#!/usr/bin/env python3
"""Promotion gate for tinygrad/llm/generated/dense_bf16_sm120_candidate_set.json (off the production path).

For every promoted route (role, m, n, k, split_k), on the live device, through the production chunk executor:
  1. correctness: routed GEMM vs a fp32 numpy (BLAS) oracle on the same bf16-rounded operands;
  2. identity (split_k == 1): bit-exact vs the safe tensor-core path (same TC opt, no candidate context), run in a
     separate process so the program cache cannot replay the candidate kernel for the safe arm; split-K routes
     reassociate the K sum and are held to the oracle bound instead;
  3. speed: median GPU kernel time of the candidate GEMM kernel and of the safe TC path, as TFLOPS against
     the measured bf16 mma.sync peak (extra/llm_research/microbench/mma_peak_cuda.cu -DBF16).

  DEV=NV python3 extra/llm_research/prefill/dense_bf16_candidate_gate.py [--reps 9] [--out gate.json]
"""
from __future__ import annotations
import argparse, contextlib, io, json, statistics, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tinygrad import Tensor, Device, Context, GlobalCounters, dtypes
from tinygrad.codegen.opt import Opt, OptOps
import tinygrad.codegen.opt.postrange as pr
from tinygrad.llm import dense_candidate_gemm as dense

MEASURED_BF16_PEAK_TFLOPS = 251.7   # mma_peak_cuda.cu -DBF16, 5090, blocks=32768, 2026-09-25


def _bf16_round(x: np.ndarray) -> np.ndarray:
  u = x.astype(np.float32).view(np.uint32)
  return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32).view(np.float32)


def _gpu_times(fn, reps: int) -> list[float]:
  out = []
  for _ in range(reps):
    GlobalCounters.reset()
    with Context(DEBUG=2), contextlib.redirect_stdout(io.StringIO()): fn().realize()
    out.append(GlobalCounters.time_sum_s)
  return out


_PROGRAMS: list[str] = []
def _spy_programs():
  import tinygrad.engine.realize as rz
  orig = rz.get_runtime
  def spy(device, ast, cache=True):
    _PROGRAMS.append(ast.arg.function_name)
    return orig(device, ast, cache)
  rz.get_runtime = spy


def _operands(m: int, n: int, k: int):
  rng = np.random.default_rng(m * 7 + n + k)
  a_np = _bf16_round(rng.standard_normal((m, k), dtype=np.float32))
  w_np = _bf16_round(rng.standard_normal((n, k), dtype=np.float32) * 0.02)
  return a_np, w_np


def run_arm(route, reps: int, *, candidate: bool) -> tuple[np.ndarray, float, list[str]]:
  """One arm in this process.  Compiled programs are cached by the pre-optimization AST, so the two arms
  must never share a process: the second would silently replay the first arm's kernel."""
  m, n, k = route.m, route.n, route.k
  a_np, w_np = _operands(m, n, k)
  a, w = Tensor(a_np).cast(dtypes.bfloat16).realize(), Tensor(w_np).cast(dtypes.bfloat16).realize()
  _PROGRAMS.clear()
  if candidate:
    gemm = lambda: dense._chunk(a, w, route)
    out = gemm().realize()
    times = _gpu_times(gemm, reps)
  else:
    key = pr.warmstart_key({m, n}, k)
    gemm = lambda: a.dot(w.T, dtype=dtypes.float)
    with pr.warmstart_candidate_state({key: (Opt(OptOps.TC, 0, (-1, 2, 1)),)}, None):
      out = gemm().realize()
      times = _gpu_times(gemm, reps)
  return out.numpy(), statistics.median(times), list(dict.fromkeys(_PROGRAMS))


def run_row(route, reps: int, index: int, scratch: Path) -> dict:
  m, n, k = route.m, route.n, route.k
  got, cand_med, cand_programs = run_arm(route, reps, candidate=True)
  dump = scratch / f"safe_{index}.npz"
  subprocess.run([sys.executable, __file__, "--safe-row", str(index), "--reps", str(reps), "--dump", str(dump)], check=True)
  safe = np.load(dump)
  ref_safe, safe_med, safe_programs = safe["out"], float(safe["median_s"]), [str(x) for x in safe["programs"]]
  a_np, w_np = _operands(m, n, k)
  oracle = (a_np @ w_np.T).astype(np.float64)   # fp32 BLAS oracle: ~1e-6 relative at these K, far under the gate
  flop = 2 * m * n * k
  tm, tn, tk = route.admission.geometry.tile
  return {"role": route.role, "m": m, "n": n, "k": k, "split_k": route.split_k,
          "geometry": [tm, tn, tk, *route.admission.geometry.waves], "canonical_identity": route.admission.canonical_identity,
          "candidate_programs": cand_programs, "safe_tc_programs": safe_programs,
          "finite": bool(np.isfinite(got).all()),
          "max_abs_err_vs_oracle": float(np.abs(got - oracle).max()),
          "max_rel_err_vs_oracle": float(np.abs(got - oracle).max() / np.abs(oracle).max()),
          "bitexact_vs_safe_tc": bool((got.view(np.uint32) == ref_safe.view(np.uint32)).all()),
          "distinct_programs": set(cand_programs).isdisjoint(safe_programs),
          "candidate_median_us": round(cand_med * 1e6, 2), "safe_tc_median_us": round(safe_med * 1e6, 2),
          "candidate_tflops": round(flop / cand_med / 1e12, 2), "safe_tc_tflops": round(flop / safe_med / 1e12, 2),
          "candidate_pct_of_measured_peak": round(100 * flop / cand_med / 1e12 / MEASURED_BF16_PEAK_TFLOPS, 1)}


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--reps", type=int, default=9)
  parser.add_argument("--out")
  parser.add_argument("--rows", help="comma list of m values to restrict to")
  parser.add_argument("--safe-row", type=int, help=argparse.SUPPRESS)
  parser.add_argument("--dump", help=argparse.SUPPRESS)
  args = parser.parse_args()
  _spy_programs()
  target = dense.device_target(Device.DEFAULT)
  routes = dense.dense_bf16_routes(target["backend"], target["arch"], target["wave_size"])
  if routes is None: raise SystemExit(f"no promoted dense bf16 candidates for {target}")
  routes = [routes[key] for key in sorted(routes)]
  if args.safe_row is not None:
    out, med, programs = run_arm(routes[args.safe_row], args.reps, candidate=False)
    np.savez(args.dump, out=out, median_s=med, programs=np.array(programs))
    return 0
  scratch = Path(tempfile.mkdtemp(prefix="dense_bf16_gate_"))
  keep = {int(x) for x in args.rows.split(",")} if args.rows else None
  rows = []
  for index, route in enumerate(routes):
    if keep is not None and route.m not in keep: continue
    rows.append(row := run_row(route, args.reps, index, scratch))
    print(json.dumps(row), flush=True)
  passed = all(r["finite"] and r["distinct_programs"] and r["max_rel_err_vs_oracle"] < 1e-4 and
               (r["bitexact_vs_safe_tc"] or r["split_k"] > 1) for r in rows)
  result = {"schema": "dense-bf16-candidate-gate.v2", "target": target, "device": Device.DEFAULT,
            "measured_bf16_peak_tflops": MEASURED_BF16_PEAK_TFLOPS, "passed": passed, "rows": rows}
  if args.out: Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps({"passed": passed, "rows": len(rows)}))
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
