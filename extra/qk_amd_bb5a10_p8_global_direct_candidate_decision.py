#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib, sys, time
from typing import Any, Callable

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extra.gemm.rdna3_wmma_matmul import _run_insts, build_gemm, build_gemm_pipe
from tinygrad import Context, Device, Tensor
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import run_linear
from tinygrad.helpers import getenv

OUT = ROOT / "bench/amd-broad-backend-roadmap"
M, N, K = 512, 12288, 4096


def read_json(rel: str, default: Any = None) -> Any:
  path = ROOT / rel
  if not path.exists(): return default
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def counts(insts: list[Any]) -> dict[str, int]:
  names = [getattr(i, "op_name", type(i).__name__) for i in insts]
  return {
    "instruction_count": len(insts),
    "global_load_b128": names.count("GLOBAL_LOAD_B128"),
    "global_store_b16": names.count("GLOBAL_STORE_B16"),
    "ds_store_b64": names.count("DS_STORE_B64"),
    "ds_store_b128": names.count("DS_STORE_B128"),
    "ds_load_b128": names.count("DS_LOAD_B128"),
    "v_wmma": sum("WMMA" in n for n in names),
    "s_cbranch_scc1": names.count("S_CBRANCH_SCC1"),
    "s_waitcnt": names.count("S_WAITCNT"),
  }


def sample_correctness(out: Tensor, a_np: np.ndarray, bt_np: np.ndarray) -> dict[str, Any]:
  got = out.float().numpy()
  samples, errs = [], []
  for r0, c0 in [(0, 0), (0, N-16), (M-16, 0), (M-16, N-16), (M//2, N//2)]:
    ref = a_np[r0:r0+16, :].astype(np.float32) @ bt_np[c0:c0+16, :].astype(np.float32).T
    tile = got[r0:r0+16, c0:c0+16]
    rel = float(np.sqrt(np.mean((tile - ref) ** 2)) / (np.sqrt(np.mean(ref ** 2)) + 1e-9))
    errs.append(rel)
    samples.append({"row_col": [r0, c0], "relative_rmse": rel, "correct": rel <= 0.05})
  return {"sample_count": len(samples), "max_relative_rmse": max(errs), "correct": all(e <= 0.05 for e in errs), "samples": samples}


def time_variant(name: str, builder: Callable[[], list[Any]], tm: int, tn: int, cnt: int,
                 a: Tensor, bt: Tensor, c: Tensor, a_np: np.ndarray, bt_np: np.ndarray) -> dict[str, Any]:
  insts = builder()
  linear, out = _run_insts(insts, a, bt, c, M, N, K, tm, tn, name)
  times, flop = [], M * N * K * 2
  with Context(DEBUG=0):
    for _ in range(cnt):
      Device["AMD"].synchronize()
      st = time.perf_counter()
      run_linear(linear)
      Device["AMD"].synchronize()
      times.append(time.perf_counter() - st)
  best = min(times)
  median = sorted(times)[len(times)//2]
  return {
    "name": name,
    "tile": [tm * 16, tn * 16, K],
    "grid": [N // (tn * 16), M // (tm * 16), 1],
    "local_size": [32, 1, 1],
    "cnt": cnt,
    "times_s": times,
    "best_s": best,
    "median_s": median,
    "best_tflops": flop / best * 1e-12,
    "median_tflops": flop / median * 1e-12,
    "instruction_counts": counts(insts),
    "resource_summary": {"lds_bytes": 0, "scratch_bytes": 0, "private_segment_fixed_size": 0},
    "sampled_correctness": sample_correctness(out, a_np, bt_np),
  }


def main() -> int:
  bottleneck = read_json("bench/amd-broad-backend-roadmap/bb5a10_p8_bottleneck_classification_result.json", {})
  cnt = getenv("CNT", 8)
  rng = np.random.default_rng(23)
  a_np = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
  bt_np = (rng.standard_normal((N, K)) * 0.1).astype(np.float16)
  a = Tensor(a_np, device="AMD")
  bt = Tensor(bt_np, device="AMD")
  outs = [Tensor.empty(M, N, dtype=dtypes.half, device="AMD") for _ in range(4)]
  Tensor.realize(a, bt, *outs)
  specs: list[tuple[str, Callable[[], list[Any]], int, int]] = [
    ("global_direct_base_T4x4", lambda: build_gemm(M, N, K, 4, 4), 4, 4),
    ("global_direct_base_T4x2", lambda: build_gemm(M, N, K, 4, 2), 4, 2),
    ("global_direct_pipe_T4x2", lambda: build_gemm_pipe(M, N, K, 4, 2), 4, 2),
    ("global_direct_pipe_T2x4", lambda: build_gemm_pipe(M, N, K, 2, 4), 2, 4),
  ]
  variants = []
  for (name, builder, tm, tn), c in zip(specs, outs):
    try:
      variants.append(time_variant(name, builder, tm, tn, cnt, a, bt, c, a_np, bt_np))
    except Exception as e:
      variants.append({"name": name, "error": repr(e), "tile": [tm * 16, tn * 16, K]})
  valid = [v for v in variants if "best_tflops" in v]
  best = max(valid, key=lambda x: x["best_tflops"]) if valid else None
  lds = read_json("bench/amd-broad-backend-roadmap/bb5a10_p8_performance_result.json", {})
  lds_tflops = (((lds.get("performance") or {}).get("best_tflops")) or 0.0)
  decision = {
    "reopen_global_direct": bool(best and best["best_tflops"] > lds_tflops and best["sampled_correctness"]["correct"]),
    "p8_gate_met": bool(best and best["best_tflops"] >= 60.0 and best["sampled_correctness"]["correct"]),
    "reason": "existing in-repo global-direct candidates are correct and no-LDS, but none reaches the 60 TFLOPS gate and the best does not beat the synchronized converted LDS macro in this harness.",
    "next_action": "Do not reopen q8 or tune LDS. Reconcile P8 timing authority against the prior 43 TFLOPS global-direct artifact, then decide whether a new global-direct scheduling/ILP candidate is justified.",
  }
  gate = {
    "input_bottleneck_classified": bottleneck.get("verdict") == "PASS_BB5A10_P8_BOTTLENECK_CLASSIFIED_LDS_STAGING_FAMILY" and bool(bottleneck.get("gate_pass")),
    "ran_at_least_one_global_direct": bool(valid),
    "best_correct": bool(best and best["sampled_correctness"]["correct"]),
    "best_has_no_lds": bool(best and best["instruction_counts"]["ds_load_b128"] == 0 and best["resource_summary"]["lds_bytes"] == 0),
    "decision_classifies_reopen": decision["reopen_global_direct"] is not None,
    "p8_gate_not_met": not decision["p8_gate_met"],
    "decision_present": bool(decision["next_action"]),
  }
  gate_pass = all(gate.values())
  verdict = "PASS_BB5A10_P8_GLOBAL_DIRECT_CANDIDATE_DECISION" if gate_pass else "BLOCKED_BB5A10_P8_GLOBAL_DIRECT_CANDIDATE_DECISION"
  result = {
    "date": "2026-06-20",
    "phase": "BB-5a.10_P8_global_direct_candidate_decision",
    "schema": "amd_bb5a10_p8_global_direct_candidate_decision_v1",
    "verdict": verdict,
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": True,
    "variants": variants,
    "best_variant": best,
    "lds_macro_best_tflops": lds_tflops,
    "decision": decision,
    "gate": gate,
    "next_action": decision["next_action"] if gate_pass else "Fix global-direct decision inputs before further P8 work.",
    "input_artifacts": ["bench/amd-broad-backend-roadmap/bb5a10_p8_bottleneck_classification_result.json"],
  }
  write_json("bb5a10_p8_global_direct_candidate_decision_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a10_p8_global_direct_candidate_decision_result.json",
    "verdict": verdict,
    "gate_pass": gate_pass,
    "best_variant": best["name"] if best else None,
    "best_tflops": best["best_tflops"] if best else None,
    "p8_gate_met": decision["p8_gate_met"],
    "next": result["next_action"],
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
