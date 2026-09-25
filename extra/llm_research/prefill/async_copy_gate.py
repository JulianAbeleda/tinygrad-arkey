#!/usr/bin/env python3
"""GPU gate for async-copy (cp.async ring) precontract candidates (off the production path).

Each case runs the candidate in this process and the safe tensor-core path (same TC opt, no candidate) in a child
process -- programs are cached by the pre-optimization AST -- and requires finite, bit-identical fp32 outputs.

  DEV=NV python3 extra/llm_research/prefill/async_copy_gate.py [--out gate.json]
"""
from __future__ import annotations
import argparse, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tinygrad import Tensor, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.kernel_pipeline import KernelStage1PipelinePlan
from tinygrad.llm.prefill_candidate_runtime import KernelCandidateContext, KernelLDSWindow, KernelTileGeometry
import tinygrad.codegen.opt.postrange as pr

# (m, n, k, tile, waves, stages, dtype): static and launch-sized arenas, 2..4 stages, bf16 and fp16
CASES = ((512, 5120, 3136, (128, 128, 32), (4, 2), 3, "bfloat16"), (512, 5120, 3136, (128, 128, 32), (4, 2), 4, "bfloat16"),
         (256, 3200, 7680, (64, 64, 32), (2, 2), 3, "bfloat16"), (384, 12544, 3136, (128, 64, 64), (4, 2), 3, "bfloat16"),
         (512, 17536, 3136, (128, 128, 32), (4, 2), 2, "bfloat16"), (512, 5120, 3136, (128, 128, 32), (4, 2), 3, "half"))


def _operands(m, n, k, dtype):
  rng = np.random.default_rng(m + 3 * n + 7 * k)
  a = rng.standard_normal((m, k), dtype=np.float32); w = rng.standard_normal((n, k), dtype=np.float32) * 0.02
  return Tensor(a).cast(dtype).realize(), Tensor(w).cast(dtype).realize()


def run(case, candidate: bool) -> np.ndarray:
  m, n, k, tile, waves, stages, dtype = case
  a, w = _operands(m, n, k, getattr(dtypes, dtype))
  key = pr.warmstart_key({m, n}, k)
  context = None
  if candidate:
    stride = tile[2] * 2 + 16
    geometry = KernelTileGeometry(tile, waves, waves[0] * waves[1] * 32, 32,
      (KernelLDSWindow("A", 0, tile[0] * stride, stride), KernelLDSWindow("B", tile[0] * stride, (tile[0] + tile[1]) * stride, stride)))
    context = {key: KernelCandidateContext("boltbeam.full_kernel_candidate.v1", "0" * 64, geometry,
                                           KernelStage1PipelinePlan(stages, geometry.lds_bytes, 1, async_copy=True))}
  with pr.warmstart_candidate_state({key: (Opt(OptOps.TC, 0, (-1, 2, 1)),)}, context):
    return a.dot(w.T, dtype=dtypes.float).realize().numpy()


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out"); parser.add_argument("--safe-case", type=int, help=argparse.SUPPRESS); parser.add_argument("--dump")
  args = parser.parse_args()
  if args.safe_case is not None:
    np.save(args.dump, run(CASES[args.safe_case], candidate=False)); return 0
  rows, scratch = [], Path(tempfile.mkdtemp(prefix="async_copy_gate_"))
  for index, case in enumerate(CASES):
    got = run(case, candidate=True)
    dump = scratch / f"safe_{index}.npy"
    subprocess.run([sys.executable, __file__, "--safe-case", str(index), "--dump", str(dump)], check=True)
    safe = np.load(dump)
    row = {"m": case[0], "n": case[1], "k": case[2], "tile": list(case[3]), "waves": list(case[4]), "stages": case[5], "dtype": case[6],
           "finite": bool(np.isfinite(got).all()), "bitexact_vs_safe_tc": bool((got.view(np.uint32) == safe.view(np.uint32)).all()),
           "max_abs_diff": float(np.abs(got - safe).max())}
    rows.append(row); print(json.dumps(row), flush=True)
  passed = all(r["finite"] and r["bitexact_vs_safe_tc"] for r in rows)
  if args.out: Path(args.out).write_text(json.dumps({"schema": "async-copy-gate.v1", "passed": passed, "rows": rows}, indent=2) + "\n")
  print(json.dumps({"passed": passed, "rows": len(rows)}))
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
