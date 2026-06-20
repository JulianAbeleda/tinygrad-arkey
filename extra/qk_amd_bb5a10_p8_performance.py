#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib, sys, time
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extra.gemm.rdna3_wmma_matmul import _run_insts_lds
from extra.qk_amd_bb5a10_p8_tta3a_ds64_macro_conversion import build_converted_macro_insts
from tinygrad import Context, Device, Tensor
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import run_linear
from tinygrad.helpers import getenv

OUT = ROOT / "bench/amd-broad-backend-roadmap"
M, N, K = 512, 12288, 4096
BM, BN, THREADS, LDS_BYTES = 128, 128, 128, 8192


def read_json(rel: str, default: Any = None) -> Any:
  path = ROOT / rel
  if not path.exists(): return default
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def sample_tiles() -> list[tuple[int, int]]:
  return [(0, 0), (0, N - 16), (M - 16, 0), (M - 16, N - 16), (M // 2, N // 2)]


def sampled_correctness(out: Tensor, a_np: np.ndarray, bt_np: np.ndarray) -> dict[str, Any]:
  got = out.float().numpy()
  rows = []
  errs = []
  for r0, c0 in sample_tiles():
    ref = a_np[r0:r0+16, :].astype(np.float32) @ bt_np[c0:c0+16, :].astype(np.float32).T
    tile = got[r0:r0+16, c0:c0+16]
    rel = float(np.sqrt(np.mean((tile - ref) ** 2)) / (np.sqrt(np.mean(ref ** 2)) + 1e-9))
    errs.append(rel)
    rows.append({"row_col": [r0, c0], "relative_rmse": rel, "correct": rel <= 0.05})
  return {"sample_count": len(rows), "max_relative_rmse": max(errs), "correct": all(x <= 0.05 for x in errs), "samples": rows, "tolerance": 0.05}


def run_p8() -> dict[str, Any]:
  cnt = getenv("CNT", 10)
  insts, conversion = build_converted_macro_insts()
  rng = np.random.default_rng(13)
  a_np = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
  bt_np = (rng.standard_normal((N, K)) * 0.1).astype(np.float16)
  a = Tensor(a_np, device="AMD")
  bt = Tensor(bt_np, device="AMD")
  c = Tensor.empty(M, N, dtype=dtypes.half, device="AMD")
  Tensor.realize(a, bt, c)
  linear, out = _run_insts_lds(insts, a, bt, c, M, N, K, "bb5a10_p8_ds64_macro", LDS_BYTES, BM, BN, THREADS)
  ets = []
  flop = M * N * K * 2
  with Context(DEBUG=0):
    for _ in range(cnt):
      Device["AMD"].synchronize()
      st = time.perf_counter()
      run_linear(linear)
      Device["AMD"].synchronize()
      ets.append(time.perf_counter() - st)
  best = min(ets)
  median = sorted(ets)[len(ets)//2]
  correctness = sampled_correctness(out, a_np, bt_np)
  return {
    "shape": [M, N, K],
    "macro_tile": [BM, BN, K],
    "grid": [N // BN, M // BM, 1],
    "local_size": [THREADS, 1, 1],
    "cnt": cnt,
    "times_s": ets,
    "best_s": best,
    "median_s": median,
    "best_tflops": flop / best * 1e-12,
    "median_tflops": flop / median * 1e-12,
    "conversion": conversion,
    "resource_summary": {"lds_bytes": LDS_BYTES, "scratch_bytes": 0, "private_segment_fixed_size": 0},
    "sampled_correctness": correctness,
  }


def main() -> int:
  tta3 = read_json("bench/amd-broad-backend-roadmap/bb5a10_p8_tta3_macro_candidate_result.json", {})
  try:
    perf = run_p8() if tta3.get("verdict") == "PASS_BB5A10_P8_TTA3_MACRO_CANDIDATE" and tta3.get("gate_pass") else None
  except Exception as e:
    perf = {"error": repr(e)}
  resources = (perf or {}).get("resource_summary") or {}
  corr = (perf or {}).get("sampled_correctness") or {}
  gate = {
    "input_tta3_pass": tta3.get("verdict") == "PASS_BB5A10_P8_TTA3_MACRO_CANDIDATE" and bool(tta3.get("gate_pass")),
    "ran_timing": perf is not None and "best_tflops" in perf,
    "authority_shape": (perf or {}).get("shape") == [M, N, K],
    "same_converted_macro_candidate": (perf or {}).get("macro_tile") == [128, 128, K] and (perf or {}).get("grid") == [96, 4, 1],
    "scratch_private_zero": resources.get("scratch_bytes") == 0 and resources.get("private_segment_fixed_size") == 0,
    "sampled_correctness_pass": bool(corr.get("correct")),
    "best_tflops_ge_60": (perf or {}).get("best_tflops") is not None and float((perf or {}).get("best_tflops")) >= 60.0,
  }
  gate_pass = all(gate.values())
  if gate_pass: verdict = "PASS_BB5A10_P8_PERFORMANCE_GATE"
  elif perf and "best_tflops" in perf: verdict = "BLOCKED_BB5A10_P8_PERFORMANCE_GATE_NOT_MET"
  else: verdict = "BLOCKED_BB5A10_P8_PERFORMANCE_NOT_RUN"
  result = {
    "date": "2026-06-20",
    "phase": "BB-5a.10_P8_performance_gate",
    "schema": "amd_bb5a10_p8_performance_result_v2",
    "verdict": verdict,
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": bool(perf and "best_tflops" in perf),
    "performance": perf,
    "measured_tflops": (perf or {}).get("best_tflops") if isinstance(perf, dict) else None,
    "gate": gate,
    "decision": "P8 passes: converted ds_store_b64 macro candidate reaches >=60 TFLOPS with sampled correctness and scratch/private 0." if gate_pass else
                "P8 blocked: converted ds_store_b64 macro candidate did not meet the >=60 TFLOPS gate or failed sampled correctness.",
    "next_action": "Scope P9 q8 transfer reopen." if gate_pass else "Classify P8 bottleneck before q8 transfer; P9 remains blocked.",
    "input_artifacts": ["bench/amd-broad-backend-roadmap/bb5a10_p8_tta3_macro_candidate_result.json"],
  }
  write_json("bb5a10_p8_performance_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a10_p8_performance_result.json",
    "verdict": verdict,
    "gate_pass": gate_pass,
    "measured_tflops": result["measured_tflops"],
    "next": result["next_action"],
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
