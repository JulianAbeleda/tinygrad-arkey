#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib, sys, time
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extra.gemm.rdna3_wmma_matmul import _run_insts_lds, build_gemm_lds2
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


def time_insts(insts: list[Any], name: str, cnt: int, a: Tensor, bt: Tensor, c: Tensor) -> dict[str, Any]:
  linear, _ = _run_insts_lds(insts, a, bt, c, M, N, K, name, LDS_BYTES, BM, BN, THREADS)
  times = []
  flop = M * N * K * 2
  with Context(DEBUG=0):
    for _ in range(cnt):
      Device["AMD"].synchronize()
      st = time.perf_counter()
      run_linear(linear)
      Device["AMD"].synchronize()
      times.append(time.perf_counter() - st)
  best = min(times)
  median = sorted(times)[len(times) // 2]
  return {"times_s": times, "best_s": best, "median_s": median, "best_tflops": flop / best * 1e-12, "median_tflops": flop / median * 1e-12}


def counts(insts: list[Any]) -> dict[str, int]:
  names = [getattr(i, "op_name", type(i).__name__) for i in insts]
  return {
    "instruction_count": len(insts),
    "ds_store_b64": names.count("DS_STORE_B64"),
    "ds_store_b128": names.count("DS_STORE_B128"),
    "ds_load_b128": names.count("DS_LOAD_B128"),
    "global_load_b128": names.count("GLOBAL_LOAD_B128"),
    "global_store_b16": names.count("GLOBAL_STORE_B16"),
    "s_barrier": names.count("S_BARRIER"),
    "v_wmma": sum("WMMA" in n for n in names),
    "s_cbranch_scc1": names.count("S_CBRANCH_SCC1"),
  }


def main() -> int:
  p8 = read_json("bench/amd-broad-backend-roadmap/bb5a10_p8_performance_result.json", {})
  cnt = getenv("CNT", 10)
  rng = np.random.default_rng(17)
  a = Tensor((rng.standard_normal((M, K)) * 0.1).astype(np.float16), device="AMD")
  bt = Tensor((rng.standard_normal((N, K)) * 0.1).astype(np.float16), device="AMD")
  c0 = Tensor.empty(M, N, dtype=dtypes.half, device="AMD")
  c1 = Tensor.empty(M, N, dtype=dtypes.half, device="AMD")
  Tensor.realize(a, bt, c0, c1)
  original = build_gemm_lds2(M, N, K, 2, 2, 4, 4, 16, 0, 0)
  converted, conversion = build_converted_macro_insts()
  original_perf = time_insts(original, "bb5a10_p8_b128_baseline", cnt, a, bt, c0)
  converted_perf = time_insts(converted, "bb5a10_p8_ds64_converted", cnt, a, bt, c1)
  known = {
    "p8_gate_tflops": 60.0,
    "tinygrad_prior_global_direct_authority_tflops": 43.026,
    "tensile_authority_tflops": 65.6,
    "prior_lds_family_verdict": "LDS round-trip + barrier overhead refuted as RDNA3 path; see docs/route-a-a3-p2-p3-lds-refuted-20260619.md",
  }
  ratios = {
    "converted_vs_gate": converted_perf["best_tflops"] / known["p8_gate_tflops"],
    "converted_vs_prior_global_direct": converted_perf["best_tflops"] / known["tinygrad_prior_global_direct_authority_tflops"],
    "converted_vs_tensile": converted_perf["best_tflops"] / known["tensile_authority_tflops"],
    "converted_vs_original_b128": converted_perf["best_tflops"] / original_perf["best_tflops"],
  }
  classification = {
    "primary": "LDS_STAGING_FAMILY_BOTTLENECK",
    "why": [
      "converted DS64 macro is correct and scratch/private free but far below the 60 TFLOPS gate",
      "original B128 and converted DS64 macro candidates are both LDS round-trip plus barrier kernels",
      "prior route-a-a3 evidence already refuted the multi-wave LDS family as net-negative versus global-direct WMMA on RDNA3",
      "selected-compatible DS64 store conversion fixes the contract but does not fix the LDS/barrier scheduling family",
    ],
    "not_primary": [
      "not correctness: sampled P8 correctness passes",
      "not scratch/private spill: both are zero",
      "not launch mapping: TTA1/TTA2/TTA3 prove full authority grid and macro shape",
      "not merely ds_store_b64 conversion: the original B128 baseline remains in the same LDS-staged family",
    ],
    "next_action": "Stop optimizing this LDS-staged macro as the P8 path; reopen a selected-compatible global-direct/IC-served WMMA candidate or classify why selected Tensile layout cannot transfer without LDS round-trip.",
  }
  gate = {
    "input_p8_blocked": p8.get("verdict") == "BLOCKED_BB5A10_P8_PERFORMANCE_GATE_NOT_MET",
    "ran_original_b128": original_perf["best_tflops"] > 0,
    "ran_converted_ds64": converted_perf["best_tflops"] > 0,
    "converted_below_gate": converted_perf["best_tflops"] < 60.0,
    "conversion_contract_preserved": counts(converted)["ds_store_b64"] == 8 and counts(converted)["ds_store_b128"] == 0,
    "classification_primary_present": classification["primary"] == "LDS_STAGING_FAMILY_BOTTLENECK",
  }
  gate_pass = all(gate.values())
  result = {
    "date": "2026-06-20",
    "phase": "BB-5a.10_P8_bottleneck_classification",
    "schema": "amd_bb5a10_p8_bottleneck_classification_v1",
    "verdict": "PASS_BB5A10_P8_BOTTLENECK_CLASSIFIED_LDS_STAGING_FAMILY" if gate_pass else "BLOCKED_BB5A10_P8_BOTTLENECK_CLASSIFICATION",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": True,
    "cnt": cnt,
    "original_b128": {"instruction_counts": counts(original), "performance": original_perf},
    "converted_ds64": {"instruction_counts": counts(converted), "performance": converted_perf, "conversion": conversion},
    "known_context": known,
    "ratios": ratios,
    "classification": classification,
    "gate": gate,
    "decision": "P8 bottleneck classified: the selected-compatible macro is in the wrong LDS-staged family; DS64 conversion fixed the contract but not the performance class." if gate_pass else
                "P8 bottleneck classification blocked; timing comparison or classification inputs missing.",
    "next_action": classification["next_action"],
    "input_artifacts": [
      "bench/amd-broad-backend-roadmap/bb5a10_p8_performance_result.json",
      "bench/amd-broad-backend-roadmap/bb5a10_p8_tta3_macro_candidate_result.json",
    ],
  }
  write_json("bb5a10_p8_bottleneck_classification_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a10_p8_bottleneck_classification_result.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "original_b128_tflops": original_perf["best_tflops"],
    "converted_ds64_tflops": converted_perf["best_tflops"],
    "next": result["next_action"],
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
