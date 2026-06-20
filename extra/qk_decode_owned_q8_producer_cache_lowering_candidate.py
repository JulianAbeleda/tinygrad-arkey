#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time
from typing import Any

import numpy as np

from tinygrad import dtypes
from tinygrad.device import Buffer, Device
from extra.q8_ffn_hcq_artifact import NORM_SOURCE, copyin_array, copyout_array, make_buffer, q8_dequant


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_owned_q8_producer_cache_lowering_candidate_result.json"


def load(rel: str, default: Any = None) -> Any:
  p = ROOT / rel
  return json.loads(p.read_text()) if p.exists() else default


def stats_ms(samples: list[float]) -> dict[str, Any]:
  return {
    "samples_ms": [round(x, 6) for x in samples],
    "min_ms": min(samples),
    "median_ms": statistics.median(samples),
    "mean_ms": statistics.fmean(samples),
    "max_ms": max(samples),
  }


def main() -> int:
  ap = argparse.ArgumentParser(description="Owned q8 producer/cache lowering candidate: tinygrad HCQ/COMGR q8_rmsnorm_side")
  ap.add_argument("--seed", type=int, default=7)
  ap.add_argument("--warmups", type=int, default=10)
  ap.add_argument("--iters", type=int, default=50)
  ap.add_argument("--out", type=pathlib.Path, default=OUT)
  args = ap.parse_args()

  scope = load("bench/qk-decode-primitive-transfer/decode_owned_q8_producer_cache_lowering_scope_result.json", {})
  target = (scope.get("candidate", {}).get("target") or {})
  target_us = float(target.get("producer_lifecycle_us_lte") or 7.501304)
  fp_abs_lte = float(target.get("reference_fp_max_abs_lte") or 1e-5)
  q8_abs_lte = float(target.get("reference_q8_max_abs_lte") or 0.02)

  rng = np.random.default_rng(args.seed)
  x = (rng.standard_normal(4096).astype(np.float32) * 0.9).astype(np.float32)
  w = (0.7 + rng.random(4096).astype(np.float32) * 0.2).astype(np.float32)
  rinv = np.float32(1.0 / np.sqrt(np.sum(x * x, dtype=np.float32) / np.float32(4096.0) + np.float32(1.0e-6)))
  ref_norm = (x * rinv * w).astype(np.float32)

  dev = Device["AMD"]
  t0 = time.perf_counter()
  prg = dev.runtime("owned_q8_rmsnorm_side_hcq_comgr", dev.compiler.compile(NORM_SOURCE))
  compile_s = time.perf_counter() - t0
  xbuf, wbuf = make_buffer(4096, dtypes.float32), make_buffer(4096, dtypes.float32)
  outbuf, q8buf = make_buffer(4096, dtypes.float32), make_buffer(128 * 36, dtypes.uint8)
  copyin_array(xbuf, x)
  copyin_array(wbuf, w)

  samples: list[float] = []
  for i in range(args.warmups + args.iters):
    ms = float(prg(outbuf._buf, q8buf._buf, xbuf._buf, wbuf._buf, global_size=(1, 1, 1), local_size=(256, 1, 1), wait=True)) * 1000.0
    if i >= args.warmups:
      samples.append(ms)

  got_norm = copyout_array(outbuf, np.empty(4096, dtype=np.float32))
  got_q8 = bytearray(128 * 36)
  q8buf.copyout(memoryview(got_q8))
  q8_x = q8_dequant(bytes(got_q8), 4096)
  fp_err = np.abs(got_norm - ref_norm)
  q8_err = np.abs(q8_x - ref_norm)
  timing = stats_ms(samples)
  producer_us = timing["median_ms"] * 1000.0
  maps = pathlib.Path("/proc/self/maps").read_text(errors="ignore")
  gates = {
    "scope_ready": scope.get("gate_pass") is True,
    "fp_correct": float(fp_err.max()) <= fp_abs_lte,
    "q8_dequant_bounded": float(q8_err.max()) <= q8_abs_lte,
    "q8_bytes_4608": len(got_q8) == 4608,
    "no_hip_runtime_in_process": "libamdhip64.so" not in maps,
    "producer_lifecycle_lte_target": producer_us <= target_us,
  }
  if not all(v for k, v in gates.items() if k != "producer_lifecycle_lte_target"):
    verdict = "BLOCKED_DECODE_OWNED_Q8_PRODUCER_CACHE_LOWERING_INCORRECT"
  elif gates["producer_lifecycle_lte_target"]:
    verdict = "PASS_DECODE_OWNED_Q8_PRODUCER_CACHE_LOWERING_CANDIDATE"
  else:
    verdict = "BLOCKED_DECODE_OWNED_Q8_PRODUCER_CACHE_LOWERING_TOO_SLOW"
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_OWNED_Q8_PRODUCER_CACHE_LOWERING_CANDIDATE",
    "schema": "decode_owned_q8_producer_cache_lowering_candidate_v1",
    "verdict": verdict,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": True,
    "candidate": {
      "name": "owned_hcq_comgr_q8_rmsnorm_side",
      "source": "extra.q8_ffn_hcq_artifact.NORM_SOURCE",
      "runtime": "tinygrad AMD HCQ / COMGR",
      "compile_s": compile_s,
      "launch": {"global_size": [1, 1, 1], "local_size": [256, 1, 1]},
    },
    "timing": timing,
    "producer_us": producer_us,
    "target_us": target_us,
    "correctness": {
      "fp_max_abs": float(fp_err.max()),
      "fp_mean_abs": float(fp_err.mean()),
      "q8_dequant_max_abs": float(q8_err.max()),
      "q8_dequant_mean_abs": float(q8_err.mean()),
      "q8_bytes": len(got_q8),
    },
    "gates": gates,
    "decision": {
      "if_pass": "Use this as the owned producer/cache row and move to owned consumer parity/lifecycle integration.",
      "if_too_slow": "Producer semantics are owned but not artifact-parity; optimize producer or keep artifact target as oracle.",
      "if_incorrect": "Fix q8 byte/reference semantics before timing.",
    },
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "producer_us": producer_us,
    "target_us": target_us,
    "correctness": result["correctness"],
    "gates": gates,
    "out": str(args.out.relative_to(ROOT) if args.out.is_absolute() and args.out.is_relative_to(ROOT) else args.out),
  }, indent=2))
  return 0 if gates["scope_ready"] and gates["fp_correct"] and gates["q8_dequant_bounded"] and gates["q8_bytes_4608"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
