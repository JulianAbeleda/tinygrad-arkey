#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from typing import Any

import numpy as np

from tinygrad import dtypes
from tinygrad.device import Device
from extra.q8_ffn_hcq_artifact import NORM_SOURCE, copyin_array, copyout_array, make_buffer, q8_dequant


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_owned_q8_producer_cache_nt_grid_result.json"


def source_nt(nt: int) -> str:
  src = NORM_SOURCE
  src = src.replace("amdgpu_flat_work_group_size(1, 256)", f"amdgpu_flat_work_group_size(1, {nt})")
  src = src.replace("float red[256]", f"float red[{nt}]")
  src = src.replace("i += 256", f"i += {nt}")
  src = src.replace("int off = 128", f"int off = {nt // 2}")
  src = src.replace("b += 256", f"b += {nt}")
  return src


def load(rel: str, default: Any = None) -> Any:
  p = ROOT / rel
  return json.loads(p.read_text()) if p.exists() else default


def main() -> int:
  ap = argparse.ArgumentParser(description="Owned q8 producer/cache workgroup-size grid")
  ap.add_argument("--nts", type=int, nargs="+", default=[128, 256, 512, 1024])
  ap.add_argument("--warmups", type=int, default=10)
  ap.add_argument("--iters", type=int, default=50)
  ap.add_argument("--seed", type=int, default=7)
  ap.add_argument("--out", type=pathlib.Path, default=OUT)
  args = ap.parse_args()

  candidate = load("bench/qk-decode-primitive-transfer/decode_owned_q8_producer_cache_lowering_candidate_result.json", {})
  target_us = float(candidate.get("target_us") or 7.501304)
  rng = np.random.default_rng(args.seed)
  x = (rng.standard_normal(4096).astype(np.float32) * 0.9).astype(np.float32)
  w = (0.7 + rng.random(4096).astype(np.float32) * 0.2).astype(np.float32)
  rinv = np.float32(1.0 / np.sqrt(np.sum(x * x, dtype=np.float32) / np.float32(4096.0) + np.float32(1.0e-6)))
  ref = (x * rinv * w).astype(np.float32)
  rows: list[dict[str, Any]] = []
  dev = Device["AMD"]
  for nt in args.nts:
    try:
      prg = dev.runtime(f"owned_q8_rmsnorm_side_nt{nt}", dev.compiler.compile(source_nt(nt)))
      xbuf, wbuf = make_buffer(4096, dtypes.float32), make_buffer(4096, dtypes.float32)
      outbuf, q8buf = make_buffer(4096, dtypes.float32), make_buffer(128 * 36, dtypes.uint8)
      copyin_array(xbuf, x)
      copyin_array(wbuf, w)
      samples: list[float] = []
      for i in range(args.warmups + args.iters):
        ms = float(prg(outbuf._buf, q8buf._buf, xbuf._buf, wbuf._buf, global_size=(1, 1, 1), local_size=(nt, 1, 1), wait=True)) * 1000.0
        if i >= args.warmups:
          samples.append(ms)
      got = copyout_array(outbuf, np.empty(4096, dtype=np.float32))
      q8 = bytearray(128 * 36)
      q8buf.copyout(memoryview(q8))
      xq = q8_dequant(bytes(q8), 4096)
      rows.append({
        "nt": nt,
        "correct": float(np.abs(got - ref).max()) <= 1e-5 and float(np.abs(xq - ref).max()) <= 0.02 and len(q8) == 4608,
        "median_us": statistics.median(samples) * 1000.0,
        "min_us": min(samples) * 1000.0,
        "fp_max_abs": float(np.abs(got - ref).max()),
        "q8_dequant_max_abs": float(np.abs(xq - ref).max()),
        "q8_bytes": len(q8),
      })
    except Exception as e:
      rows.append({"nt": nt, "correct": False, "error": repr(e)})
  correct_rows = [r for r in rows if r.get("correct") and "median_us" in r]
  best = min(correct_rows, key=lambda r: r["median_us"]) if correct_rows else {}
  gates = {
    "candidate_correctness_probe_ran": candidate.get("gates", {}).get("fp_correct") is True,
    "all_grid_rows_correct": bool(rows) and all(r.get("correct") is True for r in rows),
    "best_lte_artifact_target": bool(best) and best.get("median_us", 999.0) <= target_us,
  }
  if not gates["all_grid_rows_correct"]:
    verdict = "BLOCKED_DECODE_OWNED_Q8_PRODUCER_NT_GRID_INCORRECT"
  elif gates["best_lte_artifact_target"]:
    verdict = "PASS_DECODE_OWNED_Q8_PRODUCER_NT_GRID_TARGET_MET"
  else:
    verdict = "BLOCKED_DECODE_OWNED_Q8_PRODUCER_NT_GRID_TOO_SLOW"
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_OWNED_Q8_PRODUCER_CACHE_NT_GRID",
    "schema": "decode_owned_q8_producer_cache_nt_grid_v1",
    "verdict": verdict,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": True,
    "target_us": target_us,
    "rows": rows,
    "best": best,
    "gates": gates,
    "decision": {
      "if_too_slow": "Raw COMGR producer is semantically correct but not artifact-parity; next needs producer codegen optimization or artifact remains oracle.",
      "if_pass": "Use fastest NT row as owned producer/cache candidate.",
    },
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "target_us": target_us,
    "best": best,
    "gates": gates,
    "out": str(args.out.relative_to(ROOT) if args.out.is_absolute() and args.out.is_relative_to(ROOT) else args.out),
  }, indent=2))
  return 0 if gates["all_grid_rows_correct"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
