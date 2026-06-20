#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_owned_q8_producer_cache_reference_result.json"


def load(rel: str, default: Any = None) -> Any:
  p = ROOT / rel
  return json.loads(p.read_text()) if p.exists() else default


def q8_blocks_ref(x: np.ndarray) -> bytes:
  blocks = x.astype(np.float32).reshape(-1, 32)
  scales = np.max(np.abs(blocks), axis=1) / 127.0
  scales = np.where(scales == 0, 1.0, scales).astype(np.float32)
  qs = np.rint(blocks / scales[:, None]).clip(-128, 127).astype(np.int8)
  out = bytearray()
  for d, q in zip(scales.astype(np.float16), qs):
    out += np.float16(d).tobytes()
    out += np.float16(0.0).tobytes()
    out += q.tobytes()
  return bytes(out)


def q8_dequant_ref(q8: bytes, n: int) -> np.ndarray:
  out = np.empty(n, dtype=np.float32)
  for bi in range(n // 32):
    off = bi * 36
    d = np.frombuffer(q8[off:off + 2], dtype=np.float16).astype(np.float32)[0]
    q = np.frombuffer(q8[off + 4:off + 36], dtype=np.int8).astype(np.float32)
    out[bi * 32:(bi + 1) * 32] = d * q
  return out


def main() -> int:
  obj_result = load("bench/qk-decode-primitive-transfer/decode_owned_q8_producer_cache_object_result.json", {})
  obj = obj_result.get("object", {})
  layout = obj.get("layout", {})
  rng = np.random.default_rng(20260620)
  x = (rng.standard_normal(4096).astype(np.float32) * 0.9).astype(np.float32)
  q8 = q8_blocks_ref(x)
  xq = q8_dequant_ref(q8, 4096)
  err = np.abs(x - xq)
  first = q8[:36]
  first_scale = np.frombuffer(first[:2], dtype=np.float16).astype(np.float32)[0]
  first_sum = np.frombuffer(first[2:4], dtype=np.float16).astype(np.float32)[0]
  first_qs = np.frombuffer(first[4:36], dtype=np.int8)
  gates = {
    "object_ready": obj_result.get("gate_pass") is True,
    "q8_len_4608": len(q8) == 4608,
    "layout_matches_object": len(q8) == layout.get("total_bytes") and layout.get("block_bytes") == 36,
    "first_block_s_zero": float(first_sum) == 0.0,
    "first_block_scale_positive": float(first_scale) > 0.0,
    "qs_i8_range": int(first_qs.min()) >= -128 and int(first_qs.max()) <= 127,
    "dequant_max_abs_reasonable": float(err.max()) <= 0.02,
    "dequant_mean_abs_reasonable": float(err.mean()) <= 0.005,
    "reuse_contract_two": obj.get("reuse_count") == 2,
  }
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_OWNED_Q8_PRODUCER_CACHE_REFERENCE",
    "schema": "decode_owned_q8_producer_cache_reference_v1",
    "verdict": "PASS_DECODE_OWNED_Q8_PRODUCER_CACHE_REFERENCE_SEMANTICS" if all(gates.values()) else "BLOCKED_DECODE_OWNED_Q8_PRODUCER_CACHE_REFERENCE",
    "gate_pass": all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "reference": {
      "n": 4096,
      "q8_bytes": len(q8),
      "block_bytes": 36,
      "blocks": 128,
      "first_block": {
        "scale": float(first_scale),
        "sum": float(first_sum),
        "qs_min": int(first_qs.min()),
        "qs_max": int(first_qs.max()),
      },
      "dequant_error": {
        "max_abs": float(err.max()),
        "mean_abs": float(err.mean()),
        "p99_abs": float(np.quantile(err, 0.99)),
      },
    },
    "implementation_status": "reference_only_no_lowering",
    "next": {
      "next_probe": "owned producer/cache lowering candidate",
      "minimum_gate": "match reference semantics and lifecycle target before W==D",
      "search_status": "still blocked until candidate lowers and measures",
    },
    "gates": gates,
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "gates": gates,
    "reference": result["reference"],
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
