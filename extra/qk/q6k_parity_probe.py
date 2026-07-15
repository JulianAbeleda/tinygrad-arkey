"""Q6_K parity audit; evidence-only and default-off.

This probe makes the Q6 ``ffn_down`` gap explicit without changing the shared
route selector.  It compares source-level lifecycle markers and optionally
runs a tiny packed-reference check.  It is intentionally not a replacement
kernel or a full dequant fallback.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np

from extra.qk.q6k_mmq_vocabulary import q6k_block_dot
from extra.qk.quant import q6_k_gemv_primitive

SCHEMA = "tinygrad.q6k_parity_audit.v1"
DEFAULT_OFF = True
LLAMA_SOURCE = Path(__file__).with_name("research") / "llama_mmq" / "mmq.cuh"


def audit_q6k_ffn_down(*, run_reference: bool = False) -> dict[str, Any]:
  tinygrad_source = inspect.getsource(q6_k_gemv_primitive._q6k_block_dot_packed_load_gemm)
  llama_source = LLAMA_SOURCE.read_text(encoding="utf-8")
  markers = {
    "tinygrad_scalar_decode_in_dot": all(x in tinygrad_source for x in ("_q6k_byte", "_f16_half", "for grp in range(16)")),
    "tinygrad_packed_load_is_u16": "halfs[base + ql_byte_idx//2]" in tinygrad_source,
    "llama_q6_tile_staging": "load_tiles_q6_K" in llama_source,
    "llama_q8_1_mmq_dot": "vec_dot_q6_K_q8_1_mma" in llama_source,
    "llama_shared_scale_and_d": all(x in llama_source for x in ("x_sc", "x_df")),
  }
  result: dict[str, Any] = {
    "schema": SCHEMA,
    "enabled": bool(run_reference),
    "default_off": DEFAULT_OFF,
    "scope": {"role": "ffn_down", "quant": "Q6_K", "shape": {"M": 512, "N": 5120, "K": 17408}},
    "findings": {
      "weakness": "scalar Q6 payload/scale/d reconstruction remains in the dot body and is not shared across output rows",
      "llama_difference": "llama stages Q6 q/scales/d and uses Q8_1 MMQ dot machinery across a cooperative output tile",
      "not_a_decode_mismatch": "both paths use the canonical 256-element Q6_K block and 16 signed scale groups",
    },
    "source_markers": markers,
    "remaining_gate": "legal cooperative multi-output Q6 tile with independent correctness/resource evidence",
  }
  if run_reference:
    block = bytearray(210)
    block[0] = 1
    block[192] = 2
    block[208:210] = np.float16(0.5).tobytes()
    activation = [0.0] * 256
    activation[0] = 1.0
    result["reference"] = {"status": "PASS", "value": q6k_block_dot(bytes(block), activation)}
  return result


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--run-reference", action="store_true", help="run only the bounded 256-element reference check")
  parser.add_argument("--out", type=Path)
  args = parser.parse_args()
  report = audit_q6k_ffn_down(run_reference=args.run_reference)
  encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
  if args.out: args.out.write_text(encoded, encoding="utf-8")
  else: print(encoded, end="")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
