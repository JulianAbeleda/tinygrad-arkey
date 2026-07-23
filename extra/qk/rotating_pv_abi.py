"""Opt-in ABI contract for the future LDS-rotating full-PV attention probe.

This module deliberately has no production dispatch hook.  The backend has no
verified lowering for the accumulator StateHandle yet, so callers receive an
explicit unavailable result instead of an ordinary attention fallback.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AMDAttentionRotatingPVSpec:
  acc_blocks: int = 1
  total_blocks: int = 8
  acc_lds_bytes: int = 8192
  total_lds_bytes: int = 8704

  def validate(self) -> "AMDAttentionRotatingPVSpec":
    if (self.acc_blocks, self.total_blocks, self.acc_lds_bytes, self.total_lds_bytes) != (1, 8, 8192, 8704):
      raise ValueError("rotating PV ABI requires one of eight blocks and the exact 8192/8704 byte LDS contract")
    return self


def rotating_pv_probe_unavailable(*, q_tokens: int, q_heads: int, kv_heads: int, kv_tokens: int) -> dict:
  """Fail closed until typed LDS accumulator StateHandle lowering is implemented."""
  AMDAttentionRotatingPVSpec().validate()
  if (q_tokens, q_heads, kv_heads, kv_tokens) != (512, 32, 8, 512):
    raise ValueError("rotating PV probe is exact-8B only")
  return {"schema": "tinygrad.shared_attention.rotating_pv_probe.v1", "status": "UNAVAILABLE",
          "promotion_eligible": False, "geometry": {"q_tokens": q_tokens, "q_heads": q_heads,
          "kv_heads": kv_heads, "kv_tokens": kv_tokens, "head_dim": 128},
          "spec": AMDAttentionRotatingPVSpec().__dict__,
          "reason": "typed LDS accumulator StateHandle lowering and sequential drain are not implemented"}
