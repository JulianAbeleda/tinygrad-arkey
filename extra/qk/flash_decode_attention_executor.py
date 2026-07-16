#!/usr/bin/env python3
"""Tensor-level execution for descriptor-backed live-split flash decode."""
from __future__ import annotations

from tinygrad import Tensor, dtypes

from extra.qk.flash_decode_attention_spec import describe_flash_decode_attention


def flash_decode_live_split_block_tile(q, cache_kv, Tc_u, Hd: int, Hq: int, Hkv: int, MAXC: int, S: int,
                                       staging: str = "K_ONLY", fused_combine: bool = True, kv_scale=None, freqs=None):
  """Execute generated block-tile flash decode with live-context split geometry and return ``[Hq, Hd]``."""
  W2 = Hd + 2
  q_f = q.reshape(Hq * Hd)
  # KV-quant long-context tier dequantizes in-register; rope-at-read rotates un-roped K from freqs in-register.
  quant, rope = kv_scale is not None, freqs is not None
  inputs = (q_f, cache_kv) + ((kv_scale,) if quant else ()) + ((freqs,) if rope else ())
  spec = describe_flash_decode_attention(Hq=Hq, Hd=Hd, Hkv=Hkv, MAXC=MAXC, S=S, staging=staging,
                                         quant=quant, rope=rope)
  po = Tensor.empty(Hq * S * W2, dtype=dtypes.float32).custom_kernel(*inputs, fxn=spec.emit_tile(Tc_u))[0]
  # The old two-kernel combine was removed 2026-07-06; preserve the fail-loud contract for stale callers.
  if not fused_combine:
    raise ValueError("fused_combine=False is no longer supported for decode live-split routes")
  out = Tensor.empty(Hq * Hd, dtype=dtypes.float32).custom_kernel(po, fxn=spec.emit_combine())[0]
  return out.reshape(Hq, Hd)
