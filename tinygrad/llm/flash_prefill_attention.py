"""Shared semantic prefill-attention entry point.

The model routes meet here after Q/K/V projection. The operation deliberately
does not encode model, weight-format, or target-specific policy.
"""
from tinygrad import Tensor

def shared_prefill_attention(q:Tensor, k:Tensor, v:Tensor, *, scale:float|None=None,
                             mask:Tensor|None=None, causal:bool=False) -> Tensor:
  # Qwen uses grouped-query attention: Q has Hq heads while K/V have Hkv
  # heads. Keep this layout normalization at the one shared boundary so the
  # attention primitive stays independent of model and weight route.
  if q.shape[-3] != k.shape[-3]:
    if not isinstance(q.shape[-3], int) or not isinstance(k.shape[-3], int) or q.shape[-3] % k.shape[-3]:
      raise ValueError(f"attention requires matching heads or integral GQA groups, got Hq={q.shape[-3]}, Hkv={k.shape[-3]}")
    groups = q.shape[-3] // k.shape[-3]
    k, v = k.repeat_interleave(groups, dim=-3), v.repeat_interleave(groups, dim=-3)
  return q._semantic_attention(k, v, attn_mask=mask, is_causal=causal, scale=scale)
