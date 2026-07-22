"""Shared semantic prefill-attention entry point.

The model routes meet here after Q/K/V projection. The operation deliberately
does not encode model, weight-format, or target-specific policy.
"""
from tinygrad import Tensor

def shared_prefill_attention(q:Tensor, k:Tensor, v:Tensor, *, scale:float|None=None,
                             mask:Tensor|None=None, causal:bool=False) -> Tensor:
  return q._semantic_attention(k, v, attn_mask=mask, is_causal=causal, scale=scale)
