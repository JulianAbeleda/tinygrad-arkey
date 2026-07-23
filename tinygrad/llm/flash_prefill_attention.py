"""Shared semantic prefill-attention entry point.

The model routes meet here after Q/K/V projection. The operation deliberately
does not encode model, weight-format, or target-specific policy.
"""
from tinygrad import Tensor
from tinygrad.uop.ops import AMDAttentionGridSpec, SharedAttentionCandidateContext

def shared_prefill_attention(q:Tensor, k:Tensor, v:Tensor, *, scale:float|None=None,
                             mask:Tensor|None=None, causal:bool=False,
                             candidate_context:SharedAttentionCandidateContext|None=None) -> Tensor:
  # Preserve native GQA ownership for the exact gfx1100 prefill proof. The
  # semantic fallback expands only inside Tensor._semantic_attention, so this
  # boundary and its ATTENTION sources remain Q/Hq and K,V/Hkv.
  grid = None
  if q.shape[-3] != k.shape[-3]:
    if not isinstance(q.shape[-3], int) or not isinstance(k.shape[-3], int) or q.shape[-3] % k.shape[-3]:
      raise ValueError(f"attention requires matching heads or integral GQA groups, got Hq={q.shape[-3]}, Hkv={k.shape[-3]}")
    groups = q.shape[-3] // k.shape[-3]
    if all(isinstance(x, int) for x in (q.shape[0], q.shape[-2], q.shape[-1], k.shape[-2], q.shape[-3], k.shape[-3])) and q.shape[0] == 1:
      candidate = AMDAttentionGridSpec(q_tokens=q.shape[-2], q_heads=q.shape[-3], kv_heads=k.shape[-3],
        group_ratio=groups, kv_tokens=k.shape[-2], head_dim=q.shape[-1])
      try:
        candidate.validate()
        if (candidate.q_heads, candidate.kv_heads, candidate.q_tokens) in {(32, 8, 512), (40, 8, 512)}: grid = candidate
      except ValueError: pass
    # Preserve the established eager layout normalization outside the exact
    # opt-in proof. Those calls have no native descriptor and therefore must
    # remain byte-for-byte on the ordinary SDPA path.
    if grid is None: k, v = k.repeat_interleave(groups, dim=-3), v.repeat_interleave(groups, dim=-3)
  if candidate_context is not None:
    candidate_context.validate()
    if grid is None or (candidate_context.q_tokens,candidate_context.kv_tokens,candidate_context.hq,candidate_context.hkv,candidate_context.hd) != \
       (q.shape[-2],k.shape[-2],q.shape[-3],k.shape[-3],q.shape[-1]):
      raise ValueError("shared attention candidate context does not match the admitted GQA geometry")
  return q._semantic_attention(k, v, attn_mask=mask, is_causal=causal, scale=scale, attention_grid=grid,
                               attention_context=candidate_context)
