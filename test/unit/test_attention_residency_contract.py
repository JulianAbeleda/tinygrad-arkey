"""Structural contract for the semantic attention primitive.

This is intentionally not a fusion-performance test. It verifies the useful
pre-fusion invariant: score/probability intermediates are bounded by KV blocks,
not full T x KV tensors. Kernel-count and WMMA promotion remain separate gates.
"""
from tinygrad import Tensor, dtypes
from tinygrad.uop import Ops
from tinygrad.llm.flash_prefill_attention import shared_prefill_attention


def _primitive_buffers(t: int):
  q = Tensor.empty(1, 8, t, 128, dtype=dtypes.float16)
  k = Tensor.empty(1, 8, t, 128, dtype=dtypes.float16)
  v = Tensor.empty(1, 8, t, 128, dtype=dtypes.float16)
  attention = shared_prefill_attention(q, k, v)
  assert attention.uop.op is Ops.ATTENTION
  assert attention.uop.arg.kv_block == 64
  return [u.shape for u in attention.uop.src[1].toposort() if u.op is Ops.BUFFER and u._shape is not None]


def test_semantic_attention_primitive_has_no_full_score_or_probability_buffer():
  # 129 crosses two complete blocks and one tail, so a full T x KV temporary
  # would be visible in the Tensor graph if block ownership were lost.
  t = 129
  buffers = _primitive_buffers(t)
  assert not [shape for shape in buffers if len(shape) >= 4 and shape[-2:] == (t, t)]
