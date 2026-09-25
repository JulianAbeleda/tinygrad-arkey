"""Nemotron-H prefill attention on the promoted fused flash-prefill route.

Nemotron 3 Nano's attention (Hq=40, Hkv=8, Hd=128, no RoPE) is the Qwen3-14B
geometry already admitted for the generated `custom_kernel_prefill_attention`
route (`fused_attention.ADMITTED_GRIDS`, promoted for NV sm_120 and AMD gfx1100
by `generated/custom-kernel-prefill-attention-route-policy.json`). That kernel
takes exactly 512 fp16 query rows and bakes the query start and key count into
the program, so it cannot see a symbolic position.

The chunked prefill keeps one graph per piece while the kernel geometry is fixed
per 512-token block: every piece of at most 512 tokens starting at a multiple of
its own size lies inside one block `[b*512, (b+1)*512)`. The piece's queries are
placed at their offset inside a zero 512-row query tile; the kernel runs with
query_start=b*512 over keys [0, (b+1)*512), so each real row sees exactly its
causal prefix (keys past the piece are masked for real rows) and the padding
rows are dropped. One program per block, shared by every piece size and every
prompt length.

Q, K and V enter the kernel as fp16 (its fixed ABI); the key/value buffers stay
fp32, so the state handed to decode is unchanged.
"""
from __future__ import annotations

from tinygrad import Tensor, UOp, dtypes
from tinygrad.device import Device

BLOCK = 512


def fused_prefill_supported(model, device: str | None = None) -> bool:
  """True when every attention block matches an admitted geometry on a promoted target."""
  from tinygrad.llm.fused_attention import ADMITTED_GRIDS, warm_attention_spec_target
  from tinygrad.llm.model import _custom_kernel_prefill_attn_promoted
  device = device or Device.DEFAULT
  try:
    renderer = Device[device].renderer
    backend, arch = renderer.target.device, renderer.target.arch
  except Exception:
    return False
  if not _custom_kernel_prefill_attn_promoted(backend, arch): return False
  blocks = [b for b in model.blk if b.block_type == "attention"]
  if not all((b.n_heads, b.n_kv_heads, BLOCK) in ADMITTED_GRIDS for b in blocks): return False
  if model.config.head_dim % 16 or model.config.head_dim > 128: return False
  # custom_kernel_attention resolves the target inside a Tensor function; warm the cache eagerly.
  warm_attention_spec_target(device)
  return True


def block_admitted(block: int, heads: int, kv_heads: int, width: int) -> bool:
  """Whether the kernel's descriptor validation admits block `block` (keys [0, (block+1)*512))."""
  from tinygrad.schedule.wmma.flash_prefill import FlashPrefillAttentionSpec
  keys = (block + 1) * BLOCK
  try:
    FlashPrefillAttentionSpec(Hq=heads, Hkv=kv_heads, Hd=width, q_tokens=BLOCK, kv_tokens=keys, causal=True,
                              scale=1.0, valid_kv=keys, query_start=block * BLOCK).validate()
  except ValueError:
    return False
  return True


def query_tile(heads: int, width: int) -> Tensor:
  """The persistent fp16 query tile a piece is placed into; rows outside the piece are never read back."""
  return Tensor.zeros(1, heads, BLOCK, width, dtype=dtypes.float16).contiguous().realize()


def fused_prefill_attention(q: Tensor, k: Tensor, v: Tensor, tile: Tensor, position: UOp | int, length: int,
                            block: int) -> Tensor:
  """Causal attention of `length` queries at `position` (inside block `block`) over key rows [0, position+length).

  q: (1, Hq, length, Hd); k, v: (1, Hkv, capacity, Hd) key/value buffers with rows
  [0, position+length) written, capacity >= (block+1)*512; tile: `query_tile(Hq, Hd)`.
  Returns (1, Hq, length, Hd) float32.
  """
  from tinygrad.llm.fused_attention import custom_kernel_attention
  from tinygrad.uop.ops import SharedAttentionCandidateContext
  _, heads, _, width = q.shape
  kv_heads = k.shape[1]
  if length > BLOCK or BLOCK % length: raise ValueError("piece must be a power of two no larger than the block")
  start, keys = block * BLOCK, (block + 1) * BLOCK
  if keys > k.shape[2]: raise ValueError("key/value capacity must cover the whole 512-token block")
  offset = position - start
  if length == BLOCK: queries = q.cast(dtypes.float16)
  else:
    tile[:, :, offset:offset + length].assign(q.cast(dtypes.float16)).realize()
    queries = tile
  ctx = SharedAttentionCandidateContext("custom_kernel_prefill_attn", "DIRECT_PACKED_FALLBACK", BLOCK, keys, start,
                                        heads, kv_heads, width, True)
  out = custom_kernel_attention(queries, k[:, :, :keys].cast(dtypes.float16), v[:, :, :keys].cast(dtypes.float16),
                                scale=None, causal=True, ctx=ctx)
  return (out if length == BLOCK else out[:, :, offset:offset + length]).float()


def grouped_prefill_attention(q: Tensor, k: Tensor, v: Tensor, position: UOp | int, keys: int) -> Tensor:
  """Causal attention over key rows [0, keys) as two fp16 tensor-core matmuls with fp32 accumulation.

  The interim for blocks the flash kernel does not admit: an ordinary scheduler expression (no
  custom kernel) that keeps a symbolic position. Each KV head's query group is one matmul operand,
  so K/V are never repeated. Returns (1, Hq, length, Hd) float32.
  """
  _, heads, length, width = q.shape
  kv_heads = k.shape[1]
  group = heads // kv_heads
  rows = (Tensor.arange(length) + position).reshape(1, 1, 1, length, 1)
  mask = (Tensor.arange(keys).reshape(1, 1, 1, 1, keys) <= rows).where(0.0, float("-inf"))
  queries = q.reshape(1, kv_heads, group, length, width).cast(dtypes.float16)
  k16 = k[:, :, :keys].cast(dtypes.float16).reshape(1, kv_heads, 1, keys, width)
  v16 = v[:, :, :keys].cast(dtypes.float16).reshape(1, kv_heads, 1, keys, width)
  scores = queries.matmul(k16.transpose(-1, -2), dtype=dtypes.float) * (width ** -0.5) + mask
  return scores.softmax(-1).cast(dtypes.float16).matmul(v16, dtype=dtypes.float).reshape(1, heads, length, width)


__all__ = ["BLOCK", "block_admitted", "fused_prefill_supported", "fused_prefill_attention", "grouped_prefill_attention",
           "query_tile"]
