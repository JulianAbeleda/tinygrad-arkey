"""Shared-prefix, length-bounded decode attention for the Nemotron-H batched sampler.

A group of B sequences samples from one prompt. The prompt's keys and values
are therefore stored once, as `[1, kv_heads, prefix_capacity, head_dim]`, and
every sequence owns only its generated suffix, `[B, kv_heads, suffix_capacity,
head_dim]`. One decode step attends over two segments:

- prefix: all B queries against the shared prompt keys, over a static span
  (the prefix capacity, or a caller-chosen prompt bucket) with the prompt
  length as a symbolic mask bound, so one graph serves every prompt length.
  The batch is folded into the query-row axis next to the GQA group, so each
  key/value head is one `[B*group, head_dim] x [head_dim, span]` GEMM and the
  prefix is read once for the whole batch rather than once per sequence;
- suffix: each sequence's own keys, bounded by a length bucket (powers of two
  up to the capacity; one compiled graph per bucket) instead of the capacity.

Each segment keeps its row max, its sum of exponentials and its unnormalized
value sum, and the two are merged with an ordinary log-sum-exp combine. The
result equals softmax attention over the concatenated `[prefix, suffix]` keys
with a causal mask. Nemotron-H attention has no rotary embedding, so a key's
position never enters the arithmetic and prefix and suffix positions need no
bookkeeping beyond the mask.

Everything here is Tensor code; no kernel is hand-written.
"""
from __future__ import annotations

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import DType


def _fresh(shape: tuple[int, ...], dtype: DType) -> Tensor:
  """A new plain zeroed buffer; TinyJit replays only buffers, never lazy constants or views."""
  return Tensor.empty(*shape, dtype=dtype).assign(Tensor.zeros(*shape, dtype=dtype)).realize()


def suffix_buckets(capacity: int, minimum: int = 64) -> list[int]:
  """Suffix read lengths: powers of two from `minimum`, ending exactly at `capacity`."""
  if capacity < 1 or minimum < 1:
    raise ValueError("capacity and minimum must be positive")
  sizes, size = [], minimum
  while size < capacity:
    sizes.append(size)
    size *= 2
  return sizes + [capacity]


def suffix_bucket(step: int, capacity: int, minimum: int = 64) -> int:
  """The smallest bucket that covers suffix rows [0, step]."""
  if not 0 <= step < capacity:
    raise ValueError(f"suffix step {step} outside [0, {capacity})")
  return next(size for size in suffix_buckets(capacity, minimum) if size > step)


def _partials(scores: Tensor, values: Tensor, chunk: int) -> tuple[Tensor, Tensor, Tensor]:
  """Per-chunk row max, sum of exponentials and unnormalized value sum of one masked key segment.

  scores `[..., rows, N]` and values `[..., N, hd]` are split into `N // chunk`
  key chunks (one when `chunk` does not divide N), the ordinary-Tensor form of
  flash decode's split-K: every reduction runs over `chunk` keys and the
  chunks are parallel work, instead of one serial reduction over the whole
  segment per output. Returns `[..., splits, rows, 1]`, `[..., splits, rows, 1]`
  and `[..., splits, rows, hd]`.

  Each partial is materialized once: left to fuse, the scheduler recomputes
  the score product inside every later reduction. Scores and sums stay
  float32; the probability-value product takes the values' storage dtype for
  both operands and accumulates in float32, as the score product does, so
  bf16 storage reaches the tensor cores.
  """
  *lead, rows, span = scores.shape
  chunk = chunk if 0 < chunk < span and span % chunk == 0 else span
  splits = span // chunk
  scores = scores.reshape(*lead, rows, splits, chunk).transpose(-2, -3).contiguous()
  values = values.reshape(*lead, splits, chunk, values.shape[-1])
  peak = scores.max(-1, keepdim=True).contiguous()
  # a fully masked chunk (past the prompt or the step) has peak -inf: shift by 0 so its weights are 0, not NaN
  weights = (scores - (peak == float("-inf")).where(0.0, peak)).exp()
  return (peak, weights.sum(-1, keepdim=True).contiguous(),
          weights.cast(values.dtype).dot(values, dtype=dtypes.float).contiguous())


def two_segment_attention(q: Tensor, prefix_k: Tensor, prefix_v: Tensor, suffix_k: Tensor, suffix_v: Tensor,
                          prefix_length: UOp | int, step: UOp | int, chunk: int = 256) -> Tensor:
  """One decode query per sequence against a shared prefix and its own suffix.

  q: `[B, heads, 1, hd]`. prefix_k/v: `[1, kv_heads, P, hd]`, rows `[0, prefix_length)`
  valid. suffix_k/v: `[B, kv_heads, S, hd]`, rows `[0, step]` valid. Rows past
  those are masked, so both spans are static shapes and one compiled graph
  serves every prompt length and every step inside a suffix bucket. Returns
  `[B, heads, 1, hd]` in float32; head h reads key/value head
  `h // (heads // kv_heads)`, as `scaled_dot_product_attention(enable_gqa=True)`.
  """
  batch, heads, length, width = q.shape
  kv_heads = prefix_k.shape[1]
  if length != 1 or heads % kv_heads:
    raise ValueError("expected one query per sequence and a whole GQA group per key/value head")
  group = heads // kv_heads
  q = (q.float() * (1.0 / width ** 0.5)).reshape(batch, kv_heads, group, width)
  # prefix: batch folded next to the group, [1, kv_heads, B*group, hd], so each key/value head is one
  # GEMM that reads the shared keys once for the whole batch
  rows = q.permute(1, 0, 2, 3).reshape(1, kv_heads, batch * group, width)
  span = prefix_k.shape[2]
  scores = rows.cast(prefix_k.dtype).dot(prefix_k.transpose(-1, -2), dtype=dtypes.float)
  scores = (Tensor.arange(span) < prefix_length).reshape(1, 1, 1, span).where(scores, float("-inf"))
  prefix = [t.reshape(kv_heads, -1, batch, group, t.shape[-1]).permute(2, 0, 1, 3, 4)
            for t in _partials(scores, prefix_v, chunk)]
  # suffix: each sequence's own keys, rows after `step` masked
  span = suffix_k.shape[2]
  scores = q.cast(suffix_k.dtype).dot(suffix_k.transpose(-1, -2), dtype=dtypes.float)
  scores = (Tensor.arange(span) <= step).reshape(1, 1, 1, span).where(scores, float("-inf"))
  suffix = _partials(scores, suffix_v, chunk)
  # log-sum-exp combine over every chunk of both segments; a fully masked chunk has max -inf and weight 0
  peaks, sums, outs = (p.cat(s, dim=2) for p, s in zip(prefix, suffix))
  peak = peaks.max(2, keepdim=True)
  scale = (peaks - peak).exp()
  out = (outs * scale).sum(2) / (sums * scale).sum(2)
  return out.reshape(batch, heads, 1, width)


class SharedPrefixKV:
  """One attention layer's shared prompt keys/values and per-sequence suffix keys/values."""

  def __init__(self, batch: int, kv_heads: int, head_dim: int, prefix_capacity: int, suffix_capacity: int,
               dtype: DType = dtypes.bfloat16, chunk: int = 256):
    self.batch, self.prefix_capacity, self.suffix_capacity, self.chunk = batch, prefix_capacity, suffix_capacity, chunk
    self.prefix = {key: _fresh((1, kv_heads, prefix_capacity, head_dim), dtype) for key in ("k", "v")}
    self.suffix = {key: _fresh((batch, kv_heads, suffix_capacity, head_dim), dtype) for key in ("k", "v")}

  def load_prefix(self, k: Tensor, v: Tensor):
    """Copy prompt keys/values `[1, kv_heads, n, hd]` into rows [0, min(n, capacity)); rows past the prompt are never read."""
    for key, value in (("k", k), ("v", v)):
      rows = min(value.shape[2], self.prefix_capacity)
      self.prefix[key][:, :, :rows].assign(value[:, :, :rows].cast(self.prefix[key].dtype)).realize()

  def write(self, k: Tensor, v: Tensor, step: UOp | int):
    """Store each sequence's new key/value `[B, kv_heads, 1, hd]` at suffix row `step`."""
    for key, value in (("k", k), ("v", v)):
      self.suffix[key][:, :, step:step + 1].assign(value.cast(self.suffix[key].dtype)).realize()

  def attend(self, q: Tensor, prefix_length: UOp | int, step: UOp | int, bucket: int,
             prefix_span: int | None = None) -> Tensor:
    """Attention over prompt rows [0, prefix_length) and suffix rows [0, step].

    Reads `bucket` suffix rows and `prefix_span` prompt rows (default: the
    prefix capacity); both are static, the lengths only enter the masks.
    """
    prefix_span = self.prefix_capacity if prefix_span is None else prefix_span
    if not 0 < bucket <= self.suffix_capacity or not 0 < prefix_span <= self.prefix_capacity:
      raise ValueError(f"bucket {bucket} or prefix span {prefix_span} outside the capacities")
    return two_segment_attention(q, self.prefix["k"][:, :, :prefix_span], self.prefix["v"][:, :, :prefix_span],
                                 self.suffix["k"][:, :, :bucket], self.suffix["v"][:, :, :bucket], prefix_length, step,
                                 self.chunk)


def shared_kv_for_model(model, batch: int, prefix_capacity: int, suffix_capacity: int,
                        dtype: DType | None = None) -> list[SharedPrefixKV | None]:
  """One `SharedPrefixKV` per attention block, `None` elsewhere, aligned with `model.blk`.

  The default storage dtype is the key projection's weight dtype: bf16 for the
  released checkpoint, the precision HF and vLLM keep keys and values in.
  """
  head_dim = model.config.head_dim
  return [SharedPrefixKV(batch, block.n_kv_heads, head_dim, prefix_capacity, suffix_capacity,
                         block.attn_k.weight.dtype if dtype is None else dtype)
          if block.block_type == "attention" else None for block in model.blk]


def load_prefix_from_prefill(layers: list[SharedPrefixKV | None], prefill_buffers: list[dict | None]):
  """Fill each layer's shared prefix from `NemotronHPrefill.buffers` (or `model.prefix` caches).

  The prefill's attention buffers hold the prompt in rows [0, len(prompt));
  the whole overlap with the prefix capacity is copied, so the copy compiles
  once per configuration rather than once per prompt length.
  """
  for layer, buffer in zip(layers, prefill_buffers):
    if layer is not None:
      layer.load_prefix(buffer["k"], buffer["v"])


def decode_attention(block, normed: Tensor, layer: SharedPrefixKV, prefix_length: UOp | int, step: UOp | int,
                     bucket: int, prefix_span: int | None = None) -> Tensor:
  """One decode step of an attention block on normed hidden `[B, 1, dim]`; returns the output projection `[B, 1, dim]`.

  The new key/value is stored before attention reads the suffix.
  """
  batch, width = normed.shape[0], block.attn_q.weight.shape[0] // block.n_heads
  q = block.attn_q(normed).reshape(batch, 1, block.n_heads, width).transpose(1, 2)
  k = block.attn_k(normed).reshape(batch, 1, block.n_kv_heads, width).transpose(1, 2)
  v = block.attn_v(normed).reshape(batch, 1, block.n_kv_heads, width).transpose(1, 2)
  layer.write(k, v, step)
  attended = layer.attend(q, prefix_length, step, bucket, prefix_span)
  return block.attn_output(attended.transpose(1, 2).reshape(batch, 1, -1))


__all__ = ["SharedPrefixKV", "decode_attention", "load_prefix_from_prefill", "shared_kv_for_model", "suffix_bucket",
           "suffix_buckets", "two_segment_attention"]
