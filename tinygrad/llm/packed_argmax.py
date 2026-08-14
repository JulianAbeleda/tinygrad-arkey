"""Ordinary-UOp, finite-fp32 argmax used by the closed decode sampler experiment.

The normal :meth:`Tensor.argmax` is deliberately unchanged.  This helper is
only safe for finite float32 values: it turns each value and its *first* index
into one unsigned 64-bit reduction key.  It has no renderer source, custom
kernel, or model-shape knowledge.
"""
from __future__ import annotations

from tinygrad import Tensor, dtypes


def packed_argmax_finite_fp32(x:Tensor, axis:int=-1, keepdim:bool=False) -> Tensor:
  """Return first-index argmax for a finite fp32 tensor with one ordinary MAX.

  IEEE binary32 bit patterns have a monotonic unsigned ordering after flipping
  the sign bit for nonnegative values and complementing negative values.  Zero
  is canonicalized first so ``-0.0`` and ``+0.0`` tie exactly.  The low half of
  the key is ``(N-1-index)``; thus an unsigned max chooses the earliest index
  on equal values.

  This intentionally rejects non-fp32 inputs and dynamic/empty/too-wide axes.
  NaN policy is not silently invented here: callers must have a finite-value
  qualification before using the route.
  """
  if x.dtype != dtypes.float32: raise ValueError(f"packed argmax needs float32, got {x.dtype}")
  if x.ndim == 0: raise ValueError("packed argmax needs a non-scalar tensor")
  axis = axis + x.ndim if axis < 0 else axis
  if not 0 <= axis < x.ndim: raise ValueError(f"axis {axis} out of range for rank {x.ndim}")
  n = x.shape[axis]
  if not isinstance(n, int) or not 0 < n <= 0x1_0000_0000:
    raise ValueError(f"packed argmax needs static 1..2^32 axis, got {x.shape[axis]!r}")
  # Moving the reduced dimension to the end keeps the index construction
  # generic for all axes, then restore the requested output layout.
  y = x.transpose(axis, -1)
  bits = y.bitcast(dtypes.uint32)
  bits = y.eq(0.0).where(0, bits)  # signed zero must be one tie class
  ordered = (bits >> 31).where(~bits, bits ^ 0x80000000)
  inv_index = (n-1-Tensor.arange(n, dtype=dtypes.uint32).to(x.device)).cast(dtypes.uint64)
  key = (ordered.cast(dtypes.uint64) << 32) | inv_index
  index = (n-1-(key.max(axis=-1, keepdim=keepdim) & 0xffffffff)).cast(dtypes.int32)
  # Moving ``axis`` to the final position changes the order of every dimension
  # after it; restore that rotation (not merely a swap for rank > 3).
  if axis == x.ndim-1: return index
  if keepdim:
    order = tuple(range(axis)) + (x.ndim-1,) + tuple(range(axis+1, x.ndim-1)) + (axis,)
  else:
    order = tuple(range(axis)) + tuple(range(axis+1, x.ndim-1)) + (axis,)
  return index.permute(order)


def packed_argmax_tile_keys_fp32(x:Tensor, tile_rows:int, axis:int=-1) -> Tensor:
  """Per-tile (max, index) packed u64 keys for the fused vocab-head top-1 (P1).

  The vocab aux scatter-chain fusion (nv-vocab-aux-chain-fusion-scope-20260812.md) carries
  one (max, index) per GEMV warp tile instead of materialising the 151936-row logits chain.
  This is the ordinary-UOp mirror of that in-kernel carry: the reduced axis is grouped into
  ``tile_rows``-wide tiles, each tile becomes one u64 key ``(ordered fp32 bits) << 32 |
  (n-1-index)`` (the same monotonic IEEE reordering as :func:`packed_argmax_finite_fp32`,
  including the signed-zero canonicalisation), and a u64 MAX over the tile axis keeps the
  largest logit while the inverted index in the low half breaks ties to the FIRST index.
  The returned keys are the per-tile carry; the cross-tile reduce is one more u64 MAX
  (see :func:`packed_argmax_from_tile_keys`).  No float cast participates in the max+idx
  compare: the values only leave fp32 through a bit-exact BITCAST and integer ops.
  """
  if x.dtype != dtypes.float32: raise ValueError(f"packed argmax needs float32, got {x.dtype}")
  if x.ndim == 0: raise ValueError("packed argmax needs a non-scalar tensor")
  axis = axis + x.ndim if axis < 0 else axis
  if not 0 <= axis < x.ndim: raise ValueError(f"axis {axis} out of range for rank {x.ndim}")
  if not isinstance(tile_rows, int) or tile_rows < 1:
    raise ValueError(f"tile_rows must be a positive integer, got {tile_rows!r}")
  n = x.shape[axis]
  if not isinstance(n, int) or not 0 < n <= 0x1_0000_0000:
    raise ValueError(f"packed argmax needs static 1..2^32 axis, got {x.shape[axis]!r}")
  if n % tile_rows != 0: raise ValueError(f"axis size {n} must be divisible by tile_rows {tile_rows}")
  # Moving the reduced dimension to the end keeps the tile grouping generic for all axes,
  # then the tile axis is restored to the requested output position.
  y = x.transpose(axis, -1)
  bits = y.bitcast(dtypes.uint32)
  bits = y.eq(0.0).where(0, bits)  # signed zero must be one tie class
  ordered = (bits >> 31).where(~bits, bits ^ 0x80000000)
  inv_index = (n-1-Tensor.arange(n, dtype=dtypes.uint32).to(x.device)).cast(dtypes.uint64)
  keys = (ordered.cast(dtypes.uint64) << 32) | inv_index
  tiles = keys.reshape(y.shape[:-1] + (n // tile_rows, tile_rows))
  tile_keys = tiles.max(axis=-1)  # per-tile (max, index); first-index-wins on equal logits
  if axis == x.ndim-1: return tile_keys
  order = tuple(range(axis)) + (x.ndim-1,) + tuple(range(axis+1, x.ndim-1)) + (axis,)
  return tile_keys.permute(order)


def packed_argmax_from_tile_keys(tile_keys:Tensor, n:int, axis:int=-1, keepdim:bool=False) -> Tensor:
  """Final packed reduce over per-tile keys: one u64 MAX + unpack to the first-index token id.

  Consumes the per-tile (max, index) keys produced by :func:`packed_argmax_tile_keys_fp32`
  (or the vocab GEMV epilogue's in-kernel carry) and finishes the fused top-1: the single
  MAX over the tile axis keeps the largest packed key, and ``n-1-(key & 0xffffffff)`` unpacks
  the winning row.  Tie semantics are identical to :func:`packed_argmax_finite_fp32` and to
  today's ``r_16_8`` Tensor.argmax chain (first index wins).
  """
  if tile_keys.dtype != dtypes.uint64: raise ValueError(f"tile keys must be uint64, got {tile_keys.dtype}")
  if tile_keys.ndim == 0: raise ValueError("tile keys must be a non-scalar tensor")
  axis = axis + tile_keys.ndim if axis < 0 else axis
  if not 0 <= axis < tile_keys.ndim: raise ValueError(f"axis {axis} out of range for rank {tile_keys.ndim}")
  if not isinstance(n, int) or not 0 < n <= 0x1_0000_0000:
    raise ValueError(f"packed argmax needs static 1..2^32 axis, got n={n!r}")
  y = tile_keys.transpose(axis, -1)
  index = (n-1-(y.max(axis=-1, keepdim=keepdim) & 0xffffffff)).cast(dtypes.int32)
  if axis == tile_keys.ndim-1: return index
  if keepdim:
    order = tuple(range(axis)) + (tile_keys.ndim-1,) + tuple(range(axis+1, tile_keys.ndim-1)) + (axis,)
  else:
    order = tuple(range(axis)) + tuple(range(axis+1, tile_keys.ndim-1)) + (axis,)
  return index.permute(order)


def packed_argmax_tiles_fp32(x:Tensor, tile_rows:int, axis:int=-1, keepdim:bool=False) -> Tensor:
  """Fused per-tile (max, index) top-1 in one ordinary-UOp graph (vocab aux chain fusion, P1).

  Convenience composition of :func:`packed_argmax_tile_keys_fp32` and
  :func:`packed_argmax_from_tile_keys`: the exact two-stage reduce the fused epilogue
  performs (per-tile packed key, then one cross-tile MAX), bit-exact with
  :func:`packed_argmax_finite_fp32` and with today's Tensor.argmax chain.
  """
  keys = packed_argmax_tile_keys_fp32(x, tile_rows, axis=axis)
  return packed_argmax_from_tile_keys(keys, x.shape[axis], axis=axis, keepdim=keepdim)
