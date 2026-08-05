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
