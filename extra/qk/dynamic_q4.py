"""Dynamic-base Q4 helpers.

The functions in this file are intentionally UOp-level: ``base`` is allowed
to be a RANGE/loop value and is never converted to a Python slice.
"""
from __future__ import annotations
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import Ops, UOp


def _dynamic_load(words: UOp, index: UOp) -> UOp:
  """Load flat storage without going through Tensor's view/index machinery."""
  return words.index(index, ptr=True).load()


def q4_code(words: UOp, base: UOp, group: int, pos: UOp) -> UOp:
  """Return one Q4_K nibble from a block whose word base is dynamic."""
  if not 0 <= group < 8: raise ValueError("Q4_K group must be in [0, 8)")
  qword = _dynamic_load(words, base + 4 + (group // 2) * 8 + pos // 4)
  return qword.rshift((pos % 4) * 8 + (group % 2) * 4).bitwise_and(0xf)


def q4_scale_min(words: UOp, base: UOp, group: int) -> tuple[UOp, UOp, UOp, UOp]:
  """Return d, dmin, scale-code and min-code using a dynamic block base."""
  if not 0 <= group < 8: raise ValueError("Q4_K group must be in [0, 8)")
  meta = base + 1
  def byte(i: int) -> UOp: return _dynamic_load(words, meta + i // 4).rshift((i % 4) * 8).bitwise_and(0xff)
  w0 = _dynamic_load(words, base)
  d = w0.bitwise_and(0xffff).cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)
  dm = w0.rshift(16).bitwise_and(0xffff).cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)
  if group < 4: sc, mn = byte(group).bitwise_and(63), byte(4 + group).bitwise_and(63)
  else:
    hi = byte(8 + group - 4)
    sc = hi.bitwise_and(15).bitwise_or(byte(group - 4).rshift(6).lshift(4))
    mn = hi.rshift(4).bitwise_or(byte(4 + group - 4).rshift(6).lshift(4))
  return d, dm, sc, mn


def q4_dequant(words: UOp, base: UOp, group: int, pos: UOp) -> UOp:
  d, dm, sc, mn = q4_scale_min(words, base, group)
  q = q4_code(words, base, group, pos).cast(dtypes.float32)
  return d * sc.cast(dtypes.float32) * q - dm * mn.cast(dtypes.float32)


def activation_dequant(values: UOp, scales: UOp, base: UOp, pos: UOp, *, group_width: int = 32) -> UOp:
  """Dequantize Q8 activation values with a dynamic element base."""
  return values[base + pos].cast(dtypes.float32) * scales[base // group_width].cast(dtypes.float32)


def dynamic_effect_graph(weights: Tensor, activation: Tensor, output: Tensor, tile: UOp) -> UOp:
  """Small audit graph with the dequant chain rooted at the dynamic store.

  Keep this seam UOp-only after obtaining the backing pointers.  In
  particular, don't build Tensor indexing/view expressions here: those are
  eager graph nodes and can leave CONTIGUOUS/RESHAPE nodes outside the effect
  chain when ``tile`` is a loop value.
  """
  # Materialize inputs at this audit boundary so lazy Tensor constructors
  # (for example Tensor.zeros) cannot leak their scalar reshape/view plumbing
  # into the loop-owned effect graph.
  weights, activation, output = weights.realize(), activation.realize(), output.realize()
  w = q4_dequant(weights.uop, tile, 3, tile)
  a = activation_dequant(activation.uop, activation.uop, tile, tile)
  acc = UOp.const(dtypes.float32.vec(1), (0.0,))
  mma = UOp(Ops.WMMA, dtypes.float32.vec(1), (w, a, acc), arg=((1, 1, 1), "cpu", 1))
  return UOp.sink(output.uop.index(tile, ptr=True).store(mma))


__all__ = ["q4_code", "q4_scale_min", "q4_dequant", "activation_dequant", "dynamic_effect_graph"]
