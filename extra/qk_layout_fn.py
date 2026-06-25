#!/usr/bin/env python3
"""Queryable layout/index IR for static coalescing and composition.

This is the M1 LayoutFn from docs/layout-codegen-full-scope-20260625.md.  It deliberately stays outside model wiring:
it wraps an Ops.INDEX UOp (or a raw index expression) and exposes conservative affine queries over RANGE variables.
Unsafe CuTe-style cases that would produce silent wrong strides, notably masked PAD/WHERE and mixed-radix div/mod
RESHAPE expressions, raise LayoutFnError instead of returning a misleading coefficient.
"""
from __future__ import annotations
from dataclasses import dataclass

from tinygrad.dtype import dtypes, Invalid
from tinygrad.uop.ops import AxisType, Ops, UOp, graph_rewrite
from tinygrad.uop.symbolic import symbolic

_UNSAFE_INDEX_OPS = {Ops.WHERE, Ops.CDIV, Ops.CMOD, Ops.FLOORDIV, Ops.FLOORMOD}

class LayoutFnError(ValueError): pass


def _idx_from_index(u:UOp) -> UOp:
  if u.op is Ops.INDEX:
    if len(u.src) != 2: raise LayoutFnError(f"LayoutFn supports single-index INDEX nodes, got {len(u.src)-1} indices")
    _assert_admissible_raw(u.src[1])
    return u.src[1].get_idx()
  _assert_admissible_raw(u)
  return u.get_idx() if u.dtype.scalar() is dtypes.weakint else u


def _has_invalid(u:UOp) -> bool:
  return any(x.op is Ops.CONST and x.arg is Invalid for x in u.toposort())


def _assert_admissible_raw(idx:UOp) -> None:
  bad = [x.op for x in idx.toposort() if x.op in _UNSAFE_INDEX_OPS]
  if bad or _has_invalid(idx):
    raise LayoutFnError(f"non-affine or masked layout expression is not admissible: {bad or ['Invalid']}")

def _assert_admissible(idx:UOp) -> None:
  _assert_admissible_raw(idx)


def _coeff(idx:UOp, rng:UOp) -> int|None:
  _assert_admissible(idx)
  d = graph_rewrite(idx.substitute({rng: rng.const_like(1)}) - idx.substitute({rng: rng.const_like(0)}), symbolic)
  return int(d.arg) if d.op is Ops.CONST else None


@dataclass(frozen=True)
class LayoutFn:
  """Conservative affine layout object over an INDEX address expression."""
  idx: UOp

  @staticmethod
  def from_index(index:UOp) -> "LayoutFn":
    return LayoutFn(graph_rewrite(_idx_from_index(index), symbolic))

  @staticmethod
  def from_expr(idx:UOp) -> "LayoutFn":
    return LayoutFn(graph_rewrite(idx.get_idx(), symbolic))

  @staticmethod
  def range(size:int, axis:int=0, axistype:AxisType=AxisType.LOOP) -> UOp:
    return UOp.range(size, axis, axistype)

  @property
  def ranges(self) -> tuple[UOp, ...]:
    return tuple(r for r in self.idx.ranges if r.op is Ops.RANGE)

  def coeff(self, rng:UOp) -> int|None:
    return _coeff(self.idx, rng)

  def is_unit_stride(self, rng:UOp) -> bool:
    return self.coeff(rng) == 1

  def compose(self, other:"LayoutFn", rng:UOp|None=None) -> "LayoutFn":
    """Return A(B(c)) by substituting `other.idx` into exactly one RANGE of this layout.

    `rng` can be supplied for multi-range layouts.  Without it, the outer layout must contain exactly one RANGE.
    Both sides are checked for affine/admissible forms before and after substitution.
    """
    _assert_admissible(self.idx)
    _assert_admissible(other.idx)
    outer_ranges = self.ranges
    if rng is None:
      if len(outer_ranges) != 1: raise LayoutFnError(f"compose requires rng for {len(outer_ranges)} outer ranges")
      rng = outer_ranges[0]
    if rng not in outer_ranges: raise LayoutFnError("compose rng is not in outer layout")
    composed = graph_rewrite(self.idx.substitute({rng: other.idx}), symbolic)
    _assert_admissible(composed)
    return LayoutFn(composed)
