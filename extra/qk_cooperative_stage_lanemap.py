#!/usr/bin/env python3
"""Cooperative-staging LaneMap -- M2 of the layout/mapping IR (docs/layout-mapping-ir-design-20260625.md,
docs/decode-coalesced-load-primitive-result-20260626.md).

The coalesced-load lowering primitive (extra/qk_coalesced_load_lowering.py) vectorizes a load only when its
contiguous dimension is a small LOOP/REDUCE axis. A cooperative LDS staging that maps `i = stage*THREADS + tid`
(one element per thread per stage) puts the contiguous dimension on the *thread* (`tid`), not a loop axis -- so
the global load stays scalar (`global_load_d16=0`). This is a thread->element MAPPING problem, i.e. a LaneMap.

`CooperativeStageLaneMap` is the first-class, validated thread->element map for that pattern: T threads
cooperatively load `total` contiguous elements, each thread owning a contiguous `width`-element chunk, so the
per-thread `width` axis is a unit-stride LOOP axis the coalesced-load primitive promotes to a vectorized load
(`global_load_dwordx4`). This is the bridge-independent, reusable analogue of the GEMV's `Q4KGateUpLaneMap`
(extra/qk_gemv_g2_lanemap.py) for cooperative contiguous staging. De-risked (2026-06-26): vectorizes the masked
and unmasked cache-row staging. Pure authoring primitive -- it builds index/store UOps; no codegen wiring.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from tinygrad.uop.ops import UOp, AxisType


@dataclass(frozen=True)
class CooperativeStageLaneMap:
  """thread tid owns the contiguous chunk [chunk*width .. chunk*width+width-1], chunk = stage*threads + tid."""
  total: int            # number of contiguous elements to stage (e.g. TK*Hd)
  threads: int          # workgroup threads cooperating (e.g. 128)
  width: int = 4        # per-thread contiguous chunk = vector width the coalescer folds (float4 / half8)
  base_axis: int = 60   # axis-id base for the two staging ranges (keep clear of the kernel's other ids)

  def validate(self) -> None:
    if self.total % (self.threads * self.width) != 0:
      raise ValueError(f"total={self.total} must divide threads*width={self.threads*self.width} "
                       f"(ragged tail not supported in v1; pick width|({self.total}//{self.threads}))")
    if self.width not in (1, 2, 4, 8):
      raise ValueError(f"width={self.width} must be a hardware fold width (1/2/4/8)")

  @property
  def stages(self) -> int:
    self.validate()
    return self.total // (self.threads * self.width)

  def axes(self) -> tuple[UOp, UOp]:
    """(stage, w) -- `w` is the per-thread contiguous LOOP axis the coalesced-load primitive promotes."""
    st = UOp.range(self.stages, self.base_axis, axis_type=AxisType.REDUCE)
    w = UOp.range(self.width, self.base_axis + 1, axis_type=AxisType.LOOP)
    return st, w

  def elem_index(self, st: UOp, tid: UOp, w: UOp) -> UOp:
    """The element index thread `tid` touches at (stage st, lane w): contiguous in w by construction."""
    return (st * self.threads + tid) * self.width + w

  def stage(self, dst: UOp, tid: UOp, value_fn: Callable[[UOp], UOp]) -> UOp:
    """Build the cooperative staging store `dst[i] = value_fn(i)` over (stage, w), END-closed.

    `value_fn(i)` returns the value to store for element index `i` (e.g. the cache row element, masked + cast).
    The returned UOp is the END(END(store, w), st) -- ready to feed a barrier/group. With
    `COALESCED_LOAD_LOWERING` on, the global load inside `value_fn(i)` folds to a vector load and the `dst`
    store folds to a vector store (both contiguous in `w`)."""
    st, w = self.axes()
    i = self.elem_index(st, tid, w)
    return dst[i].store(value_fn(i)).end(w).end(st)
