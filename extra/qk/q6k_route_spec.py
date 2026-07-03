#!/usr/bin/env python3
"""TG-P3: Q6_K decode-GEMV route SPEC + spec-driven lowering.

The shipped Q6_K decode route (extra/qk/quant/q6_k_gemv_primitive.py q6k_coop_partial_kernel / q6k_gemv_partial_kernel) is
a pair of hand-authored UOp kernel templates. This module makes the route MACHINE-AUTHORED: the kernel is emitted
from a data `Q6KGEMVRouteSpec` by a generic lowering (emit_q6k_gemv_kernel), exactly as the Q4_K G3 route is
lowered from a Q4KGateUpLaneMap. The dequant grammar (_q6k_block_dot / _q6k_weight, the Q6_K block format) is
shared and unchanged; only the *selection + assembly* of the kernel moves from a hand-written function into a
spec-driven lowering. The generated kernels are numerically byte-identical to the shipped ones (proven in
extra/qk/q6k_generated_coop_gate.py), so this is a provenance conversion, not a math change.

Route families (both finalize with an external stage-2 sum over the partial axis, exactly like the shipped route):
  * coop     -> pos(0..15) is a LOCAL lane axis (coalesced packed-weight loads); row_tile rows share a workgroup;
                blocks are the REDUCE axis; output partials[rows, 16]. Mirrors q6k_coop_partial_kernel.
  * partial  -> `parts` K-slices as an outer range; (blk_part, pos) are REDUCE axes; output partials[rows, parts].
                Mirrors q6k_gemv_partial_kernel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tinygrad import dtypes
from tinygrad.helpers import cdiv
from tinygrad.uop.ops import AxisType, KernelInfo, UOp

from extra.qk.quant.q6_k_gemv_primitive import _q6k_block_dot, parse_opt
from extra.qk.layout import Q6_K_BLOCK_ELEMS, Q6K_HALFWORDS_PER_BLOCK

# the Q6_K dequant grammar reads 16 within-block byte positions per block; the coop lane axis and the partial
# reduce axis both have this extent. Kept as a named constant so the spec is data, not a magic literal.
Q6K_POS_EXTENT = 16

_ALLOWED_ROUTE_FAMILIES = ("q6k_coop", "q6k_partial")
_ALLOWED_REDUCTION = ("external_sum",)
_ALLOWED_STORAGE = ("packed_u16",)


@dataclass(frozen=True)
class Q6KGEMVRouteSpec:
  """Data description of a Q6_K decode-GEMV route. `route_family` picks the lowering; every other field is a
  structural parameter of that family. This is DATA (serializable via to_json); the lowering is emit_q6k_gemv_kernel."""
  rows: int
  k: int
  role: str = ""
  route_family: str = "q6k_coop"
  target: str = "amd_gfx1100"
  row_tile: int = 4                 # coop: rows per workgroup (the LOCAL row axis extent)
  lane_extent: int = Q6K_POS_EXTENT # within-block pos lane extent (coop LOCAL / partial REDUCE)
  parts: int = 1                    # partial: number of K-slices (the stage-2 sum axis)
  pos_axis: str = "local"           # coop -> local ; partial -> reduce
  block_axis: str = "reduce"
  reduction: str = "external_sum"
  storage: str = "packed_u16"
  quant: str = "Q6_K"
  opts: tuple = field(default_factory=tuple)

  @property
  def k_blocks(self) -> int:
    return self.k // Q6_K_BLOCK_ELEMS

  @property
  def partial_axis_extent(self) -> int:
    """The stage-2 external-sum axis width (coop: pos lanes; partial: K-parts)."""
    return self.lane_extent if self.route_family == "q6k_coop" else self.parts

  def validate(self) -> None:
    if self.quant != "Q6_K": raise ValueError(f"Q6KGEMVRouteSpec quant must be Q6_K, got {self.quant!r}")
    if self.route_family not in _ALLOWED_ROUTE_FAMILIES:
      raise ValueError(f"unknown route_family {self.route_family!r}; allowed {_ALLOWED_ROUTE_FAMILIES}")
    if self.reduction not in _ALLOWED_REDUCTION:
      raise ValueError(f"unsupported reduction {self.reduction!r}; allowed {_ALLOWED_REDUCTION}")
    if self.storage not in _ALLOWED_STORAGE:
      raise ValueError(f"unsupported storage {self.storage!r}; allowed {_ALLOWED_STORAGE}")
    if self.k % Q6_K_BLOCK_ELEMS != 0: raise ValueError(f"k={self.k} must be a multiple of {Q6_K_BLOCK_ELEMS}")
    if self.lane_extent != Q6K_POS_EXTENT:
      raise ValueError(f"lane_extent must be {Q6K_POS_EXTENT} (Q6_K within-block pos), got {self.lane_extent}")
    if self.route_family == "q6k_coop":
      if self.pos_axis != "local": raise ValueError("coop route requires pos_axis=local")
      if self.row_tile < 1 or self.rows % self.row_tile != 0:
        raise ValueError(f"coop route requires rows({self.rows}) % row_tile({self.row_tile}) == 0")
    else:
      if self.pos_axis != "reduce": raise ValueError("partial route requires pos_axis=reduce")
      if self.parts < 1: raise ValueError(f"partial route requires parts>=1, got {self.parts}")

  def to_json(self) -> dict[str, Any]:
    return {"quant": self.quant, "rows": self.rows, "k": self.k, "role": self.role,
            "route_family": self.route_family, "target": self.target, "row_tile": self.row_tile,
            "lane_extent": self.lane_extent, "parts": self.parts, "pos_axis": self.pos_axis,
            "block_axis": self.block_axis, "reduction": self.reduction, "storage": self.storage}

  @property
  def kernel_name(self) -> str:
    if self.route_family == "q6k_coop": return f"q6k_gen_coop_{self.rows}_{self.k}"
    return f"q6k_gen_partial_{self.rows}_{self.k}_{self.parts}"


def spec_for_role(rows:int, k:int, *, role:str="", parts:int=1, row_tile:int=4, use_coop:bool=True,
                  target:str="amd_gfx1100", opts:tuple=()) -> Q6KGEMVRouteSpec:
  """Build the spec that reproduces the CURRENT default for a (rows, k, role, parts) Q6_K tensor: the coop family
  when the shipped `use_coop` gate would fire (parts==1), else the partial family with the tensor's parts."""
  if use_coop and parts == 1:
    return Q6KGEMVRouteSpec(rows=rows, k=k, role=role, route_family="q6k_coop", row_tile=row_tile,
                            pos_axis="local", target=target)
  return Q6KGEMVRouteSpec(rows=rows, k=k, role=role, route_family="q6k_partial", parts=parts,
                          pos_axis="reduce", target=target, opts=opts)


def emit_q6k_gemv_kernel(spec:Q6KGEMVRouteSpec):
  """Lower a Q6KGEMVRouteSpec to a named UOp kernel fn (partials, halfs, x) -> sink. Byte-identical body to the
  shipped route; distinct program name so capture/gates can tell the generated route from the hand template."""
  spec.validate()
  if spec.route_family == "q6k_coop": return _emit_coop(spec)
  return _emit_partial(spec)


def _emit_coop(spec:Q6KGEMVRouteSpec):
  rows, k, row_tile, k_blocks, name = spec.rows, spec.k, spec.row_tile, spec.k_blocks, spec.kernel_name
  def kernel(partials:UOp, halfs:UOp, x:UOp) -> UOp:
    row_o = UOp.range(cdiv(rows, row_tile), 0)
    row_i = UOp.range(row_tile, 1, axis_type=AxisType.LOCAL)
    pos = UOp.range(Q6K_POS_EXTENT, 2, axis_type=AxisType.LOCAL)
    blk = UOp.range(k_blocks, 3, axis_type=AxisType.REDUCE)
    row = row_o * row_tile + row_i
    base = (row * k_blocks + blk) * Q6K_HALFWORDS_PER_BLOCK
    contrib = _q6k_block_dot(halfs, x, base, blk, pos)
    acc = partials[row, pos].set(0.0)
    acc = partials[row, pos].set(acc.after(blk)[row, pos] + contrib, end=blk)
    return acc.end(row_o, row_i, pos).sink(arg=KernelInfo(name=name, opts_to_apply=()))
  return kernel


def _emit_partial(spec:Q6KGEMVRouteSpec):
  rows, k, parts, k_blocks, name, opts = spec.rows, spec.k, spec.parts, spec.k_blocks, spec.kernel_name, spec.opts
  blocks_per_part = cdiv(k_blocks, parts)
  def kernel(partials:UOp, halfs:UOp, x:UOp) -> UOp:
    row = UOp.range(rows, 0)
    part = UOp.range(parts, 1)
    blk_part = UOp.range(blocks_per_part, 2, axis_type=AxisType.REDUCE)
    pos = UOp.range(Q6K_POS_EXTENT, 3, axis_type=AxisType.REDUCE)
    blk = part * blocks_per_part + blk_part
    in_range = blk < k_blocks
    base = (row * k_blocks + blk) * Q6K_HALFWORDS_PER_BLOCK
    contrib = in_range.where(_q6k_block_dot(halfs, x, base, blk, pos), UOp.const(dtypes.float32, 0.0))
    acc = partials[row, part].set(0.0)
    acc = partials[row, part].set(acc.after(blk_part, pos)[row, part] + contrib, end=pos)
    return acc.end(row, part, blk_part).sink(arg=KernelInfo(name=name, opts_to_apply=opts))
  return kernel
