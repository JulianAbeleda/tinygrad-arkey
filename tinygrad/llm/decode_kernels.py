"""Production Q4_K G3 and Q6_K decode kernel lowerings.

These are statically promoted results: search and qualification live outside the
runtime, while this module owns the selected data descriptions and generic UOp
lowerings used for inference.  Keep this module independent of ``extra``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tinygrad import dtypes
from tinygrad.codegen.late.warp_reduce import WARP, _staged_shfl, _warp_reduce_sum_staged
from tinygrad.dtype import AddrSpace, DType
from tinygrad.helpers import cdiv
from tinygrad.llm.qk_layout import Q4_K_BLOCK_ELEMS, Q4K_WORDS_PER_BLOCK, Q6_K, Q6_K_BLOCK_ELEMS, Q6K_HALFWORDS_PER_BLOCK, QuantFormat
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp


# Q4_K G3 selected lane-map lowering.
@dataclass(frozen=True)
class Q4KGateUpLaneMap:
  k: int
  n: int
  qk_k: int = Q4_K_BLOCK_ELEMS
  lane_extent: int = WARP
  block_groups: int = 4
  words_per_group: int = 8
  q4k_words_per_block: int = Q4K_WORDS_PER_BLOCK

  @property
  def k_blocks(self) -> int: return self.k // self.qk_k

  @property
  def blocks_per_group(self) -> int: return self.k_blocks // self.block_groups

  def validate(self) -> None:
    if self.k % self.qk_k != 0: raise ValueError(f"k={self.k} must divide qk_k={self.qk_k}")
    if self.k_blocks % self.block_groups != 0:
      raise ValueError(f"k_blocks={self.k_blocks} must divide block_groups={self.block_groups}")
    if self.lane_extent != self.block_groups * self.words_per_group:
      raise ValueError(f"lane extent mismatch: {self.lane_extent} != {self.block_groups} * {self.words_per_group}")
    if self.words_per_group != 8: raise ValueError("Q4_K G3 lane map requires eight packed words per group")
    if self.q4k_words_per_block != 36: raise ValueError("Q4_K block layout must be 36 uint32 words")


@dataclass(frozen=True)
class LanePartition:
  lane: UOp
  lane_extent: int = WARP
  words_per_group: int = 8

  def validate(self) -> None:
    if self.lane_extent != WARP: raise ValueError(f"only wave{WARP} is supported, got {self.lane_extent}")
    if self.words_per_group <= 0 or self.lane_extent % self.words_per_group != 0:
      raise ValueError(f"words_per_group must divide lane_extent, got {self.words_per_group} / {self.lane_extent}")
    if self.lane.dtype.scalar() is not dtypes.weakint: raise ValueError(f"lane must be weakint index dtype, got {self.lane.dtype}")

  @property
  def word_col(self) -> UOp: return self.lane % self.words_per_group

  @property
  def block_group(self) -> UOp: return self.lane // self.words_per_group


def _lane_partition_reduce_sum(partial:UOp, part:LanePartition) -> UOp:
  part.validate()
  if partial.dtype.scalar() not in (dtypes.float32, dtypes.float):
    raise ValueError(f"lane partition sum supports float partials, got {partial.dtype}")
  if any(u.op is Ops.RANGE and u.arg[-1].name in ("UPCAST", "UNROLL") for u in partial.toposort()):
    raise ValueError("vectorized lane-partition partials are not supported")
  return _warp_reduce_sum_staged(partial, part.lane, part.lane_extent)


def _f16_word(word:UOp, high:bool) -> UOp:
  bits = (word.rshift(16) if high else word).bitwise_and(0xffff)
  return bits.cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)


def _q4k_group_params(words:UOp, base:UOp, grp:int) -> tuple[UOp, UOp, UOp, UOp]:
  d, dmin = _f16_word(words[base], False), _f16_word(words[base], True)
  def scale_byte(idx:int) -> UOp: return words[base + 1 + idx//4].rshift((idx%4)*8).bitwise_and(0xff)
  if grp < 4:
    sc, mn = scale_byte(grp).bitwise_and(63), scale_byte(4+grp).bitwise_and(63)
  else:
    high = scale_byte(8+grp-4)
    sc = high.bitwise_and(0xf).bitwise_or(scale_byte(grp-4).rshift(6).lshift(4))
    mn = high.rshift(4).bitwise_or(scale_byte(4+grp-4).rshift(6).lshift(4))
  return d, dmin, sc, mn


def _q4k_group_dot_packed_load(words:UOp, x:UOp, base:UOp, x_block:UOp, grp:int, lane4:UOp) -> UOp:
  d, dmin, sc, mn = _q4k_group_params(words, base, grp)
  qpack = words[base + 4 + (grp//2)*8 + lane4].rshift((grp%2)*4).bitwise_and(0x0F0F0F0F)
  contrib = UOp.const(dtypes.float32, 0.0)
  for nib in range(4):
    pos = lane4 * 4 + nib
    q = qpack.rshift(nib*8).bitwise_and(0xf)
    weight = d * sc.cast(dtypes.float32) * q.cast(dtypes.float32) - dmin * mn.cast(dtypes.float32)
    contrib = contrib + weight * x[x_block*Q4_K_BLOCK_ELEMS + grp*32 + pos].cast(dtypes.float32)
  return contrib


def _q4k_block_dot_packed_load(words:UOp, x:UOp, base:UOp, x_block:UOp, lane4:UOp) -> UOp:
  contrib = UOp.const(dtypes.float32, 0.0)
  for grp in range(8): contrib = contrib + _q4k_group_dot_packed_load(words, x, base, x_block, grp, lane4)
  return contrib


def q4k_g3_lanemap_gemv_kernel(rows:int, k:int, lanes:int=WARP):
  """Lower the search-selected G3 lane map without changing its UOps or name."""
  lm = Q4KGateUpLaneMap(k=k, n=rows, lane_extent=lanes)
  lm.validate()
  def kernel(out:UOp, words:UOp, x:UOp) -> UOp:
    row, lane = UOp.special(rows, "gidx0"), UOp.special(lanes, "lidx0")
    part = LanePartition(lane, lane_extent=lm.lane_extent, words_per_group=lm.words_per_group)
    lblk = UOp.range(lm.blocks_per_group, 0, axis_type=AxisType.REDUCE)
    blk = part.block_group * lm.blocks_per_group + lblk
    base = (row * lm.k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = _q4k_block_dot_packed_load(words, x, base, blk, part.word_col)
    acc = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(lblk)[0] + contrib).end(lblk))
    total = _lane_partition_reduce_sum(acc[0], part)
    return out[row].store(total).sink(arg=KernelInfo(name=f"q4k_g3_lanemap_gemv_{rows}_{k}", opts_to_apply=()))
  return kernel


# Q6_K selected spec-driven lowering and shared decode quant grammar.
Q6K_POS_EXTENT = 16
Q6K_VOCAB_SCALAR_REDUCE_MIN_ROWS = 131072

# Per-target coop row_tile values. This is route-config data, not an emitter branch: the emitter
# below is one shared lowering, and a target without a measured row keeps the safe default (the
# AMD gfx1100 machine-search value). NV:sm_120 measured 2026-08-02 on the RTX 5090 (P1 sweep,
# d512 fixed-depth): vocab coop 397.4 -> 330.1us, down coop 49.7 -> 35.5us, decode tok/s
# 163.2 -> 172.4, token sha unchanged.
Q6K_COOP_ROW_TILE_BY_TARGET: dict[tuple[str, str], int] = {("NV", "sm_120"): 2}
Q6K_COOP_ROW_TILE_DEFAULT = 4


def q6k_coop_row_tile_for_target(backend: str | None, architecture: str | None) -> int:
  """Resolve the coop row_tile for a resolved (backend, architecture) fact pair."""
  return Q6K_COOP_ROW_TILE_BY_TARGET.get((backend, architecture), Q6K_COOP_ROW_TILE_DEFAULT)


def _f16_half(half:UOp) -> UOp: return half.cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)


def _q6k_byte(halfs:UOp, base:UOp, byte_idx:UOp|int) -> UOp:
  idx = UOp.const(dtypes.int32, byte_idx) if isinstance(byte_idx, int) else byte_idx
  return halfs[base + idx//2].rshift((idx%2)*8).bitwise_and(0xff)


def _i8(byte:UOp) -> UOp: return byte.cast(dtypes.uint8).bitcast(dtypes.int8).cast(dtypes.float32)


def _q6k_weight(halfs:UOp, base:UOp, grp:int, pos:UOp) -> UOp:
  half, pgrp = grp // 8, grp % 8
  ql_byte_idx, ql_shift = half*64 + (pgrp%4)*16 + pos, 4 if pgrp >= 4 else 0
  qh_byte_idx, qh_shift = 128 + half*32 + (pgrp%2)*16 + pos, (pgrp//2) * 2
  ql = _q6k_byte(halfs, base, ql_byte_idx).rshift(ql_shift).bitwise_and(0xf)
  qh = _q6k_byte(halfs, base, qh_byte_idx).rshift(qh_shift).bitwise_and(0x3).lshift(4)
  q = ql.bitwise_or(qh).cast(dtypes.float32) - UOp.const(dtypes.float32, 32.0)
  return _f16_half(halfs[base + 104]) * q * _i8(_q6k_byte(halfs, base, 192 + grp))


def _q6k_block_dot(halfs:UOp, x:UOp, base:UOp, x_block:UOp, pos:UOp) -> UOp:
  contrib = UOp.const(dtypes.float32, 0.0)
  for grp in range(16):
    contrib = contrib + _q6k_weight(halfs, base, grp, pos) * x[x_block*Q6_K_BLOCK_ELEMS + grp*16 + pos].cast(dtypes.float32)
  return contrib


def _q6k_coop_pos_reduce_sum(val:UOp, lane:UOp, row_tile:int, slot_base:int=90) -> UOp:
  """Cross-lane sum over the coop route's 16 pos lanes. The lane map is row_i-fastest
  (tid = pos*row_tile + row_i), so each ladder step must advance by row_tile lane ids to
  stay within one (row, pos) group; the standard offset ladder (width/2..1) would mix the
  row_i bit on the last step. Requires a single warp: row_tile * Q6K_POS_EXTENT <= 32."""
  off = Q6K_POS_EXTENT >> 1
  while off >= 1:
    val = val + _staged_shfl(val, off * row_tile, lane, slot_base)
    off >>= 1
    slot_base += 1
  return val


@dataclass(frozen=True)
class DecodeRMSNormSpec:
  """One fused decode RMSNorm kernel: per-row sumsq reduce + `x * rsqrt(sumsq/dim + eps) * w`
  epilogue in a single UOp builder (l1-decode-plumbing-fusion-design-20260802.md section 6,
  norm family). One warp per row; the epilogue reuses the legacy graph's exact ops (DIV by dim,
  SQRT, RECIPROCAL, then (x*scale)*w) so only the sumsq summation ORDER differs from the generic
  reduce, and that delta is gated by the fixed-depth token sha like every other fused variant."""
  rows: int
  dim: int
  eps: float
  lane_width: int = 32
  warps_per_row: int = 1
  x_dtype: DType = dtypes.float32
  weight_dtype: DType = dtypes.float32
  out_dtype: DType = dtypes.float32
  x_rank: int = 1
  target: str = "amd_gfx1100"

  @property
  def kernel_name(self) -> str: return f"decode_rmsnorm_{self.rows}_{self.dim}"

  def validate(self) -> None:
    if self.rows < 1: raise ValueError(f"DecodeRMSNormSpec requires rows>=1, got {self.rows}")
    if self.dim < self.lane_width or self.dim % self.lane_width != 0:
      raise ValueError(f"DecodeRMSNormSpec requires dim >= lane_width and dim % lane_width == 0, "
                       f"got dim={self.dim} lane_width={self.lane_width}")
    if self.warps_per_row < 1 or self.dim % (self.lane_width * self.warps_per_row) != 0:
      raise ValueError(f"DecodeRMSNormSpec requires dim % (lane_width * warps_per_row) == 0, got "
                       f"dim={self.dim} lane_width={self.lane_width} warps_per_row={self.warps_per_row}")
    if not isinstance(self.eps, float) or self.eps <= 0: raise ValueError(f"DecodeRMSNormSpec requires eps>0, got {self.eps!r}")
    if self.x_rank not in (1, 3): raise ValueError(f"DecodeRMSNormSpec requires x_rank in (1, 3), got {self.x_rank}")
    if self.out_dtype not in (dtypes.float16, dtypes.float32):
      raise ValueError(f"DecodeRMSNormSpec requires out_dtype float16/float32, got {self.out_dtype}")


def emit_decode_rmsnorm_kernel(spec:DecodeRMSNormSpec):
  spec.validate()
  rows, dim, lane, warps = spec.rows, spec.dim, spec.lane_width, spec.warps_per_row
  per_lane = dim // (lane * warps)
  dim_f = UOp.const(dtypes.float32, float(dim))
  eps = UOp.const(dtypes.float32, spec.eps)
  def kernel(out:UOp, x:UOp, w:UOp) -> UOp:
    row = UOp.range(rows, 0)
    laneid = UOp.range(lane, 1, axis_type=AxisType.LOCAL)
    warp = UOp.range(warps, 2, axis_type=AxisType.LOCAL)
    red = UOp.range(per_lane, 3, axis_type=AxisType.REDUCE)
    base = row * dim + warp * (per_lane * lane) + laneid + red * lane
    # The input is passed as a flat (numel,) view and the custom-kernel transport
    # contiguous()s it into the buffer this flat base indexes. That per-call
    # materialization is the measured reason the norm-fusion route is closed-default
    # non-landing (m3-fused-norm-measurement-record-20260802.md). Rank 3 inputs
    # (B,T,dim activations) index the two leading extent-1 dims; rank 1 inputs
    # (q/k slices) index the flat base directly.
    x_sel = x[UOp.const(dtypes.int, 0), UOp.const(dtypes.int, 0), base] if spec.x_rank == 3 else x[base]
    acc = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    xv = x_sel.cast(dtypes.float32)
    acc = acc.after(acc[0].store(acc.after(red)[0] + xv * xv).end(red))
    warp_total = _warp_reduce_sum_staged(acc[0], laneid, lane, slot_base=90)
    if warps > 1:
      smem = UOp.placeholder((warps,), dtypes.float32, 230, addrspace=AddrSpace.LOCAL)
      wstore = smem[warp].store(warp_total, laneid.eq(0))
      barrier = UOp.barrier(UOp.group(wstore))
      total = UOp.const(dtypes.float32, 0.0)
      for wi in range(warps):
        total = total + smem.after(barrier)[wi]
    else:
      total = warp_total
    scale = UOp(Ops.RECIPROCAL, dtypes.float32, (UOp(Ops.SQRT, dtypes.float32, (total / dim_f + eps,)),))
    epi = UOp.range(per_lane, 3)
    obase = row * dim + warp * (per_lane * lane) + laneid + epi * lane
    wv = w[warp * (per_lane * lane) + laneid + epi * lane].cast(dtypes.float32)
    x_epi = x[UOp.const(dtypes.int, 0), UOp.const(dtypes.int, 0), obase] if spec.x_rank == 3 else x[obase]
    return out[obase].store(((x_epi.cast(dtypes.float32) * scale) * wv).cast(spec.out_dtype)).end(row, laneid, warp, epi).sink(
      arg=KernelInfo(name=spec.kernel_name, opts_to_apply=()))
  return kernel


@dataclass(frozen=True)
class Q6KGEMVRouteSpec:
  rows: int
  k: int
  role: str = ""
  route_family: str = "q6k_coop"
  target: str = "amd_gfx1100"
  row_tile: int = 4
  lane_extent: int = Q6K_POS_EXTENT
  parts: int = 1
  pos_axis: str = "local"
  block_axis: str = "reduce"
  reduction: str = "external_sum"
  storage: str = "packed_u16"
  quant: QuantFormat = Q6_K
  opts: tuple = field(default_factory=tuple)

  @property
  def k_blocks(self) -> int: return self.k // Q6_K_BLOCK_ELEMS

  @property
  def partial_axis_extent(self) -> int: return self.lane_extent if self.route_family == "q6k_coop" else self.parts

  @property
  def kernel_name(self) -> str:
    suffix = "_inkernel" if self.reduction == "in_kernel" else ""
    return (f"q6k_gen_coop_{self.rows}_{self.k}" if self.route_family == "q6k_coop"
            else f"q6k_gen_partial_{self.rows}_{self.k}_{self.parts}") + suffix

  def validate(self) -> None:
    if self.quant is not Q6_K: raise ValueError(f"Q6KGEMVRouteSpec quant must be Q6_K, got {self.quant!r}")
    if self.route_family not in ("q6k_coop", "q6k_partial"): raise ValueError(f"unknown route_family {self.route_family!r}")
    if self.reduction not in ("external_sum", "in_kernel"): raise ValueError(f"unsupported reduction {self.reduction!r}")
    if self.reduction == "in_kernel" and self.route_family == "q6k_partial":
      raise ValueError("in_kernel reduction is not implemented for the q6k_partial family (M2 non-landing, "
                       "l1-decode-plumbing-fusion-design-20260802.md section 6 class 9); use external_sum")
    if self.storage != "packed_u16": raise ValueError(f"unsupported storage {self.storage!r}")
    if self.k % Q6_K_BLOCK_ELEMS != 0: raise ValueError(f"k={self.k} must be a multiple of {Q6_K_BLOCK_ELEMS}")
    if self.lane_extent != Q6K_POS_EXTENT: raise ValueError(f"lane_extent must be {Q6K_POS_EXTENT}, got {self.lane_extent}")
    if self.route_family == "q6k_coop":
      if self.pos_axis != "local": raise ValueError("coop route requires pos_axis=local")
      if self.row_tile < 1 or self.rows % self.row_tile != 0:
        raise ValueError(f"coop route requires rows({self.rows}) % row_tile({self.row_tile}) == 0")
      if self.reduction == "in_kernel" and self.row_tile * self.lane_extent > 32:
        raise ValueError("coop in_kernel reduce requires a single warp: row_tile * lane_extent <= 32, got "
                         f"row_tile={self.row_tile} lane_extent={self.lane_extent}")
    else:
      if self.pos_axis != "reduce": raise ValueError("partial route requires pos_axis=reduce")
      if self.parts < 1: raise ValueError(f"partial route requires parts>=1, got {self.parts}")

  def to_json(self) -> dict[str, Any]:
    return {"quant": self.quant.name, "rows": self.rows, "k": self.k, "role": self.role, "route_family": self.route_family,
            "target": self.target, "row_tile": self.row_tile, "lane_extent": self.lane_extent, "parts": self.parts,
            "pos_axis": self.pos_axis, "block_axis": self.block_axis, "reduction": self.reduction, "storage": self.storage}


def q6k_spec_for_role(rows:int, k:int, *, role:str="", parts:int=1, row_tile:int=4, use_coop:bool=True,
                      target:str="amd_gfx1100", opts:tuple=(), reduction:str="external_sum") -> Q6KGEMVRouteSpec:
  if use_coop and parts == 1:
    return Q6KGEMVRouteSpec(rows=rows, k=k, role=role, route_family="q6k_coop", row_tile=row_tile,
                            pos_axis="local", target=target, reduction=reduction)
  return Q6KGEMVRouteSpec(rows=rows, k=k, role=role, route_family="q6k_partial", parts=parts,
                          pos_axis="reduce", target=target, opts=opts, reduction=reduction)


def emit_q6k_gemv_kernel(spec:Q6KGEMVRouteSpec):
  spec.validate()
  return _emit_q6k_coop(spec) if spec.route_family == "q6k_coop" else _emit_q6k_partial(spec)


def q6k_vocab_scalar_reduce_eligible(spec:Q6KGEMVRouteSpec) -> bool:
  return spec.route_family == "q6k_coop" and spec.rows >= Q6K_VOCAB_SCALAR_REDUCE_MIN_ROWS and spec.partial_axis_extent == Q6K_POS_EXTENT


def emit_q6k_vocab_scalar_reduce_kernel(spec:Q6KGEMVRouteSpec):
  if not q6k_vocab_scalar_reduce_eligible(spec): raise ValueError(f"Q6_K scalar vocab reduction is not admitted for {spec.to_json()}")
  def kernel(out:UOp, partials:UOp) -> UOp:
    row = UOp.range(spec.rows, 0)
    pos = UOp.range(spec.partial_axis_extent, 1, axis_type=AxisType.REDUCE)
    return out[row].store(partials[row, pos].reduce(pos, arg=Ops.ADD)).end(row).sink(
      arg=KernelInfo(name=f"q6k_vocab_scalar_reduce_{spec.rows}_{spec.k}", opts_to_apply=()))
  return kernel


def _emit_q6k_coop(spec:Q6KGEMVRouteSpec):
  rows, row_tile, k_blocks, name = spec.rows, spec.row_tile, spec.k_blocks, spec.kernel_name
  in_kernel = spec.reduction == "in_kernel"
  def kernel(partials:UOp, halfs:UOp, x:UOp) -> UOp:
    row_o = UOp.range(cdiv(rows, row_tile), 0)
    row_i = UOp.range(row_tile, 1, axis_type=AxisType.LOCAL)
    pos = UOp.range(Q6K_POS_EXTENT, 2, axis_type=AxisType.LOCAL)
    blk = UOp.range(k_blocks, 3, axis_type=AxisType.REDUCE)
    row, base = row_o * row_tile + row_i, ((row_o * row_tile + row_i) * k_blocks + blk) * Q6K_HALFWORDS_PER_BLOCK
    contrib = _q6k_block_dot(halfs, x, base, blk, pos)
    if in_kernel:
      acc = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
      acc = acc.after(acc[0].store(0.0))
      acc = acc.after(acc[0].store(acc.after(blk)[0] + contrib).end(blk))
      total = _q6k_coop_pos_reduce_sum(acc[0], pos, row_tile)
      return partials[row].store(total).end(row_o, row_i, pos).sink(arg=KernelInfo(name=name, opts_to_apply=()))
    acc = partials[row, pos].set(0.0)
    acc = partials[row, pos].set(acc.after(blk)[row, pos] + contrib, end=blk)
    return acc.end(row_o, row_i, pos).sink(arg=KernelInfo(name=name, opts_to_apply=()))
  return kernel


def _emit_q6k_partial(spec:Q6KGEMVRouteSpec):
  rows, parts, k_blocks, name, opts = spec.rows, spec.parts, spec.k_blocks, spec.kernel_name, spec.opts
  blocks_per_part = cdiv(k_blocks, parts)
  def kernel(partials:UOp, halfs:UOp, x:UOp) -> UOp:
    row, part = UOp.range(rows, 0), UOp.range(parts, 1)
    blk_part = UOp.range(blocks_per_part, 2, axis_type=AxisType.REDUCE)
    pos = UOp.range(Q6K_POS_EXTENT, 3, axis_type=AxisType.REDUCE)
    blk = part * blocks_per_part + blk_part
    base = (row * k_blocks + blk) * Q6K_HALFWORDS_PER_BLOCK
    contrib = (blk < k_blocks).where(_q6k_block_dot(halfs, x, base, blk, pos), UOp.const(dtypes.float32, 0.0))
    acc = partials[row, part].set(0.0)
    acc = partials[row, part].set(acc.after(blk_part, pos)[row, part] + contrib, end=pos)
    return acc.end(row, part, blk_part).sink(arg=KernelInfo(name=name, opts_to_apply=opts))
  return kernel
