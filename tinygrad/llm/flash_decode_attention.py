"""Production live-split flash-decode runtime.

This module owns the selected G4/G5 descriptors, their generated UOp builders,
and the Tensor executor. Search campaigns and qualification harnesses live under
``extra/llm_research``; production inference does not depend on them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from tinygrad import Tensor, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import getenv
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp
from tinygrad.llm.kernel_program import (DeclaredTypedOutput, KernelProgram, KernelProgramProvenance, OutputSpec,
                                         TypedLayout, execute_promoted_program)

_LOG2E = 1.4426950408889634
_F32 = dtypes.float32


def _fexp(x:UOp) -> UOp:
  arg = x * _LOG2E
  if getenv("DECODE_FAST_EXP2", 0):
    from tinygrad.codegen.late.flash_decode_intrinsics import exp2f
    return exp2f(arg)
  return arg.exp2()


def _fc(v:float) -> UOp: return UOp.const(_F32, v)
def _ceildiv(a, b:int): return (a + b - 1) // b
def _ceildiv_uop(a, b:int): return (a + (b - 1)) // b
def _kernel_info(name:str, *, coalesced_loads:bool=False) -> KernelInfo:
  return KernelInfo(name=name, opts_to_apply=(), coalesced_loads=coalesced_loads)


def _promoted_route_stage_width(Hq:int, split_count:int, query_group_size:int|None) -> int|None:
  """The frozen stage_width of the promoted route matching (Hq, split_count, query_group_size), else None.

  G4 (32, 48, None) stages with width 1 and G5 (40, 32, 2) with width 4. Those two geometries are the
  historical "default geometry": their kernel names are byte-exact artifacts (the name is the rendered C
  function name), so the naming rule must not append a suffix to them even though G5's stage_width differs
  from the descriptor default."""
  if (Hq, split_count, query_group_size) == (32, 48, None): return 1
  if (Hq, split_count, query_group_size) == (40, 32, 2): return 4
  return None


def _promoted_route_split_count(Hq:int, query_group_size:int|None) -> int|None:
  """The frozen split_count of the promoted route matching (Hq, query_group_size), else None.

  G4 (32, None) uses S=48 and G5 (40, 2) uses S=32. split_count is shape-derived but participates in the
  emitted program, so a non-default split count must not collide with the historical kernel name."""
  if (Hq, query_group_size) == (32, None): return 48
  if (Hq, query_group_size) == (40, 2): return 32
  return None


def flash_decode_coarse_split_override() -> int:
  """Env-gated research override for the promoted G4 route's KV split count (0 = unset).

  When ``FLASH_DECODE_COARSE_SPLIT`` is set to a positive int, decode_routes runs the production
  G4 decode route with that split count instead of the promoted S=48. Unset env is byte-identical
  to today: the promoted route, its admission guard, and its kernel names are untouched. G5
  (40 heads, S=32) is not affected by this override.
  """
  return getenv("FLASH_DECODE_COARSE_SPLIT", 0)


def _tile_geometry_suffix(*, Hq:int, split_count:int, lane_width:int, token_block:int, stage_width:int|None,
                          reduce_structure:str|None, dot_pair_width:int, score_group_width:int|None, warps:int|None,
                          query_group_size:int|None, QG:int) -> str:
  """Deterministic JIT-cache suffix for non-default tile geometry; empty for the promoted-route geometry.

  Only fields that differ from their production default are included, so G4/G5 keep their exact historical
  kernel names while differently emitted programs get distinct, deterministic names (the canonical JSON is
  the authoritative candidate identity; this suffix is the short-name side of the same contract)."""
  route_split = _promoted_route_split_count(Hq, query_group_size)
  if route_split is not None and split_count == route_split \
      and lane_width == 32 and token_block == 16 and score_group_width is None and warps is None \
      and reduce_structure in (None, "staged") and dot_pair_width == 2 and \
      (stage_width == _promoted_route_stage_width(Hq, split_count, query_group_size) or
       (stage_width is None and _promoted_route_stage_width(Hq, split_count, query_group_size) == 1)):
    return ""
  geom = ""
  if split_count != route_split: geom += f"_s{split_count}"
  if lane_width != 32: geom += f"_lw{lane_width}"
  if token_block != 16: geom += f"_tk{token_block}"
  route_sw = _promoted_route_stage_width(Hq, split_count, query_group_size)
  if stage_width != (route_sw if route_sw is not None else 1): geom += f"_sw{stage_width}"
  if reduce_structure not in (None, "staged"): geom += f"_r{reduce_structure[0]}"
  if dot_pair_width != 2: geom += f"_dpw{dot_pair_width}"
  if score_group_width is not None: geom += f"_sgw{score_group_width}"
  if warps is not None and warps != QG: geom += f"_w{warps}"
  return geom


@dataclass(frozen=True)
class CooperativeStageLaneMap:
  """Map each staging thread to a contiguous, vectorizable element chunk."""
  total: int
  threads: int
  width: int = 4
  base_axis: int = 60

  def validate(self) -> None:
    if self.total % (self.threads * self.width) != 0:
      raise ValueError(f"total={self.total} must divide threads*width={self.threads*self.width}")
    if self.width not in (1, 2, 4, 8):
      raise ValueError(f"width={self.width} must be one of 1,2,4,8")

  @property
  def stages(self) -> int:
    self.validate()
    return self.total // (self.threads * self.width)

  def axes(self) -> tuple[UOp, UOp]:
    return (UOp.range(self.stages, self.base_axis, axis_type=AxisType.REDUCE),
            UOp.range(self.width, self.base_axis + 1, axis_type=AxisType.LOOP))

  def elem_index(self, stage:UOp, tid:UOp, width:UOp) -> UOp:
    return (stage * self.threads + tid) * self.width + width

  def stage(self, dst:UOp, tid:UOp, value_fn:Callable[[UOp], UOp]) -> UOp:
    stage, width = self.axes()
    idx = self.elem_index(stage, tid, width)
    return dst[idx].store(value_fn(idx)).end(width).end(stage)


def make_kv_element_loader(cache:UOp, Hd:int, kvscale:UOp|None=None, freqs:UOp|None=None, pos_of=None):
  """Return a K/V loader with optional in-register dequant and K rope-at-read."""
  quant, rope, half_dim = kvscale is not None, freqs is not None, Hd // 2
  if pos_of is None: pos_of = lambda tok: tok

  def raw(which, kvh, tok, elem):
    val = cache[which, 0, kvh, tok, elem].cast(dtypes.float16)
    if quant: val = val * kvscale[which, 0, kvh, tok].cast(dtypes.float16)
    return val

  def load(which, kvh, tok, elem):
    val = raw(which, kvh, tok, elem)
    if rope and which == 0:
      pos, rotary_elem, low = pos_of(tok), elem % half_dim, elem < half_dim
      pair = raw(0, kvh, tok, low.where(elem + half_dim, elem - half_dim))
      cos = freqs[pos, rotary_elem].cast(dtypes.float16)
      sin = freqs[pos, half_dim + rotary_elem].cast(dtypes.float16)
      val = val * cos + low.where(-(pair * sin), pair * sin)
    return val
  return load


def flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, L:int, S, Tc,
                                                              staging:str="KV_BOTH", quant:bool=False,
                                                              rope:bool=False, query_group_size:int|None=None,
                                                              stage_width:int|None=None, token_block:int=16,
                                                              lane_width:int=32, score_group_width:int|None=None,
                                                              warps:int|None=None, reduce_structure:str|None=None,
                                                              dot_pair_width:int=2):
  """Emit the selected live-split tile: LDS K/V, online softmax, and sharded PV."""
  if Hd % (lane_width * dot_pair_width) != 0:
    raise ValueError(f"block tile requires Hd%(lane_width*dot_pair_width)==0, "
                     f"got Hd={Hd} lane_width={lane_width} dot_pair_width={dot_pair_width}")
  if staging not in {"KV_BOTH", "K_ONLY"}: raise ValueError(f"unsupported staging={staging!r}")
  if token_block < 1: raise ValueError(f"token_block must be >= 1, got {token_block}")
  if lane_width < 1 or lane_width & (lane_width - 1):
    raise ValueError(f"lane_width must be a positive power of two, got {lane_width}")
  if dot_pair_width < 1: raise ValueError(f"dot_pair_width must be >= 1, got {dot_pair_width}")
  if reduce_structure not in (None, "staged", "inline"):
    raise ValueError(f"reduce_structure must be one of 'staged','inline', got {reduce_structure!r}")
  G = Hq // Hkv
  QG = G if query_group_size is None else query_group_size
  if QG < 1 or QG > G: raise ValueError(f"query_group_size must be in 1..{G}, got {QG}")
  if warps is not None and warps < QG: raise ValueError(f"warps must be >= QG={QG}, got {warps}")
  # Column-parallel score groups are not implemented: dot ownership is elem =
  # pair_axis*(LANES*dot_pair_width) + lane*dot_pair_width, so every lane must
  # contribute to cover Hd. Narrowing the reduce width below lane_width would
  # sum only a fraction of the dot and produce a wrong score.
  if score_group_width is not None and score_group_width != lane_width:
    raise ValueError(f"score_group_width must equal lane_width={lane_width} or be None, got {score_group_width}")
  group_width = score_group_width or lane_width
  NG, W, LANES, WARPS, TK = _ceildiv(G, QG), Hd + 2, lane_width, QG if warps is None else warps, token_block
  THREADS, R, RP = LANES * WARPS, Hd // LANES, Hd // (LANES * dot_pair_width)
  STAGES, NB, scale = _ceildiv(TK * Hd, THREADS), _ceildiv(L, TK), 1.0 / (Hd ** 0.5)

  def kernel(pout:UOp, q:UOp, cache:UOp, *extra) -> UOp:
    from tinygrad.codegen.late.warp_reduce import _warp_reduce_sum_staged, warp_reduce_sum
    from tinygrad.codegen.late.flash_decode_intrinsics import fdot2 as _lower_fdot2
    optional = list(extra)
    kvscale = (optional.pop(0) if optional else None) if quant else None
    freqs = (optional.pop(0) if optional else None) if rope else None
    if quant and kvscale is None: raise ValueError("quant=True requires a scale buffer bound after cache")
    if rope and freqs is None: raise ValueError("rope=True requires a freqs (cos|sin) buffer bound after cache/scale")
    kv_load = make_kv_element_loader(cache, Hd, kvscale=kvscale, freqs=freqs)
    kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
    split = UOp.range(S, 1, AxisType.GLOBAL)
    query_group = UOp.range(NG, 9, AxisType.GLOBAL)
    lane = UOp.range(LANES, 10, AxisType.LOCAL)
    warp = UOp.range(WARPS, 11, AxisType.LOCAL)
    grouped_head = query_group * QG + warp
    warp_active = grouped_head < G
    raw_head = kvh * G + grouped_head
    head = warp_active.where(raw_head, raw_head.const_like(0))
    tid = warp * LANES + lane
    ksh = UOp.placeholder((TK * Hd,), dtypes.float16, 230, addrspace=AddrSpace.LOCAL)
    vsh = UOp.placeholder((TK * Hd,), dtypes.float16, 231, addrspace=AddrSpace.LOCAL) if staging == "KV_BOTH" else None
    acc = UOp.placeholder((R,), _F32, 232, addrspace=AddrSpace.REG)
    den = UOp.placeholder((1,), _F32, 233, addrspace=AddrSpace.REG)
    mx = UOp.placeholder((1,), _F32, 234, addrspace=AddrSpace.REG)
    zero_axis = UOp.range(R, 2)
    init = acc.after(kvh, split)[zero_axis].store(0.0).end(zero_axis)
    init = den.after(init)[0].store(0.0)
    init = mx.after(init)[0].store(-float("inf"))
    acc, den, mx = acc.after(init), den.after(init), mx.after(init)
    block = UOp.range(NB, 3, axis_type=AxisType.REDUCE)
    selected_width = getenv("DECODE_STAGE_COALESCE") if stage_width is None else stage_width
    try:
      if selected_width:
        lane_map = CooperativeStageLaneMap(total=TK * Hd, threads=THREADS, width=selected_width, base_axis=60)
        lane_map.validate()
        stage, width = lane_map.axes()
        idx = lane_map.elem_index(stage, tid, width)
    except ValueError:
      selected_width = 0
    if not selected_width:
      stage = UOp.range(STAGES, 4, axis_type=AxisType.REDUCE)
      idx = stage * THREADS + tid
    token_stage, elem_stage = idx // Hd, idx % Hd
    token = split * L + block * TK + token_stage
    in_stage = (token_stage < TK) & (token < Tc)
    token_safe = in_stage.where(token, token.const_like(0))
    gate = () if selected_width else (idx < (TK * Hd),)
    kstore = ksh[idx].store(kv_load(0, kvh, token_safe, elem_stage), *gate)
    if staging == "KV_BOTH":
      vstore = vsh.after(kstore)[idx].store(kv_load(1, kvh, token_safe, elem_stage), *gate)
      barrier = UOp.barrier(UOp.group(vstore.end(width).end(stage) if selected_width else vstore.end(stage)))
    else:
      barrier = UOp.barrier(UOp.group(kstore.end(width).end(stage) if selected_width else kstore.end(stage)))

    def dot_reduce(token_in_tile):
      dot = UOp.placeholder((1,), _F32, 235, addrspace=AddrSpace.REG)
      dot_init = dot.after(block, token_in_tile)[0].store(0.0)
      dot = dot.after(dot_init)
      pair_axis = UOp.range(RP, 6, axis_type=AxisType.REDUCE)
      elem = pair_axis * (LANES * dot_pair_width) + lane * dot_pair_width
      qpair = UOp(Ops.STACK, dtypes.float16.vec(2), (q[head * Hd + elem].cast(dtypes.float16), q[head * Hd + elem + 1].cast(dtypes.float16)))
      kpair = UOp(Ops.STACK, dtypes.float16.vec(2), (ksh.after(barrier)[token_in_tile * Hd + elem],
                                                 ksh.after(barrier)[token_in_tile * Hd + elem + 1]))
      fdot = _lower_fdot2(dot.after(pair_axis)[0], qpair, kpair)
      update = dot[0].store(fdot).end(pair_axis)
      # reduce_structure is the descriptor owner; the env var is honored ONLY as a legacy alias when the
      # caller passes reduce_structure=None, never as a production default (the spec always passes a value).
      inline_reduce = (bool(getenv("DECODE_ATTN_BLOCK_TILE_INLINE_REDUCE", 0)) if reduce_structure is None
                       else reduce_structure == "inline")
      reduced = (warp_reduce_sum(dot.after(update)[0], lane, group_width) if inline_reduce
                 else _warp_reduce_sum_staged(dot.after(update)[0], lane, group_width))
      return reduced * scale

    def merge_tail(token_in_tile, new_max, correction, probability):
      dim_axis = UOp.range(R, 7)
      dim = lane * R + dim_axis
      value = (vsh.after(barrier)[token_in_tile * Hd + dim].cast(_F32) if staging == "KV_BOTH" else
               kv_load(1, kvh, split * L + block * TK + token_in_tile, dim).cast(_F32))
      acc_update = acc[dim_axis].store(acc.after(token_in_tile)[dim_axis] * correction + probability * value).end(dim_axis)
      den_update = den.after(acc_update)[0].store(den.after(token_in_tile)[0] * correction + probability)
      max_update = mx.after(den_update)[0].store(new_max).end(token_in_tile)
      return UOp.barrier(UOp.group(max_update)).end(block)

    token_in_tile = UOp.range(TK, 5, axis_type=AxisType.REDUCE)
    in_range = (split * L + block * TK + token_in_tile) < Tc
    score = in_range.where(dot_reduce(token_in_tile), _fc(-float("inf")))
    old_max = mx.after(token_in_tile)[0]
    new_max = old_max.maximum(score)
    correction = in_range.where(_fexp(old_max - new_max), _fc(1.0))
    probability = in_range.where(_fexp(score - new_max), _fc(0.0))
    merged = merge_tail(token_in_tile, new_max, correction, probability)
    final_acc, final_den, final_max = acc.after(merged), den.after(merged), mx.after(merged)
    base = (head * S + split) * W
    output_axis = UOp.range(R, 8)
    output_dim = lane * R + output_axis
    pv = pout[base + output_dim].store(final_acc[output_axis], warp_active).end(output_axis)
    ls = pout.after(pv)[base + Hd].store(final_den[0], lane.eq(0) & warp_active)
    ms = pout.after(ls)[base + (Hd + 1)].store(final_max[0], lane.eq(0) & warp_active)
    suffix = "" if QG == G else f"_qg{QG}"
    geom_suffix = _tile_geometry_suffix(Hq=Hq, split_count=S, lane_width=lane_width, token_block=token_block,
                                        stage_width=stage_width, reduce_structure=reduce_structure,
                                        dot_pair_width=dot_pair_width, score_group_width=score_group_width,
                                        warps=warps, query_group_size=query_group_size, QG=QG)
    return ms.end(kvh, split, query_group, lane, warp).sink(arg=_kernel_info(
      f"flash_block_tiled_xlane_score_pv_tile_whole_cache_{Hq}_{Hd}{suffix}{geom_suffix}",
      coalesced_loads=bool(selected_width)))
  return kernel


def flash_fused_gmax_combine_kernel(Hd:int, Hq:int, S:int, stride:int|None=None, output_fp16:bool=False,
                                    lane_width:int=32):
  W, L_COL, M_COL, LANES, R = Hd + 2, Hd, Hd + 1, lane_width, Hd // lane_width
  if Hd % LANES != 0: raise ValueError(f"fused combine needs Hd%{LANES}==0, got {Hd}")
  NW, stride = _ceildiv(S, LANES), S if stride is None else stride

  def kernel(out:UOp, pout:UOp) -> UOp:
    head = UOp.range(Hq, 0, AxisType.GLOBAL)
    lane = UOp.range(LANES, 1, AxisType.LOCAL)
    weights = UOp.placeholder((S,), _F32, 240, addrspace=AddrSpace.LOCAL)
    global_max = UOp.placeholder((1,), _F32, 241, addrspace=AddrSpace.REG)
    split = UOp.range(S, 2, axis_type=AxisType.REDUCE)
    max_init = global_max.after(head, lane)[0].set(-1e30)
    max_update = max_init[0].set(max_init.after(split)[0].maximum(pout[(head * stride + split) * W + M_COL]), end=split)
    maximum = max_init.after(max_update)[0]
    weight_iteration = UOp.range(NW, 3)
    split_idx = weight_iteration * LANES + lane
    valid_weight = split_idx < S
    safe_split = valid_weight.where(split_idx, split_idx.const_like(0))
    weight_store = weights.after(max_update)[safe_split].store(
      _fexp(pout[(head * stride + safe_split) * W + M_COL] - maximum), valid_weight).end(weight_iteration)
    barrier = UOp.barrier(UOp.group(weight_store))
    acc = UOp.placeholder((R,), _F32, 242, addrspace=AddrSpace.REG)
    den = UOp.placeholder((1,), _F32, 243, addrspace=AddrSpace.REG)
    zero_axis = UOp.range(R, 5)
    acc_init = acc.after(barrier, head, lane)[zero_axis].store(0.0).end(zero_axis)
    den_init = den.after(acc_init)[0].store(0.0)
    acc, den = acc.after(den_init), den.after(den_init)
    split_reduce = UOp.range(S, 4, axis_type=AxisType.REDUCE)
    weight = weights.after(barrier)[split_reduce]
    dim_axis = UOp.range(R, 6)
    dim = lane * R + dim_axis
    acc_update = acc[dim_axis].store(acc.after(split_reduce)[dim_axis] +
      weight * pout[(head * stride + split_reduce) * W + dim]).end(dim_axis)
    den_update = den.after(acc_update)[0].store(den.after(split_reduce)[0] +
      weight * pout[(head * stride + split_reduce) * W + L_COL]).end(split_reduce)
    final_acc, final_den = acc.after(den_update), den.after(den_update)[0]
    output_axis = UOp.range(R, 7)
    output_dim = lane * R + output_axis
    value = final_acc[output_axis] / final_den
    if output_fp16: value = value.cast(dtypes.float16)
    s_suffix = "" if S == {32: 48, 40: 32}.get(Hq) else f"_s{S}"
    lw_suffix = "" if lane_width == 32 else f"_lw{lane_width}"
    combine_name = (f"flash_fused_gmax_combine_f16_{Hq}_{Hd}{s_suffix}{lw_suffix}" if output_fp16
                    else f"flash_fused_gmax_combine_{Hq}_{Hd}{s_suffix}{lw_suffix}")
    return out[head * Hd + output_dim].store(value).end(output_axis).end(head, lane).sink(
      arg=KernelInfo(name=combine_name, opts_to_apply=()))
  return kernel


def flash_single_stage_d512_kernel(Hd:int, Hq:int, Hkv:int, L:int, Tc, *, output_fp16:bool=True):
  """Closed-default construction candidate: S=4 split score/PV + ordered combine in one workgroup.

  This emitter is deliberately not wired into any route.  Its fixed ownership is one warp per
  (split, GQA-head) pair and exists to qualify whether ordinary UOps can carry the communication
  boundary without a global partial buffer.
  """
  S, LANES, TK = 4, 32, 16
  if (Hd, Hq, Hkv) != (128, 32, 8): raise ValueError("single-stage d512 candidate is fixed to Hd=128,Hq=32,Hkv=8")
  G, WARPS, THREADS, R, RP, NB = Hq // Hkv, S * (Hq // Hkv), S * (Hq // Hkv) * LANES, Hd // LANES, Hd // 64, _ceildiv(L, TK)
  W, scale = Hd + 2, 1.0 / (Hd ** 0.5)

  def kernel(out:UOp, q:UOp, cache:UOp) -> UOp:
    from tinygrad.codegen.late.warp_reduce import _warp_reduce_sum_staged
    from tinygrad.codegen.late.flash_decode_intrinsics import fdot2 as _lower_fdot2
    kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
    lane = UOp.range(LANES, 10, AxisType.LOCAL)
    warp = UOp.range(WARPS, 11, AxisType.LOCAL)
    owner_split, grouped_head = warp // G, warp % G
    head, tid = kvh * G + grouped_head, warp * LANES + lane
    ksh = UOp.placeholder((S * TK * Hd,), dtypes.float16, 250, addrspace=AddrSpace.LOCAL)
    vsh = UOp.placeholder((S * TK * Hd,), dtypes.float16, 251, addrspace=AddrSpace.LOCAL)
    partial = UOp.placeholder((WARPS * W,), _F32, 252, addrspace=AddrSpace.LOCAL)
    acc = UOp.placeholder((R,), _F32, 253, addrspace=AddrSpace.REG)
    den = UOp.placeholder((1,), _F32, 254, addrspace=AddrSpace.REG)
    mx = UOp.placeholder((1,), _F32, 255, addrspace=AddrSpace.REG)
    za = UOp.range(R, 2)
    init = acc.after(kvh)[za].store(0.0).end(za)
    init = den.after(init)[0].store(0.0)
    init = mx.after(init)[0].store(-float("inf"))
    acc, den, mx = acc.after(init), den.after(init), mx.after(init)
    block = UOp.range(NB, 3, AxisType.REDUCE)

    # All 512 threads cooperatively stage one K/V tile for each split. The staging loop and
    # barriers are uniform; only arithmetic ownership is warp-specific.
    stage = UOp.range(_ceildiv(S * TK * Hd, THREADS), 4, AxisType.REDUCE)
    idx = stage * THREADS + tid
    stage_split, split_elem = idx // (TK * Hd), idx % (TK * Hd)
    token_stage, elem = split_elem // Hd, split_elem % Hd
    token = stage_split * L + block * TK + token_stage
    valid = (stage_split < S) & (token_stage < TK) & (token < Tc)
    safe_token = valid.where(token, token.const_like(0))
    kstore = ksh[idx].store(cache[0, 0, kvh, safe_token, elem].cast(dtypes.float16), idx < (S * TK * Hd))
    vstore = vsh.after(kstore)[idx].store(cache[1, 0, kvh, safe_token, elem].cast(dtypes.float16), idx < (S * TK * Hd))
    barrier = UOp.barrier(UOp.group(vstore.end(stage)))

    token_in_tile = UOp.range(TK, 5, AxisType.REDUCE)
    owned_token = owner_split * L + block * TK + token_in_tile
    in_range = owned_token < Tc
    dot = UOp.placeholder((1,), _F32, 256, addrspace=AddrSpace.REG)
    dot_init = dot.after(block, token_in_tile)[0].store(0.0)
    pair_axis = UOp.range(RP, 6, AxisType.REDUCE)
    qelem = pair_axis * 64 + lane * 2
    tile_base = owner_split * TK * Hd + token_in_tile * Hd + qelem
    qpair = UOp(Ops.STACK, dtypes.float16.vec(2), (q[head * Hd + qelem].cast(dtypes.float16), q[head * Hd + qelem + 1].cast(dtypes.float16)))
    kpair = UOp(Ops.STACK, dtypes.float16.vec(2), (ksh.after(barrier)[tile_base], ksh.after(barrier)[tile_base + 1]))
    dot_update = dot.after(dot_init)[0].store(_lower_fdot2(dot.after(pair_axis)[0], qpair, kpair)).end(pair_axis)
    score = in_range.where(_warp_reduce_sum_staged(dot.after(dot_update)[0], lane, LANES) * scale, _fc(-float("inf")))
    old_max = mx.after(token_in_tile)[0]
    new_max = old_max.maximum(score)
    correction = in_range.where(_fexp(old_max - new_max), _fc(1.0))
    probability = in_range.where(_fexp(score - new_max), _fc(0.0))
    da = UOp.range(R, 7)
    dim = lane * R + da
    value = vsh.after(barrier)[owner_split * TK * Hd + token_in_tile * Hd + dim].cast(_F32)
    au = acc[da].store(acc.after(token_in_tile)[da] * correction + probability * value).end(da)
    du = den.after(au)[0].store(den.after(token_in_tile)[0] * correction + probability)
    mu = mx.after(du)[0].store(new_max).end(token_in_tile)
    tile_done = UOp.barrier(UOp.group(mu)).end(block)

    # Preserve the legacy ABI internally, but exchange it through LOCAL rather than global memory.
    pa = UOp.range(R, 8)
    pdim = lane * R + pa
    pbase = warp * W
    ps = partial.after(tile_done)[pbase + pdim].store(acc.after(tile_done)[pa]).end(pa)
    ps = partial.after(ps)[pbase + Hd].store(den.after(tile_done)[0], lane.eq(0))
    ps = partial.after(ps)[pbase + Hd + 1].store(mx.after(tile_done)[0], lane.eq(0))
    pbar = UOp.barrier(UOp.group(ps))

    # Only the split-0 owner warp for each grouped head performs the legacy ordered combine.
    active = warp < G
    output_head = active.where(head, head.const_like(0))
    gm = UOp.placeholder((1,), _F32, 257, addrspace=AddrSpace.REG)
    si = UOp.range(S, 12, AxisType.REDUCE)
    split_warp = si * G + grouped_head
    ginit = gm.after(pbar)[0].store(-1e30)
    gupdate = gm.after(ginit)[0].store(gm.after(si)[0].maximum(partial.after(pbar)[split_warp * W + Hd + 1])).end(si)
    maximum = gm.after(gupdate)[0]
    ca = UOp.placeholder((R,), _F32, 258, addrspace=AddrSpace.REG)
    cd = UOp.placeholder((1,), _F32, 259, addrspace=AddrSpace.REG)
    cia = UOp.range(R, 14)
    ci = ca.after(gupdate)[cia].store(0.0).end(cia)
    ci = cd.after(ci)[0].store(0.0)
    ca, cd = ca.after(ci), cd.after(ci)
    sr = UOp.range(S, 13, AxisType.REDUCE)
    sw = sr * G + grouped_head
    weight = _fexp(partial.after(pbar)[sw * W + Hd + 1] - maximum)
    cra = UOp.range(R, 15)
    rdim = lane * R + cra
    cua = ca[cra].store(ca.after(sr)[cra] + weight * partial.after(pbar)[sw * W + rdim]).end(cra)
    cud = cd.after(cua)[0].store(cd.after(sr)[0] + weight * partial.after(pbar)[sw * W + Hd]).end(sr)
    coa = UOp.range(R, 16)
    odim = lane * R + coa
    result = ca.after(cud)[coa] / cd.after(cud)[0]
    if output_fp16: result = result.cast(dtypes.float16)
    store = out[output_head * Hd + odim].store(result, active).end(coa)
    suffix = "f16" if output_fp16 else "f32"
    return store.end(kvh, lane, warp).sink(arg=_kernel_info(f"flash_single_stage_d512_{suffix}_{Hq}_{Hd}"))
  return kernel


def flash_vec_llama_score_pv_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, S:int, Tc, *, wide_kv:bool=False,
                                    wide_q:bool=True, token_bound:int|None=None, guard_kv_loads:bool=False,
                                    separate_kv:bool=False):
  """Llama ``flash_attn_ext_vec`` substrate for d512 decode (closed-default, not routed).

  Faithful transcription of the traced llama kernel (docs/task_workflow/input/
  nv-flash-score-llama-trace-20260813.md): Q is loaded once into registers; K/V are
  streamed straight from global/L2 with no per-tile LDS staging; each 8-lane group scores one KV column
  with a 3-shuffle-stage reduce (4 columns in flight per warp, 16 per block); online softmax and PV
  accumulation happen in registers in the same pass. The only cross-warp exchange is the final PV/den/max
  combine, and the output is the legacy ``pout`` partial ABI so ``flash_fused_gmax_combine_kernel`` merges
  the S=4 splits unchanged.

  Fixed to the d512 shape family (Hd=128, Hq=32, Hkv=8, GQA=4). ``S`` is the KV split count (llama uses 4 at
  context 512); ``MAXC`` bounds the symbolic context so the chunk loop is static. No quant/rope here: this is
  the fp16-KV substrate proof.
  """
  if (Hd, Hq, Hkv) != (128, 32, 8): raise ValueError("llama-vec substrate is fixed to Hd=128,Hq=32,Hkv=8")
  G = Hq // Hkv
  NKQ, LANES, WARPS, THREADS = 8, 32, 4, 128
  GROUPS = LANES // NKQ                        # 4 groups per warp
  R = Hd // NKQ                                # 16 dims per thread
  RP = R // 2                                  # 8 half2 per thread
  COLS_PER_WARP = GROUPS * NKQ                 # 32 columns per warp per chunk
  COLS_PER_CHUNK = THREADS                     # 128 columns per chunk
  CHUNK_STRIDE = S * COLS_PER_CHUNK            # 512: each split owns one interleaved 128-col chunk
  if token_bound is not None and (token_bound > MAXC or token_bound % COLS_PER_CHUNK):
    raise ValueError(f"token_bound must be <= MAXC and a multiple of {COLS_PER_CHUNK}, got {token_bound}")
  NCHUNK = _ceildiv(MAXC if token_bound is None else token_bound, CHUNK_STRIDE)
  W = Hd + 2
  scale = 1.0 / (Hd ** 0.5)

  def build_kernel(pout:UOp, q:UOp, cache:UOp, cache_v:UOp|None) -> UOp:
    from tinygrad.codegen.late.warp_reduce import (_warp_reduce_sum_staged, warp_reduce_sum_across_groups,
                                                   warp_reduce_max_across_groups)
    from tinygrad.codegen.late.flash_decode_intrinsics import fdot2 as _lower_fdot2
    head = UOp.range(Hq, 0, AxisType.GLOBAL)    # 32 heads
    split = UOp.range(S, 1, AxisType.GLOBAL)    # 4 KV splits
    lane = UOp.range(LANES, 10, AxisType.LOCAL)
    warp = UOp.range(WARPS, 11, AxisType.LOCAL)
    kvh = head // G
    # Bitwise lane split keeps the 32-lane range intact (no pm_split_ranges decomposition), so CUDA's
    # __shfl_xor_sync sees the lane along threadIdx.x and the 8-lane group reduce stays warp-aligned.
    glane = lane & (NKQ - 1)
    group = lane >> 3

    def packed_half8(ptr:UOp, idx:UOp, gate:UOp|None=None) -> tuple[UOp, ...]:
      # Match llama's 16-byte cooperative copy. A direct half8 is devectorized
      # into two 8-byte half4 loads; a uint4 input view keeps the aligned
      # 128-bit transfer. The research caller supplies q/cache as zero-copy
      # uint32 bitcast views because pointer reinterpret casts are erased by
      # the current C-style devectorizer before LOAD folding.
      raw_ptr = ptr.src[0] if ptr.op is Ops.RESHAPE else ptr
      if raw_ptr.dtype.base != dtypes.uint32: raise ValueError("wide_kv requires uint32 bitcast views for q and cache")
      indexed = raw_ptr.index(idx // 2)
      words = indexed.load(dtype=dtypes.uint32.vec(4)) if gate is None else \
        indexed.load(UOp.const(dtypes.uint32.vec(4), 0), gate, dtype=dtypes.uint32.vec(4))
      return tuple(words.gep(i // 2).rshift((i & 1) * 16).cast(dtypes.uint16).bitcast(dtypes.float16) for i in range(8))

    def owned_dim(i:UOp|int) -> UOp:
      return (i // 8) * 64 + glane * 8 + (i % 8) if wide_kv else glane * R + i

    # Register-resident Q: 16 scalar halves per thread, loaded once. The 8 lanes of a group cover all 128
    # dims; the 4 groups x 4 warps hold redundant copies so every group can score a column independently.
    # Scalar (not half2-typed) registers keep the DEFINE_REG index pipeline devectorizer-friendly; the packed
    # half2 is rebuilt with STACK at the dot, exactly as the legacy tile builds its q/k pairs.
    if wide_kv and wide_q:
      qlanes = packed_half8(q, head * Hd + glane * 8) + packed_half8(q, head * Hd + 64 + glane * 8)
    else:
      qreg = UOp.placeholder((R,), dtypes.float16, 300, addrspace=AddrSpace.REG)
      qp = UOp.range(R, 40)
      qe = owned_dim(qp)
      qload = qreg[qp].store(q[head * Hd + qe].cast(dtypes.float16)).end(qp)
      qreg = qreg.after(qload)
      qlanes = tuple(qreg[i] for i in range(R))

    acc = UOp.placeholder((R,), _F32, 301, addrspace=AddrSpace.REG)
    den = UOp.placeholder((1,), _F32, 302, addrspace=AddrSpace.REG)
    mx = UOp.placeholder((1,), _F32, 303, addrspace=AddrSpace.REG)
    za = UOp.range(R, 41)
    init = acc.after(head, split)[za].store(0.0).end(za)
    init = den.after(init)[0].store(0.0)
    init = mx.after(init)[0].store(-float("inf"))
    acc, den, mx = acc.after(init), den.after(init), mx.after(init)

    chunk = UOp.range(NCHUNK, 3, axis_type=AxisType.REDUCE)

    # Per-column scores for this group's 8 columns, held in registers. Each 8-lane group scores
    # column (warp*32 + group*8 + j); the 8 lanes hold complementary 16-dim Q slices and the
    # 8-lane reduce broadcasts the full dot to every lane of the group.
    score = UOp.placeholder((NKQ,), _F32, 304, addrspace=AddrSpace.REG)
    j = UOp.range(NKQ, 5, axis_type=AxisType.REDUCE)
    col = split * COLS_PER_CHUNK + chunk * CHUNK_STRIDE + warp * COLS_PER_WARP + group * NKQ + j
    token = col
    valid = token < Tc
    dot = UOp.placeholder((1,), _F32, 305, addrspace=AddrSpace.REG)
    dot_init = dot.after(chunk, j)[0].store(0.0)
    dot = dot.after(dot_init)
    if wide_kv:
      kbase = (kvh * MAXC + token) * Hd
      kgate = valid if guard_kv_loads else None
      klanes = packed_half8(cache, kbase + glane * 8, kgate) + packed_half8(cache, kbase + 64 + glane * 8, kgate)
      dot_value = dot[0]
      for pi in range(RP):
        qpair = UOp(Ops.STACK, dtypes.float16.vec(2), (qlanes[pi * 2], qlanes[pi * 2 + 1]))
        kpair = UOp(Ops.STACK, dtypes.float16.vec(2), (klanes[pi * 2], klanes[pi * 2 + 1]))
        dot_value = _lower_fdot2(dot_value, qpair, kpair)
      dot_update = dot[0].store(dot_value)
    else:
      p = UOp.range(RP, 6, axis_type=AxisType.REDUCE)
      ke = glane * R + p * 2
      qpair = UOp(Ops.STACK, dtypes.float16.vec(2), (qreg[p * 2], qreg[p * 2 + 1]))
      kpair = UOp(Ops.STACK, dtypes.float16.vec(2), (cache[0, 0, kvh, token, ke].cast(dtypes.float16),
                                                    cache[0, 0, kvh, token, ke + 1].cast(dtypes.float16)))
      dot_update = dot[0].store(_lower_fdot2(dot.after(p)[0], qpair, kpair)).end(p)
    sc = valid.where(_warp_reduce_sum_staged(dot.after(dot_update)[0], lane, NKQ) * scale, _fc(-float("inf")))
    score_store = score[j].store(sc).end(j)
    score = score.after(score_store)

    # Group-wide max over this chunk's 8 columns, then cross-group reduce to the warp-wide max.
    group_max = UOp.placeholder((1,), _F32, 306, addrspace=AddrSpace.REG)
    jm = UOp.range(NKQ, 7, axis_type=AxisType.REDUCE)
    gm_init = group_max.after(score_store)[0].set(-float("inf"))
    gm = gm_init[0].set(gm_init.after(jm)[0].maximum(score[jm]), end=jm)
    warp_max = warp_reduce_max_across_groups(gm_init.after(gm)[0], lane, NKQ)

    # Online-softmax rescale by the warp max, then PV/den accumulation for this group's 8 columns.
    valid_chunk = warp_max > -1e30
    old_max = mx.after(score_store)[0]
    rescale = valid_chunk.where(_fexp(old_max - warp_max), _fc(1.0))
    new_max = valid_chunk.where(warp_max, old_max)
    da = UOp.range(R, 8)
    au = acc[da].store(acc.after(score_store)[da] * rescale).end(da)
    du = den.after(au)[0].store(den.after(score_store)[0] * rescale)
    mu = mx.after(du)[0].store(new_max)
    acc, den, mx = acc.after(au), den.after(du), mx.after(mu)

    jv = UOp.range(NKQ, 9, axis_type=AxisType.REDUCE)
    tokv = split * COLS_PER_CHUNK + chunk * CHUNK_STRIDE + warp * COLS_PER_WARP + group * NKQ + jv
    validv = tokv < Tc
    prob = validv.where(_fexp(score[jv] - new_max), _fc(0.0))
    if wide_kv:
      vptr = cache_v if separate_kv else cache
      assert vptr is not None
      vbase = (kvh * MAXC + tokv) * Hd if separate_kv else ((Hkv + kvh) * MAXC + tokv) * Hd
      vgate = validv if guard_kv_loads else None
      vlanes = packed_half8(vptr, vbase + glane * 8, vgate) + packed_half8(vptr, vbase + 64 + glane * 8, vgate)
      a2 = UOp.group(*[acc[di].store(acc.after(jv)[di] + prob * vlanes[di].cast(_F32)) for di in range(R)])
    else:
      dv = UOp.range(R, 12)
      vdim = glane * R + dv
      vval = cache[1, 0, kvh, tokv, vdim].cast(_F32)
      a2 = acc[dv].store(acc.after(jv)[dv] + prob * vval).end(dv)
    # mu (the running-max store) is a loop-carried register too; order it inside the chunk so it is not
    # hoisted out as a dead tail and the next chunk reads the updated softmax frame.
    d2 = den.after(a2, mu)[0].store(den.after(jv)[0] + prob)
    chunk_end = d2.end(jv).end(chunk)

    # Cross-group sum of PV and den (the 4 groups own disjoint columns but the same 16-dim slices).
    # Only lanes 0..7 are group-0 owners, so they alone write each dim slice to avoid a 4-way same-value
    # store race in shared memory. den/max are already warp-uniform after the cross-group reduce.
    dr = UOp.range(R, 13)
    warp_acc = warp_reduce_sum_across_groups(acc.after(chunk_end)[dr], lane, NKQ, slot_base=320)
    warp_den = warp_reduce_sum_across_groups(den.after(chunk_end)[0], lane, NKQ, slot_base=340)
    warp_mx = warp_reduce_max_across_groups(mx.after(chunk_end)[0], lane, NKQ, slot_base=350)

    sh_pv = UOp.placeholder((WARPS * Hd,), _F32, 360, addrspace=AddrSpace.LOCAL)
    sh_den = UOp.placeholder((WARPS,), _F32, 361, addrspace=AddrSpace.LOCAL)
    sh_mx = UOp.placeholder((WARPS,), _F32, 362, addrspace=AddrSpace.LOCAL)
    ps = sh_pv[warp * Hd + owned_dim(dr)].store(warp_acc, lane < NKQ).end(dr)
    ps = sh_den.after(ps)[warp].store(warp_den, lane.eq(0))
    ps = sh_mx.after(ps)[warp].store(warp_mx, lane.eq(0))
    barrier = UOp.barrier(UOp.group(ps))

    # Block-wide max over the 4 warp partials, then re-normalize each warp's PV/den by exp(warp_max -
    # block_max) before summing. This is llama's fattn-vec.cuh:434-500: warp partials are only valid in a
    # common softmax frame after the global-max rescale.
    block_max = UOp.placeholder((1,), _F32, 370, addrspace=AddrSpace.REG)
    wmax = UOp.range(WARPS, 14, axis_type=AxisType.REDUCE)
    bm_init = block_max.after(barrier)[0].set(-float("inf"))
    bm = bm_init[0].set(bm_init.after(wmax)[0].maximum(sh_mx.after(barrier)[wmax]), end=wmax)
    global_max = bm_init.after(bm)[0]

    block_pv = UOp.placeholder((R,), _F32, 371, addrspace=AddrSpace.REG)
    block_den = UOp.placeholder((1,), _F32, 372, addrspace=AddrSpace.REG)
    zr = UOp.range(R, 15)
    pv_init = block_pv.after(bm)[zr].store(0.0).end(zr)
    den_init = block_den.after(pv_init)[0].store(0.0)
    block_pv, block_den = block_pv.after(den_init), block_den.after(den_init)

    ws = UOp.range(WARPS, 16, axis_type=AxisType.REDUCE)
    # A physical partition may be wholly outside logical Tc (the current llama-like
    # d512 geometry has six 128-token parts, while Tc is only 513).  In that case
    # every warp max and the block max are -inf.  exp(-inf - -inf) is NaN and used
    # to poison the empty partial before the outer combine can give it zero weight.
    # Preserve the partial ABI for an empty part: PV=0, denominator=0, max=-inf.
    block_valid = global_max > -1e30
    weight = block_valid.where(_fexp(sh_mx.after(barrier)[ws] - global_max), _fc(0.0))
    wr = UOp.range(R, 17)
    pv_up = block_pv[wr].store(block_pv.after(ws)[wr] + weight * sh_pv.after(barrier)[ws * Hd + owned_dim(wr)]).end(wr)
    den_up = block_den.after(pv_up)[0].store(block_den.after(ws)[0] + weight * sh_den.after(barrier)[ws]).end(ws)
    final_pv, final_den = block_pv.after(den_up), block_den.after(den_up)

    base = (head * S + split) * W
    output_axis = UOp.range(R, 18)
    output_dim = owned_dim(output_axis)
    pv = pout[base + output_dim].store(final_pv[output_axis]).end(output_axis)
    ls = pout.after(pv)[base + Hd].store(final_den[0], lane.eq(0))
    ms = pout.after(ls)[base + (Hd + 1)].store(global_max, lane.eq(0))
    return ms.end(head, split, lane, warp).sink(arg=_kernel_info(
      f"flash_vec_llama_score_pv_{Hq}_{Hd}_{S}{'_widekv16' if wide_kv else ''}"
      f"{'_guardkv' if guard_kv_loads else ''}{'_separatekv' if separate_kv else ''}", coalesced_loads=True))

  if separate_kv:
    def kernel_separate(pout:UOp, q:UOp, cache_k:UOp, cache_v:UOp) -> UOp:
      return build_kernel(pout, q, cache_k, cache_v)
    return kernel_separate

  def kernel_combined(pout:UOp, q:UOp, cache:UOp) -> UOp:
    return build_kernel(pout, q, cache, None)
  return kernel_combined


@dataclass(frozen=True)
class LiveSplitGeometrySpec:
  split_count: int
  token_block: int = 16

  def validate(self) -> None:
    if self.split_count < 1: raise ValueError(f"split_count must be >= 1, got {self.split_count!r}")
    if self.token_block < 1: raise ValueError(f"token_block must be >= 1, got {self.token_block!r}")

  def per_split_length(self, Tc): return _ceildiv_uop(Tc, self.split_count)
  def aligned_per_split_length(self, Tc): return _ceildiv_uop(self.per_split_length(Tc), self.token_block) * self.token_block
  def blocks(self, Tc): return _ceildiv_uop(self.aligned_per_split_length(Tc), self.token_block)


@dataclass(frozen=True)
class BufferRole:
  name: str
  dtype: str
  shape: tuple[int, ...]
  optional_on: str|None = None


@dataclass(frozen=True)
class FlashDecodeTileSpec:
  Hq: int
  Hd: int
  Hkv: int
  MAXC: int
  split_count: int
  staging: str = "KV_BOTH"
  quant: bool = False
  rope: bool = False
  token_block: int = 16
  query_group_size: int|None = None
  stage_width: int = 1
  lane_width: int = 32
  score_group_width: int|None = None
  warps: int|None = None
  reduce_structure: str = "staged"
  dot_pair_width: int = 2
  target: str|None = None

  def validate(self) -> None:
    if min(self.Hq, self.Hd, self.Hkv, self.MAXC) <= 0: raise ValueError("Hq, Hd, Hkv and MAXC must be positive")
    if self.staging != "KV_BOTH": raise ValueError(f"production flash decode requires staging='KV_BOTH', got {self.staging!r}")
    if self.token_block < 1: raise ValueError(f"token_block must be >= 1, got {self.token_block}")
    if self.Hq % self.Hkv != 0: raise ValueError(f"Hq must be divisible by Hkv, got Hq={self.Hq} Hkv={self.Hkv}")
    if self.query_group_size is not None and not 1 <= self.query_group_size <= self.Hq // self.Hkv:
      raise ValueError(f"query_group_size must be in 1..{self.Hq // self.Hkv}, got {self.query_group_size}")
    if self.stage_width not in (1, 2, 4, 8): raise ValueError(f"stage_width must be one of 1,2,4,8, got {self.stage_width}")
    if self.lane_width < 1 or self.lane_width & (self.lane_width - 1):
      raise ValueError(f"lane_width must be a positive power of two, got {self.lane_width}")
    if self.score_group_width is not None and self.score_group_width != self.lane_width:
      raise ValueError(f"score_group_width must equal lane_width={self.lane_width} or be None, "
                       f"got {self.score_group_width}")
    qg = self.query_group_size if self.query_group_size is not None else self.Hq // self.Hkv
    if self.warps is not None and self.warps < qg:
      raise ValueError(f"warps must be >= query_group_size={qg} when set, got {self.warps}")
    if self.dot_pair_width < 1: raise ValueError(f"dot_pair_width must be >= 1, got {self.dot_pair_width}")
    if self.Hd % (self.lane_width * self.dot_pair_width) != 0:
      raise ValueError(f"Hd must be divisible by lane_width*dot_pair_width={self.lane_width * self.dot_pair_width}, "
                       f"got Hd={self.Hd}")
    if self.reduce_structure not in {"staged", "inline"}:
      raise ValueError(f"reduce_structure must be one of 'staged','inline', got {self.reduce_structure!r}")
    self.geometry.validate()

  @property
  def geometry(self) -> LiveSplitGeometrySpec: return LiveSplitGeometrySpec(self.split_count, self.token_block)

  @property
  def buffer_roles(self) -> tuple[BufferRole, ...]:
    roles = [BufferRole("pout", "float32", (self.Hq * self.split_count * (self.Hd + 2),)),
             BufferRole("q", "float16", (self.Hq * self.Hd,)),
             BufferRole("cache", "float16", (2, 1, self.Hkv, self.MAXC, self.Hd))]
    if self.quant: roles.append(BufferRole("kvscale", "float16", (2, 1, self.Hkv, self.MAXC), "quant"))
    if self.rope: roles.append(BufferRole("freqs", "float16", (2, self.MAXC, self.Hd // 2), "rope"))
    return tuple(roles)

  @property
  def kernel_name(self) -> str:
    G = self.Hq // self.Hkv
    QG = G if self.query_group_size is None else self.query_group_size
    suffix = "" if QG == G else f"_qg{QG}"
    geom_suffix = _tile_geometry_suffix(Hq=self.Hq, split_count=self.split_count, lane_width=self.lane_width,
                                        token_block=self.token_block, stage_width=self.stage_width,
                                        reduce_structure=self.reduce_structure, dot_pair_width=self.dot_pair_width,
                                        score_group_width=self.score_group_width, warps=self.warps,
                                        query_group_size=self.query_group_size, QG=QG)
    return f"flash_block_tiled_xlane_score_pv_tile_whole_cache_{self.Hq}_{self.Hd}{suffix}{geom_suffix}"

  def emit(self, Tc:UOp):
    self.validate()
    return flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel(
      self.Hd, self.Hq, self.Hkv, self.MAXC, self.geometry.aligned_per_split_length(Tc), self.split_count, Tc,
      staging=self.staging, quant=self.quant, rope=self.rope, query_group_size=self.query_group_size,
      stage_width=self.stage_width, token_block=self.token_block, lane_width=self.lane_width,
      score_group_width=self.score_group_width, warps=self.warps, reduce_structure=self.reduce_structure,
      dot_pair_width=self.dot_pair_width)

  def to_json(self) -> dict[str, Any]:
    return {key:getattr(self, key) for key in ("Hq", "Hd", "Hkv", "MAXC", "split_count", "staging", "quant", "rope",
                                                 "token_block", "query_group_size", "stage_width", "lane_width",
                                                 "score_group_width", "warps", "reduce_structure",
                                                 "dot_pair_width", "target")}


@dataclass(frozen=True)
class FlashCombineSpec:
  Hd: int
  Hq: int
  split_count: int
  stride: int|None = None
  output_fp16: bool = False
  lane_width: int = 32

  def validate(self) -> None:
    if min(self.Hd, self.Hq, self.split_count) <= 0: raise ValueError("Hd, Hq and split_count must be positive")
    if self.stride is not None and self.stride < 1: raise ValueError(f"stride must be >= 1, got {self.stride}")
    if self.lane_width < 1 or self.lane_width & (self.lane_width - 1):
      raise ValueError(f"lane_width must be a positive power of two, got {self.lane_width}")
    if self.Hd % self.lane_width != 0:
      raise ValueError(f"Hd must be divisible by lane_width={self.lane_width}, got Hd={self.Hd}")

  @property
  def kernel_name(self) -> str:
    prefix = "flash_fused_gmax_combine_f16" if self.output_fp16 else "flash_fused_gmax_combine"
    route_split = {32: 48, 40: 32}.get(self.Hq)
    suffix = ""
    if self.split_count != route_split: suffix += f"_s{self.split_count}"
    if self.lane_width != 32: suffix += f"_lw{self.lane_width}"
    return f"{prefix}_{self.Hq}_{self.Hd}{suffix}"
  def emit(self):
    self.validate()
    return flash_fused_gmax_combine_kernel(self.Hd, self.Hq, self.split_count, self.stride, self.output_fp16,
                                           self.lane_width)
  def to_json(self) -> dict[str, Any]:
    return {key:getattr(self, key) for key in ("Hd", "Hq", "split_count", "stride", "output_fp16", "lane_width")}


@dataclass(frozen=True)
class FlashDecodeAttentionSpec:
  tile: FlashDecodeTileSpec
  combine: FlashCombineSpec|None = None

  @property
  def descriptor_artifact(self) -> str: return "FlashDecodeAttentionSpec"
  def validate(self): self.tile.validate(); self.combine.validate() if self.combine is not None else None
  def emit_tile(self, Tc:UOp): self.validate(); return self.tile.emit(Tc)
  def emit_combine(self):
    self.validate()
    if self.combine is None: raise ValueError("combine was not requested")
    return self.combine.emit()
  @property
  def emitted_kernel_names(self) -> tuple[str, ...]:
    return (self.tile.kernel_name,) if self.combine is None else (self.tile.kernel_name, self.combine.kernel_name)


def describe_flash_decode_attention(Hq:int, Hd:int, Hkv:int, MAXC:int, S:int, *, staging:str="KV_BOTH",
                                    fused_combine:bool=True, quant:bool=False, rope:bool=False,
                                    combine_stride:int|None=None, query_group_size:int|None=None,
                                    stage_width:int=1, token_block:int=16, lane_width:int=32,
                                    score_group_width:int|None=None, warps:int|None=None,
                                    reduce_structure:str="staged", dot_pair_width:int=2,
                                    combine_lane_width:int|None=None,
                                    combine_fp16:bool=False) -> FlashDecodeAttentionSpec:
  tile = FlashDecodeTileSpec(Hq, Hd, Hkv, MAXC, S, staging, quant, rope, query_group_size=query_group_size,
                             stage_width=stage_width, token_block=token_block, lane_width=lane_width,
                             score_group_width=score_group_width, warps=warps, reduce_structure=reduce_structure,
                             dot_pair_width=dot_pair_width)
  return FlashDecodeAttentionSpec(
    tile,
    FlashCombineSpec(Hd, Hq, S, combine_stride, output_fp16=combine_fp16,
                     lane_width=tile.lane_width if combine_lane_width is None else combine_lane_width)
    if fused_combine else None)


def emit_flash_decode_tile(spec:FlashDecodeAttentionSpec, Tc:UOp): return spec.emit_tile(Tc)
def emit_flash_decode_combine(spec:FlashDecodeAttentionSpec): return spec.emit_combine()


@dataclass(frozen=True)
class FlashDecodeCapability:
  """TG7 capability authority (docs/task_workflow/input/target-capability-policy-decoupling-scope-20260730.md):
  can the resolved target's renderer express what `dot_reduce` (flash_block_tiled_xlane_score_pv_tile_...)
  requires? Every field is copied verbatim from the renderer that was actually resolved -- never inferred
  from a device/backend string, and never conflated with promotion (see FlashDecodeAdmission below). Mirrors
  QKPrimitiveCapability's shape (tinygrad/llm/qk_primitives.py, TG3) so the same two-question pattern is
  reused, not reinvented.

  `supports_warp_shfl_xor` covers the lane-reduction ladder every score reduces through
  (codegen/late/warp_reduce.py, TG1); `supports_fdot2` covers the packed-half2 QK dot product this package
  makes renderer-lowered (codegen/late/flash_decode_intrinsics.py). The opt-in DECODE_FAST_EXP2 fast path
  (also renderer-lowered by this package, as `exp2f`) is deliberately NOT part of `satisfied`: it is off by
  default in production, and a target that lacks it still fails loudly at lowering if the env flag is set --
  gating route admission on an experimental, rarely-used knob would be a second, redundant fail-safe."""
  supports_warp_shfl_xor: bool | None = None
  supports_fdot2: bool | None = None

  @property
  def satisfied(self) -> bool:
    return self.supports_warp_shfl_xor is True and self.supports_fdot2 is True


def flash_decode_capability_from_renderer(renderer:object|None) -> FlashDecodeCapability:
  """Copy only the immutable renderer capability facts flash decode requires. `renderer` should be an
  already-open target's renderer (e.g. `Device[device].renderer`, called only on a device already opened
  elsewhere in the model pipeline) -- never a fresh probe here, and never a parallel capability object."""
  if renderer is None: return FlashDecodeCapability()
  return FlashDecodeCapability(getattr(renderer, "supports_warp_shfl_xor", None),
                                getattr(renderer, "supports_flash_decode_fdot2", None))


def flash_decode_capability_from_device_facts(device_facts:object|None) -> FlashDecodeCapability:
  """Same shape as qk_primitive_capability_from_device_facts (tinygrad/llm/qk_primitives.py, TG3): read the
  load-entry DeviceFacts scan (tinygrad/llm/device_facts.py) verbatim, never a fresh probe, never a parallel
  facts object. Not yet wired to a production call site -- doing so requires threading the load-entry
  DeviceFacts object into decode_routes.py's flash-decode bind, which belongs to the model.py owner (see
  model.py's `_flash_decode` at the FLASH_DECODE_CANDIDATE.bind call). Provided so that wiring is a one-line
  change, not a new capability path."""
  if device_facts is None: return FlashDecodeCapability()
  capabilities = getattr(device_facts, "capabilities", None)
  return FlashDecodeCapability(getattr(capabilities, "supports_warp_shfl_xor", None),
                                getattr(capabilities, "supports_flash_decode_fdot2", None))


def flash_decode_target_promoted(route_plan:object|None, target:tuple[str|None, str|None]) -> bool:
  """TG3-shaped policy check (mirrors qk_primitives._qk_target_promoted): absence of a route plan (or of its
  target_promoted method) reads as undecided-by-target -- admitted -- not denied; see
  ModelRoutePlan.target_promoted's own docstring for why. No route_plan reaches flash decode's bind today (no
  production call site threads one through -- see flash_decode_capability_from_device_facts above), so this
  currently always resolves True; the mechanism is wired so a future promotion record is a one-line change."""
  check = getattr(route_plan, "target_promoted", None)
  return True if check is None else check(target)


@dataclass(frozen=True)
class FlashDecodeAdmission:
  """The three independent TG7 answers retained for one bind attempt: shape (decode_routes.py's existing
  authority -- unchanged from the pre-TG7 `supports()` shape check), capability (this file, read from
  renderer facts), and promotion (ModelRoutePlan.target_promoted, tinygrad/llm/model_route_plan.py --
  BoltBeam-sourced route policy, TG3's authority, reused rather than restated). `reason` gives each rejection
  a distinct, observable label instead of the pre-TG7 silent `device == "AMD"` fallback.
  `epilogue_fusion_promoted` is the L1 decode epilogue-fusion answer (closed default, resolved by
  decode_routes.py bind from model_route_plan.decode_epilogue_fusion_promoted; it gates the fused combine/
  epilogue variants only -- the legacy `admitted` route is unchanged by it). `combine_fusion_promoted` is the
  L1 M5 flash-combine fp16 absorption answer (closed default, resolved by decode_routes.py bind from
  model_route_plan.decode_flash_combine_fusion_promoted; it gates the fp16 combine variant
  flash_fused_gmax_combine_f16_* only -- deliberately SEPARATE from M2's epilogue-fusion record, and the
  legacy fp32 combine is unchanged by it)."""
  shape_ok: bool
  capability: FlashDecodeCapability
  target_promoted: bool
  epilogue_fusion_promoted: bool = False
  combine_fusion_promoted: bool = False

  @property
  def admitted(self) -> bool:
    return self.shape_ok and self.capability.satisfied and self.target_promoted

  @property
  def fusion_admitted(self) -> bool: return self.admitted and self.epilogue_fusion_promoted

  @property
  def combine_fusion_admitted(self) -> bool: return self.admitted and self.combine_fusion_promoted

  @property
  def reason(self) -> str | None:
    if self.admitted: return None
    if not self.shape_ok: return "shape_not_supported"
    if not self.capability.satisfied: return "capability_missing"
    return "policy_target_not_promoted"


@dataclass(frozen=True)
class FlashDecodeRouteConfig:
  candidate_id: str
  route_id: str
  query_heads: int
  split_size: int
  query_group_size: int|None
  stage_width: int
  kv_heads: int = 8
  head_dim: int = 128
  staging: str = "KV_BOTH"

  def shape_ok(self, B:int, Hq:int, Hkv:int, Hd:int) -> bool:
    """Shape admissibility only (scope section 3.1: bind() in decode_routes.py owns shape gates). Unchanged
    from the pre-TG7 `supports()` shape check -- the only thing that moved is the backend/capability/policy
    question this used to be ANDed with, split out into FlashDecodeAdmission above."""
    return (B, Hq, Hkv, Hd) == (1, self.query_heads, self.kv_heads, self.head_dim)

  def evaluate(self, B:int, Hq:int, Hkv:int, Hd:int, capability:FlashDecodeCapability, target_promoted:bool,
               epilogue_fusion_promoted:bool=False, combine_fusion_promoted:bool=False) -> FlashDecodeAdmission:
    return FlashDecodeAdmission(self.shape_ok(B, Hq, Hkv, Hd), capability, target_promoted,
                                epilogue_fusion_promoted, combine_fusion_promoted)


FLASH_DECODE_G4 = FlashDecodeRouteConfig("attention_decode.flash_live_split", "decode_flash_live_split_g4_kvboth",
                                          32, 48, None, 1)
FLASH_DECODE_G5 = FlashDecodeRouteConfig("attention_decode.flash_live_split_g5", "decode_flash_live_split_g5_kvboth",
                                          40, 32, 2, 4)

def _adaptive_split_lease_admitted(route:FlashDecodeRouteConfig|None, S:int, query_group_size:int|None,
                                   staging:str, MAXC:int) -> bool:
  return route is not None and route.query_heads == FLASH_DECODE_G4.query_heads and MAXC <= 1024 and \
    (64, route.query_group_size, route.staging) == (S, query_group_size, staging)


def flash_decode_live_split_block_tile(q:Tensor, cache_kv:Tensor, Tc:UOp, Hd:int, Hq:int, Hkv:int, MAXC:int, S:int,
                                       staging:str="KV_BOTH", fused_combine:bool=True, kv_scale:Tensor|None=None,
                                       freqs:Tensor|None=None, query_group_size:int|None=None, stage_width:int=1,
                                       token_block:int=16, lane_width:int=32, score_group_width:int|None=None,
                                       warps:int|None=None, reduce_structure:str="staged", dot_pair_width:int=2,
                                       combine_lane_width:int|None=None, combine_fp16:bool=False,
                                       split_count_leased:bool=False) -> Tensor:
  """Execute the selected live-split flash decode and return ``[Hq, Hd]``."""
  if not fused_combine: raise ValueError("fused_combine=False is no longer supported for decode live-split routes")
  # TG7: this is a pure shape-based route SELECTION (which of G4/G5 matches, to label the emitted
  # KernelProgram) -- capability and promotion were already decided by the caller's bind() before this
  # function is ever invoked (decode_routes.flash_decode_attention_route only calls here once binding
  # succeeded), so re-deriving them from `device` here would be both redundant and -- inside a Tensor Function
  # dispatch, which is exactly where this runs -- unable to open a device at all (see
  # decode_routes._flash_decode_capability_and_target_for_device's docstring).
  route = next((row for row in (FLASH_DECODE_G4, FLASH_DECODE_G5) if row.shape_ok(1, Hq, Hkv, Hd)), None)
  # stage_width/reduce_structure/dot_pair_width are searchable geometry overrides (P3); the promoted-route
  # identity is the split/query-group/staging triple, not the staging coalesce width.
  admitted = route is not None and (route.split_size, route.query_group_size, route.staging) == \
      (S, query_group_size, staging)
  # Env-gated coarse-split research override (FLASH_DECODE_COARSE_SPLIT): admit the env-selected
  # split for the G4 d512 route in addition to the promoted one. Unset env leaves `admitted` exactly
  # as before, so the promoted route stays byte-identical to today.
  coarse_split = flash_decode_coarse_split_override()
  if not admitted and route is not None and route.query_heads == FLASH_DECODE_G4.query_heads and coarse_split:
    admitted = (coarse_split, route.query_group_size, route.staging) == (S, query_group_size, staging)
  # Closed-default graph-local lease used by the qualified adaptive policy.
  # Arbitrary split geometry remains inadmissible.
  if not admitted and split_count_leased:
    admitted = _adaptive_split_lease_admitted(route, S, query_group_size, staging, MAXC)
  if not admitted:
    raise ValueError("flash decode geometry is not an admitted promoted route")
  quant, rope = kv_scale is not None, freqs is not None
  inputs = (q.reshape(Hq * Hd), cache_kv) + ((kv_scale,) if quant else ()) + ((freqs,) if rope else ())
  spec = describe_flash_decode_attention(Hq, Hd, Hkv, MAXC, S, staging=staging, quant=quant, rope=rope,
                                         query_group_size=query_group_size, stage_width=stage_width,
                                         token_block=token_block, lane_width=lane_width,
                                         score_group_width=score_group_width, warps=warps,
                                         reduce_structure=reduce_structure, dot_pair_width=dot_pair_width,
                                         combine_lane_width=combine_lane_width,
                                         combine_fp16=combine_fp16)
  tile_program = KernelProgram(route.route_id, f"{route.candidate_id}.tile",
    KernelProgramProvenance.MACHINE_SEARCH_GENERATED, spec.emit_tile(Tc),
    output_spec=OutputSpec((Hq * S * (Hd + 2),), dtypes.float32))
  partial = execute_promoted_program(None, *inputs, program=tile_program)
  # M5 typed boundary (m5-variant-reopen-boundary-p0-scope-20260803.md section 3.1): the fp16
  # combine declares its typed output layout -- fp16 (Hq*Hd,) row-major, viewable as (Hq, Hd),
  # no permutation/stride/padding. The emitted kernel is unchanged; only the boundary's declared
  # output ABI gains the layout/view metadata. combine_fp16 is set by the only call site from
  # FlashDecodeAdmission.combine_fusion_admitted (decode_routes.py), so recording it here IS the
  # producer-side combine-fusion gate state the consumer validator requires (scope 4(d)).
  combine_typed = None
  if combine_fp16:
    combine_typed = DeclaredTypedOutput(TypedLayout(dtypes.float16, (Hq * Hd,), (Hq, Hd)),
                                        combine_fusion_admitted=combine_fp16)
  combine_program = KernelProgram(route.route_id, f"{route.candidate_id}.combine",
    KernelProgramProvenance.MACHINE_SEARCH_GENERATED, spec.emit_combine(),
    output_spec=OutputSpec((Hq * Hd,), dtypes.float16 if combine_fp16 else dtypes.float32,
                           typed_output=combine_typed))
  out = execute_promoted_program(None, partial, program=combine_program)
  return out.reshape(Hq, Hd)
