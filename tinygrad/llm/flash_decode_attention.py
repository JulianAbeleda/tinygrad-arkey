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
                                                              stage_width:int|None=None):
  """Emit the selected live-split tile: LDS K/V, online softmax, and sharded PV."""
  if Hd % 64 != 0: raise ValueError(f"block tile requires Hd%64==0, got {Hd}")
  if staging not in {"KV_BOTH", "K_ONLY"}: raise ValueError(f"unsupported staging={staging!r}")
  G = Hq // Hkv
  QG = G if query_group_size is None else query_group_size
  if QG < 1 or QG > G: raise ValueError(f"query_group_size must be in 1..{G}, got {QG}")
  NG, W, LANES, WARPS, TK = _ceildiv(G, QG), Hd + 2, 32, QG, 16
  THREADS, R, RP = LANES * WARPS, Hd // LANES, Hd // 64
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
      elem = pair_axis * 64 + lane * 2
      qpair = UOp(Ops.STACK, dtypes.float16.vec(2), (q[head * Hd + elem].cast(dtypes.float16), q[head * Hd + elem + 1].cast(dtypes.float16)))
      kpair = UOp(Ops.STACK, dtypes.float16.vec(2), (ksh.after(barrier)[token_in_tile * Hd + elem],
                                                 ksh.after(barrier)[token_in_tile * Hd + elem + 1]))
      fdot = _lower_fdot2(dot.after(pair_axis)[0], qpair, kpair)
      update = dot[0].store(fdot).end(pair_axis)
      reduced = (warp_reduce_sum(dot.after(update)[0], lane, LANES) if getenv("DECODE_ATTN_BLOCK_TILE_INLINE_REDUCE", 0)
                 else _warp_reduce_sum_staged(dot.after(update)[0], lane, LANES))
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
    return ms.end(kvh, split, query_group, lane, warp).sink(arg=_kernel_info(
      f"flash_block_tiled_xlane_score_pv_tile_whole_cache_{Hq}_{Hd}{suffix}", coalesced_loads=bool(selected_width)))
  return kernel


def flash_fused_gmax_combine_kernel(Hd:int, Hq:int, S:int, stride:int|None=None, output_fp16:bool=False):
  W, L_COL, M_COL, LANES, R = Hd + 2, Hd, Hd + 1, 32, Hd // 32
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
    combine_name = f"flash_fused_gmax_combine_f16_{Hq}_{Hd}" if output_fp16 else f"flash_fused_gmax_combine_{Hq}_{Hd}"
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


def flash_vec_llama_score_pv_kernel(Hd:int, Hq:int, Hkv:int, MAXC:int, S:int, Tc):
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
  NCHUNK = _ceildiv(MAXC, CHUNK_STRIDE)        # 9 at MAXC=4608
  W = Hd + 2
  scale = 1.0 / (Hd ** 0.5)

  def kernel(pout:UOp, q:UOp, cache:UOp) -> UOp:
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

    # Register-resident Q: 16 scalar halves per thread, loaded once. The 8 lanes of a group cover all 128
    # dims; the 4 groups x 4 warps hold redundant copies so every group can score a column independently.
    # Scalar (not half2-typed) registers keep the DEFINE_REG index pipeline devectorizer-friendly; the packed
    # half2 is rebuilt with STACK at the dot, exactly as the legacy tile builds its q/k pairs.
    qreg = UOp.placeholder((R,), dtypes.float16, 300, addrspace=AddrSpace.REG)
    qp = UOp.range(R, 40)
    qe = glane * R + qp
    qload = qreg[qp].store(q[head * Hd + qe].cast(dtypes.float16)).end(qp)
    qreg = qreg.after(qload)

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
    ps = sh_pv[warp * Hd + glane * R + dr].store(warp_acc, lane < NKQ).end(dr)
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
    weight = _fexp(sh_mx.after(barrier)[ws] - global_max)
    wr = UOp.range(R, 17)
    pv_up = block_pv[wr].store(block_pv.after(ws)[wr] + weight * sh_pv.after(barrier)[ws * Hd + glane * R + wr]).end(wr)
    den_up = block_den.after(pv_up)[0].store(block_den.after(ws)[0] + weight * sh_den.after(barrier)[ws]).end(ws)
    final_pv, final_den = block_pv.after(den_up), block_den.after(den_up)

    base = (head * S + split) * W
    output_axis = UOp.range(R, 18)
    output_dim = glane * R + output_axis
    pv = pout[base + output_dim].store(final_pv[output_axis]).end(output_axis)
    ls = pout.after(pv)[base + Hd].store(final_den[0], lane.eq(0))
    ms = pout.after(ls)[base + (Hd + 1)].store(global_max, lane.eq(0))
    return ms.end(head, split, lane, warp).sink(arg=_kernel_info(
      f"flash_vec_llama_score_pv_{Hq}_{Hd}_{S}", coalesced_loads=True))
  return kernel


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
  target: str = "amd_gfx1100"

  def validate(self) -> None:
    if min(self.Hq, self.Hd, self.Hkv, self.MAXC) <= 0: raise ValueError("Hq, Hd, Hkv and MAXC must be positive")
    if self.staging != "KV_BOTH": raise ValueError(f"production flash decode requires staging='KV_BOTH', got {self.staging!r}")
    if self.token_block != 16: raise ValueError(f"token_block must currently be 16, got {self.token_block}")
    if self.Hq % self.Hkv != 0: raise ValueError(f"Hq must be divisible by Hkv, got Hq={self.Hq} Hkv={self.Hkv}")
    if self.query_group_size is not None and not 1 <= self.query_group_size <= self.Hq // self.Hkv:
      raise ValueError(f"query_group_size must be in 1..{self.Hq // self.Hkv}, got {self.query_group_size}")
    if self.stage_width not in (1, 2, 4, 8): raise ValueError(f"stage_width must be one of 1,2,4,8, got {self.stage_width}")
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
    suffix = "" if self.query_group_size is None else f"_qg{self.query_group_size}"
    return f"flash_block_tiled_xlane_score_pv_tile_whole_cache_{self.Hq}_{self.Hd}{suffix}"

  def emit(self, Tc:UOp):
    self.validate()
    return flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel(
      self.Hd, self.Hq, self.Hkv, self.MAXC, self.geometry.aligned_per_split_length(Tc), self.split_count, Tc,
      staging=self.staging, quant=self.quant, rope=self.rope, query_group_size=self.query_group_size,
      stage_width=self.stage_width)

  def to_json(self) -> dict[str, Any]:
    return {key:getattr(self, key) for key in ("Hq", "Hd", "Hkv", "MAXC", "split_count", "staging", "quant", "rope",
                                                 "token_block", "query_group_size", "stage_width", "target")}


@dataclass(frozen=True)
class FlashCombineSpec:
  Hd: int
  Hq: int
  split_count: int
  stride: int|None = None
  output_fp16: bool = False

  def validate(self) -> None:
    if min(self.Hd, self.Hq, self.split_count) <= 0: raise ValueError("Hd, Hq and split_count must be positive")
    if self.stride is not None and self.stride < 1: raise ValueError(f"stride must be >= 1, got {self.stride}")

  @property
  def kernel_name(self) -> str:
    prefix = "flash_fused_gmax_combine_f16" if self.output_fp16 else "flash_fused_gmax_combine"
    return f"{prefix}_{self.Hq}_{self.Hd}"
  def emit(self): self.validate(); return flash_fused_gmax_combine_kernel(self.Hd, self.Hq, self.split_count, self.stride, self.output_fp16)


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
                                    stage_width:int=1, combine_fp16:bool=False) -> FlashDecodeAttentionSpec:
  return FlashDecodeAttentionSpec(
    FlashDecodeTileSpec(Hq, Hd, Hkv, MAXC, S, staging, quant, rope, query_group_size=query_group_size,
                        stage_width=stage_width),
    FlashCombineSpec(Hd, Hq, S, combine_stride, output_fp16=combine_fp16) if fused_combine else None)


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


def flash_decode_live_split_block_tile(q:Tensor, cache_kv:Tensor, Tc:UOp, Hd:int, Hq:int, Hkv:int, MAXC:int, S:int,
                                       staging:str="KV_BOTH", fused_combine:bool=True, kv_scale:Tensor|None=None,
                                       freqs:Tensor|None=None, query_group_size:int|None=None, stage_width:int=1,
                                       combine_fp16:bool=False) -> Tensor:
  """Execute the selected live-split flash decode and return ``[Hq, Hd]``."""
  if not fused_combine: raise ValueError("fused_combine=False is no longer supported for decode live-split routes")
  # TG7: this is a pure shape-based route SELECTION (which of G4/G5 matches, to label the emitted
  # KernelProgram) -- capability and promotion were already decided by the caller's bind() before this
  # function is ever invoked (decode_routes.flash_decode_attention_route only calls here once binding
  # succeeded), so re-deriving them from `device` here would be both redundant and -- inside a Tensor Function
  # dispatch, which is exactly where this runs -- unable to open a device at all (see
  # decode_routes._flash_decode_capability_and_target_for_device's docstring).
  route = next((row for row in (FLASH_DECODE_G4, FLASH_DECODE_G5) if row.shape_ok(1, Hq, Hkv, Hd)), None)
  if route is None or (route.split_size, route.query_group_size, route.stage_width, route.staging) != \
      (S, query_group_size, stage_width, staging):
    raise ValueError("flash decode geometry is not an admitted promoted route")
  quant, rope = kv_scale is not None, freqs is not None
  inputs = (q.reshape(Hq * Hd), cache_kv) + ((kv_scale,) if quant else ()) + ((freqs,) if rope else ())
  spec = describe_flash_decode_attention(Hq, Hd, Hkv, MAXC, S, staging=staging, quant=quant, rope=rope,
                                         query_group_size=query_group_size, stage_width=stage_width,
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
