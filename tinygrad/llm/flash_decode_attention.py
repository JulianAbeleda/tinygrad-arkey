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
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, execute_promoted_program

_LOG2E = 1.4426950408889634
_F32 = dtypes.float32


def _fexp(x:UOp) -> UOp:
  arg = x * _LOG2E
  if getenv("DECODE_FAST_EXP2", 0):
    return UOp(Ops.CUSTOMI, arg.dtype, (arg,), arg="__builtin_amdgcn_exp2f({0})")
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
    val = cache[which, 0, kvh, tok, elem].cast(dtypes.half)
    if quant: val = val * kvscale[which, 0, kvh, tok].cast(dtypes.half)
    return val

  def load(which, kvh, tok, elem):
    val = raw(which, kvh, tok, elem)
    if rope and which == 0:
      pos, rotary_elem, low = pos_of(tok), elem % half_dim, elem < half_dim
      pair = raw(0, kvh, tok, low.where(elem + half_dim, elem - half_dim))
      cos = freqs[pos, rotary_elem].cast(dtypes.half)
      sin = freqs[pos, half_dim + rotary_elem].cast(dtypes.half)
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
    ksh = UOp.placeholder((TK * Hd,), dtypes.half, 230, addrspace=AddrSpace.LOCAL)
    vsh = UOp.placeholder((TK * Hd,), dtypes.half, 231, addrspace=AddrSpace.LOCAL) if staging == "KV_BOTH" else None
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
      qpair = UOp(Ops.STACK, dtypes.half.vec(2), (q[head * Hd + elem].cast(dtypes.half), q[head * Hd + elem + 1].cast(dtypes.half)))
      kpair = UOp(Ops.STACK, dtypes.half.vec(2), (ksh.after(barrier)[token_in_tile * Hd + elem],
                                                 ksh.after(barrier)[token_in_tile * Hd + elem + 1]))
      fdot = UOp(Ops.CUSTOMI, _F32, (dot.after(pair_axis)[0], qpair, kpair),
                 arg="__builtin_amdgcn_fdot2({1}, {2}, {0}, false)")
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


def flash_fused_gmax_combine_kernel(Hd:int, Hq:int, S:int, stride:int|None=None):
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
    return out[head * Hd + output_dim].store(final_acc[output_axis] / final_den).end(output_axis).end(head, lane).sink(
      arg=KernelInfo(name=f"flash_fused_gmax_combine_{Hq}_{Hd}", opts_to_apply=()))
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

  def validate(self) -> None:
    if min(self.Hd, self.Hq, self.split_count) <= 0: raise ValueError("Hd, Hq and split_count must be positive")
    if self.stride is not None and self.stride < 1: raise ValueError(f"stride must be >= 1, got {self.stride}")

  @property
  def kernel_name(self) -> str: return f"flash_fused_gmax_combine_{self.Hq}_{self.Hd}"
  def emit(self): self.validate(); return flash_fused_gmax_combine_kernel(self.Hd, self.Hq, self.split_count, self.stride)


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
                                    stage_width:int=1) -> FlashDecodeAttentionSpec:
  return FlashDecodeAttentionSpec(
    FlashDecodeTileSpec(Hq, Hd, Hkv, MAXC, S, staging, quant, rope, query_group_size=query_group_size,
                        stage_width=stage_width),
    FlashCombineSpec(Hd, Hq, S, combine_stride) if fused_combine else None)


def emit_flash_decode_tile(spec:FlashDecodeAttentionSpec, Tc:UOp): return spec.emit_tile(Tc)
def emit_flash_decode_combine(spec:FlashDecodeAttentionSpec): return spec.emit_combine()


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

  def supports(self, B:int, Hq:int, Hkv:int, Hd:int, device:str) -> bool:
    return (device == "AMD" or device.startswith("AMD:")) and (B, Hq, Hkv, Hd) == (1, self.query_heads, self.kv_heads, self.head_dim)


FLASH_DECODE_G4 = FlashDecodeRouteConfig("attention_decode.flash_live_split", "decode_flash_live_split_g4_kvboth",
                                          32, 48, None, 1)
FLASH_DECODE_G5 = FlashDecodeRouteConfig("attention_decode.flash_live_split_g5", "decode_flash_live_split_g5_kvboth",
                                          40, 32, 2, 4)


def flash_decode_live_split_block_tile(q:Tensor, cache_kv:Tensor, Tc:UOp, Hd:int, Hq:int, Hkv:int, MAXC:int, S:int,
                                       staging:str="KV_BOTH", fused_combine:bool=True, kv_scale:Tensor|None=None,
                                       freqs:Tensor|None=None, query_group_size:int|None=None, stage_width:int=1) -> Tensor:
  """Execute the selected live-split flash decode and return ``[Hq, Hd]``."""
  if not fused_combine: raise ValueError("fused_combine=False is no longer supported for decode live-split routes")
  route = next((row for row in (FLASH_DECODE_G4, FLASH_DECODE_G5) if row.supports(1, Hq, Hkv, Hd, str(q.device))), None)
  if route is None or (route.split_size, route.query_group_size, route.stage_width, route.staging) != \
      (S, query_group_size, stage_width, staging):
    raise ValueError("flash decode geometry is not an admitted promoted route")
  quant, rope = kv_scale is not None, freqs is not None
  inputs = (q.reshape(Hq * Hd), cache_kv) + ((kv_scale,) if quant else ()) + ((freqs,) if rope else ())
  spec = describe_flash_decode_attention(Hq, Hd, Hkv, MAXC, S, staging=staging, quant=quant, rope=rope,
                                         query_group_size=query_group_size, stage_width=stage_width)
  tile_program = KernelProgram(route.route_id, f"{route.candidate_id}.tile",
    KernelProgramProvenance.MACHINE_SEARCH_GENERATED, spec.emit_tile(Tc))
  partial = execute_promoted_program(Tensor.empty(Hq * S * (Hd + 2), dtype=dtypes.float32, device=q.device),
    *inputs, program=tile_program)
  combine_program = KernelProgram(route.route_id, f"{route.candidate_id}.combine",
    KernelProgramProvenance.MACHINE_SEARCH_GENERATED, spec.emit_combine())
  out = execute_promoted_program(Tensor.empty(Hq * Hd, dtype=dtypes.float32, device=q.device), partial, program=combine_program)
  return out.reshape(Hq, Hd)
