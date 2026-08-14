"""Closed-default, one-group mixed-Q4/Q6 shared-Q8 decode boundary.

This is deliberately a *bounded* integration boundary, rather than a new
per-linear mode.  An admission object is installed on one TransformerBlock by
the qualification harness; no load policy or environment variable can enable
it.  The call checks the complete Q/K/V triple again at trace time and returns
``None`` on every mismatch, leaving the ordinary primitive calls untouched.
"""
from __future__ import annotations

from dataclasses import dataclass

from tinygrad import Tensor, dtypes
from tinygrad.codegen.late.int8_dot import int8x4_dot
from tinygrad.codegen.late.warp_reduce import _staged_shfl as _warp_shfl, _warp_reduce_sum_staged, warp_reduce_max
from tinygrad.dtype import AddrSpace
from tinygrad.llm.decode_kernels import (Q4K_WORDS_PER_BLOCK, Q6K_HALFWORDS_PER_BLOCK, _f16_half, _f16_word, _i8,
  _q6k_byte, _q4k_group_params, _staged_shfl)
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec, execute_promoted_program
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, ReduceOutputSpec, UOp

_K, _Q_ROWS, _KV_ROWS = 4096, 4096, 1024
_Q8_PACKS, _Q8_GROUPS = _K//4, _K//32

@dataclass(frozen=True)
class SharedQ8AttentionAdmission:
  """A deliberately non-promoted qualification lease for one exact block.

  The model loader never constructs this.  Its explicit presence gives a
  qualification harness a structural, inspectable scope and makes an
  accidental all-layer rollout impossible.
  """
  block_index: int
  target: tuple[str, str] = ("NV", "sm_120")
  cooperative_q4: bool = False
  # Kept separate from the Q4 cooperative lease: this selects only the Q6 V
  # direct-output consumer in a real Q4/Q4/Q6 attention group.  It is an
  # explicit qualification lease, never a model-load policy.
  q6_direct_output: bool = False

  def __post_init__(self):
    if not isinstance(self.block_index, int) or self.block_index < 0: raise ValueError("block_index must be non-negative")
    if self.target != ("NV", "sm_120"): raise ValueError("only the isolated NV sm_120 qualification target is supported")
    if not isinstance(self.cooperative_q4, bool): raise ValueError("cooperative_q4 must be bool")
    if not isinstance(self.q6_direct_output, bool): raise ValueError("q6_direct_output must be bool")

def _pack4(vs):
  r=UOp.const(dtypes.uint32,0)
  for i,v in enumerate(vs): r=r.bitwise_or(v.cast(dtypes.uint8).cast(dtypes.uint32).lshift(8*i))
  return r

def _i8lane(p, lane): return p.rshift(lane*8).bitwise_and(255).cast(dtypes.uint8).bitcast(dtypes.int8).cast(dtypes.int32)

def _q8_d(packed, group):
  return packed[_Q8_PACKS+group].bitwise_and(0xffff).cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)

def _q8_s(packed, group):
  return packed[_Q8_PACKS+group].rshift(16).cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)

def _emit_q8_provider():
  """One-program llama-CUDA Q8_1 provider: 1024 int8x4 packets + 128 d|s half2 metadata.

  CUDA's live MMQ provider stores ``d`` and the raw fp32 input sum as fp16
  (`make_half2(d, sum)` in ggml-cuda/quantize.cu). This differs from the CPU
  model-file reference's `sum(qs)*d`; decode parity must match the live CUDA
  construction. Computing both here avoids the four-program Tensor graph.
  """
  def kernel(out, x):
    group=UOp.range(_Q8_GROUPS,0)
    # The provider accepts the promised producer view directly. Baseline Q4/Q6
    # consumers round fp32 norm output to fp16 at their prelude; do that same
    # round in-kernel so no cast/contiguous adapter is needed at the function
    # boundary.
    rounded=[x[group*32+i].cast(dtypes.float16).cast(dtypes.float32) for i in range(32)]
    amax=UOp.const(dtypes.float32,0.)
    for value in rounded: amax=amax.maximum(value.abs())
    d=amax/UOp.const(dtypes.float32,127.0)
    inv=d.eq(0).where(UOp.const(dtypes.float32,0.),d.reciprocal())
    qstores=[]
    for pack in range(8):
      qs=[]
      for i in range(4):
        q=(rounded[pack*4+i]*inv).round().maximum(
          UOp.const(dtypes.float32,-128.)).minimum(UOp.const(dtypes.float32,127.)).cast(dtypes.int8)
        qs.append(q)
      qstores.append(out[group*8+pack].store(_pack4(qs)))
    # Match CUDA's warp_reduce_sum<QK8_1> association exactly: shfl-down
    # offsets 16,8,4,2,1. The fp16 ``s`` can otherwise move by one ULP even
    # though the mathematical sum is unchanged.
    sums=rounded
    for off in (16,8,4,2,1): sums=[sums[i]+sums[i+off] for i in range(off)]
    xsum=sums[0]
    dh=d.cast(dtypes.float16).bitcast(dtypes.uint16).cast(dtypes.uint32)
    sh=xsum.cast(dtypes.float16).bitcast(dtypes.uint16).cast(dtypes.uint32)
    mstore=out[_Q8_PACKS+group].store(dh.bitwise_or(sh.lshift(16)))
    return UOp.group(*qstores,mstore).end(group).sink(arg=KernelInfo(name="q8_1_llama_provider_4096",opts_to_apply=()))
  return kernel

def _emit_rmsnorm_q8_provider(spec:ReduceOutputSpec, x_dtype, weight_dtype):
  """One-block RMSNorm -> llama-CUDA Q8_1 provider.

  The first phase is the already-qualified 16-warp reduction association used
  by ``reduce_output_rmsnorm_1_4096``.  After its one workgroup barrier, each
  warp owns eight consecutive Q8 groups.  Keeping one lane per activation
  preserves llama CUDA's 32-lane max/sum association; four adjacent quantized
  lanes are gathered only for the final int8x4 store.
  """
  if not (spec.rows == 1 and spec.dim == _K and spec.recipe == "sumsq_rsqrt_affine" and spec.affine):
    raise ValueError("fused RMSNorm/Q8 requires the exact 1x4096 affine recipe")
  if x_dtype not in (dtypes.float16, dtypes.float32) or weight_dtype != dtypes.float16:
    raise ValueError("fused RMSNorm/Q8 requires fp16/fp32 x and a materialized fp16 weight")
  def kernel(out, x, weight):
    lane=UOp.range(32,0,axis_type=AxisType.LOCAL)
    warp=UOp.range(16,1,axis_type=AxisType.LOCAL)
    red=UOp.range(8,2,axis_type=AxisType.REDUCE)
    base=warp*256+lane+red*32
    xv=x[base].cast(dtypes.float32)
    acc=UOp.placeholder((1,),dtypes.float32,20,addrspace=AddrSpace.REG)
    acc=acc.after(acc[0].store(0.0))
    acc=acc.after(acc[0].store(acc.after(red)[0]+xv*xv).end(red))
    warp_total=_warp_reduce_sum_staged(acc[0],lane,32,slot_base=90)
    smem=UOp.placeholder((16,),dtypes.float32,230,addrspace=AddrSpace.LOCAL)
    published=smem[warp].store(warp_total,lane.eq(0))
    ready=UOp.barrier(UOp.group(published))
    total=UOp.const(dtypes.float32,0.0)
    for wi in range(16): total=total+smem.after(ready)[wi]
    scale=(total/UOp.const(dtypes.float32,float(_K))+UOp.const(dtypes.float32,spec.eps)).sqrt().reciprocal()

    group_loop=UOp.range(8,3,axis_type=AxisType.LOOP)
    group=warp*8+group_loop
    idx=group*32+lane
    # This is the ordinary attention RMSNorm epilogue followed by the legacy
    # projection prelude's fp16 cast.  The widened value is precisely what the
    # standalone llama-Q8 provider receives in the proved construction.
    scaled=x[idx].cast(dtypes.float32)*scale
    # Mirror emit_reduce_output_rmsnorm's input-dtype rounding before the
    # affine multiply, then mirror the ordinary Q8 provider's fp16 prelude.
    # The production residual is fp32, but keeping the admitted fp16 input
    # contract exact prevents this closed route from having a latent dtype
    # widening bug.
    normed=scaled.cast(x_dtype)
    affine=(normed*weight[idx].cast(x_dtype)).cast(spec.out_dtype)
    rounded=affine.cast(dtypes.float16).cast(dtypes.float32)
    amax=warp_reduce_max(rounded.abs(),lane,32,slot_base=100)
    d=amax/UOp.const(dtypes.float32,127.0)
    inv=d.eq(0).where(UOp.const(dtypes.float32,0.0),d.reciprocal())
    qi=(rounded*inv).round().maximum(UOp.const(dtypes.float32,-128.0)).minimum(
      UOp.const(dtypes.float32,127.0)).cast(dtypes.int8).cast(dtypes.int32)
    q1=_warp_shfl(qi,1,lane,110)
    q2=_warp_shfl(qi,2,lane,111)
    q3=_warp_shfl(qi,3,lane,112)
    qstore=out[group*8+lane//4].store(_pack4((qi,q1,q2,q3)),lane.bitwise_and(3).eq(0))
    xsum=_warp_reduce_sum_staged(rounded,lane,32,slot_base=120)
    dh=d.cast(dtypes.float16).bitcast(dtypes.uint16).cast(dtypes.uint32)
    sh=xsum.cast(dtypes.float16).bitcast(dtypes.uint16).cast(dtypes.uint32)
    # Keep lane ownership explicit in both the address and STORE gate. GPU-dim
    # lowering adds an implicit lane-zero Invalid index when a global-store
    # address omits a local axis; combining that with an existing explicit
    # gate leaves an unrenderable Invalid. The inactive address is arbitrary
    # but in bounds, while lane zero retains the exact Q8_1 metadata address.
    lane0=lane.eq(0)
    metadata_idx=lane0.where(_Q8_PACKS+group,UOp.const(dtypes.weakint,0))
    mstore=out[metadata_idx].store(dh.bitwise_or(sh.lshift(16)),lane0)
    return UOp.group(qstore,mstore).end(lane,warp,group_loop).sink(
      arg=KernelInfo(name="rmsnorm_q8_1_llama_provider_4096",opts_to_apply=()))
  return kernel

def _reduce_output_rmsnorm_marker(x:Tensor) -> tuple[UOp, ReduceOutputSpec]|None:
  marker=x.uop
  while marker.op in (Ops.MEMORY_SEMANTIC, Ops.RESHAPE): marker=marker.src[0]
  if marker.op is not Ops.REDUCE_OUTPUT or not isinstance(marker.arg,ReduceOutputSpec): return None
  if not (marker.arg.rows == 1 and marker.arg.dim == _K and marker.arg.recipe == "sumsq_rsqrt_affine" and marker.arg.affine): return None
  return marker,marker.arg

def _q6signed(h,b,g,p):
  half,pg=g//8,g%8
  lo=_q6k_byte(h,b,half*64+(pg%4)*16+p).rshift(4 if pg>=4 else 0).bitwise_and(15)
  hi=_q6k_byte(h,b,128+half*32+(pg%2)*16+p).rshift((pg//2)*2).bitwise_and(3).lshift(4)
  return lo.bitwise_or(hi).cast(dtypes.int32)-32

def _q6signed_dynamic_group(h,b,g,p):
  """Select one fixed Q6_K layout without an unsupported dynamic shift."""
  ret=UOp.const(dtypes.int32,0)
  for static_group in range(16): ret=g.eq(static_group).where(_q6signed(h,b,static_group,p),ret)
  return ret

def _emit_q4(rows, rt=2):
  def kernel(out,w,xp):
    ro,ri=UOp.range(rows//rt,0),UOp.range(rt,1,axis_type=AxisType.LOCAL)
    p4,b=UOp.range(8,2,axis_type=AxisType.LOCAL),UOp.range(_K//256,3,axis_type=AxisType.REDUCE)
    row=ro*rt+ri; base=(row*(_K//256)+b)*Q4K_WORDS_PER_BLOCK; c=UOp.const(dtypes.float32,0.)
    for g in range(8):
      d,dm,sc,mn=_q4k_group_params(w,base,g); qw=w[base+4+(g//2)*8+p4].rshift((g%2)*4).bitwise_and(0x0F0F0F0F)
      xv=xp[b*64+g*8+p4]; dot=int8x4_dot(UOp.const(dtypes.int32,0),qw,xv).cast(dtypes.float32)
      sx=_i8lane(xv,0)+_i8lane(xv,1)+_i8lane(xv,2)+_i8lane(xv,3)
      # llama's decode MMVQ path intentionally ignores Q8_1.s and computes the
      # Q4 minimum correction from the int8 lane sum times fp16 d (see
      # vec_dot_q4_K_q8_1_impl_vmmq). Each p4 lane owns four values; the staged
      # reduction combines all eight lane sums exactly once.
      c=c+_q8_d(xp,b*8+g)*(d*sc.cast(dtypes.float32)*dot-dm*mn.cast(dtypes.float32)*sx.cast(dtypes.float32))
    a=UOp.placeholder((1,),dtypes.float32,20,addrspace=AddrSpace.REG); a=a.after(a[0].store(0.)); a=a.after(a[0].store(a.after(b)[0]+c).end(b)); t=a[0]
    for slot,off in enumerate((4,2,1),90): t=t+_staged_shfl(t,off*rt,p4,slot)
    # Keep the isolated-PASS kernel identity byte-for-byte. Route ownership is
    # carried by KernelProgram, not by perturbing generated source metadata.
    return out[row].store(t).end(ro,ri,p4).sink(arg=KernelInfo(name=f"q4k_q8_dp4a_{rows}_{_K}",opts_to_apply=()))
  return kernel

def _emit_q4_cooperative(rows, block_count:UOp):
  """Closed-lease four-warp Q4/Q8 consumer; four partials require an external sum."""
  if rows not in (_Q_ROWS,_KV_ROWS): raise ValueError("cooperative Q4 requires a production attention shape")
  def kernel(out,w,xp):
    row,lid=UOp.special(rows,"gidx0"),UOp.special(128,"lidx0")
    warp,lane=lid//32,lid%32; group,word_base=lane//4,(lane%4)*2
    br=UOp.range(block_count,2,axis_type=AxisType.LOOP); block=warp*4+br
    base=(row*(_K//256)+block)*Q4K_WORDS_PER_BLOCK
    w0,w1,w2,w3=w[base],w[base+1],w[base+2],w[base+3]
    d,dm=_f16_word(w0,False),_f16_word(w0,True); g4=group%4
    b1=w1.rshift(g4*8).bitwise_and(0xff); b2=w2.rshift(g4*8).bitwise_and(0xff); hb=w3.rshift(g4*8).bitwise_and(0xff)
    sc=(group<4).where(b1.bitwise_and(63),hb.bitwise_and(0xf).bitwise_or(b1.rshift(6).lshift(4)))
    mn=(group<4).where(b2.bitwise_and(63),hb.rshift(4).bitwise_or(b2.rshift(6).lshift(4)))
    c=UOp.const(dtypes.float32,0.)
    for ws in range(2):
      word=word_base+ws; qw=w[base+4+(group//2)*8+word].rshift((group%2)*4).bitwise_and(0x0F0F0F0F)
      xv=xp[block*64+group*8+word]; dot=int8x4_dot(UOp.const(dtypes.int32,0),qw,xv).cast(dtypes.float32)
      sx=_i8lane(xv,0)+_i8lane(xv,1)+_i8lane(xv,2)+_i8lane(xv,3)
      c=c+_q8_d(xp,block*8+group)*(d*sc.cast(dtypes.float32)*dot-dm*mn.cast(dtypes.float32)*sx.cast(dtypes.float32))
    a=UOp.placeholder((1,),dtypes.float32,20,addrspace=AddrSpace.REG); a=a.after(a[0].store(0.)); a=a.after(a[0].store(a.after(br)[0]+c).end(br)); t=a[0]
    for slot,off in enumerate((16,8,4,2,1),90): t=t+_staged_shfl(t,off,lane,slot)
    return out[row,warp].store(t,lane.eq(0)).sink(
      arg=KernelInfo(name=f"q4k_warp_coop_q8_dp4a_partial_{rows}_{_K}",opts_to_apply=()))
  return kernel

def _emit_q6(rows, rt=2):
  def kernel(out,h,xp):
    ro,ri=UOp.range(rows//rt,0),UOp.range(rt,1,axis_type=AxisType.LOCAL)
    p4,b=UOp.range(4,2,axis_type=AxisType.LOCAL),UOp.range(_K//256,3,axis_type=AxisType.REDUCE)
    row=ro*rt+ri; base=(row*(_K//256)+b)*Q6K_HALFWORDS_PER_BLOCK; c=UOp.const(dtypes.float32,0.)
    for g in range(16):
      qw=_pack4([_q6signed(h,base,g,p4*4+i) for i in range(4)])
      dot=int8x4_dot(UOp.const(dtypes.int32,0),qw,xp[b*64+g*4+p4]).cast(dtypes.float32)
      c=c+dot*_f16_half(h[base+104])*_i8(_q6k_byte(h,base,192+g))*_q8_d(xp,b*8+g//2)
    a=UOp.placeholder((1,),dtypes.float32,20,addrspace=AddrSpace.REG); a=a.after(a[0].store(0.)); a=a.after(a[0].store(a.after(b)[0]+c).end(b)); t=a[0]
    for slot,off in enumerate((2,1),90): t=t+_staged_shfl(t,off*rt,p4,slot)
    return out[row].store(t).end(ro,ri,p4).sink(arg=KernelInfo(name=f"q6k_q8_dp4a_{rows}_{_K}",opts_to_apply=()))
  return kernel

def _emit_q6_warp_direct(rows):
  """Q6/Q8 llama-MMVQ direct-output consumer for the shared packed-Q8 ABI.

  One row per CTA and 32 lanes x 4 warps, matching llama's
  ``vec_dot_q6_K_q8_1_impl_mmvq``: each lane loads two packed uint32 words
  from ql/qh, produces two signed int8x4 values with bit ops, then dp4a's them
  against the shared Q8_1 packets.  Blocks are strided across warps
  (``block = warp + 4*block_rel``), so each warp owns one Q6 block per loop.
  """
  if rows != _KV_ROWS: raise ValueError("Q6 direct shared-Q8 consumer requires the exact 1024x4096 V shape")
  def kernel(out,h,xp):
    row,lid=UOp.special(rows,"gidx0"),UOp.special(128,"lidx0")
    warp,lane=lid//32,lid%32
    block_rel=UOp.range(4,0,axis_type=AxisType.REDUCE); block=warp*4+block_rel
    base=(row*(_K//256)+block)*Q6K_HALFWORDS_PER_BLOCK
    contrib=UOp.const(dtypes.float32,0.)
    # get_int_b2(bq6_K->ql, lane): two halfwords, low halfword first.
    vl=h[base+lane*2].cast(dtypes.uint32).bitwise_or(h[base+lane*2+1].cast(dtypes.uint32).lshift(16))
    # get_int_b2(bq6_K->qh, 8*(lane//16)+lane%8) >> 2*((lane%16)//8).
    qh_half=16*(lane//16)+2*(lane%8)
    vh=h[base+64+qh_half].cast(dtypes.uint32).bitwise_or(
      h[base+65+qh_half].cast(dtypes.uint32).lshift(16))
    vh=vh.rshift(2*((lane%16)//8))
    scale_idx=8*(lane//16)+(lane%16)//4
    q8_group0=4*(lane//16)+(lane%16)//8
    for i in range(2):
      low=vl.rshift(4*i).bitwise_and(0x0F0F0F0F)
      high=vh.rshift(4*i).lshift(4).bitwise_and(0x30303030)
      qword=low.bitwise_or(high)
      q8_group=q8_group0+2*i
      xword=xp[block*64+q8_group*8+lane%8]
      scale=_q6k_byte(h,base,192+scale_idx+4*i).cast(dtypes.uint8).bitcast(dtypes.int8).cast(dtypes.int32)
      dot=int8x4_dot(UOp.const(dtypes.int32,0),qword,xword)
      xsum=int8x4_dot(UOp.const(dtypes.int32,0),UOp.const(dtypes.uint32,0x01010101),xword)
      # (qword - 32) dot xword is algebraically llama's __vsubss4(qword, 0x20).
      contrib=contrib+(dot-UOp.const(dtypes.int32,32)*xsum).cast(dtypes.float32)*scale.cast(dtypes.float32)*_q8_d(xp,block*8+q8_group)
    contrib=contrib*_f16_half(h[base+104])
    acc=UOp.placeholder((1,),dtypes.float32,20,addrspace=AddrSpace.REG)
    acc=acc.after(acc[0].store(0.)); acc=acc.after(acc[0].store(acc.after(block_rel)[0]+contrib).end(block_rel))
    total=acc[0]
    for slot,off in enumerate((16,8,4,2,1),90): total=total+_staged_shfl(total,off,lane,slot)
    smem=UOp.placeholder((4,),dtypes.float32,230,addrspace=AddrSpace.LOCAL)
    published=smem[warp].store(total,lane.eq(0)); ready=UOp.barrier(UOp.group(published))
    merged=UOp.const(dtypes.float32,0.)
    for wi in range(4): merged=merged+smem.after(ready)[wi]
    return out[row].store(merged,lid.eq(0)).sink(arg=KernelInfo(name=f"q6k_q8_warp_direct_{rows}_{_K}",opts_to_apply=()))
  return kernel

def shared_q8_attention_call(admission, q_linear, k_linear, v_linear, x:Tensor, start_pos:int|UOp=0,
                             norm_weight:Tensor|None=None):
  """Return Q/K/V projections for an admitted real Qwen Q4/Q4/{Q4,Q6} group, else ``None``."""
  if not isinstance(admission, SharedQ8AttentionAdmission): return None
  from tinygrad.llm.qk_primitives import Q4KPrimitiveLinear, Q6KPrimitiveLinear
  if not isinstance(q_linear, Q4KPrimitiveLinear) or not isinstance(k_linear, Q4KPrimitiveLinear): return None
  if not isinstance(v_linear, (Q4KPrimitiveLinear, Q6KPrimitiveLinear)): return None
  if tuple(x.shape) != (1,1,_K) or (q_linear.in_features, k_linear.in_features, v_linear.in_features) != (_K,_K,_K): return None
  if (q_linear.out_features, k_linear.out_features, v_linear.out_features) != (_Q_ROWS,_KV_ROWS,_KV_ROWS): return None
  # Preserve a real four-iteration runtime loop without adding a scalar
  # token-time provider. start_pos is an existing nonnegative graph variable,
  # so n//(n+1) is exactly zero. Unlike a literal four this survives NVRTC
  # without body cloning, and unlike a scalar buffer LOAD its estimate is
  # resolvable from the graph's existing var_vals during replay.
  graph_pos=None
  if isinstance(start_pos,UOp):
    if start_pos.op is Ops.DEFINE_VAR: graph_pos=start_pos
    elif (start_pos.op is Ops.BIND and len(start_pos.src)==2 and start_pos.src[0].op is Ops.DEFINE_VAR and
          start_pos.src[1].op is Ops.CONST): graph_pos=start_pos.src[0]
  if graph_pos is not None and graph_pos.arg[1] < 0: graph_pos=None
  cooperative_blocks=graph_pos//(graph_pos+1)+4 if graph_pos is not None else None
  if admission.cooperative_q4 and cooperative_blocks is None: return None
  if not all(getattr(linear, "decode_enabled", False) and getattr(getattr(linear, "route_admission", None), "admitted", False)
                 for linear in (q_linear, k_linear, v_linear)): return None
  marker=_reduce_output_rmsnorm_marker(x)
  if marker is not None and norm_weight is not None and tuple(norm_weight.shape) == (_K,) and norm_weight.dtype == dtypes.float16:
    marker_uop,spec=marker
    route_kind="q4q4q4" if isinstance(v_linear,Q4KPrimitiveLinear) else "q4q4q6"
    provider=KernelProgram("decode_shared_q8_attention",f"{route_kind}.blk{admission.block_index}.rmsnorm_provider",
      KernelProgramProvenance.MACHINE_SEARCH_GENERATED,_emit_rmsnorm_q8_provider(spec,marker_uop.src[1].dtype,norm_weight.dtype),
      output_spec=OutputSpec((_Q8_PACKS+_Q8_GROUPS,),dtypes.uint32))
    xp=execute_promoted_program(None,Tensor(marker_uop.src[1]).reshape(_K),norm_weight,program=provider)
  else:
    route_kind="q4q4q4" if isinstance(v_linear,Q4KPrimitiveLinear) else "q4q4q6"
    provider=KernelProgram("decode_shared_q8_attention",f"{route_kind}.blk{admission.block_index}.provider",
      KernelProgramProvenance.MACHINE_SEARCH_GENERATED,_emit_q8_provider(),output_spec=OutputSpec((_Q8_PACKS+_Q8_GROUPS,),dtypes.uint32))
    xp=execute_promoted_program(None,x[:,0,:].reshape(_K),program=provider)
  def run(linear, rows, emitter, storage):
    cooperative=admission.cooperative_q4 and emitter is _emit_q4
    q6_direct=admission.q6_direct_output and emitter is _emit_q6
    emitted=(lambda rows:_emit_q4_cooperative(rows,cooperative_blocks)) if cooperative else (_emit_q6_warp_direct if q6_direct else emitter)
    shape=(rows,4) if cooperative else (rows,)
    suffix='.coop' if cooperative else '.q6_direct' if q6_direct else ''
    program=KernelProgram("decode_shared_q8_attention", f"{route_kind}.blk{admission.block_index}.{rows}{suffix}",
      KernelProgramProvenance.MACHINE_SEARCH_GENERATED, emitted(rows), output_spec=OutputSpec(shape,dtypes.float32))
    ret=execute_promoted_program(None,storage.to(x.device),xp,program=program)
    if cooperative: ret=ret.sum(axis=1).contiguous()
    return ret.reshape(1,1,rows)
  v_emitter,v_storage=(_emit_q4,v_linear.q4k_storage.words) if isinstance(v_linear,Q4KPrimitiveLinear) else (_emit_q6,v_linear.q6k_storage.halfs)
  return run(q_linear,_Q_ROWS,_emit_q4,q_linear.q4k_storage.words), run(k_linear,_KV_ROWS,_emit_q4,k_linear.q4k_storage.words), run(v_linear,_KV_ROWS,v_emitter,v_storage)

__all__ = ["SharedQ8AttentionAdmission", "shared_q8_attention_call"]
