#!/usr/bin/env python3
"""Exact-shape, included-cost Q4/Q4/Q6 attention-group shared-Q8 gate.

Research only.  The candidate has precisely one Q8 producer, two Q4 consumers
and one Q6 consumer; it is not a per-linear cache or a production route.
"""
from __future__ import annotations
import argparse, json, statistics, time
import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.codegen.late.int8_dot import int8x4_dot
from tinygrad.dtype import AddrSpace
from tinygrad.llm.decode_kernels import (_f16_half, _i8, _q6k_byte, _staged_shfl, _q4k_group_params,
  Q6K_HALFWORDS_PER_BLOCK, Q4K_WORDS_PER_BLOCK, emit_q6k_gemv_kernel, q6k_spec_for_role, q4k_g3_lanemap_gemv_kernel)
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, execute_research_program
from tinygrad.uop.ops import AxisType, KernelInfo, UOp
from extra.llm_research.decode.route_class_numerics import _make_q4k_words, _make_q6k_halfs

K=4096
def prog(name, fn): return KernelProgram("research.q4q4q6_shared_q8", name, KernelProgramProvenance.RESEARCH_ONLY, fn)
def pack4(vs):
  r=UOp.const(dtypes.uint32,0)
  for i,v in enumerate(vs): r=r.bitwise_or(v.cast(dtypes.uint8).cast(dtypes.uint32).lshift(8*i))
  return r
def i8lane(p, lane): return p.rshift(lane*8).bitwise_and(255).cast(dtypes.uint8).bitcast(dtypes.int8).cast(dtypes.int32)
def q6signed(h,b,g,p):
  half,pg=g//8,g%8
  lo=_q6k_byte(h,b,half*64+(pg%4)*16+p).rshift(4 if pg>=4 else 0).bitwise_and(15)
  hi=_q6k_byte(h,b,128+half*32+(pg%2)*16+p).rshift((pg//2)*2).bitwise_and(3).lshift(4)
  return lo.bitwise_or(hi).cast(dtypes.int32)-32
def emit_q4(rows, rt=2):
  # Eight x4 activation packets cover the 32-element Q4 group.
  def kernel(out,w,xp,xs):
    ro,ri=UOp.range(rows//rt,0),UOp.range(rt,1,axis_type=AxisType.LOCAL)
    p4,b=UOp.range(8,2,axis_type=AxisType.LOCAL),UOp.range(K//256,3,axis_type=AxisType.REDUCE)
    row=ro*rt+ri; base=(row*(K//256)+b)*Q4K_WORDS_PER_BLOCK; c=UOp.const(dtypes.float32,0.)
    for g in range(8):
      d,dm,sc,mn=_q4k_group_params(w,base,g)
      qw=w[base+4+(g//2)*8+p4].rshift((g%2)*4).bitwise_and(0x0F0F0F0F)
      xv=xp[b*64+g*8+p4]; dot=int8x4_dot(UOp.const(dtypes.int32,0),qw,xv).cast(dtypes.float32)
      sx=i8lane(xv,0)+i8lane(xv,1)+i8lane(xv,2)+i8lane(xv,3)
      c=c+xs[b*8+g]*(d*sc.cast(dtypes.float32)*dot-dm*mn.cast(dtypes.float32)*sx.cast(dtypes.float32))
    a=UOp.placeholder((1,),dtypes.float32,20,addrspace=AddrSpace.REG); a=a.after(a[0].store(0.)); a=a.after(a[0].store(a.after(b)[0]+c).end(b)); t=a[0]
    for slot,off in enumerate((4,2,1),90): t=t+_staged_shfl(t,off*rt,p4,slot)
    return out[row].store(t).end(ro,ri,p4).sink(arg=KernelInfo(name=f"q4k_q8_dp4a_{rows}_{K}",opts_to_apply=()))
  return kernel
def emit_q6(rows,rt=2):
  def kernel(out,h,xp,xs):
    ro,ri=UOp.range(rows//rt,0),UOp.range(rt,1,axis_type=AxisType.LOCAL); p4,b=UOp.range(4,2,axis_type=AxisType.LOCAL),UOp.range(K//256,3,axis_type=AxisType.REDUCE)
    row=ro*rt+ri; base=(row*(K//256)+b)*Q6K_HALFWORDS_PER_BLOCK; c=UOp.const(dtypes.float32,0.)
    for g in range(16):
      qw=pack4([q6signed(h,base,g,p4*4+i) for i in range(4)]); dot=int8x4_dot(UOp.const(dtypes.int32,0),qw,xp[b*64+g*4+p4]).cast(dtypes.float32)
      c=c+dot*_f16_half(h[base+104])*_i8(_q6k_byte(h,base,192+g))*xs[b*8+g//2]
    a=UOp.placeholder((1,),dtypes.float32,20,addrspace=AddrSpace.REG); a=a.after(a[0].store(0.)); a=a.after(a[0].store(a.after(b)[0]+c).end(b)); t=a[0]
    for slot,off in enumerate((2,1),90): t=t+_staged_shfl(t,off*rt,p4,slot)
    return out[row].store(t).end(ro,ri,p4).sink(arg=KernelInfo(name=f"q6k_q8_dp4a_{rows}_{K}",opts_to_apply=()))
  return kernel
def q8(x):
  g=x.cast(dtypes.float32).reshape(K//32,32); s=(g.abs().max(axis=1)/127.).maximum(1e-12); q=(g/s.reshape(-1,1)).round().clip(-127,127).cast(dtypes.int8).reshape(K//4,4).cast(dtypes.uint8).cast(dtypes.uint32)
  return (q[:,0]|(q[:,1]<<8)|(q[:,2]<<16)|(q[:,3]<<24)).contiguous(),s.contiguous()
def run(replays=100,reps=5,v_q4=False):
  d=Device.DEFAULT
  if not str(d).startswith("NV"): raise RuntimeError("native NV required")
  wq,_=_make_q4k_words(4096,K,1); wk,_=_make_q4k_words(1024,K,2); vv=_make_q4k_words(1024,K,3)[0] if v_q4 else _make_q6k_halfs(1024,K,3); x=Tensor(np.random.default_rng(4).normal(0,.2,K).astype(np.float16),dtype=dtypes.float16,device=d).contiguous().realize()
  qW=Tensor(wq,device=d).contiguous().realize(); kW=Tensor(wk,device=d).contiguous().realize(); vW=Tensor(vv,dtype=None if v_q4 else dtypes.uint16,device=d).contiguous().realize()
  bq=prog("q4_base_q",q4k_g3_lanemap_gemv_kernel(4096,K)); bk=prog("q4_base_k",q4k_g3_lanemap_gemv_kernel(1024,K)); sp=q6k_spec_for_role(1024,K,role="attn_kv",parts=4,use_coop=False,reduction="external_sum"); bv=prog("q4_base_v" if v_q4 else "q6_base_v",q4k_g3_lanemap_gemv_kernel(1024,K) if v_q4 else emit_q6k_gemv_kernel(sp))
  cq,ck,cv=prog("q4_q8_q",emit_q4(4096)),prog("q4_q8_k",emit_q4(1024)),prog("q4_q8_v" if v_q4 else "q6_q8_v",emit_q4(1024) if v_q4 else emit_q6(1024))
  @TinyJit
  def base(a,b,c,z): return execute_research_program(Tensor.empty(4096,dtype=dtypes.float32,device=d),a,z,program=bq).sum()+execute_research_program(Tensor.empty(1024,dtype=dtypes.float32,device=d),b,z,program=bk).sum()+(execute_research_program(Tensor.empty(1024,dtype=dtypes.float32,device=d),c,z,program=bv) if v_q4 else execute_research_program(Tensor.empty((1024,4),dtype=dtypes.float32,device=d),c,z,program=bv)).sum()
  @TinyJit
  def cand(a,b,c,z):
    xp,xs=q8(z); return execute_research_program(Tensor.empty(4096,dtype=dtypes.float32,device=d),a,xp,xs,program=cq).sum()+execute_research_program(Tensor.empty(1024,dtype=dtypes.float32,device=d),b,xp,xs,program=ck).sum()+execute_research_program(Tensor.empty(1024,dtype=dtypes.float32,device=d),c,xp,xs,program=cv).sum()
  base(qW,kW,vW,x).realize(); cand(qW,kW,vW,x).realize(); Device[d].synchronize()
  # Numerical characterization is intentionally independent of the scalar
  # timing sink: Q8 is approximate, so this reports rather than assumes an
  # exact-baseline contract.
  bpv=execute_research_program(Tensor.empty(1024 if v_q4 else (1024,4),dtype=dtypes.float32,device=d),vW,x,program=bv)
  if not v_q4: bpv=bpv.sum(axis=1)
  xp,xs=q8(x)
  bo=execute_research_program(Tensor.empty(4096,dtype=dtypes.float32,device=d),qW,x,program=bq).cat(execute_research_program(Tensor.empty(1024,dtype=dtypes.float32,device=d),kW,x,program=bk),bpv).numpy()
  co=execute_research_program(Tensor.empty(4096,dtype=dtypes.float32,device=d),qW,xp,xs,program=cq).cat(execute_research_program(Tensor.empty(1024,dtype=dtypes.float32,device=d),kW,xp,xs,program=ck),execute_research_program(Tensor.empty(1024,dtype=dtypes.float32,device=d),vW,xp,xs,program=cv)).numpy()
  max_abs=float(np.max(np.abs(bo-co))); rel=float(max_abs/max(1e-12,float(np.max(np.abs(bo)))))
  for _ in range(200): base(qW,kW,vW,x).realize(); cand(qW,kW,vW,x).realize()
  def tm(f):
    z=[]
    for _ in range(reps):
      Device[d].synchronize(); t=time.perf_counter_ns()
      for _ in range(replays): f(qW,kW,vW,x).realize()
      Device[d].synchronize(); z.append((time.perf_counter_ns()-t)/1e3/replays)
    return z
  A,B,C=tm(base),tm(cand),tm(base); mid=(statistics.median(A)+statistics.median(C))/2
  return {"baseline":A,"candidate":B,"baseline_mid_us":mid,"candidate_us":statistics.median(B),"delta_us":statistics.median(B)-mid,"numerics":{"max_abs_vs_fp16_activation":max_abs,"max_rel_vs_fp16_activation":rel},"gate":"PASS" if statistics.median(B)<mid else "FAIL"}
if __name__=="__main__":
  p=argparse.ArgumentParser();p.add_argument('--replays',type=int,default=100);p.add_argument('--reps',type=int,default=5);p.add_argument('--v-q4',action='store_true');p.add_argument('--out');a=p.parse_args();r=run(a.replays,a.reps,a.v_q4);print(json.dumps(r,indent=2));open(a.out,'w').write(json.dumps(r,indent=2)+'\n') if a.out else None
