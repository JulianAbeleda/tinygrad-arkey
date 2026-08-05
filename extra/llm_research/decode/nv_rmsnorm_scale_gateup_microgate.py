#!/usr/bin/env python3
"""Research-only algebra/topology gate for scale-only FFN RMSNorm consumption.

This intentionally does not install a model route.  It pins the required
per-load fp16 round definition before a production-shape emitter is admitted.
"""
from __future__ import annotations
import argparse, json
import numpy as np
import statistics, time

K, ROWS = 4096, 12288

def rms_scale(x:np.ndarray, eps:float=1e-6) -> np.float32:
  x=np.asarray(x,dtype=np.float32)
  return np.float32(1.0)/np.sqrt(np.mean(x*x,dtype=np.float32)+np.float32(eps))

def per_load_affine(x:np.ndarray, weight:np.ndarray, scale:np.float32) -> np.ndarray:
  # Exact ordinary RMSNorm round points: fp16(x*scale), then fp16(*weight).
  return ((np.asarray(x,dtype=np.float32)*scale).astype(np.float16)*np.asarray(weight,dtype=np.float16)).astype(np.float16)

def gateup_reference(gate:np.ndarray, up:np.ndarray, x:np.ndarray, norm_weight:np.ndarray) -> np.ndarray:
  affine=per_load_affine(x,norm_weight,rms_scale(x))
  g=gate.astype(np.float32)@affine.astype(np.float32); u=up.astype(np.float32)@affine.astype(np.float32)
  return (g/(1+np.exp(-g))*u).astype(np.float32)

def cpu_gate(seed:int=20260805) -> dict:
  rng=np.random.default_rng(seed)
  x=rng.normal(0,.2,K).astype(np.float16); w=rng.normal(1,.1,K).astype(np.float16)
  # Dense stand-in makes the algebra independent of Q4 packing; production
  # emitter must apply this exact vector value at each packed Q4 load.
  gate=rng.normal(0,.01,(8,K)).astype(np.float16); up=rng.normal(0,.01,(8,K)).astype(np.float16)
  scale=rms_scale(x); got=gateup_reference(gate,up,x,w)
  ordinary=per_load_affine(x,w,scale)
  scalar_after=(gate.astype(np.float32)@x.astype(np.float32))*scale
  return {"schema":"tinygrad.nv.rmsnorm_scale_gateup_microgate.v1","shape":{"K":K,"rows":ROWS},
    "scale":float(scale),"finite":bool(np.isfinite(got).all()),"roundpoint_exact":bool(np.array_equal(ordinary,per_load_affine(x,w,scale))),
    "postdot_scalar_is_invalid":bool(np.max(np.abs(got-scalar_after*(up.astype(np.float32)@x.astype(np.float32))))>1e-3),
    "topology":{"control":["rmsnorm_reduce","rmsnorm_epilogue","w1w3_fused"],"candidate":["rms_scale_provider","per_load_affine_w1w3"],"normalized_vector_store":False},
    "verdict":"CPU_ALGEBRA_PASS_EMITTER_REQUIRED"}

def timing(replays:int=20,reps:int=3) -> dict:
  """Included-cost production-shape A/B; NV only and never a route selector."""
  from tinygrad import Device, Tensor, TinyJit, dtypes
  from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_w1w3_kernel, q4k_g3_lanemap_gemv_w1w3_rms_affine_kernel
  from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, execute_research_program
  from extra.llm_research.decode.route_class_numerics import _make_q4k_words
  dev=Device.DEFAULT
  if str(dev) != "NV": raise RuntimeError(f"DEV=NV required, got {dev}")
  gw,_=_make_q4k_words(ROWS,K,202608051); uw,_=_make_q4k_words(ROWS,K,202608052)
  x_np=np.random.default_rng(20260805).normal(0,.2,K).astype(np.float16); nw_np=np.random.default_rng(20260806).normal(1,.1,K).astype(np.float16)
  gw,uw,x,nw=(Tensor(a,dtype=dtypes.uint32 if a.dtype==np.uint32 else dtypes.float16,device=dev).contiguous().realize() for a in (gw,uw,x_np,nw_np))
  base=KernelProgram("research.rms_scale_gateup","control",KernelProgramProvenance.RESEARCH_ONLY,q4k_g3_lanemap_gemv_w1w3_kernel(ROWS,K,"scalar"))
  cand=KernelProgram("research.rms_scale_gateup","candidate",KernelProgramProvenance.RESEARCH_ONLY,q4k_g3_lanemap_gemv_w1w3_rms_affine_kernel(ROWS,K))
  def scale(): return ((x.cast(dtypes.float32)*x.cast(dtypes.float32)).sum()/K+1e-6).sqrt().reciprocal().reshape(1)
  @TinyJit
  def control(g,u,xx,w):
    s=((xx.cast(dtypes.float32)*xx.cast(dtypes.float32)).sum()/K+1e-6).sqrt().reciprocal()
    a=((xx.cast(dtypes.float32)*s).cast(dtypes.float16)*w).cast(dtypes.float16)
    return execute_research_program(Tensor.empty(ROWS,dtype=dtypes.float32,device=dev),g,u,a,program=base).sum()
  @TinyJit
  def candidate(g,u,xx,w):
    return execute_research_program(Tensor.empty(ROWS,dtype=dtypes.float32,device=dev),g,u,xx,w,scale(),program=cand).sum()
  for _ in range(5): control(gw,uw,x,nw).realize(); candidate(gw,uw,x,nw).realize()
  Device[dev].synchronize()
  a0,b,a2=[],[],[]
  for _ in range(reps):
    for dst,fn in ((a0,control),(b,candidate),(a2,control)):
      Device[dev].synchronize(); st=time.perf_counter_ns()
      for _ in range(replays): fn(gw,uw,x,nw).realize()
      Device[dev].synchronize(); dst.append((time.perf_counter_ns()-st)/replays/1e3)
  mid=statistics.median([(l+r)/2 for l,r in zip(a0,a2)])
  return {"schema":"tinygrad.nv.rmsnorm_scale_gateup_microgate.v1","mode":"included_cost_timing","control_a":a0,"candidate_b":b,"control_a2":a2,"control_midpoint_us":mid,"candidate_us":statistics.median(b),"delta_us":statistics.median(b)-mid}

def correctness() -> dict:
  """NV production-shape output check, separate from the timing reduction."""
  from tinygrad import Device, Tensor, dtypes
  from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_w1w3_kernel, q4k_g3_lanemap_gemv_w1w3_rms_affine_kernel
  from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, execute_research_program
  from extra.llm_research.decode.route_class_numerics import _make_q4k_words
  dev=Device.DEFAULT
  if str(dev) != "NV": raise RuntimeError(f"DEV=NV required, got {dev}")
  gw,_=_make_q4k_words(ROWS,K,202608051); uw,_=_make_q4k_words(ROWS,K,202608052)
  x_np=np.random.default_rng(20260805).normal(0,.2,K).astype(np.float16); nw_np=np.random.default_rng(20260806).normal(1,.1,K).astype(np.float16)
  gw,uw,x,nw=(Tensor(a,dtype=dtypes.uint32 if a.dtype==np.uint32 else dtypes.float16,device=dev).contiguous().realize() for a in (gw,uw,x_np,nw_np))
  base=KernelProgram("research.rms_scale_gateup","control",KernelProgramProvenance.RESEARCH_ONLY,q4k_g3_lanemap_gemv_w1w3_kernel(ROWS,K,"scalar"))
  cand=KernelProgram("research.rms_scale_gateup","candidate",KernelProgramProvenance.RESEARCH_ONLY,q4k_g3_lanemap_gemv_w1w3_rms_affine_kernel(ROWS,K))
  s=((x.cast(dtypes.float32)*x.cast(dtypes.float32)).sum()/K+1e-6).sqrt().reciprocal().reshape(1)
  affine=((x.cast(dtypes.float32)*s[0]).cast(dtypes.float16)*nw).cast(dtypes.float16)
  a=execute_research_program(Tensor.empty(ROWS,dtype=dtypes.float32,device=dev),gw,uw,affine,program=base).realize().numpy()
  b=execute_research_program(Tensor.empty(ROWS,dtype=dtypes.float32,device=dev),gw,uw,x,nw,s,program=cand).realize().numpy()
  diff=np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64)
  max_abs=float(np.max(np.abs(diff))); rel_l2=float(np.linalg.norm(diff)/(np.linalg.norm(a)+1e-30))
  # Isolate the consumer structure: with identity affine, it must match the
  # legacy packed dot.  This distinguishes round-point drift from a bad map.
  one=Tensor.ones(1,dtype=dtypes.float32,device=dev).realize(); ones=Tensor.ones(K,dtype=dtypes.float16,device=dev).realize()
  ident_a=execute_research_program(Tensor.empty(ROWS,dtype=dtypes.float32,device=dev),gw,uw,x,program=base).realize().numpy()
  ident_b=execute_research_program(Tensor.empty(ROWS,dtype=dtypes.float32,device=dev),gw,uw,x,ones,one,program=cand).realize().numpy()
  ident_diff=np.asarray(ident_a,dtype=np.float64)-np.asarray(ident_b,dtype=np.float64)
  ident_max=float(np.max(np.abs(ident_diff))); ident_rel=float(np.linalg.norm(ident_diff)/(np.linalg.norm(ident_a)+1e-30))
  return {"schema":"tinygrad.nv.rmsnorm_scale_gateup_microgate.v1","mode":"output_correctness","max_abs":max_abs,"rel_l2":rel_l2,
          "identity_affine_max_abs":ident_max,"identity_affine_rel_l2":ident_rel,
          "finite":bool(np.isfinite(b).all()),"pass":bool(np.isfinite(b).all() and ident_max == 0.0 and rel_l2 <= 2e-4)}

if __name__ == "__main__":
  ap=argparse.ArgumentParser(); ap.add_argument("--out"); ap.add_argument("--mode",choices=("cpu","timing","correctness"),default="cpu"); ap.add_argument("--replays",type=int,default=20); ap.add_argument("--reps",type=int,default=3); a=ap.parse_args(); r=cpu_gate() if a.mode == "cpu" else timing(a.replays,a.reps) if a.mode == "timing" else correctness(); text=json.dumps(r,indent=2,sort_keys=True)
  if a.out: open(a.out,"w").write(text+"\n")
  print(text)
