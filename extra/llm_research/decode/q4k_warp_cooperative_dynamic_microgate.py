#!/usr/bin/env python3
"""Included-cost native-NV gate for the research-only dynamic Q4 four-warp map."""
from __future__ import annotations
import argparse, hashlib, json, statistics, subprocess, time
import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_kernel
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, execute_research_program
from tinygrad.llm.shared_q8_attention import _emit_q8_provider
from tinygrad.uop.ops import UOp
from extra.llm_research.decode.q4k_warp_cooperative_dynamic import ROWS, K, emit_q4k_warp_cooperative_q8_partial
from extra.llm_research.decode.route_class_numerics import _make_q4k_words
from extra.llm_research.layout import q4_k_reference

def _p(name, fn): return KernelProgram("research.q4k_warp_coop_dynamic", name, KernelProgramProvenance.RESEARCH_ONLY, fn)

def run(replays=200, reps=7):
  dev=Device.DEFAULT
  if not str(dev).startswith("NV"): raise RuntimeError(f"native NV required, got {dev}")
  raw,_=_make_q4k_words(ROWS,K,20260805)
  xn=np.random.default_rng(20260805).normal(0,.2,K).astype(np.float16)
  w=Tensor(raw, dtype=dtypes.uint32, device=dev).contiguous().realize(); x=Tensor(xn,dtype=dtypes.float16,device=dev).contiguous().realize()
  basep=_p("q4_installed",q4k_g3_lanemap_gemv_kernel(ROWS,K)); providerp=_p("q8_provider",_emit_q8_provider())
  block_var=UOp.variable("q4k_coop_blocks",1,4); bound_blocks=block_var.bind(4)
  @TinyJit
  def baseline(ww,xx): return execute_research_program(Tensor.empty((ROWS,),dtype=dtypes.float32,device=dev),ww,xx,program=basep)
  @TinyJit
  def candidate(ww,xx,blocks):
    # Included cost: the actual packed llama-Q8 provider ABI, cooperative Q4,
    # then the four-partial reduction. AFTER makes the runtime scalar binding
    # visible to scheduling without adding a kernel or changing the Q8 buffer.
    xp=execute_research_program(Tensor.empty((K//4+K//32,),dtype=dtypes.uint32,device=dev),xx,program=providerp)
    xp=Tensor(xp.uop.after(blocks))
    # The BIND carries value=4 to scheduling; the kernel AST itself receives
    # the underlying DEFINE_VAR, which is the legal runtime scalar parameter.
    extent=blocks.src[0]
    candp=_p("q4_coop_q8",emit_q4k_warp_cooperative_q8_partial(block_count=extent))
    return execute_research_program(Tensor.empty((ROWS,4),dtype=dtypes.float32,device=dev),ww,xp,program=candp).sum(axis=1).contiguous()
  bo=baseline(w,x).realize(); co=candidate(w,x,bound_blocks).realize(); Device[dev].synchronize()

  # CPU references first. The Q8 provider quantizes from the fp16-rounded
  # input with fp32 d, stores d as fp16, and the consumer dequantizes q*d_fp16.
  # This separates mapping correctness from the admitted Q8 approximation.
  raw_bytes=raw.view(np.uint8).reshape(-1).copy()
  weights=q4_k_reference(Tensor(raw_bytes,dtype=dtypes.uint8),ROWS*K).numpy().astype(np.float32).reshape(ROWS,K)
  x32=xn.astype(np.float32); groups=x32.reshape(-1,32); d=np.max(np.abs(groups),axis=1)/127.0
  inv=np.divide(1.0,d,out=np.zeros_like(d),where=d!=0)
  qi=np.rint(groups*inv[:,None]).clip(-128,127).astype(np.int8)
  xq=(qi.astype(np.float32)*d.astype(np.float16).astype(np.float32)[:,None]).reshape(-1)
  fp16_ref=weights@x32; q8_ref=weights@xq
  bg,cg=bo.numpy().astype(np.float32),co.numpy().astype(np.float32)
  base_err=float(np.max(np.abs(bg-fp16_ref))); cand_err=float(np.max(np.abs(cg-q8_ref)))
  q8_rel_l2=float(np.linalg.norm((cg-bg).astype(np.float64))/max(np.linalg.norm(bg.astype(np.float64)),1e-30))
  base_tol=max(1e-2,float(np.max(np.abs(fp16_ref)))*1e-2); cand_tol=max(1e-2,float(np.max(np.abs(q8_ref)))*1e-2)
  correctness_pass=bool(base_err<=base_tol and cand_err<=cand_tol and np.isfinite(cg).all())
  if not correctness_pass: raise RuntimeError(f"correctness gate failed: baseline={base_err}/{base_tol} candidate={cand_err}/{cand_tol}")
  for _ in range(200): baseline(w,x).realize(); candidate(w,x,bound_blocks).realize()
  def timed(f):
    out=[]
    for _ in range(reps):
      Device[dev].synchronize(); st=time.perf_counter_ns()
      for _ in range(replays): f(w,x).realize()
      Device[dev].synchronize(); out.append((time.perf_counter_ns()-st)/1e3/replays)
    return out
  def timed_candidate(_ww,_xx): return candidate(_ww,_xx,bound_blocks)
  a,b,c=timed(baseline),timed(timed_candidate),timed(baseline); mid=(statistics.median(a)+statistics.median(c))/2
  return {"schema":"tinygrad.q4k_warp_coop_dynamic_microgate.v3","git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    "shape":{"rows":ROWS,"k":K,"local":128,"runtime_blocks":4},"payload":{"q4_sha256":hashlib.sha256(raw_bytes).hexdigest(),"x_sha256":hashlib.sha256(xn.tobytes()).hexdigest()},
    "census":{"control_programs_per_replay":1,"candidate_programs_per_replay":3,"candidate_programs":["q8_1_llama_provider_4096","q4k_warp_coop_q8_dp4a_partial_4096_4096","partial_sum_4096x4"]},
    "prior_invalid_construction":{"candidate_max_abs_cpu_q8":0.9392952919006348,"candidate_tol":0.09806095123291016,
      "disposition":"header nibble-stride bug fixed before this corrected retry; no timing was run"},
    "correctness":{"pass":correctness_pass,"baseline_max_abs_cpu_fp16":base_err,"baseline_tol":base_tol,"candidate_max_abs_cpu_q8":cand_err,"candidate_tol":cand_tol,"q8_vs_fp16_relative_l2":q8_rel_l2,"representation":"llama Q8_1 approximate; no full-logit promotion contract"},
    "timing":{"unit":"us_per_graph_replay","replays":replays,"reps":reps,"control_a":a,"candidate_b":b,"control_c":c,"control_midpoint_median":mid,"candidate_median":statistics.median(b),"delta":statistics.median(b)-mid,
      "gate":"PASS" if statistics.median(b)<mid else "FAIL"}}

if __name__ == "__main__":
  ap=argparse.ArgumentParser(); ap.add_argument("--replays",type=int,default=200); ap.add_argument("--reps",type=int,default=7); ap.add_argument("--out"); a=ap.parse_args()
  r=run(a.replays,a.reps); print(json.dumps(r,indent=2));
  if a.out: open(a.out,"w").write(json.dumps(r,indent=2)+"\n")
