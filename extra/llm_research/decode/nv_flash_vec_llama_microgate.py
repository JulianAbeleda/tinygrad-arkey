#!/usr/bin/env python3
"""Native-NV correctness/resource/wall gate for the llama-vec single-pass flash score substrate."""
from __future__ import annotations

import argparse, hashlib, json, statistics, subprocess, time
from pathlib import Path
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.llm.flash_decode_attention import (flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel,
  flash_fused_gmax_combine_kernel, flash_vec_llama_score_pv_kernel)
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec, execute_research_program
from tinygrad.uop.ops import UOp

Hq, Hkv, Hd, S, MAXC, Tc = 32, 8, 128, 4, 4608, 513
L = 144  # legacy contiguous-split tile length (multiple of TK=16) so both kernels cover tokens 0..512 exactly once
W = Hd + 2


def _programs(output_fp16: bool):
  tc = UOp.const(dtypes.int, Tc)
  legacy = KernelProgram("research.nv_flash_vec_llama", "legacy_score", KernelProgramProvenance.RESEARCH_ONLY,
    flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel(Hd,Hq,Hkv,MAXC,L,S,tc,stage_width=1),
    output_spec=OutputSpec((Hq*S*W,), dtypes.float32))
  candidate = KernelProgram("research.nv_flash_vec_llama", "vec_score", KernelProgramProvenance.RESEARCH_ONLY,
    flash_vec_llama_score_pv_kernel(Hd,Hq,Hkv,MAXC,S,tc),
    output_spec=OutputSpec((Hq*S*W,), dtypes.float32))
  combine = KernelProgram("research.nv_flash_vec_llama", "combine", KernelProgramProvenance.RESEARCH_ONLY,
    flash_fused_gmax_combine_kernel(Hd,Hq,S,output_fp16=output_fp16),
    output_spec=OutputSpec((Hq*Hd,), dtypes.float16 if output_fp16 else dtypes.float32))
  return legacy, candidate, combine


def _score(program, q, cache):
  return execute_research_program(Tensor.empty(Hq*S*W,dtype=dtypes.float32,device=q.device), q, cache, program=program)


def _combine(combine, partial):
  return execute_research_program(Tensor.empty(Hq*Hd,dtype=combine.output_spec.dtype,device=partial.device), partial, program=combine)


def _inputs(case: str):
  rng = np.random.default_rng(20260813)
  if case == "zero":
    q = np.zeros(Hq*Hd,np.float16); cache = np.zeros((2,1,Hkv,MAXC,Hd),np.float16)
  elif case == "dynamic":
    q = np.linspace(-4,4,Hq*Hd,dtype=np.float32).astype(np.float16)
    base = np.linspace(-8,8,MAXC*Hd,dtype=np.float32).reshape(MAXC,Hd).astype(np.float16)
    cache = np.empty((2,1,Hkv,MAXC,Hd),np.float16)
    for h in range(Hkv): cache[0,0,h]=base*((h+1)/8); cache[1,0,h]=base[::-1]*((8-h)/8)
  else:
    q = rng.normal(0,.2,Hq*Hd).astype(np.float16)
    cache = rng.normal(0,.2,(2,1,Hkv,MAXC,Hd)).astype(np.float16)
  return Tensor(q,device="NV").contiguous().realize(), Tensor(cache,device="NV").contiguous().realize()


def run(replays:int=200,reps:int=5):
  if Device.DEFAULT != "NV": raise RuntimeError(f"DEV=NV required, got {Device.DEFAULT}")
  correctness=[]
  normal_inputs=None
  for output_fp16 in (False,True):
    legacy,candidate,combine=_programs(output_fp16)
    for case in ("normal","zero","dynamic"):
      q,cache=_inputs(case)
      a=_combine(combine,_score(legacy,q,cache)); a.realize(); Device["NV"].synchronize(); av=np.asarray(a.numpy()).astype(np.float32).copy()
      b=_combine(combine,_score(candidate,q,cache)); b.realize(); Device["NV"].synchronize(); bv=np.asarray(b.numpy()).astype(np.float32).copy()
      correctness.append({"dtype":"fp16" if output_fp16 else "fp32","case":case,
        "finite":bool(np.isfinite(av).all() and np.isfinite(bv).all()),"bitwise":bool(np.array_equal(av,bv)),
        "max_abs":float(np.max(np.abs(av-bv))),"legacy_sha256":hashlib.sha256(av.tobytes()).hexdigest(),
        "candidate_sha256":hashlib.sha256(bv.tobytes()).hexdigest()})
      if output_fp16 and case=="normal": normal_inputs=(q,cache,legacy,candidate,combine)
  assert normal_inputs is not None
  q,cache,legacy,candidate,combine=normal_inputs
  @TinyJit
  def legacy_graph(q,cache): return _combine(combine,_score(legacy,q,cache))
  @TinyJit
  def candidate_graph(q,cache): return _combine(combine,_score(candidate,q,cache))
  legacy_graph(q,cache).realize(); legacy_graph(q,cache).realize(); candidate_graph(q,cache).realize(); candidate_graph(q,cache).realize(); Device["NV"].synchronize()
  def timed(fn):
    vals=[]
    for _ in range(reps):
      Device["NV"].synchronize(); st=time.perf_counter_ns()
      for _ in range(replays): fn(q,cache).realize()
      Device["NV"].synchronize(); vals.append((time.perf_counter_ns()-st)/1e3/replays)
    return vals
  a1,b,a2=timed(legacy_graph),timed(candidate_graph),timed(legacy_graph)
  amid=(statistics.median(a1)+statistics.median(a2))/2
  return {"schema":"tinygrad.nv_flash_vec_llama_microgate.v1","device":str(Device.DEFAULT),
    "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    "shape":{"Hq":Hq,"Hkv":Hkv,"Hd":Hd,"S":S,"L":L,"Tc":Tc,"MAXC":MAXC},
    "correctness":correctness,
    "timing":{"unit":"us_per_included_graph","replays":replays,"reps":reps,"control_a":a1,"candidate_b":b,"control_c":a2,
      "control_midpoint_median":amid,"candidate_median":statistics.median(b),"delta":statistics.median(b)-amid}}


if __name__ == "__main__":
  ap=argparse.ArgumentParser(); ap.add_argument("--replays",type=int,default=200); ap.add_argument("--reps",type=int,default=5); ap.add_argument("--out",type=Path)
  args=ap.parse_args(); got=run(args.replays,args.reps); print(json.dumps(got,indent=2,sort_keys=True))
  if args.out: args.out.write_text(json.dumps(got,indent=2,sort_keys=True)+"\n")
