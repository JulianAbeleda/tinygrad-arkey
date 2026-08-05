#!/usr/bin/env python3
"""Isolated native-NV correctness/resource/wall gate for d512 single-stage flash decode."""
from __future__ import annotations

import argparse, hashlib, json, statistics, subprocess, time
from pathlib import Path
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.codegen import to_program
from tinygrad.engine.realize import get_runtime
from tinygrad.llm.flash_decode_attention import (flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel,
  flash_fused_gmax_combine_kernel, flash_single_stage_d512_kernel)
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec, execute_research_program
from tinygrad.uop.ops import Ops, UOp

Hq, Hkv, Hd, S, L, Tc, MAXC = 32, 8, 128, 4, 128, 513, 4608
W = Hd + 2


def _programs(output_fp16:bool):
  tc = UOp.const(dtypes.int, Tc)
  tile = KernelProgram("research.nv_flash_single_stage", "legacy_tile", KernelProgramProvenance.RESEARCH_ONLY,
    flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel(Hd,Hq,Hkv,MAXC,L,S,tc,stage_width=1),
    output_spec=OutputSpec((Hq*S*W,), dtypes.float32))
  combine = KernelProgram("research.nv_flash_single_stage", "legacy_combine", KernelProgramProvenance.RESEARCH_ONLY,
    flash_fused_gmax_combine_kernel(Hd,Hq,S,output_fp16=output_fp16),
    output_spec=OutputSpec((Hq*Hd,), dtypes.float16 if output_fp16 else dtypes.float32))
  candidate = KernelProgram("research.nv_flash_single_stage", "single_stage", KernelProgramProvenance.RESEARCH_ONLY,
    flash_single_stage_d512_kernel(Hd,Hq,Hkv,L,tc,output_fp16=output_fp16),
    output_spec=OutputSpec((Hq*Hd,), dtypes.float16 if output_fp16 else dtypes.float32))
  return tile, combine, candidate


def _legacy(tile, combine, q, cache):
  partial = execute_research_program(Tensor.empty(Hq*S*W,dtype=dtypes.float32,device=q.device), q, cache, program=tile)
  return execute_research_program(Tensor.empty(Hq*Hd,dtype=combine.output_spec.dtype,device=q.device), partial, program=combine)


def _candidate(candidate, q, cache):
  return execute_research_program(Tensor.empty(Hq*Hd,dtype=candidate.output_spec.dtype,device=q.device), q, cache, program=candidate)


def _lower_one(out:Tensor):
  linear, var_vals = out.linear_with_vars()
  if var_vals: raise RuntimeError(f"static probe expected, got {var_vals}")
  rows=[]
  for call in linear.src:
    ast=call.src[0]
    if ast.op is Ops.SINK: ast=to_program(ast,Device["NV"].renderer)
    if ast.op is not Ops.PROGRAM: continue
    prg=get_runtime("NV",ast)
    rows.append({"name":prg.name,"regs_usage":prg.regs_usage,"shmem_usage":prg.shmem_usage,
      "lcmem_usage":prg.lcmem_usage,"slm_per_thread":prg.dev.slm_per_thread,
      "source_sha256":hashlib.sha256(ast.src[4].arg if isinstance(ast.src[4].arg,bytes) else ast.src[4].arg.encode()).hexdigest()})
  return rows


def _inputs(case:str):
  rng=np.random.default_rng(20260805)
  if case == "zero":
    q=np.zeros(Hq*Hd,np.float16); cache=np.zeros((2,1,Hkv,MAXC,Hd),np.float16)
  elif case == "dynamic":
    q=np.linspace(-4,4,Hq*Hd,dtype=np.float32).astype(np.float16)
    base=np.linspace(-8,8,MAXC*Hd,dtype=np.float32).reshape(MAXC,Hd).astype(np.float16)
    cache=np.empty((2,1,Hkv,MAXC,Hd),np.float16)
    for h in range(Hkv): cache[0,0,h]=base*((h+1)/8); cache[1,0,h]=base[::-1]*((8-h)/8)
  else:
    q=rng.normal(0,.2,Hq*Hd).astype(np.float16)
    cache=rng.normal(0,.2,(2,1,Hkv,MAXC,Hd)).astype(np.float16)
  return Tensor(q,device="NV").contiguous().realize(), Tensor(cache,device="NV").contiguous().realize()


def run(replays:int=1000,reps:int=7):
  if Device.DEFAULT != "NV": raise RuntimeError(f"DEV=NV required, got {Device.DEFAULT}")
  correctness=[]
  normal_inputs=None
  for output_fp16 in (False,True):
    tile,combine,candidate=_programs(output_fp16)
    for case in ("normal","zero","dynamic"):
      q,cache=_inputs(case)
      a=_legacy(tile,combine,q,cache)
      a.realize(); Device["NV"].synchronize(); av=np.asarray(a.numpy()).astype(np.float32).copy()
      b=_candidate(candidate,q,cache)
      b.realize(); Device["NV"].synchronize(); bv=np.asarray(b.numpy()).astype(np.float32).copy()
      correctness.append({"dtype":"fp16" if output_fp16 else "fp32","case":case,
        "finite":bool(np.isfinite(av).all() and np.isfinite(bv).all()),"bitwise":bool(np.array_equal(av,bv)),
        "max_abs":float(np.max(np.abs(av-bv))),"legacy_sha256":hashlib.sha256(av.tobytes()).hexdigest(),
        "candidate_sha256":hashlib.sha256(bv.tobytes()).hexdigest()})
      if output_fp16 and case=="normal": normal_inputs=(q,cache,tile,combine,candidate)
  assert normal_inputs is not None
  # Resource inspection is deliberately separate from numerical ownership: lowering a pending custom graph
  # must not perturb the held buffers used as the differential oracle.
  rt,rc,rs=_programs(False); rq,rk=_inputs("normal")
  resource={"legacy":_lower_one(_legacy(rt,rc,rq,rk)),"candidate":_lower_one(_candidate(rs,rq,rk))}
  q,cache,tile,combine,candidate=normal_inputs
  @TinyJit
  def baseline(q,cache): return _legacy(tile,combine,q,cache)
  @TinyJit
  def one_stage(q,cache): return _candidate(candidate,q,cache)
  baseline(q,cache).realize(); baseline(q,cache).realize(); one_stage(q,cache).realize(); one_stage(q,cache).realize(); Device["NV"].synchronize()
  def timed(fn):
    vals=[]
    for _ in range(reps):
      Device["NV"].synchronize(); st=time.perf_counter_ns()
      for _ in range(replays): fn(q,cache).realize()
      Device["NV"].synchronize(); vals.append((time.perf_counter_ns()-st)/1e3/replays)
    return vals
  a1,b,a2=timed(baseline),timed(one_stage),timed(baseline)
  amid=(statistics.median(a1)+statistics.median(a2))/2
  return {"schema":"tinygrad.nv_flash_single_stage_microgate.v1","device":str(Device.DEFAULT),
    "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    "shape":{"Hq":Hq,"Hkv":Hkv,"Hd":Hd,"S":S,"L":L,"Tc":Tc,"MAXC":MAXC},
    "correctness":correctness,"resources":resource,
    "timing":{"unit":"us_per_included_graph","replays":replays,"reps":reps,"control_a":a1,"candidate_b":b,"control_c":a2,
      "control_midpoint_median":amid,"candidate_median":statistics.median(b),"delta":statistics.median(b)-amid}}


if __name__ == "__main__":
  ap=argparse.ArgumentParser(); ap.add_argument("--replays",type=int,default=1000); ap.add_argument("--reps",type=int,default=7); ap.add_argument("--out",type=Path)
  args=ap.parse_args(); got=run(args.replays,args.reps); encoded=json.dumps(got,indent=2,sort_keys=True)
  print(encoded)
  if args.out: args.out.write_text(encoded+"\n")
