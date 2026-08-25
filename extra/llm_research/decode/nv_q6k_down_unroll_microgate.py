#!/usr/bin/env python3
"""Bit-exact microgate for Q6 packed-lane block-loop unroll variants."""
from __future__ import annotations

import argparse, hashlib, json, pathlib, statistics, subprocess, time
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, execute_research_program
from tinygrad.llm.q6k_ffn_down_mmvq import K, ROWS, emit_q6k_four_warp_fp16_direct
from extra.llm_research.decode.route_class_numerics import _make_q6k_halfs


def program(unroll:int|None) -> KernelProgram:
  key = "control" if unroll is None else f"u{unroll}"
  return KernelProgram("research.q6k_ffn_down_unroll", key, KernelProgramProvenance.RESEARCH_ONLY,
    emit_q6k_four_warp_fp16_direct(packed_lanemap=True, unroll_blocks=unroll))


def run(replays:int,reps:int)->dict:
  dev=Device.DEFAULT
  if not str(dev).startswith("NV"): raise RuntimeError(f"native NV required, got {dev}")
  halfs_np=_make_q6k_halfs(ROWS,K,2026082404)
  x_np=np.random.default_rng(2026082405).normal(0,0.2,K).astype(np.float16)
  h_np=np.random.default_rng(2026082406).normal(0,0.05,ROWS).astype(np.float32)
  halfs=Tensor(halfs_np.copy(),dtype=dtypes.uint16,device=dev).contiguous().realize()
  x=Tensor(x_np.copy(),dtype=dtypes.float16,device=dev).contiguous().realize()
  h=Tensor(h_np.copy(),dtype=dtypes.float32,device=dev).contiguous().realize()

  variants=(None,2,3,4,6,12)
  def make(unroll):
    p=program(unroll)
    @TinyJit
    def fn(w:Tensor,a:Tensor,residual:Tensor):
      return execute_research_program(Tensor.empty((ROWS,),dtype=dtypes.float32,device=dev),w,a,residual,program=p)
    return fn
  fns={u:make(u) for u in variants}; outputs={}
  for u,fn in fns.items(): fn(halfs,x,h).realize(); outputs[u]=fn(halfs,x,h).realize().numpy().astype(np.float32)
  Device[dev].synchronize(); ref=outputs[None]
  exact={str(u):{"bitwise_identical":bool(np.array_equal(ref.view(np.uint32),outputs[u].view(np.uint32))),
    "max_abs_diff":float(np.max(np.abs(ref-outputs[u]))),"finite":bool(np.isfinite(outputs[u]).all())} for u in variants[1:]}
  def timed(fn):
    vals=[]
    for _ in range(reps):
      Device[dev].synchronize(); t=time.perf_counter_ns()
      for _ in range(replays): fn(halfs,x,h).realize()
      Device[dev].synchronize(); vals.append((time.perf_counter_ns()-t)/1e3/replays)
    return vals
  samples={"control":timed(fns[None]),**{str(u):timed(fns[u]) for u in variants[1:]}}
  med={k:statistics.median(v) for k,v in samples.items()}; best=min((str(u) for u in variants[1:]),key=med.__getitem__)
  return {"schema":"tinygrad.nv_q6k_down_unroll_microgate.v1","commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    "payload":{"q6_sha256":hashlib.sha256(halfs_np.view(np.uint8)).hexdigest(),"x_sha256":hashlib.sha256(x_np.tobytes()).hexdigest()},
    "mechanism":"unroll the 12-block packed-lane loop to expose independent loads against long-scoreboard stalls",
    "correctness":exact,"timing":{"unit":"us_per_launch_host_synchronized","replays":replays,"reps":reps,
      "samples":samples,"medians":med,"best_unroll":int(best),"best_recovery_us":med["control"]-med[best]}}


def main()->int:
  ap=argparse.ArgumentParser();ap.add_argument("--replays",type=int,default=200);ap.add_argument("--reps",type=int,default=7)
  ap.add_argument("--out",type=pathlib.Path,required=True);args=ap.parse_args();result=run(args.replays,args.reps)
  args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
  print(json.dumps(result,indent=2,sort_keys=True));return 0 if all(x["bitwise_identical"] for x in result["correctness"].values()) else 1


if __name__=="__main__":raise SystemExit(main())
