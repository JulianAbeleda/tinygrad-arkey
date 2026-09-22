#!/usr/bin/env python3
"""Included-cost native-NV gate for ordinary-CALL cooperative RMSNorm."""
from __future__ import annotations
import argparse, json, os, statistics, subprocess, time
import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes, nn

DIM = 4096

def _names(out):
  linear, _ = out.linear_with_vars()
  return [x.src[0].arg.name for x in linear.src]

def run(replays=1000, reps=7):
  dev = Device.DEFAULT
  if not str(dev).startswith("NV"): raise RuntimeError(f"native NV required, got {dev}")
  rng = np.random.default_rng(20260805)
  x = Tensor(rng.normal(0,.2,(1,DIM)).astype(np.float16), dtype=dtypes.float16, device=dev).realize()
  w = Tensor(rng.normal(1,.05,(DIM,)).astype(np.float16), dtype=dtypes.float16, device=dev).realize()
  ordinary, candidate = nn.RMSNorm(DIM), nn.RMSNorm(DIM)
  ordinary.weight = candidate.weight = w
  candidate._reduce_output_rmsnorm_promoted = True
  @TinyJit
  def a(v): return ordinary(v)
  @TinyJit
  def b(v): return candidate(v)
  a(x).realize(); av=a(x).realize(); b(x).realize(); bv=b(x).realize(); Device[dev].synchronize()
  max_abs=float(np.max(np.abs(av.numpy().astype(np.float32)-bv.numpy().astype(np.float32))))
  def timed(fn):
    vals=[]
    for _ in range(reps):
      Device[dev].synchronize(); st=time.perf_counter_ns()
      for _ in range(replays): fn(x).realize()
      Device[dev].synchronize(); vals.append((time.perf_counter_ns()-st)/1e3/replays)
    return vals
  aa,bb,cc=timed(a),timed(b),timed(a)
  midpoint=(statistics.median(aa)+statistics.median(cc))/2
  return {"schema":"tinygrad.nv_reduce_output_rmsnorm_microgate.v1", "device":str(dev),
    "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    "contract":{"shape":[1,DIM],"realized_inputs":True,"ordinary_call":True,"custom_kernel":False,
      "singleton_graph_for_fair_replay":os.environ.get("GRAPH_ONE_KERNEL")=="1"},
    "topology":{"baseline":_names(ordinary(x)),"candidate":_names(candidate(x))},
    "correctness":{"max_abs":max_abs,"bitwise":max_abs==0.0},
    "timing":{"unit":"us_per_graph_replay","replays":replays,"reps":reps,"control_a":aa,"candidate_b":bb,"control_c":cc,
      "control_midpoint_median":midpoint,"candidate_median":statistics.median(bb),"delta":statistics.median(bb)-midpoint}}

if __name__ == "__main__":
  ap=argparse.ArgumentParser(); ap.add_argument("--replays",type=int,default=1000); ap.add_argument("--reps",type=int,default=7); ap.add_argument("--out")
  args=ap.parse_args(); got=run(args.replays,args.reps); rendered=json.dumps(got,indent=2,sort_keys=True)
  if args.out:
    with open(args.out,"w") as f: f.write(rendered+"\n")
  print(rendered)
