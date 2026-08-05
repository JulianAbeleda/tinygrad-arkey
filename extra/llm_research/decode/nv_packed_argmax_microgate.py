#!/usr/bin/env python3
"""Included-cost native-NV microgate for the closed packed-greedy argmax route."""
from __future__ import annotations
import argparse, hashlib, json, statistics, subprocess, time
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.llm.packed_argmax import packed_argmax_finite_fp32

VOCAB = 151936

def run(replays:int=300, reps:int=7) -> dict:
  dev=Device.DEFAULT
  if not str(dev).startswith("NV"): raise RuntimeError(f"native NV required, got {dev}")
  # Values cover both signs and use deliberate equal maxima to qualify the
  # first-index rule, while retaining the full LM-head-sized input/copy cost.
  host=np.random.default_rng(20260805).standard_normal((1,VOCAB)).astype(np.float32)
  host[0,17]=host[0,923]=np.finfo(np.float32).max
  x=Tensor(host,dtype=dtypes.float32,device=dev).contiguous().realize()
  @TinyJit
  def legacy(a): return a.argmax(-1, keepdim=True)
  @TinyJit
  def packed(a): return packed_argmax_finite_fp32(a, -1, keepdim=True)
  old,new=legacy(x).realize(),packed(x).realize(); Device[dev].synchronize()
  same=bool(np.array_equal(old.numpy(),new.numpy()) and int(new.item()) == 17)
  # Prime outside the clock window.  A/B/A includes the Tensor route's index
  # construction and every ordinary reduction -- not only its final kernel.
  for _ in range(100): legacy(x).realize(); packed(x).realize()
  Device[dev].synchronize()
  def timed(fn):
    out=[]
    for _ in range(reps):
      Device[dev].synchronize(); st=time.perf_counter_ns()
      for _ in range(replays): fn(x).realize()
      Device[dev].synchronize(); out.append((time.perf_counter_ns()-st)/1e3/replays)
    return out
  a,b,c=timed(legacy),timed(packed),timed(legacy)
  mid=(statistics.median(a)+statistics.median(c))/2
  return {"schema":"tinygrad.nv.packed_argmax_microgate.v1", "device":str(dev),
    "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    "contract":{"shape":[1,VOCAB],"dtype":"float32","finite_qualified":True,"tie_first":True,
      "preserves":"ordinary Tensor.argmax semantics for finite fp32 only"},
    "correctness":{"legacy":old.numpy().tolist(),"packed":new.numpy().tolist(),"exact":same,
      "input_sha256":hashlib.sha256(host.tobytes()).hexdigest()},
    "timing":{"unit":"us_per_full_argmax_graph","replays":replays,"reps":reps,"control_a":a,"candidate_b":b,"control_c":c,
      "control_midpoint_median":mid,"candidate_median":statistics.median(b),"delta":statistics.median(b)-mid,
      "gate":"PASS" if same and statistics.median(b) < mid else "FAIL"},
    "topology":{"legacy":"ordinary max + equal + index max chain","candidate":"ordered fp32/u32 key + one uint64 MAX",
      "note":"wall debit requires a later full-token capture; this isolated gate is direction only"}}

def main():
  p=argparse.ArgumentParser(); p.add_argument("--replays",type=int,default=300); p.add_argument("--reps",type=int,default=7); p.add_argument("--out")
  a=p.parse_args(); result=run(a.replays,a.reps); text=json.dumps(result,indent=2,sort_keys=True)
  if a.out: open(a.out,"w").write(text+"\n")
  print(text); return 0 if result["timing"]["gate"] == "PASS" else 1
if __name__ == "__main__": raise SystemExit(main())
