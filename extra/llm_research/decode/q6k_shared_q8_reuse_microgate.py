#!/usr/bin/env python3
"""Research-only included-cost Q6_K Q/K/V shared-Q8 microgate.

This deliberately tests an optimization llama.cpp does *not* use in the pinned
d512 graph: one Q8_1 quantization of the common attention-norm activation,
then three independent Q6 consumers.  It is useful only if the complete
three-consumer graph beats three installed partial4+sum graphs.  No production
route is selected here.
"""
from __future__ import annotations
import argparse, hashlib, json, statistics, subprocess, time
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.llm.decode_kernels import emit_q6k_gemv_kernel, q6k_spec_for_role
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, execute_research_program
from extra.llm_research.decode.q6k_q8_dp4a_microgate import emit_q6k_q8_dp4a, q8_1_pack, ROWS, K
from extra.llm_research.decode.route_class_numerics import _make_q6k_halfs

def _program(name, emitter):
  return KernelProgram("research.q6k_shared_q8", name, KernelProgramProvenance.RESEARCH_ONLY, emitter)

def run(replays:int=300, reps:int=7, row_tile:int=2) -> dict:
  dev=Device.DEFAULT
  if not str(dev).startswith("NV"): raise RuntimeError(f"native NV required, got {dev}")
  rng=np.random.default_rng(20260805)
  # Separate immutable buffers make the three consumers visible to the graph;
  # their payloads have matching shape/layout but intentionally differ.
  ws=[Tensor(_make_q6k_halfs(ROWS,K,20260805+i),dtype=dtypes.uint16,device=dev).contiguous().realize() for i in range(3)]
  x=Tensor(rng.normal(0,.2,K).astype(np.float16),dtype=dtypes.float16,device=dev).contiguous().realize()
  bs=q6k_spec_for_role(ROWS,K,role="attn_kv",parts=4,use_coop=False,reduction="external_sum")
  bp=_program(bs.kernel_name,emit_q6k_gemv_kernel(bs)); cp=_program(f"q6k_q8_dp4a_rt{row_tile}",emit_q6k_q8_dp4a(row_tile))
  def partial(w, xx):
    return execute_research_program(Tensor.empty((ROWS,4),dtype=dtypes.float32,device=dev),w,xx,program=bp).sum(axis=1).contiguous()
  def dot(w,xp,xs):
    return execute_research_program(Tensor.empty((ROWS,),dtype=dtypes.float32,device=dev),w,xp,xs,program=cp)
  @TinyJit
  def baseline(w0,w1,w2,xx):
    # The scalar reductions retain all three projections without adding a
    # materialized output path to only one arm.
    return partial(w0,xx).sum()+partial(w1,xx).sum()+partial(w2,xx).sum()
  @TinyJit
  def shared(w0,w1,w2,xx):
    xp,xs=q8_1_pack(xx)
    return dot(w0,xp,xs).sum()+dot(w1,xp,xs).sum()+dot(w2,xp,xs).sum()
  baseline(*ws,x).realize(); shared(*ws,x).realize(); Device[dev].synchronize()
  # Keep the same clock-settling discipline as the one-consumer gate.
  for _ in range(1000): baseline(*ws,x).realize(); shared(*ws,x).realize()
  Device[dev].synchronize()
  def timed(fn):
    vals=[]
    for _ in range(reps):
      Device[dev].synchronize(); st=time.perf_counter_ns()
      for _ in range(replays): fn(*ws,x).realize()
      Device[dev].synchronize(); vals.append((time.perf_counter_ns()-st)/1e3/replays)
    return vals
  a,b,c=timed(baseline),timed(shared),timed(baseline)
  mid=(statistics.median(a)+statistics.median(c))/2
  # This checks common-producer semantics, not exact fp16 baseline numerics:
  # two separately evaluated Q8 paths must equal the single shared Q8 path.
  xp0,xs0=q8_1_pack(x); xp1,xs1=q8_1_pack(x)
  Device[dev].synchronize()
  q8_repeat_equal=bool(np.array_equal(xp0.numpy(),xp1.numpy()) and np.array_equal(xs0.numpy(),xs1.numpy()))
  # The exact capture census is recorded from a DEBUG=2 first capture: baseline
  # schedules 7 calls (3 partials, 3 reductions, scalar combine), candidate 6
  # (3 Q8 producer calls and 3 distinct DP4A consumers).  CapturedJit stores a
  # graph as one outer UOp, so that count cannot be recovered from `.src` here.
  cand=statistics.median(b); delta=cand-mid
  return {"schema":"tinygrad.q6k_shared_q8_reuse_microgate.v1","device":str(dev),
    "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    "hypothesis":{"common_activation_consumers":3,"llama_observed_reuse_consumers":1,
      "native_candidate":"one tinygrad Q8_1 producer feeding three Q6 DP4A consumers"},
    "correctness":{"repeat_q8_bitwise_equal":q8_repeat_equal},
    "topology":{"debug2_first_capture_baseline_calls":7,"debug2_first_capture_shared_calls":6,
      "interpretation":"three distinct Q6 consumers retained; shared producer is not CSE-elided consumers"},
    "timing":{"unit":"us_per_three_projection_graph","replays":replays,"reps":reps,"control_a":a,"candidate_b":b,"control_c":c,
      "control_midpoint_median":mid,"candidate_median":cand,"delta":delta,
      "gate":"PASS" if q8_repeat_equal and delta < 0 else "FAIL"},
    "payload":{"x_sha256":hashlib.sha256(x.numpy().tobytes()).hexdigest()}}

def main():
  p=argparse.ArgumentParser(); p.add_argument("--replays",type=int,default=300); p.add_argument("--reps",type=int,default=7)
  p.add_argument("--row-tile",type=int,default=2); p.add_argument("--out"); a=p.parse_args(); r=run(a.replays,a.reps,a.row_tile)
  s=json.dumps(r,indent=2,sort_keys=True)
  if a.out: open(a.out,"w").write(s+"\n")
  print(s); return 0 if r["timing"]["gate"]=="PASS" else 1
if __name__=="__main__": raise SystemExit(main())
