#!/usr/bin/env python3
"""Deferred included-cost P3b primitive microgate (never a route selector).

Run only under the shared GPU lock after the root scheduler releases it.  The
control is the landed scalar W1/W3 producer, materialized fp16 boundary, and
ordinary Q4 FFN-down; candidate is the cooperative packet producer plus its
Q4/Q8 down consumer.  Timing returns one scalar sink so all boundary cost is
inside the bracket.
"""
from __future__ import annotations
import argparse, json, statistics, time
import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.engine.jit import GraphAdmissionCensus, observe_graph_admissions
from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_kernel, q4k_g3_lanemap_gemv_w1w3_kernel
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, execute_research_program
from extra.llm_research.decode.ffn_q8_cooperative_producer import (
  DOWN_K, DOWN_ROWS, K, ROWS, Q8_METADATA_BASE, emit_ffn_w1w3_q8_cooperative, emit_q4_q8_ffn_down_consumer,
  pack_q8_1_private, q4_q8_ffn_down_row_reference, q4_w1w3_packet_reference)
from extra.llm_research.decode.route_class_numerics import _make_q4k_words


TOPOLOGY={"control_programs_min":3,"candidate_programs":2,"strictly_smaller":True,
          "control":["landed_w1w3","fp16_boundary","q4_ffn_down"],
          "candidate":["cooperative_w1w3_q8_provider","q4_q8_ffn_down_consumer"]}
NUMERICAL_CONTRACT={"kind":"shared_q8_semantic_not_bitwise", "provider_round_point":"fp32 GLU -> fp16 -> Q8_1",
  "required_before_wall":["finite candidate output","independent packed-ABI oracle", "full-model tokens/argmax/top-k semantic gate"],
  "forbidden_claim":"candidate equals fp16 control bitwise"}

def _p(name, fn): return KernelProgram("research.nv_ffn_q8_cooperative",name,KernelProgramProvenance.RESEARCH_ONLY,fn)

def _raw_inputs():
  # Exact target shapes; deterministic synthetic quantized storage is enough
  # for an included-cost primitive construction gate.
  gw,_=_make_q4k_words(ROWS,K,202608051); uw,_=_make_q4k_words(ROWS,K,202608052); dw,_=_make_q4k_words(DOWN_ROWS,DOWN_K,202608053)
  x=np.random.default_rng(20260805).normal(0,.2,K).astype(np.float16)
  return gw,uw,dw,x

def _inputs(device,raw=None):
  gw,uw,dw,x=_raw_inputs() if raw is None else raw
  return tuple(Tensor(a,dtype=dtypes.uint32,device=device).contiguous().realize() for a in (gw,uw,dw)) + \
    (Tensor(x,dtype=dtypes.float16,device=device).contiguous().realize(),)

def build(device):
  scalar=_p("landed_w1w3",q4k_g3_lanemap_gemv_w1w3_kernel(ROWS,K,"scalar"))
  down=_p("landed_q4_down",q4k_g3_lanemap_gemv_kernel(DOWN_ROWS,DOWN_K))
  provider=_p("cooperative_q8_provider",emit_ffn_w1w3_q8_cooperative())
  consumer=_p("cooperative_q4_q8_down",emit_q4_q8_ffn_down_consumer())
  @TinyJit
  def control(gw,uw,dw,x):
    z=execute_research_program(Tensor.empty(ROWS,dtype=dtypes.float32,device=device),gw,uw,x,program=scalar)
    # This cast is intentionally explicit: it is the physical baseline
    # boundary P3b seeks to remove, and it must remain in timed control.
    z16=z.cast(dtypes.float16).contiguous()
    return execute_research_program(Tensor.empty(DOWN_ROWS,dtype=dtypes.float32,device=device),dw,z16,program=down).sum()
  @TinyJit
  def candidate(gw,uw,dw,x):
    xp=execute_research_program(Tensor.empty(3456,dtype=dtypes.uint32,device=device),gw,uw,x,program=provider)
    return execute_research_program(Tensor.empty(DOWN_ROWS,dtype=dtypes.float32,device=device),dw,xp,program=consumer).sum()
  return control,candidate

def census(arm:str):
  dev=Device.DEFAULT
  if not str(dev).startswith("NV"): raise RuntimeError(f"native NV required, got {dev}")
  args=_inputs(dev); control,candidate=build(dev); fn={"control":control,"candidate":candidate}[arm]
  fn(*args).realize(); Device[dev].synchronize()
  census=GraphAdmissionCensus()
  with observe_graph_admissions(census): fn(*args).realize()
  names=[x.program_name for x in census.records if x.program_name]
  # Both arms return the same scalar timing sink. It is outside the replaced
  # FFN chain and is retained in raw_program_count as a positive control.
  chain=[name for name in names if not name.startswith("r_16_256_")]
  return {"schema":"tinygrad.nv_ffn_q8_cooperative_microgate.v1","mode":"census","arm":arm,
    "topology":TOPOLOGY,"numerical_contract":NUMERICAL_CONTRACT,"program_names":names,"raw_program_count":len(names),
    "chain_program_names":chain,"chain_program_count":len(chain),
    "topology_gate":"PASS" if len(chain) == (3 if arm == "control" else 2) else "FAIL"}

def timing(replays:int,reps:int):
  dev=Device.DEFAULT
  if not str(dev).startswith("NV"): raise RuntimeError(f"native NV required, got {dev}")
  args=_inputs(dev); control,candidate=build(dev)
  for _ in range(30): control(*args).realize(); candidate(*args).realize()
  def samples(fn):
    out=[]
    for _ in range(reps):
      Device[dev].synchronize(); start=time.perf_counter_ns()
      for _ in range(replays): fn(*args).realize()
      Device[dev].synchronize(); out.append((time.perf_counter_ns()-start)/replays/1e3)
    return out
  a,b,c=samples(control),samples(candidate),samples(control); mid=(statistics.median(a)+statistics.median(c))/2
  return {"schema":"tinygrad.nv_ffn_q8_cooperative_microgate.v1","mode":"timing","topology":TOPOLOGY,
    "numerical_contract":NUMERICAL_CONTRACT,"unit":"us_per_graph_replay","control_a":a,"candidate_b":b,"control_c":c,
    "control_midpoint":mid,"candidate_median":statistics.median(b),"delta_us":statistics.median(b)-mid}

def correctness():
  dev=Device.DEFAULT
  if not str(dev).startswith("NV"): raise RuntimeError(f"native NV required, got {dev}")
  raw=_raw_inputs(); gw,uw,dw,x=_inputs(dev,raw)
  scalar=_p("landed_w1w3",q4k_g3_lanemap_gemv_w1w3_kernel(ROWS,K,"scalar"))
  down=_p("landed_q4_down",q4k_g3_lanemap_gemv_kernel(DOWN_ROWS,DOWN_K))
  provider=_p("cooperative_q8_provider",emit_ffn_w1w3_q8_cooperative())
  consumer=_p("cooperative_q4_q8_down",emit_q4_q8_ffn_down_consumer())
  z=execute_research_program(Tensor.empty(ROWS,dtype=dtypes.float32,device=dev),gw,uw,x,program=scalar).realize()
  base=execute_research_program(Tensor.empty(DOWN_ROWS,dtype=dtypes.float32,device=dev),dw,z.cast(dtypes.float16).contiguous(),program=down).realize()
  xp=execute_research_program(Tensor.empty(3456,dtype=dtypes.uint32,device=dev),gw,uw,x,program=provider).realize()
  cand=execute_research_program(Tensor.empty(DOWN_ROWS,dtype=dtypes.float32,device=dev),dw,xp,program=consumer).realize()
  Device[dev].synchronize(); base_np,cand_np,xp_np=base.numpy(),cand.numpy(),xp.numpy()
  provider_rows=[]
  for packet in (0,191,383):
    oracle_values=q4_w1w3_packet_reference(raw[0],raw[1],raw[3],packet)
    oracle=pack_q8_1_private(np.pad(oracle_values,(packet*32,ROWS-(packet+1)*32)))
    got_payload=xp_np[packet*8:packet*8+8]; exp_payload=oracle[packet*8:packet*8+8]
    got_meta=xp_np[Q8_METADATA_BASE+packet]; exp_meta=oracle[Q8_METADATA_BASE+packet]
    provider_rows.append({"packet":packet,"payload_equal":bool(np.array_equal(got_payload,exp_payload)),
      "d_equal":bool((got_meta&0xffff)==(exp_meta&0xffff)),"s_equal":bool((got_meta>>16)==(exp_meta>>16)),
      "payload_mismatch_bytes":int(np.sum(got_payload.view(np.uint8)!=exp_payload.view(np.uint8)))})
  consumer_rows=[]
  for row in (0,1,2047,4095):
    ref=float(q4_q8_ffn_down_row_reference(raw[2],xp_np,row)); got=float(cand_np[row]); err=abs(got-ref)
    consumer_rows.append({"row":row,"got":got,"oracle":ref,"abs_error":err,"relative_error":err/max(abs(ref),1e-12)})
  delta=(cand_np-base_np).astype(np.float64); rel_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(base_np.astype(np.float64)),1e-30))
  consumer_pass=all(x["relative_error"] <= 2e-5 or x["abs_error"] <= 2e-4 for x in consumer_rows)
  provider_pass=all(x["payload_equal"] and x["d_equal"] and x["s_equal"] for x in provider_rows)
  gate=bool(np.isfinite(cand_np).all() and provider_pass and consumer_pass)
  return {"schema":"tinygrad.nv_ffn_q8_cooperative_microgate.v1","mode":"correctness","topology":TOPOLOGY,
    "numerical_contract":NUMERICAL_CONTRACT,"finite":bool(np.isfinite(cand_np).all()),"provider_oracle":provider_rows,
    "consumer_oracle":consumer_rows,"provider_oracle_pass":provider_pass,"consumer_oracle_pass":consumer_pass,
    "vs_fp16_control":{"relative_l2":rel_l2,"max_abs":float(np.max(np.abs(delta)))},"gate":"PASS" if gate else "FAIL"}

if __name__ == "__main__":
  ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=("census","correctness","timing"),required=True); ap.add_argument("--arm",choices=("control","candidate"))
  ap.add_argument("--replays",type=int,default=100); ap.add_argument("--reps",type=int,default=5); ap.add_argument("--out")
  ns=ap.parse_args(); result=census(ns.arm) if ns.mode == "census" else correctness() if ns.mode == "correctness" else timing(ns.replays,ns.reps)
  print(json.dumps(result,indent=2));
  if ns.out: open(ns.out,"w").write(json.dumps(result,indent=2)+"\n")
