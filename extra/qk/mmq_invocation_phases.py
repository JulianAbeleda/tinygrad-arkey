#!/usr/bin/env python3
"""Randomized invocation-layer phase accounting for the bounded MMQ wrappers."""
from __future__ import annotations
import hashlib,json,random,statistics,time
from pathlib import Path
from typing import Any,Callable
import numpy as np
from tinygrad import Tensor,dtypes
from tinygrad.device import Device
from tinygrad.codegen import to_program
from extra.qk.mmq_bounded_harness import ACTIVATION_LAYOUT_MMQ_DS4,_finite_q4k_bytes,_q8_activation_inputs
from extra.qk.mmq_compile_evidence import build_mmq_sink,capture_loaded_mmq_program,compile_mmq_program
from extra.qk.mmq_experiment import canonical_candidate
from extra.qk.mmq_q4k_q8_atom import (_as_u32_words,_q4k_q8_1_bounded_ds4_coop_tile_kernel,
  _staged_ds4_lifecycle_for_spec)
from extra.qk.mmq_q4k_q8_reference import Q4KQ81MMQTileSpec,Q8_1_MMQ_DS4_LAYOUT

SCHEMA="tinygrad.mmq_invocation_phases.v1"

def _clock_overhead(samples=2000):
  values=[]
  for _ in range(samples):a=time.perf_counter_ns();b=time.perf_counter_ns();values.append(b-a)
  return statistics.median(values)

def _timed(fn:Callable[[],Any],overhead_ns:float):
  start=time.perf_counter_ns();value=fn();elapsed=time.perf_counter_ns()-start
  return value,elapsed/1e6,max(0.0,(elapsed-overhead_ns)/1e6)

def _one(mode,q4,ds4,program,runtime,prebuilt_sink,overhead_ns):
  phases={}
  def take(name,fn):
    value,raw,corrected=_timed(fn,overhead_ns);phases[name]={"raw_ms":raw,"corrected_ms":corrected};return value
  q4_view=take("numpy_validation_views",lambda:np.asarray(q4,dtype=np.uint8))
  words=take("q4_tensor_construct_realize_transfer",lambda:Tensor(_as_u32_words(q4_view),dtype=dtypes.uint32,device="AMD").realize())
  def ds(name,array,dtype,npdtype):return take(name,lambda:Tensor(np.ascontiguousarray(np.asarray(array,dtype=npdtype).reshape(-1)),dtype=dtype,device="AMD").realize())
  values=ds("ds4_values_construct_realize",ds4.values,dtypes.int8,np.int8)
  scales=ds("ds4_scales_construct_realize",ds4.scales,dtypes.float32,np.float32)
  sums=ds("ds4_sums_construct_realize",ds4.sums,dtypes.float32,np.float32)
  def alloc_out():
    tensor=Tensor.empty(16,16,dtype=dtypes.float32,device="AMD");tensor.uop.buffer.allocate();return tensor
  out=take("output_allocation",alloc_out)
  fxn=_q4k_q8_1_bounded_ds4_coop_tile_kernel(16,16,256,"ffn_gate_up",mode)
  lazy=take("custom_kernel_uop_construction",lambda:out.custom_kernel(words,values,scales,sums,fxn=fxn)[0])
  take("schedule_creation",lambda:Tensor.linear_with_vars(lazy))
  before=(program.src[3].arg,program.src[4].arg)
  cached=take("lowering_compile_cache_lookup",lambda:to_program(prebuilt_sink,Device["AMD"].renderer))
  if (cached.src[3].arg,cached.src[4].arg)!=before:raise RuntimeError("compile cache lookup changed program")
  buffers={0:out.uop.buffer._buf,1:words.uop.buffer._buf,2:values.uop.buffer._buf,3:scales.uop.buffer._buf,4:sums.uop.buffer._buf}
  args=tuple(buffers[i] for i in program.arg.globals)
  take("device_launch_sync",lambda:runtime(*args,global_size=program.arg.global_size,local_size=program.arg.local_size,wait=True))
  take("lifecycle_report_construction",lambda:_staged_ds4_lifecycle_for_spec(Q4KQ81MMQTileSpec(role="ffn_gate_up",m=16,n=16,k=256,m_tile=16,n_tile=16,activation_layout=Q8_1_MMQ_DS4_LAYOUT)))
  take("numpy_readback_cast",lambda:out.numpy().astype(np.float32))
  phases["total_corrected_ms"]={"corrected_ms":sum(v["corrected_ms"] for v in phases.values())}
  return phases

def run_invocation_phase_probe(*,rounds=30,warmups=3,seed=20260711,system_snapshot_id:str):
  if rounds<30:raise ValueError("phase probe requires at least 30 samples per mode")
  overhead=_clock_overhead();q4=_finite_q4k_bytes(16,256,seed);activation=_q8_activation_inputs(16,256,seed+1,ACTIVATION_LAYOUT_MMQ_DS4);ds4=activation.ds4_activation;assert ds4
  state={};
  for mode in ("gated_matrix_v0","direct_owner_v0"):
    spec=canonical_candidate(mode,seed=seed);prg=compile_mmq_program(spec)
    # Warm the exact wrapper/runtime once before randomized measurement.
    from extra.qk.mmq_q4k_q8_atom import run_q4k_q8_1_mmq_bounded_amd_ds4_coop_tile
    run_q4k_q8_1_mmq_bounded_amd_ds4_coop_tile(q4,ds4,role="ffn_gate_up",writeback_mode=mode)
    ev=capture_loaded_mmq_program(spec);from tinygrad.engine.realize import runtime_cache
    state[mode]=(prg,runtime_cache[(prg.key,"AMD")],ev.hashes["binary_sha256"],build_mmq_sink(spec))
  order=[m for m in state for _ in range(rounds)];random.Random(seed).shuffle(order);samples={m:[] for m in state}
  for mode in order:samples[mode].append(_one(mode,q4,ds4,state[mode][0],state[mode][1],state[mode][3],overhead))
  names=[k for k in samples["gated_matrix_v0"][0] if k!="total_corrected_ms"]
  summary={m:{name:statistics.median(x[name]["corrected_ms"] for x in rows) for name in names}|
             {"total_corrected_ms":statistics.median(x["total_corrected_ms"]["corrected_ms"] for x in rows)} for m,rows in samples.items()}
  return {"schema":SCHEMA,"system_snapshot_id":system_snapshot_id,"protocol":{"rounds":rounds,"warmups":warmups,"randomized_interleaved":True,"seed":seed},
    "timer":{"clock":"perf_counter_ns","median_empty_pair_ns":overhead,"subtraction":"each phase corrected=max(0,raw-empty_pair_median)"},
    "binary_sha256":{m:v[2] for m,v in state.items()},"samples":samples,"summary_median_ms":summary,"production_dispatch_changed":False}

def write_probe(result,path):Path(path).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
