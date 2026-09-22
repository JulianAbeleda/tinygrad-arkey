#!/usr/bin/env python3
"""Included-cost native-NV gate for Q4_K 4096x12288 Q8_1/four-warp DP4A.

The control is the installed fp16-input Q4 FFN-down primitive.  The candidate
starts from the same already materialized fp16 activation and includes both
Q8_1 quantization and a direct-output four-warp Q4 consumer.  W1/W3 production
is intentionally outside both arms.
"""
from __future__ import annotations

import argparse, ctypes, hashlib, json, pathlib, statistics, subprocess, time
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.device import Buffer
from tinygrad.engine.realize import time_call
from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_kernel
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, execute_research_program
from tinygrad.runtime.autogen import cuda
from tinygrad.runtime.ops_cuda import check
from tinygrad.uop.ops import UOp
from extra.llm_research.decode.ffn_q8_cooperative_producer import (
  llama_q8_1_oracle_private, pack_q8_1_private, q4_q8_ffn_down_row_reference)
from extra.llm_research.decode.q4k_ffn_down_mmvq import (
  BLOCKS_PER_WARP,K,Q8_PAYLOAD_WORDS,Q8_WORDS,ROWS,emit_four_warp_direct,emit_q8_provider)
from extra.llm_research.decode.route_class_numerics import _make_q4k_words
from scratchpad.llama_cuda_quantized_live_oracle import ENTRY_Q8, device_pointer, fastdiv_values


MATERIAL_FRACTION, MATERIAL_ABS_US = 0.05, 1.0
LLAMA_Q8_CUBIN=pathlib.Path("/tmp/llama-oracle-cubins/libggml-cuda.so.0.14.44.sm_120a.cubin")


def _program(name,emitter):
  return KernelProgram("research.q4k_ffn_down_mmvq",name,KernelProgramProvenance.RESEARCH_ONLY,emitter)


def _inputs(device):
  words,raw=_make_q4k_words(ROWS,K,202608056)
  x=np.random.default_rng(202608057).normal(0,.2,K).astype(np.float16)
  return (Tensor(words,dtype=dtypes.uint32,device=device).contiguous().realize(),
          Tensor(x,dtype=dtypes.float16,device=device).contiguous().realize(),words,raw,x)


def _llama_cuda_q8_oracle_private(values:np.ndarray) -> np.ndarray:
  """Run the pinned live llama CUDA Q8_1 producer and transpose only storage.

  This is the authoritative provider oracle. The CPU reference is retained as
  a diagnostic because it computes ``x*(1/d)`` and an integer-derived ``s``;
  live CUDA computes ``x/d`` and a warp-reduced raw-input ``s``.
  """
  if not LLAMA_Q8_CUBIN.is_file(): raise FileNotFoundError(f"missing pinned llama Q8 cubin: {LLAMA_Q8_CUBIN}")
  source=np.ascontiguousarray(np.asarray(values,dtype=np.float16).astype(np.float32))
  src=Buffer("CUDA",K,dtypes.float32,initial_value=bytearray(source.tobytes()))
  raw_bytes=(K//32)*36
  dst=Buffer("CUDA",raw_bytes,dtypes.uint8,preallocate=True)
  Device["CUDA"].synchronize()
  module,function,stream=cuda.CUmodule(),cuda.CUfunction(),cuda.CUstream()
  try:
    check(cuda.cuModuleLoad(ctypes.byref(module),str(LLAMA_Q8_CUBIN).encode()))
    check(cuda.cuModuleGetFunction(ctypes.byref(function),module,ENTRY_Q8.encode()))
    check(cuda.cuStreamCreate(ctypes.byref(stream),cuda.CU_STREAM_NON_BLOCKING))
    args=[device_pointer(src),device_pointer(dst),ctypes.c_int64(K),ctypes.c_int64(K),ctypes.c_int64(K),
          ctypes.c_int64(K),ctypes.c_int64(K),ctypes.c_uint32(1),fastdiv_values(1)]
    params=(ctypes.c_void_p*len(args))(*[ctypes.cast(ctypes.pointer(arg),ctypes.c_void_p) for arg in args])
    check(cuda.cuLaunchKernel(function,(K+255)//256,1,1,256,1,1,0,stream,params,None))
    check(cuda.cuStreamSynchronize(stream))
    raw=bytearray(raw_bytes); dst.copyout(memoryview(raw))
  finally:
    if stream: cuda.cuStreamDestroy_v2(stream)
    if module: cuda.cuModuleUnload(module)
  blocks=np.frombuffer(raw,dtype=np.uint8).reshape(-1,36)
  out=np.zeros(Q8_WORDS,dtype=np.uint32)
  payload=blocks[:,4:].reshape(-1,4).astype(np.uint32)
  out[:Q8_PAYLOAD_WORDS]=payload@np.array([1,256,65536,16777216],dtype=np.uint32)
  d=blocks[:,:2].copy().reshape(-1).view(np.uint16).astype(np.uint32)
  s=blocks[:,2:4].copy().reshape(-1).view(np.uint16).astype(np.uint32)
  out[Q8_PAYLOAD_WORDS:]=d|(s<<np.uint32(16))
  return out


def build(device):
  control_program=_program("installed_q4_ffn_down",q4k_g3_lanemap_gemv_kernel(ROWS,K))
  provider_program=_program("q8_provider",emit_q8_provider())
  block_var=UOp.variable("q4k_ffn_down_blocks",1,BLOCKS_PER_WARP)
  bound_blocks=block_var.bind(BLOCKS_PER_WARP)

  @TinyJit
  def control(words,x):
    return execute_research_program(Tensor.empty((ROWS,),dtype=dtypes.float32,device=device),words,x,program=control_program)

  @TinyJit
  def candidate(words,x,blocks):
    packed=execute_research_program(Tensor.empty((Q8_WORDS,),dtype=dtypes.uint32,device=device),x,program=provider_program)
    # Carry the existing bound graph scalar without adding a provider kernel.
    packed=Tensor(packed.uop.after(blocks))
    consumer_program=_program("q4_mmvq_direct",emit_four_warp_direct(blocks.src[0]))
    return execute_research_program(Tensor.empty((ROWS,),dtype=dtypes.float32,device=device),words,packed,program=consumer_program)
  return control,candidate,bound_blocks


def _correctness(device,words_t,x_t,words,x):
  control,candidate,bound_blocks=build(device)
  baseline=control(words_t,x_t).realize()
  result=candidate(words_t,x_t,bound_blocks).realize()
  # Materialize the provider independently so its complete ABI can be checked.
  provider=_program("q8_provider_oracle",emit_q8_provider())
  packed_t=execute_research_program(Tensor.empty((Q8_WORDS,),dtype=dtypes.uint32,device=device),x_t,program=provider).realize()
  Device[device].synchronize()
  baseline_np,result_np,packed=baseline.numpy(),result.numpy(),packed_t.numpy()

  live=_llama_cuda_q8_oracle_private(x)
  private=pack_q8_1_private(x)
  external=llama_q8_1_oracle_private(x)
  live_equal=bool(np.array_equal(packed,live))
  cpu_payload_mismatches=int(np.count_nonzero(packed[:Q8_PAYLOAD_WORDS] != external[:Q8_PAYLOAD_WORDS]))
  cpu_d_equal=bool(np.array_equal((packed[Q8_PAYLOAD_WORDS:]&0xffff),(external[Q8_PAYLOAD_WORDS:]&0xffff)))
  private_mismatches=int(np.count_nonzero(packed != private))
  rows=[]
  for row in (0,1,2047,4095):
    reference=float(q4_q8_ffn_down_row_reference(words,packed,row))
    got=float(result_np[row]); error=abs(got-reference)
    rows.append({"row":row,"got":got,"oracle":reference,"abs_error":error,
      "relative_error":error/max(abs(reference),1e-12)})
  algebra_pass=all(row["relative_error"] <= 2e-5 or row["abs_error"] <= 2e-4 for row in rows)
  relative_l2=float(np.linalg.norm((result_np-baseline_np).astype(np.float64))/
                    max(np.linalg.norm(baseline_np.astype(np.float64)),1e-30))
  passed=bool(np.isfinite(result_np).all() and live_equal and algebra_pass)
  return {"pass":passed,"finite":bool(np.isfinite(result_np).all()),
    "provider":{"complete_equal_live_llama_cuda":live_equal,"live_llama_q8_cubin":str(LLAMA_Q8_CUBIN),
      "cpu_reference_payload_word_mismatches":cpu_payload_mismatches,"cpu_reference_d_equal":cpu_d_equal,
      "private_reciprocal_reference_word_mismatches":private_mismatches},
    "independent_q4_times_q8_rows":rows,"independent_algebra_pass":algebra_pass,
    "q8_vs_fp16_control":{"relative_l2":relative_l2,"max_abs":float(np.max(np.abs(result_np-baseline_np)))}}


def _device_event_decomposition(device,words_t,x_t,reps:int=11):
  """Measure concrete kernel calls with device completion timing, not host replay wall."""
  control_program=_program("installed_q4_ffn_down_event",q4k_g3_lanemap_gemv_kernel(ROWS,K))
  control=execute_research_program(Tensor.empty((ROWS,),dtype=dtypes.float32,device=device),words_t,x_t,program=control_program)
  provider=execute_research_program(Tensor.empty((Q8_WORDS,),dtype=dtypes.uint32,device=device),x_t,
    program=_program("q8_provider_event",emit_q8_provider()))
  blocks=UOp.variable("q4k_ffn_down_event_blocks",1,BLOCKS_PER_WARP).bind(BLOCKS_PER_WARP)
  provider=Tensor(provider.uop.after(blocks))
  candidate=execute_research_program(Tensor.empty((ROWS,),dtype=dtypes.float32,device=device),words_t,provider,
    program=_program("q4_mmvq_direct_event",emit_four_warp_direct(blocks.src[0])))
  rows={}
  for arm,out in (("control",control),("candidate",candidate)):
    linear,var_vals=out.linear_with_vars()
    for index,call in enumerate(linear.src):
      name=getattr(getattr(call.src[0],"arg",None),"name",None) or f"call_{index}"
      # One untimed execution initializes the output/intermediate and proves
      # the isolated call can be launched with its concrete graph buffers.
      time_call(call,var_vals)
      values=[time_call(call,var_vals)*1e6 for _ in range(reps)]
      rows[f"{arm}:{name}"]={"samples_us":values,"median_us":statistics.median(values)}
  control_rows=[v["median_us"] for k,v in rows.items() if k.startswith("control:")]
  candidate_rows=[v["median_us"] for k,v in rows.items() if k.startswith("candidate:")]
  if len(control_rows) != 1 or len(candidate_rows) != 2:
    raise RuntimeError(f"unexpected device-event topology: {list(rows)}")
  control_us=control_rows[0]; candidate_us=sum(candidate_rows)
  return {"unit":"device_event_us_per_kernel_launch","reps":reps,"kernels":rows,
    "control_sum_us":control_us,"candidate_included_sum_us":candidate_us,
    "delta_us":candidate_us-control_us,"saving_fraction":(control_us-candidate_us)/control_us,
    "topology_pass":True,"expected":{"control_kernels":1,"candidate_kernels":2,"adapter_kernels":0}}


def run(replays:int=200,reps:int=7):
  device=Device.DEFAULT
  if not str(device).startswith("NV"): raise RuntimeError(f"native NV required, got {device}")
  words_t,x_t,words,raw,x=_inputs(device)
  correctness=_correctness(device,words_t,x_t,words,x)
  if not correctness["pass"]: raise RuntimeError(f"correctness gate failed: {correctness}")
  device_events=_device_event_decomposition(device,words_t,x_t)

  control,candidate,bound_blocks=build(device)
  control(words_t,x_t).realize(); candidate(words_t,x_t,bound_blocks).realize(); Device[device].synchronize()
  for _ in range(100): control(words_t,x_t).realize(); candidate(words_t,x_t,bound_blocks).realize()

  def samples(fn):
    values=[]
    for _ in range(reps):
      Device[device].synchronize(); start=time.perf_counter_ns()
      for _ in range(replays): fn().realize()
      Device[device].synchronize(); values.append((time.perf_counter_ns()-start)/replays/1e3)
    return values
  arm_a=samples(lambda:control(words_t,x_t))
  arm_b=samples(lambda:candidate(words_t,x_t,bound_blocks))
  arm_c=samples(lambda:control(words_t,x_t))
  midpoint=(statistics.median(arm_a)+statistics.median(arm_c))/2
  candidate_median=statistics.median(arm_b); delta=candidate_median-midpoint
  fraction=-delta/midpoint
  material=bool(delta <= -MATERIAL_ABS_US and fraction >= MATERIAL_FRACTION)
  full_logits={"status":"DECLARED_NOT_RUN" if material else "NOT_ADMITTED_NO_PRIMITIVE_WIN",
    "required":["all candidate logits finite","argmax exact","top10 set exact",
      "relative_l2 <= 1e-3","maximum perturbation below the control top1-top2 margin","generated token stream exact"],
    "route_scope":"default-off one Q4_K FFN-down role first; no all-18 rollout before one-role pass"}
  return {"schema":"tinygrad.q4k_ffn_down_mmvq_microgate.v1",
    "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    "shape":{"rows":ROWS,"k":K,"local":128,"warps_per_row":4,"runtime_blocks_per_warp":BLOCKS_PER_WARP},
    "payload":{"q4_sha256":hashlib.sha256(raw.reshape(-1).tobytes()).hexdigest(),
      "x_sha256":hashlib.sha256(x.tobytes()).hexdigest()},
    "boundary":{"control":["installed q4k fp16-input direct output"],
      "candidate":["q8_1_llama_provider_12288","q4k_q8_mmvq_direct_4096_12288"],
      "excluded":"W1/W3 production is identical and outside both arms"},
    "correctness":correctness,"device_event_decomposition":device_events,
    "timing":{"unit":"us_per_graph_replay","replays":replays,"reps":reps,
      "control_a":arm_a,"candidate_b":arm_b,"control_c":arm_c,"control_midpoint_median":midpoint,
      "candidate_median":candidate_median,"delta_us":delta,"saving_fraction":fraction,
      "material_gate":{"minimum_fraction":MATERIAL_FRACTION,"minimum_absolute_us":MATERIAL_ABS_US,
        "pass":material}},
    "full_logit_contract":full_logits,
    "verdict":"MATERIAL_PRIMITIVE_WIN" if material else "NO_GO_INCLUDED_COST"}


if __name__ == "__main__":
  parser=argparse.ArgumentParser(); parser.add_argument("--replays",type=int,default=200); parser.add_argument("--reps",type=int,default=7)
  parser.add_argument("--out"); args=parser.parse_args(); report=run(args.replays,args.reps)
  print(json.dumps(report,indent=2))
  if args.out: open(args.out,"w").write(json.dumps(report,indent=2)+"\n")
