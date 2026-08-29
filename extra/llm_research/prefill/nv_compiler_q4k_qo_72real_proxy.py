#!/usr/bin/env python3
"""Captured 72-real-weight Q/O proxy for compiler-owned Q4_K/Q8 IMMA."""
from __future__ import annotations

import argparse, json, pathlib, statistics, time
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.postrange import warmstart_candidate_state, warmstart_key
from tinygrad.uop.ops import Ops, UOp
from extra.llm_research.prefill.nv_compiler_q4k_production_gate import _activation_carrier, _weight_carrier
from extra.llm_research.prefill.nv_compiler_q4k_qo_gate import M,N,K,_context,_record


def _call_count(linear:UOp) -> int:
  total=0
  for call in linear.src:
    program=call.src[0]
    total += _call_count(program.src[0]) if program.op is Ops.CUSTOM_FUNCTION and program.arg=="graph" else 1
  return total


def main():
  ap=argparse.ArgumentParser();ap.add_argument("--model",default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--rounds",type=int,default=9);ap.add_argument("--out",required=True);args=ap.parse_args()
  from tinygrad.llm.generate import load_model_and_tokenizer
  model,_=load_model_and_tokenizer(args.model,4608,seed=20260617)
  linears=[p for block in model.blk for p in (block.attn_q,block.attn_output)]
  weights=[p.prefill_packed_weight().contiguous().realize() for p in linears]
  expected_words=N*(K//256)*36
  if len(weights)!=72 or any(w.dtype!=dtypes.uint32 or w.numel()!=expected_words for w in weights):
    raise RuntimeError("proxy requires 72 canonical Qwen3-8B Q/O Q4_K buffers")
  if any(getattr(p,"_pf16_w",None) is None for p in linears):raise RuntimeError("current FP16 comparator overlays unavailable")

  q8_np,scales_np,_,record_np=_record();record=Tensor(record_np,device="NV").contiguous().realize()
  # Reconstruct exactly representable fp16 values from the compact packet for
  # the matched resident-FP16 service comparator.
  x_np=(q8_np.astype(np.float32)*np.repeat(scales_np,32,axis=1)).astype(np.float16)
  x=Tensor(x_np,device="NV").contiguous().realize()
  wt,at,identity,context=_context();key=warmstart_key({M,N},K,wt.storage_dtype)

  @TinyJit
  def packed_batch(record_arg:Tensor):
    return tuple(_activation_carrier(record_arg,at).matmul(_weight_carrier(words,wt).transpose(),dtype=dtypes.int)
      .cast(dtypes.float).contiguous().realize() for words in weights)

  @TinyJit
  def fp16_batch(x_arg:Tensor):
    return tuple(x_arg.matmul(p._pf16_w.transpose()).contiguous().realize() for p in linears)

  def bench(jit,arg,ctx=None):
    samples=[]
    manager=warmstart_candidate_state({key:(Opt(OptOps.TC,0,(-1,2,1)),)},{key:context}) if ctx else \
      warmstart_candidate_state(model._pf16_warmstart or {},{})
    with manager:
      for iteration in range(args.rounds+3):
        Device["NV"].synchronize();started=time.perf_counter_ns();outputs=jit(arg);Device["NV"].synchronize()
        if iteration>=3:samples.append((time.perf_counter_ns()-started)/1e6)
    return outputs,samples

  packed,packed_times=bench(packed_batch,record,True)
  fp16,fp16_times=bench(fp16_batch,x,False)
  sample_indices=np.asarray([0,127,128,16383,M*N-1]); picks=(0,35,71)
  before=[packed[i].numpy().reshape(-1)[sample_indices].copy() for i in picks]
  with warmstart_candidate_state({key:(Opt(OptOps.TC,0,(-1,2,1)),)},{key:context}):packed=packed_batch(record)
  Device["NV"].synchronize();after=[packed[i].numpy().reshape(-1)[sample_indices] for i in picks]
  replay_exact=all(np.array_equal(a,b) for a,b in zip(before,after))
  sample_distinct=all(not np.array_equal(after[i],after[j]) for i in range(3) for j in range(i+1,3))
  packed_calls=_call_count(packed_batch.captured.linear) if packed_batch.captured is not None else 0
  fp16_calls=_call_count(fp16_batch.captured.linear) if fp16_batch.captured is not None else 0
  pmin,fmin=min(packed_times),min(fp16_times);pper=pmin*1000/72;fper=fmin*1000/72
  rec={"schema":"tinygrad.nv_compiler_q4k_qo_72real_proxy.v1","shape":{"M":M,"N":N,"K":K,"tile_k":64},
    "roles":{"q":36,"o":36},"real_weight_buffers":len(weights),"identity":identity,
    "captured_calls":{"packed":packed_calls,"fp16":fp16_calls},"sample_replay_exact":replay_exact,
    "sample_distinct_real_weights":sample_distinct,"sample_finite":bool(all(np.isfinite(v).all() for v in after)),
    "timing":{"packed_r9_ms":packed_times,"packed_min_ms":pmin,"packed_median_ms":statistics.median(packed_times),
      "packed_per_call_us":pper,"fp16_r9_ms":fp16_times,"fp16_min_ms":fmin,"fp16_median_ms":statistics.median(fp16_times),
      "fp16_per_call_us":fper,"packed_over_fp16":pmin/fmin,"llama_estimate_per_call_us":61.5,
      "packed_over_llama_estimate":pper/61.5},
    "representation":{"expanded_global_packed_weight":False,"partial_workspace_bytes":0,"fixup_calls":0}}
  rec["passed"]=bool(replay_exact and sample_distinct and rec["sample_finite"] and packed_calls==fp16_calls==72 and pmin<=fmin)
  path=pathlib.Path(args.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(rec,indent=2)+"\n")
  print(json.dumps(rec,sort_keys=True));
  if not rec["passed"]:raise SystemExit(1)

if __name__=="__main__":main()
