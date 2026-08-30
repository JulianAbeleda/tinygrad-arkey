#!/usr/bin/env python3
"""Fresh-process whole-model gate for the graph-owned llama Q4_K gate/up pair."""
from __future__ import annotations

import argparse,json,os,pathlib,statistics,time
from collections import Counter
import numpy as np

from tinygrad import Device,Tensor,TinyJit
from extra.llm_research.prefill.nv_compiler_q4k_model_gate import _call_and_sync,_call_name,_compile_scope,_configure,_numpy_output,_program_calls


def main():
  ap=argparse.ArgumentParser();ap.add_argument("--arm",choices=("candidate","control"),required=True);ap.add_argument("--model",required=True)
  ap.add_argument("--out",required=True);ap.add_argument("--npz",required=True);ap.add_argument("--warmups",type=int,default=3);ap.add_argument("--rounds",type=int,default=9);a=ap.parse_args()
  enabled=os.environ.get("NV_LLAMA_PACKED_Q4K_PP512");promoted=os.environ.get("NV_LLAMA_FULL_PACKED_PP512","1")!="0"
  if a.arm=="candidate" and enabled!="1" and not promoted:raise SystemExit("candidate requires the promoted route or NV_LLAMA_PACKED_Q4K_PP512=1")
  if a.arm=="control" and (enabled is not None or promoted):raise SystemExit("control requires NV_LLAMA_FULL_PACKED_PP512=0 and NV_LLAMA_PACKED_Q4K_PP512 unset")
  from tinygrad.llm.generate import load_model_and_tokenizer
  model,_=load_model_and_tokenizer(a.model,4608,seed=20260617)
  final_row_prune=bool(os.environ.get("NV_LLAMA_FINAL_ROW_PRUNE_PP512"))
  if final_row_prune:model.blk[-1]._final_row_prune_requested_row=511
  last_down_name=type(model.blk[-1].ffn_down).__name__
  capture=q6_capture=q4_down_capture=qkv_capture=o_capture=compiler_o_capture=None
  if a.arm=="candidate":
    from extra.llm_research.prefill.nv_llama_packed_q4k_pp512_binding import binding_for
    asset=binding_for("NV");asset.prepare_pairs(36);capture=asset.new_capture()
    if promoted or os.environ.get("NV_LLAMA_PACKED_Q6K_DOWN_PP512"):
      from extra.llm_research.prefill.nv_llama_packed_q6k_down_pp512_binding import binding_for as q6_binding_for
      q6_asset=q6_binding_for("NV");q6_asset.prepare_records(18);q6_capture=q6_asset.new_capture()
    if promoted or os.environ.get("NV_LLAMA_PACKED_Q4K_DOWN_PP512"):
      from extra.llm_research.prefill.nv_llama_packed_q4k_down_pp512_binding import binding_for as q4_down_binding_for
      q4_asset=q4_down_binding_for("NV");q4_asset.prepare_records(18);q4_down_capture=q4_asset.new_capture()
    if promoted or os.environ.get("NV_LLAMA_PACKED_QKV_PP512"):
      from extra.llm_research.prefill.nv_qkv_packed_pp512_binding import binding_for as qkv_binding_for
      qkv_capture=qkv_binding_for("NV").new_capture()
    if os.environ.get("NV_COMPILER_Q4_IMMA_O_PP512"):
      from extra.llm_research.prefill.nv_compiler_q4k_qo_binding import binding_for as compiler_o_binding_for
      compiler_o_asset=compiler_o_binding_for("NV");compiler_o_asset.prepare(36);compiler_o_capture=compiler_o_asset.new_capture()
    elif promoted or os.environ.get("NV_LLAMA_PACKED_O_PP512"):
      from extra.llm_research.prefill.nv_llama_packed_q4k_o_pp512_binding import binding_for as o_binding_for
      o_asset=o_binding_for("NV");o_asset.prepare_records(36);o_capture=o_asset.new_capture()
  _configure(model,capture)
  if q6_capture is not None:
    for block in model.blk:block._nv_llama_packed_q6k_down_pp512_binding=q6_capture
  if q4_down_capture is not None:
    for block in model.blk:block._nv_llama_packed_q4k_down_pp512_binding=q4_down_capture
  if qkv_capture is not None:
    for block in model.blk:block._nv_qkv_packed_pp512_binding=qkv_capture
  if o_capture is not None:
    for block in model.blk:block._nv_llama_packed_o_pp512_binding=o_capture
  if compiler_o_capture is not None:
    for block in model.blk:block._nv_compiler_q4k_o_pp512_binding=compiler_o_capture
  @TinyJit
  def run(tokens,temp):return model.forward_greedy_with_logits(tokens,0,temp)
  chunk_a=Tensor([[(i*7)%1000 for i in range(512)]],dtype="int32").contiguous();chunk_b=Tensor([[(i*11+3)%1000 for i in range(512)]],dtype="int32").contiguous();temp=Tensor([0.0])
  with _compile_scope(model):
    for _ in range(3):
      if capture is not None:capture.begin_trace()
      if q6_capture is not None:q6_capture.begin_trace()
      if q4_down_capture is not None:q4_down_capture.begin_trace()
      if qkv_capture is not None:qkv_capture.begin_trace()
      if o_capture is not None:o_capture.begin_trace()
      if compiler_o_capture is not None:compiler_o_capture.begin_trace()
      _call_and_sync(run,chunk_a,temp)
  a0=_numpy_output(_call_and_sync(run,chunk_a,temp));b=_numpy_output(_call_and_sync(run,chunk_b,temp));a1=_numpy_output(_call_and_sync(run,chunk_a,temp))
  for _ in range(a.warmups):_call_and_sync(run,chunk_a,temp)
  samples=[]
  for _ in range(a.rounds):
    Device[Device.DEFAULT].synchronize();st=time.perf_counter_ns();_call_and_sync(run,chunk_a,temp);samples.append((time.perf_counter_ns()-st)/1e6)
  calls=_program_calls(run.captured.linear);names=Counter(_call_name(call) for call in calls)
  q6=promoted or bool(os.environ.get("NV_LLAMA_PACKED_Q6K_DOWN_PP512"));q4d=promoted or bool(os.environ.get("NV_LLAMA_PACKED_Q4K_DOWN_PP512"))
  qkv=promoted or bool(os.environ.get("NV_LLAMA_PACKED_QKV_PP512"));oproj=promoted or bool(os.environ.get("NV_LLAMA_PACKED_O_PP512"))
  gate_epi=bool(os.environ.get("NV_LLAMA_PACKED_GATE_UP_EPILOGUE_PP512"))
  compiler_o=bool(os.environ.get("NV_COMPILER_Q4_IMMA_O_PP512"))
  flash_mma=os.environ.get("NV_LLAMA_FATTN_MMA_PP512",os.environ.get("NV_LLAMA_FULL_PACKED_PP512","1")) not in ("", "0")
  q4_main_total=sum(v for k,v in names.items() if k.startswith("_Z15dense_mul_mat_qIL9ggml_type12"))
  q4_fixup_total=sum(v for k,v in names.items() if k.startswith("_Z30dense_mul_mat_q_stream_k_fixupIL9ggml_type12"))
  census={"producer":names.get("q8_ds4_fp16_pp512",0),
          "q4_main_total":q4_main_total,"q4_fixup_total":q4_fixup_total,
          "q6_down_producer":names.get("q8_d4_fp16_down_pp512",0),
          "q6_main_total":sum(v for k,v in names.items() if k.startswith("_Z15dense_mul_mat_qIL9ggml_type14")),
          "q6_fixup_total":sum(v for k,v in names.items() if k.startswith("_Z30dense_mul_mat_q_stream_k_fixupIL9ggml_type14")),
          "q4_down_producer":names.get("q8_ds4_fp16_down_pp512",0),
          "qkv_ds4_producer":names.get("q8_ds4_fp16_qkv_pp512",0),
          "qkv_d4_producer":names.get("q8_d4_fp16_qkv_pp512",0),
          "o_producer":names.get("q8_ds4_fp16_o_pp512",0),
          "compiler_o_producer":names.get("q8_compact_record_fp16",0),
          "gate_up_epilogue":names.get("nv_llama_gate_up_silu_mul_fp16",0),
          "flash_mma_pp512":names.get("nv_llama_fattn_mma_pp512",0)}
  replay={"finite":bool(np.isfinite(a1[1]).all()),"same_activation_exact":bool(a0[0]==a1[0] and np.array_equal(a0[1],a1[1])),
          "distinct_activation_output":bool(a1[0]!=b[0] or not np.array_equal(a1[1],b[1]))}
  last_q6=final_row_prune and last_down_name=="Q6KPrimitiveLinear";last_q4=final_row_prune and last_down_name=="Q4KPrimitiveLinear"
  expected=(36-int(final_row_prune),72+(18 if q4d else 0)+(90 if qkv else 0)+(36 if oproj and not compiler_o else 0)-2*int(final_row_prune)-int(last_q4),
    72+(18 if q4d else 0)+(90 if qkv else 0)+(36 if oproj and not compiler_o else 0)-2*int(final_row_prune)-int(last_q4),
    (18 if q6 else 0)-int(last_q6),(18 if q6 else 0)+(18 if qkv else 0)-int(last_q6),
    (18 if q6 else 0)+(18 if qkv else 0)-int(last_q6),(18 if q4d else 0)-int(last_q4),
    36 if qkv else 0,18 if qkv else 0,36 if oproj and not compiler_o else 0,36 if compiler_o else 0,36 if gate_epi else 0,
    36 if flash_mma else 0) if a.arm=="candidate" else (0,)*13
  passed=all(replay.values()) and tuple(census.values())==expected
  payload={"schema":"tinygrad.nv_llama_packed_q4k_model_arm.v1","arm":a.arm,"status":"PASS" if passed else "FAIL","token":a1[0],
           "final_row_prune":{"enabled":final_row_prune,"requested_row":511 if final_row_prune else None,"last_down_type":last_down_name},
           "replay":replay,"census":census,"wall":{"samples_ms":samples,"min_ms":min(samples),"median_ms":statistics.median(samples),"tok_s":512/min(samples)*1000}}
  path=pathlib.Path(a.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,indent=2)+"\n")
  np.savez(a.npz,token=np.int64(a1[0]),logits=a1[1]);print(json.dumps(payload,indent=2))
  if not passed:raise SystemExit(1)


if __name__=="__main__":main()
