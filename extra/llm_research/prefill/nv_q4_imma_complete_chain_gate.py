#!/usr/bin/env python3
"""Fail-closed complete-chain qualification gate for a future packed Q4 IMMA provider.

The fixture and Q8 producer execute today.  A provider is admitted only through
an explicit JSON manifest matching ABI v1; absent/incomplete providers are
reported as BLOCKED and are never silently replaced by expanded weights.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, statistics, time
import numpy as np

M,N,K=512,12288,4096
ABI="tinygrad.nv_q4_imma_chain_abi.v1"

def sha(a): return hashlib.sha256(a.tobytes()).hexdigest()
def timing(fn, rounds):
 from tinygrad import Device
 xs=[]
 for _ in range(rounds):
  Device["NV"].synchronize();t=time.perf_counter_ns();fn();Device["NV"].synchronize();xs.append((time.perf_counter_ns()-t)/1e6)
 return {"n":len(xs),"min_ms":min(xs),"median_ms":statistics.median(xs),"max_ms":max(xs),"samples_ms":xs}

def validate_provider(path):
 p=json.loads(pathlib.Path(path).read_text())
 required={"schema","symbol","artifact","args","grid","block","shared_bytes","writes_partials","fixup_symbol","output_dtype"}
 missing=sorted(required-p.keys())
 if missing: raise ValueError(f"provider manifest missing {missing}")
 if p["schema"]!=ABI: raise ValueError("provider schema mismatch")
 if p["args"]!=["q4_words_u32","q8_values_i8","q8_scales_f32","q8_sums_f32","output_or_partials_f32","M_i32","N_i32","K_i32"]: raise ValueError("provider argument ABI mismatch")
 if p["output_dtype"]!="float32" or not p["writes_partials"] or not p["fixup_symbol"]: raise ValueError("complete main+fixup lifecycle required")
 if not pathlib.Path(p["artifact"]).is_file(): raise ValueError("provider artifact missing")
 return p

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',default='/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf');ap.add_argument('--rounds',type=int,default=9);ap.add_argument('--provider-manifest',default='');ap.add_argument('--out',required=True);a=ap.parse_args()
 from tinygrad import Tensor,Device,TinyJit
 from tinygrad.llm.generate import load_model_and_tokenizer
 from extra.llm_research.layout import read_metadata,packed_u32_slice,GGML_Q4_K
 from extra.llm_research.mmq_ds4_logical_emitter import packed_row_major_candidate,pack_q8_1_mmq_fused
 from extra.llm_research.layout import q8_1_quantize
 model,_=load_model_and_tokenizer(a.model,512,seed=20260617)
 toks=Tensor([[(i*7)%1000 for i in range(M)]],dtype='int32').contiguous()
 # Legal real-model carrier at the FFN boundary; shape and distribution come
 # from block-0 model weights and normalization, without invoking the candidate.
 x=model.blk[0].ffn_norm(model.blk[0].attn_norm(model.token_embd(toks).float())).reshape(M,K).cast('float32').contiguous().realize()
 cand=packed_row_major_candidate(M,N,K,role='ffn_gate_up',target='amd_gfx1100')
 @TinyJit
 def producer():
  q,s,_unused=pack_q8_1_mmq_fused(x,cand)
  # Q4_K min correction follows llama Q8_1: raw pre-quantization FP32
  # group sum, rounded through fp16 storage, then widened for this ABI.
  u=x.reshape(M,K//32,32).sum(axis=2).cast('float16').cast('float32').reshape(-1).contiguous()
  return q.realize(),s.realize(),u.realize()
 q,s,u=producer();producer();producer();Device['NV'].synchronize()
 prod_t=timing(producer,a.rounds)
 qn,sn,un=q.numpy(),s.numpy(),u.numpy();refq,refs=q8_1_quantize(x)
 refqn,refsn=refq.numpy(),refs.numpy()
 # Min-correction sums use raw pre-quantization FP32 values with the exact
 # fp16 metadata round point; they are not reconstructed as d8*sum(q8).
 refun=x.numpy().reshape(M,K//32,32).sum(2,dtype=np.float32).astype(np.float16).astype(np.float32).reshape(-1)
 meta=read_metadata(pathlib.Path(a.model));info=next(i for i in meta.infos if i.name=='blk.0.ffn_gate.weight')
 if info.typ!=GGML_Q4_K or tuple(reversed(info.dims))!=(N,K): raise RuntimeError(f'illegal gate fixture {info}')
 words=packed_u32_slice(pathlib.Path(a.model),meta,info,device='CPU').numpy()
 guards={"q8_values_shape":list(qn.shape)==[M*K],"q8_scales_shape":list(sn.shape)==[M*K//32],"q8_sums_shape":list(un.shape)==[M*K//32],"q4_words":words.size==N*(K//256)*36}
 provider={"status":"BLOCKED","reason":"provider_manifest_not_supplied"}
 if a.provider_manifest:
  try: provider={"status":"ABI_ACCEPTED","manifest":validate_provider(a.provider_manifest),"reason":"execution adapter awaits stable provider launch ABI"}
  except Exception as e: provider={"status":"REJECTED","reason":str(e)}
 rec={"schema":"tinygrad.nv_q4_imma_complete_chain_gate.v1","abi":ABI,"shape":{"M":M,"N":N,"K":K},"fixture":{"tensor":info.name,"q4_words_sha256":sha(words),"activation_sha256":sha(x.numpy()),"real_model_boundary":True},"guards":guards,
  "q8_producer":{"timing":prod_t,"finite":bool(np.isfinite(sn).all() and np.isfinite(un).all()),"values_exact":bool(np.array_equal(qn,refqn)),"scales_exact":bool(np.array_equal(sn,refsn)),"sums_max_abs":float(np.max(np.abs(un-refun))),"hashes":{"values":sha(qn),"scales":sha(sn),"sums":sha(un)}},
  "provider":provider,"control":{"corrected_fp16_us":569.7,"source":"nv-llama-prefill-lifecycle-audit.md"},"promotion":"CLOSED_UNTIL_PROVIDER_EXECUTES_MAIN_PLUS_FIXUP_AND_QUALIFIES"}
 path=pathlib.Path(a.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec,indent=2))
if __name__=='__main__':main()
