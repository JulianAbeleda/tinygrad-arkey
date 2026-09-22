#!/usr/bin/env python3
"""Reference real-weight Q6_K -> Q5_K quality/byte feasibility."""
from __future__ import annotations
import argparse,ctypes,json,pathlib,statistics,subprocess,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tinygrad import Tensor,dtypes
from tinygrad.llm.gguf import ggml_data_to_tensor,gguf_load,gguf_load_metadata
from extra.llm_research.decode.nv_q6_to_q4_v_feasibility import _metrics
LIB="/home/ubuntu/env/llama.cpp/build-cuda/bin/libggml-base.so.0.14.0";Q6B,Q5B=210,176

def convert(raw,n):
  lib=ctypes.CDLL(LIB);deq=lib.dequantize_row_q6_K;deq.argtypes=(ctypes.c_void_p,ctypes.POINTER(ctypes.c_float),ctypes.c_int64)
  quant=lib.quantize_row_q5_K_ref;quant.argtypes=(ctypes.POINTER(ctypes.c_float),ctypes.c_void_p,ctypes.c_int64)
  dense=np.empty(n,dtype=np.float32);q5=np.empty((n//256)*Q5B,dtype=np.uint8)
  deq(raw.ctypes.data_as(ctypes.c_void_p),dense.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),n)
  quant(dense.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),q5.ctypes.data_as(ctypes.c_void_p),n);return dense,q5

def main():
  ap=argparse.ArgumentParser();ap.add_argument("--model",default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf");ap.add_argument("--out",required=True);a=ap.parse_args()
  model=pathlib.Path(a.model);_,meta=gguf_load_metadata(model);infos={x[0]:x for x in meta["tensor_infos"]};_,state=gguf_load(model)
  vacts=state["token_embd.weight"][:2].numpy().astype(np.float32);rng=np.random.default_rng(20260825);dacts=rng.normal(0,.2,(2,12288)).astype(np.float32)
  pops={"V":[0,1,2,3,6,9,12,15,18,21],"down":[0,1,2,3,6,9,12,15,18,21,24,27,30,31,32,33,34,35]};rows=[]
  for role,blocks in pops.items():
    R,K=(1024,4096) if role=="V" else (4096,12288);acts=vacts if role=="V" else dacts
    for bi in blocks:
      name=f"blk.{bi}.{'attn_v' if role=='V' else 'ffn_down'}.weight";_n,shape,typ,off=infos[name]
      if tuple(shape)!=(K,R) or typ!=14:raise RuntimeError(infos[name])
      n=R*K;n6=(n//256)*Q6B;raw=np.asarray(np.memmap(model,mode="r",dtype=np.uint8,offset=meta["data_start"]+off,shape=(n6,))).copy();dense,q5=convert(raw,n)
      q5d=ggml_data_to_tensor(Tensor(q5.copy(),dtype=dtypes.uint8),n,13).numpy().reshape(R,K).astype(np.float32);dense=dense.reshape(R,K)
      outs=[_metrics(dense@x,q5d@x) for x in acts]
      rows.append({"role":role,"block":bi,"tensor":name,"q6_bytes":n6,"q5_bytes":q5.nbytes,"bytes_saved":n6-q5.nbytes,"weight":_metrics(dense,q5d),"outputs":outs})
  rel=[m["relative_l2"] for r in rows for m in r["outputs"]];cos=[m["cosine"] for r in rows for m in r["outputs"]]
  byrole={}
  for role in pops:
    rr=[r for r in rows if r["role"]==role];rv=[m["relative_l2"] for r in rr for m in r["outputs"]];cv=[m["cosine"] for r in rr for m in r["outputs"]]
    byrole[role]={"bytes_saved":sum(r["bytes_saved"] for r in rr),"relative_l2_max":max(rv),"relative_l2_median":statistics.median(rv),"cosine_min":min(cv)}
  ret={"schema":"tinygrad.nv_q6_to_q5_feasibility.v1","commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"quality_contract":{"relative_l2_max":.05,"cosine_min":.999},"rows":rows,
    "summary":{"q6_bytes":sum(r["q6_bytes"] for r in rows),"q5_bytes":sum(r["q5_bytes"] for r in rows),"bytes_saved":sum(r["bytes_saved"] for r in rows),"bytes_saved_fraction":1-sum(r["q5_bytes"] for r in rows)/sum(r["q6_bytes"] for r in rows),
    "relative_l2_max":max(rel),"relative_l2_median":statistics.median(rel),"cosine_min":min(cos),"by_role":byrole,"feasibility_pass":max(rel)<=.05 and min(cos)>=.999}}
  text=json.dumps(ret,indent=2,sort_keys=True);pathlib.Path(a.out).write_text(text+"\n");print(json.dumps(ret["summary"],indent=2));return 0 if ret["summary"]["feasibility_pass"] else 1
if __name__=="__main__":raise SystemExit(main())
