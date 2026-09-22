#!/usr/bin/env python3
"""Real-weight Q6_K -> Q4_K FFN-down feasibility and direct-device gate."""
from __future__ import annotations
import argparse, json, pathlib, statistics, subprocess, sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tinygrad import Device,Tensor,dtypes
from tinygrad.codegen import to_program
from tinygrad.engine.realize import get_runtime
from tinygrad.llm.gguf import ggml_data_to_tensor,gguf_load_metadata
from tinygrad.llm.q4k_ffn_down_mmvq import emit_four_warp_fp16_direct
from tinygrad.llm.q6k_ffn_down_mmvq import emit_q6k_four_warp_fp16_direct
from tinygrad.uop.ops import UOp
from extra.llm_research.decode.nv_q6_to_q4_v_feasibility import _reference_convert as _v_convert,_metrics

ROWS,K,Q6B,Q4B=4096,12288,210,144

def _convert(raw):
  # The reference routines operate on arbitrary multiples of 256; temporarily
  # override the V module's fixed shape by calling the same exported symbols.
  import ctypes
  lib=ctypes.CDLL("/home/ubuntu/env/llama.cpp/build-cuda/bin/libggml-base.so.0.14.0")
  deq=lib.dequantize_row_q6_K;deq.argtypes=(ctypes.c_void_p,ctypes.POINTER(ctypes.c_float),ctypes.c_int64)
  quant=lib.quantize_row_q4_K_ref;quant.argtypes=(ctypes.POINTER(ctypes.c_float),ctypes.c_void_p,ctypes.c_int64)
  dense=np.empty(ROWS*K,dtype=np.float32);q4=np.empty(ROWS*(K//256)*Q4B,dtype=np.uint8)
  deq(raw.ctypes.data_as(ctypes.c_void_p),dense.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),dense.size)
  quant(dense.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),q4.ctypes.data_as(ctypes.c_void_p),dense.size)
  return dense.reshape(ROWS,K),q4

def _programs():
  out=UOp.placeholder((ROWS,),dtypes.float32,0);q6=UOp.placeholder((ROWS*(K//256)*105,),dtypes.uint16,1)
  q4=UOp.placeholder((ROWS*(K//256)*36,),dtypes.uint32,1);x=UOp.placeholder((K,),dtypes.float16,2);h=UOp.placeholder((ROWS,),dtypes.float32,3)
  return (to_program(emit_q6k_four_warp_fp16_direct(packed_lanemap=True,unroll_blocks=4)(out,q6,x,h),Device["NV"].renderer),
          to_program(emit_four_warp_fp16_direct(UOp.const(dtypes.weakint,3),resadd=True,load_style="vector")(out,q4,x,h),Device["NV"].renderer))

def _runner(p,o,w,x,h):
  rt=get_runtime("NV",p);gs,ls=p.arg.launch_dims({});buf=[t.uop.buffer.get_buf("NV") for t in (o,w,x,h)]
  return lambda wait=False:rt(*[buf[i] for i in p.arg.globals],global_size=gs,local_size=ls,vals=(),wait=wait)

def main():
  ap=argparse.ArgumentParser();ap.add_argument("--model",default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--blocks",default="0,1,2,3,6,9,12,15,18,21,24,27,30,31,32,33,34,35");ap.add_argument("--activations",type=int,default=8)
  ap.add_argument("--timing-reps",type=int,default=21);ap.add_argument("--out",required=True);a=ap.parse_args()
  model=pathlib.Path(a.model);_,meta=gguf_load_metadata(model);infos={x[0]:x for x in meta["tensor_infos"]};blocks=[int(x) for x in a.blocks.split(",") if x]
  rng=np.random.default_rng(20260825);acts=rng.normal(0,.2,(a.activations,K)).astype(np.float16);p6,p4=_programs();rows=[]
  for bi in blocks:
    name=f"blk.{bi}.ffn_down.weight";_n,shape,typ,off=infos[name]
    if tuple(shape)!=(K,ROWS) or typ!=14:raise RuntimeError(f"not Q6 down: {infos[name]!r}")
    n6=ROWS*(K//256)*Q6B;raw=np.asarray(np.memmap(model,mode="r",dtype=np.uint8,offset=meta["data_start"]+off,shape=(n6,))).copy();dense,q4=_convert(raw)
    q4d=ggml_data_to_tensor(Tensor(q4.copy(),dtype=dtypes.uint8),ROWS*K,12).numpy().reshape(ROWS,K).astype(np.float32)
    tq6=Tensor(raw.view(np.uint16).copy(),dtype=dtypes.uint16,device="NV").realize();tq4=Tensor(q4.view(np.uint32).copy(),dtype=dtypes.uint32,device="NV").realize()
    x=Tensor(acts[0],dtype=dtypes.float16,device="NV").realize();h=Tensor.zeros(ROWS,dtype=dtypes.float32,device="NV").realize();o6=Tensor.empty(ROWS,dtype=dtypes.float32,device="NV").realize();o4=Tensor.empty(ROWS,dtype=dtypes.float32,device="NV").realize()
    r6,r4=_runner(p6,o6,tq6,x,h),_runner(p4,o4,tq4,x,h);r6(True);r4(True);outs=[]
    for av in acts:x.assign(Tensor(av,dtype=dtypes.float16,device="NV")).realize();r6(True);r4(True);outs.append(_metrics(o6.numpy(),o4.numpy()))
    t6=[r6(True)*1e6 for _ in range(a.timing_reps)];t4=[r4(True)*1e6 for _ in range(a.timing_reps)]
    rows.append({"block":bi,"tensor":name,"q6_bytes":n6,"q4_bytes":q4.nbytes,"bytes_saved":n6-q4.nbytes,"weight":_metrics(dense,q4d),"outputs":outs,
      "timing_us":{"q6_median":statistics.median(t6),"q4_median":statistics.median(t4),"delta":statistics.median(t4)-statistics.median(t6)}})
  rel=[m["relative_l2"] for r in rows for m in r["outputs"]];cos=[m["cosine"] for r in rows for m in r["outputs"]]
  ret={"schema":"tinygrad.nv_q6_to_q4_down_feasibility.v1","commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"blocks":blocks,"rows":rows,
    "quality_contract":{"projection_relative_l2_max":.05,"projection_cosine_min":.999,"authority":"feasibility_only"},"summary":{"q6_bytes":sum(r["q6_bytes"] for r in rows),"q4_bytes":sum(r["q4_bytes"] for r in rows),"bytes_saved":sum(r["bytes_saved"] for r in rows),
    "bytes_saved_fraction":1-sum(r["q4_bytes"] for r in rows)/sum(r["q6_bytes"] for r in rows),"output_relative_l2_max":max(rel),"output_relative_l2_median":statistics.median(rel),"output_cosine_min":min(cos),
    "device_recovery_us_sum":sum(-r["timing_us"]["delta"] for r in rows),"feasibility_pass":max(rel)<=.05 and min(cos)>=.999}}
  text=json.dumps(ret,indent=2,sort_keys=True);pathlib.Path(a.out).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(a.out).write_text(text+"\n");print(json.dumps(ret["summary"],indent=2));return 0 if ret["summary"]["feasibility_pass"] else 1
if __name__=="__main__":raise SystemExit(main())
