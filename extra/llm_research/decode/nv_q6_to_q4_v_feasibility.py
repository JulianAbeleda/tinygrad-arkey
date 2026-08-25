#!/usr/bin/env python3
"""Real-weight Q6_K -> Q4_K attention-V feasibility and direct-device gate."""
from __future__ import annotations

import argparse, ctypes, hashlib, json, pathlib, statistics, subprocess, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.engine.realize import get_runtime
from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_kernel
from tinygrad.llm.gguf import gguf_load, gguf_load_metadata
from tinygrad.llm.q6k_v_mmvq import emit_q6k_v_four_warp_fp16_direct
from tinygrad.uop.ops import UOp

ROWS, K, Q6_BLOCK_BYTES, Q4_BLOCK_BYTES = 1024, 4096, 210, 144
BASE_LIB = "/home/ubuntu/env/llama.cpp/build-cuda/bin/libggml-base.so.0.14.0"


def _reference_convert(raw_q6:np.ndarray) -> tuple[np.ndarray,np.ndarray]:
  lib=ctypes.CDLL(BASE_LIB)
  deq=lib.dequantize_row_q6_K
  deq.argtypes=(ctypes.c_void_p,ctypes.POINTER(ctypes.c_float),ctypes.c_int64)
  quant=lib.quantize_row_q4_K_ref
  quant.argtypes=(ctypes.POINTER(ctypes.c_float),ctypes.c_void_p,ctypes.c_int64)
  dense=np.empty(ROWS*K,dtype=np.float32)
  q4=np.empty(ROWS*(K//256)*Q4_BLOCK_BYTES,dtype=np.uint8)
  deq(raw_q6.ctypes.data_as(ctypes.c_void_p),dense.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),dense.size)
  quant(dense.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),q4.ctypes.data_as(ctypes.c_void_p),dense.size)
  return dense.reshape(ROWS,K),q4


def _programs():
  out=UOp.placeholder((ROWS,),dtypes.float32,0)
  q6=UOp.placeholder((ROWS*(K//256)*105,),dtypes.uint16,1)
  q4=UOp.placeholder((ROWS*(K//256)*36,),dtypes.uint32,1)
  x=UOp.placeholder((K,),dtypes.float16,2)
  return (to_program(emit_q6k_v_four_warp_fp16_direct()(out,q6,x),Device["NV"].renderer),
          to_program(q4k_g3_lanemap_gemv_kernel(ROWS,K,load_style="vector")(out,q4,x),Device["NV"].renderer))


def _direct_runner(prg, out:Tensor, packed:Tensor, x:Tensor):
  rt=get_runtime("NV",prg); gs,ls=prg.arg.launch_dims({})
  bufs=[t.uop.buffer.get_buf("NV") for t in (out,packed,x)]
  return lambda wait=False: rt(*[bufs[i] for i in prg.arg.globals],global_size=gs,local_size=ls,vals=(),wait=wait)


def _metrics(ref:np.ndarray,got:np.ndarray) -> dict:
  r=ref.astype(np.float64).reshape(-1); g=got.astype(np.float64).reshape(-1); diff=g-r
  rn=float(np.linalg.norm(r)); gn=float(np.linalg.norm(g))
  return {"relative_l2":float(np.linalg.norm(diff)/max(rn,1e-30)),
          "cosine":float(np.dot(r,g)/max(rn*gn,1e-30)),
          "max_abs":float(np.max(np.abs(diff))),"mean_abs":float(np.mean(np.abs(diff))),
          "finite":bool(np.isfinite(got).all())}


def main() -> int:
  ap=argparse.ArgumentParser(); ap.add_argument("--model",default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--blocks",default="0,1,2,3,6,9,12,15,18,21"); ap.add_argument("--activations",type=int,default=8)
  ap.add_argument("--timing-reps",type=int,default=31); ap.add_argument("--out",required=True); args=ap.parse_args()
  model=pathlib.Path(args.model); blocks=[int(x) for x in args.blocks.split(",") if x]
  _,meta=gguf_load_metadata(model); infos={x[0]:x for x in meta["tensor_infos"]}
  _,state=gguf_load(model)
  acts=state["token_embd.weight"][:args.activations].numpy().astype(np.float16)
  p6,p4=_programs(); rows=[]
  for bi in blocks:
    name=f"blk.{bi}.attn_v.weight"; _n,shape,typ,off=infos[name]
    if tuple(shape)!=(K,ROWS) or typ!=14: raise RuntimeError(f"not Q6 V: {infos[name]!r}")
    n6=ROWS*(K//256)*Q6_BLOCK_BYTES
    raw=np.asarray(np.memmap(model,mode="r",dtype=np.uint8,offset=meta["data_start"]+off,shape=(n6,))).copy()
    dense,q4=_reference_convert(raw)
    # Reference dequantization of the candidate is tinygrad's independent Q4 decoder.
    from tinygrad.llm.gguf import ggml_data_to_tensor
    q4_dense=ggml_data_to_tensor(Tensor(q4.copy(),dtype=dtypes.uint8),ROWS*K,12).numpy().reshape(ROWS,K).astype(np.float32)
    weight_metrics=_metrics(dense,q4_dense)
    tq6=Tensor(raw.view(np.uint16).copy(),dtype=dtypes.uint16,device="NV").realize()
    tq4=Tensor(q4.view(np.uint32).copy(),dtype=dtypes.uint32,device="NV").realize()
    x=Tensor(acts[0],dtype=dtypes.float16,device="NV").realize(); o6=Tensor.empty(ROWS,dtype=dtypes.float32,device="NV").realize(); o4=Tensor.empty(ROWS,dtype=dtypes.float32,device="NV").realize()
    r6,r4=_direct_runner(p6,o6,tq6,x),_direct_runner(p4,o4,tq4,x); r6(True);r4(True)
    output=[]
    for av in acts:
      x.assign(Tensor(av,dtype=dtypes.float16,device="NV")).realize(); r6(True);r4(True)
      output.append(_metrics(o6.numpy(),o4.numpy()))
    t6=[r6(True)*1e6 for _ in range(args.timing_reps)];t4=[r4(True)*1e6 for _ in range(args.timing_reps)]
    rows.append({"block":bi,"tensor":name,"q6_bytes":n6,"q4_bytes":q4.nbytes,"bytes_saved":n6-q4.nbytes,
      "weight":weight_metrics,"outputs":output,"timing_us":{"q6_median":statistics.median(t6),"q4_median":statistics.median(t4),
      "delta":statistics.median(t4)-statistics.median(t6)}})
  rel=[m["relative_l2"] for r in rows for m in r["outputs"]]; cos=[m["cosine"] for r in rows for m in r["outputs"]]
  result={"schema":"tinygrad.nv_q6_to_q4_v_feasibility.v1","commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    "model":str(model),"model_sha256_prefix":hashlib.sha256(model.read_bytes()[:1<<20]).hexdigest(),"blocks":blocks,
    "quality_contract":{"projection_relative_l2_max":0.05,"projection_cosine_min":0.999,"authority":"feasibility_only"},
    "rows":rows,"summary":{"q6_bytes":sum(r["q6_bytes"] for r in rows),"q4_bytes":sum(r["q4_bytes"] for r in rows),
      "bytes_saved":sum(r["bytes_saved"] for r in rows),"bytes_saved_fraction":1-sum(r["q4_bytes"] for r in rows)/sum(r["q6_bytes"] for r in rows),
      "output_relative_l2_max":max(rel),"output_relative_l2_median":statistics.median(rel),"output_cosine_min":min(cos),
      "feasibility_pass":max(rel)<=0.05 and min(cos)>=0.999 and all(m["finite"] for r in rows for m in r["outputs"])}}
  text=json.dumps(result,indent=2,sort_keys=True);pathlib.Path(args.out).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(args.out).write_text(text+"\n");print(text)
  return 0 if result["summary"]["feasibility_pass"] else 1
if __name__=="__main__": raise SystemExit(main())
