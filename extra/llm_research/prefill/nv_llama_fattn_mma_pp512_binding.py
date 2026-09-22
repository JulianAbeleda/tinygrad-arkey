"""Exact whole-tile llama MMA Flash binding for NV Qwen3-8B pp512."""
from __future__ import annotations
from pathlib import Path
from tinygrad import Tensor, dtypes
from extra.llm_research.prefill.nv_native_program_uop import native_nv_program

S,D,HQ,HK=512,128,32,8
SYMBOL="nv_llama_fattn_mma_pp512"
CUBIN=Path(__file__).parents[3]/"docs/task_workflow/evidence/nv-cleanroom-flash-production-boundary-20260830/flash-mma-pp512-specialized.sm_120a.cubin"

def program():
  return native_nv_program(SYMBOL,CUBIN.read_bytes(),global_size=(256,1,1),local_size=(32,4,1),
    # QMD records total shared memory, unlike CUDA's launch API which accepts
    # only the dynamic portion. This cubin also owns 1024 bytes of static smem.
    globals=(0,1,2,3,4),outs=(4,),ins=(0,1,2,3),shared_mem=37120+1024)

_PROGRAM=None
def project(q:Tensor,k:Tensor,v:Tensor,mask:Tensor)->Tensor:
  global _PROGRAM
  if tuple(q.shape)!=(1,HQ,S,D) or q.dtype!=dtypes.float32 or q.device!="NV":raise ValueError("MMA Flash Q contract mismatch")
  if tuple(k.shape)!=(1,HK,S,D) or tuple(v.shape)!=(1,HK,S,D) or k.dtype!=dtypes.float16 or v.dtype!=dtypes.float16:raise ValueError("MMA Flash KV contract mismatch")
  if tuple(mask.shape)!=(1,1,S,S) or mask.dtype!=dtypes.float16:raise ValueError("MMA Flash mask contract mismatch")
  if _PROGRAM is None:_PROGRAM=program()
  out=Tensor.empty((1,S,HQ,D),dtype=dtypes.float32,device="NV")
  q,k,v,mask,out=q.uop_program(k,v,mask,out,fxn=lambda *_:_PROGRAM)
  return out.permute(0,2,1,3)
