"""Exact llama.cpp Q6_K vocabulary MMVQ binding contract (NV, pp512).

The cubins are unmodified llama CUDA artifacts. This module only admits the
terminal M=1 shape; callers must provide the post-output_norm row and retain
the Q8 producer -> Q6 consumer dependency.
"""
from pathlib import Path
import struct
from extra.llm_research.prefill.nv_native_program_uop import native_nv_program

ARTIFACTS = Path(__file__).resolve().parents[3] / "scratchpad/llama_cuda_quantized_oracle_dump"
Q8_SYMBOL = "_Z17quantize_mmq_q8_1IL18mmq_q8_1_ds_layout2EEvPKfPKiPvlllllii"
Q6_SYMBOL = "_Z13mul_mat_vec_qIL9ggml_type14ELi1ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj"

# CUDA parameter-bank offsets recovered from the real llama graph node.
Q6_ARG_OFFSETS = (0, 8, 16, 24, 56, 64, 68, 80, 84, 88, 92, 104, 108, 112, 116, 128, 132, 136, 140)
Q6_ARG_SIZES = (8, 8, 8, 32, 8, 4, 12, 4, 4, 4, 12, 4, 4, 4, 12, 4, 4, 4, 4)

def enabled(config) -> bool:
  return bool(getattr(config, "prefill_ubatch", None) == 512 and __import__("os").environ.get("NV_LLAMA_Q6_VOCAB_PP512") == "1")

def artifacts() -> dict:
  return {"q8": ARTIFACTS / "libggml-cuda.q8_1.sm_120a.cubin", "q6": Path(__file__).resolve().parents[3] / "docs/task_workflow/evidence/nv-llama-q6k-vocab-standalone-20260830/q6k-mmvq-nopdl.sm_120a.cubin"}

def validate() -> None:
  a = artifacts()
  if not all(p.is_file() and p.read_bytes()[:4] == b"\x7fELF" for p in a.values()): raise FileNotFoundError("exact llama vocabulary cubins missing")
  if Q6_ARG_OFFSETS[-1] + Q6_ARG_SIZES[-1] != 144: raise ValueError("invalid captured Q6 parameter-bank layout")

def programs():
  validate()
  q8_layout = (("ptr",0,8,8,0),("ptr",-1,8,8,8),("ptr",1,8,8,16)) + tuple(("u64",4096,8,8,24+i*8) for i in range(5)) + (("u32",1,4,4,64),("u32",1,4,4,68))
  q8 = native_nv_program("llama_q8_1_vocab_4096", artifacts()["q8"].read_bytes(), global_size=(128,1,1), local_size=(256,1,1), globals=(0,1), outs=(1,), ins=(0,), arg_layout=q8_layout)
  layout=[]
  sizes=Q6_ARG_SIZES; offsets=Q6_ARG_OFFSETS
  ptrs={0:0,1:1,2:-1,4:2}
  for i,(size,off) in enumerate(zip(sizes,offsets)):
    if i in ptrs: kind,src="ptr",ptrs[i]
    elif i==3: kind,src="blob",bytes(32)
    elif i in (6,10,14): kind,src= "blob", struct.pack('<III',1,1,1)
    elif i in (8,): kind,src="u32",128
    elif i in (9,): kind,src="u32",151936
    elif i in (11,12,13): kind,src="u32",1
    elif i in (15,16,17): kind,src="u32",1
    elif i in (5,): kind,src="u32",4096
    elif i in (7,): kind,src="u32",16
    elif i in (9,13,17): kind,src="u32",151936
    else: kind,src="u32",1
    layout.append((kind,src,size,4,off))
  import os
  q6 = native_nv_program("llama_q6k_vocab_151936", artifacts()["q6"].read_bytes(), global_size=(int(os.environ.get("Q6_TEST_ROWS", "151936")),1,1), local_size=(32,4,1), globals=(0,1,2), arg_layout=tuple(layout))
  return q8, q6
