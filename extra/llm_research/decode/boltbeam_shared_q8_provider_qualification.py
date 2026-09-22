#!/usr/bin/env python3
"""Compare the exact BoltBeam shared-Q8 provider with llama's Q8_1 reference."""
from __future__ import annotations

import argparse, ctypes, hashlib, json, pathlib, statistics, subprocess
import numpy as np

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.device import Buffer
from tinygrad.engine.realize import get_runtime
from tinygrad.llm.boltbeam_authority import lower_authorized_candidate
from tinygrad.uop.ops import UOp
from scratchpad.llama_cuda_quantized_live_oracle import device_pointer

K,PACK,GROUPS,PACKETS=4096,32,128,1024
BASE_LIB="/home/ubuntu/env/llama.cpp/build-cuda/bin/libggml-base.so.0.14.0"
CUDA_LIB="/home/ubuntu/env/llama.cpp/build-cuda/bin/libggml-cuda.so.0.14.0"


def _private_layout(raw_bytes:bytes) -> np.ndarray:
  blocks=np.frombuffer(raw_bytes,dtype=np.uint8).reshape(GROUPS,36);out=np.zeros(PACKETS+GROUPS,dtype=np.uint32)
  payload=blocks[:,4:].reshape(-1,4).astype(np.uint32)
  out[:PACKETS]=payload@np.array([1,256,65536,16777216],dtype=np.uint32)
  d=blocks[:,:2].copy().reshape(-1).view(np.uint16).astype(np.uint32)
  s=blocks[:,2:4].copy().reshape(-1).view(np.uint16).astype(np.uint32)
  out[PACKETS:]=d|(s<<np.uint32(16));return out


def llama_cpu_reference(values:np.ndarray) -> np.ndarray:
  lib=ctypes.CDLL(BASE_LIB,mode=ctypes.RTLD_LOCAL);fn=lib.quantize_row_q8_1_ref;fn.restype=None
  fn.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.c_void_p,ctypes.c_int64]
  source=np.ascontiguousarray(values.astype(np.float32));raw=(ctypes.c_uint8*(GROUPS*36))()
  fn(source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),raw,K)
  return _private_layout(bytes(raw))


def llama_cuda_reference(values:np.ndarray) -> np.ndarray:
  lib=ctypes.CDLL(CUDA_LIB,mode=ctypes.RTLD_LOCAL)
  fn=getattr(lib,"_Z22quantize_row_q8_1_cudaPKfPKiPv9ggml_typellllllllP11CUstream_st")
  fn.restype=None;fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]+[ctypes.c_int64]*8+[ctypes.c_void_p]
  source=np.ascontiguousarray(values.astype(np.float32));src=Buffer("CUDA",K,dtypes.float32,initial_value=bytearray(source.tobytes()))
  dst=Buffer("CUDA",GROUPS*36,dtypes.uint8,preallocate=True)
  fn(device_pointer(src),None,device_pointer(dst),0,K,K,K,K,K,1,1,1,None);Device["CUDA"].synchronize()
  raw=bytearray(dst.nbytes);dst.copyout(memoryview(raw));return _private_layout(bytes(raw))


def main() -> int:
  parser=argparse.ArgumentParser();parser.add_argument("--reps",type=int,default=11);parser.add_argument("--out",required=True);args=parser.parse_args()
  if not str(Device.DEFAULT).startswith("NV"): raise RuntimeError(f"native NV required, got {Device.DEFAULT}")
  values=np.random.default_rng(202609013).normal(0,.2,K).astype(np.float16)
  out=Tensor.empty(PACKETS+GROUPS,dtype=dtypes.uint32,device="NV").realize();x=Tensor(values.copy(),dtype=dtypes.float16,device="NV").realize()
  emitter,tickets=lower_authorized_candidate({"family":"shared_q8_provider.v1","source_dtype":"fp16","k":K},
    (("decode_shared_q8_attention","shared_q8_provider"),))
  ast=emitter(UOp.placeholder((PACKETS+GROUPS,),dtypes.uint32,0),UOp.placeholder((K,),dtypes.float16,1))
  program=to_program(ast,Device["NV"].renderer);runtime=get_runtime("NV",program);gs,ls=program.arg.launch_dims({})
  buffers=[out.uop.buffer.get_buf("NV"),x.uop.buffer.get_buf("NV")]
  def run(wait=False): return runtime(*[buffers[i] for i in program.arg.globals],global_size=gs,local_size=ls,vals=(),wait=wait)
  run(True);got=out.numpy();reference=llama_cuda_reference(values);cpu_reference=llama_cpu_reference(values);timing=[run(True)*1e6 for _ in range(args.reps)]
  mismatches=int(np.count_nonzero(got!=reference));cpu_payload=int(np.count_nonzero(got[:PACKETS]!=cpu_reference[:PACKETS]))
  cpu_metadata=int(np.count_nonzero(got[PACKETS:]!=cpu_reference[PACKETS:]));report={"schema":"tinygrad.boltbeam_shared_q8_provider_qualification.v1",
    "device":str(Device.DEFAULT),"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    "route_id":"decode_shared_q8_attention","component":"shared_q8_provider",
    "tickets":[ticket.to_dict() for ticket in tickets.tickets],"payload_sha256":hashlib.sha256(values.tobytes()).hexdigest(),
    "correctness":{"pass":mismatches==0,"live_cuda_mismatched_words":mismatches,"live_cuda_bitwise_identical":mismatches==0,
      "cpu_reference_diagnostic":{"payload_mismatched_words":cpu_payload,"metadata_mismatched_words":cpu_metadata}},
    "timing_us":{"samples":timing,"median":statistics.median(timing)}}
  text=json.dumps(report,indent=2,sort_keys=True);path=pathlib.Path(args.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text+"\n");print(text)
  return 0 if mismatches==0 else 1


if __name__ == "__main__": raise SystemExit(main())
