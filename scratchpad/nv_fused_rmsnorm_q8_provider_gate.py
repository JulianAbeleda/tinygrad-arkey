#!/usr/bin/env python3
"""Isolated numerical gate for the fused RMSNorm -> Q8_1 provider.

This does not instantiate a model or a decode graph.  It compares the fused
one-program provider against the exact two-program construction it replaces:
the qualified ordinary RMSNorm emitter followed by the ordinary llama-Q8_1
provider.  The packed int8 payload and fp16 d|sum metadata must match bitwise.
"""
from __future__ import annotations

import hashlib, json
import numpy as np

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen.late.reduce_output import emit_reduce_output_rmsnorm
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec, execute_promoted_program
from tinygrad.llm.shared_q8_attention import _emit_q8_provider, _emit_rmsnorm_q8_provider
from tinygrad.uop.ops import ReduceOutputSpec

K, PACKED = 4096, 1152


def _program(name, emitter, shape, dtype):
  return KernelProgram("research.nv_fused_rmsnorm_q8_provider_gate", name,
    KernelProgramProvenance.MACHINE_SEARCH_GENERATED, emitter, output_spec=OutputSpec(shape, dtype))


def _run_case(name:str, x_np:np.ndarray, weight_np:np.ndarray) -> dict:
  dev=Device.DEFAULT
  x=Tensor(x_np,dtype=dtypes.float32,device=dev).contiguous().realize()
  weight=Tensor(weight_np,dtype=dtypes.float16,device=dev).contiguous().realize()
  spec=ReduceOutputSpec(1,K,1e-6,dtypes.float32)
  rms_program=_program("ordinary_rmsnorm",emit_reduce_output_rmsnorm(spec,dtypes.float32,dtypes.float16),(K,),dtypes.float32)
  q8_program=_program("ordinary_q8_provider",_emit_q8_provider(),(PACKED,),dtypes.uint32)
  fused_program=_program("fused_rmsnorm_q8_provider",_emit_rmsnorm_q8_provider(spec,dtypes.float32,dtypes.float16),(PACKED,),dtypes.uint32)

  normed=execute_promoted_program(None,x,weight,program=rms_program).realize()
  ordinary=execute_promoted_program(None,normed,program=q8_program).realize()
  fused=execute_promoted_program(None,x,weight,program=fused_program).realize()
  Device[dev].synchronize()
  ordinary_np=ordinary.numpy().astype(np.uint32,copy=False)
  fused_np=fused.numpy().astype(np.uint32,copy=False)
  mismatch=np.flatnonzero(ordinary_np != fused_np)
  return {
    "case":name,
    "bitwise_equal":bool(not mismatch.size),
    "mismatch_count":int(mismatch.size),
    "first_mismatch":None if not mismatch.size else int(mismatch[0]),
    "ordinary_sha256":hashlib.sha256(ordinary_np.tobytes()).hexdigest(),
    "fused_sha256":hashlib.sha256(fused_np.tobytes()).hexdigest(),
  }


def main():
  if not str(Device.DEFAULT).startswith("NV"): raise RuntimeError("run with DEV=NV")
  rng=np.random.default_rng(20260805)
  cases=[
    ("normal",rng.normal(0.0,0.25,K).astype(np.float32),rng.normal(1.0,0.1,K).astype(np.float16)),
    ("all_zero",np.zeros(K,dtype=np.float32),np.ones(K,dtype=np.float16)),
    ("dynamic_range",np.geomspace(2**-12,2**6,K,dtype=np.float32)*rng.choice(np.array([-1.,1.],dtype=np.float32),K),
      rng.uniform(0.5,1.5,K).astype(np.float16)),
  ]
  rows=[_run_case(name,x,w) for name,x,w in cases]
  result={"schema":"tinygrad.nv_fused_rmsnorm_q8_provider_gate.v1","device":str(Device.DEFAULT),"cases":rows,
          "gate":"PASS" if all(x["bitwise_equal"] for x in rows) else "FAIL"}
  print(json.dumps(result,indent=2,sort_keys=True))
  if result["gate"] != "PASS": raise SystemExit(1)


if __name__ == "__main__": main()
