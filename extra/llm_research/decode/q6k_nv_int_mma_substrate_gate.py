#!/usr/bin/env python3
"""CPU/compiler-only gate for native CUDA signed-int8 TensorCore support.

The generic TensorCore descriptor deliberately remains fail-closed: int8
WMMA is admitted only by an exact typed compiler candidate.  The renderer
probe separately proves that such a qualified candidate can lower the native
instruction once its lane/fragment contract has been established.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from tinygrad import dtypes
from tinygrad.codegen.opt import tc
from tinygrad.helpers import Target
from tinygrad.renderer.ptx import PTXRenderer, render_wmma
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import Ops, UOp

MMA_SIGNATURE="mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"


def _nvrtc_int_mma_probe() -> dict:
  source=r'''extern "C" __global__ void int_mma_probe(int *out) {
  unsigned a0=0, a1=0, a2=0, a3=0, b0=0, b1=0;
  int d0, d1, d2, d3, c0=0, c1=0, c2=0, c3=0;
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
    : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
      "r"(c0), "r"(c1), "r"(c2), "r"(c3));
  if (threadIdx.x == 0) { out[0]=d0; out[1]=d1; out[2]=d2; out[3]=d3; }
}'''
  try:
    ptx=NVRTCCompiler("sm_120",ptx=True,cache_key="q6k_int_mma_substrate").compile(source).decode()
  except Exception as exc: return {"pass":False,"error":f"{type(exc).__name__}: {exc}"}
  return {"pass":MMA_SIGNATURE in ptx,"signature":MMA_SIGNATURE,"ptx_contains_signature":MMA_SIGNATURE in ptx,
    "fragment_registers":{"a_b32":4,"b_b32":2,"acc_s32":4}}


def _tinygrad_descriptor_probe() -> dict:
  # Use the CUDA device spelling so this CPU/compiler-only audit selects the
  # direct PTX compiler rather than native NV's nvJitLink runtime path.
  renderer=PTXRenderer(Target.parse("CUDA:CUDA:sm_120"))
  cuda_pairs=sorted({(str(x.dtype_in),str(x.dtype_out),x.dims) for x in tc.cuda_sm80})
  admitted=sorted({(str(x.dtype_in),str(x.dtype_out),x.dims) for x in renderer.tensor_cores})
  wanted=(str(dtypes.int8),str(dtypes.int32),(8,16,32))
  return {"wanted":{"dtype_in":"dtypes.char","dtype_out":"dtypes.int","dims_nmk":[8,16,32]},
    "cuda_descriptor_present":wanted in cuda_pairs,"ptx_renderer_admits":wanted in admitted,
    "cuda_descriptors":[[a,b,list(c)] for a,b,c in cuda_pairs],"renderer_descriptors":[[a,b,list(c)] for a,b,c in admitted]}


def _tinygrad_render_probe() -> dict:
  # Exercise render_wmma rather than inferring support from source text.  This
  # is the lowering used only after an exact candidate supplied the fragment
  # ABI; it is not ambient generic TensorCore admission.
  a=UOp(Ops.NOOP,dtypes.int8.vec(16)); b=UOp(Ops.NOOP,dtypes.int8.vec(8)); c=UOp(Ops.NOOP,dtypes.int32.vec(4))
  w=UOp(Ops.WMMA,dtypes.int32.vec(4),(a,b,c),arg=("int_mma_probe",(8,16,32),dtypes.int8,dtypes.int32))
  ctx=SimpleNamespace(wmma_r=[[f"a{i}" for i in range(4)],[f"b{i}" for i in range(2)],[f"c{i}" for i in range(4)]],
    r={a:[f"aa{i}" for i in range(16)],b:[f"bb{i}" for i in range(8)],c:[f"cc{i}" for i in range(4)],w:[f"dd{i}" for i in range(4)]})
  try: lines=list(render_wmma(ctx,w))
  except Exception as exc: return {"pass":False,"error":f"{type(exc).__name__}: {exc}"}
  return {"pass":any(MMA_SIGNATURE in line for line in lines),"lines":lines}


def audit() -> dict:
  hardware=_nvrtc_int_mma_probe(); descriptor=_tinygrad_descriptor_probe(); renderer=_tinygrad_render_probe()
  generic_closed=not descriptor["cuda_descriptor_present"] and not descriptor["ptx_renderer_admits"]
  qualified_ready=hardware["pass"] and generic_closed and renderer["pass"]
  return {"schema":"tinygrad.q6k_nv_int_mma_substrate_gate.v2","hardware_compiler":hardware,
    "tinygrad_descriptor":descriptor,"tinygrad_renderer":renderer,
    "generic_descriptor_remains_fail_closed":generic_closed,"qualified_candidate_substrate_ready":qualified_ready,
    "blocker_localized_to_tinygrad_substrate":False,
    "next_gate":"retain exact typed fragment/provider admission and run adversarial value, real-shape, SASS, and lifecycle gates"}


if __name__ == "__main__": print(json.dumps(audit(),indent=2,sort_keys=True))
