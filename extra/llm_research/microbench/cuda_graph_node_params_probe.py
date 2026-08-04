#!/usr/bin/env python3
"""Debug probe: what do cuGraphKernelNodeGetParams (v1/v2) return for kernels
captured with the extra-array launch style (vargs), on this driver?

Answers whether the B2 capture lowerer can match captured nodes back to calls
by the 'extra' pointer. One flocked GPU session.
"""
import ctypes
import tinygrad.runtime.autogen.cuda as cuda
from tinygrad.runtime.ops_cuda import check
from tinygrad.runtime.support.c import init_c_var

MODULE_BYTES = open("/tmp/k_probe.cubin", "rb").read()

check(cuda.cuInit(0))
dev = init_c_var(cuda.CUdevice, lambda x: check(cuda.cuDeviceGet(ctypes.byref(x), 0)))
ctx = init_c_var(cuda.CUcontext, lambda x: check(cuda.cuCtxCreate_v2(ctypes.byref(x), 0, dev)))
mod = init_c_var(cuda.CUmodule, lambda x: check(cuda.cuModuleLoadData(ctypes.byref(x), MODULE_BYTES)))
fn = init_c_var(cuda.CUfunction, lambda x: check(cuda.cuModuleGetFunction(ctypes.byref(x), mod, b"_Z1kPf")))

stream = init_c_var(cuda.CUstream, lambda x: check(cuda.cuStreamCreate(ctypes.byref(x), 0)))

class Args(ctypes.Structure):
  _fields_ = [("f0", cuda.CUdeviceptr_v2)]

for trial, (kp_style, ex_style) in enumerate([("none", "extra"), ("kernelParams", "none")]):
  a = Args(0x1234)
  vargs = (ctypes.c_void_p * 5)(ctypes.c_void_p(1), ctypes.cast(ctypes.byref(a), ctypes.c_void_p),
                                ctypes.c_void_p(2), ctypes.cast(ctypes.pointer(ctypes.c_size_t(ctypes.sizeof(a))), ctypes.c_void_p),
                                ctypes.c_void_p(0))
  check(cuda.cuStreamBeginCapture_v2(stream, cuda.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL))
  kp = vargs if kp_style == "kernelParams" else None
  ex = vargs if ex_style == "extra" else None
  # three kernels with distinguishable grids, plus a fork/join event pair
  check(cuda.cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, stream, kp, ex))
  check(cuda.cuLaunchKernel(fn, 2, 1, 1, 32, 1, 1, 0, stream, kp, ex))
  ev = init_c_var(cuda.CUevent, lambda x: check(cuda.cuEventCreate(ctypes.byref(x), 0)))
  check(cuda.cuEventRecord(ev, stream))
  check(cuda.cuLaunchKernel(fn, 3, 1, 1, 32, 1, 1, 0, stream, kp, ex))
  graph = cuda.CUgraph()
  check(cuda.cuStreamEndCapture(stream, ctypes.byref(graph)))
  num = ctypes.c_size_t()
  check(cuda.cuGraphGetNodes(graph, None, ctypes.byref(num)))
  nodes = (cuda.CUgraphNode * num.value)()
  check(cuda.cuGraphGetNodes(graph, nodes, ctypes.byref(num)))
  print(f"trial {trial} ({kp_style},{ex_style}): {num.value} nodes in GetNodes order:", flush=True)
  for i in range(num.value):
    t = ctypes.c_uint()
    cuda.cuGraphNodeGetType(nodes[i], ctypes.byref(t))
    if t.value == cuda.CU_GRAPH_NODE_TYPE_KERNEL:
      kpn = cuda.CUDA_KERNEL_NODE_PARAMS_v1()
      cuda.cuGraphKernelNodeGetParams(nodes[i], ctypes.byref(kpn))
      print(f"  node{i} KERNEL grid=({kpn.gridDimX},{kpn.gridDimY},{kpn.gridDimZ}) func={ctypes.cast(kpn.func, ctypes.c_void_p).value:x}")
    else:
      print(f"  node{i} type={t.value} ({cuda.enum_CUgraphNodeType_enum.get(t.value, '?')})")
  cuda.cuGraphDestroy(graph)
