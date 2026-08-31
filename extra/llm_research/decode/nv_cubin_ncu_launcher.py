#!/usr/bin/env python3
"""Load a captured production NV cubin through the CUDA driver API.

The production NV backend submits QMDs directly through the NV ioctl path, so
Nsight Compute reports no kernels. This standalone harness loads the exact
cubin with cuModuleLoad and launches it with cuLaunchKernel in a normal CUDA
context, which makes the kernel visible to NCU. It is measurement tooling only
and never participates in a production route.
"""
from __future__ import annotations

import argparse, ctypes, hashlib, json, pathlib

import sys
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tinygrad.runtime.autogen import cuda
from tinygrad.runtime.ops_cuda import check


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--cubin", type=pathlib.Path, required=True)
  ap.add_argument("--symbol", required=True)
  ap.add_argument("--grid-x", type=int, default=12288)
  ap.add_argument("--block-x", type=int, default=32)
  ap.add_argument("--grid", help="comma-separated grid x,y,z (overrides --grid-x)")
  ap.add_argument("--block", help="comma-separated block x,y,z (overrides --block-x)")
  ap.add_argument("--shared-mem", type=int, default=0, help="dynamic shared-memory bytes")
  ap.add_argument("--n-bufs", type=int, default=4)
  ap.add_argument("--buf-bytes", type=int, default=64 << 20)
  ap.add_argument("--buf-sizes", help="comma-separated exact byte sizes, one per buffer")
  ap.add_argument("--vals", help="comma-separated integer scalar kernel arguments (after pointers)")
  ap.add_argument("--val-groups", help="comma-separated scalar parameter widths; use 3 for uint3")
  ap.add_argument("--reps", type=int, default=3)
  ap.add_argument("--warmup", type=int, default=20)
  ap.add_argument("--condition-mib", type=int, default=0,
                  help="between a target reheat and timed target, stream this many MiB")
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()

  buf_sizes = [int(x) for x in args.buf_sizes.split(",")] if args.buf_sizes else None
  if buf_sizes is not None:
    if len(buf_sizes) != args.n_bufs:
      raise SystemExit(f"--buf-sizes has {len(buf_sizes)} entries but --n-bufs is {args.n_bufs}")

  grid = [int(x) for x in args.grid.split(",")] if args.grid else [args.grid_x, 1, 1]
  block = [int(x) for x in args.block.split(",")] if args.block else [args.block_x, 1, 1]
  if len(grid) != 3 or len(block) != 3:
    raise SystemExit("--grid and --block must be comma-separated x,y,z triples")

  blob = args.cubin.read_bytes()
  check(cuda.cuInit(0))
  device = ctypes.c_int(0)
  check(cuda.cuDeviceGet(ctypes.byref(device), 0))
  ctx = cuda.CUcontext()
  # cuMemAlloc_v2 rejects a non-primary context on this driver (error 201), so
  # retain the primary context like a normal CUDA runtime launch.
  check(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), device))
  check(cuda.cuCtxSetCurrent(ctx))

  module = cuda.CUmodule()
  function = cuda.CUfunction()
  condition_module = cuda.CUmodule()
  try:
    check(cuda.cuModuleLoadData(ctypes.byref(module), blob))
    check(cuda.cuModuleGetFunction(ctypes.byref(function), module, args.symbol.encode()))
    if args.shared_mem:
      check(cuda.cuFuncSetAttribute(function, cuda.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, args.shared_mem))

    bufs = []
    for i in range(args.n_bufs):
      ptr = cuda.CUdeviceptr()
      check(cuda.cuMemAlloc_v2(ctypes.byref(ptr), buf_sizes[i] if buf_sizes else args.buf_bytes))
      check(cuda.cuMemsetD8_v2(ptr, 0, buf_sizes[i] if buf_sizes else args.buf_bytes))
      bufs.append(ptr)

    # Pointer parameters (EIATTR ordinals 0..n-1), then any scalar int args.
    vals = [int(x) for x in args.vals.split(",")] if args.vals else []
    groups = [int(x) for x in args.val_groups.split(",")] if args.val_groups else [1] * len(vals)
    if sum(groups) != len(vals): raise SystemExit(f"--val-groups covers {sum(groups)} values, expected {len(vals)}")
    scalar_holders, scalar_ptrs, cursor = [], [], 0
    for width in groups:
      holder = (ctypes.c_int32 * width)(*vals[cursor:cursor+width]); cursor += width; scalar_holders.append(holder)
      scalar_ptrs.append(ctypes.cast(holder, ctypes.c_void_p))
    params = (ctypes.c_void_p * (args.n_bufs + len(groups)))(
      *[ctypes.cast(ctypes.pointer(b), ctypes.c_void_p) for b in bufs],
      *scalar_ptrs,
    )
    stream = cuda.CUstream()
    check(cuda.cuStreamCreate(ctypes.byref(stream), cuda.CU_STREAM_NON_BLOCKING))
    try:
      condition_function, condition_bufs, condition_params = None, [], None
      if args.condition_mib:
        from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
        condition_blob = NVRTCCompiler("sm_120a", ptx=False, cache_key="flash_common_condition").compile(r'''
extern "C" __global__ void flash_common_condition(const float *src, unsigned long long words, float *sink) {
  unsigned long long i = (unsigned long long)blockIdx.x*blockDim.x + threadIdx.x;
  if (i < words) { float v = src[i]; if (v < 0.0f) sink[0] = v; }
}''')
        check(cuda.cuModuleLoadData(ctypes.byref(condition_module), condition_blob))
        condition_function = cuda.CUfunction()
        check(cuda.cuModuleGetFunction(ctypes.byref(condition_function), condition_module, b"flash_common_condition"))
        for size in (args.condition_mib << 20, 4):
          p = cuda.CUdeviceptr(); check(cuda.cuMemAlloc_v2(ctypes.byref(p), size)); condition_bufs.append(p)
          check(cuda.cuMemsetD8_v2(p, 0, size))
        words = ctypes.c_uint64((args.condition_mib << 20)//4)
        condition_params = (ctypes.c_void_p * 3)(ctypes.cast(ctypes.pointer(condition_bufs[0]), ctypes.c_void_p),
          ctypes.cast(ctypes.pointer(words), ctypes.c_void_p), ctypes.cast(ctypes.pointer(condition_bufs[1]), ctypes.c_void_p))
      def launch_sequence():
        check(cuda.cuLaunchKernel(function, grid[0], grid[1], grid[2], block[0], block[1], block[2], args.shared_mem, stream, params, None))
        if condition_function is not None:
          blocks = (((args.condition_mib << 20)//4 + 255)//256)
          check(cuda.cuLaunchKernel(condition_function, blocks, 1, 1, 256, 1, 1, 0, stream, condition_params, None))
          check(cuda.cuLaunchKernel(function, grid[0], grid[1], grid[2], block[0], block[1], block[2], args.shared_mem, stream, params, None))
      for _ in range(args.warmup):
        launch_sequence()
      check(cuda.cuStreamSynchronize(stream))
      begin, end = cuda.CUevent(), cuda.CUevent()
      check(cuda.cuEventCreate(ctypes.byref(begin), 0)); check(cuda.cuEventCreate(ctypes.byref(end), 0))
      check(cuda.cuEventRecord(begin, stream))
      for _ in range(args.reps):
        launch_sequence()
      check(cuda.cuEventRecord(end, stream)); check(cuda.cuEventSynchronize(end))
      elapsed_ms = ctypes.c_float()
      check(cuda.cuEventElapsedTime(ctypes.byref(elapsed_ms), begin, end))
      cuda.cuEventDestroy_v2(begin); cuda.cuEventDestroy_v2(end)
      for p in condition_bufs: cuda.cuMemFree_v2(p)
    finally:
      cuda.cuStreamDestroy_v2(stream)
    for b in bufs:
      cuda.cuMemFree_v2(b)

    result = {
      "schema": "tinygrad.nv_cubin_ncu_launcher.v1",
      "cubin": str(args.cubin),
      "cubin_sha256": hashlib.sha256(blob).hexdigest(),
      "symbol": args.symbol,
      "grid": grid,
      "block": block,
      "shared_mem": args.shared_mem,
      "n_bufs": args.n_bufs,
      "buf_bytes": args.buf_bytes,
      "buf_sizes": buf_sizes or [args.buf_bytes] * args.n_bufs,
      "vals": vals,
      "val_groups": groups,
      "reps": args.reps,
      "warmup": args.warmup,
      "condition_mib": args.condition_mib,
      "event_us_per_launch": 1000.0 * float(elapsed_ms.value) / args.reps,
      "verdict": "CUDA_LAUNCH_OK",
    }
  finally:
    if condition_module: cuda.cuModuleUnload(condition_module)
    cuda.cuModuleUnload(module)
    cuda.cuDevicePrimaryCtxRelease(device)

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
