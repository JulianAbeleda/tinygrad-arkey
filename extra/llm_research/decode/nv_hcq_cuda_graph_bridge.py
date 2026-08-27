#!/usr/bin/env python3
"""Matched CUDA-launch/CUDA-graph replay for the HCQ dispatch-slope cubin.

Run separately from ``nv_hcq_dispatch_slope.py``: both consume the exact same
production cubin, launch geometry, buffer sizes, and serial dependency chain.
The output permits a native-HCQ versus CUDA-front-end drain-slope comparison
without changing kernel code or model semantics.
"""
from __future__ import annotations

import argparse, ctypes, hashlib, json, pathlib, statistics, subprocess

import sys
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tinygrad.runtime.autogen import cuda
from tinygrad.runtime.ops_cuda import check, encode_args

CUBIN = ROOT / "docs/task_workflow/evidence/nv-qk-head-norm-predecessor-20260822/reduce_output_rmsnorm_8_128.cubin"
SYMBOL = b"reduce_output_rmsnorm_8_128"
GRID, BLOCK = (8, 1, 1), (2, 16, 1)
NS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def slope(xs: list[int], ys: list[float]) -> tuple[float, float]:
  mx, my = statistics.mean(xs), statistics.mean(ys)
  den = sum((x-mx)**2 for x in xs)
  m = sum((x-mx)*(y-my) for x,y in zip(xs,ys))/den
  return m, my-m*mx


def event_time_us(stream, fn, reps: int) -> list[float]:
  out = []
  for _ in range(reps):
    st, en = cuda.CUevent(), cuda.CUevent()
    check(cuda.cuEventCreate(ctypes.byref(st), 0)); check(cuda.cuEventCreate(ctypes.byref(en), 0))
    check(cuda.cuEventRecord(st, stream)); fn(); check(cuda.cuEventRecord(en, stream)); check(cuda.cuEventSynchronize(en))
    ms = ctypes.c_float(); check(cuda.cuEventElapsedTime(ctypes.byref(ms), st, en))
    check(cuda.cuEventDestroy_v2(st)); check(cuda.cuEventDestroy_v2(en)); out.append(ms.value*1000.0)
  return out


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--warmup", type=int, default=5)
  ap.add_argument("--reps", type=int, default=15)
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()

  check(cuda.cuInit(0)); dev = ctypes.c_int(); check(cuda.cuDeviceGet(ctypes.byref(dev), 0))
  ctx = cuda.CUcontext(); check(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)); check(cuda.cuCtxSetCurrent(ctx))
  module, fn, stream = cuda.CUmodule(), cuda.CUfunction(), cuda.CUstream()
  graphs: list[tuple[cuda.CUgraph, cuda.CUgraphExec]] = []
  bufs: list[cuda.CUdeviceptr] = []
  try:
    blob = CUBIN.read_bytes(); check(cuda.cuModuleLoadData(ctypes.byref(module), blob)); check(cuda.cuModuleGetFunction(ctypes.byref(fn), module, SYMBOL))
    check(cuda.cuStreamCreate(ctypes.byref(stream), cuda.CU_STREAM_NON_BLOCKING))
    for _ in range(3):
      p = cuda.CUdeviceptr(); check(cuda.cuMemAlloc_v2(ctypes.byref(p), 4096)); check(cuda.cuMemsetD8_v2(p, 0, 4096)); bufs.append(p)
    params = (ctypes.c_void_p*3)(*[ctypes.cast(ctypes.pointer(p), ctypes.c_void_p) for p in bufs])
    c_args, vargs = encode_args(bufs, [])

    rows = []
    for n in NS:
      graph, instance = cuda.CUgraph(), cuda.CUgraphExec(); check(cuda.cuGraphCreate(ctypes.byref(graph), 0))
      prior = None
      # Keep params alive for the lifetime of the graph even though every node
      # has the same immutable pointer tuple.
      node_params = []
      for _ in range(n):
        node = cuda.CUgraphNode()
        deps = None if prior is None else (cuda.CUgraphNode*1)(prior)
        kp = cuda.CUDA_KERNEL_NODE_PARAMS_v1(fn, *GRID, *BLOCK, 0, ctypes.cast(0, ctypes.POINTER(ctypes.c_void_p)), vargs)
        check(cuda.cuGraphAddKernelNode(ctypes.byref(node), graph, deps, 0 if prior is None else 1, ctypes.byref(kp)))
        node_params.append(kp); prior = node
      check(cuda.cuGraphInstantiate_v2(ctypes.byref(instance), graph, None, None, 0)); graphs.append((graph, instance))

      def ordinary():
        for _ in range(n): check(cuda.cuLaunchKernel(fn, *GRID, *BLOCK, 0, stream, params, None))
      def graphed(): check(cuda.cuGraphLaunch(instance, stream))
      for launch in (ordinary, graphed):
        for _ in range(args.warmup): launch()
        check(cuda.cuStreamSynchronize(stream))
      ordinary_us = event_time_us(stream, ordinary, args.reps)
      graph_us = event_time_us(stream, graphed, args.reps)
      rows.append({"n":n, "ordinary_median_us":statistics.median(ordinary_us), "graph_median_us":statistics.median(graph_us),
                   "ordinary_samples_us":ordinary_us, "graph_samples_us":graph_us})

    om, oi = slope([r["n"] for r in rows], [r["ordinary_median_us"] for r in rows])
    gm, gi = slope([r["n"] for r in rows], [r["graph_median_us"] for r in rows])
    result = {"schema":"tinygrad.nv_hcq_cuda_graph_bridge.v1", "commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
      "cubin":str(CUBIN.relative_to(ROOT)), "cubin_sha256":hashlib.sha256(blob).hexdigest(), "symbol":SYMBOL.decode(),
      "grid":GRID, "block":BLOCK, "warmup":args.warmup, "reps":args.reps, "rows":rows,
      "slopes":{"ordinary_cuda_us_per_kernel":om,"ordinary_intercept_us":oi,"cuda_graph_us_per_kernel":gm,"cuda_graph_intercept_us":gi}}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"rows":[{k:v for k,v in r.items() if "samples" not in k} for r in rows],"slopes":result["slopes"]},indent=2))
  finally:
    for graph,instance in graphs: check(cuda.cuGraphExecDestroy(instance)); check(cuda.cuGraphDestroy(graph))
    for p in bufs: check(cuda.cuMemFree_v2(p))
    if stream: check(cuda.cuStreamDestroy_v2(stream))
    if module: check(cuda.cuModuleUnload(module))
    check(cuda.cuDevicePrimaryCtxRelease(dev))
  return 0


if __name__ == "__main__": raise SystemExit(main())
