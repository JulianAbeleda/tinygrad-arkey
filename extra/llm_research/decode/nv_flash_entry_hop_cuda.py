#!/usr/bin/env python3
"""CUDA/NCU bridge for an exact Flash entry-hop launch manifest.

Allocations are shared by captured native virtual-address identity, preserving
producer/output/cache aliasing across the replayed prefix. Each repetition is
``score reheat -> selected prefix -> score target``; NCU profiles the second
score launch with cache flushing disabled.
"""
from __future__ import annotations

import argparse, ctypes, json, pathlib, sys

ROOT=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))

from tinygrad.runtime.autogen import cuda
from tinygrad.runtime.ops_cuda import check

ARMS={
  "hot":0,
  "gate":1,
  "ffn":2,
  "ffn_provider":3,
  "through_q":4,
  "through_kv":5,
  "through_qdone":6,
  "full_entry":7,
}


def main() -> int:
  ap=argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--manifest",type=pathlib.Path,required=True); ap.add_argument("--arm",choices=ARMS,required=True)
  ap.add_argument("--reps",type=int,default=1); ap.add_argument("--out",type=pathlib.Path,required=True); args=ap.parse_args()
  rows=json.loads(args.manifest.read_text())["launch_manifest"]
  if len(rows)!=8: raise RuntimeError(f"expected eight launch records, got {len(rows)}")
  prefix=rows[:ARMS[args.arm]]; target=rows[-1]

  check(cuda.cuInit(0)); device=ctypes.c_int(0); check(cuda.cuDeviceGet(ctypes.byref(device),0))
  ctx=cuda.CUcontext(); check(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx),device)); check(cuda.cuCtxSetCurrent(ctx))
  stream=cuda.CUstream(); check(cuda.cuStreamCreate(ctypes.byref(stream),cuda.CU_STREAM_NON_BLOCKING))
  modules=[]; functions={}; allocations={}
  try:
    for row in rows:
      module=cuda.CUmodule(); function=cuda.CUfunction(); blob=pathlib.Path(row["cubin"]["path"]).read_bytes()
      check(cuda.cuModuleLoadData(ctypes.byref(module),blob)); modules.append(module)
      check(cuda.cuModuleGetFunction(ctypes.byref(function),module,row["name"].encode())); functions[row["name"]]=function
      for meta in row["buf_meta"]:
        key=int(meta["va_addr"])
        if key not in allocations:
          ptr=cuda.CUdeviceptr(); check(cuda.cuMemAlloc_v2(ctypes.byref(ptr),int(meta["size"])))
          allocations[key]=(ptr,int(meta["size"])); check(cuda.cuMemsetD8Async(ptr,0,int(meta["size"]),stream))
    check(cuda.cuStreamSynchronize(stream))

    def launch(row):
      ptr_holders=[allocations[int(meta["va_addr"])][0] for meta in row["buf_meta"]]
      scalar_holders=[ctypes.c_int32(int(v)) for v in row["vals"]]
      params=(ctypes.c_void_p*(len(ptr_holders)+len(scalar_holders)))(
        *[ctypes.cast(ctypes.pointer(x),ctypes.c_void_p) for x in ptr_holders],
        *[ctypes.cast(ctypes.pointer(x),ctypes.c_void_p) for x in scalar_holders])
      g,b=row["global_size"],row["local_size"]
      check(cuda.cuLaunchKernel(functions[row["name"]],g[0],g[1],g[2],b[0],b[1],b[2],0,stream,params,None))

    for _ in range(args.reps):
      launch(target)
      for row in prefix: launch(row)
      launch(target)
    check(cuda.cuStreamSynchronize(stream))
  finally:
    for ptr,_ in allocations.values(): cuda.cuMemFree_v2(ptr)
    for module in modules: cuda.cuModuleUnload(module)
    cuda.cuStreamDestroy_v2(stream); cuda.cuDevicePrimaryCtxRelease(device)
  payload={"schema":"tinygrad.nv_flash_entry_hop_cuda.v1","arm":args.arm,"prefix_names":[x["name"] for x in prefix],
    "target":target["name"],"reps":args.reps,"allocation_count":len(allocations),"verdict":"CUDA_SEQUENCE_OK"}
  args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
  print(json.dumps(payload,indent=2,sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
