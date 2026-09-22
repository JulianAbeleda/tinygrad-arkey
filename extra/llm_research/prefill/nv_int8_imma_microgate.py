#!/usr/bin/env python3
"""Research-only signed-int8 TensorCore execution and cubin capture gate."""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, time
import numpy as np


def main() -> None:
  parser=argparse.ArgumentParser()
  parser.add_argument("--out", required=True)
  parser.add_argument("--artifacts", required=True)
  parser.add_argument("--rounds", type=int, default=9)
  args=parser.parse_args()
  # TC=1 asks the normal optimizer to select an exact tensor-core descriptor.
  os.environ.setdefault("TC", "1")
  os.environ.setdefault("DEV", "NV")

  from tinygrad import Device, Tensor, dtypes
  from tinygrad.renderer.cuda import CUDARenderer
  from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler

  artifacts=pathlib.Path(args.artifacts); artifacts.mkdir(parents=True, exist_ok=True)
  rendered, cubins = [], []
  old_render, old_compile = CUDARenderer.render, NVRTCCompiler.compile
  def capture_render(self, uops):
    source=old_render(self,uops); rendered.append(source); return source
  def capture_compile(self, source):
    binary=old_compile(self,source); cubins.append(binary); return binary
  CUDARenderer.render, NVRTCCompiler.compile = capture_render, capture_compile
  try:
    rng=np.random.default_rng(20260828)
    av=rng.integers(-127,128,(16,32),dtype=np.int8)
    bv=rng.integers(-127,128,(32,8),dtype=np.int8)
    a,b=Tensor(av),Tensor(bv)
    out=a.dot(b,dtype=dtypes.int).realize()
    ref=av.astype(np.int32) @ bv.astype(np.int32)
    got=out.numpy()
    samples=[]
    for _ in range(args.rounds):
      Device["NV"].synchronize(); st=time.perf_counter_ns()
      a.dot(b,dtype=dtypes.int).realize(); Device["NV"].synchronize()
      samples.append((time.perf_counter_ns()-st)/1e3)
  finally:
    CUDARenderer.render, NVRTCCompiler.compile = old_render, old_compile

  sources=[s for s in rendered if "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32" in s]
  if not sources: raise RuntimeError("execution did not render signed-int8 IMMA")
  source_path=artifacts/"int8_imma.cu"; source_path.write_text(sources[-1])
  # Compile the captured execution source explicitly as well: runtime program
  # caching can legitimately bypass NVRTCCompiler.compile on a hot run.
  binary=old_compile(NVRTCCompiler("sm_120", ptx=False, cache_key="nv_int8_imma_microgate"), sources[-1])
  cubin_path=artifacts/"int8_imma.cubin"; cubin_path.write_bytes(binary)
  diff=np.abs(got.astype(np.int64)-ref.astype(np.int64))
  result={"schema":"tinygrad.nv_int8_imma_microgate.v1", "shape":[16,8,32],
    "finite":bool(np.isfinite(got).all()), "exact":bool(np.array_equal(got,ref)),
    "max_abs":int(diff.max()), "output_sha256":hashlib.sha256(got.tobytes()).hexdigest(),
    "samples_us":samples, "min_us":min(samples), "median_us":float(np.median(samples)),
    "source":str(source_path), "cubin":str(cubin_path), "production_route":False}
  out_path=pathlib.Path(args.out); out_path.parent.mkdir(parents=True,exist_ok=True)
  out_path.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,sort_keys=True))


if __name__ == "__main__": main()
