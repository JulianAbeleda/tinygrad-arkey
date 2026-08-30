#!/usr/bin/env python3
"""Bit-exact hot timing and cold counters for Q4_K four-warp FFN-down vector loads."""
from __future__ import annotations

import argparse, csv, io, json, os, re, statistics, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, execute_research_program
from tinygrad.llm.q4k_ffn_down_mmvq import ROWS, K, Q4_BLOCKS, SUB_BLOCKS, emit_four_warp_fp16_direct
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops, UOp
from extra.llm_research.decode.route_class_numerics import _make_q4k_words

CUDA_BIN = "/usr/local/cuda-13.2/bin"
NCU = "/usr/local/bin/ncu"
SCALAR = "q4k_fp16_mmvq_direct_4096_12288_epi_ffnresadd"
VECTOR = "q4k_fp16_mmvq_direct_vec_4096_12288_epi_ffnresadd"


def _args() -> tuple[UOp, UOp, UOp, UOp]:
  return (UOp.placeholder((ROWS,), dtypes.float32, 0),
          UOp.placeholder((ROWS * Q4_BLOCKS * 36,), dtypes.uint32, 1),
          UOp.placeholder((K,), dtypes.float16, 2),
          UOp.placeholder((ROWS,), dtypes.float32, 3))


def _emitter(style:str):
  return emit_four_warp_fp16_direct(UOp.const(dtypes.weakint, SUB_BLOCKS), resadd=True, load_style=style)


def _native_exact() -> dict:
  dev = Device.DEFAULT
  if not str(dev).startswith("NV"): raise RuntimeError(f"native NV required, got {dev}")
  words_np, raw = _make_q4k_words(ROWS, K, 202608241)
  rng = np.random.default_rng(202608242)
  x_np = rng.normal(0, 0.2, K).astype(np.float16)
  h_np = rng.normal(0, 0.05, ROWS).astype(np.float32)
  words = Tensor(words_np, dtype=dtypes.uint32, device=dev).contiguous().realize()
  x = Tensor(x_np, dtype=dtypes.float16, device=dev).contiguous().realize()
  h = Tensor(h_np, dtype=dtypes.float32, device=dev).contiguous().realize()

  def run(style:str) -> np.ndarray:
    program = KernelProgram("research.q4k_ffn_down_vector_load", style,
      KernelProgramProvenance.RESEARCH_ONLY, _emitter(style))
    return execute_research_program(Tensor.empty((ROWS,), dtype=dtypes.float32, device=dev),
      words, x, h, program=program).realize().numpy()

  scalar, vector = run("scalar"), run("vector")
  Device[dev].synchronize()
  return {"bitwise_identical": bool(np.array_equal(scalar.view(np.uint32), vector.view(np.uint32))),
          "max_abs_diff": float(np.max(np.abs(scalar - vector))),
          "finite": bool(np.isfinite(scalar).all() and np.isfinite(vector).all()),
          "q4_sha256": __import__("hashlib").sha256(raw.tobytes()).hexdigest(),
          "x_sha256": __import__("hashlib").sha256(x_np.tobytes()).hexdigest()}


def _render() -> dict[str, str]:
  ren = CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=False)
  def source(style:str) -> str:
    ast = _emitter(style)(*_args())
    text = next(x.arg for x in to_program(ast, ren).src if x.op is Ops.SOURCE)
    return text[text.index('extern "C" __global__'):]
  return {"scalar": source("scalar"), "vector": source("vector")}


HARNESS = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#ifndef INFINITY
#define INFINITY (__int_as_float(0x7f800000))
#endif
#ifndef NAN
#define NAN (__int_as_float(0x7fffffff))
#endif
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f=v; return u.t; }
struct __align__(8) half4 { half x, y, z, w; };
__device__ half4 make_half4(half x, half y, half z, half w) { half4 r={x,y,z,w}; return r; }

__SCALAR_SRC__
__VECTOR_SRC__

static void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); exit(2); }
}

static double time_scalar(float* out, unsigned int* w, half* x, float* h, int passes) {
  cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); cudaEventRecord(s);
  for (int i=0;i<passes;i++) __SCALAR__<<<4096,128>>>(out,w,x,h);
  cudaEventRecord(e); check(cudaDeviceSynchronize(),"sync scalar"); float ms=0; cudaEventElapsedTime(&ms,s,e);
  cudaEventDestroy(s); cudaEventDestroy(e); return ms*1000.0/passes;
}
static double time_vector(float* out, unsigned int* w, half* x, float* h, int passes) {
  cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); cudaEventRecord(s);
  for (int i=0;i<passes;i++) __VECTOR__<<<4096,128>>>(out,w,x,h);
  cudaEventRecord(e); check(cudaDeviceSynchronize(),"sync vector"); float ms=0; cudaEventElapsedTime(&ms,s,e);
  cudaEventDestroy(s); cudaEventDestroy(e); return ms*1000.0/passes;
}

int main(int argc,char** argv) {
  int passes=argc>1?atoi(argv[1]):100, reps=argc>2?atoi(argv[2]):7;
  float *a=nullptr,*b=nullptr,*h=nullptr; unsigned int* w=nullptr; half* x=nullptr;
  check(cudaMalloc(&a,4096*sizeof(float)),"a"); check(cudaMalloc(&b,4096*sizeof(float)),"b");
  check(cudaMalloc(&w,7077888*sizeof(unsigned int)),"w"); check(cudaMalloc(&x,12288*sizeof(half)),"x");
  check(cudaMalloc(&h,4096*sizeof(float)),"h");
  check(cudaMemset(a,0,4096*sizeof(float)),"zero a"); check(cudaMemset(b,0,4096*sizeof(float)),"zero b");
  check(cudaMemset(w,0,7077888*sizeof(unsigned int)),"zero w"); check(cudaMemset(x,0,12288*sizeof(half)),"zero x");
  check(cudaMemset(h,0,4096*sizeof(float)),"zero h");
  __SCALAR__<<<4096,128>>>(a,w,x,h); __VECTOR__<<<4096,128>>>(b,w,x,h);
  check(cudaDeviceSynchronize(),"warmup");
  float ha[4096],hb[4096]; check(cudaMemcpy(ha,a,sizeof(ha),cudaMemcpyDeviceToHost),"copy a");
  check(cudaMemcpy(hb,b,sizeof(hb),cudaMemcpyDeviceToHost),"copy b");
  printf("zero_input_bitwise=%d\n",memcmp(ha,hb,sizeof(ha))==0);
  for (int r=0;r<reps;r++) printf("rep=%d scalar=%.4f vector=%.4f\n",r,time_scalar(a,w,x,h,passes),time_vector(b,w,x,h,passes));
  return 0;
}
"""


def _ncu(binary:str, symbol:str) -> list[dict[str,str]]:
  metrics = ",".join(["dram__bytes.sum", "dram__bytes_op_read.sum", "dram__bytes_op_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed", "gpu__time_duration.sum",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed", "lts__t_bytes.sum",
    "lts__t_sector_op_read_hit_rate.pct", "sm__inst_executed.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed", "launch__registers_per_thread"])
  cp = subprocess.run(["sudo","-n",NCU,"-k",symbol,"--launch-skip","1","--launch-count","1",
    "--cache-control","all","--metrics",metrics,"--csv",binary,"1","1"], capture_output=True, text=True)
  if cp.returncode: raise RuntimeError(f"ncu {symbol} failed: {cp.stderr[-4000:]}")
  rows, header = [], None
  for cols in csv.reader(io.StringIO(cp.stdout)):
    if cols and cols[0] == "ID": header=cols; continue
    if header is not None and len(cols)==len(header):
      row=dict(zip(header,cols)); rows.append({"metric":row["Metric Name"],"unit":row["Metric Unit"],"value":row["Metric Value"]})
  return rows


def main() -> int:
  ap=argparse.ArgumentParser(); ap.add_argument("--passes",type=int,default=100); ap.add_argument("--reps",type=int,default=7)
  ap.add_argument("--ncu",action="store_true"); ap.add_argument("--out",type=Path,required=True); args=ap.parse_args()
  exact=_native_exact()
  if not exact["bitwise_identical"] or not exact["finite"]: raise RuntimeError(f"exact gate failed: {exact}")
  src=_render(); cu=(HARNESS.replace("__SCALAR_SRC__",src["scalar"]).replace("__VECTOR_SRC__",src["vector"])
    .replace("__SCALAR__",SCALAR).replace("__VECTOR__",VECTOR))
  with tempfile.TemporaryDirectory(prefix="q4k_down_vec_") as td:
    cupath, binary = os.path.join(td,"down.cu"), os.path.join(td,"down")
    Path(cupath).write_text(cu)
    env={**os.environ,"PATH":f"{CUDA_BIN}:"+os.environ.get("PATH","")}
    cp=subprocess.run(["nvcc","-arch=sm_120a","-O3","-std=c++17","--ptxas-options=-v",cupath,"-o",binary],
      capture_output=True,text=True,env=env)
    if cp.returncode: raise RuntimeError(cp.stderr[-8000:])
    run=subprocess.run([binary,str(args.passes),str(args.reps)],capture_output=True,text=True,check=True)
    scalar,vector=[],[]
    for line in run.stdout.splitlines():
      if (m:=re.match(r"rep=\d+ scalar=([0-9.]+) vector=([0-9.]+)",line)):
        scalar.append(float(m.group(1))); vector.append(float(m.group(2)))
    result={"schema":"tinygrad.q4k_ffn_down_vector_load_microgate.v1","native_exact":exact,
      "method":"production emitter -> rendered CUDA -> nvcc sm_120a",
      "shape":{"rows":ROWS,"k":K,"grid":4096,"block":128},
      "timing":{"unit":"us_per_launch_cuda_event","scalar":scalar,"vector":vector,
        "scalar_median":statistics.median(scalar),"vector_median":statistics.median(vector),
        "recovery_us":statistics.median(scalar)-statistics.median(vector)},
      "ptxas":cp.stderr.strip().splitlines(),"raw_stdout":run.stdout.strip().splitlines()}
    if args.ncu:
      result["ncu"]={"method":"Nsight Compute --cache-control all; one post-warmup launch",
        "scalar":{"symbol":SCALAR,"rows":_ncu(binary,SCALAR)},
        "vector":{"symbol":VECTOR,"rows":_ncu(binary,VECTOR)}}
  args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
  print(json.dumps(result,indent=2,sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
