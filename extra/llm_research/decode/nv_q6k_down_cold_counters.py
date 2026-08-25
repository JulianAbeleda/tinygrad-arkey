#!/usr/bin/env python3
"""Cold counters for the current packed-lane Q6_K dense FFN-down kernel."""
from __future__ import annotations

import argparse, csv, io, json, os, pathlib, re, statistics, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.llm.q6k_ffn_down_mmvq import ROWS, K, emit_q6k_four_warp_fp16_direct
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops, UOp

CUDA_BIN = "/usr/local/cuda-13.2/bin"
NCU = "/usr/local/bin/ncu"
CONTROL_SYMBOL = "q6k_fp16_packed_lanemap_4096_12288_epi_ffnresadd"
HALFWORDS = ROWS * (K // 256) * 105


def symbol(unroll:int|None) -> str:
  return CONTROL_SYMBOL if unroll is None else f"q6k_fp16_packed_lanemap_u{unroll}_4096_12288_epi_ffnresadd"


def render(unroll:int|None=None) -> str:
  out = UOp.placeholder((ROWS,), dtypes.float32, 0)
  halfs = UOp.placeholder((HALFWORDS,), dtypes.uint16, 1)
  x = UOp.placeholder((K,), dtypes.float16, 2)
  h = UOp.placeholder((ROWS,), dtypes.float32, 3)
  ast = emit_q6k_four_warp_fp16_direct(packed_lanemap=True, unroll_blocks=unroll)(out, halfs, x, h)
  src = next(x.arg for x in to_program(ast, CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=False)).src
             if x.op is Ops.SOURCE)
  return src[src.index('extern "C" __global__'):]


HARNESS = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#ifndef INFINITY
#define INFINITY (__int_as_float(0x7f800000))
#endif
#ifndef NAN
#define NAN (__int_as_float(0x7fffffff))
#endif
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f=v; return u.t; }
struct __align__(8) half4 { half x, y, z, w; };
__device__ half4 make_half4(half x, half y, half z, half w) { half4 r={x,y,z,w}; return r; }
__SRC__
static void check(cudaError_t e,const char* what) { if(e!=cudaSuccess) { fprintf(stderr,"%s: %s\n",what,cudaGetErrorString(e)); exit(2); } }
int main(int argc,char** argv) {
  int passes=argc>1?atoi(argv[1]):50, reps=argc>2?atoi(argv[2]):5;
  float *out=nullptr,*h=nullptr; unsigned short* w=nullptr; half* x=nullptr;
  check(cudaMalloc(&out,ROWS_ARG*sizeof(float)),"out"); check(cudaMalloc(&h,ROWS_ARG*sizeof(float)),"h");
  check(cudaMalloc(&w,HALFWORDS_ARG*sizeof(unsigned short)),"w"); check(cudaMalloc(&x,K_ARG*sizeof(half)),"x");
  check(cudaMemset(out,0,ROWS_ARG*sizeof(float)),"zero out"); check(cudaMemset(h,0,ROWS_ARG*sizeof(float)),"zero h");
  check(cudaMemset(w,0,HALFWORDS_ARG*sizeof(unsigned short)),"zero w"); check(cudaMemset(x,0,K_ARG*sizeof(half)),"zero x");
  SYMBOL_ARG<<<ROWS_ARG,128>>>(out,w,x,h); check(cudaDeviceSynchronize(),"warmup");
  for(int r=0;r<reps;r++) {
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); cudaEventRecord(s);
    for(int i=0;i<passes;i++) SYMBOL_ARG<<<ROWS_ARG,128>>>(out,w,x,h);
    cudaEventRecord(e); check(cudaDeviceSynchronize(),"sync"); float ms=0; cudaEventElapsedTime(&ms,s,e);
    printf("rep=%d current=%.4f\n",r,ms*1000.0f/passes); cudaEventDestroy(s); cudaEventDestroy(e);
  }
  return 0;
}
"""


def ncu(binary: str, kernel_symbol:str) -> list[dict[str, str]]:
  metrics = ",".join([
    "dram__bytes.sum", "dram__bytes_op_read.sum", "dram__bytes_op_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed", "gpu__time_duration.sum",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed", "lts__t_bytes.sum",
    "lts__t_sector_op_read_hit_rate.pct", "sm__inst_executed.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed", "launch__registers_per_thread",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_wait_per_warp_active.pct",
  ])
  cp = subprocess.run(["sudo", "-n", NCU, "-k", kernel_symbol, "--launch-skip", "1", "--launch-count", "1",
    "--cache-control", "all", "--metrics", metrics, "--csv", binary, "1", "1"],
    capture_output=True, text=True)
  if cp.returncode: raise RuntimeError(cp.stderr[-4000:])
  rows, header = [], None
  for cols in csv.reader(io.StringIO(cp.stdout)):
    if cols and cols[0] == "ID": header = cols; continue
    if header is not None and len(cols) == len(header):
      row = dict(zip(header, cols)); rows.append({"metric":row["Metric Name"], "unit":row["Metric Unit"], "value":row["Metric Value"]})
  return rows


def main() -> int:
  ap=argparse.ArgumentParser(); ap.add_argument("--passes",type=int,default=50); ap.add_argument("--reps",type=int,default=5)
  ap.add_argument("--unroll",type=int,choices=(2,3,4,6,12)); ap.add_argument("--ncu",action="store_true")
  ap.add_argument("--out",type=pathlib.Path,required=True); args=ap.parse_args()
  kernel_symbol=symbol(args.unroll)
  cu = HARNESS.replace("__SRC__",render(args.unroll)).replace("ROWS_ARG",str(ROWS)).replace("K_ARG",str(K)) \
    .replace("HALFWORDS_ARG",str(HALFWORDS)).replace("SYMBOL_ARG",kernel_symbol)
  with tempfile.TemporaryDirectory(prefix="q6k_down_cold_") as td:
    cupath,binary=pathlib.Path(td)/"q6.cu",pathlib.Path(td)/"q6"
    cupath.write_text(cu)
    env={**os.environ,"PATH":f"{CUDA_BIN}:"+os.environ.get("PATH","")}
    cp=subprocess.run(["nvcc","-arch=sm_120a","-O3","-std=c++17","--ptxas-options=-v",str(cupath),"-o",str(binary)],
      capture_output=True,text=True,env=env)
    if cp.returncode: raise RuntimeError(cp.stderr[-8000:])
    run=subprocess.run([str(binary),str(args.passes),str(args.reps)],capture_output=True,text=True,check=True)
    samples=[float(m.group(1)) for line in run.stdout.splitlines() if (m:=re.match(r"rep=\d+ current=([0-9.]+)",line))]
    result={"schema":"tinygrad.nv_q6k_down_cold_counters.v1","commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
      "symbol":kernel_symbol,"unroll":args.unroll,"shape":{"rows":ROWS,"k":K,"grid":ROWS,"block":128},"weight_halfwords":HALFWORDS,
      "timing":{"unit":"us_per_launch_cuda_event","samples":samples,"median":statistics.median(samples)},
      "ptxas":cp.stderr.strip().splitlines()}
    if args.ncu: result["ncu"]={"method":"Nsight Compute --cache-control all; one post-warmup launch","rows":ncu(str(binary),kernel_symbol)}
  args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
  print(json.dumps(result,indent=2,sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
