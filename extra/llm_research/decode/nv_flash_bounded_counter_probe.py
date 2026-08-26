#!/usr/bin/env python3
"""Standalone CUDA/ncu counters for current S8 versus depth-bounded S4 Flash."""
from __future__ import annotations

import argparse, csv, io, json, pathlib, re, statistics, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.llm.flash_decode_attention import flash_vec_llama_score_pv_kernel
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops, UOp

Hq, Hkv, Hd, MAXC, Tc, W = 32, 8, 128, 1024, 512, 130
NVCC, NCU = "/usr/local/cuda-13.2/bin/nvcc", "/usr/local/bin/ncu"
METRICS = ",".join((
  "dram__bytes.sum", "dram__bytes_op_read.sum", "dram__bytes_op_write.sum",
  "dram__throughput.avg.pct_of_peak_sustained_elapsed", "gpu__time_duration.sum", "lts__t_bytes.sum",
  "lts__t_sector_op_read_hit_rate.pct", "l1tex__t_bytes.sum", "sm__inst_executed.sum",
  "sm__throughput.avg.pct_of_peak_sustained_elapsed", "launch__registers_per_thread",
  "smsp__warps_active.avg.pct_of_peak_sustained_active",
  "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
  "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct"))


def _render(splits:int, token_bound:int|None) -> tuple[str, str]:
  out = UOp.placeholder((Hq*splits*W,), dtypes.float32, 0)
  q = UOp.placeholder((Hq*Hd,), dtypes.float32, 1)
  cache = UOp.placeholder((2*Hkv*MAXC*Hd//2,), dtypes.uint32, 2)
  sink = flash_vec_llama_score_pv_kernel(Hd, Hq, Hkv, MAXC, splits, UOp.const(dtypes.int, Tc),
    wide_kv=True, wide_q=False, token_bound=token_bound)(out, q, cache)
  program = to_program(sink, CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=False))
  source = next(x.arg for x in program.src if x.op is Ops.SOURCE)
  return program.arg.name, source


HARNESS = r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
__CONTROL_SOURCE__
__CANDIDATE_SOURCE__
static void ck(cudaError_t e,const char* w){if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e));exit(2);}}
static double run_control(float* o,float* q,unsigned int* c,int n){cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);for(int i=0;i<n;i++)__CONTROL__<<<dim3(8,32,1),dim3(32,4,1)>>>(o,q,c);cudaEventRecord(b);ck(cudaEventSynchronize(b),"control");float ms;cudaEventElapsedTime(&ms,a,b);return 1000.0*ms/n;}
static double run_candidate(float* o,float* q,unsigned int* c,int n){cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);for(int i=0;i<n;i++)__CANDIDATE__<<<dim3(__CAND_SPLITS__,32,1),dim3(32,4,1)>>>(o,q,c);cudaEventRecord(b);ck(cudaEventSynchronize(b),"candidate");float ms;cudaEventElapsedTime(&ms,a,b);return 1000.0*ms/n;}
int main(int ac,char**av){int n=ac>1?atoi(av[1]):400,r=ac>2?atoi(av[2]):9;float *oc,*on,*q;unsigned int*c;ck(cudaMalloc(&oc,33280*4),"oc");ck(cudaMalloc(&on,__CAND_OUT__*4),"on");ck(cudaMalloc(&q,4096*4),"q");ck(cudaMalloc(&c,1048576*4),"c");float*hq=(float*)malloc(4096*4);unsigned int*hc=(unsigned int*)malloc(1048576*4);for(int i=0;i<4096;i++)hq[i]=((i*17+3)%127-63)/256.0f;for(int i=0;i<1048576;i++)hc[i]=(i*2654435761u)^0x3c003c00u;cudaMemcpy(q,hq,4096*4,cudaMemcpyHostToDevice);cudaMemcpy(c,hc,1048576*4,cudaMemcpyHostToDevice);free(hq);free(hc);__CONTROL__<<<dim3(8,32,1),dim3(32,4,1)>>>(oc,q,c);__CANDIDATE__<<<dim3(__CAND_SPLITS__,32,1),dim3(32,4,1)>>>(on,q,c);ck(cudaDeviceSynchronize(),"warm");for(int i=0;i<r;i++)printf("rep=%d control=%.6f candidate=%.6f\n",i,run_control(oc,q,c,n),run_candidate(on,q,c,n));}
'''


def _ncu(binary:pathlib.Path, symbol:str, cache_control:str) -> list[dict[str, str]]:
  cp = subprocess.run(["sudo", "-n", NCU, "-k", symbol, "--launch-skip", "1", "--launch-count", "1",
    "--cache-control", cache_control, "--metrics", METRICS, "--csv", str(binary), "2", "1"],
    capture_output=True, text=True)
  if cp.returncode: raise RuntimeError(f"ncu {symbol}/{cache_control} failed: {cp.stderr[-5000:]}")
  rows, header = [], None
  for cols in csv.reader(io.StringIO(cp.stdout)):
    if cols and cols[0] == "ID": header = cols; continue
    if header is not None and len(cols) == len(header):
      row = dict(zip(header, cols)); rows.append({"metric":row["Metric Name"], "unit":row["Metric Unit"],
                                                  "value":row["Metric Value"]})
  return rows


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--passes", type=int, default=400)
  ap.add_argument("--reps", type=int, default=9); ap.add_argument("--ncu", action="store_true")
  ap.add_argument("--candidate-splits", type=int, default=4); ap.add_argument("--token-bound", type=int, default=512)
  ap.add_argument("--out", type=pathlib.Path, required=True); args = ap.parse_args()
  if args.token_bound != args.candidate_splits*128 or args.token_bound > MAXC:
    raise ValueError("token-bound must equal candidate-splits*128 and fit MAXC")
  control, control_src = _render(8, None); candidate, candidate_src = _render(args.candidate_splits, args.token_bound)
  candidate_src = candidate_src[candidate_src.index('extern "C"'):]
  source = HARNESS.replace("__CONTROL_SOURCE__", control_src).replace("__CANDIDATE_SOURCE__", candidate_src)
  source = source.replace("__CONTROL__", control).replace("__CANDIDATE__", candidate)
  source = source.replace("__CAND_SPLITS__", str(args.candidate_splits)).replace("__CAND_OUT__", str(Hq*args.candidate_splits*W))
  with tempfile.TemporaryDirectory(prefix="nv_flash_bounded_") as td:
    cu, binary = pathlib.Path(td)/"probe.cu", pathlib.Path(td)/"probe"
    cu.write_text(source)
    build = subprocess.run([NVCC, "-arch=sm_120a", "-O3", "-std=c++17", "--ptxas-options=-v", str(cu), "-o", str(binary)],
                           capture_output=True, text=True)
    if build.returncode: raise RuntimeError(build.stderr[-10000:])
    run = subprocess.run([str(binary), str(args.passes), str(args.reps)], capture_output=True, text=True, check=True)
    counters = {arm:{state:_ncu(binary, symbol, "none" if state == "hot" else "all") for state in ("hot", "cold")}
                for arm,symbol in (("control",control),("candidate",candidate))} if args.ncu else None
  cv, nv = [], []
  for line in run.stdout.splitlines():
    if m := re.match(r"rep=\d+ control=([0-9.]+) candidate=([0-9.]+)", line): cv.append(float(m.group(1))); nv.append(float(m.group(2)))
  payload = {"schema":"tinygrad.nv_flash_bounded_counter_probe.v1", "shape":{"Hq":Hq,"Hkv":Hkv,"Hd":Hd,"MAXC":MAXC,"Tc":Tc},
    "control":{"symbol":control,"splits":8,"token_bound":None,"samples_us":cv,"median_us":statistics.median(cv)},
    "candidate":{"symbol":candidate,"splits":args.candidate_splits,"token_bound":args.token_bound,
                 "samples_us":nv,"median_us":statistics.median(nv)},
    "ratio":statistics.median(nv)/statistics.median(cv), "ptxas":build.stderr.splitlines()}
  if counters is not None: payload["ncu"] = {"method":"one post-warmup launch; cache-control none=hot, all=cold", "arms":counters}
  args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
  print(json.dumps(payload, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
