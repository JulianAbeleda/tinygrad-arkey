#!/usr/bin/env python3
"""Bit-exact isolated geometry sweep for the promoted 1x4096 native RMSNorm body."""
from __future__ import annotations

import argparse, json, pathlib, re, sqlite3, statistics, subprocess, sys, tempfile, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.llm.decode_kernels import DecodeRMSNormSpec, emit_decode_rmsnorm_kernel
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec, execute_research_program
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops, UOp

WARPS = (1, 2, 4, 8, 16)

def _program(warps:int, retain_input:bool=False) -> KernelProgram:
  spec = DecodeRMSNormSpec(rows=1, dim=4096, eps=1e-6, warps_per_row=warps,
    x_dtype=dtypes.float16, weight_dtype=dtypes.float16, out_dtype=dtypes.float16, x_rank=1, native=True,
    retain_input=retain_input)
  suffix="_retain" if retain_input else ""
  return KernelProgram("research.nv_rmsnorm_native_geometry", f"rmsnorm_native_1_4096_w{warps}{suffix}",
    KernelProgramProvenance.RESEARCH_ONLY, emit_decode_rmsnorm_kernel(spec),
    output_spec=OutputSpec((4096,), dtypes.float16))

def measure(out:pathlib.Path, replays:int, warmup:int, reps:int) -> None:
  dev = Device.DEFAULT
  if not str(dev).startswith("NV"): raise RuntimeError(f"DEV=NV required, got {dev}")
  rng = np.random.default_rng(20260826)
  x = Tensor(rng.normal(0, .2, 4096).astype(np.float16), device=dev).contiguous().realize()
  w = Tensor(rng.normal(1, .05, 4096).astype(np.float16), device=dev).contiguous().realize()
  rows, ref = [], None
  for warps,retain in [(w,False) for w in WARPS]+[(16,True)]:
    program, dst = _program(warps,retain), Tensor.empty(4096, dtype=dtypes.float16, device=dev)
    @TinyJit
    def run(a:Tensor, weight:Tensor): return execute_research_program(dst, a, weight, program=program)
    run(x,w).realize(); got=run(x,w).realize()
    Device[dev].synchronize(); arr = np.asarray(got.numpy())
    if ref is None: ref = arr.copy()
    exact = bool(np.array_equal(arr, ref))
    for _ in range(warmup): run(x,w).realize()
    Device[dev].synchronize()
    samples=[]
    for _ in range(reps):
      Device[dev].synchronize(); start=time.perf_counter_ns()
      for _ in range(replays): run(x,w).realize()
      Device[dev].synchronize(); samples.append((time.perf_counter_ns()-start)/1000/replays)
    rows.append({"warps":warps, "retain_input":retain, "kernel":program.program_id, "bit_exact_to_w1":exact,
      "samples_us":samples, "median_us":statistics.median(samples)})
  out.write_text(json.dumps({"schema":"tinygrad.nv_rmsnorm_native_geometry.v1", "replays":replays,
    "warmup":warmup, "reps":reps, "control_warps":16,
    "verdict_order":[r["warps"] for r in sorted(rows,key=lambda r:r["median_us"])], "rows":rows}, indent=2, sort_keys=True)+"\n")

def parse(meta:pathlib.Path, trace:pathlib.Path, out:pathlib.Path) -> None:
  data=json.loads(meta.read_text()); con=sqlite3.connect(trace)
  names={int(i):str(v) for i,v in con.execute("select id,value from StringIds")}
  vals={r["kernel"]:[] for r in data["rows"]}
  for start,end,short in con.execute("select start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL"):
    name=names.get(int(short),"")
    if name in vals: vals[name].append((end-start)/1000.0)
  for row in data["rows"]:
    samples=sorted(vals[row["kernel"]]); row["instances"]=len(samples)
    row["median_us"]=float(np.median(samples)) if samples else None
  control=next(r["median_us"] for r in data["rows"] if r["warps"]==16)
  for row in data["rows"]: row["delta_vs_w16_us"]=None if row["median_us"] is None else row["median_us"]-control
  data["control_warps"]=16; data["verdict_order"]=[r["warps"] for r in sorted(data["rows"], key=lambda r:r["median_us"])]
  out.write_text(json.dumps(data, indent=2, sort_keys=True)+"\n")

def cuda_gate(out:pathlib.Path, replays:int, reps:int) -> None:
  ren=CUDARenderer(Target("NV",arch="sm_120"),use_nvcc=False); p=UOp.placeholder
  def src(retain:bool) -> str:
    u=emit_decode_rmsnorm_kernel(DecodeRMSNormSpec(rows=1,dim=4096,eps=1e-6,warps_per_row=16,
      x_dtype=dtypes.float16,weight_dtype=dtypes.float16,out_dtype=dtypes.float16,x_rank=1,native=True,
      retain_input=retain))(p((4096,),dtypes.float16,0),p((4096,),dtypes.float16,1),p((4096,),dtypes.float16,2))
    s=next(v.arg for v in to_program(u,ren).src if v.op is Ops.SOURCE); return s[s.index('extern "C" __global__'):]
  c,r=src(False),src(True).replace("rmsnorm_native_1_4096","rmsnorm_native_1_4096_retain",1)
  harness=r'''#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v){union U{F f;T t;};U u;u.f=v;return u.t;}
__C__
__R__
static void ck(cudaError_t e,const char*w){if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e));exit(2);}}
static double tc(half*o,half*x,half*w,int n){cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);for(int i=0;i<n;i++)rmsnorm_native_1_4096<<<1,dim3(32,16,1)>>>(o,x,w);cudaEventRecord(b);ck(cudaEventSynchronize(b),"c");float ms;cudaEventElapsedTime(&ms,a,b);return ms*1000/n;}
static double tr(half*o,half*x,half*w,int n){cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);for(int i=0;i<n;i++)rmsnorm_native_1_4096_retain<<<1,dim3(32,16,1)>>>(o,x,w);cudaEventRecord(b);ck(cudaEventSynchronize(b),"r");float ms;cudaEventElapsedTime(&ms,a,b);return ms*1000/n;}
int main(int ac,char**av){int n=ac>1?atoi(av[1]):1000,reps=ac>2?atoi(av[2]):9;half *a,*b,*x,*w;cudaMalloc(&a,8192);cudaMalloc(&b,8192);cudaMalloc(&x,8192);cudaMalloc(&w,8192);cudaMemset(x,1,8192);cudaMemset(w,2,8192);rmsnorm_native_1_4096<<<1,dim3(32,16,1)>>>(a,x,w);rmsnorm_native_1_4096_retain<<<1,dim3(32,16,1)>>>(b,x,w);ck(cudaDeviceSynchronize(),"warm");half ha[4096],hb[4096];cudaMemcpy(ha,a,8192,cudaMemcpyDeviceToHost);cudaMemcpy(hb,b,8192,cudaMemcpyDeviceToHost);printf("bit_exact=%d\n",memcmp(ha,hb,8192)==0);for(int i=0;i<reps;i++)printf("rep=%d control=%.6f retain=%.6f\n",i,tc(a,x,w,n),tr(b,x,w,n));}'''.replace("__C__",c).replace("__R__",r)
  with tempfile.TemporaryDirectory(prefix="normretain_") as td:
    sp=pathlib.Path(td)/"g.cu"; bp=pathlib.Path(td)/"g"; sp.write_text(harness)
    cp=subprocess.run(["/usr/local/cuda-13.2/bin/nvcc","-arch=sm_120a","-O3","--ptxas-options=-v",str(sp),"-o",str(bp)],capture_output=True,text=True)
    if cp.returncode: raise RuntimeError(cp.stderr[-6000:])
    run=subprocess.run([str(bp),str(replays),str(reps)],capture_output=True,text=True,check=True)
  cv,rv=[],[]
  for line in run.stdout.splitlines():
    if m:=re.match(r"rep=\d+ control=([0-9.]+) retain=([0-9.]+)",line):cv.append(float(m.group(1)));rv.append(float(m.group(2)))
  result={"schema":"tinygrad.nv_rmsnorm_retain_input_cuda_gate.v1","bit_exact":bool(re.search(r"bit_exact=1",run.stdout)),
    "control_us":cv,"retain_us":rv,"control_median_us":statistics.median(cv),"retain_median_us":statistics.median(rv),
    "delta_us":statistics.median(rv)-statistics.median(cv),"ptxas":cp.stderr.splitlines()}
  out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,indent=2,sort_keys=True))

if __name__ == "__main__":
  ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="mode",required=True)
  m=sub.add_parser("measure"); m.add_argument("--out",type=pathlib.Path,required=True);m.add_argument("--replays",type=int,default=1000);m.add_argument("--warmup",type=int,default=50);m.add_argument("--reps",type=int,default=7)
  p=sub.add_parser("parse");p.add_argument("--meta",type=pathlib.Path,required=True);p.add_argument("--trace",type=pathlib.Path,required=True);p.add_argument("--out",type=pathlib.Path,required=True)
  c=sub.add_parser("cuda");c.add_argument("--out",type=pathlib.Path,required=True);c.add_argument("--replays",type=int,default=1000);c.add_argument("--reps",type=int,default=9)
  a=ap.parse_args(); measure(a.out,a.replays,a.warmup,a.reps) if a.mode=="measure" else parse(a.meta,a.trace,a.out) if a.mode=="parse" else cuda_gate(a.out,a.replays,a.reps)
