#!/usr/bin/env python3
"""First-class bounded-persistent Q4_K O emitter and Flash handoff gate.

Research only.  Unlike the earlier generated-source wrapping attempt, the O
consumer is emitted directly from UOps.  A bounded number of 128-thread CTAs
wait on one producer release epoch, then execute the complete installed
Q4_K vector-load arithmetic in a deterministic strided row loop.  The control
uses the installed 4096-CTA O kernel after the same full-grid Flash producer.
"""
from __future__ import annotations

import argparse, json, re, shutil, statistics, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Target
from tinygrad.llm.decode_kernels import (LanePartition, Q4KGEMVEpilogue, Q4KGateUpLaneMap,
  Q4K_WORDS_PER_BLOCK, _lane_partition_reduce_sum, _q4k_block_dot_packed_load_vec,
  q4k_g3_lanemap_gemv_kernel)
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp
from extra.llm_research.decode.nv_flash_last_cta_combine_gate import source as flash_source
from extra.llm_research.decode.nv_segmented_flash_o_complete_span import _preambles

ROWS = K = 4096
O_WORDS = ROWS * (K // 256) * Q4K_WORDS_PER_BLOCK
CACHE_WORDS = 1_048_576
ROTATIONS = 16
NVCC = "/usr/local/cuda-13.2/bin/nvcc"


def _render(ast: UOp) -> tuple[str, str]:
  prg = to_program(ast, CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=False))
  return prg.arg.name, next(x.arg for x in prg.src if x.op is Ops.SOURCE)


def persistent_o_emitter(workers: int):
  """Four rows/CTA per wave, exact installed lane association and reduction."""
  if workers <= 0 or (ROWS // 4) % workers: raise ValueError("workers must divide 1024")
  waves = ROWS // (workers * 4)
  lm = Q4KGateUpLaneMap(k=K, n=ROWS)
  lm.validate()

  def kernel(out: UOp, words: UOp, x: UOp, residual: UOp, ready: UOp) -> UOp:
    worker, lid = UOp.special(workers, "gidx0"), UOp.special(128, "lidx0")
    warp, lane = lid // 32, lid % 32
    # One lane polls per CTA; all four warps are released together.  Returning
    # zero makes the readiness operation an address dependency without moving
    # the logical row.
    wait = UOp(Ops.CUSTOMI, dtypes.uint32, (ready.index(UOp.const(dtypes.int, 0), ptr=True),),
      arg='([&]() {{ if (threadIdx.x == 0) while (*((volatile unsigned int*){0}) == 0u) __nanosleep(64); '
          '__syncthreads(); return 0u; }})()')
    wave = UOp.range(waves, 18, axis_type=AxisType.LOOP)
    row = (wave * workers + worker) * 4 + warp + wait.cast(dtypes.weakint)
    part = LanePartition(lane, lane_extent=lm.lane_extent, words_per_group=lm.words_per_group)
    lblk = UOp.range(lm.blocks_per_group, 0, axis_type=AxisType.REDUCE)
    blk = part.block_group * lm.blocks_per_group + lblk
    base = (row * lm.k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = _q4k_block_dot_packed_load_vec(words, x, base, blk, part.word_col)
    acc = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
    # The register is worker-local, so reset it at the start of every row
    # wave.  Depending on both wave and wait keeps the initialization inside
    # the outer loop and after producer publication.
    acc = acc.after(acc.after(wait, wave)[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(lblk)[0] + contrib).end(lblk))
    total = _lane_partition_reduce_sum(acc[0], part)
    stored = out[row].store(total + residual[row].cast(dtypes.float32))
    return stored.end(wave).sink(arg=KernelInfo(name=f"q4k_o_bounded_persistent_w{workers}", opts_to_apply=()))
  return kernel


def build(workers: int) -> tuple[str, dict[str, str]]:
  p = UOp.placeholder
  oname, osrc = _render(q4k_g3_lanemap_gemv_kernel(ROWS, K, epilogue=Q4KGEMVEpilogue("residual_add"),
    load_style="vector")(p((ROWS,), dtypes.float32, 0), p((O_WORDS,), dtypes.uint32, 1),
      p((K,), dtypes.float16, 2), p((ROWS,), dtypes.float32, 3)))
  pname, psrc = _render(persistent_o_emitter(workers)(p((ROWS,), dtypes.float32, 0),
    p((O_WORDS,), dtypes.uint32, 1), p((K,), dtypes.float16, 2), p((ROWS,), dtypes.float32, 3),
    p((1,), dtypes.uint32, 4)))

  fsrc, _, _, fname = flash_source()
  start = fsrc.index('extern "C"', fsrc.index(fname) - 80)
  fbody = fsrc[start:]
  # source() places the last-CTA kernel last. Extend only its explicit ABI and
  # publish after the thirty-second per-head combine has completed.
  fbody = fbody.replace("unsigned int* counters) {",
    "unsigned int* counters, unsigned int* ready, unsigned int* heads_done) {", 1)
  old = "if (tg_lane == 0) counters[gidx1]=0u;"
  new = """if (tg_lane == 0) {
      counters[gidx1]=0u;
      __threadfence();
      if (atomicAdd(heads_done, 1u) == 31u) {
        heads_done[0]=0u;
        __threadfence();
        ready[0]=1u;
      }
    }"""
  if old not in fbody: raise RuntimeError("last-CTA publication anchor moved")
  fbody = fbody.replace(old, new, 1)
  src = _preambles(fsrc, osrc, psrc) + fbody + osrc[osrc.index('extern "C"'):] + psrc[psrc.index('extern "C"'):]
  return src, {"flash": fname, "control_o": oname, "persistent_o": pname}


HARNESS = r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
__SRC__
#define ROWS 4096
#define O_WORDS 2359296
#define CACHE_WORDS 1048576
#define ROTATIONS 16
#define WORKERS __WORKERS__
static void ck(cudaError_t e,const char*w){if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e));exit(2);}}
struct Ctx{cudaStream_t producer,consumer;cudaEvent_t start,done;};
static void init(Ctx&c){int least,greatest;ck(cudaDeviceGetStreamPriorityRange(&least,&greatest),"priority");
  ck(cudaStreamCreateWithPriority(&c.producer,cudaStreamNonBlocking,greatest),"producer-stream");
  ck(cudaStreamCreateWithPriority(&c.consumer,cudaStreamNonBlocking,least),"consumer-stream");
  ck(cudaEventCreate(&c.start),"start-event");ck(cudaEventCreate(&c.done),"done-event");}
static float elapsed(cudaEvent_t a,cudaEvent_t b){float ms;ck(cudaEventElapsedTime(&ms,a,b),"elapsed");return ms*1000.0f;}
static void prep(unsigned*ready,unsigned*heads,unsigned*ctr,cudaStream_t s){ck(cudaMemsetAsync(ready,0,4,s),"ready0");
  ck(cudaMemsetAsync(heads,0,4,s),"heads0");ck(cudaMemsetAsync(ctr,0,128,s),"ctr0");}
static float control(Ctx&c,float*out,half*att,float*partial,float*q,unsigned*cache,unsigned*w,float*res,unsigned*ctr,unsigned*ready,unsigned*heads){
  prep(ready,heads,ctr,c.producer);ck(cudaEventRecord(c.start,c.producer),"c-start");
  __FLASH__<<<dim3(6,32),dim3(32,4),0,c.producer>>>(partial,q,cache,att,ctr,ready,heads);
  __CONTROL_O__<<<ROWS,32,0,c.producer>>>(out,w,att,res);ck(cudaEventRecord(c.done,c.producer),"c-done");
  ck(cudaEventSynchronize(c.done),"c-sync");return elapsed(c.start,c.done);}
static float candidate(Ctx&c,float*out,half*att,float*partial,float*q,unsigned*cache,unsigned*w,float*res,unsigned*ctr,unsigned*ready,unsigned*heads){
  prep(ready,heads,ctr,c.producer);ck(cudaEventRecord(c.start,c.producer),"p-start");
  ck(cudaStreamWaitEvent(c.consumer,c.start),"consumer-start");
  __PERSISTENT_O__<<<WORKERS,128,0,c.consumer>>>(out,w,att,res,ready);
  __FLASH__<<<dim3(6,32),dim3(32,4),0,c.producer>>>(partial,q,cache,att,ctr,ready,heads);
  ck(cudaEventRecord(c.done,c.consumer),"p-done");ck(cudaEventSynchronize(c.done),"p-sync");return elapsed(c.start,c.done);}
static float candidate_after(Ctx&c,float*out,half*att,float*partial,float*q,unsigned*cache,unsigned*w,float*res,unsigned*ctr,unsigned*ready,unsigned*heads){
  prep(ready,heads,ctr,c.producer);ck(cudaEventRecord(c.start,c.producer),"a-start");
  __FLASH__<<<dim3(6,32),dim3(32,4),0,c.producer>>>(partial,q,cache,att,ctr,ready,heads);
  __PERSISTENT_O__<<<WORKERS,128,0,c.producer>>>(out,w,att,res,ready);
  ck(cudaEventRecord(c.done,c.producer),"a-done");ck(cudaEventSynchronize(c.done),"a-sync");return elapsed(c.start,c.done);}
static float installed_ready(Ctx&c,float*out,half*x,unsigned*w,float*res,unsigned*ready){ck(cudaMemsetAsync(ready,1,4,c.producer),"r1");
  ck(cudaEventRecord(c.start,c.producer),"i-start");__CONTROL_O__<<<ROWS,32,0,c.producer>>>(out,w,x,res);
  ck(cudaEventRecord(c.done,c.producer),"i-done");ck(cudaEventSynchronize(c.done),"i-sync");return elapsed(c.start,c.done);}
static float persistent_ready(Ctx&c,float*out,half*x,unsigned*w,float*res,unsigned*ready){ck(cudaMemsetAsync(ready,1,4,c.producer),"r1");
  ck(cudaEventRecord(c.start,c.producer),"pr-start");__PERSISTENT_O__<<<WORKERS,128,0,c.producer>>>(out,w,x,res,ready);
  ck(cudaEventRecord(c.done,c.producer),"pr-done");ck(cudaEventSynchronize(c.done),"pr-sync");return elapsed(c.start,c.done);}
static void legal_words(unsigned*w,size_t n,unsigned seed){for(size_t i=0;i<n;i++)w[i]=(unsigned)((i*2654435761u)^(seed*2246822519u)^0x9e3779b9u);
  for(size_t b=0;b<n;b+=36){w[b]=0x30003000u+(seed&3u);w[b+1]=0x10101010u;}}
int main(int ac,char**av){int hp=atoi(av[1]),cp=atoi(av[2]),reps=atoi(av[3]);Ctx c;init(c);
  float *a,*b,*pa,*pb,*q,*res;half *aa,*bb;unsigned *wa,*wb,*ka,*kb,*ca,*cb,*ra,*rb,*ha,*hb;
  ck(cudaMalloc(&a,ROWS*4),"a");ck(cudaMalloc(&b,ROWS*4),"b");ck(cudaMalloc(&pa,24960*4),"pa");ck(cudaMalloc(&pb,24960*4),"pb");
  ck(cudaMalloc(&aa,8192),"aa");ck(cudaMalloc(&bb,8192),"bb");ck(cudaMalloc(&q,16384),"q");ck(cudaMalloc(&res,ROWS*4),"res");
  ck(cudaMalloc(&wa,(size_t)ROTATIONS*O_WORDS*4),"wa");ck(cudaMalloc(&wb,(size_t)ROTATIONS*O_WORDS*4),"wb");
  ck(cudaMalloc(&ka,(size_t)ROTATIONS*CACHE_WORDS*4),"ka");ck(cudaMalloc(&kb,(size_t)ROTATIONS*CACHE_WORDS*4),"kb");
  ck(cudaMalloc(&ca,128),"ca");ck(cudaMalloc(&cb,128),"cb");ck(cudaMalloc(&ra,4),"ra");ck(cudaMalloc(&rb,4),"rb");
  ck(cudaMalloc(&ha,4),"ha");ck(cudaMalloc(&hb,4),"hb");
  std::vector<unsigned> hwa((size_t)ROTATIONS*O_WORDS),hwb((size_t)ROTATIONS*O_WORDS);for(int r=0;r<ROTATIONS;r++){
    legal_words(hwa.data()+(size_t)r*O_WORDS,O_WORDS,17+r);legal_words(hwb.data()+(size_t)r*O_WORDS,O_WORDS,17+r);}
  std::vector<float> hq(4096),hr(ROWS);for(int i=0;i<4096;i++)hq[i]=float((i*17%127)-63)/256.0f;
  for(int i=0;i<ROWS;i++)hr[i]=float((i%31)-15)/128.0f;
  ck(cudaMemcpy(wa,hwa.data(),hwa.size()*4,cudaMemcpyHostToDevice),"wa-copy");ck(cudaMemcpy(wb,hwb.data(),hwb.size()*4,cudaMemcpyHostToDevice),"wb-copy");
  ck(cudaMemcpy(q,hq.data(),16384,cudaMemcpyHostToDevice),"q-copy");ck(cudaMemcpy(res,hr.data(),ROWS*4,cudaMemcpyHostToDevice),"res-copy");
  ck(cudaMemset(ka,0x30,(size_t)ROTATIONS*CACHE_WORDS*4),"ka-init");ck(cudaMemset(kb,0x30,(size_t)ROTATIONS*CACHE_WORDS*4),"kb-init");
  bool exact=true,finite=true;std::vector<float> xa(ROWS),xb(ROWS);for(int rot: {0,5,11}){
    control(c,a,aa,pa,q,ka+(size_t)rot*CACHE_WORDS,wa+(size_t)rot*O_WORDS,res,ca,ra,ha);
    candidate(c,b,bb,pb,q,kb+(size_t)rot*CACHE_WORDS,wb+(size_t)rot*O_WORDS,res,cb,rb,hb);
    ck(cudaMemcpy(xa.data(),a,ROWS*4,cudaMemcpyDeviceToHost),"a-copy");ck(cudaMemcpy(xb.data(),b,ROWS*4,cudaMemcpyDeviceToHost),"b-copy");
    exact&=memcmp(xa.data(),xb.data(),ROWS*4)==0;for(int i=0;i<ROWS;i++)finite&=std::isfinite(xa[i])&&std::isfinite(xb[i]);}
  printf("validate exact=%d finite=%d\n",(int)exact,(int)finite);
  for(int r=0;r<reps;r++){double ih=0,ph=0,ic=0,pc0=0,ch=0,hh=0,ah=0,cc=0,hc=0,ac=0;
    for(int i=0;i<hp;i++){if((r+i)&1){ph+=persistent_ready(c,b,bb,wb,res,rb);ih+=installed_ready(c,a,aa,wa,res,ra);hh+=candidate(c,b,bb,pb,q,kb,wb,res,cb,rb,hb);ah+=candidate_after(c,b,bb,pb,q,kb,wb,res,cb,rb,hb);ch+=control(c,a,aa,pa,q,ka,wa,res,ca,ra,ha);}
      else{ih+=installed_ready(c,a,aa,wa,res,ra);ph+=persistent_ready(c,b,bb,wb,res,rb);ch+=control(c,a,aa,pa,q,ka,wa,res,ca,ra,ha);ah+=candidate_after(c,b,bb,pb,q,kb,wb,res,cb,rb,hb);hh+=candidate(c,b,bb,pb,q,kb,wb,res,cb,rb,hb);}}
    for(int i=0;i<cp;i++){int z=(r*cp+i)%ROTATIONS;if((r+i)&1){pc0+=persistent_ready(c,b,bb,wb+(size_t)z*O_WORDS,res,rb);ic+=installed_ready(c,a,aa,wa+(size_t)z*O_WORDS,res,ra);hc+=candidate(c,b,bb,pb,q,kb+(size_t)z*CACHE_WORDS,wb+(size_t)z*O_WORDS,res,cb,rb,hb);ac+=candidate_after(c,b,bb,pb,q,kb+(size_t)z*CACHE_WORDS,wb+(size_t)z*O_WORDS,res,cb,rb,hb);cc+=control(c,a,aa,pa,q,ka+(size_t)z*CACHE_WORDS,wa+(size_t)z*O_WORDS,res,ca,ra,ha);}
      else{ic+=installed_ready(c,a,aa,wa+(size_t)z*O_WORDS,res,ra);pc0+=persistent_ready(c,b,bb,wb+(size_t)z*O_WORDS,res,rb);cc+=control(c,a,aa,pa,q,ka+(size_t)z*CACHE_WORDS,wa+(size_t)z*O_WORDS,res,ca,ra,ha);ac+=candidate_after(c,b,bb,pb,q,kb+(size_t)z*CACHE_WORDS,wb+(size_t)z*O_WORDS,res,cb,rb,hb);hc+=candidate(c,b,bb,pb,q,kb+(size_t)z*CACHE_WORDS,wb+(size_t)z*O_WORDS,res,cb,rb,hb);}}
    printf("rep=%d installed_hot=%.6f persistent_hot=%.6f installed_cold=%.6f persistent_cold=%.6f control_span_hot=%.6f candidate_span_hot=%.6f after_span_hot=%.6f control_span_cold=%.6f candidate_span_cold=%.6f after_span_cold=%.6f\n",
      r,ih/hp,ph/hp,ic/cp,pc0/cp,ch/hp,hh/hp,ah/hp,cc/cp,hc/cp,ac/cp);}
  return exact&&finite?0:5;}
'''

SAMPLE = re.compile(r"rep=(\d+) installed_hot=([0-9.]+) persistent_hot=([0-9.]+) installed_cold=([0-9.]+) persistent_cold=([0-9.]+) control_span_hot=([0-9.]+) candidate_span_hot=([0-9.]+) after_span_hot=([0-9.]+) control_span_cold=([0-9.]+) candidate_span_cold=([0-9.]+) after_span_cold=([0-9.]+)")


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--workers", type=int, required=True)
  ap.add_argument("--hot", type=int, default=32)
  ap.add_argument("--cold", type=int, default=16)
  ap.add_argument("--reps", type=int, default=9)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--artifact-dir", type=Path)
  args = ap.parse_args()
  src, names = build(args.workers)
  cu = (HARNESS.replace("__SRC__", src).replace("__WORKERS__", str(args.workers))
    .replace("__FLASH__", names["flash"]).replace("__CONTROL_O__", names["control_o"])
    .replace("__PERSISTENT_O__", names["persistent_o"]))
  with tempfile.TemporaryDirectory(prefix="nv_flash_o_bounded_") as td:
    cup, binary = Path(td)/"gate.cu", Path(td)/"gate"
    cup.write_text(cu)
    buildp = subprocess.run([NVCC, "-O3", "-std=c++20", "-arch=sm_120a", "--ptxas-options=-v", str(cup), "-o", str(binary)],
      capture_output=True, text=True)
    if buildp.returncode: raise RuntimeError(buildp.stdout + "\n" + buildp.stderr[-16000:])
    run = subprocess.run([str(binary), str(args.hot), str(args.cold), str(args.reps)], capture_output=True, text=True, timeout=900)
    if args.artifact_dir:
      args.artifact_dir.mkdir(parents=True, exist_ok=True)
      shutil.copy2(cup, args.artifact_dir/"gate.cu"); shutil.copy2(binary, args.artifact_dir/"gate")
      (args.artifact_dir/"ptxas.txt").write_text(buildp.stderr)
    if run.returncode: raise RuntimeError(run.stdout + "\n" + run.stderr)
  rows = []
  for line in run.stdout.splitlines():
    if m := SAMPLE.fullmatch(line):
      vals = [float(x) for x in m.groups()[1:]]
      rows.append(dict(zip(("installed_hot_us", "persistent_hot_us", "installed_cold_us", "persistent_cold_us",
        "control_span_hot_us", "candidate_span_hot_us", "after_span_hot_us", "control_span_cold_us", "candidate_span_cold_us",
        "after_span_cold_us"), vals), rep=int(m[1])))
  if not rows or "validate exact=1 finite=1" not in run.stdout: raise RuntimeError("unparsed or invalid output\n" + run.stdout)
  keys = tuple(k for k in rows[0] if k != "rep")
  med = {k: statistics.median(row[k] for row in rows) for k in keys}
  med["o_cold_recovery_us"] = med["installed_cold_us"] - med["persistent_cold_us"]
  med["span_cold_recovery_us"] = med["control_span_cold_us"] - med["candidate_span_cold_us"]
  med["after_span_cold_recovery_us"] = med["control_span_cold_us"] - med["after_span_cold_us"]
  result = {"schema":"tinygrad.nv_flash_o_bounded_persistent_emitter.v1", "commit":subprocess.check_output(
    ["git","rev-parse","HEAD"], cwd=ROOT, text=True).strip(), "workers":args.workers, "exact":True, "names":names,
    "median":med, "samples":rows, "stdout":run.stdout, "ptxas":buildp.stderr}
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__": raise SystemExit(main())
