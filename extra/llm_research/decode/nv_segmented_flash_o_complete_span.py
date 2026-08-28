#!/usr/bin/env python3
"""Exact two-wave Flash -> Q4_K O complete-span native gate (research only).

The candidate splits the installed S6 score/combine work by query-head half.
The first O half materializes the exact lanes 0..15 accumulators; the second
half forms the installed offset-16 pair and finishes the unchanged 8/4/2/1
shuffle ladder.  No activation compression or production routing is involved.
"""
from __future__ import annotations

import argparse, json, re, statistics, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.codegen.late.warp_reduce import _warp_reduce_sum_staged
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Target
from tinygrad.llm.decode_kernels import (LanePartition, Q4KGEMVEpilogue, Q4KGateUpLaneMap,
  _q4k_block_dot_packed_load_vec, q4k_g3_lanemap_gemv_kernel)
from tinygrad.llm.flash_decode_attention import flash_fused_gmax_combine_kernel, flash_vec_llama_score_pv_kernel
from tinygrad.llm.qk_layout import Q4K_WORDS_PER_BLOCK
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp

H, HKV, HD, S, MAXC, TC = 32, 8, 128, 6, 1024, 513
W, ROWS, K = HD + 2, 4096, 4096
O_WORDS = ROWS * (K // 256) * Q4K_WORDS_PER_BLOCK
ROTATIONS = 16
NVCC = "/usr/local/cuda-13.2/bin/nvcc"


def _render(ast: UOp) -> tuple[str, str]:
  prg = to_program(ast, CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=False))
  return prg.arg.name, next(x.arg for x in prg.src if x.op is Ops.SOURCE)


def _head_slice(src: str, name: str, *, axis: str, offset: int, count: int) -> tuple[str, str]:
  sliced = f"{name}_h{count}o{offset}"
  body = src[src.index('extern "C"'):].replace(name, sliced, 1)
  pat = rf"int {axis} = blockIdx\.(y if axis == 'gidx1' else x); /\* 32 \*/"
  # Keep the specialization literal so ptxas can fold the head offset.
  block_axis = "y" if axis == "gidx1" else "x"
  needle = rf"int {axis} = blockIdx\.{block_axis}; /\* 32 \*/"
  repl = f"int {axis} = blockIdx.{block_axis} + {offset}; /* {count} of 32 */"
  body, n = re.subn(needle, repl, body, count=1)
  if n != 1: raise RuntimeError(f"could not specialize {name} on {axis}: pattern {pat}")
  return sliced, body


def _o_half_emitter(*, first_half: bool):
  """Two rows/warp, sixteen physical lanes/row, exact installed lane grammar."""
  lm = Q4KGateUpLaneMap(k=K, n=ROWS)
  lm.validate()
  # The high-priority producer owns heads 16..31. Store that half first, then
  # let the low-half finisher spell the installed low + high offset-16 add.
  lane_base = 16 if first_half else 0
  name = "q4k_o_segment16_high_first" if first_half else "q4k_o_segment16_low_finish_epi_resadd"

  def kernel(out: UOp, words: UOp, x: UOp, *extra: UOp) -> UOp:
    if len(extra) != (0 if first_half else 2): raise ValueError("segmented O argument mismatch")
    row_pair, lane = UOp.special(ROWS // 2, "gidx0"), UOp.special(32, "lidx0")
    lane16, row = lane.bitwise_and(15), row_pair * 2 + lane.rshift(4)
    logical_lane = lane16 + lane_base
    part = LanePartition(logical_lane, lane_extent=lm.lane_extent, words_per_group=lm.words_per_group)
    lblk = UOp.range(lm.blocks_per_group, 0, axis_type=AxisType.REDUCE)
    blk = part.block_group * lm.blocks_per_group + lblk
    base = (row * lm.k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = _q4k_block_dot_packed_load_vec(words, x, base, blk, part.word_col)
    acc = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(lblk)[0] + contrib).end(lblk))
    partial = acc[0]
    if first_half:
      return out[row * 16 + lane16].store(partial).sink(arg=KernelInfo(name=name, opts_to_apply=()))
    scratch, residual = extra
    # Installed offset-16 semantics for each low logical lane are low + high.
    paired = partial + scratch[row * 16 + lane16]
    total = _warp_reduce_sum_staged(paired, lane, 16, slot_base=90)
    result = total + residual[row].cast(dtypes.float32)
    return out[row].store(result, lane16.eq(0)).sink(arg=KernelInfo(name=name, opts_to_apply=()))
  return kernel


def _preambles(*sources: str) -> str:
  # Generated declarations are one-per-line. Unioning them retains every vector
  # type needed by score/combine/O without redefining tg_bitcast or structs.
  lines: list[str] = []
  for src in sources:
    for line in src[:src.index('extern "C"')].splitlines():
      if line not in lines: lines.append(line)
  return "\n".join(lines) + "\n"


def _build_source() -> tuple[str, dict[str, str]]:
  p = UOp.placeholder
  score_name, score_src = _render(flash_vec_llama_score_pv_kernel(
    HD, H, HKV, MAXC, S, UOp.const(dtypes.int, TC), wide_kv=True, wide_q=False,
    token_bound=768, v_pipeline_tail=1)(
      p((H*S*W,), dtypes.float32, 0), p((H*HD,), dtypes.float32, 1),
      p((2*HKV*MAXC*HD//2,), dtypes.uint32, 2)))
  combine_name, combine_src = _render(flash_fused_gmax_combine_kernel(
    HD, H, S, output_fp16=True, lane_width=128, register_weights=False)(
      p((H*HD,), dtypes.float16, 0), p((H*S*W,), dtypes.float32, 1)))
  o_name, o_src = _render(q4k_g3_lanemap_gemv_kernel(
    ROWS, K, epilogue=Q4KGEMVEpilogue("residual_add"), load_style="vector")(
      p((ROWS,), dtypes.float32, 0), p((O_WORDS,), dtypes.uint32, 1),
      p((K,), dtypes.float16, 2), p((ROWS,), dtypes.float32, 3)))
  first_name, first_src = _render(_o_half_emitter(first_half=True)(
    p((ROWS*16,), dtypes.float32, 0), p((O_WORDS,), dtypes.uint32, 1), p((K,), dtypes.float16, 2)))
  finish_name, finish_src = _render(_o_half_emitter(first_half=False)(
    p((ROWS,), dtypes.float32, 0), p((O_WORDS,), dtypes.uint32, 1), p((K,), dtypes.float16, 2),
    p((ROWS*16,), dtypes.float32, 3), p((ROWS,), dtypes.float32, 4)))
  score0, score0_src = _head_slice(score_src, score_name, axis="gidx1", offset=0, count=16)
  score1, score1_src = _head_slice(score_src, score_name, axis="gidx1", offset=16, count=16)
  combine0, combine0_src = _head_slice(combine_src, combine_name, axis="gidx0", offset=0, count=16)
  combine1, combine1_src = _head_slice(combine_src, combine_name, axis="gidx0", offset=16, count=16)
  sources = (score_src, combine_src, o_src, first_src, finish_src)
  functions = [score_src[score_src.index('extern "C"'):], combine_src[combine_src.index('extern "C"'):],
    o_src[o_src.index('extern "C"'):], first_src[first_src.index('extern "C"'):],
    finish_src[finish_src.index('extern "C"'):], score0_src, score1_src, combine0_src, combine1_src]
  names = {"score":score_name, "combine":combine_name, "o":o_name, "first":first_name, "finish":finish_name,
           "score0":score0, "score1":score1, "combine0":combine0, "combine1":combine1}
  return _preambles(*sources) + "\n".join(functions), names


def _harness(kernels: str, n: dict[str, str]) -> str:
  return kernels + f'''
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define HQ {H}
#define HD {HD}
#define SPLITS {S}
#define STRIDE {W}
#define ROWS {ROWS}
#define O_WORDS {O_WORDS}
#define ROTATIONS {ROTATIONS}
#define CACHE_WORDS {2*HKV*MAXC*HD//2}
static void ck(cudaError_t e,const char*w){{if(e!=cudaSuccess){{fprintf(stderr,"%s: %s\\n",w,cudaGetErrorString(e));exit(2);}}}}
struct Ctx {{ cudaStream_t hi,lo,o; cudaEvent_t start,done,r0,r1,f1s,f1e,o0s,o0e; }};
static void init_ctx(Ctx&c){{int least,greatest;ck(cudaDeviceGetStreamPriorityRange(&least,&greatest),"priority");
  ck(cudaStreamCreateWithPriority(&c.hi,cudaStreamNonBlocking,greatest),"hi");
  ck(cudaStreamCreateWithPriority(&c.lo,cudaStreamNonBlocking,least),"lo");
  ck(cudaStreamCreateWithFlags(&c.o,cudaStreamNonBlocking),"o");
  for(cudaEvent_t*e:{{&c.start,&c.done,&c.r0,&c.r1,&c.f1s,&c.f1e,&c.o0s,&c.o0e}})ck(cudaEventCreate(e),"event");}}
static float elapsed(cudaEvent_t a,cudaEvent_t b){{float ms;ck(cudaEventElapsedTime(&ms,a,b),"elapsed");return ms*1000.0f;}}
static double control(Ctx&c,float*out,half*attn,float*partial,float*q,unsigned int*cache,unsigned int*w,float*res){{
  ck(cudaEventRecord(c.start,c.o),"cstart");
  {n['score']}<<<dim3(SPLITS,HQ,1),dim3(32,4,1),0,c.o>>>(partial,q,cache);
  {n['combine']}<<<HQ,128,0,c.o>>>(attn,partial);
  {n['o']}<<<ROWS,32,0,c.o>>>(out,w,attn,res);
  ck(cudaEventRecord(c.done,c.o),"cdone");ck(cudaEventSynchronize(c.done),"csync");return elapsed(c.start,c.done);
}}
static double candidate(Ctx&c,float*out,half*attn,float*partial,float*q,unsigned int*cache,unsigned int*w,float*scratch,float*res,bool trace){{
  ck(cudaEventRecord(c.start,c.o),"start");ck(cudaStreamWaitEvent(c.hi,c.start),"hi-start");ck(cudaStreamWaitEvent(c.lo,c.start),"lo-start");
  // Give the high logical half the high-priority queue, then preserve low+high
  // association when the low-half kernel finishes the O projection.
  {n['score1']}<<<dim3(SPLITS,16,1),dim3(32,4,1),0,c.hi>>>(partial,q,cache);
  {n['combine1']}<<<16,128,0,c.hi>>>(attn,partial);ck(cudaEventRecord(c.r1,c.hi),"r1");
  ck(cudaEventRecord(c.f1s,c.lo),"f1s");
  {n['score0']}<<<dim3(SPLITS,16,1),dim3(32,4,1),0,c.lo>>>(partial,q,cache);
  {n['combine0']}<<<16,128,0,c.lo>>>(attn,partial);ck(cudaEventRecord(c.f1e,c.lo),"f1e");ck(cudaEventRecord(c.r0,c.lo),"r0");
  ck(cudaStreamWaitEvent(c.o,c.r1),"wait-r1");ck(cudaEventRecord(c.o0s,c.o),"o0s");
  {n['first']}<<<ROWS/2,32,0,c.o>>>(scratch,w,attn);ck(cudaEventRecord(c.o0e,c.o),"o0e");
  ck(cudaStreamWaitEvent(c.o,c.r0),"wait-r0");
  {n['finish']}<<<ROWS/2,32,0,c.o>>>(out,w,attn,scratch,res);
  ck(cudaEventRecord(c.done,c.o),"done");ck(cudaEventSynchronize(c.done),"sync");
  if(trace){{double f1a=elapsed(c.start,c.f1s),f1b=elapsed(c.start,c.f1e),oa=elapsed(c.start,c.o0s),ob=elapsed(c.start,c.o0e);
    double ov=std::max(0.0,std::min(f1b,ob)-std::max(f1a,oa));printf("timeline flash1_start=%.6f flash1_end=%.6f o0_start=%.6f o0_end=%.6f overlap=%.6f\\n",f1a,f1b,oa,ob,ov);}}
  return elapsed(c.start,c.done);
}}
int main(int ac,char**av){{int hp=atoi(av[1]),cp=atoi(av[2]),reps=atoi(av[3]);Ctx c;init_ctx(c);
  float *co,*so,*cp0,*sp,*q,*res,*scratch;half *ca,*sa;unsigned int *ws,*caches;unsigned char*evict;
  ck(cudaMalloc(&co,ROWS*4),"co");ck(cudaMalloc(&so,ROWS*4),"so");ck(cudaMalloc(&cp0,HQ*SPLITS*STRIDE*4),"cp");ck(cudaMalloc(&sp,HQ*SPLITS*STRIDE*4),"sp");
  ck(cudaMalloc(&ca,HQ*HD*2),"ca");ck(cudaMalloc(&sa,HQ*HD*2),"sa");ck(cudaMalloc(&q,HQ*HD*4),"q");ck(cudaMalloc(&res,ROWS*4),"res");
  ck(cudaMalloc(&scratch,ROWS*16*4),"scratch");ck(cudaMalloc(&ws,(size_t)ROTATIONS*O_WORDS*4),"ws");
  ck(cudaMalloc(&caches,(size_t)ROTATIONS*CACHE_WORDS*4),"cache");ck(cudaMalloc(&evict,128ull<<20),"evict");
  std::vector<float> hq(HQ*HD),hr(ROWS);for(size_t i=0;i<hq.size();i++)hq[i]=float((int(i*17+3)%127)-63)/256.0f;for(int i=0;i<ROWS;i++)hr[i]=float((i%31)-15)/128.0f;
  ck(cudaMemcpy(q,hq.data(),hq.size()*4,cudaMemcpyHostToDevice),"qcopy");ck(cudaMemcpy(res,hr.data(),hr.size()*4,cudaMemcpyHostToDevice),"rcopy");
  ck(cudaMemset(ws,7,(size_t)ROTATIONS*O_WORDS*4),"winit");ck(cudaMemset(caches,0x30,(size_t)ROTATIONS*CACHE_WORDS*4),"cinit");
  control(c,co,ca,cp0,q,caches,ws,res);candidate(c,so,sa,sp,q,caches,ws,scratch,res,false);
  std::vector<unsigned char>a(HQ*SPLITS*STRIDE*4),b(a.size());cudaMemcpy(a.data(),cp0,a.size(),cudaMemcpyDeviceToHost);cudaMemcpy(b.data(),sp,b.size(),cudaMemcpyDeviceToHost);printf("score_bitwise=%d\\n",memcmp(a.data(),b.data(),a.size())==0);
  a.resize(HQ*HD*2);b.resize(a.size());cudaMemcpy(a.data(),ca,a.size(),cudaMemcpyDeviceToHost);cudaMemcpy(b.data(),sa,b.size(),cudaMemcpyDeviceToHost);printf("combine_bitwise=%d\\n",memcmp(a.data(),b.data(),a.size())==0);
  a.resize(ROWS*4);b.resize(a.size());cudaMemcpy(a.data(),co,a.size(),cudaMemcpyDeviceToHost);cudaMemcpy(b.data(),so,b.size(),cudaMemcpyDeviceToHost);printf("final_bitwise=%d\\n",memcmp(a.data(),b.data(),a.size())==0);
  for(int i=0;i<20;i++){{control(c,co,ca,cp0,q,caches,ws,res);candidate(c,so,sa,sp,q,caches,ws,scratch,res,false);}}
  for(int i=0;i<reps;i++)candidate(c,so,sa,sp,q,caches,ws,scratch,res,true);
  for(int r=0;r<reps;r++)for(int oi=0;oi<2;oi++){{int arm=(r&1)?1-oi:oi;double hot=0,cold=0;
    for(int i=0;i<hp;i++)hot+=(arm?candidate(c,so,sa,sp,q,caches,ws,scratch,res,false):control(c,co,ca,cp0,q,caches,ws,res));
    for(int i=0;i<cp;i++){{ck(cudaMemset(evict,(r*cp+i)&255,128ull<<20),"evict");ck(cudaDeviceSynchronize(),"evictsync");int rot=(r*cp+i)%ROTATIONS;
      cold+=(arm?candidate(c,so,sa,sp,q,caches+(size_t)rot*CACHE_WORDS,ws+(size_t)rot*O_WORDS,scratch,res,false):control(c,co,ca,cp0,q,caches+(size_t)rot*CACHE_WORDS,ws+(size_t)rot*O_WORDS,res));}}
    printf("rep=%d arm=%d hot=%.6f cold=%.6f\\n",r,arm,hot/hp,cold/cp);
  }}return 0;}}
'''


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--hot-passes", type=int, default=32)
  ap.add_argument("--cold-passes", type=int, default=4)
  ap.add_argument("--reps", type=int, default=9)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--artifacts", type=Path, required=True)
  args = ap.parse_args()
  args.artifacts.mkdir(parents=True, exist_ok=True)
  kernels, names = _build_source()
  cu, binary = args.artifacts / "gate.cu", args.artifacts / "gate"
  cu.write_text(_harness(kernels, names))
  build = subprocess.run([NVCC, "-O3", "-std=c++17", "-arch=sm_120a", "--ptxas-options=-v", str(cu), "-o", str(binary)],
                         text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  (args.artifacts / "build.stdout.txt").write_text(build.stdout)
  (args.artifacts / "build.stderr.txt").write_text(build.stderr)
  if build.returncode: raise RuntimeError(build.stderr[-16000:])
  run = subprocess.run([str(binary), str(args.hot_passes), str(args.cold_passes), str(args.reps)],
                       text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  (args.artifacts / "run.stdout.txt").write_text(run.stdout)
  (args.artifacts / "run.stderr.txt").write_text(run.stderr)
  if run.returncode: raise RuntimeError(f"gate rc={run.returncode}: {run.stderr[-8000:]}")
  rows = []
  for line in run.stdout.splitlines():
    if m := re.match(r"rep=(\d+) arm=(\d+) hot=([0-9.]+) cold=([0-9.]+)", line):
      rows.append({"rep":int(m[1]), "arm":int(m[2]), "hot_us":float(m[3]), "cold_us":float(m[4])})
  med = {}
  for arm, name in enumerate(("control", "segmented_two_wave")):
    rr = [x for x in rows if x["arm"] == arm]
    med[name] = {"hot_us":statistics.median(x["hot_us"] for x in rr),
                 "cold_us":statistics.median(x["cold_us"] for x in rr)}
  tms = re.findall(r"timeline flash1_start=([0-9.]+) flash1_end=([0-9.]+) o0_start=([0-9.]+) o0_end=([0-9.]+) overlap=([0-9.]+)", run.stdout)
  timelines = [{"flash1_start_us":float(x[0]),"flash1_end_us":float(x[1]),"o_first_start_us":float(x[2]),
                "o_first_end_us":float(x[3]),"overlap_us":float(x[4])} for x in tms]
  exact = {k:f"{k}=1" in run.stdout for k in ("score_bitwise", "combine_bitwise", "final_bitwise")}
  recovery = med["control"]["cold_us"] - med["segmented_two_wave"]["cold_us"]
  result = {"schema":"tinygrad.nv_segmented_flash_o_complete_span.v1",
    "commit":subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT, text=True).strip(),
    "shape":{"heads":H,"kv_heads":HKV,"head_dim":HD,"splits":S,"max_context":MAXC,"tc":TC,"rows":ROWS},
    "accounting":{"control_kernels":3,"candidate_kernels":6,"o_weight_bytes":O_WORDS*4,
      "scratch_bytes":ROWS*16*4,"scratch_store_plus_read_bytes":ROWS*16*8},
    "kernel_names":names,"exact":exact,"samples":rows,"median":med,"cold_recovery_us_per_layer":recovery,
    "timeline_samples":timelines,"median_overlap_us":statistics.median(x["overlap_us"] for x in timelines) if timelines else None,
    "ptxas":build.stderr,"verdict":"PRIMITIVE_PASS" if all(exact.values()) and recovery >= 1.0 and timelines and
      statistics.median(x["overlap_us"] for x in timelines) > 0 else "NO_GO_PRIMITIVE"}
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(run.stdout.strip()); print(json.dumps(result, indent=2, sort_keys=True))
  return 0 if all(exact.values()) else 5


if __name__ == "__main__": raise SystemExit(main())
