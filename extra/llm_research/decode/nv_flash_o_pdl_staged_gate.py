#!/usr/bin/env python3
"""Exact full-grid Flash -> single staged O programmatic-launch microgate."""
from __future__ import annotations
import argparse, json, re, statistics, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tinygrad import dtypes
from tinygrad.codegen.late.warp_reduce import _warp_reduce_sum_staged
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, UOp
from tinygrad.llm.decode_kernels import (LanePartition, Q4KGEMVEpilogue, Q4KGateUpLaneMap,
  _q4k_block_dot_packed_load_vec, q4k_g3_lanemap_gemv_kernel)
from tinygrad.llm.qk_layout import Q4K_WORDS_PER_BLOCK
from extra.llm_research.decode.nv_flash_last_cta_combine_gate import source as flash_source, NVCC
from extra.llm_research.decode.nv_segmented_flash_o_complete_span import _render, _preambles, ROWS, O_WORDS, K


def _extract(src: str, name: str) -> str:
  for chunk in ('extern "C"' + x for x in src.split('extern "C"')[1:]):
    if name in chunk.split('{', 1)[0]: return chunk
  raise RuntimeError(f"missing {name}")


def _matching_brace(src: str, start: int) -> int:
  depth = 0
  for i in range(start, len(src)):
    if src[i] == '{': depth += 1
    elif src[i] == '}':
      depth -= 1
      if depth == 0: return i + 1
  raise RuntimeError("unbalanced generated source")


def _packed_staged_o_emitter():
  """Two rows/warp; each lane retains exact low/high lane-pair ownership."""
  lm = Q4KGateUpLaneMap(k=K, n=ROWS)
  lm.validate()
  def kernel(out: UOp, words: UOp, x: UOp, residual: UOp) -> UOp:
    lane, row_pair = UOp.special(32, 'lidx0'), UOp.special(ROWS//2, 'gidx0')
    lane16, row = lane.bitwise_and(15), row_pair * 2 + lane.rshift(4)
    def half(lane_base: int, rid: int, slot: int, dep: UOp|None = None) -> UOp:
      logical_lane = lane16 + lane_base
      part = LanePartition(logical_lane, lane_extent=lm.lane_extent, words_per_group=lm.words_per_group)
      lblk = UOp.range(lm.blocks_per_group, rid, axis_type=AxisType.REDUCE)
      blk = part.block_group * lm.blocks_per_group + lblk
      base = (row * lm.k_blocks + blk) * Q4K_WORDS_PER_BLOCK
      contrib = _q4k_block_dot_packed_load_vec(words, x, base, blk, part.word_col)
      acc = UOp.placeholder((1,), dtypes.float32, slot, addrspace=AddrSpace.REG)
      init = acc.after(*(x for x in (dep,) if x is not None))[0].store(0.0)
      acc = acc.after(init)
      acc = acc.after(acc[0].store(acc.after(lblk)[0] + contrib).end(lblk))
      return acc[0]
    low = half(0, 0, 20)
    high = half(16, 1, 21, low)
    # This is exactly the installed xor-16 pair, followed by its 8/4/2/1
    # ladder, expressed as two independent 16-lane rows in one physical warp.
    total = _warp_reduce_sum_staged(low + high, lane, 16, slot_base=90)
    return out[row].store(total + residual[row].cast(dtypes.float32), lane16.eq(0)).sink(
      arg=KernelInfo(name='q4k_o_pdl_staged16_packed_epi_resadd', opts_to_apply=()))
  return kernel


def build(prefetch_lines: int = 0) -> tuple[str, tuple[str, str, str]]:
  src, score_name, combine_name, fused_name = flash_source()
  # Preserve the passing last-CTA combine, add two 16-head release epochs, and
  # turn its first epoch into a PDL trigger.  Every high-head CTA triggers at
  # entry; every low-head CTA triggers only after its ordinary score/combine
  # work, so dependent launch release coincides with low-half readiness.
  src = src.replace('unsigned int* counters) {',
    'unsigned int* counters, unsigned int* groups, unsigned int* ready) {', 1)
  src = src.replace('if (tg_lane == 0) counters[gidx1]=0u;',
    'if (tg_lane == 0) { __threadfence(); unsigned int gg=gidx1>>4; '
    'if (atomicAdd(groups+gg,1u)==15u) { groups[gg]=0u; __threadfence(); ready[gg]=1u; } '
    'counters[gidx1]=0u; }', 1)
  fused = _extract(src, fused_name)
  fused_pdl_name = fused_name + f'_pdl_low16_pf{prefetch_lines}'
  fused = fused.replace(fused_name, fused_pdl_name, 1)
  if prefetch_lines:
    fused = fused.replace('unsigned int* ready) {', 'unsigned int* ready, unsigned char* ow) {', 1)
  head_decl = re.search(r'int gidx1 = blockIdx\.y;[^\n]*\n', fused)
  if head_decl is None: raise RuntimeError("missing head declaration")
  fused = fused[:head_decl.end()] + '  if (gidx1 >= 16) cudaTriggerProgrammaticLaunchCompletion();\n' + fused[head_decl.end():]
  # All low-head CTAs reach this point after their partial; the last CTA for
  # each head has also completed and published its exact combine output.
  tail = '  if (gidx1 < 16) cudaTriggerProgrammaticLaunchCompletion();\n'
  if prefetch_lines:
    tail += ('  unsigned int pfid=((gidx1*6u+(unsigned int)gidx0)*128u+(unsigned int)(lidx1*32+lidx0));\n'
             f'  #pragma unroll\n  for (int pfi=0;pfi<{prefetch_lines};pfi++) {{ unsigned int pfl=pfid+pfi*24576u; '
             'if (pfl<73728u) { const void* pfa=ow+(size_t)pfl*128u; asm volatile("prefetch.global.L2 [%0];" :: "l"(pfa)); } }\n')
  fused = fused[:fused.rfind('}')] + tail + fused[fused.rfind('}'):]

  p = UOp.placeholder
  o_name, o_src = _render(q4k_g3_lanemap_gemv_kernel(
    ROWS, K, epilogue=Q4KGEMVEpilogue('residual_add'), load_style='vector')(
      p((ROWS,), dtypes.float32, 0), p((O_WORDS,), dtypes.uint32, 1),
      p((K,), dtypes.float16, 2), p((ROWS,), dtypes.float32, 3)))
  staged_name, staged_src = _render(_packed_staged_o_emitter()(
    p((ROWS,), dtypes.float32, 0), p((O_WORDS,), dtypes.uint32, 1),
    p((K,), dtypes.float16, 2), p((ROWS,), dtypes.float32, 3)))
  ob = _extract(staged_src, staged_name).replace(') {', ', unsigned int* ready) {', 1)
  lane_decl = re.search(r'int lidx0 = threadIdx\.x;[^\n]*\n', ob)
  if lane_decl is None: raise RuntimeError('missing O lane declaration')
  ob = ob[:lane_decl.end()] + '  cudaGridDependencySynchronize();\n' + ob[lane_decl.end():]
  high_loop = ob.index('  for (int Ridx1')
  ob = ob[:high_loop] + '  while (((volatile unsigned int*)ready)[1] == 0u) __nanosleep(64);\n' + ob[high_loop:]
  combined = (_preambles(src, o_src) + _extract(src, score_name) + _extract(src, combine_name) +
              fused + _extract(o_src, o_name) + ob)
  return combined, (score_name, fused_pdl_name, staged_name)


HARNESS = r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
__SRC__
#define ROT 16
static void ck(cudaError_t e,const char*w){if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e));exit(2);}}
struct C{cudaStream_t s;cudaEvent_t st,done;};
static void init(C&c){ck(cudaStreamCreateWithFlags(&c.s,cudaStreamNonBlocking),"s");ck(cudaEventCreate(&c.st),"st");ck(cudaEventCreate(&c.done),"done");}
static float ctrl(C&c,float*out,half*att,float*part,float*q,unsigned*cache,unsigned*w,float*res){ck(cudaEventRecord(c.st,c.s),"st");__S__<<<dim3(6,32),dim3(32,4),0,c.s>>>(part,q,cache);flash_fused_gmax_combine_f16_32_128_s6_lw128<<<32,128,0,c.s>>>(att,part);q4k_g3_lanemap_gemv_vec_epi_resadd_4096_4096<<<4096,32,0,c.s>>>(out,w,att,res);ck(cudaEventRecord(c.done,c.s),"dn");ck(cudaEventSynchronize(c.done),"sy");float ms;ck(cudaEventElapsedTime(&ms,c.st,c.done),"el");return ms*1000;}
static float cand(C&c,float*out,half*att,float*part,float*q,unsigned*cache,unsigned*w,float*res,unsigned*ctr,unsigned*groups,unsigned*ready){ck(cudaEventRecord(c.st,c.s),"st");ck(cudaMemsetAsync(ctr,0,128,c.s),"cz");ck(cudaMemsetAsync(groups,0,8,c.s),"gz");ck(cudaMemsetAsync(ready,0,8,c.s),"rz");cudaLaunchConfig_t cfg={};cudaLaunchAttribute at[1]={};at[0].id=cudaLaunchAttributeProgrammaticStreamSerialization;at[0].val.programmaticStreamSerializationAllowed=1;cfg.attrs=at;cfg.numAttrs=1;cfg.stream=c.s;cfg.gridDim=dim3(6,32);cfg.blockDim=dim3(32,4);__FLAUNCH__ cfg.gridDim=dim3(4096);cfg.blockDim=dim3(32);ck(cudaLaunchKernelEx(&cfg,__O__,out,w,att,res,ready),"opdl");ck(cudaEventRecord(c.done,c.s),"dn");ck(cudaEventSynchronize(c.done),"sy");float ms;ck(cudaEventElapsedTime(&ms,c.st,c.done),"el");return ms*1000;}
int main(int ac,char**av){int hp=atoi(av[1]),cp=atoi(av[2]),reps=atoi(av[3]);C c;init(c);float *a,*b,*pa,*pb,*q,*res;half *aa,*bb;unsigned *wa,*wb,*ka,*kb,*ctr,*groups,*ready;cudaMalloc(&a,16384);cudaMalloc(&b,16384);cudaMalloc(&pa,99840);cudaMalloc(&pb,99840);cudaMalloc(&aa,8192);cudaMalloc(&bb,8192);cudaMalloc(&q,16384);cudaMalloc(&res,16384);cudaMalloc(&wa,(size_t)ROT*O_WORDS*4);cudaMalloc(&wb,(size_t)ROT*O_WORDS*4);cudaMalloc(&ka,(size_t)ROT*1048576*4);cudaMalloc(&kb,(size_t)ROT*1048576*4);cudaMalloc(&ctr,128);cudaMalloc(&groups,8);cudaMalloc(&ready,8);cudaMemset(ctr,0,128);cudaMemset(groups,0,8);cudaMemset(wa,7,(size_t)ROT*O_WORDS*4);cudaMemset(wb,7,(size_t)ROT*O_WORDS*4);cudaMemset(ka,0x30,(size_t)ROT*1048576*4);cudaMemset(kb,0x30,(size_t)ROT*1048576*4);std::vector<float>x(4096),r(4096);for(int i=0;i<4096;i++){x[i]=float((i*17%127)-63)/256;r[i]=float((i%31)-15)/128;}cudaMemcpy(q,x.data(),16384,cudaMemcpyHostToDevice);cudaMemcpy(res,r.data(),16384,cudaMemcpyHostToDevice);ctrl(c,a,aa,pa,q,ka,wa,res);cand(c,b,bb,pb,q,kb,wb,res,ctr,groups,ready);std::vector<unsigned char>ha(16384),hb(16384);cudaMemcpy(ha.data(),a,16384,cudaMemcpyDeviceToHost);cudaMemcpy(hb.data(),b,16384,cudaMemcpyDeviceToHost);printf("bitwise=%d\n",memcmp(ha.data(),hb.data(),16384)==0);for(int r0=0;r0<reps;r0++){double ah=0,bh=0,ac=0,bc=0;for(int i=0;i<hp;i++){ah+=ctrl(c,a,aa,pa,q,ka,wa,res);bh+=cand(c,b,bb,pb,q,kb,wb,res,ctr,groups,ready);}for(int i=0;i<cp;i++){int z=(r0*cp+i)%ROT;ac+=ctrl(c,a,aa,pa,q,ka+(size_t)z*1048576,wa+(size_t)z*O_WORDS,res);bc+=cand(c,b,bb,pb,q,kb+(size_t)z*1048576,wb+(size_t)z*O_WORDS,res,ctr,groups,ready);}printf("rep=%d control_hot=%.6f candidate_hot=%.6f control_cold=%.6f candidate_cold=%.6f\n",r0,ah/hp,bh/hp,ac/cp,bc/cp);}return 0;}
'''


def main() -> int:
  ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--hot',type=int,default=16);ap.add_argument('--cold',type=int,default=4);ap.add_argument('--reps',type=int,default=9);ap.add_argument('--prefetch-lines',type=int,default=0,choices=range(4));a=ap.parse_args()
  src,n=build(a.prefetch_lines);fargs='ck(cudaLaunchKernelEx(&cfg,'+n[1]+',part,q,cache,att,ctr,groups,ready'+(',((unsigned char*)w)' if a.prefetch_lines else '')+'),"fpdl");'
  cu=HARNESS.replace('__SRC__',src).replace('__S__',n[0]).replace('__F__',n[1]).replace('__O__',n[2]).replace('__FLAUNCH__',fargs).replace('O_WORDS',str(O_WORDS))
  with tempfile.TemporaryDirectory(prefix='flash_o_pdl_') as td:
    p=Path(td);(p/'g.cu').write_text(cu);b=subprocess.run([NVCC,'-O3','-arch=sm_120a','--ptxas-options=-v',str(p/'g.cu'),'-o',str(p/'g')],capture_output=True,text=True)
    if b.returncode: raise RuntimeError(b.stderr[-16000:])
    r=subprocess.run([str(p/'g'),str(a.hot),str(a.cold),str(a.reps)],capture_output=True,text=True,timeout=120)
  print(r.stdout);rows=[]
  for m in re.finditer(r'rep=(\d+) control_hot=([0-9.]+) candidate_hot=([0-9.]+) control_cold=([0-9.]+) candidate_cold=([0-9.]+)',r.stdout): rows.append([float(x) for x in m.groups()[1:]])
  med=[statistics.median(x[i] for x in rows) for i in range(4)]
  out={'schema':'tinygrad.nv_flash_o_pdl_staged.v1','prefetch_lines':a.prefetch_lines,'bitwise':'bitwise=1' in r.stdout,'medians':dict(zip(('control_hot','candidate_hot','control_cold','candidate_cold'),med)),'cold_recovery_us':med[2]-med[3],'samples':rows,'ptxas':b.stderr}
  a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));return 0 if out['bitwise'] else 5
if __name__=='__main__': raise SystemExit(main())
