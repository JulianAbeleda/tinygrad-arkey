#!/usr/bin/env python3
"""GPU Gate-2 for the research-only U4Z8_G64_P256 FP16 O projection.

The control is the installed production-rendered residual-fused Q4_K kernel.  The
numerical candidate consumes 34 uint32 words per 256 weights: words 0..1 hold
four FP16 scales and words 2..33 hold offset-binary nibbles in [group-pair][word-col]
layout.  No production route or storage is modified.
"""
from __future__ import annotations

import argparse, csv, io, json, os, re, shutil, statistics, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Target
from tinygrad.llm.decode_kernels import (LanePartition, Q4KGEMVEpilogue, Q4KGateUpLaneMap,
  _half4_lane, _lane_partition_reduce_sum, q4k_g3_lanemap_gemv_kernel)
from tinygrad.llm.qk_layout import Q4_K_BLOCK_ELEMS, Q4K_WORDS_PER_BLOCK
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp

ROWS, K = 4096, 4096
K_BLOCKS = K // Q4_K_BLOCK_ELEMS
CONTROL_WORDS = ROWS * K_BLOCKS * Q4K_WORDS_PER_BLOCK
CANDIDATE_WORDS = ROWS * K_BLOCKS * 34
CONTROL_WEIGHT_BYTES, CANDIDATE_WEIGHT_BYTES, ROTATIONS = CONTROL_WORDS * 4, CANDIDATE_WORDS * 4, 16
CUDA_BIN, NCU = "/usr/local/cuda-13.2/bin", "/usr/local/bin/ncu"
CONTROL = f"q4k_g3_lanemap_gemv_vec_epi_resadd_{ROWS}_{K}"
CANDIDATE = f"u4z8_g64_p256_lanemap_gemv_vec_epi_resadd_{ROWS}_{K}"


def emit_s4_o():
  lm = Q4KGateUpLaneMap(k=K, n=ROWS)
  lm.validate()

  def kernel(out:UOp, words:UOp, x:UOp, residual:UOp) -> UOp:
    row, lane = UOp.special(ROWS, "gidx0"), UOp.special(32, "lidx0")
    part = LanePartition(lane, lane_extent=lm.lane_extent, words_per_group=lm.words_per_group)
    lblk = UOp.range(lm.blocks_per_group, 0, axis_type=AxisType.REDUCE)
    blk = part.block_group * lm.blocks_per_group + lblk
    base = (row * lm.k_blocks + blk) * 34
    hdr = words.index(base).load(dtype=dtypes.uint32.vec(2))
    contrib = UOp.const(dtypes.float32, 0.0)
    for group_pair in range(4):
      qw = words[base + 2 + group_pair * 8 + part.word_col]
      for pair_member in range(2):
        grp = 2 * group_pair + pair_member
        scale_bits = hdr.gep(group_pair // 2).rshift((group_pair % 2) * 16).bitwise_and(0xffff)
        scale = scale_bits.cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)
        qpack = qw.rshift(pair_member * 4).bitwise_and(0x0F0F0F0F)
        xv = x.index(blk * Q4_K_BLOCK_ELEMS + grp * 32 + part.word_col * 4).load(dtype=dtypes.float16.vec(4))
        for nib in range(4):
          q = qpack.rshift(nib * 8).bitwise_and(0xf).cast(dtypes.float32)
          contrib = contrib + scale * (q - 8.0) * _half4_lane(xv, nib)
    acc = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(lblk)[0] + contrib).end(lblk))
    total = _lane_partition_reduce_sum(acc[0], part)
    return out[row].store(total + residual[row].cast(dtypes.float32)).sink(
      arg=KernelInfo(name=CANDIDATE, opts_to_apply=()))
  return kernel


def _render() -> tuple[str, str]:
  p = UOp.placeholder
  control = q4k_g3_lanemap_gemv_kernel(ROWS, K, epilogue=Q4KGEMVEpilogue("residual_add"), load_style="vector")(
    p((ROWS,), dtypes.float32, 0), p((CONTROL_WORDS,), dtypes.uint32, 1), p((K,), dtypes.float16, 2), p((ROWS,), dtypes.float32, 3))
  candidate = emit_s4_o()(p((ROWS,), dtypes.float32, 0), p((CANDIDATE_WORDS,), dtypes.uint32, 1),
    p((K,), dtypes.float16, 2), p((ROWS,), dtypes.float32, 3))
  ren = CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=False)
  def source(u:UOp) -> str:
    text = next(x.arg for x in to_program(u, ren).src if x.op is Ops.SOURCE)
    return text[text.index('extern "C" __global__'):]
  return source(control), source(candidate)


HARNESS = r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#define ROWS 4096
#define K 4096
#define K_BLOCKS 16
#define CONTROL_WORDS 2359296
#define CANDIDATE_WORDS 2228224
#define ROTATIONS 16
#define GUARD 32
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f=v; return u.t; }
struct __align__(8) half4 { half x,y,z,w; };
__device__ half4 make_half4(half x,half y,half z,half w) { half4 r={x,y,z,w}; return r; }

__CONTROL_SOURCE__
__CANDIDATE_SOURCE__

static void ck(cudaError_t e,const char* what) { if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",what,cudaGetErrorString(e));exit(2);} }
static uint32_t step(uint32_t& s) { s=1664525u*s+1013904223u; return s; }

static void fill_q4(uint32_t* w,int fixture) {
  uint32_t s=0x91427ab3u^(uint32_t(fixture)*0x9e3779b9u);
  const uint16_t hb[4]={0x2c00u,0x3000u,0x3400u,0x3800u};
  for(size_t b=0;b<(size_t)ROWS*K_BLOCKS;b++) { size_t z=b*36; int p=int((b+fixture)&3);
    w[z]=uint32_t(hb[p])|(uint32_t(hb[(p+1)&3])<<16);
    for(int i=1;i<36;i++) w[z+i]=step(s);
  }
}

// Legal U4Z8_G64: finite non-negative FP16 scales and all offset-binary nibble patterns.
static void fill_s4(uint32_t* w,half* x,float* residual,int fixture) {
  uint32_t s=0x53475032u^(uint32_t(fixture)*0x85ebca6bu);
  const uint16_t scales[8]={0x0000u,0x2800u,0x2c00u,0x3000u,0x3200u,0x3400u,0x3600u,0x3800u};
  for(size_t b=0;b<(size_t)ROWS*K_BLOCKS;b++) { size_t z=b*34;
    for(int i=0;i<2;i++) { uint16_t lo=scales[(b+i*2+fixture)%8],hi=scales[(b+i*2+1+fixture)%8]; w[z+i]=uint32_t(lo)|(uint32_t(hi)<<16); }
    for(int gp=0;gp<4;gp++) for(int wc=0;wc<8;wc++) {
      uint32_t q=fixture==0?step(s):(fixture==1?0xfedcba98u^(uint32_t(b)+uint32_t(gp*17+wc)):0x807f10e1u);
      w[z+2+gp*8+wc]=q;
    }
  }
  for(int i=0;i<K;i++) { float v=fixture==0?float(int(step(s)>>16)%511-255)/512.0f:
      fixture==1?float((i%257)-128)/256.0f:float((i%17)-8)/64.0f; x[i]=__float2half(v); }
  for(int i=0;i<ROWS;i++) residual[i]=fixture==0?float(int(step(s)>>16)%255-127)/1024.0f:
      fixture==1?float((i%31)-15)/128.0f:float((i%7)-3)/32.0f;
}

static float half_from_bits(uint16_t bits) { half h; memcpy(&h,&bits,2); return __half2float(h); }

// Independent scalar host oracle: direct logical weight indexing, double summation.
static double oracle_row(const uint32_t* w,const half* x,const float* residual,int row) {
  double sum=0.0;
  for(int blk=0;blk<K_BLOCKS;blk++) { const uint32_t* b=w+((size_t)row*K_BLOCKS+blk)*34;
    for(int g=0;g<8;g++) { int gp=g/2; uint16_t sb=uint16_t(b[gp/2]>>(16*(gp&1))); double sc=half_from_bits(sb);
      for(int j=0;j<32;j++) { int wc=j/4,nib=j&3; uint32_t qw=b[2+gp*8+wc]; int n=int((qw>>(4*(g&1)+8*nib))&15)-8;
        sum += sc*double(n)*double(__half2float(x[blk*256+g*32+j]));
      }
    }
  }
  return sum+double(residual[row]);
}

static void launch(int arm,float* out,uint32_t* w,half* x,float* residual,cudaStream_t s=0) {
  if(arm==0) q4k_g3_lanemap_gemv_vec_epi_resadd_4096_4096<<<ROWS,32,0,s>>>(out,w,x,residual);
  else u4z8_g64_p256_lanemap_gemv_vec_epi_resadd_4096_4096<<<ROWS,32,0,s>>>(out,w,x,residual);
}
static double hot(int arm,float* out,uint32_t* w,half* x,float* r,int passes,cudaEvent_t a,cudaEvent_t b) {
  ck(cudaEventRecord(a),"hot start"); for(int i=0;i<passes;i++)launch(arm,out,w,x,r); ck(cudaEventRecord(b),"hot stop");ck(cudaEventSynchronize(b),"hot sync");
  float ms=0;ck(cudaEventElapsedTime(&ms,a,b),"hot elapsed");return ms*1000.0/passes;
}
static double rotated(int arm,float* out,uint32_t* ring,half* x,float* r,int passes,cudaEvent_t a,cudaEvent_t b) {
  size_t words=arm?CANDIDATE_WORDS:CONTROL_WORDS;double total=0;for(int i=0;i<passes;i++){uint32_t* w=ring+(size_t)(i%ROTATIONS)*words;ck(cudaEventRecord(a),"cold start");launch(arm,out,w,x,r);ck(cudaEventRecord(b),"cold stop");ck(cudaEventSynchronize(b),"cold sync");float ms=0;ck(cudaEventElapsedTime(&ms,a,b),"cold elapsed");total+=ms*1000.0;}return total/passes;
}
static double rotated_batch(int arm,float* out,uint32_t* ring,half* x,float* r,int passes,cudaEvent_t a,cudaEvent_t b) {
  size_t words=arm?CANDIDATE_WORDS:CONTROL_WORDS;ck(cudaEventRecord(a),"batch cold start");for(int i=0;i<passes;i++){uint32_t* w=ring+(size_t)(i%ROTATIONS)*words;launch(arm,out,w,x,r);}ck(cudaEventRecord(b),"batch cold stop");ck(cudaEventSynchronize(b),"batch cold sync");float ms=0;ck(cudaEventElapsedTime(&ms,a,b),"batch cold elapsed");return ms*1000.0/passes;
}

int main(int argc,char** argv) {
  int hp=argc>1?atoi(argv[1]):300,cp=argc>2?atoi(argv[2]):32,reps=argc>3?atoi(argv[3]):9; bool profile=argc>1&&!strcmp(argv[1],"profile"),batched=argc>4&&!strcmp(argv[4],"batch");
  uint32_t *wc=nullptr,*ws=nullptr;half* x=nullptr;float *r=nullptr,*ocbase=nullptr,*osbase=nullptr;float *oc,*os;
  ck(cudaMalloc(&wc,(size_t)ROTATIONS*CONTROL_WORDS*4),"control weights");ck(cudaMalloc(&ws,(size_t)ROTATIONS*CANDIDATE_WORDS*4),"s4 weights");
  ck(cudaMalloc(&x,K*sizeof(half)),"x");ck(cudaMalloc(&r,ROWS*4),"residual");ck(cudaMalloc(&ocbase,(ROWS+2*GUARD)*4),"control output");ck(cudaMalloc(&osbase,(ROWS+2*GUARD)*4),"s4 output");oc=ocbase+GUARD;os=osbase+GUARD;
  uint32_t *hqc=(uint32_t*)malloc((size_t)CONTROL_WORDS*4),*hqs=(uint32_t*)malloc((size_t)CANDIDATE_WORDS*4),*check=(uint32_t*)malloc((size_t)CANDIDATE_WORDS*4);half* hx=(half*)malloc(K*2),*xcheck=(half*)malloc(K*2);float *hr=(float*)malloc(ROWS*4),*rcheck=(float*)malloc(ROWS*4),*hout=(float*)malloc((ROWS+2*GUARD)*4);
  if(!hqc||!hqs||!check||!hx||!xcheck||!hr||!rcheck||!hout){fprintf(stderr,"host allocation failed\n");return 3;}
  int all=1; const float sentinel=12345.25f;
  for(int fixture=0;fixture<3;fixture++) {fill_q4(hqc,fixture);fill_s4(hqs,hx,hr,fixture);ck(cudaMemcpy(wc,hqc,(size_t)CONTROL_WORDS*4,cudaMemcpyHostToDevice),"q4 fixture");ck(cudaMemcpy(ws,hqs,(size_t)CANDIDATE_WORDS*4,cudaMemcpyHostToDevice),"s4 fixture");ck(cudaMemcpy(x,hx,K*2,cudaMemcpyHostToDevice),"x fixture");ck(cudaMemcpy(r,hr,ROWS*4,cudaMemcpyHostToDevice),"r fixture");
    for(int i=0;i<ROWS+2*GUARD;i++)hout[i]=sentinel;ck(cudaMemcpy(ocbase,hout,(ROWS+2*GUARD)*4,cudaMemcpyHostToDevice),"q4 guards");ck(cudaMemcpy(osbase,hout,(ROWS+2*GUARD)*4,cudaMemcpyHostToDevice),"s4 guards");launch(0,oc,wc,x,r);launch(1,os,ws,x,r);ck(cudaDeviceSynchronize(),"fixture sync");
    ck(cudaMemcpy(hout,osbase,(ROWS+2*GUARD)*4,cudaMemcpyDeviceToHost),"s4 result");ck(cudaMemcpy(check,ws,(size_t)CANDIDATE_WORDS*4,cudaMemcpyDeviceToHost),"s4 readonly");ck(cudaMemcpy(xcheck,x,K*2,cudaMemcpyDeviceToHost),"x readonly");ck(cudaMemcpy(rcheck,r,ROWS*4,cudaMemcpyDeviceToHost),"r readonly");
    int guards=1,finite=1,bad=0;double maxabs=0,maxrel=0;for(int i=0;i<GUARD;i++)guards&=hout[i]==sentinel&&hout[GUARD+ROWS+i]==sentinel;for(int row=0;row<ROWS;row++){double ref=oracle_row(hqs,hx,hr,row),got=hout[GUARD+row],ae=fabs(got-ref),re=ae/fmax(1.0,fabs(ref));finite&=isfinite(got)&&isfinite(ref);bad+=!(ae<=0.02+2e-5*fabs(ref));maxabs=fmax(maxabs,ae);maxrel=fmax(maxrel,re);}int readonly=!memcmp(check,hqs,(size_t)CANDIDATE_WORDS*4)&&!memcmp(xcheck,hx,K*2)&&!memcmp(rcheck,hr,ROWS*4);
    printf("fixture=%d finite=%d guards=%d readonly=%d bad=%d max_abs=%.9g max_rel=%.9g\n",fixture,finite,guards,readonly,bad,maxabs,maxrel);all&=finite&&guards&&readonly&&bad==0;
  }
  fill_q4(hqc,0);fill_s4(hqs,hx,hr,0);for(int i=0;i<ROTATIONS;i++){ck(cudaMemcpy(wc+(size_t)i*CONTROL_WORDS,hqc,(size_t)CONTROL_WORDS*4,cudaMemcpyHostToDevice),"q4 rotation");ck(cudaMemcpy(ws+(size_t)i*CANDIDATE_WORDS,hqs,(size_t)CANDIDATE_WORDS*4,cudaMemcpyHostToDevice),"s4 rotation");}ck(cudaMemcpy(x,hx,K*2,cudaMemcpyHostToDevice),"timing x");ck(cudaMemcpy(r,hr,ROWS*4,cudaMemcpyHostToDevice),"timing residual");free(hqc);free(hqs);free(check);free(hx);free(xcheck);free(hr);free(rcheck);free(hout);
  for(int i=0;i<20;i++){launch(0,oc,wc,x,r);launch(1,os,ws,x,r);}ck(cudaDeviceSynchronize(),"warm sync");if(profile){launch(0,oc,wc,x,r);launch(1,os,ws,x,r);ck(cudaDeviceSynchronize(),"profile sync");return all?0:5;}
  cudaEvent_t a,b;ck(cudaEventCreate(&a),"event");ck(cudaEventCreate(&b),"event");for(int z=0;z<reps;z++){double ch,sh,cc,sc;if(!(z&1)){ch=hot(0,oc,wc,x,r,hp,a,b);sh=hot(1,os,ws,x,r,hp,a,b);cc=batched?rotated_batch(0,oc,wc,x,r,cp,a,b):rotated(0,oc,wc,x,r,cp,a,b);sc=batched?rotated_batch(1,os,ws,x,r,cp,a,b):rotated(1,os,ws,x,r,cp,a,b);}else{sh=hot(1,os,ws,x,r,hp,a,b);ch=hot(0,oc,wc,x,r,hp,a,b);sc=batched?rotated_batch(1,os,ws,x,r,cp,a,b):rotated(1,os,ws,x,r,cp,a,b);cc=batched?rotated_batch(0,oc,wc,x,r,cp,a,b):rotated(0,oc,wc,x,r,cp,a,b);}printf("rep=%d hot_control_us=%.6f hot_candidate_us=%.6f cold_control_us=%.6f cold_candidate_us=%.6f\n",z,ch,sh,cc,sc);}return all?0:5;
}
'''


def _sass(binary:Path, symbol:str) -> dict:
  try: text=subprocess.check_output([f"{CUDA_BIN}/cuobjdump","--dump-sass",str(binary)],text=True,stderr=subprocess.STDOUT,env={**os.environ,"NVDISASM_PATH":str(ROOT/".venv/lib/python3.12/site-packages/triton/backends/nvidia/bin")})
  except (OSError,subprocess.CalledProcessError) as e:return {"available":False,"reason":str(e),"detail":getattr(e,"output","")[-2000:]}
  marker=f"Function : {symbol}"
  if marker not in text:return {"available":False,"reason":f"missing {symbol}"}
  body=text.split(marker,1)[1].split("Function :",1)[0];ops=re.findall(r"/\*[0-9a-f]+\*/\s+(?:@[!P0-9]+\s+)?([A-Z][A-Z0-9_.]+)",body);ldg=[x for x in ops if x.startswith("LDG")]
  return {"available":True,"instructions":len(ops),"ldg":len(ldg),"ldg_128":sum(".128" in x for x in ldg),"ldg_64":sum(".64" in x for x in ldg),"opcodes":{x:ops.count(x) for x in sorted(set(ops))}}


def _ncu(binary:Path,symbol:str,artifact:Path)->dict:
  metrics=",".join(["dram__bytes.sum","dram__bytes_op_read.sum","dram__throughput.avg.pct_of_peak_sustained_elapsed","gpu__time_duration.sum","lts__t_bytes.sum","lts__t_sector_op_read_hit_rate.pct","sm__inst_executed.sum","sm__throughput.avg.pct_of_peak_sustained_elapsed","launch__registers_per_thread","smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"])
  cp=subprocess.run(["sudo","-n",NCU,"-k",symbol,"--launch-skip","1","--launch-count","1","--cache-control","all","--metrics",metrics,"--csv",str(binary),"profile"],capture_output=True,text=True);artifact.write_text(cp.stdout+"\nSTDERR\n"+cp.stderr);rows=[];header=None
  for cols in csv.reader(io.StringIO(cp.stdout)):
    if cols and cols[0]=="ID":header=cols;continue
    if header is not None and len(cols)==len(header):
      row=dict(zip(header,cols));rows.append({"metric":row.get("Metric Name"),"unit":row.get("Metric Unit"),"value":row.get("Metric Value")})
  return {"returncode":cp.returncode,"rows":rows,"stderr_tail":cp.stderr[-2000:]}


def main()->int:
  ap=argparse.ArgumentParser();ap.add_argument("--hot-passes",type=int,default=300);ap.add_argument("--cold-passes",type=int,default=32);ap.add_argument("--reps",type=int,default=9);ap.add_argument("--cold-mode",choices=("per_call","batch"),default="per_call");ap.add_argument("--threshold-us",type=float,default=0.15);ap.add_argument("--out",type=Path,required=True);ap.add_argument("--artifact-dir",type=Path);ap.add_argument("--ncu",action="store_true");a=ap.parse_args()
  control,candidate=_render();source=HARNESS.replace("__CONTROL_SOURCE__",control).replace("__CANDIDATE_SOURCE__",candidate)
  with tempfile.TemporaryDirectory(prefix="nv_s4_g32_p256_o_") as td:
    src,binary=Path(td)/"gate.cu",Path(td)/"gate";src.write_text(source);env={**os.environ,"PATH":f"{CUDA_BIN}:"+os.environ.get("PATH","")};build=subprocess.run(["nvcc","-arch=sm_120a","-O3","-std=c++17","--ptxas-options=-v",str(src),"-o",str(binary)],capture_output=True,text=True,env=env)
    if build.returncode: print(build.stderr[-12000:],file=sys.stderr);return 3
    artifact=a.artifact_dir
    if artifact:artifact.mkdir(parents=True,exist_ok=True);shutil.copy2(src,artifact/"gate.cu");shutil.copy2(binary,artifact/"gate");(artifact/"ptxas.txt").write_text(build.stderr)
    run=subprocess.run([str(binary),str(a.hot_passes),str(a.cold_passes),str(a.reps),a.cold_mode],capture_output=True,text=True);print(run.stdout.strip())
    if run.returncode not in (0,5):print(run.stderr[-8000:],file=sys.stderr);return 4
    fixtures=[{"fixture":int(m[1]),"finite":bool(int(m[2])),"guards":bool(int(m[3])),"readonly":bool(int(m[4])),"bad":int(m[5]),"max_abs":float(m[6]),"max_rel":float(m[7])} for m in re.finditer(r"fixture=(\d+) finite=(\d+) guards=(\d+) readonly=(\d+) bad=(\d+) max_abs=([0-9.eE+-]+) max_rel=([0-9.eE+-]+)",run.stdout)]
    samples={k:[] for k in ("hot_control","hot_candidate","cold_control","cold_candidate")};pat=re.compile(r"hot_control_us=([0-9.]+) hot_candidate_us=([0-9.]+) cold_control_us=([0-9.]+) cold_candidate_us=([0-9.]+)")
    for m in pat.finditer(run.stdout):
      for k,v in zip(samples,m.groups()):samples[k].append(float(v))
    med={k:statistics.median(v) for k,v in samples.items()};cold=med["cold_control"]-med["cold_candidate"];correct=len(fixtures)==3 and all(x["finite"] and x["guards"] and x["readonly"] and x["bad"]==0 for x in fixtures)
    result={"schema":"tinygrad.nv_u4z8_g64_p256_o_microgate.v1","commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"method":f"production-rendered Q4_K residual O control versus numerical U4Z8_G64_P256; cudaEvent hot and 16-copy rotated-cold {a.cold_mode} r{a.reps}","shape":{"rows":ROWS,"k":K,"blocks_per_row":K_BLOCKS,"control_words_per_block":36,"candidate_words_per_block":34,"control_weight_bytes":CONTROL_WEIGHT_BYTES,"candidate_weight_bytes":CANDIDATE_WEIGHT_BYTES,"control_rotated_weight_bytes":ROTATIONS*CONTROL_WEIGHT_BYTES,"candidate_rotated_weight_bytes":ROTATIONS*CANDIDATE_WEIGHT_BYTES},"correctness":{"contract":"independent double host oracle; abs <= 0.02 + 2e-5*abs(reference)","fixtures":fixtures,"pass":correct},"storage_contract":{"scales":"words 0..1: four FP16 via uint16 bitcast","codes":"words 2..33: unsigned nibbles decoded as q-8 [4 group-pairs][8 word-cols]","bytes_per_256":136,"byte_reduction_fraction":1-CANDIDATE_WEIGHT_BYTES/CONTROL_WEIGHT_BYTES},"timing":{"unit":"us_per_call","cold_mode":a.cold_mode,"hot_passes":a.hot_passes,"cold_passes":a.cold_passes,"reps":a.reps,"samples":samples,"medians":med,"hot_recovery_us":med["hot_control"]-med["hot_candidate"],"cold_recovery_us":cold,"control_cold_rate_tb_s":CONTROL_WEIGHT_BYTES/med["cold_control"]/1e6,"candidate_cold_rate_tb_s":CANDIDATE_WEIGHT_BYTES/med["cold_candidate"]/1e6},"sass":{"control":_sass(binary,CONTROL),"candidate":_sass(binary,CANDIDATE)},"ptxas":build.stderr.strip().splitlines(),"threshold":{"cold_recovery_us_per_call":a.threshold_us},"verdict":"PASS" if correct and cold>=a.threshold_us else "STOP"}
    if a.ncu:
      if artifact is None:artifact=a.out.parent/"artifacts";artifact.mkdir(parents=True,exist_ok=True)
      result["ncu"]={"control":_ncu(binary,CONTROL,artifact/"ncu-control.csv.txt"),"candidate":_ncu(binary,CANDIDATE,artifact/"ncu-candidate.csv.txt")}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2,sort_keys=True));return 0 if correct else 5

if __name__=="__main__":raise SystemExit(main())
