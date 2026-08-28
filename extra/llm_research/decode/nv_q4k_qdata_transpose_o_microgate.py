#!/usr/bin/env python3
"""Research-only exact Q4_K qdata-transpose gate for the FP16 O projection.

The installed control uses the promoted residual-fused vector-load Q4_K body.
The candidate changes only the packed-Q storage inside each 36-word Q4_K block:

  control:   qdata[group_pair][word_col] at base + 4 + group_pair*8 + word_col
  candidate: qdata[word_col][group_pair] at base + 4 + word_col*4 + group_pair

The four qdata words owned by one lane are consequently one aligned uint4 load.
Headers, activation loads, dequantization, accumulation order, warp reduction,
output type, residual epilogue, launch geometry, and total weight bytes stay fixed.
No production route or model storage is modified by this experiment.
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
  _half4_lane, _lane_partition_reduce_sum, _q4k_group_params_from_words, q4k_g3_lanemap_gemv_kernel)
from tinygrad.llm.qk_layout import Q4_K_BLOCK_ELEMS, Q4K_WORDS_PER_BLOCK
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp

ROWS, K = 4096, 4096
K_BLOCKS = K // Q4_K_BLOCK_ELEMS
WORDS = ROWS * K_BLOCKS * Q4K_WORDS_PER_BLOCK
WEIGHT_BYTES = WORDS * 4
ROTATIONS = 16
CUDA_BIN = "/usr/local/cuda-13.2/bin"
NCU = "/usr/local/bin/ncu"
CONTROL = f"q4k_g3_lanemap_gemv_vec_epi_resadd_{ROWS}_{K}"
CANDIDATE = f"q4k_g3_lanemap_gemv_qdata_t_epi_resadd_{ROWS}_{K}"


def emit_qdata_transposed_o():
  """Exact installed O body with qdata [4][8] repacked as [8][4]."""
  lm = Q4KGateUpLaneMap(k=K, n=ROWS)
  lm.validate()

  def kernel(out:UOp, words:UOp, x:UOp, residual:UOp) -> UOp:
    row, lane = UOp.special(ROWS, "gidx0"), UOp.special(32, "lidx0")
    part = LanePartition(lane, lane_extent=lm.lane_extent, words_per_group=lm.words_per_group)
    lblk = UOp.range(lm.blocks_per_group, 0, axis_type=AxisType.REDUCE)
    blk = part.block_group * lm.blocks_per_group + lblk
    base = (row * lm.k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    hdr = words.index(base).load(dtype=dtypes.uint32.vec(4))
    qdata = words.index(base + 4 + part.word_col * 4).load(dtype=dtypes.uint32.vec(4))
    contrib = UOp.const(dtypes.float32, 0.0)
    # This loop order is deliberately identical to _q4k_block_dot_packed_load_vec.
    for group_pair in range(4):
      qw = qdata.gep(group_pair)
      for pair_member in range(2):
        grp = 2 * group_pair + pair_member
        d, dmin, sc, mn = _q4k_group_params_from_words(hdr.gep(0), hdr.gep(1), hdr.gep(2), hdr.gep(3), grp)
        qpack = qw.rshift(pair_member * 4).bitwise_and(0x0F0F0F0F)
        xv = x.index(blk * Q4_K_BLOCK_ELEMS + grp * 32 + part.word_col * 4).load(dtype=dtypes.float16.vec(4))
        for nib in range(4):
          q = qpack.rshift(nib * 8).bitwise_and(0xf)
          weight = d * sc.cast(dtypes.float32) * q.cast(dtypes.float32) - dmin * mn.cast(dtypes.float32)
          contrib = contrib + weight * _half4_lane(xv, nib)
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
    p((ROWS,), dtypes.float32, 0), p((WORDS,), dtypes.uint32, 1), p((K,), dtypes.float16, 2), p((ROWS,), dtypes.float32, 3))
  candidate = emit_qdata_transposed_o()(
    p((ROWS,), dtypes.float32, 0), p((WORDS,), dtypes.uint32, 1), p((K,), dtypes.float16, 2), p((ROWS,), dtypes.float32, 3))
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
#define WORDS 2359296
#define ROTATIONS 16
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f=v; return u.t; }
struct __align__(8) half4 { half x,y,z,w; };
__device__ half4 make_half4(half x,half y,half z,half w) { half4 r={x,y,z,w}; return r; }

__CONTROL_SOURCE__
__CANDIDATE_SOURCE__

static void ck(cudaError_t e,const char* what) { if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",what,cudaGetErrorString(e));exit(2);} }
static uint32_t step(uint32_t& s) { s=1664525u*s+1013904223u; return s; }

static void transpose_blocks(const uint32_t* src,uint32_t* dst) {
  for(size_t b=0;b<(size_t)ROWS*K_BLOCKS;b++) {
    size_t base=b*36;
    for(int i=0;i<4;i++) dst[base+i]=src[base+i];
    for(int gp=0;gp<4;gp++) for(int wc=0;wc<8;wc++)
      dst[base+4+wc*4+gp]=src[base+4+gp*8+wc];
  }
}

// All fixtures use finite positive fp16 d/dmin metadata and legal packed Q/scales.
static void fill_fixture(uint32_t* w,half* x,float* residual,int fixture) {
  uint32_t state=0x1234567u ^ (uint32_t(fixture)*0x9e3779b9u);
  const uint16_t dbits[4]={0x2c00u,0x3000u,0x3400u,0x3800u};
  const uint16_t mbits[4]={0x2800u,0x2c00u,0x3000u,0x3400u};
  for(size_t b=0;b<(size_t)ROWS*K_BLOCKS;b++) {
    size_t base=b*36; int p=int((b+fixture)%4);
    w[base]=uint32_t(dbits[p]) | (uint32_t(mbits[(p+1)&3])<<16);
    if(fixture==0) {
      w[base+1]=step(state); w[base+2]=step(state); w[base+3]=step(state);
      for(int i=4;i<36;i++) w[base+i]=step(state);
    } else if(fixture==1) {
      w[base+1]=0x01020304u^(uint32_t)b; w[base+2]=0x10203040u+(uint32_t)b; w[base+3]=0x3f2f1f0fu;
      for(int i=4;i<36;i++) w[base+i]=0x01234567u*uint32_t(i)+(uint32_t)b*0x11111111u;
    } else {
      w[base+1]=0x3f3f3f3fu; w[base+2]=0x15151515u; w[base+3]=0x2a2a2a2au;
      for(int i=4;i<36;i++) w[base+i]=(i&1)?0xfedcba98u:0x76543210u;
    }
  }
  for(int i=0;i<K;i++) {
    float v=fixture==0?float((int(step(state)>>16)%511)-255)/512.0f:
            fixture==1?float((i%257)-128)/256.0f:float((i%17)-8)/64.0f;
    x[i]=__float2half(v);
  }
  for(int i=0;i<ROWS;i++) residual[i]=fixture==0?float((int(step(state)>>16)%255)-127)/1024.0f:
                                      fixture==1?float((i%31)-15)/128.0f:float((i%7)-3)/32.0f;
}

static void launch(int arm,float* out,uint32_t* w,half* x,float* residual,cudaStream_t s=0) {
  if(arm==0) q4k_g3_lanemap_gemv_vec_epi_resadd_4096_4096<<<ROWS,32,0,s>>>(out,w,x,residual);
  else q4k_g3_lanemap_gemv_qdata_t_epi_resadd_4096_4096<<<ROWS,32,0,s>>>(out,w,x,residual);
}

static double hot(int arm,float* out,uint32_t* w,half* x,float* residual,int passes,cudaEvent_t start,cudaEvent_t stop) {
  ck(cudaEventRecord(start),"hot start");
  for(int i=0;i<passes;i++) launch(arm,out,w,x,residual);
  ck(cudaEventRecord(stop),"hot stop"); ck(cudaEventSynchronize(stop),"hot sync");
  float ms=0; ck(cudaEventElapsedTime(&ms,start,stop),"hot elapsed"); return ms*1000.0/passes;
}

static double rotated(int arm,float* out,uint32_t* rotations,half* x,float* residual,int passes,cudaEvent_t start,cudaEvent_t stop) {
  double total=0.0;
  for(int i=0;i<passes;i++) {
    uint32_t* w=rotations+(size_t)(i%ROTATIONS)*WORDS;
    ck(cudaEventRecord(start),"cold start"); launch(arm,out,w,x,residual); ck(cudaEventRecord(stop),"cold stop");
    ck(cudaEventSynchronize(stop),"cold sync"); float ms=0; ck(cudaEventElapsedTime(&ms,start,stop),"cold elapsed"); total+=ms*1000.0;
  }
  return total/passes;
}

int main(int argc,char** argv) {
  int hot_passes=argc>1?atoi(argv[1]):300,cold_passes=argc>2?atoi(argv[2]):32,reps=argc>3?atoi(argv[3]):9;
  bool profile=argc>1 && !strcmp(argv[1],"profile");
  uint32_t *wc=nullptr,*wt=nullptr; half* x=nullptr; float *residual=nullptr,*outc=nullptr,*outt=nullptr;
  ck(cudaMalloc(&wc,(size_t)ROTATIONS*WORDS*4),"control weights");
  ck(cudaMalloc(&wt,(size_t)ROTATIONS*WORDS*4),"candidate weights");
  ck(cudaMalloc(&x,K*sizeof(half)),"x"); ck(cudaMalloc(&residual,ROWS*sizeof(float)),"residual");
  ck(cudaMalloc(&outc,ROWS*sizeof(float)),"control output"); ck(cudaMalloc(&outt,ROWS*sizeof(float)),"candidate output");
  uint32_t* hw=(uint32_t*)malloc((size_t)WORDS*4),*ht=(uint32_t*)malloc((size_t)WORDS*4);
  half* hx=(half*)malloc(K*sizeof(half)); float* hr=(float*)malloc(ROWS*sizeof(float));
  if(!hw||!ht||!hx||!hr){fprintf(stderr,"host allocation failed\n");return 3;}

  int exact_all=1;
  for(int fixture=0;fixture<3;fixture++) {
    fill_fixture(hw,hx,hr,fixture); transpose_blocks(hw,ht);
    ck(cudaMemcpy(wc,hw,(size_t)WORDS*4,cudaMemcpyHostToDevice),"control fixture");
    ck(cudaMemcpy(wt,ht,(size_t)WORDS*4,cudaMemcpyHostToDevice),"candidate fixture");
    ck(cudaMemcpy(x,hx,K*sizeof(half),cudaMemcpyHostToDevice),"x fixture");
    ck(cudaMemcpy(residual,hr,ROWS*sizeof(float),cudaMemcpyHostToDevice),"residual fixture");
    launch(0,outc,wc,x,residual); launch(1,outt,wt,x,residual); ck(cudaDeviceSynchronize(),"fixture sync");
    float *hc=(float*)malloc(ROWS*4),*htout=(float*)malloc(ROWS*4);
    ck(cudaMemcpy(hc,outc,ROWS*4,cudaMemcpyDeviceToHost),"control result"); ck(cudaMemcpy(htout,outt,ROWS*4,cudaMemcpyDeviceToHost),"candidate result");
    int mismatch=0,finite=1; double max_abs=0.0;
    for(int i=0;i<ROWS;i++){uint32_t a,b;memcpy(&a,hc+i,4);memcpy(&b,htout+i,4);mismatch+=a!=b;finite&=isfinite(hc[i])&&isfinite(htout[i]);max_abs=fmax(max_abs,fabs(double(hc[i])-double(htout[i])));}
    printf("fixture=%d finite=%d mismatched_words=%d max_abs=%.9g\n",fixture,finite,mismatch,max_abs);
    exact_all &= finite && mismatch==0; free(hc); free(htout);
  }

  // Fixture zero is the timed production-shaped legal/random input.  Replicate
  // both physical layouts across a >L2 ring without including copies in timing.
  fill_fixture(hw,hx,hr,0); transpose_blocks(hw,ht);
  for(int r=0;r<ROTATIONS;r++) {
    ck(cudaMemcpy(wc+(size_t)r*WORDS,hw,(size_t)WORDS*4,cudaMemcpyHostToDevice),"control rotation");
    ck(cudaMemcpy(wt+(size_t)r*WORDS,ht,(size_t)WORDS*4,cudaMemcpyHostToDevice),"candidate rotation");
  }
  ck(cudaMemcpy(x,hx,K*sizeof(half),cudaMemcpyHostToDevice),"timing x"); ck(cudaMemcpy(residual,hr,ROWS*sizeof(float),cudaMemcpyHostToDevice),"timing residual");
  free(hw);free(ht);free(hx);free(hr);
  for(int i=0;i<20;i++){launch(0,outc,wc,x,residual);launch(1,outt,wt,x,residual);} ck(cudaDeviceSynchronize(),"warm sync");
  if(profile){launch(0,outc,wc,x,residual);launch(1,outt,wt,x,residual);ck(cudaDeviceSynchronize(),"profile sync");return exact_all?0:5;}
  cudaEvent_t start,stop;ck(cudaEventCreate(&start),"event");ck(cudaEventCreate(&stop),"event");
  for(int r=0;r<reps;r++) {
    double ch,th,cc,tc;
    if((r&1)==0) {ch=hot(0,outc,wc,x,residual,hot_passes,start,stop);th=hot(1,outt,wt,x,residual,hot_passes,start,stop);cc=rotated(0,outc,wc,x,residual,cold_passes,start,stop);tc=rotated(1,outt,wt,x,residual,cold_passes,start,stop);}
    else {th=hot(1,outt,wt,x,residual,hot_passes,start,stop);ch=hot(0,outc,wc,x,residual,hot_passes,start,stop);tc=rotated(1,outt,wt,x,residual,cold_passes,start,stop);cc=rotated(0,outc,wc,x,residual,cold_passes,start,stop);}
    printf("rep=%d hot_control_us=%.6f hot_candidate_us=%.6f cold_control_us=%.6f cold_candidate_us=%.6f\n",r,ch,th,cc,tc);
  }
  return exact_all?0:5;
}
'''


def _sass_census(binary:Path, symbol:str) -> dict:
  try:
    text = subprocess.check_output([f"{CUDA_BIN}/cuobjdump", "--dump-sass", str(binary)], text=True, stderr=subprocess.STDOUT,
      env={**os.environ, "NVDISASM_PATH":str(ROOT/".venv/lib/python3.12/site-packages/triton/backends/nvidia/bin")})
  except (OSError, subprocess.CalledProcessError) as exc:
    return {"available":False,"reason":str(exc),"detail":getattr(exc,"output","")[-2000:]}
  marker=f"Function : {symbol}"
  if marker not in text:return {"available":False,"reason":f"missing {symbol}"}
  body=text.split(marker,1)[1].split("Function :",1)[0]
  ops=re.findall(r"/\*[0-9a-f]+\*/\s+(?:@[!P0-9]+\s+)?([A-Z][A-Z0-9_.]+)",body)
  ldg=[x for x in ops if x.startswith("LDG")]
  return {"available":True,"instructions":len(ops),"ldg":len(ldg),"ldg_128":sum(".128" in x for x in ldg),
          "ldg_64":sum(".64" in x for x in ldg),"opcodes":{op:ops.count(op) for op in sorted(set(ops))}}


def _ncu(binary:Path, symbol:str, artifact:Path) -> dict:
  metrics=",".join(["dram__bytes.sum","dram__bytes_op_read.sum","dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__time_duration.sum","lts__t_bytes.sum","lts__t_sector_op_read_hit_rate.pct","sm__inst_executed.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed","launch__registers_per_thread",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"])
  cp=subprocess.run(["sudo","-n",NCU,"-k",symbol,"--launch-skip","1","--launch-count","1","--cache-control","all",
    "--metrics",metrics,"--csv",str(binary),"profile"],capture_output=True,text=True)
  artifact.write_text(cp.stdout+"\nSTDERR\n"+cp.stderr)
  rows=[];header=None
  for cols in csv.reader(io.StringIO(cp.stdout)):
    if cols and cols[0]=="ID":header=cols;continue
    if header is not None and len(cols)==len(header):
      row=dict(zip(header,cols));rows.append({"metric":row.get("Metric Name"),"unit":row.get("Metric Unit"),"value":row.get("Metric Value")})
  return {"returncode":cp.returncode,"rows":rows,"stderr_tail":cp.stderr[-2000:]}


def main() -> int:
  ap=argparse.ArgumentParser()
  ap.add_argument("--hot-passes",type=int,default=300);ap.add_argument("--cold-passes",type=int,default=32)
  ap.add_argument("--reps",type=int,default=9);ap.add_argument("--threshold-us",type=float,default=0.15)
  ap.add_argument("--out",type=Path,required=True);ap.add_argument("--artifact-dir",type=Path);ap.add_argument("--ncu",action="store_true")
  args=ap.parse_args();control,candidate=_render();source=HARNESS.replace("__CONTROL_SOURCE__",control).replace("__CANDIDATE_SOURCE__",candidate)
  with tempfile.TemporaryDirectory(prefix="nv_q4k_qdata_transpose_o_") as td:
    src,binary=Path(td)/"gate.cu",Path(td)/"gate";src.write_text(source)
    env={**os.environ,"PATH":f"{CUDA_BIN}:"+os.environ.get("PATH","")}
    build=subprocess.run(["nvcc","-arch=sm_120a","-O3","-std=c++17","--ptxas-options=-v",str(src),"-o",str(binary)],capture_output=True,text=True,env=env)
    if build.returncode:print(build.stderr[-12000:],file=sys.stderr);return 3
    artifact=args.artifact_dir
    if artifact:
      artifact.mkdir(parents=True,exist_ok=True);shutil.copy2(src,artifact/"gate.cu");shutil.copy2(binary,artifact/"gate");(artifact/"ptxas.txt").write_text(build.stderr)
    run=subprocess.run([str(binary),str(args.hot_passes),str(args.cold_passes),str(args.reps)],capture_output=True,text=True)
    print(run.stdout.strip())
    if run.returncode not in (0,5):print(run.stderr[-8000:],file=sys.stderr);return 4
    fixtures=[]
    for m in re.finditer(r"fixture=(\d+) finite=(\d+) mismatched_words=(\d+) max_abs=([0-9.eE+-]+)",run.stdout):
      fixtures.append({"fixture":int(m[1]),"finite":bool(int(m[2])),"mismatched_words":int(m[3]),"max_abs":float(m[4])})
    samples={k:[] for k in ("hot_control","hot_candidate","cold_control","cold_candidate")}
    pat=re.compile(r"hot_control_us=([0-9.]+) hot_candidate_us=([0-9.]+) cold_control_us=([0-9.]+) cold_candidate_us=([0-9.]+)")
    for m in pat.finditer(run.stdout):
      for k,v in zip(samples,m.groups()):samples[k].append(float(v))
    med={k:statistics.median(v) for k,v in samples.items()}
    hot_recovery=med["hot_control"]-med["hot_candidate"];cold_recovery=med["cold_control"]-med["cold_candidate"]
    exact=len(fixtures)==3 and all(x["finite"] and x["mismatched_words"]==0 for x in fixtures)
    result={"schema":"tinygrad.nv_q4k_qdata_transpose_o_microgate.v1","commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "method":f"production-rendered installed residual-fused FP16 O control versus exact qdata-only [4][8]->[8][4] repack; cudaEvent hot and 16-copy rotated-cold r{args.reps}",
      "shape":{"rows":ROWS,"k":K,"q4k_blocks":ROWS*K_BLOCKS,"weight_bytes":WEIGHT_BYTES,"rotated_weight_bytes_per_arm":ROTATIONS*WEIGHT_BYTES},
      "correctness":{"fixtures":fixtures,"bitwise_all":exact},"timing":{"unit":"us_per_call","hot_passes":args.hot_passes,"cold_passes":args.cold_passes,
      "reps":args.reps,"samples":samples,"medians":med,"hot_recovery_us":hot_recovery,"cold_recovery_us":cold_recovery,
      "control_cold_rate_tb_s":WEIGHT_BYTES/med["cold_control"]/1e6,"candidate_cold_rate_tb_s":WEIGHT_BYTES/med["cold_candidate"]/1e6},
      "storage_contract":{"metadata":"unchanged words 0..3","qdata_control":"[4 group-pairs][8 word-cols]","qdata_candidate":"[8 word-cols][4 group-pairs]","bytes_changed":0},
      "sass":{"control":_sass_census(binary,CONTROL),"candidate":_sass_census(binary,CANDIDATE)},"ptxas":build.stderr.strip().splitlines(),
      "threshold":{"cold_recovery_us_per_call":args.threshold_us},"verdict":"ADVANCE_TO_PRODUCTION_WALL" if exact and cold_recovery>=args.threshold_us else "STOP"}
    if args.ncu:
      if artifact is None:artifact=args.out.parent/"artifacts";artifact.mkdir(parents=True,exist_ok=True)
      result["ncu"]={"control":_ncu(binary,CONTROL,artifact/"ncu-control.csv.txt"),"candidate":_ncu(binary,CANDIDATE,artifact/"ncu-candidate.csv.txt")}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True));return 0 if exact else 5


if __name__=="__main__":raise SystemExit(main())
