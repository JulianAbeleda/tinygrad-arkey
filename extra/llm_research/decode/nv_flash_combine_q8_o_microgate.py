#!/usr/bin/env python3
"""Research-only Flash-combine-owned Q8_1 O-projection primitive gate."""
from __future__ import annotations
import argparse,json,re,statistics,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.llm.flash_decode_attention import flash_fused_gmax_combine_kernel
from tinygrad.llm.shared_q8_attention import _emit_q8_provider,_emit_q4_cooperative
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops,UOp
H,HD,S,W,ROWS,K=32,128,6,130,4096,4096
O_WORDS=ROWS*(K//256)*36;Q8_WORDS=1024+128;ROTATIONS=16
NVCC="/usr/local/cuda-13.2/bin/nvcc"

def render(u):
  p=to_program(u,CUDARenderer(Target("NV",arch="sm_120"),use_nvcc=False))
  return p.arg.name,next(x.arg for x in p.src if x.op is Ops.SOURCE)

def fused_source(base:str)->tuple[str,str]:
  old="flash_fused_gmax_combine_f16_32_128_s6_lw128_regw";name=old+"_q8o"
  src=base[base.index('extern "C"'):].replace(old,name,1)
  src=src.replace("float* data1_24960) {","float* data1_24960, unsigned int* q8out) {",1)
  needle="  *(data0_4096+(lidx0+(gidx0<<7))) = ((half)(((*(buf1+0))*(1/(*(buf2+0))))));"
  if needle not in src: raise RuntimeError("combine output spelling changed")
  code=r'''  float q8v = (float)((half)((*(buf1+0))*(1/(*(buf2+0)))));
  *(data0_4096+(lidx0+(gidx0<<7))) = (half)q8v;
  int q8lane = lidx0 & 31;
  int q8group = (gidx0<<2) + (lidx0>>5);
  float q8max = fabsf(q8v);
  #pragma unroll
  for (int off=16; off; off>>=1) q8max=fmaxf(q8max,__shfl_down_sync(0xffffffffu,q8max,off));
  float q8d=q8max*(1.0f/127.0f);
  int q8q=q8d==0.0f?0:__float2int_rn(q8v/q8d); q8q=max(-128,min(127,q8q));
  int q81=__shfl_down_sync(0xffffffffu,q8q,1),q82=__shfl_down_sync(0xffffffffu,q8q,2),q83=__shfl_down_sync(0xffffffffu,q8q,3);
  if((q8lane&3)==0) q8out[q8group*8+(q8lane>>2)]=(unsigned char)q8q|((unsigned int)(unsigned char)q81<<8)|((unsigned int)(unsigned char)q82<<16)|((unsigned int)(unsigned char)q83<<24);
  float q8sum=q8v;
  #pragma unroll
  for (int off=16; off; off>>=1) q8sum+=__shfl_down_sync(0xffffffffu,q8sum,off);
  if(q8lane==0) q8out[1024+q8group]=(unsigned int)__half_as_ushort((half)q8d)|((unsigned int)__half_as_ushort((half)q8sum)<<16);'''
  return name,src.replace(needle,code,1)

def main()->int:
  ap=argparse.ArgumentParser();ap.add_argument("--hot-passes",type=int,default=500);ap.add_argument("--cold-passes",type=int,default=16);ap.add_argument("--reps",type=int,default=9);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args();p=UOp.placeholder
  cn,cs=render(flash_fused_gmax_combine_kernel(HD,H,S,output_fp16=True,lane_width=128,register_weights=True)(p((4096,),dtypes.float16,0),p((H*S*W,),dtypes.float32,1)))
  pn,ps=render(_emit_q8_provider()(p((Q8_WORDS,),dtypes.uint32,0),p((K,),dtypes.float16,1)))
  on,os=render(_emit_q4_cooperative(ROWS,UOp.const(dtypes.weakint,4),direct_output=True)(p((ROWS,),dtypes.float32,0),p((O_WORDS,),dtypes.uint32,1),p((Q8_WORDS,),dtypes.uint32,2)))
  fn,fs=fused_source(cs);bodies=cs+"\n"+fs+"\n"+ps[ps.index('extern "C"'):]+"\n"+os[os.index('extern "C"'):]
  h=f'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
{bodies}
static void ck(cudaError_t e,const char*w){{if(e!=cudaSuccess){{fprintf(stderr,"%s: %s\\n",w,cudaGetErrorString(e));exit(2);}}}}
static void chain(int arm,float*out,half*attn,float*partial,unsigned int*w,unsigned int*q8){{if(arm==0){{{cn}<<<32,128>>>(attn,partial);{pn}<<<128,1>>>(q8,attn);}}else {fn}<<<32,128>>>(attn,partial,q8);{on}<<<4096,128>>>(out,w,q8);}}
static double hot(int arm,float*out,half*attn,float*partial,unsigned int*w,unsigned int*q8,int n){{cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);cudaEventRecord(s);for(int i=0;i<n;i++)chain(arm,out,attn,partial,w,q8);cudaEventRecord(e);cudaEventSynchronize(e);float ms;cudaEventElapsedTime(&ms,s,e);return ms*1000.0/n;}}
static double cold(int arm,float*out,half*attn,float*partial,unsigned int*ws,unsigned int*q8,unsigned char*ev,int n){{cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);double us=0;for(int i=0;i<n;i++){{cudaMemset(ev,i,128ull<<20);cudaDeviceSynchronize();cudaEventRecord(s);chain(arm,out,attn,partial,ws+(i%{ROTATIONS})*{O_WORDS},q8);cudaEventRecord(e);cudaEventSynchronize(e);float ms;cudaEventElapsedTime(&ms,s,e);us+=ms*1000.0;}}return us/n;}}
int main(int ac,char**av){{int hp=atoi(av[1]),cp=atoi(av[2]),reps=atoi(av[3]);float *out[2],*partial;half*attn[2];unsigned int*ws,*q8[2];unsigned char*ev;for(int i=0;i<2;i++){{cudaMalloc(&out[i],4096*4);cudaMalloc(&attn[i],4096*2);cudaMalloc(&q8[i],{Q8_WORDS}*4);}}cudaMalloc(&partial,{H*S*W}*4);cudaMalloc(&ws,(size_t){ROTATIONS}*{O_WORDS}*4);cudaMalloc(&ev,128ull<<20);cudaMemset(partial,1,{H*S*W}*4);cudaMemset(ws,7,(size_t){ROTATIONS}*{O_WORDS}*4);chain(0,out[0],attn[0],partial,ws,q8[0]);chain(1,out[1],attn[1],partial,ws,q8[1]);cudaDeviceSynchronize();unsigned int*h0=(unsigned int*)malloc({Q8_WORDS}*4),*h1=(unsigned int*)malloc({Q8_WORDS}*4);cudaMemcpy(h0,q8[0],{Q8_WORDS}*4,cudaMemcpyDeviceToHost);cudaMemcpy(h1,q8[1],{Q8_WORDS}*4,cudaMemcpyDeviceToHost);printf("q8_bitwise=%d\\n",memcmp(h0,h1,{Q8_WORDS}*4)==0);float*o0=(float*)malloc(4096*4),*o1=(float*)malloc(4096*4);cudaMemcpy(o0,out[0],4096*4,cudaMemcpyDeviceToHost);cudaMemcpy(o1,out[1],4096*4,cudaMemcpyDeviceToHost);printf("output_bitwise=%d\\n",memcmp(o0,o1,4096*4)==0);for(int r=0;r<reps;r++)for(int i=0;i<2;i++){{int arm=(r&1)?1-i:i;printf("rep=%d arm=%d hot=%.6f cold=%.6f\\n",r,arm,hot(arm,out[arm],attn[arm],partial,ws,q8[arm],hp),cold(arm,out[arm],attn[arm],partial,ws,q8[arm],ev,cp));}}}}
'''
  with tempfile.TemporaryDirectory(prefix="nv_flash_q8_o_") as td:
    cu,bi=Path(td)/"g.cu",Path(td)/"g";cu.write_text(h);b=subprocess.run([NVCC,"-O3","-std=c++17","-arch=sm_120a","--ptxas-options=-v",str(cu),"-o",str(bi)],capture_output=True,text=True)
    if b.returncode:raise RuntimeError(b.stderr[-12000:])
    run=subprocess.run([str(bi),str(a.hot_passes),str(a.cold_passes),str(a.reps)],capture_output=True,text=True,check=True)
  print(run.stdout.strip());rows=[]
  for line in run.stdout.splitlines():
    if m:=re.match(r"rep=(\d+) arm=(\d+) hot=([0-9.]+) cold=([0-9.]+)",line):rows.append({"rep":int(m[1]),"arm":int(m[2]),"hot_us":float(m[3]),"cold_us":float(m[4])})
  med={};
  for arm,name in enumerate(("standalone_provider","combine_owned")):
    rr=[x for x in rows if x["arm"]==arm];med[name]={"hot_us":statistics.median(x["hot_us"] for x in rr),"cold_us":statistics.median(x["cold_us"] for x in rr)}
  out={"schema":"tinygrad.nv_flash_combine_q8_o_microgate.v1","q8_bitwise":"q8_bitwise=1" in run.stdout,"output_bitwise":"output_bitwise=1" in run.stdout,"median":med,"samples":rows,"ptxas":b.stderr};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");return 0 if out["q8_bitwise"] and out["output_bitwise"] else 5
if __name__=="__main__":raise SystemExit(main())
