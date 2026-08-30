#!/usr/bin/env python3
"""Full-grid last-CTA Flash readiness to native persistent exact O gate."""
from __future__ import annotations
import argparse,json,re,statistics,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tinygrad import dtypes
from tinygrad.uop.ops import UOp
from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_kernel,Q4KGEMVEpilogue
from extra.llm_research.decode.nv_flash_last_cta_combine_gate import source as flash_source,NVCC
from extra.llm_research.decode.nv_segmented_flash_o_complete_span import _render,_preambles,_o_half_emitter,ROWS,O_WORDS,K

def build(workers):
  src,sn,cn,fn=flash_source()
  src=src.replace('unsigned int* counters) {','unsigned int* counters, unsigned int* groups, unsigned int* ready) {',1)
  src=src.replace('if (tg_lane == 0) counters[gidx1]=0u;',
    'if (tg_lane == 0) { __threadfence(); unsigned int gg=gidx1>>4; if (atomicAdd(groups+gg,1u)==15u) { groups[gg]=0u; __threadfence(); ready[gg]=1u; } counters[gidx1]=0u; }',1)
  p=UOp.placeholder
  hn,hs=_render(_o_half_emitter(first_half=True,persistent_workers=workers,low_first=True)(
    p((ROWS*16,),dtypes.float32,0),p((O_WORDS,),dtypes.uint32,1),p((K,),dtypes.float16,2)))
  ln,ls=_render(_o_half_emitter(first_half=False,persistent_workers=workers,low_first=True)(
    p((ROWS,),dtypes.float32,0),p((O_WORDS,),dtypes.uint32,1),p((K,),dtypes.float16,2),p((ROWS*16,),dtypes.float32,3),
    p((ROWS,),dtypes.float32,4)))
  def add_wait(s,name,group):
    b=s[s.index('extern "C"'):]
    b=b.replace(') {',', unsigned int* ready) {',1)
    # The producer fences before publication.  Poll with a volatile load so
    # hundreds of O CTAs do not serialize on a read-only global atomic.  A
    # consumer-side global fence per thread is unnecessary and very expensive.
    return b.replace('{',f'{{ while (((volatile unsigned int*)ready)[{group}]==0u) __nanosleep(64);',1)
  hs,ls=add_wait(hs,hn,0),add_wait(ls,ln,1)
  on,os=_render(q4k_g3_lanemap_gemv_kernel(ROWS,K,epilogue=Q4KGEMVEpilogue('residual_add'),load_style='vector')(
    p((ROWS,),dtypes.float32,0),p((O_WORDS,),dtypes.uint32,1),p((K,),dtypes.float16,2),p((ROWS,),dtypes.float32,3)))
  def extract(s,name):
    for chunk in ('extern "C"'+x for x in s.split('extern "C"')[1:]):
      if name in chunk.split('{',1)[0]: return chunk
    raise RuntimeError(f'missing {name}')
  combined=_preambles(src,os,hs,ls)+extract(src,sn)+extract(src,cn)+extract(src,fn)+os[os.index('extern "C"'):]+hs+ls
  return combined,(sn,cn,fn,hn,ln)

HARNESS=r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
__SRC__
#define ROT 16
static void ck(cudaError_t e,const char*w){if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e));exit(2);}}
struct C{cudaStream_t f,o;cudaEvent_t st,done;};
static void init(C&c){int least,great;ck(cudaDeviceGetStreamPriorityRange(&least,&great),"pr");ck(cudaStreamCreateWithPriority(&c.f,cudaStreamNonBlocking,great),"f");ck(cudaStreamCreateWithPriority(&c.o,cudaStreamNonBlocking,least),"o");ck(cudaEventCreate(&c.st),"st");ck(cudaEventCreate(&c.done),"done");}
static float ctrl(C&c,float*out,half*att,float*part,float*q,unsigned*cache,unsigned*w,float*res){ck(cudaEventRecord(c.st,c.f),"st");__S__<<<dim3(6,32),dim3(32,4),0,c.f>>>(part,q,cache);__C__<<<32,128,0,c.f>>>(att,part);q4k_g3_lanemap_gemv_vec_epi_resadd_4096_4096<<<4096,32,0,c.f>>>(out,w,att,res);ck(cudaEventRecord(c.done,c.f),"dn");ck(cudaEventSynchronize(c.done),"sy");float ms;ck(cudaEventElapsedTime(&ms,c.st,c.done),"el");return ms*1000;}
static float cand(C&c,float*out,half*att,float*part,float*q,unsigned*cache,unsigned*w,float*scratch,float*res,unsigned*ctr,unsigned*groups,unsigned*ready){ck(cudaEventRecord(c.st,c.f),"st");ck(cudaMemsetAsync(ready,0,8,c.f),"rz");ck(cudaStreamWaitEvent(c.o,c.st),"ow");__F__<<<dim3(6,32),dim3(32,4),0,c.f>>>(part,q,cache,att,ctr,groups,ready);__H__<<<__WORKERS__,32,0,c.o>>>(scratch,w,att,ready);__L__<<<__WORKERS__,32,0,c.o>>>(out,w,att,scratch,res,ready);ck(cudaEventRecord(c.done,c.o),"dn");ck(cudaEventSynchronize(c.done),"sy");float ms;ck(cudaEventElapsedTime(&ms,c.st,c.done),"el");return ms*1000;}
static float octrl(C&c,float*out,half*att,unsigned*w,float*res){ck(cudaEventRecord(c.st,c.f),"ost");q4k_g3_lanemap_gemv_vec_epi_resadd_4096_4096<<<4096,32,0,c.f>>>(out,w,att,res);ck(cudaEventRecord(c.done,c.f),"odn");ck(cudaEventSynchronize(c.done),"osy");float ms;ck(cudaEventElapsedTime(&ms,c.st,c.done),"oel");return ms*1000;}
static float osplit(C&c,float*out,half*att,unsigned*w,float*scratch,float*res,unsigned*ready){ck(cudaMemsetAsync(ready,1,8,c.f),"rone");ck(cudaEventRecord(c.st,c.f),"sst");__H__<<<__WORKERS__,32,0,c.f>>>(scratch,w,att,ready);__L__<<<__WORKERS__,32,0,c.f>>>(out,w,att,scratch,res,ready);ck(cudaEventRecord(c.done,c.f),"sdn");ck(cudaEventSynchronize(c.done),"ssy");float ms;ck(cudaEventElapsedTime(&ms,c.st,c.done),"sel");return ms*1000;}
int main(int ac,char**av){int hp=atoi(av[1]),cp=atoi(av[2]),reps=atoi(av[3]);C c;init(c);float *a,*b,*pa,*pb,*q,*res,*scratch;half *aa,*bb;unsigned *wa,*wb,*ka,*kb,*ctr,*groups,*ready;cudaMalloc(&a,16384);cudaMalloc(&b,16384);cudaMalloc(&pa,99840);cudaMalloc(&pb,99840);cudaMalloc(&aa,8192);cudaMalloc(&bb,8192);cudaMalloc(&q,16384);cudaMalloc(&res,16384);cudaMalloc(&scratch,262144);cudaMalloc(&wa,(size_t)ROT*O_WORDS*4);cudaMalloc(&wb,(size_t)ROT*O_WORDS*4);cudaMalloc(&ka,(size_t)ROT*1048576*4);cudaMalloc(&kb,(size_t)ROT*1048576*4);cudaMalloc(&ctr,128);cudaMalloc(&groups,8);cudaMalloc(&ready,8);cudaMemset(ctr,0,128);cudaMemset(groups,0,8);cudaMemset(wa,7,(size_t)ROT*O_WORDS*4);cudaMemset(wb,7,(size_t)ROT*O_WORDS*4);cudaMemset(ka,0x30,(size_t)ROT*1048576*4);cudaMemset(kb,0x30,(size_t)ROT*1048576*4);std::vector<float>x(4096),r(4096);for(int i=0;i<4096;i++){x[i]=float((i*17%127)-63)/256;r[i]=float((i%31)-15)/128;}cudaMemcpy(q,x.data(),16384,cudaMemcpyHostToDevice);cudaMemcpy(res,r.data(),16384,cudaMemcpyHostToDevice);ctrl(c,a,aa,pa,q,ka,wa,res);cand(c,b,bb,pb,q,kb,wb,scratch,res,ctr,groups,ready);std::vector<unsigned char>ha(16384),hb(16384);cudaMemcpy(ha.data(),a,16384,cudaMemcpyDeviceToHost);cudaMemcpy(hb.data(),b,16384,cudaMemcpyDeviceToHost);printf("bitwise=%d\n",memcmp(ha.data(),hb.data(),16384)==0);double oh=0,sh=0,oc=0,sc=0;for(int i=0;i<hp;i++){oh+=octrl(c,a,aa,wa,res);sh+=osplit(c,b,aa,wb,scratch,res,ready);}for(int i=0;i<cp;i++){int z=i%ROT;oc+=octrl(c,a,aa,wa+(size_t)z*O_WORDS,res);sc+=osplit(c,b,aa,wb+(size_t)z*O_WORDS,scratch,res,ready);}printf("o_control_hot=%.6f o_split_hot=%.6f o_control_cold=%.6f o_split_cold=%.6f\n",oh/hp,sh/hp,oc/cp,sc/cp);for(int r0=0;r0<reps;r0++){double ah=0,bh=0,ac=0,bc=0;for(int i=0;i<hp;i++){ah+=ctrl(c,a,aa,pa,q,ka,wa,res);bh+=cand(c,b,bb,pb,q,kb,wb,scratch,res,ctr,groups,ready);}for(int i=0;i<cp;i++){int z=(r0*cp+i)%ROT;ac+=ctrl(c,a,aa,pa,q,ka+(size_t)z*1048576,wa+(size_t)z*O_WORDS,res);bc+=cand(c,b,bb,pb,q,kb+(size_t)z*1048576,wb+(size_t)z*O_WORDS,scratch,res,ctr,groups,ready);}printf("rep=%d control_hot=%.6f candidate_hot=%.6f control_cold=%.6f candidate_cold=%.6f\n",r0,ah/hp,bh/hp,ac/cp,bc/cp);}return 0;}
'''
def main():
  ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--hot',type=int,default=16);ap.add_argument('--cold',type=int,default=4);ap.add_argument('--reps',type=int,default=9);a=ap.parse_args();src,n=build(a.workers);cu=HARNESS.replace('__SRC__',src).replace('__S__',n[0]).replace('__C__',n[1]).replace('__F__',n[2]).replace('__H__',n[3]).replace('__L__',n[4]).replace('__WORKERS__',str(a.workers)).replace('O_WORDS',str(O_WORDS))
  with tempfile.TemporaryDirectory(prefix='fullgrid_o_') as td:
    p=Path(td);(p/'g.cu').write_text(cu);b=subprocess.run([NVCC,'-O3','-arch=sm_120a','--ptxas-options=-v',str(p/'g.cu'),'-o',str(p/'g')],capture_output=True,text=True)
    if b.returncode:raise RuntimeError(b.stderr[-12000:])
    r=subprocess.run([str(p/'g'),str(a.hot),str(a.cold),str(a.reps)],capture_output=True,text=True,timeout=120)
  print(r.stdout);rows=[]
  for m in re.finditer(r'rep=(\d+) control_hot=([0-9.]+) candidate_hot=([0-9.]+) control_cold=([0-9.]+) candidate_cold=([0-9.]+)',r.stdout):rows.append([float(x) for x in m.groups()[1:]])
  med=[statistics.median(x[i] for x in rows) for i in range(4)];o={'schema':'tinygrad.nv_flash_o_fullgrid_persistent.v1','workers':a.workers,'bitwise':'bitwise=1' in r.stdout,'medians':dict(zip(('control_hot','candidate_hot','control_cold','candidate_cold'),med)),'cold_recovery_us':med[2]-med[3],'samples':rows,'ptxas':b.stderr};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2));return 0 if o['bitwise'] else 5
if __name__=='__main__':raise SystemExit(main())
