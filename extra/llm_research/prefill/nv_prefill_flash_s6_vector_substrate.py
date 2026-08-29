#!/usr/bin/env python3
"""Standalone NVIDIA S6 vector Flash substrate (research-only, default-off).

The kernel deliberately exposes the six-way score/reduction partition ABI:
each CTA owns one (query head, query token, 128-key partition), and writes
``(m, l, acc[128])``.  A second kernel performs the six-part online combine.
This is an independent primitive/oracle gate, not a production route.
"""
import argparse, json, pathlib, statistics, os
import numpy as np

Hq,Hkv,Hd,T,KV,PARTS,PK = 32,8,128,512,768,6,128
SRC=r'''
#include <cuda_fp16.h>
__device__ __forceinline__ float dot_coop(const half *q,const half *k) {
  float z=0.0f; for(int x=threadIdx.x; x<128; x+=32) z += __half2float(q[x])*__half2float(k[x]);
  for(int off=16;off;off>>=1) z += __shfl_down_sync(0xffffffff,z,off);
  return __shfl_sync(0xffffffff,z,0);
}
extern "C" __global__ void s6coop(const half* q,const half* k,const half* v,float* p) {
  int d=threadIdx.x, h=blockIdx.x, t=blockIdx.y, part=blockIdx.z, ks=part*128;
  if (d>=128 || h>=32 || t>=512 || part>=6) return;
  int qh=h, kh=h/4; float mx=-1e30f, sm=0.0f; int upto=min(768,ks+128), causal=t;
  __shared__ float score;
  for(int j=ks;j<upto;j++) { if(j>causal) continue; if(threadIdx.x<32) { float s=dot_coop(&q[(qh*512+t)*128],&k[(kh*768+j)*128]); if(threadIdx.x==0) score=s*.08838834764831843f; } __syncthreads(); mx=fmaxf(mx,score); }
  for(int j=ks;j<upto;j++) { if(j>causal) continue; if(threadIdx.x<32) { float s=dot_coop(&q[(qh*512+t)*128],&k[(kh*768+j)*128]); if(threadIdx.x==0) score=s*.08838834764831843f; } __syncthreads(); sm+=expf(score-mx); }
  int base=(((h*512+t)*6+part)*130); p[base]=mx; p[base+1]=sm;
  float a=0; for(int j=ks;j<upto;j++) { if(j>causal) continue; if(threadIdx.x<32) { float s=dot_coop(&q[(qh*512+t)*128],&k[(kh*768+j)*128]); if(threadIdx.x==0) score=s*.08838834764831843f; } __syncthreads(); a+=expf(score-mx)*__half2float(v[(kh*768+j)*128+d]); }
  p[base+2+d]=a;
}
extern "C" __global__ void s6x(const half* q,const half* k,const half* v,float* p) {
  int d=threadIdx.x, h=blockIdx.x, t=blockIdx.y, part=blockIdx.z, ks=part*128;
  if (d>=128 || h>=32 || t>=512 || part>=6) return;
  int qh=h, kh=h/4; float mx=-1e30f, sm=0.0f;
  int upto=min(768,ks+128); int causal=t;
  for(int j=ks;j<upto;j++) { if(j>causal) continue; float s=0; for(int x=0;x<128;x++) s+=__half2float(q[(qh*512+t)*128+x])*__half2float(k[(kh*768+j)*128+x]); s*=0.08838834764831843f; mx=fmaxf(mx,s); }
  for(int j=ks;j<upto;j++) { if(j>causal) continue; float s=0; for(int x=0;x<128;x++) s+=__half2float(q[(qh*512+t)*128+x])*__half2float(k[(kh*768+j)*128+x]); sm+=expf(s*0.08838834764831843f-mx); }
  int base=(((h*512+t)*6+part)*130); p[base]=mx; p[base+1]=sm;
  float a=0; for(int j=ks;j<upto;j++) { if(j>causal) continue; float s=0; for(int x=0;x<128;x++) s+=__half2float(q[(qh*512+t)*128+x])*__half2float(k[(kh*768+j)*128+x]); a+=expf(s*0.08838834764831843f-mx)*__half2float(v[(kh*768+j)*128+d]); }
  p[base+2+d]=a;
}
extern "C" __global__ void s6head(const half* q,const half* k,const half* v,float* p) {
 int d=threadIdx.x, h=blockIdx.x, part=blockIdx.y, ks=part*128;
 if(d>=128 || h>=32 || part>=6) return;
 int kh=h/4, upto=min(768,ks+128);
 for(int t=0;t<512;t++) {
   float mx=-1e30f, sm=0.0f;
   for(int j=ks;j<upto && j<=t;j++) { float s=0; for(int x=0;x<128;x++) s+=__half2float(q[(h*512+t)*128+x])*__half2float(k[(kh*768+j)*128+x]); s*=0.08838834764831843f; mx=fmaxf(mx,s); }
   for(int j=ks;j<upto && j<=t;j++) { float s=0; for(int x=0;x<128;x++) s+=__half2float(q[(h*512+t)*128+x])*__half2float(k[(kh*768+j)*128+x]); sm+=expf(s*0.08838834764831843f-mx); }
   int base=(((h*512+t)*6+part)*130); p[base]=mx; p[base+1]=sm;
   float aa=0; for(int j=ks;j<upto && j<=t;j++) { float s=0; for(int x=0;x<128;x++) s+=__half2float(q[(h*512+t)*128+x])*__half2float(k[(kh*768+j)*128+x]); aa+=expf(s*0.08838834764831843f-mx)*__half2float(v[(kh*768+j)*128+d]); }
   p[base+2+d]=aa;
 }
}
extern "C" __global__ void combine(const float* p,half* o) {
 int d=threadIdx.x,h=blockIdx.x,t=blockIdx.y; if(d>=128)return; float gm=-1e30f,den=0,num=0;
 for(int z=0;z<6;z++){int b=(((h*512+t)*6+z)*130);gm=fmaxf(gm,p[b]);}
 for(int z=0;z<6;z++){int b=(((h*512+t)*6+z)*130);float w=expf(p[b+0]-gm);den+=w*p[b+1];num+=w*p[b+2+d];}
 o[(h*512+t)*128+d]=__float2half_rn(num/den);
}
'''

def oracle(q,k,v):
 q=q.astype(np.float32); k=np.repeat(k.astype(np.float32),4,axis=1); v=np.repeat(v.astype(np.float32),4,axis=1)
 s=q@k.transpose(0,1,3,2)/np.sqrt(128); mask=np.arange(768)[None,:] <= np.arange(512)[:,None]; s=np.where(mask[None,None],s,-np.inf); s-=np.max(s,-1,keepdims=True); w=np.exp(s); w/=w.sum(-1,keepdims=True); return (w@v).astype(np.float16)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--heads',type=int,default=32); ap.add_argument('--queries',type=int,default=512); ap.add_argument('--parts',type=int,default=6); ap.add_argument('--variant',choices=('naive','coop','head'),default='naive'); a=ap.parse_args()
 from tinygrad import Tensor,Device,dtypes
 from tinygrad.runtime.ops_nv import NVProgram
 from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
 if Device.DEFAULT!='NV': raise RuntimeError('DEV=NV required')
 rng=np.random.default_rng(20260829); q=rng.normal(0,.04,(Hq,T,Hd)).astype(np.float16); k=rng.normal(0,.04,(Hkv,KV,Hd)).astype(np.float16); v=rng.normal(0,.04,(Hkv,KV,Hd)).astype(np.float16)
 tq,tk,tv=[Tensor(x,device='NV').contiguous().realize() for x in (q,k,v)]; tp=Tensor.empty(Hq*T*PARTS*130,dtype=dtypes.float32,device='NV').realize(); to=Tensor.empty(Hq*T*Hd,dtype=dtypes.float16,device='NV').realize(); dev=Device['NV']; score_src=SRC[:SRC.index('extern "C" __global__ void combine')]; score_lib=NVRTCCompiler(dev.arch,ptx=False,cache_key='nv_flash_s6_score_v4').compile(score_src); combine_lib=NVRTCCompiler(dev.arch,ptx=False,cache_key='nv_flash_s6_combine_v4').compile('#include <cuda_fp16.h>\n'+SRC[SRC.index('extern "C" __global__ void combine'):]); ps=NVProgram(dev,'s6x',score_lib); pc=NVProgram(dev,'combine',combine_lib); bufs=tuple(x.uop.buffer.get_buf('NV') for x in (tq,tk,tv,tp)); q0,k0,v0=q.copy(),k.copy(),v.copy()
 grid=(a.heads,a.parts,1) if a.variant == 'head' else (a.heads,a.queries,a.parts); score_name = {'coop':'s6coop','head':'s6head'}.get(a.variant,'s6x'); ps=NVProgram(dev,score_name,score_lib); tscore=ps(*bufs,global_size=grid,local_size=(128,1,1),wait=True)*1e6
 if a.parts==6: tcomb=pc(tp.uop.buffer.get_buf('NV'),to.uop.buffer.get_buf('NV'),global_size=(a.heads,a.queries,1),local_size=(128,1,1),wait=True)*1e6
 else: tcomb=None
 Device['NV'].synchronize(); got=to.numpy().reshape(Hq,T,Hd); q1,k1,v1=tq.numpy(),tk.numpy(),tv.numpy(); ref=oracle(q[None],k[None],v[None])[0]; sub=got[:a.heads,:a.queries]; rsub=ref[:a.heads,:a.queries]; rec={'schema':'tinygrad.nv_prefill_flash_s6_vector_substrate.v1','variant':a.variant,'status':'PASS' if tcomb is not None and np.allclose(sub,rsub,atol=2e-2,rtol=2e-2) else ('PRIMITIVE_ONLY' if tcomb is None else 'FAIL'),'shape':{'Hq':Hq,'Hkv':Hkv,'Hd':Hd,'q_tokens':T,'kv_extent':KV,'partitions':a.parts,'partition_tokens':PK,'launched_heads':a.heads,'launched_queries':a.queries},'timing_us':{'score':tscore,'combine':tcomb},'finite':bool(np.isfinite(sub).all()),'readonly_inputs':bool(np.array_equal(q1,q) and np.array_equal(k1,k) and np.array_equal(v1,v)),'max_abs':float(np.max(np.abs(sub.astype(np.float32)-rsub.astype(np.float32)))),'allclose':bool(np.allclose(sub,rsub,atol=2e-2,rtol=2e-2)),'resources':{'score_regs':ps.regs_usage,'score_shared':ps.shmem_usage,'combine_regs':pc.regs_usage,'combine_shared':pc.shmem_usage}}
 pathlib.Path(a.out).parent.mkdir(parents=True,exist_ok=True); pathlib.Path(a.out).write_text(json.dumps(rec,indent=2)+'\n'); print(json.dumps(rec,indent=2))
if __name__=='__main__': main()
