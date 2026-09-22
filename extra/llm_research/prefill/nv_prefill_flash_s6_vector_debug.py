#!/usr/bin/env python3
"""Progressive ABI/memory/math probes for the S6 Flash CUDA substrate."""
import argparse, json, pathlib
import numpy as np

SRC=r'''
#include <cuda_fp16.h>
extern "C" __global__ void s0(const half*q,const half*k,const half*v,float*o){ }
extern "C" __global__ void s1(const half*q,const half*k,const half*v,float*o){ if(threadIdx.x==0)o[0]=1.0f; }
extern "C" __global__ void s2(const half*q,const half*k,const half*v,float*o){ if(threadIdx.x==0)o[0]=__half2float(q[0])+__half2float(k[0])+__half2float(v[0]); }
extern "C" __global__ void s3(const half*q,const half*k,const half*v,float*o){ int d=threadIdx.x; if(d<128)o[d]=__half2float(q[d])+__half2float(k[d])+__half2float(v[d]); }
extern "C" __global__ void s4(const half*q,const half*k,const half*v,float*o){ int d=threadIdx.x; if(d>=128)return; float a=0; for(int x=0;x<128;x++)a+=__half2float(q[x])*__half2float(k[x]); o[d]=a+__half2float(v[d]); }
extern "C" __global__ void s5(const half*q,const half*k,const half*v,float*o){ int d=threadIdx.x; if(d>=128)return; float mx=-1e30f,sm=0; for(int j=0;j<128;j++){float a=0;for(int x=0;x<128;x++)a+=__half2float(q[x])*__half2float(k[j*128+x]);mx=fmaxf(mx,a*.08838835f);} for(int j=0;j<128;j++){float a=0;for(int x=0;x<128;x++)a+=__half2float(q[x])*__half2float(k[j*128+x]);sm+=expf(a*.08838835f-mx);} o[d]=sm; }
extern "C" __global__ void s6(const half*q,const half*k,const half*v,float*o){ int d=threadIdx.x,h=blockIdx.x,t=blockIdx.y,part=blockIdx.z,ks=part*128; if(d>=128||h>=32||t>=512||part>=6)return; int qh=h,kh=h/4; float mx=-1e30f,sm=0; int upto=min(768,ks+128),causal=t; for(int j=ks;j<upto;j++){if(j>causal)continue;float s=0;for(int x=0;x<128;x++)s+=__half2float(q[(qh*512+t)*128+x])*__half2float(k[(kh*768+j)*128+x]);s*=.08838834764831843f;mx=fmaxf(mx,s);} for(int j=ks;j<upto;j++){if(j>causal)continue;float s=0;for(int x=0;x<128;x++)s+=__half2float(q[(qh*512+t)*128+x])*__half2float(k[(kh*768+j)*128+x]);sm+=expf(s*.08838834764831843f-mx);} int base=(((h*512+t)*6+part)*130);o[0]=mx;o[1]=sm;float z=0;for(int j=ks;j<upto;j++){if(j>causal)continue;float s=0;for(int x=0;x<128;x++)s+=__half2float(q[(qh*512+t)*128+x])*__half2float(k[(kh*768+j)*128+x]);z+=expf(s*.08838834764831843f-mx)*__half2float(v[(kh*768+j)*128+d]);}o[2+d]=z; }
extern "C" __global__ void s7(const half*q,const half*k,const half*v,float*o){ int d=threadIdx.x,h=blockIdx.x,t=blockIdx.y,part=blockIdx.z,ks=part*128; if(d>=128||h>=32||t>=512||part>=6)return; int qh=h,kh=h/4; float mx=-1e30f,sm=0; int upto=min(768,ks+128),causal=t; for(int j=ks;j<upto;j++){if(j>causal)continue;float s=0;for(int x=0;x<128;x++)s+=__half2float(q[(qh*512+t)*128+x])*__half2float(k[(kh*768+j)*128+x]);s*=.08838834764831843f;mx=fmaxf(mx,s);} for(int j=ks;j<upto;j++){if(j>causal)continue;float s=0;for(int x=0;x<128;x++)s+=__half2float(q[(qh*512+t)*128+x])*__half2float(k[(kh*768+j)*128+x]);sm+=expf(s*.08838834764831843f-mx);} int base=(((h*512+t)*6+part)*130);o[base]=mx;o[base+1]=sm;float z=0;for(int j=ks;j<upto;j++){if(j>causal)continue;float s=0;for(int x=0;x<128;x++)s+=__half2float(q[(qh*512+t)*128+x])*__half2float(k[(kh*768+j)*128+x]);z+=expf(s*.08838834764831843f-mx);}o[base+2+d]=z; }
extern "C" __global__ void s8(const half*q,const half*k,const half*v,float*o){ int d=threadIdx.x,h=blockIdx.x,t=blockIdx.y,part=blockIdx.z,ks=part*128; if(d>=128||h>=32||t>=512||part>=6)return; int qh=h,kh=h/4; float mx=-1e30f,sm=0; int upto=min(768,ks+128),causal=t; for(int j=ks;j<upto;j++){if(j>causal)continue;float s=0;for(int x=0;x<128;x++)s+=__half2float(q[(qh*512+t)*128+x])*__half2float(k[(kh*768+j)*128+x]);s*=.08838834764831843f;mx=fmaxf(mx,s);} for(int j=ks;j<upto;j++){if(j>causal)continue;float s=0;for(int x=0;x<128;x++)s+=__half2float(q[(qh*512+t)*128+x])*__half2float(k[(kh*768+j)*128+x]);sm+=expf(s*.08838834764831843f-mx);} int base=(((h*512+t)*6+part)*130);o[base]=mx;o[base+1]=sm;float z=0;for(int j=ks;j<upto;j++){if(j>causal)continue;float s=0;for(int x=0;x<128;x++)s+=__half2float(q[(qh*512+t)*128+x])*__half2float(k[(kh*768+j)*128+x]);z+=expf(s*.08838834764831843f-mx)*__half2float(v[(kh*768+j)*128+d]);}o[base+2+d]=z; }
'''
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
 from tinygrad import Tensor,Device,dtypes
 from tinygrad.runtime.ops_nv import NVProgram
 from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
 if Device.DEFAULT!='NV': raise RuntimeError('DEV=NV required')
 rng=np.random.default_rng(20260829); q=Tensor(rng.normal(0,.04,(32,512,128)).astype(np.float16),device='NV').contiguous().realize(); k=Tensor(rng.normal(0,.04,(8,768,128)).astype(np.float16),device='NV').contiguous().realize(); v=Tensor(rng.normal(0,.04,(8,768,128)).astype(np.float16),device='NV').contiguous().realize(); o=Tensor.empty(130,dtype=dtypes.float32,device='NV').realize(); dev=Device['NV'];lib=NVRTCCompiler(dev.arch,ptx=False,cache_key='nv_flash_s6_debug_v2').compile(SRC); bufs=tuple(x.uop.buffer.get_buf('NV') for x in (q,k,v,o)); rows=[]
 for name in ('s0','s1','s2','s3','s4','s5','s6','s7','s8'):
  p=NVProgram(dev,name,lib)
  try:
   us=p(*bufs,global_size=(1,1,1),local_size=(128,1,1),wait=True)*1e6; got=o.numpy().copy(); rows.append({'stage':name,'status':'PASS','us':us,'finite':bool(np.isfinite(got).all()),'sample':got[:4].tolist()})
  except Exception as e:
   rows.append({'stage':name,'status':'FAIL','error':str(e)});break
 rec={'schema':'tinygrad.nv_prefill_flash_s6_debug.v1','launch':{'global_size':[1,1,1],'local_size':[128,1,1]},'stages':rows,'status':rows[-1]['status']}; pathlib.Path(a.out).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(a.out).write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec,indent=2))
if __name__=='__main__':main()
