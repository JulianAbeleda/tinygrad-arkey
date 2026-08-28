#!/usr/bin/env python3
"""Isolated llama-geometry Q8_1 producer for pp512 hidden-size 4096."""
from __future__ import annotations
import argparse, hashlib, json, pathlib, statistics
import numpy as np

M,K=512,4096
SRC=r'''
#include <cuda_fp16.h>
#include <cuda_runtime.h>
__device__ __forceinline__ signed char tg_round_i8(float v,float d){
 float y=v*(1.0f/d),a=y-0.5f,b=y+0.5f,ta=truncf(a),tb=truncf(b),th=truncf(y)*0.5f;
 float lo=ta<a?ta+1.0f:ta,hi=b<tb?tb-1.0f:tb;
 float r=((0.0f<y)!=(truncf(th)==th))?hi:lo;
 r=fminf(127.0f,fmaxf(-128.0f,r));return (signed char)r;
}
extern "C" __global__ __launch_bounds__(128,1) void q8_compact(const float* __restrict__ x, signed char* __restrict__ q,
 float* __restrict__ scales,float* __restrict__ sums) {
 int row=blockIdx.x, seg=blockIdx.y, t=threadIdx.x, i=seg*512+t*4, base=row*4096+i;
 float4 v=*reinterpret_cast<const float4*>(x+base);
 float a=fmaxf(fmaxf(fabsf(v.x),fabsf(v.y)),fmaxf(fabsf(v.z),fabsf(v.w)));
 float s=(v.x+v.y)+(v.z+v.w);
 #pragma unroll
 for(int off=4;off;off>>=1){a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,off));s+=__shfl_xor_sync(0xffffffff,s,off);}
 float d=a==0.0f?1.0f:a*0x1.020408p-7f;
 char4 z;z.x=tg_round_i8(v.x,d);z.y=tg_round_i8(v.y,d);z.z=tg_round_i8(v.z,d);z.w=tg_round_i8(v.w,d);
 *reinterpret_cast<char4*>(q+base)=z;
 if((t&7)==0){int g=row*128+seg*16+t/8;scales[g]=d;sums[g]=__half2float(__float2half_rn(s));}
}
'''

# Same packet ABI as SRC, but consumes the model's already-materialized fp16
# norm output directly. half->float conversion is exact and precedes the
# existing reduction/rounding sequence.
SRC_FP16=SRC.replace("void q8_compact(const float* __restrict__ x", "void q8_compact_fp16(const half* __restrict__ x").replace(
  "float4 v=*reinterpret_cast<const float4*>(x+base);",
  "half2 h0=*reinterpret_cast<const half2*>(x+base),h1=*reinterpret_cast<const half2*>(x+base+2); "
  "float4 v=make_float4(__half2float(__low2half(h0)),__half2float(__high2half(h0)),"
  "__half2float(__low2half(h1)),__half2float(__high2half(h1)));"
)

def sha(a):return hashlib.sha256(a.tobytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--rounds',type=int,default=9);ap.add_argument('--out',required=True);a=ap.parse_args()
 from tinygrad import Tensor,Device,dtypes
 from tinygrad.llm.generate import load_model_and_tokenizer
 from tinygrad.runtime.ops_nv import NVProgram
 from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
 from extra.llm_research.layout import q8_1_quantize
 model,_=load_model_and_tokenizer('/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf',512,seed=20260617)
 toks=Tensor([[(i*7)%1000 for i in range(M)]],dtype='int32').contiguous()
 x=model.blk[0].ffn_norm(model.blk[0].attn_norm(model.token_embd(toks).float())).reshape(M,K).cast('float32').contiguous().realize()
 q=Tensor.empty(M*K,dtype=dtypes.int8,device='NV').realize();s=Tensor.empty(M*K//32,dtype=dtypes.float32,device='NV').realize();u=Tensor.empty(M*K//32,dtype=dtypes.float32,device='NV').realize()
 dev=Device['NV'];lib=NVRTCCompiler(dev.arch,ptx=False,cache_key='nv_q8_compact_pp512_v1').compile(SRC);p=NVProgram(dev,'q8_compact',lib)
 bufs=tuple(t.uop.buffer.get_buf('NV') for t in (x,q,s,u));grid=(M,8,1);block=(128,1,1)
 for _ in range(3):p(*bufs,global_size=grid,local_size=block,wait=True)
 ts=[p(*bufs,global_size=grid,local_size=block,wait=True)*1e6 for _ in range(a.rounds)]
 qn,sn,un=q.numpy(),s.numpy(),u.numpy();rq,rs=q8_1_quantize(x);rqn,rsn=rq.numpy(),rs.numpy()
 bad=np.flatnonzero(qn!=rqn);xn=x.numpy().reshape(-1)
 examples=[{'index':int(i),'candidate':int(qn[i]),'reference':int(rqn[i]),'x':float(xn[i]),'scale':float(rsn[i//32]),'ratio':float(xn[i]/rsn[i//32])} for i in bad[:16]]
 run=x.numpy().reshape(M,K//32,32).sum(2,dtype=np.float32).astype(np.float16).astype(np.float32).reshape(-1)
 rec={'schema':'tinygrad.nv_q8_compact_producer_gate.v1','shape':{'M':M,'K':K},'grid':grid,'block':block,'ctas':M*8,'warps_per_cta':4,
  'timing':{'n':len(ts),'min_us':min(ts),'median_us':statistics.median(ts),'max_us':max(ts),'samples_us':ts},
  'correctness':{'finite':bool(np.isfinite(sn).all() and np.isfinite(un).all()),'values_exact':bool(np.array_equal(qn,rqn)),'values_mismatch':int(len(bad)),'mismatch_examples':examples,'scales_max_abs':float(np.max(np.abs(sn-rsn))),'scales_mismatch':int(np.count_nonzero(sn!=rsn)),'scales_exact':bool(np.array_equal(sn,rsn)),'sums_max_abs':float(np.max(np.abs(un-run))),'sums_exact':bool(np.array_equal(un,run)),'hashes':{'q':sha(qn),'scales':sha(sn),'sums':sha(un)}},
  'resources':{'registers':p.regs_usage,'shared_bytes':p.shmem_usage,'local_bytes':p.lcmem_usage},'source':'llama quantize_mmq_q8_1 topology: float4/thread, 128 threads, 8 K-segment CTAs per row'}
 pathlib.Path(a.out).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(a.out).write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec,indent=2))
if __name__=='__main__':main()
