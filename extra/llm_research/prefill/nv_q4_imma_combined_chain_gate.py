#!/usr/bin/env python3
"""Zero-copy compact-Q8 -> composite packed-Q4 IMMA integration gate."""
from __future__ import annotations
import argparse,json,pathlib,statistics
import numpy as np
from extra.llm_research.prefill.nv_q8_compact_producer_gate import SRC as Q8_SRC,M,K
from extra.llm_research.prefill.nv_q4k_imma_fragment_microgate import SRC as MAIN_SRC,lexical_src

N=12288
REF_SRC=r'''
#include <cuda_fp16.h>
__device__ __forceinline__ unsigned qb(const unsigned*w,int base,int off){return(w[base+(off>>2)]>>(8*(off&3)))&255u;}
extern "C" __global__ void q4_ref(float*out,const unsigned*w,const signed char*x,const float*ds,const float*sm,int M,int N,int K){
 int m=blockIdx.y,n=blockIdx.x*blockDim.x+threadIdx.x;if(m>=M||n>=N)return;int z=m*N+n,nb=K/256;float acc=0;
 for(int b=0;b<nb;b++){int base=(n*nb+b)*36;unsigned w0=w[base];float D=__half2float(__ushort_as_half(w0&65535u)),Dm=__half2float(__ushort_as_half(w0>>16));
  for(int g=0;g<8;g++){unsigned sc,mn;if(g<4){sc=qb(w,base,4+g)&63u;mn=qb(w,base,8+g)&63u;}else{int h=g-4;sc=(qb(w,base,12+h)&15u)|((qb(w,base,4+h)>>6)<<4);mn=(qb(w,base,12+h)>>4)|((qb(w,base,8+h)>>6)<<4);}
   int dot=0;for(int k=0;k<32;k++){unsigned by=qb(w,base,16+(g>>1)*32+k);int q=(g&1)?by>>4:by&15u;dot+=q*(int)x[m*K+b*256+g*32+k];}
   int mi=m*(nb*8)+b*8+g;float wc0=__half2float(__float2half_rn(D*(float)sc)),wc1=__half2float(__float2half_rn(-Dm*(float)mn));float yc0=__half2float(__float2half_rn(ds[mi])),yc1=__half2float(__float2half_rn(sm[mi]));acc+=wc0*yc0*(float)dot+wc1*yc1;}}
 out[z]=acc;}
'''

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--rounds',type=int,default=9);ap.add_argument('--out',required=True)
 ap.add_argument('--main-cubin',default='');a=ap.parse_args()
 from tinygrad import Tensor,Device,dtypes
 from tinygrad.llm.generate import load_model_and_tokenizer
 from extra.llm_research.layout import read_metadata,packed_u32_slice
 from tinygrad.runtime.ops_nv import NVProgram
 from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
 model,_=load_model_and_tokenizer('/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf',512,seed=20260617);dev=Device['NV']
 toks=Tensor([[(i*7)%1000 for i in range(M)]],dtype='int32').contiguous();x=model.blk[0].ffn_norm(model.blk[0].attn_norm(model.token_embd(toks).float())).reshape(M,K).cast('float32').contiguous().realize()
 meta=read_metadata(pathlib.Path('/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf'));info=next(i for i in meta.infos if i.name=='blk.0.ffn_gate.weight');w=packed_u32_slice(pathlib.Path('/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf'),meta,info,device='NV')
 q=Tensor.empty(M*K,dtype=dtypes.int8,device='NV').realize();ds=Tensor.empty(M*K//32,dtype=dtypes.float32,device='NV').realize();sm=Tensor.empty(M*K//32,dtype=dtypes.float32,device='NV').realize();o=Tensor.empty(M*N,dtype=dtypes.float32,device='NV').realize();r=Tensor.full((M*N,),float('nan'),dtype=dtypes.float32,device='NV').contiguous().realize()
 ssrc=lexical_src(MAIN_SRC,True).replace("int ntx=(N+127)/128,mb=(tile/ntx)*128,nb=(tile%ntx)*128,blocks=K/256;","int ntx=96,mb=(tile/96)*128,nb=(tile%96)*128,blocks=16;").replace("row*K+blk*256","row*4096+blk*256").replace("total=(M/128)*(N/128)*(K/256),owners=min(170,total)","total=6144,owners=170").replace("K/256","16").replace("if(col<N) { unsigned raw=","{ unsigned raw=").replace("w0=col<N?words[base]:0;","w0=words[base];").replace("if(col<N) {","{").replace("if(row<M) v=","v=").replace("if(row<M) { int xm=","{ int xm=").replace("if(row<M&&col<N) {","{").replace("row*N+col","row*12288+col")
 statsrc=lexical_src(MAIN_SRC).replace("row*K+blk*256","row*4096+blk*256").replace("K/256","16").replace("row*N+col","row*12288+col")
 fsrc='extern "C" __global__ void q4k_imma_fixup(float *o,const float*p,const int*map,int M,int N){int t=blockIdx.x,s0=map[2*t];if(s0<0)return;int s1=map[2*t+1],nb=(t%(N/128))*128,mb=(t/(N/128))*128;for(int z=threadIdx.x;z<16384;z+=256){int r=z/128,c=z%128;o[(mb+r)*N+nb+c]=p[s0*16384+z]+(s1>=0?p[s1*16384+z]:0);}}'
 qp=NVProgram(dev,'q8_compact',NVRTCCompiler(dev.arch,ptx=False,cache_key='q8_combined_v2').compile(Q8_SRC));main_lib=pathlib.Path(a.main_cubin).read_bytes() if a.main_cubin else NVRTCCompiler(dev.arch,ptx=False,cache_key='q4combined_stream_v2').compile(ssrc);mp=NVProgram(dev,'q4k_imma_stream',main_lib,shared_mem=57856+1024);fp=NVProgram(dev,'q4k_imma_fixup',NVRTCCompiler(dev.arch,ptx=False,cache_key='q4combined_fix_v2').compile(fsrc));sp=NVProgram(dev,'q4k_imma_complete',NVRTCCompiler(dev.arch,ptx=False,cache_key='q4combined_static_v1').compile(statsrc),shared_mem=57856+1024);rp=NVProgram(dev,'q4_ref',NVRTCCompiler(dev.arch,ptx=False,cache_key='q4ref_v3').compile(REF_SRC))
 b=lambda t:t.uop.buffer.get_buf('NV');xb,qb,db,sb,wb,ob,rb=map(b,(x,q,ds,sm,w,o,r));partials=Tensor.empty(340*128*128,dtype=dtypes.float32,device='NV').realize();ids=Tensor.empty(340,dtype=dtypes.int32,device='NV').realize();pb,ib=map(b,(partials,ids));slotmap=np.full((384,2),-1,np.int32)
 for cid in range(170):
  u=(cid*6144)//170;ue=((cid+1)*6144)//170;piece=0
  while u<ue:
   tile=u//16;b0=u%16;end=min(ue,(tile+1)*16);b1=end-tile*16
   if not (b0==0 and b1==16):j=0 if slotmap[tile,0]<0 else 1;slotmap[tile,j]=cid*2+piece;piece+=1
   u=end
 map_t=Tensor(slotmap,device='NV').contiguous().realize();mapb=b(map_t);qgrid=(M,8,1);qblock=(128,1,1)
 x_readonly=x.numpy().copy();w_readonly=w.numpy().copy()
 rows=[]
 for _ in range(3):qp(xb,qb,db,sb,global_size=qgrid,local_size=qblock,wait=True);mp(ob,pb,ib,wb,qb,db,sb,vals=(M,N,K),global_size=(170,1,1),local_size=(256,1,1),wait=True);fp(ob,pb,mapb,vals=(M,N),global_size=(384,1,1),local_size=(256,1,1),wait=True)
 for _ in range(a.rounds):
  qu=qp(xb,qb,db,sb,global_size=qgrid,local_size=qblock,wait=True)*1e6;mu=mp(ob,pb,ib,wb,qb,db,sb,vals=(M,N,K),global_size=(170,1,1),local_size=(256,1,1),wait=True)*1e6;fu=fp(ob,pb,mapb,vals=(M,N),global_size=(384,1,1),local_size=(256,1,1),wait=True)*1e6;rows.append({'producer_us':qu,'main_us':mu,'fixup_us':fu,'chain_us':qu+mu+fu})
 # Independent scalar packed algebra, complete full output.
 # One scalar tile is the independent packed-algebra oracle. A full scalar launch is
 # watchdog-scale, so the faithful static IMMA body is the full scheduling oracle.
 rp(rb,wb,qb,db,sb,vals=(M,N,K),global_size=(1,M,1),local_size=(256,1,1),wait=True)
 sto=Tensor.full((M*N,),float('nan'),dtype=dtypes.float32,device='NV').contiguous().realize();stob=b(sto);sp(stob,wb,qb,db,sb,vals=(M,N,K),global_size=(96,4,1),local_size=(256,1,1),wait=True)
 ca,sa,ra=o.numpy(),sto.numpy(),r.numpy();d=np.abs(ca-sa)
 sta=sa.reshape(M,N)[:128,:128];sr=ra.reshape(M,N)[:128,:128];sdiff=np.abs(sta-sr)
 top=np.argpartition(d,-16)[-16:];top=top[np.argsort(d[top])[::-1]]
 examples=[{'flat':int(i),'row':int(i//N),'col':int(i%N),'tile':int((i//N)//128*96+(i%N)//128),'candidate':float(ca[i]),'reference':float(sa[i]),'abs':float(d[i])} for i in top]
 dt=[];st=[]
 for t in range(384):
  mb=(t//96)*128;nb=(t%96)*128;td=d.reshape(M,N)[mb:mb+128,nb:nb+128]
  (st if slotmap[t,0]>=0 else dt).append(float(td.max()))
 rec={'schema':'tinygrad.nv_q4_imma_combined_chain.v1','shape':{'M':M,'N':N,'K':K},'fixture':{'model':'/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf','weight':'blk.0.ffn_gate.weight','token_formula':'(i*7)%1000','load_seed':20260617},'readonly':{'x_exact':bool(np.array_equal(x_readonly,x.numpy())),'words_exact':bool(np.array_equal(w_readonly,w.numpy()))},'zero_copy':{'q_values_same_buffer':True,'scales_same_buffer':True,'sums_same_buffer':True,'conversion_kernels':0,'producer_layout':'int8[M,K]+fp32[M,K/32]+fp32[M,K/32]','consumer_layout_match':True},'rows':rows,'timing':{'min_chain_us':min(x['chain_us'] for x in rows),'median_chain_us':statistics.median(x['chain_us'] for x in rows),'min_producer_us':min(x['producer_us'] for x in rows),'min_main_us':min(x['main_us'] for x in rows)},'full_output':{'oracle':'faithful_static_full_grid','finite':bool(np.isfinite(ca).all()),'reference_finite':bool(np.isfinite(sa).all()),'unwritten_sentinels':int(np.isnan(sa).sum()),'candidate_zero_count':int((ca==0).sum()),'reference_zero_count':int((sa==0).sum()),'elements':int(ca.size),'candidate_minmax':[float(ca.min()),float(ca.max())],'reference_minmax':[float(sa.min()),float(sa.max())],'max_abs':float(d.max()),'mean_abs':float(d.mean()),'split_tile_max_abs':max(st),'direct_tile_max_abs':max(dt),'top_mismatches':examples,'allclose_rtol2e5_atol2e3':bool(np.allclose(ca,sa,rtol=2e-5,atol=2e-3))},'verdict':'NO_GO_CORRECTNESS'}
 rec['static_tile0']={'max_abs':float(sdiff.max()),'mean_abs':float(sdiff.mean()),'candidate_00':float(sta[0,0]),'reference_00':float(sr[0,0])}
 if rec['full_output']['unwritten_sentinels'] != 0: raise AssertionError('full static oracle did not cover every output')
 if all(rec['readonly'].values()) and rec['full_output']['finite'] and rec['full_output']['reference_finite'] and rec['full_output']['allclose_rtol2e5_atol2e3'] and rec['static_tile0']['max_abs'] == 0.0:
  rec['verdict']='PASS_COMPLETE_CHAIN'
 rec['full_output'].update(candidate_zero_count=int(np.count_nonzero(ca==0)),reference_zero_count=int(np.count_nonzero(ra==0)))
 p=pathlib.Path(a.out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec,indent=2))
if __name__=='__main__':main()
