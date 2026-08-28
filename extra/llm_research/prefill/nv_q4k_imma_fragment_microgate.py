#!/usr/bin/env python3
"""Composite research microgate for a packed Q4_K m16n8k32 fragment."""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np
from tinygrad import Device
from tinygrad.device import BufferSpec
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler

SRC=r'''
extern "C" __global__ void q4k_imma_fragment(int *out, const unsigned int *words, const signed char *x, int group) {
  int lane=threadIdx.x, lr=lane>>2, lc=lane&3;
  int ar[4], br[2], cr[4]={0,0,0,0};
  #pragma unroll
  for (int r=0;r<4;r++) {
    unsigned v=0;
    #pragma unroll
    for (int b=0;b<4;b++) {
      int row=lr+8*(r&1), k=4*lc+16*(r>>1)+b;
      v |= ((unsigned)(unsigned char)x[row*256+group*32+k]) << (8*b);
    }
    ar[r]=(int)v;
  }
  #pragma unroll
  for (int r=0;r<2;r++) {
    unsigned v=0;
    #pragma unroll
    for (int b=0;b<4;b++) {
      int k=4*(lc+4*r)+b, col=lr, pair=group>>1;
      unsigned byte=(words[col*36 + 4 + pair*8 + (k>>2)] >> (8*(k&3))) & 255u;
      unsigned q=(group&1) ? (byte>>4) : (byte&15u);
      v |= q << (8*b);
    }
    br[r]=(int)v;
  }
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
    : "+r"(cr[0]),"+r"(cr[1]),"+r"(cr[2]),"+r"(cr[3])
    : "r"(ar[0]),"r"(ar[1]),"r"(ar[2]),"r"(ar[3]),"r"(br[0]),"r"(br[1]));
  #pragma unroll
  for (int r=0;r<4;r++) {
    int row=lr+8*(r>>1), col=2*lc+(r&1);
    out[row*8+col]=cr[r];
  }
}

#include <cuda_fp16.h>
__device__ __forceinline__ unsigned q4byte(const unsigned int *w, int base, int off) {
  return (w[base+(off>>2)]>>(8*(off&3)))&255u;
}
template<bool local_out> __device__ __forceinline__ void q4k_run_tile(float *out, const unsigned int *words, const signed char *x,
    const float *xscale, const float *xsum, int M, int N, int K, int tile, int b0, int b1) {
  extern __shared__ unsigned int arena[];
  unsigned int *ids=arena, *sy=ids+128, *sw=sy+128*36;
  int lane=threadIdx.x&31, warp=threadIdx.x>>5, lr=lane>>2, lc=lane&3;
  int ntx=(N+127)/128,mb=(tile/ntx)*128,nb=(tile%ntx)*128,blocks=K/256;
  int nr=(warp>>1)*32, mp=(warp&1)*8;
  float acc[2][8][4]={0};
  for (int blk=b0;blk<b1;blk++) {
    for(int rr=warp;rr<128;rr+=8) { int col=nb+rr,base=(col*blocks+blk)*36;
      for(int z=lane;z<64;z+=32) { int g=z>>3,kw=z&7;unsigned v=0;
        if(col<N) { unsigned raw=words[base+4+(g>>1)*8+kw];v=(g&1)?((raw>>4)&0x0f0f0f0fu):(raw&0x0f0f0f0fu); } sw[rr*76+z]=v;
      }
      if(lane<8) { unsigned g=lane,sc=0,mn=0,w0=col<N?words[base]:0;
        if(col<N) {
          if(g<4) { sc=q4byte(words,base,4+g)&63u;mn=q4byte(words,base,8+g)&63u; }
          else { int h=g-4;sc=(q4byte(words,base,12+h)&15u)|((q4byte(words,base,4+h)>>6)<<4);
                 mn=(q4byte(words,base,12+h)>>4)|((q4byte(words,base,8+h)>>6)<<4); }
        }
        float D=__half2float(__ushort_as_half(w0&65535u)),Dm=__half2float(__ushort_as_half(w0>>16));
        *(__half2 *)&sw[rr*76+64+g]=__floats2half2_rn(D*(float)sc,-Dm*(float)mn);
      }
      if(lane<4) sw[rr*76+72+lane]=0;
    }
    asm volatile("bar.sync 0, 256;" ::: "memory");
    #pragma unroll
    for(int half=0;half<2;half++) {
      for(int z=threadIdx.x;z<128*32;z+=256) { int rr=z>>5,kw=z&31,row=mb+rr;unsigned v=0;
        if(row<M) v=((const unsigned int *)(x+row*K+blk*256+half*128))[kw];sy[rr*36+kw]=v;
      }
      for(int z=threadIdx.x;z<128*4;z+=256) { int rr=z>>2,g=z&3,row=mb+rr;
        float ds=0,sm=0;if(row<M) { int xm=row*(blocks*8)+blk*8+half*4+g;ds=xscale[xm];sm=xsum[xm]; }
        *(__half2 *)&sy[rr*36+32+g]=__floats2half2_rn(ds,sm);
      }
      asm volatile("bar.sync 0, 256;" ::: "memory");
      #pragma unroll
      for(int gg=0;gg<4;gg++) { int group=half*4+gg;
    #pragma unroll
    for(int nt=0;nt<2;nt++) { int ar[4];
      int *lp=(int *)(sw+(nr+nt*16)*76+group*8)+(lane&15)*76+(lane>>4)*4;
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
        :"=r"(ar[0]),"=r"(ar[1]),"=r"(ar[2]),"=r"(ar[3]):"l"(lp));
      #pragma unroll
      for(int mt=0;mt<8;mt++) { int br[2],cr[4]={0,0,0,0};
        #pragma unroll
        for(int r=0;r<2;r++){unsigned v=0;
          #pragma unroll
          for(int q=0;q<4;q++){int k=4*(lc+4*r)+q,mr=mp+16*mt+lr;
            v|=((sy[mr*36+gg*8+(k>>2)]>>(8*(k&3)))&255u)<<(8*q);}br[r]=(int)v;}
      asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      : "+r"(cr[0]),"+r"(cr[1]),"+r"(cr[2]),"+r"(cr[3])
      : "r"(ar[0]),"r"(ar[1]),"r"(ar[2]),"r"(ar[3]),"r"(br[0]),"r"(br[1]));
      #pragma unroll
      for(int r=0;r<4;r++) { int col=nb+nr+nt*16+lr+((r>>1)?8:0),row=mb+mp+16*mt+2*lc+(r&1); if(row<M&&col<N) {
        __half2 wc=*(__half2 *)&sw[(nr+nt*16+lr+((r>>1)?8:0))*76+64+group];
        __half2 yc=*(__half2 *)&sy[(mp+16*mt+2*lc+(r&1))*36+32+gg];
        acc[nt][mt][r]+=__half2float(__low2half(wc))*__half2float(__low2half(yc))*(float)cr[r]
                       +__half2float(__high2half(wc))*__half2float(__high2half(yc));
      }}
    }
    }
      }
      asm volatile("bar.sync 0, 256;" ::: "memory");
    }
  }
  #pragma unroll
  for(int nt=0;nt<2;nt++) {
    #pragma unroll
    for(int mt=0;mt<8;mt++) {
      #pragma unroll
      for(int r=0;r<4;r++) { int col=nb+nr+nt*16+lr+((r>>1)?8:0),row=mb+mp+16*mt+2*lc+(r&1);if(row<M&&col<N) {
        if(local_out) out[(row-mb)*128+col-nb]=acc[nt][mt][r]; else out[row*N+col]=acc[nt][mt][r];
      }}
    }
  }
}
extern "C" __global__ void q4k_imma_complete(float *out, const unsigned int *words, const signed char *x,
    const float *xscale, const float *xsum, int M, int N, int K) {
  q4k_run_tile<false>(out,words,x,xscale,xsum,M,N,K,blockIdx.y*((N+127)/128)+blockIdx.x,0,K/256);
}
extern "C" __global__ void q4k_imma_stream(float *out, float *partials, int *ids, const unsigned int *words,
    const signed char *x, const float *xscale, const float *xsum, int M, int N, int K) {
  int cid=blockIdx.x,total=(M/128)*(N/128)*(K/256),u=(cid*total)/170,ue=((cid+1)*total)/170,piece=0;
  ids[cid*2]=ids[cid*2+1]=-1;
  while(u<ue) { int tile=u/(K/256),b0=u%(K/256),end=min(ue,(tile+1)*(K/256)),b1=end-tile*(K/256);
    if(b0==0 && b1==K/256) q4k_run_tile<false>(out,words,x,xscale,xsum,M,N,K,tile,b0,b1);
    else { int slot=cid*2+piece++;ids[slot]=tile;
      q4k_run_tile<true>(partials+slot*128*128,words,x,xscale,xsum,M,N,K,tile,b0,b1);
    }
    u=end;
  }
}
extern "C" __global__ void q4k_imma_fixup(float *out, const float *partials, const int *ids, int M, int N) {
  int slot=blockIdx.x,tile=ids[slot];if(tile<0)return;int nb=(tile%(N/128))*128,mb=(tile/(N/128))*128;
  for(int z=threadIdx.x;z<128*128;z+=256) { int r=z/128,c=z%128;if(mb+r<M&&nb+c<N)atomicAdd(out+(mb+r)*N+nb+c,partials[slot*128*128+z]); }
}
'''

def lexical_src(src:str, stream_only=False) -> str:
  """Specialize the barrier-heavy tile body directly into kernel entry points."""
  hs=src.index("template<bool local_out>")
  op=src.index("{",hs); depth=0; cl=-1
  for i in range(op,len(src)):
    depth += (src[i]=="{")-(src[i]=="}")
    if depth==0: cl=i; break
  body=src[op+1:cl]
  prefix=src[:hs]
  if stream_only: prefix=src[src.index("#include <cuda_fp16.h>"):hs]
  def site(local:bool, out_expr:str, indent="  "):
    old="if(local_out) out[(row-mb)*128+col-nb]=acc[mt][ns][r]; else out[row*N+col]=acc[mt][ns][r];"
    new=("out[(row-mb)*128+col-nb]=acc[mt][ns][r];" if local else "out[row*N+col]=acc[mt][ns][r];")
    b=body.replace(old,new)
    b=b.replace("if(local_out)",f"if({'true' if local else 'false'})")
    b=b.replace("out[","tile_out[")
    return f"{{ float *tile_out={out_expr};\n"+b+"\n}"
  static_site=site(False,"out")
  static_site=static_site.replace("int ntx=(N+127)/128,mb=(tile/ntx)*128,nb=(tile%ntx)*128,blocks=K/256;",
                                  "int mb=blockIdx.y*128,nb=blockIdx.x*128,blocks=K/256;").replace(
                                  "for (int blk=b0;blk<b1;blk++)","for (int blk=0;blk<blocks;blk++)")
  full_site=site(False,"out")
  part_site=site(True,"partials+slot*128*128")
  static_kernel=f'''
extern "C" __global__ void q4k_imma_complete(float *out, const unsigned int *words, const signed char *x,
    const float *xscale, const float *xsum, int M, int N, int K) {{
  {static_site}
}}
'''
  if not stream_only:return prefix+static_kernel
  return prefix+f'''
extern "C" __global__ void q4k_imma_stream(float *out, float *partials, int *ids, const unsigned int *words,
    const signed char *x, const float *xscale, const float *xsum, int M, int N, int K) {{
  int cid=blockIdx.x,total=(M/128)*(N/128)*(K/256),owners=min(170,total),u=(cid*total)/owners,ue=((cid+1)*total)/owners,piece=0;
  ids[cid*2]=ids[cid*2+1]=-1;
  while(u<ue) {{ int tile=u/(K/256),b0=u%(K/256),end=min(ue,(tile+1)*(K/256)),b1=end-tile*(K/256);
    if(b0==0 && b1==K/256) {full_site}
    else {{int slot=cid*2+piece++;ids[slot]=tile;{part_site}}}
    u=end;
  }}
}}
'''

def production_slotmap() -> np.ndarray:
  """Map each production output tile to its one or two Stream-K boundary slots."""
  slotmap=np.full((384,2),-1,np.int32)
  for cid in range(170):
    u=(cid*6144)//170;ue=((cid+1)*6144)//170;piece=0
    while u<ue:
      tile=u//16;b0=u%16;end=min(ue,(tile+1)*16);b1=end-tile*16
      if not (b0==0 and b1==16):
        j=0 if slotmap[tile,0]<0 else 1;slotmap[tile,j]=cid*2+piece;piece+=1
      u=end
  return slotmap

def alloc(dev,n): return dev.allocator._alloc(n,BufferSpec())
def copyin(dev,b,a): dev.allocator._copyin(b,memoryview(np.ascontiguousarray(a).tobytes()))
def copyout(dev,b,dtype,shape):
  h=memoryview(bytearray(b.size));dev.allocator._copyout(h,b);return np.frombuffer(h,dtype=dtype,count=int(np.prod(shape))).reshape(shape).copy()

def main():
  ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--artifacts',required=True);ap.add_argument('--production',action='store_true');a=ap.parse_args()
  rng=np.random.default_rng(20260828); raw=rng.integers(0,256,(128,144),dtype=np.uint8); x=rng.integers(-127,128,(128,256),dtype=np.int8)
  raw[:,:4]=np.frombuffer(np.array([.0625,.03125],dtype=np.float16).tobytes(),dtype=np.uint8)
  words=np.frombuffer(raw.tobytes(),dtype=np.uint32).copy();dev=Device['NV']; wb,xb,ob=alloc(dev,words.nbytes),alloc(dev,x.nbytes),alloc(dev,16*8*4)
  copyin(dev,wb,words);copyin(dev,xb,x);art=pathlib.Path(a.artifacts);art.mkdir(parents=True,exist_ok=True)
  csrc=lexical_src(SRC)
  cubin=NVRTCCompiler(dev.arch,ptx=False,cache_key='q4k_imma_fragment_v21').compile(csrc);(art/'q4k_imma_fragment.cubin').write_bytes(cubin);(art/'q4k_imma_fragment.cu').write_text(csrc)
  if a.production:
    print('production timing start',flush=True)
    ssrc=lexical_src(SRC,True)
    ssrc=ssrc.replace("int ntx=(N+127)/128,mb=(tile/ntx)*128,nb=(tile%ntx)*128,blocks=K/256;",
      "int ntx=96,mb=(tile/96)*128,nb=(tile%96)*128,blocks=16;").replace("row*K+blk*256","row*4096+blk*256")
    ssrc=ssrc.replace("total=(M/128)*(N/128)*(K/256),owners=min(170,total)","total=6144,owners=170").replace("K/256","16")
    ssrc=ssrc.replace("if(col<N) { unsigned raw=", "{ unsigned raw=").replace("w0=col<N?words[base]:0;","w0=words[base];").replace("if(col<N) {","{")
    ssrc=ssrc.replace("if(row<M) v=", "v=").replace("if(row<M) { int xm=", "{ int xm=")
    ssrc=ssrc.replace("if(row<M&&col<N) {","{").replace("if(row<M&&col<N) {","{").replace("row*N+col","row*12288+col")
    scubin=NVRTCCompiler(dev.arch,ptx=False,cache_key='q4k_imma_stream_prod_v5').compile(ssrc)
    (art/'q4k_imma_stream.cubin').write_bytes(scubin);(art/'q4k_imma_stream.cu').write_text(ssrc)
    pm,pn,pk=512,12288,4096;zs=(np.zeros((pn,pk//256,36),np.uint32),np.zeros((pm,pk),np.int8),np.zeros((pm,pk//32),np.float32),np.zeros((pm,pk//32),np.float32))
    bs=[alloc(dev,z.nbytes) for z in zs];[copyin(dev,b,z) for b,z in zip(bs,zs)]
    oo,pp,ii=alloc(dev,pm*pn*4),alloc(dev,340*128*128*4),alloc(dev,340*4)
    fsrc='extern "C" __global__ void q4k_imma_fixup(float *o,const float*p,const int*map,int M,int N){int t=blockIdx.x,s0=map[2*t];if(s0<0)return;int s1=map[2*t+1],nb=(t%(N/128))*128,mb=(t/(N/128))*128;for(int z=threadIdx.x;z<16384;z+=256){int r=z/128,c=z%128;if(mb+r<M&&nb+c<N)o[(mb+r)*N+nb+c]=p[s0*16384+z]+(s1>=0?p[s1*16384+z]:0);}}'
    fcubin=NVRTCCompiler(dev.arch,ptx=False,cache_key='q4k_imma_fixup_v5').compile(fsrc)
    stream=NVProgram(dev,'q4k_imma_stream',scubin,shared_mem=57856+1024);fixup=NVProgram(dev,'q4k_imma_fixup',fcubin)
    slotmap=production_slotmap()
    mapb=alloc(dev,slotmap.nbytes);copyin(dev,mapb,slotmap)
    sm=stream(oo,pp,ii,*bs,vals=(pm,pn,pk),global_size=(170,1,1),local_size=(256,1,1),wait=True,timeout=10)*1e3
    fx=fixup(oo,pp,mapb,vals=(pm,pn),global_size=(384,1,1),local_size=(256,1,1),wait=True,timeout=10)*1e3
    print('stream smoke',sm,fx,flush=True)
    print('stream repeat',stream(oo,pp,ii,*bs,vals=(pm,pn,pk),global_size=(170,1,1),local_size=(256,1,1),wait=True,timeout=10)*1e3,flush=True)
    early_sts=[]
    for _ in range(12):
      mt=stream(oo,pp,ii,*bs,vals=(pm,pn,pk),global_size=(170,1,1),local_size=(256,1,1),wait=True)*1e3
      ft=fixup(oo,pp,mapb,vals=(pm,pn),global_size=(384,1,1),local_size=(256,1,1),wait=True)*1e3;early_sts.append(mt+ft)
    print('early stream timing done',flush=True)
  prg=NVProgram(dev,'q4k_imma_fragment',cubin); rows=[]
  qs=raw[:,16:].reshape(128,4,32)
  for g in range(8):
    us=prg(ob,wb,xb,vals=(g,),global_size=(1,1,1),local_size=(32,1,1),wait=True);got=copyout(dev,ob,np.int32,(16,8))
    q=(qs[:8,g//2,:]>>(4*(g&1)))&15;ref=x[:16,g*32:(g+1)*32].astype(np.int32)@q.astype(np.int32).T
    rows.append({'group':g,'us':us*1e6,'exact':bool(np.array_equal(got,ref)),'max_abs':int(np.abs(got-ref).max()),'nonzero':int(np.count_nonzero(got))})
  xs=(rng.random((128,8),dtype=np.float32)*.02).astype(np.float32); xsum=(rng.standard_normal((128,8))*5).astype(np.float32)
  sb,sumb,fob=alloc(dev,xs.nbytes),alloc(dev,xsum.nbytes),alloc(dev,128*128*4);copyin(dev,sb,xs);copyin(dev,sumb,xsum)
  complete=NVProgram(dev,'q4k_imma_complete',cubin,shared_mem=57856+1024);complete_us=complete(fob,wb,xb,sb,sumb,vals=(128,128,256),global_size=(1,1,1),local_size=(256,1,1),wait=True,timeout=5)*1e6
  final=copyout(dev,fob,np.float32,(128,128))
  # Independent per-output reference uses the canonical oracle.
  from extra.llm_research.prefill.q4k_q8_imma_oracle import unpack_scales,unpack_qs
  final_ref=np.empty((128,128),np.float32)
  llama_ref=np.empty((128,128),np.float32)
  for mi in range(128):
    for ni in range(128):
      scn,mnn=unpack_scales(raw[ni,4:16].tobytes());qn=unpack_qs(raw[ni,16:].tobytes()).astype(np.int32)
      dots=(x[mi].reshape(8,32).astype(np.int32)*qn).sum(1,dtype=np.int32)
      final_ref[mi,ni]=np.float32(.0625)*np.sum(xs[mi]*scn*dots,dtype=np.float32)-np.float32(.03125)*np.sum(mnn*xsum[mi],dtype=np.float32)
      wc0=(np.float16(np.float32(.0625)*scn)).astype(np.float32);wc1=(np.float16(-np.float32(.03125)*mnn)).astype(np.float32)
      yc0=xs[mi].astype(np.float16).astype(np.float32);yc1=xsum[mi].astype(np.float16).astype(np.float32)
      llama_ref[mi,ni]=np.sum(wc0*yc0*dots+wc1*yc1,dtype=np.float32)
  fd=np.abs(final-final_ref)
  final_row={'us':complete_us,'finite':bool(np.isfinite(final).all()),'max_abs':float(fd.max()),'mean_abs':float(fd.mean()),'allclose':bool(np.allclose(final,final_ref,rtol=2e-5,atol=2e-3))}
  ld=np.abs(final-llama_ref);final_row.update({'llama_max_abs':float(ld.max()),'llama_mean_abs':float(ld.mean()),'llama_allclose':bool(np.allclose(final,llama_ref,rtol=2e-5,atol=2e-3))})
  if a.production:
    pm,pn,pk=512,12288,4096; pw=np.zeros((pn,pk//256,36),np.uint32);px=np.zeros((pm,pk),np.int8)
    ps=np.zeros((pm,pk//32),np.float32);pu=np.zeros_like(ps)
    pbufs=[alloc(dev,z.nbytes) for z in (pw,px,ps,pu,np.empty((pm,pn),np.float32))]
    for b,z in zip(pbufs[:4],(pw,px,ps,pu)):copyin(dev,b,z)
    ts=[]
    for _ in range(12):ts.append(complete(pbufs[4],pbufs[0],pbufs[1],pbufs[2],pbufs[3],vals=(pm,pn,pk),global_size=(96,4,1),local_size=(256,1,1),wait=True,timeout=10)*1e3)
    final_row['production_ms']={'min':float(min(ts[3:])),'median':float(np.median(ts[3:])),'samples':ts}
    print('static timing done',flush=True)
    sts=early_sts
    final_row['stream_complete_ms']={'min':float(min(sts[3:])),'median':float(np.median(sts[3:])),'samples':sts}
    print('stream timing done',flush=True)
  result={'schema':'nv.q4k_imma_fragment.v2','passed':all(r['exact'] and r['nonzero'] for r in rows) and final_row['llama_allclose'],
    'rows':rows,'complete':final_row,'cubin':str(art/'q4k_imma_fragment.cubin')}
  pathlib.Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
