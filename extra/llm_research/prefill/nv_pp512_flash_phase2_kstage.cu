#include <cuda_fp16.h>
#include <cuda_runtime.h>
extern "C" __global__ void nv_pp512_flash_phase2_kstage(const float*q,const half*k,const half*v,float*out){
  constexpr int D=128,S=512; int h=blockIdx.y,t=blockIdx.x,tid=threadIdx.x,kvh=h/4;
  __shared__ float qs[D], acc[D], sk[D], red[4], score;
  if(tid<D) { qs[tid]=q[(h*S+t)*D+tid]; acc[tid]=0; }
  __syncthreads(); float m=-INFINITY,z=0; 
  for(int j=0;j<=t;j++){
    if(tid<32){ const uint4 w=((const uint4*)(k+(kvh*S+j)*D))[tid]; ((uint4*)sk)[tid]=w; }
    __syncthreads(); float d=0; for(int x=tid;x<D;x+=128) d+=qs[x]*__half2float(((half*)sk)[x]);
    for(int off=16;off;off>>=1)d+=__shfl_down_sync(0xffffffff,d,off);
    if((tid&31)==0) red[tid>>5]=d;
    __syncthreads();
    if(tid<32){float x=tid<4?red[tid]:0;for(int off=16;off;off>>=1)x+=__shfl_down_sync(0xffffffff,x,off);if(tid==0)score=x*rsqrtf(128.0f);}
    __syncthreads(); d=score; float nm=fmaxf(m,d),a=expf(m-nm),e=expf(d-nm);
    float vv=__half2float(v[(kvh*S+j)*D+tid]); acc[tid]=acc[tid]*a+e*vv; z=z*a+e; m=nm; __syncthreads();
  }
  if(tid<D) out[(h*S+t)*D+tid]=acc[tid]/z;
}
