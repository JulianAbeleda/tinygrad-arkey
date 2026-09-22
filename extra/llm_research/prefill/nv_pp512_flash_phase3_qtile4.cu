#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
// Clean-room semantic transcription baseline: four query rows per CTA.
extern "C" __global__ void nv_pp512_flash_phase3_qtile4(const half*q,const half*k,const half*v,float*out){
 constexpr int D=128,S=512,HQ=32,G=4,TQ=4; int h0=blockIdx.y*TQ,t0=blockIdx.x*TQ,lane=threadIdx.y*32+threadIdx.x;
 if(lane>=D||h0>=HQ||t0>=S)return; __shared__ half qs[TQ][D]; __shared__ float acc[TQ][D],wm[TQ],wd[TQ],wdot[TQ][4],alpha[TQ],beta[TQ];
 for(int r=0;r<TQ;r++){int h=h0+r,t=t0+r;if(h<HQ&&t<S){qs[r][lane]=q[(h*S+t)*D+lane];acc[r][lane]=0;}if(lane==0){wm[r]=-INFINITY;wd[r]=0;}} __syncthreads();
 for(int j=0;j<S;j++){for(int r=0;r<TQ;r++){int h=h0+r,t=t0+r;if(h>=HQ||t>=S||j>t)continue;float x=__half2float(qs[r][lane])*__half2float(k[((h/G)*S+j)*D+lane]);for(int o=16;o;o>>=1)x+=__shfl_down_sync(0xffffffff,x,o);if((threadIdx.x==0))wdot[r][threadIdx.y]=x;}__syncthreads();
  for(int r=0;r<TQ;r++){int h=h0+r,t=t0+r;if(h>=HQ||t>=S||j>t)continue;if(lane==0){float s=(wdot[r][0]+wdot[r][1]+wdot[r][2]+wdot[r][3])*rsqrtf(128.f),nm=fmaxf(wm[r],s);alpha[r]=isinf(wm[r])?0.0f:expf(wm[r]-nm);beta[r]=expf(s-nm);wm[r]=nm;wd[r]=wd[r]*alpha[r]+beta[r];}__syncthreads();float aa=alpha[r],bb=beta[r];acc[r][lane]=acc[r][lane]*aa+bb*__half2float(v[((h/G)*S+j)*D+lane]);__syncthreads();}}
 for(int r=0;r<TQ;r++){int h=h0+r,t=t0+r;if(h<HQ||t>=S)out[(h*S+t)*D+lane]=acc[r][lane]/fmaxf(wd[r],1e-30f);}
}
