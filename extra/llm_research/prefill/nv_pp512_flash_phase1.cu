#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

// Phase-1 clean-room substrate.  Contract: Q fp32 [B,Hq,S,D], K/V fp16
// [B,Hkv,Skv,D], contiguous, S=Skv=512,D=128,Hq=32,Hkv=8.
extern "C" __global__ void nv_pp512_flash_phase1(const float *q, const half *k,
    const half *v, float *out, int start_pos, float scale) {
  constexpr int D=128, S=512, Hq=32, Hkv=8;
  const int h=blockIdx.y, t=blockIdx.x, b=blockIdx.z, kvh=h/4;
  if (h>=Hq || t>=S || b!=0) return;
  __shared__ float qs[D], acc[D];
  if (threadIdx.x < D) { qs[threadIdx.x]=q[(h*S+t)*D+threadIdx.x]; acc[threadIdx.x]=0.0f; }
  __syncthreads();
  if (threadIdx.x==0) {
    float m=-INFINITY, z=0.0f;
    for (int j=0; j<S; j++) {
      if (j > t+start_pos) break;
      float dot=0.0f;
      // 16-byte aligned/coalesced K/V vector loads.  The admitted contract
      // proves D and base offsets are multiples of eight half elements.
      const uint4 *kp=(const uint4 *)(k+(kvh*S+j)*D);
      const uint4 *vp=(const uint4 *)(v+(kvh*S+j)*D);
      const uint4 *qp=(const uint4 *)(qs);
      for (int x=0;x<D/8;x++) {
        uint4 kw=kp[x], vw=vp[x];
        half *kh=(half*)&kw; half *vh=(half*)&vw;
        dot += __half2float(kh[0])*qs[x*8+0]+__half2float(kh[1])*qs[x*8+1];
        dot += __half2float(kh[2])*qs[x*8+2]+__half2float(kh[3])*qs[x*8+3];
        dot += __half2float(kh[4])*qs[x*8+4]+__half2float(kh[5])*qs[x*8+5];
        dot += __half2float(kh[6])*qs[x*8+6]+__half2float(kh[7])*qs[x*8+7];
        (void)qp;
      }
      float s=dot*scale, nm=fmaxf(m,s), a=expf(m-nm), e=expf(s-nm);
      for (int x=0;x<D;x++) acc[x]=acc[x]*a+e*__half2float(((half*)vp)[x]);
      z=z*a+e; m=nm;
    }
    for (int x=0;x<D;x++) out[(h*S+t)*D+x]=z>0 ? acc[x]/z : 0.0f;
  }
}
