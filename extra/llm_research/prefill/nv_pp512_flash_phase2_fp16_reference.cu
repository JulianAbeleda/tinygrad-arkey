#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

// Correctness-first reference: one CTA per (query head, query token).
// q/k/v are logical [Hq,T,D], [Hkv,MAXC,D], and [Hkv,MAXC,D].
extern "C" __global__ void nv_pp512_flash_phase2_fp16_reference(
    const half *q, const half *k, const half *v, float *out, int start_pos, int ntok, int maxc) {
  constexpr int D=128, Hq=32, Hkv=8, G=4;
  const int h=blockIdx.y, t=blockIdx.x, lane=threadIdx.y*32+threadIdx.x;
  if (h>=Hq || t>=ntok || lane>=D) return;
  const int kvh=h/G, qpos=start_pos+t;
  __shared__ half qs[D];
  __shared__ float warp_dot[4], score, running_max, running_den, decay, weight;
  __shared__ float accum[D];
  qs[lane]=q[(h*ntok+t)*D+lane];
  accum[lane]=0.0f;
  if (lane==0) { running_max=-INFINITY; running_den=0.0f; }
  __syncthreads();
  for (int j=0; j<=qpos; ++j) {
    float part=__half2float(qs[lane])*__half2float(k[((kvh*maxc+j)*D)+lane]);
    for (int off=16; off; off>>=1) part += __shfl_down_sync(0xffffffff,part,off);
    if ((threadIdx.x==0)) warp_dot[threadIdx.y]=part;
    __syncthreads();
    if (lane==0) score=(warp_dot[0]+warp_dot[1]+warp_dot[2]+warp_dot[3])*(1.0f/sqrtf((float)D));
    __syncthreads();
    if (lane==0) { float nm=fmaxf(running_max,score); float a=expf(running_max-nm), b=expf(score-nm); decay=a; weight=b; running_den=running_den*a+b; running_max=nm; }
    __syncthreads();
    float vv=__half2float(v[(kvh*maxc+j)*D+lane]);
    accum[lane]=accum[lane]*decay+weight*vv;
    __syncthreads();
  }
  out[(h*ntok+t)*D+lane]=accum[lane]/fmaxf(running_den,1e-30f);
}
