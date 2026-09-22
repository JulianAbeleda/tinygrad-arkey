
#include "ggml-cuda/fattn-common.cuh"
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(128) flash_fused_gmax_combine_f16_32_128_s6_lw128(half* data0_4096, float* data1_24960) {
  int gidx0 = blockIdx.x; /* 32 */
  float buf0[1];
  (*(buf0+0)) = -1e+30f;
  int alu1 = (gidx0*780);
  for (int Ridx2 = 0; Ridx2 < 6; Ridx2++) {
    float val0 = (*(data1_24960+(alu1+(Ridx2*130)+129)));
    float alu2 = (((*(buf0+0))<val0)?val0:(*(buf0+0)));
    (*(buf0+0)) = alu2;
  }
  int lidx0 = threadIdx.x; /* 128 */
  bool alu5 = (lidx0<6);
  int alu6 = (alu5?lidx0:0);
  float val1 = (*(data1_24960+(alu1+(alu6*130)+129)));
  __shared__ __align__(16) float buf1[6];
  float buf2[1];
  float buf3[1];
  if (alu5) {
    *(buf1+alu6) = exp2(((val1-(*(buf0+0)))*1.4426950408889634f));
  }
  __syncthreads();
  (*(buf2+0)) = 0.0f;
  (*(buf3+0)) = 0.0f;
  for (int Ridx4 = 0; Ridx4 < 6; Ridx4++) {
    int alu13 = (alu1+(Ridx4*130));
    float val2 = (*(data1_24960+(lidx0+alu13)));
    float val3 = (*(data1_24960+(alu13+128)));
    float val4 = (*(buf1+Ridx4));
    (*(buf2+0)) = ((*(buf2+0))+(val4*val2));
    (*(buf3+0)) = ((*(buf3+0))+(val4*val3));
  }
  *(data0_4096+(lidx0+(gidx0<<7))) = ((half)(((*(buf2+0))*(1/(*(buf3+0))))));
}
extern "C" __global__ void __launch_bounds__(128) flash_fused_gmax_combine_f16_32_128_s6_lw128_regw(half* data0_4096, float* data1_24960) {
  int gidx0 = blockIdx.x; /* 32 */
  float buf0[1];
  (*(buf0+0)) = -1e+30f;
  int alu1 = (gidx0*780);
  for (int Ridx2 = 0; Ridx2 < 6; Ridx2++) {
    float val0 = (*(data1_24960+(alu1+(Ridx2*130)+129)));
    float alu2 = (((*(buf0+0))<val0)?val0:(*(buf0+0)));
    (*(buf0+0)) = alu2;
  }
  int lidx0 = threadIdx.x; /* 128 */
  int alu5 = (lidx0&31);
  bool alu6 = (alu5<6);
  int alu7 = (alu6?alu5:0);
  float val1 = (*(data1_24960+(alu1+(alu7*130)+129)));
  float buf1[1];
  float buf2[1];
  float buf3[1];
  float alu8 = (alu6?exp2(((val1-(*(buf0+0)))*1.4426950408889634f)):0.0f);
  (*(buf3+0)) = alu8;
  (*(buf1+0)) = 0.0f;
  (*(buf2+0)) = 0.0f;
  for (int Ridx4 = 0; Ridx4 < 6; Ridx4++) {
    int alu12 = (alu1+(Ridx4*130));
    float val2 = (*(data1_24960+(lidx0+alu12)));
    float val3 = (*(data1_24960+(alu12+128)));
    float alu13 = __shfl_sync(0xffffffffu, (*(buf3+0)), ((((unsigned int)((Ridx4<<2)))) >> 2));
    (*(buf1+0)) = ((*(buf1+0))+(alu13*val2));
    (*(buf2+0)) = ((*(buf2+0))+(alu13*val3));
  }
  *(data0_4096+(lidx0+(gidx0<<7))) = ((half)(((*(buf1+0))*(1/(*(buf2+0))))));
}
#include <cstdio>
int main() {
  constexpr int H=32,D=128,S=6,W=130; float *tp,*lp,*lm,*lo; half *to; cudaEvent_t a,b;
  cudaMalloc(&tp,H*S*W*4);cudaMalloc(&to,H*D*2);cudaMalloc(&lp,H*S*D*4);cudaMalloc(&lm,H*S*8);cudaMalloc(&lo,H*D*4);
  cudaMemset(tp,1,H*S*W*4);cudaMemset(lp,1,H*S*D*4);cudaMemset(lm,1,H*S*8);cudaEventCreate(&a);cudaEventCreate(&b);
  auto one=[&](int arm,int n){cudaEventRecord(a);for(int i=0;i<n;i++){
    if(arm==0) flash_fused_gmax_combine_f16_32_128_s6_lw128<<<H,128>>>(to,tp); else if(arm==1) flash_fused_gmax_combine_f16_32_128_s6_lw128_regw<<<H,128>>>(to,tp);
    else flash_attn_combine_results<D><<<dim3(1,H,1),128,S*sizeof(float2)>>>(lp,(float2*)lm,lo,S);
  }cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);return 1000.0f*ms/n;};
  for(int arm=0;arm<3;arm++)for(int i=0;i<200;i++)one(arm,1);
  for(int r=0;r<1;r++)printf("rep=%d tiny_shared=%.6f tiny_register=%.6f llama=%.6f\n",r,one(0,100),one(1,100),one(2,100));
}
