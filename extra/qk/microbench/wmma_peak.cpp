#include <hip/hip_runtime.h>
#include <cstdio>
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float    f8  __attribute__((ext_vector_type(8)));
#define NACC 8                        // independent accumulators to cover WMMA latency
__global__ __launch_bounds__(256) void wmma_peak(float* out, int iters){
  h16 a, b;
  for (int i=0;i<16;i++){ a[i]=(_Float16)(1.0f+0.001f*i); b[i]=(_Float16)(0.5f+0.002f*i); }
  f8 c[NACC];
  #pragma unroll
  for (int j=0;j<NACC;j++) for (int i=0;i<8;i++) c[j][i]=0.0f;
  for (int t=0;t<iters;t++){
    #pragma unroll
    for (int j=0;j<NACC;j++) c[j]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a,b,c[j]);
  }
  float s=0; 
  #pragma unroll
  for (int j=0;j<NACC;j++) for (int i=0;i<8;i++) s+=c[j][i];
  if (s==1234.5f) out[0]=s;           // keep it live, never taken
}
int main(){
  int iters=20000, blocks=2048, tpb=256;      // 2048 blocks x 8 waves = 16384 waves
  float* d; hipMalloc(&d, 4);
  hipLaunchKernelGGL(wmma_peak, dim3(blocks), dim3(tpb), 0, 0, d, 1000); hipDeviceSynchronize();
  hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
  hipEventRecord(s);
  hipLaunchKernelGGL(wmma_peak, dim3(blocks), dim3(tpb), 0, 0, d, iters);
  hipEventRecord(e); hipDeviceSynchronize();
  float ms; hipEventElapsedTime(&ms,s,e);
  double waves = (double)blocks * (tpb/32);
  double flop = waves * iters * NACC * 2.0*16*16*16;   // 8192 FLOP per wave-WMMA
  printf("waves=%.0f iters=%d nacc=%d  time=%.3f ms  -> %.1f TFLOPS\n",
         waves, iters, NACC, ms, flop/(ms*1e-3)/1e12);
  return 0;
}
