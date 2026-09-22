// fma_peak_cuda.cu - measured achievable fp32 FMA peak (sm_120 / RTX 5090)
//
// The CUDA-core FP analogue of dp4a_peak_cuda.cu. Isolated back-to-back
// FFMA on register-resident operands: NACC independent accumulators,
// runtime trip count so the loop is not folded, and a never-taken store so the
// result stays live. Zero loads in the hot loop.
//
//   nvcc -O3 -arch=sm_120 -DNACC=8 fma_peak_cuda.cu -o fma_peak_cuda
//   ./fma_peak_cuda 32768

// Accounting note: the printed "G FMA/s" is warp-instruction issue rate and
// matches dp4a_peak_cuda.cu's "G dp4a/s" (both count warps * iters * NACC).
// TFLOPS multiplies by the 32 lanes per warp and 2 FLOP/FMA.

#include <cuda_runtime.h>
#include <cstdio>

#ifndef NACC
#define NACC 8
#endif

__global__ __launch_bounds__(256) void fma_peak(float* out, int iters) {
  float a[NACC], b[NACC], c[NACC];
  #pragma unroll
  for (int j = 0; j < NACC; j++) {
    a[j] = 1.0000001f * (j + 1);
    b[j] = 0.9999999f * (j + 1);
    c[j] = 0.0f;
  }
  for (int t = 0; t < iters; t++) {
    #pragma unroll
    for (int j = 0; j < NACC; j++) {
      asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(c[j]) : "f"(a[j]), "f"(b[j]));
    }
  }
  float s = 0.0f;
  #pragma unroll
  for (int j = 0; j < NACC; j++) s += c[j];
  if (s == 1234567.0f) out[0] = s;   // keep it live, never taken
}

int main(int argc, char** argv) {
  int blocks = argc > 1 ? atoi(argv[1]) : 32768;
  int iters = 200000 / NACC;
  int tpb = 256;
  float* d;
  cudaMalloc(&d, 4);
  fma_peak<<<blocks, tpb>>>(d, 1000);
  cudaDeviceSynchronize();
  cudaEvent_t s, e;
  cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  fma_peak<<<blocks, tpb>>>(d, iters);
  cudaEventRecord(e);
  cudaDeviceSynchronize();
  float ms; cudaEventElapsedTime(&ms, s, e);
  double warps = (double)blocks * (tpb / 32);
  double fma_warp_instr = warps * iters * NACC;   // warp-instructions
  double fma_thread = fma_warp_instr * 32.0;      // per-lane FMA ops
  double flops = fma_thread * 2.0;                // one FMA = 2 FLOP
  printf("blocks=%d tpb=%d iters=%d nacc=%d  time=%.2f ms  -> %.1f G FMA/s (warp-issue) | %.1f TMAC/s | %.1f TFLOPS\n",
         blocks, tpb, iters, NACC, ms, fma_warp_instr / (ms * 1e-3) / 1e9,
         fma_thread / (ms * 1e-3) / 1e12, flops / (ms * 1e-3) / 1e12);
  return 0;
}
