// dp4a_peak_cuda.cu - measured achievable dp4a peak (NVIDIA sm_120 / RTX 5090)
//
// The int8 analogue of mma_peak_cuda.cu, for llama.cpp's MMQ mechanism: Q4_K/Q6_K
// weights dequantized to q8_1 on the fly, then int8 dp4a dot products. Isolated
// back-to-back dp4a.s32.s32.s32.s32 on register-resident int8x4 operands: NACC
// independent accumulators, runtime trip count so the loop is not folded, and a
// never-taken store so the result stays live. Zero loads in the hot loop.
//
//   export PATH=/usr/local/cuda-13.2/bin:$PATH
//   nvcc -O3 -arch=sm_120 -DNACC=8 dp4a_peak_cuda.cu -o dp4a_peak_cuda && ./dp4a_peak_cuda 32768
//
// Verify purity before believing a number (cuobjdump --dump-sass): the hot loop
// contains only the dp4a SASS instruction (IMAD.I4 on sm_120) NACC times per
// unrolled body, zero LDG/LDS/STS, and exactly one gated STG sentinel.

#include <cuda_runtime.h>
#include <cstdio>

#ifndef NACC
#define NACC 8
#endif

__global__ __launch_bounds__(256) void dp4a_peak(int* out, int iters) {
  // int8x4 operands held in 32-bit registers; distinct A per accumulator so the
  // compiler cannot CSE the operands.
  unsigned a[NACC], b[NACC];
  #pragma unroll
  for (int j = 0; j < NACC; j++) {
    a[j] = (unsigned)(0x01020304 * (j + 1)) | 0x80000000u;   // keep signs mixed, never zero
    b[j] = (unsigned)(0x04030201 * (j + 1)) ^ 0x7fffffff;
  }

  int c[NACC];
  #pragma unroll
  for (int j = 0; j < NACC; j++) c[j] = 0;

  for (int t = 0; t < iters; t++) {
    #pragma unroll
    for (int j = 0; j < NACC; j++) {
      asm volatile(
        "dp4a.s32.s32 %0, %1, %2, %0;"
        : "+r"(c[j])
        : "r"(a[j]), "r"(b[j]));
    }
  }

  int s = 0;
  #pragma unroll
  for (int j = 0; j < NACC; j++) s += c[j];
  if (s == 1234567) out[0] = s;   // keep it live, never taken
}

int main(int argc, char** argv) {
  int blocks = argc > 1 ? atoi(argv[1]) : 2048;
  int iters = argc > 2 ? atoi(argv[2]) : 1600000 / NACC;   // roughly constant work per config
  int tpb = 256;
  int* d;
  cudaMalloc(&d, 4);
  dp4a_peak<<<blocks, tpb>>>(d, 1000);   // warmup (module load, clock ramp)
  cudaDeviceSynchronize();
  cudaEvent_t s, e;
  cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  dp4a_peak<<<blocks, tpb>>>(d, iters);
  cudaEventRecord(e);
  cudaDeviceSynchronize();
  float ms; cudaEventElapsedTime(&ms, s, e);
  double warps = (double)blocks * (tpb/32);
  double dp4a = warps * iters * NACC;                       // instructions
  double int8ops = dp4a * 8.0;                              // 4 MACs x 2 ops per dp4a
  printf("blocks=%d tpb=%d iters=%d nacc=%d  time=%.2f ms  -> %.1f G dp4a/s | %.1f INT8 TOPS | %.1f fp16-equiv TFLOPS\n",
         blocks, tpb, iters, NACC, ms, dp4a/(ms*1e-3)/1e9, int8ops/(ms*1e-3)/1e12, int8ops/(ms*1e-3)/1e12);
  return 0;
}
