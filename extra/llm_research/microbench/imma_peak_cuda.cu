// imma_peak_cuda.cu - measured achievable int8 mma.sync peak (NVIDIA sm_120 / RTX 5090)
//
// The int8 tensor-core twin of mma_peak_cuda.cu, and the number that says what a prompt-reading
// kernel is actually allowed to reach. mma_peak measures m16n8k16 f16->f32; this measures
// mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32, the instruction llama.cpp's CUDA library
// carries 38,880 times. dp4a_peak_cuda.cu measures the integer PIPE; this measures the integer
// TENSOR unit, and they are not the same thing or the same order of magnitude.
//
// Same regime as the f16 twin, so the two numbers are comparable: NACC independent accumulators
// to cover the mma dependency latency, register-resident fragments, runtime trip count so the
// loop is not folded, a never-taken store so the result stays live, zero loads in the hot loop.
//
//   nvcc -O3 -arch=sm_120 -DNACC=8 imma_peak_cuda.cu -o imma_peak && ./imma_peak
//
// Verify purity before believing a number (cuobjdump --dump-sass): IMMA appears NACC times per
// loop body, with zero LDG/LDS/STS inside the loop.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#ifndef NACC
#define NACC 8
#endif

__global__ __launch_bounds__(256) void imma_peak(int* out, int iters) {
  // m16n8k32 s8->s32 fragments: A=16x32 s8 (4 b32 regs), B=32x8 s8 (2 b32), C=16x8 s32 (4 s32)
  unsigned a0 = 0x01020304u, a1 = 0x05060708u, a2 = 0x090a0b0cu, a3 = 0x0d0e0f10u;
  unsigned b0 = 0x11121314u, b1 = 0x15161718u;

  int c[NACC][4];
  #pragma unroll
  for (int j = 0; j < NACC; j++) for (int i = 0; i < 4; i++) c[j][i] = 0;

  for (int t = 0; t < iters; t++) {
    #pragma unroll
    for (int j = 0; j < NACC; j++) {
      asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(c[j][0]), "+r"(c[j][1]), "+r"(c[j][2]), "+r"(c[j][3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
  }

  int s = 0;
  #pragma unroll
  for (int j = 0; j < NACC; j++) for (int i = 0; i < 4; i++) s += c[j][i];
  if (s == 1234567) out[0] = s;   // keep it live, never taken
}

int main(int argc, char** argv) {
  int blocks = argc > 1 ? atoi(argv[1]) : 2048;
  int iters = argc > 2 ? atoi(argv[2]) : 1600000 / NACC;   // same mma count per config as the f16 twin
  int tpb = 256;
  int* d;
  cudaMalloc(&d, 4);
  imma_peak<<<blocks, tpb>>>(d, 1000);   // warmup (module load, clock ramp)
  cudaDeviceSynchronize();
  cudaEvent_t s, e;
  cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  imma_peak<<<blocks, tpb>>>(d, iters);
  cudaEventRecord(e);
  cudaDeviceSynchronize();
  float ms; cudaEventElapsedTime(&ms, s, e);
  double warps = (double)blocks * (tpb/32);
  double ops = warps * iters * NACC * 2.0*16*8*32;   // 8192 int8 ops per mma.m16n8k32
  printf("blocks=%d tpb=%d iters=%d nacc=%d  time=%.2f ms  -> %.1f INT8 TOPS\n",
         blocks, tpb, iters, NACC, ms, ops/(ms*1e-3)/1e12);
  return 0;
}
