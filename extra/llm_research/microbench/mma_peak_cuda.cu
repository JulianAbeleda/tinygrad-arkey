// mma_peak_cuda.cu - measured achievable mma.sync peak (NVIDIA sm_120 / RTX 5090)
//
// The CUDA analogue of wmma_peak.cpp: isolated back-to-back mma.sync.aligned.m16n8k16
// (the shape tinygrad's cuda_sm89 tensor-core descriptor emits for fp16->fp32, see
// tinygrad/codegen/opt/tc.py get_cuda) on register-resident fragments: NACC independent
// accumulators to cover the mma dependency latency, runtime trip count so the loop is not
// folded, and a never-taken store so the result stays live. Zero loads in the hot loop.
//
//   nvcc -O3 -arch=sm_120 -DNACC=8 mma_peak_cuda.cu -o mma_peak_cuda && ./mma_peak_cuda
//
// Verify purity before believing a number (cuobjdump --dump-sass):
// HMMA.16816 appears NACC times per loop body, with zero LDG/LDS/STS inside the loop.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

#ifndef NACC
#define NACC 8
#endif

__global__ __launch_bounds__(256) void mma_peak(float* out, int iters) {
  // m16n8k16 f16->f32 fragments: A=16x16 f16 (4 b32 regs), B=16x8 f16 (2 b32), C=16x8 f32 (4 f32)
  unsigned a0, a1, a2, a3, b0, b1;
  __half2 ha[4], hb[2];
  #pragma unroll
  for (int i = 0; i < 4; i++) ha[i] = __floats2half2_rn(1.0f + 0.001f*(2*i), 1.0f + 0.001f*(2*i+1));
  #pragma unroll
  for (int i = 0; i < 2; i++) hb[i] = __floats2half2_rn(0.5f + 0.002f*(2*i), 0.5f + 0.002f*(2*i+1));
  a0 = *reinterpret_cast<unsigned*>(&ha[0]); a1 = *reinterpret_cast<unsigned*>(&ha[1]);
  a2 = *reinterpret_cast<unsigned*>(&ha[2]); a3 = *reinterpret_cast<unsigned*>(&ha[3]);
  b0 = *reinterpret_cast<unsigned*>(&hb[0]); b1 = *reinterpret_cast<unsigned*>(&hb[1]);

  float c[NACC][4];
  #pragma unroll
  for (int j = 0; j < NACC; j++) for (int i = 0; i < 4; i++) c[j][i] = 0.0f;

  for (int t = 0; t < iters; t++) {
    #pragma unroll
    for (int j = 0; j < NACC; j++) {
      asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
  }

  float s = 0.0f;
  #pragma unroll
  for (int j = 0; j < NACC; j++) for (int i = 0; i < 4; i++) s += c[j][i];
  if (s == 1234.5f) out[0] = s;   // keep it live, never taken
}

int main(int argc, char** argv) {
  int blocks = argc > 1 ? atoi(argv[1]) : 2048;
  int iters = argc > 2 ? atoi(argv[2]) : 1600000 / NACC;   // roughly constant mma work per config
  int tpb = 256;
  float* d;
  cudaMalloc(&d, 4);
  mma_peak<<<blocks, tpb>>>(d, 1000);   // warmup (module load, clock ramp)
  cudaDeviceSynchronize();
  cudaEvent_t s, e;
  cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  mma_peak<<<blocks, tpb>>>(d, iters);
  cudaEventRecord(e);
  cudaDeviceSynchronize();
  float ms; cudaEventElapsedTime(&ms, s, e);
  double warps = (double)blocks * (tpb/32);
  double flop = warps * iters * NACC * 2.0*16*8*16;   // 4096 FLOP per mma.m16n8k16
  printf("blocks=%d tpb=%d iters=%d nacc=%d  time=%.2f ms  -> %.1f TFLOPS\n",
         blocks, tpb, iters, NACC, ms, flop/(ms*1e-3)/1e12);
  return 0;
}
