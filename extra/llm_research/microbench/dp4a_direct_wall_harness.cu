// dp4a_direct_wall_harness.cu - minimal launch harness for the four-warp Q8/DP4A
// FFN-down GEMV (q4k_q8_mmvq_direct_4096_12288). Same geometry as the fp16 direct
// harness, so the only difference is the int8 DP4A datapath vs fp32 FMA.
//
//   nvcc -O3 -arch=sm_120a -o dp4a_direct_wall \
//     /tmp/q4k_dp4a_sum.cu extra/llm_research/microbench/dp4a_direct_wall_harness.cu
//   ./dp4a_direct_wall 2000         # wall time per 4096-row launch

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define ROWS 4096
#define K 12288
#define W ((size_t)ROWS * (K / 256) * 36)
#define Q8_WORDS ((size_t)K / 4 + K / 32)

extern "C" __global__ void q4k_q8_mmvq_direct_4096_12288(
  float* out, unsigned int* words, unsigned int* packed);

int main(int argc, char** argv) {
  int passes = argc > 1 ? atoi(argv[1]) : 200;
  float *out; unsigned int* words; unsigned int* packed;
  if (cudaMalloc(&out, ROWS * 4) != cudaSuccess) return 2;
  if (cudaMalloc(&words, W * 4) != cudaSuccess) return 4;
  if (cudaMalloc(&packed, Q8_WORDS * 4) != cudaSuccess) return 5;
  cudaMemset(out, 0, ROWS * 4);
  cudaMemset(words, 0, W * 4);
  cudaMemset(packed, 0, Q8_WORDS * 4);

  q4k_q8_mmvq_direct_4096_12288<<<ROWS, 128>>>(out, words, packed);
  if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "launch failed\n"); return 2; }
  cudaDeviceSynchronize();

  cudaEvent_t s, e;
  cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  for (int i = 0; i < passes; i++)
    q4k_q8_mmvq_direct_4096_12288<<<ROWS, 128>>>(out, words, packed);
  cudaEventRecord(e);
  cudaDeviceSynchronize();
  float ms; cudaEventElapsedTime(&ms, s, e);
  printf("passes=%d grid=%d block=%d  total=%.3f ms  per_launch=%.3f us  (%.1f GB/s weights)\n",
         passes, ROWS, 128, ms, ms * 1000.0 / passes, (double)W * 4 * passes / (ms * 1e-3) / 1e9);
  return 0;
}
