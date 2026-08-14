// fp16_direct_wall_harness.cu - minimal launch harness for the four-warp fp16
// FFN-down GEMV (q4k_fp16_mmvq_direct_4096_12288), the isolated geometry lever
// from the FFN-down occupancy proof. Compiles alongside the rendered source.
//
//   nvcc -O3 -arch=sm_120a -o fp16_direct_wall \
//     /tmp/q4k_fp16_direct.cu extra/llm_research/microbench/fp16_direct_wall_harness.cu
//   ./fp16_direct_wall 1000          # wall time per 4096-row launch

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>

#define ROWS 4096
#define K 12288
#define W ((size_t)ROWS * (K / 256) * 36)

extern "C" __global__ void q4k_fp16_mmvq_direct_4096_12288(
  float* out, unsigned int* words, half* x);

int main(int argc, char** argv) {
  int passes = argc > 1 ? atoi(argv[1]) : 200;
  float *out; unsigned int* words; half* x;
  if (cudaMalloc(&out, ROWS * 4) != cudaSuccess) return 2;
  if (cudaMalloc(&words, W * 4) != cudaSuccess) return 4;
  if (cudaMalloc(&x, K * 2) != cudaSuccess) return 5;
  cudaMemset(out, 0, ROWS * 4);
  cudaMemset(words, 0, W * 4);
  cudaMemset(x, 0, K * 2);

  // warmup: module load + clock ramp
  q4k_fp16_mmvq_direct_4096_12288<<<ROWS, 128>>>(out, words, x);
  if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "launch failed\n"); return 2; }
  cudaDeviceSynchronize();

  cudaEvent_t s, e;
  cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  for (int i = 0; i < passes; i++)
    q4k_fp16_mmvq_direct_4096_12288<<<ROWS, 128>>>(out, words, x);
  cudaEventRecord(e);
  cudaDeviceSynchronize();
  float ms; cudaEventElapsedTime(&ms, s, e);
  printf("passes=%d grid=%d block=%d  total=%.3f ms  per_launch=%.3f us  (%.1f GB/s weights)\n",
         passes, ROWS, 128, ms, ms * 1000.0 / passes, (double)W * 4 * passes / (ms * 1e-3) / 1e9);
  return 0;
}
