// bw_peak_cuda.cu - measured achievable streaming bandwidth (NVIDIA sm_120 / RTX 5090)
//
// The BW analogue of mma_peak_cuda.cu, per docs/bringing-up-a-new-target-20260731.md
// Phase 0: a streaming benchmark at the sizes actually used, zero math in the hot loop.
// Three modes:
//   read  - grid-stride float4 loads, sum kept live via a never-taken store
//   write - grid-stride float4 streaming stores
//   copy  - read float4 from src, store to dst (the 2x-traffic case)
//
//   nvcc -O3 -arch=sm_120 bw_peak_cuda.cu -o bw_peak_cuda
//   ./bw_peak_cuda read  4294967296 32
//   ./bw_peak_cuda copy  4294967296 16
//
// Verify purity before believing a number (cuobjdump --dump-sass): the read hot loop
// contains only LDG/FFMA (plus one gated STG sentinel), write only STG, copy LDG+STG.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#ifndef TPB
#define TPB 256
#endif

__global__ __launch_bounds__(TPB) void read_peak(const float4* __restrict__ src,
                                                 float* out, long long n4) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long stride = (long long)gridDim.x * blockDim.x;
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  for (; i < n4; i += stride) {
    float4 v = __ldcs(src + i);
    acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
  }
  float s = acc.x + acc.y + acc.z + acc.w;
  if (s == 1234.5f) out[0] = s;   // keep it live, never taken
}

__global__ __launch_bounds__(TPB) void write_peak(float4* __restrict__ dst, long long n4) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long stride = (long long)gridDim.x * blockDim.x;
  float4 v = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
  for (; i < n4; i += stride) __stcs(dst + i, v);
}

__global__ __launch_bounds__(TPB) void copy_peak(const float4* __restrict__ src,
                                                 float4* __restrict__ dst, long long n4) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long stride = (long long)gridDim.x * blockDim.x;
  for (; i < n4; i += stride) __stcs(dst + i, __ldcs(src + i));
}

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "read";
  long long bytes = argc > 2 ? atoll(argv[2]) : 1LL << 32;   // default 4 GiB
  int passes = argc > 3 ? atoi(argv[3]) : 32;
  int blocks = argc > 4 ? atoi(argv[4]) : 4096;
  bool is_copy = strcmp(mode, "copy") == 0;
  long long n4 = bytes / 16;
  if (bytes % 16 != 0 || n4 <= 0) { fprintf(stderr, "bytes must be a positive multiple of 16\n"); return 1; }

  float4* src = nullptr; float4* dst = nullptr;
  float* out = nullptr;
  if (is_copy || strcmp(mode, "read") == 0) { if (cudaMalloc(&src, bytes) != cudaSuccess) return 2; }
  if (is_copy || strcmp(mode, "write") == 0) { if (cudaMalloc(&dst, bytes) != cudaSuccess) return 3; }
  cudaMalloc(&out, 4);

  if (strcmp(mode, "read") == 0) {
    read_peak<<<blocks, TPB>>>(src, out, n4);
  } else if (strcmp(mode, "write") == 0) {
    write_peak<<<blocks, TPB>>>(dst, n4);
  } else if (is_copy) {
    copy_peak<<<blocks, TPB>>>(src, dst, n4);
  } else {
    fprintf(stderr, "mode must be read|write|copy\n"); return 1;
  }
  cudaDeviceSynchronize();   // warmup (module load, clock ramp, TLB)

  cudaEvent_t s, e;
  cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  for (int p = 0; p < passes; p++) {
    if (strcmp(mode, "read") == 0) read_peak<<<blocks, TPB>>>(src, out, n4);
    else if (strcmp(mode, "write") == 0) write_peak<<<blocks, TPB>>>(dst, n4);
    else copy_peak<<<blocks, TPB>>>(src, dst, n4);
  }
  cudaEventRecord(e);
  cudaDeviceSynchronize();
  float ms; cudaEventElapsedTime(&ms, s, e);
  double traffic = (double)passes * bytes * (is_copy ? 2.0 : 1.0);
  double gbps = traffic / (ms * 1e-3) / 1e9;
  printf("mode=%s bytes=%.3f GiB passes=%d blocks=%d  time=%.2f ms  -> %.1f GB/s (%.1f%% of 1792 GB/s sheet)\n",
         mode, bytes / 1073741824.0, passes, blocks, ms, gbps, 100.0 * gbps / 1792.0);
  return 0;
}
