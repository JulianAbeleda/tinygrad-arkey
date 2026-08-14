// llama_mmq_wall_harness.cu - direct launch of llama's Q4_K mul_mat_vec_q
// cubin (the exact decode FFN-down shape) for a wall/occupancy apples-to-apples
// against the tinygrad FFN-down GEMV. Weight and Q8_1 activation are packed by
// the pinned llama CPU reference (see llama_cuda_quantized_live_oracle.py).
//
//   nvcc -O3 -arch=sm_120a -o llama_mmq_wall llama_mmq_wall_harness.cu
//   ./llama_mmq_wall <cubin> 200

#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

struct UInt3 { uint32_t x, y, z; };
struct FusionArgs { void* x_bias; void* gate; void* gate_bias; int32_t glu_op; };

static void check(CUresult e, const char* what) {
  if (e != CUDA_SUCCESS) { const char* s; cuGetErrorString(e, &s); fprintf(stderr, "%s: %s\n", what, s); exit(1); }
}

static UInt3 fastdiv(uint32_t d) {
  uint32_t level = 0; while (level < 32 && (1u << level) < d) level++;
  uint64_t mul = (((1ull << 32) * ((1ull << level) - d)) / d + 1) & 0xffffffffu;
  return { (uint32_t)mul, level, d };
}

int main(int argc, char** argv) {
  const char* cubin_path = argc > 1 ? argv[1] :
    "/home/ubuntu/tinygrad-arkey/scratchpad/llama_cuda_quantized_oracle_dump/libggml-cuda.so.0.14.36.sm_120a.cubin";
  int passes = argc > 2 ? atoi(argv[2]) : 200;
  const int nrows = 4096, k = 12288, qk = 256;
  const char* entry = "_Z13mul_mat_vec_qIL9ggml_type12ELi1ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj";

  CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
  check(cuInit(0), "cuInit"); check(cuDeviceGet(&dev, 0), "cuDeviceGet");
  check(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
  check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
  check(cuModuleLoad(&mod, cubin_path), "cuModuleLoad");
  check(cuModuleGetFunction(&fn, mod, entry), "cuModuleGetFunction");

  size_t wbytes = (size_t)nrows * (k / qk) * 144;
  size_t q8bytes = (size_t)k;  // 8-bit activation
  void *weight, *q8, *out;
  check(cuMemAlloc((CUdeviceptr*)&weight, wbytes), "weight");
  check(cuMemAlloc((CUdeviceptr*)&q8, q8bytes), "q8");
  check(cuMemAlloc((CUdeviceptr*)&out, (size_t)nrows * 4), "out");
  cuMemsetD8((CUdeviceptr)weight, 0, wbytes);
  cuMemsetD8((CUdeviceptr)q8, 0, q8bytes);
  cuMemsetD8((CUdeviceptr)out, 0, (size_t)nrows * 4);

  void* vx = weight;            // const void* vx_ptr (src0 quantized weights)
  void* vy = q8;                // const void* vy_ptr (src1 q8_1 staging)
  int32_t* ids = nullptr;       // const int32_t* ids_ptr
  FusionArgs fusion = {nullptr, nullptr, nullptr, 0};  // passed by value, 32 bytes
  float* outptr = (float*)out;
  uint32_t k32 = k;
  UInt3 zero = {0,0,0}, one = fastdiv(1);
  uint32_t row_blocks = k / qk, q8_blocks = k / 32, nrows32 = nrows;
  uint32_t nrb = nrows * row_blocks, qb = q8_blocks, z32 = 0;
  void* args[19] = {
    &vx, &vy, &ids, &fusion, &outptr, &k32, &zero, &row_blocks, &q8_blocks, &nrows32, &one,
    &nrb, &qb, &nrows32, &one, &nrb, &qb, &nrows32, &z32
  };

  // warmup
  check(cuLaunchKernel(fn, nrows, 1, 1, 32, 4, 1, 0, nullptr, args, nullptr), "launch warmup");
  cuCtxSynchronize();

  cudaEvent_t s, e;
  cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  for (int i = 0; i < passes; i++)
    check(cuLaunchKernel(fn, nrows, 1, 1, 32, 4, 1, 0, nullptr, args, nullptr), "launch");
  cudaEventRecord(e);
  cudaEventSynchronize(e);
  float ms; cudaEventElapsedTime(&ms, s, e);
  printf("passes=%d grid=%d block=32x4  total=%.3f ms  per_launch=%.3f us  (%.1f GB/s weights)\n",
         passes, nrows, ms, ms * 1000.0 / passes, (double)wbytes * passes / (ms * 1e-3) / 1e9);
  return 0;
}
