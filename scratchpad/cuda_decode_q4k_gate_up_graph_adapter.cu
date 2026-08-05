#include <cuda_fp16.h>

extern "C" __global__ void half_to_float_4096(const __half *in, float *out) {
  unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < 4096) out[i] = __half2float(in[i]);
}

extern "C" __global__ void float_to_half_12288(const float *in, __half *out) {
  unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < 12288) out[i] = __float2half_rn(in[i]);
}
