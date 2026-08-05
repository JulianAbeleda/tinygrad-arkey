#include <cuda_fp16.h>
extern "C" __global__ void half_to_float(const __half *in, float *out, unsigned n) {
  unsigned i=blockIdx.x*blockDim.x+threadIdx.x; if (i<n) out[i]=__half2float(in[i]);
}
extern "C" __global__ void scatter_to_16_partials(const float *in, float *out, unsigned n) {
  unsigned i=blockIdx.x*blockDim.x+threadIdx.x; if (i<16*n) out[i]=(i&15) ? 0.0f : in[i>>4];
}
