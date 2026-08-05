#include <cuda_fp16.h>
extern "C" __global__ void half_to_float(const __half *in, float *out, unsigned n) {
  unsigned i=blockIdx.x*blockDim.x+threadIdx.x; if (i<n) out[i]=__half2float(in[i]);
}
extern "C" __global__ void scatter_to_partials(const float *in, float *out, unsigned n) {
  unsigned i=blockIdx.x*blockDim.x+threadIdx.x; if (i<4*n) out[i]=(i&3) ? 0.0f : in[i>>2];
}
