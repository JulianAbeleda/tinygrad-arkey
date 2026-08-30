#include "/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh"
extern "C" __global__ void q6k_vocab_hcq(const void *w, const void *q8, float *out, unsigned nrows) {
  const unsigned row=blockIdx.x;
  if (row>=nrows) return;
  float sum=0.0f;
  const block_q8_1 *y=(const block_q8_1 *)q8;
  const char *x=(const char *)w + row*210;
  for (int k=0;k<128;k++) sum += vec_dot_q6_K_q8_1(x, y, 0, 0);
  if (threadIdx.x==0) out[row]=sum;
}
