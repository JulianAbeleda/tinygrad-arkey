// HCQ-safe specialization of llama.cpp's Q6_K x Q8_1 MMVQ kernel.
// Fixed contract: K=4096, M=1, four warps, one vocabulary row per CTA.
#include "/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh"

extern "C" __global__ __launch_bounds__(128, 1)
void q6k_vocab_hcq(const void * weights, const void * q8, float * out, unsigned nrows) {
  constexpr int qk = 256;
  constexpr int qi = 32;
  constexpr int vdr = 1;
  constexpr int nwarps = 4;
  constexpr int warp_size = 32;
  constexpr int blocks_per_row = 4096 / qk;
  constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi;

  const unsigned row = blockIdx.x;
  if (row >= nrows) return;
  const int tid = warp_size * threadIdx.y + threadIdx.x;
  float sum = 0.0f;
  const block_q8_1 * y = static_cast<const block_q8_1 *>(q8);

  for (int kbx = tid / (qi / vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
    const int kby = kbx * (qk / QK8_1);
    const int kqs = vdr * (tid % (qi / vdr));
    sum += vec_dot_q6_K_q8_1(weights, &y[kby], row * blocks_per_row + kbx, kqs);
  }

  __shared__ float partial[nwarps - 1][warp_size];
  if (threadIdx.y > 0) partial[threadIdx.y - 1][threadIdx.x] = sum;
  __syncthreads();
  if (threadIdx.y > 0) return;

#pragma unroll
  for (int warp = 0; warp < nwarps - 1; ++warp) sum += partial[warp][threadIdx.x];
  sum = warp_reduce_sum<warp_size>(sum);
  if (threadIdx.x == 0) out[row] = sum;
}
