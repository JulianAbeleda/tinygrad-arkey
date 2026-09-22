// q6k_vocab_coop_ceiling_cuda.cu - L4 Scope B Stage 1: values ceiling of the landed Q6_K vocab
// coop head at row_tile=2 (NV sm_120 / RTX 5090). wmma_peak-style method per
// docs/task_workflow/input/decode-gemv-efficiency-forward-scope-20260803.md section 4:
// isolate the steady-state reduction loop with hoisted operand setup and accumulator staging,
// sweep the knob, and inspect the rendered source to verify purity before believing a number.
//
// The installed kernel (q6k_gen_coop_151936_4096, row_tile=2) measured 330.1 us / 1.55 TB/s
// (86% of the 1792 GB/s sheet) in decode; llama's single mmq vocab kernel is 303.75 us on the
// same 151936x4096 shape. This probe replicates the coop inner loop EXACTLY (same Q6_K byte
// extraction, same 16 pos lanes x row_tile=2 rows per warp, same 16 k_blocks reduction, same
// (N,16) partial output) and sweeps the remaining values surface at row_tile=2:
//   NACC       - independent accumulators over the 16 k_blocks (accumulator staging)
//   ROW_GROUPS - row-groups per block (1 = the installed 32-thread block; 2/4 group more
//                warps per block, total warp count fixed at 75968 = occupancy knob)
//   XSH        - 0: x read from global each k_block (installed behavior, L2-resident);
//                1: x hoisted into shared memory once per block (hoisted operand setup)
// One launch == one full vocab pass (grid 75968/ROW_GROUPS blocks x 32*ROW_GROUPS threads);
// the host loops `passes` launches under one CUDA event pair, exactly like bw_peak_cuda.cu.
//
//   export PATH=/usr/local/cuda-13.2/bin:$PATH
//   nvcc -O3 -arch=sm_120 q6k_vocab_coop_ceiling_cuda.cu -o q6k_vocab_coop_ceiling_cuda
//   ./q6k_vocab_coop_ceiling_cuda <NACC> <ROW_GROUPS> <XSH> <passes>
//
// Verify purity before believing a number (cuobjdump --dump-sass): the k_blocks hot loop must
// contain only the Q6_K byte-extraction ALU + u16 weight loads + one FMA per group, zero LDS
// at XSH=0, and exactly one gated STG sentinel. 0 spills at every config. The reference row
// (NACC=1, ROW_GROUPS=1, XSH=0) must land at the installed kernel's time (~330 us).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>

#define ROWS 151936
#define K 4096
#define K_BLOCKS (K / 256)          // 16 Q6_K blocks per row
#define HALFS_PER_BLOCK 105
#define ROW_TILE 2
#define POS 16                      // lane_extent (Q6K_POS_EXTENT)

// exact port of decode_kernels.py _q6k_byte/_q6k_weight: same byte reads, shifts, and casts.
__device__ __forceinline__ float q6k_weight(const unsigned short* __restrict__ halfs,
                                            int base, int grp, int pos) {
  int half = grp / 8, pgrp = grp % 8;
  int ql_byte_idx = half * 64 + (pgrp % 4) * 16 + pos;
  int qh_byte_idx = 128 + half * 32 + (pgrp % 2) * 16 + pos;
  int ql_shift = (pgrp >= 4) ? 4 : 0;
  int qh_shift = (pgrp / 2) * 2;
  unsigned short ql_u16 = halfs[base + ql_byte_idx / 2];
  unsigned short qh_u16 = halfs[base + qh_byte_idx / 2];
  int ql = (ql_u16 >> ((ql_byte_idx % 2) * 8) >> ql_shift) & 0xf;
  int qh = ((qh_u16 >> ((qh_byte_idx % 2) * 8) >> qh_shift) & 0x3) << 4;
  int q = (ql | qh) - 32;
  float scale = __half2float(__ushort_as_half(halfs[base + 104]));
  unsigned short d_u16 = halfs[base + (192 + grp) / 2];
  int d_byte = (d_u16 >> (((192 + grp) % 2) * 8)) & 0xff;
  return scale * (float)q * (float)(signed char)d_byte;
}

// One warp per row-group: 32 lanes = ROW_TILE rows x POS pos; lane map matches the emitter
// (tid = pos*ROW_TILE + row_i -> row_i fastest). x index per emitter: x[blk*256 + grp*16 + pos].
template <int nacc, int row_groups, int xsh>
__global__ __launch_bounds__(ROW_TILE * POS * row_groups) void coop_ceiling(
    float* __restrict__ out, const unsigned short* __restrict__ halfs,
    const half* __restrict__ x_g, float* __restrict__ sentinel) {
  extern __shared__ half x_s[];
  int tid = threadIdx.x;
  int warp = tid / (ROW_TILE * POS);
  int lane = tid % (ROW_TILE * POS);
  int row_i = lane % ROW_TILE;
  int pos = lane / ROW_TILE;
  int row_o = blockIdx.x * row_groups + warp;
  if (xsh) {
    for (int i = tid; i < K; i += blockDim.x) x_s[i] = x_g[i];
    __syncthreads();
  }
  const half* __restrict__ xr = xsh ? x_s : x_g;

  int row = row_o * ROW_TILE + row_i;
  float a[nacc];
  #pragma unroll
  for (int j = 0; j < nacc; j++) a[j] = 0.0f;
  #pragma unroll 2
  for (int blk = 0; blk < K_BLOCKS; blk++) {
    int base = (row * K_BLOCKS + blk) * HALFS_PER_BLOCK;
    #pragma unroll
    for (int grp = 0; grp < 16; grp++) {
      float w = q6k_weight(halfs, base, grp, pos);
      float xv = __half2float(xr[blk * 256 + grp * 16 + pos]);
      a[grp % nacc] += w * xv;
    }
  }
  float s = 0.0f;
  #pragma unroll
  for (int j = 0; j < nacc; j++) s += a[j];
  // nacc==1 keeps the installed serial-reduce order; nacc>1 changes only the summation order
  // of the 16 group products (fp32 reassociation), which is what "accumulator staging" means.
  int out_idx = row * POS + pos;
  out[out_idx] = s;                               // (N,16) partial, like the installed kernel
  if (s == 1234.5f) sentinel[0] = s;              // never taken
}

template <int nacc, int row_groups, int xsh>
static void launch(dim3 grid, dim3 block, size_t smem, float* out, const unsigned short* h,
                   const half* xg, float* sentinel) {
  coop_ceiling<nacc,row_groups,xsh><<<grid, block, smem>>>(out, h, xg, sentinel);
}

int main(int argc, char** argv) {
  int nacc = argc > 1 ? atoi(argv[1]) : 1;
  int row_groups = argc > 2 ? atoi(argv[2]) : 1;
  int xsh = argc > 3 ? atoi(argv[3]) : 0;
  int passes = argc > 4 ? atoi(argv[4]) : 16;

  size_t halfs_n = (size_t)ROWS * K_BLOCKS * HALFS_PER_BLOCK;   // 510.5 MB
  size_t out_n = (size_t)ROWS * POS;
  unsigned short* h; half* xg; float* out; float* sentinel;
  if (cudaMalloc(&h, halfs_n * 2) != cudaSuccess) return 2;
  if (cudaMalloc(&xg, K * 2) != cudaSuccess) return 3;
  if (cudaMalloc(&out, out_n * 4) != cudaSuccess) return 4;
  if (cudaMalloc(&sentinel, 4) != cudaSuccess) return 5;
  cudaMemset(h, 0x37, halfs_n * 2);              // deterministic nonzero bytes
  cudaMemset(xg, 0x3f, K * 2);
  cudaMemset(out, 0, out_n * 4);

  // One 32-lane warp covers ROW_TILE=2 rows (2 lanes x 16 pos), so the grid is
  // ROWS/ROW_TILE = 75968 blocks / row_groups (NOT ROWS/32 - that launches 1/16 of
  // the work and reads 330us down to ~20.5us, the bug this line guards against).
  int warps = ROWS / ROW_TILE / row_groups;                     // 75968 / row_groups
  int tpb = ROW_TILE * POS * row_groups;
  dim3 grid(warps), block(tpb);
  size_t smem = xsh ? K * 2 : 0;

  void (*timed)(dim3,dim3,size_t,float*,const unsigned short*,const half*,float*) = nullptr;
  if (nacc == 1 && row_groups == 1 && xsh == 0) timed = launch<1,1,0>;
  else if (nacc == 1 && row_groups == 1 && xsh == 1) timed = launch<1,1,1>;
  else if (nacc == 2 && row_groups == 1 && xsh == 1) timed = launch<2,1,1>;
  else if (nacc == 4 && row_groups == 1 && xsh == 1) timed = launch<4,1,1>;
  else if (nacc == 8 && row_groups == 1 && xsh == 1) timed = launch<8,1,1>;
  else if (nacc == 16 && row_groups == 1 && xsh == 1) timed = launch<16,1,1>;
  else if (nacc == 8 && row_groups == 2 && xsh == 1) timed = launch<8,2,1>;
  else if (nacc == 8 && row_groups == 4 && xsh == 1) timed = launch<8,4,1>;
  else if (nacc == 8 && row_groups == 2 && xsh == 0) timed = launch<8,2,0>;
  else if (nacc == 8 && row_groups == 4 && xsh == 0) timed = launch<8,4,0>;
  else if (nacc == 2 && row_groups == 1 && xsh == 0) timed = launch<2,1,0>;
  else if (nacc == 4 && row_groups == 1 && xsh == 0) timed = launch<4,1,0>;
  else if (nacc == 16 && row_groups == 1 && xsh == 0) timed = launch<16,1,0>;
  else { fprintf(stderr, "unsupported config nacc=%d row_groups=%d xsh=%d\n", nacc, row_groups, xsh); return 1; }

  for (int i = 0; i < 2; i++) timed(grid, block, smem, out, h, xg, sentinel);   // warmup
  if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "launch failed: %s\n", cudaGetErrorString(cudaGetLastError())); return 6; }
  cudaDeviceSynchronize();

  cudaEvent_t s, e;
  cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  for (int i = 0; i < passes; i++) timed(grid, block, smem, out, h, xg, sentinel);
  cudaEventRecord(e);
  cudaDeviceSynchronize();
  float ms; cudaEventElapsedTime(&ms, s, e);
  double us_per_pass = ms * 1000.0 / passes;
  double bytes = (double)ROWS * K_BLOCKS * HALFS_PER_BLOCK * 2;   // 510.5 MB weights
  double gbps = bytes / (us_per_pass * 1e-6) / 1e9;
  printf("nacc=%d row_groups=%d xsh=%d passes=%d  %.2f us/pass  %.0f GB/s  (%.1f%% of 1792 GB/s)\n",
         nacc, row_groups, xsh, passes, us_per_pass, gbps, 100.0 * gbps / 1792.0);
  return 0;
}
