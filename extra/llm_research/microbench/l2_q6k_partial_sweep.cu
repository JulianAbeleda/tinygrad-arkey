// l2_q6k_partial_sweep.cu - L2 Q6K partial single-pass diagnostic (scope A, NV sm_120)
//
// Wmma_peak-style characterization of the in-kernel reduce on the fixed parts=4 packed
// storage of `q6k_gen_partial_1024_4096_4` (1024 rows x 4096 k, 16 blocks/row, 4 blocks
// per part, Q6K_HALFWORDS_PER_BLOCK=105). The knob is the thread decomposition of the
// in-kernel part merge over the same packed layout (no load-time repack): rows per block
// x threads per part (split-reduce) x part lanes, covering at minimum the two recorded
// shapes (4-thread part blocks; 8-row x 4-part 32-thread blocks) plus 16-row variants
// and split-reduce variants.
//
// Three probe modes, all following the wmma_peak discipline (operand setup hoisted,
// multiple independent accumulators, runtime trip count, never-taken keep-alive store,
// inspect the rendered source before believing a number):
//
//   mem    - faithful reproduction: real packed-storage loads (identical byte layout and
//            per-thread work to the installed kernel), per-decomposition launch config,
//            in-kernel XOR ladder + gated store for the merged variants, partials store
//            for the legacy external_sum shape. This is the go/no-go evidence (us per
//            pass, TB/s) and the mandatory control reproduction (17.15 / 25.3 / 466.6 us).
//   dot    - pure steady-state dot chain on register-resident fp32 operands, NACC
//            independent accumulators, zero loads in the hot loop: the ALU ceiling.
//   dequant- same, but the full Q6K dequant+FMA chain runs on register-resident packed
//            bytes each iteration: the realistic ALU+dequant instruction mix, zero loads.
//
//   nvcc -O3 -arch=sm_120 -std=c++17 l2_q6k_partial_sweep.cu -o l2_q6k_partial_sweep
//   ./l2_q6k_partial_sweep --mode mem
//   ./l2_q6k_partial_sweep --mode dot --nacc 8
//
// Verify purity (cuobjdump --dump-sass): `mem` hot loop contains the same LDG mix as the
// installed kernel plus SHFL ladder + gated STG (no LDS/STS, 0 spills); `dot`/`dequant`
// hot loops contain zero LDG/LDS/STS and exactly one gated STG sentinel.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>

#define ROWS 1024
#define K 4096
#define PARTS 4
#define KB 16            // blocks per row (K / 256)
#define BPP (KB / PARTS) // 4 blocks per part
#define HWPB 105         // halfwords per Q6_K block (210 bytes)
#define POSN 16          // Q6K_POS_EXTENT
#define GRPS 16

#ifndef NACC
#define NACC 8
#endif

// ---- Q6_K packed dequant grammar, byte-identical to tinygrad's `_q6k_weight`
// (decode_kernels.py): ql bytes 0..127, qh bytes 128..191, int8 scales 192..207, fp16 d at
// halfword 104. The per-pos window of 12 halfwords mirrors the installed render exactly.
__device__ __forceinline__ unsigned char q6k_byte(const unsigned short* halfs, int base, int idx) {
  return (unsigned char)((halfs[base + (idx >> 1)] >> ((idx & 1) << 3)) & 0xff);
}

__device__ __forceinline__ float q6k_weight(const unsigned short* halfs, int base, int grp, int pos) {
  int half = grp >> 3, pgrp = grp & 7;
  int qlb = half*64 + (pgrp & 3)*16 + pos, qls = (pgrp >= 4) ? 4 : 0;
  int qhb = 128 + half*32 + (pgrp & 1)*16 + pos, qhs = (pgrp >> 1) * 2;
  unsigned char ql = (q6k_byte(halfs, base, qlb) >> qls) & 0xf;
  unsigned char qh = (q6k_byte(halfs, base, qhb) >> qhs) & 0x3;
  float q = (float)((int)(ql | (qh << 4)) - 32);
  float d = __half2float(__ushort_as_half(halfs[base + 104]));
  float sc = (float)(signed char)q6k_byte(halfs, base, 192 + grp);
  return d * q * sc;
}

// Per-block hoisted context, mirroring the installed render (9 halfwords 96..104 loaded
// once per block: 8 int8 scale pairs + the fp16 d at halfword 104).
struct Q6KBlockCtx { float d; float sc[GRPS]; };

__device__ __forceinline__ Q6KBlockCtx q6k_block_ctx(const unsigned short* halfs, int base) {
  Q6KBlockCtx c;
  c.d = __half2float(__ushort_as_half(halfs[base + 104]));
#pragma unroll
  for (int grp = 0; grp < GRPS; grp++) {
    unsigned short w = halfs[base + 96 + (grp >> 1)];
    c.sc[grp] = (float)(signed char)((w >> ((grp & 1) << 3)) & 0xff);
  }
  return c;
}

// Full block reduce for one (blk, pos): 16 groups. Mirrors the installed kernel exactly:
// direct per-halfword loads with compile-time offsets after pos unroll (no window array),
// scales/d from the hoisted ctx. ptxas CSEs the 12 window halfwords across the 16 groups.
__device__ __forceinline__ float q6k_block_dot(Q6KBlockCtx c, const unsigned short* halfs,
                                               const __half* x, int base, int blk, int pos) {
  float acc = 0.0f;
#pragma unroll
  for (int grp = 0; grp < GRPS; grp++) {
    int half = grp >> 3, pgrp = grp & 7;
    int qlb = half*64 + (pgrp & 3)*16 + pos, qls = (pgrp >= 4) ? 4 : 0;
    int qhb = 128 + half*32 + (pgrp & 1)*16 + pos, qhs = (pgrp >> 1) * 2;
    unsigned char ql = (q6k_byte(halfs, base, qlb) >> qls) & 0xf;
    unsigned char qh = (q6k_byte(halfs, base, qhb) >> qhs) & 0x3;
    float q = (float)((int)(ql | (qh << 4)) - 32);
    acc += c.d * q * c.sc[grp] * __half2float(x[blk*256 + grp*16 + pos]);
  }
  return acc;
}

// ---- legacy installed row (external_sum, LOCAL:0:32): grid (PARTS, ROWS/32) x 32 threads,
// one (row, part) per thread, serial 4-block x 16-pos reduce, partials store.
__global__ void __launch_bounds__(32) k_legacy(const unsigned short* __restrict__ halfs,
                                               const __half* __restrict__ x,
                                               float* __restrict__ partials, int iters) {
  int part = blockIdx.x;      // 4
  int rowg = blockIdx.y;      // ROWS/32
  int rowl = threadIdx.x;     // 32
  for (int t = 0; t < iters; t++) {
  int row = (rowg*32 + rowl + t) & (ROWS - 1);   // rotate so loads cannot be hoisted
  float acc = 0.0f;
  for (int b = 0; b < BPP; b++) {
    int blk = part*BPP + b;
    int base = (row*KB + blk) * HWPB;
    Q6KBlockCtx ctx = q6k_block_ctx(halfs, base);
    for (int pos = 0; pos < POSN; pos++) acc += q6k_block_dot(ctx, halfs, x, base, blk, pos);
  }
    partials[row*PARTS + part] = acc;
  }
}

// ---- in-kernel merge variants: R rows x (4 parts x S threads per part) per block.
// lane = part*S + split within each row's 4*S-lane group; intra-part ladder over the S
// split lanes, then inter-part ladder over the 4 part lanes; gated store on lane 0.
template<int R, int S>
__global__ void __launch_bounds__(R*4*S) k_merge(const unsigned short* __restrict__ halfs,
                                                 const __half* __restrict__ x,
                                                 float* __restrict__ out, int iters) {
  int tid = threadIdx.x;
  int row_local = tid / (4*S);
  int lane = tid % (4*S);
  int part = lane / S, split = lane % S;
  unsigned mask = (blockDim.x < 32) ? ((1u << blockDim.x) - 1u) : 0xffffffffu;
  for (int t = 0; t < iters; t++) {
    int row = (blockIdx.x*R + row_local + t) & (ROWS - 1);
    float acc = 0.0f;
    for (int b = 0; b < BPP/S; b++) {
      int blk = part*BPP + split*(BPP/S) + b;
      int base = (row*KB + blk) * HWPB;
      Q6KBlockCtx ctx = q6k_block_ctx(halfs, base);
      for (int pos = 0; pos < POSN; pos++) acc += q6k_block_dot(ctx, halfs, x, base, blk, pos);
    }
    for (int off = S>>1; off >= 1; off >>= 1) acc += __shfl_xor_sync(mask, acc, off);
    acc += __shfl_xor_sync(mask, acc, 2*S);
    acc += __shfl_xor_sync(mask, acc, S);
    if (lane == 0) out[row] = acc;
  }
}

// ---- pure steady-state dot chain: fp32 operands hoisted once from a per-thread runtime
// buffer (dp4a_peak structure), NACC independent accumulators, zero loads in the hot loop.
template<int R, int S>
__global__ void __launch_bounds__(R*4*S) k_dot_peak(float* __restrict__ out,
                                                    const float* __restrict__ ops, int iters) {
  float w[16], xr[16], c[NACC];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const float* op = ops + (size_t)(tid % 4096) * 32;   // distinct per-thread values, hoisted
#pragma unroll
  for (int i = 0; i < 16; i++) { w[i] = op[i]; xr[i] = op[16 + i]; }
#pragma unroll
  for (int j = 0; j < NACC; j++) c[j] = 0.0f;
  for (int t = 0; t < iters; t++) {
#pragma unroll
    for (int j = 0; j < NACC; j++)
#pragma unroll
      for (int i = 0; i < 16; i++) c[j] += w[i] * xr[i];
  }
  float s = 0.0f;
#pragma unroll
  for (int j = 0; j < NACC; j++) s += c[j];
  if (s == 1234.5f) out[0] = s;   // never-taken keep-alive
}

// ---- dequant+FMA chain on register-resident packed bytes: one 16-pos group per
// accumulator, full Q6K weight math per iteration, zero loads in the hot loop. Operands
// (ql/qh/sc/d/xr) are hoisted once from a per-thread runtime buffer so the chain cannot
// fold to compile-time constants.
template<int R, int S>
__global__ void __launch_bounds__(R*4*S) k_dequant_peak(float* __restrict__ out,
                                                        const float* __restrict__ ops, int iters) {
  const unsigned* uops = reinterpret_cast<const unsigned*>(ops);
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t o = (size_t)(tid % 1024) * 256;   // per-thread 256-u32 chunk, hoisted
  unsigned ql[NACC][4], qh[NACC][4];
  float sc[NACC], d[NACC], c[NACC];
  __half2 xr[NACC][8];
#pragma unroll
  for (int j = 0; j < NACC; j++) {
    c[j] = 0.0f;
    d[j] = __uint_as_float(uops[o + 72 + j]);
    sc[j] = (float)(signed char)(uops[o + 64 + j] & 0xff);
    for (int q = 0; q < 4; q++) {
      ql[j][q] = uops[o + j*4 + q];
      qh[j][q] = uops[o + 32 + j*4 + q];
    }
    for (int i = 0; i < 8; i++) xr[j][i] = *(const __half2*)(uops + o + 80 + (j*8 + i)*2);
  }
  // grp layout chosen so shifts are constants: pgrp=0 -> qls=0, qhs=0; half=0.
  for (int t = 0; t < iters; t++) {
#pragma unroll
    for (int j = 0; j < NACC; j++) {
#pragma unroll
      for (int pos = 0; pos < 16; pos++) {
        unsigned char qlb = (unsigned char)((ql[j][pos >> 2] >> ((pos & 3) << 3)) & 0xff);
        unsigned char qhb = (unsigned char)((qh[j][pos >> 2] >> ((pos & 3) << 3)) & 0xff);
        float q = (float)((int)(((qlb & 0xf) | ((qhb & 0x3) << 4))) - 32);
        __half2 h = xr[j][pos >> 1];
        float xv = (pos & 1) ? __half2float(h.y) : __half2float(h.x);
        c[j] += d[j] * q * sc[j] * xv;
      }
    }
  }
  float s = 0.0f;
#pragma unroll
  for (int j = 0; j < NACC; j++) s += c[j];
  if (s == 1234.5f) out[0] = s;
}

struct Cfg { const char* name; int r; int s; int blocks; int thr; bool legacy; };

static const Cfg CFGS[] = {
  {"legacy_32",        32, 1, 4*32,          32,  true},  // installed LOCAL:0:32 (grid x=parts, y=row-groups)
  {"p4_4thr",           1, 1, ROWS,           4, false},  // 4-thread part blocks (M2 control, 25.3 us)
  {"r8p4_32thr",        8, 1, ROWS/8,        32, false},  // 8-row x 4-part 32-thread (M2 anomaly, 466.6 us)
  {"r16p4_64thr",      16, 1, ROWS/16,       64, false},  // 16-row variant
  {"r32p4_128thr",     32, 1, ROWS/32,      128, false},  // 32-row variant
  {"r8_split2_64thr",   8, 2, ROWS/8,        64, false},  // split-reduce: 2 threads per part
  {"r16_split2_128thr",16, 2, ROWS/16,      128, false},
  {"r8_split4_128thr",  8, 4, ROWS/8,       128, false},  // split-reduce: 4 threads per part
  {"r16_split4_256thr",16, 4, ROWS/16,      256, false},
};
static const int NCFGS = sizeof(CFGS) / sizeof(CFGS[0]);

static void launch_mem(const Cfg& c, const unsigned short* halfs, const __half* x,
                       float* out, int iters, cudaStream_t st) {
  if (c.legacy) k_legacy<<<dim3(PARTS, ROWS/32), 32, 0, st>>>(halfs, x, out, iters);
  else if (c.r == 1 && c.s == 1) k_merge<1,1><<<c.blocks, c.thr, 0, st>>>(halfs, x, out, iters);
  else if (c.r == 8 && c.s == 1) k_merge<8,1><<<c.blocks, c.thr, 0, st>>>(halfs, x, out, iters);
  else if (c.r == 16 && c.s == 1) k_merge<16,1><<<c.blocks, c.thr, 0, st>>>(halfs, x, out, iters);
  else if (c.r == 32 && c.s == 1) k_merge<32,1><<<c.blocks, c.thr, 0, st>>>(halfs, x, out, iters);
  else if (c.r == 8 && c.s == 2) k_merge<8,2><<<c.blocks, c.thr, 0, st>>>(halfs, x, out, iters);
  else if (c.r == 16 && c.s == 2) k_merge<16,2><<<c.blocks, c.thr, 0, st>>>(halfs, x, out, iters);
  else if (c.r == 8 && c.s == 4) k_merge<8,4><<<c.blocks, c.thr, 0, st>>>(halfs, x, out, iters);
  else if (c.r == 16 && c.s == 4) k_merge<16,4><<<c.blocks, c.thr, 0, st>>>(halfs, x, out, iters);
}

static void launch_compute(const Cfg& c, float* out, const float* ops, int iters,
                           bool dequant, cudaStream_t st) {
  if (dequant) {
    if (c.legacy) k_dequant_peak<32,1><<<c.blocks, c.thr, 0, st>>>(out, ops, iters);
    else if (c.r == 1) k_dequant_peak<1,1><<<c.blocks, c.thr, 0, st>>>(out, ops, iters);
    else if (c.r == 8 && c.s == 1) k_dequant_peak<8,1><<<c.blocks, c.thr, 0, st>>>(out, ops, iters);
    else if (c.r == 16 && c.s == 1) k_dequant_peak<16,1><<<c.blocks, c.thr, 0, st>>>(out, ops, iters);
    else if (c.r == 32 && c.s == 1) k_dequant_peak<32,1><<<c.blocks, c.thr, 0, st>>>(out, ops, iters);
    else if (c.r == 8 && c.s == 2) k_dequant_peak<8,2><<<c.blocks, c.thr, 0, st>>>(out, ops, iters);
    else if (c.r == 16 && c.s == 2) k_dequant_peak<16,2><<<c.blocks, c.thr, 0, st>>>(out, ops, iters);
    else if (c.r == 8 && c.s == 4) k_dequant_peak<8,4><<<c.blocks, c.thr, 0, st>>>(out, ops, iters);
    else k_dequant_peak<16,4><<<c.blocks, c.thr, 0, st>>>(out, ops, iters);
  } else {
    if (c.legacy) k_dot_peak<32,1><<<c.blocks, c.thr, 0, st>>>(out, ops, iters);
    else if (c.r == 1) k_dot_peak<1,1><<<c.blocks, c.thr, 0, st>>>(out, ops, iters);
    else if (c.r == 8 && c.s == 1) k_dot_peak<8,1><<<c.blocks, c.thr, 0, st>>>(out, ops, iters);
    else if (c.r == 16 && c.s == 1) k_dot_peak<16,1><<<c.blocks, c.thr, 0, st>>>(out, ops, iters);
    else if (c.r == 32 && c.s == 1) k_dot_peak<32,1><<<c.blocks, c.thr, 0, st>>>(out, ops, iters);
    else if (c.r == 8 && c.s == 2) k_dot_peak<8,2><<<c.blocks, c.thr, 0, st>>>(out, ops, iters);
    else if (c.r == 16 && c.s == 2) k_dot_peak<16,2><<<c.blocks, c.thr, 0, st>>>(out, ops, iters);
    else if (c.r == 8 && c.s == 4) k_dot_peak<8,4><<<c.blocks, c.thr, 0, st>>>(out, ops, iters);
    else k_dot_peak<16,4><<<c.blocks, c.thr, 0, st>>>(out, ops, iters);
  }
}

int main(int argc, char** argv) {
  const char* mode = "mem";
  const char* only = nullptr;
  int iters = 2000, reps = 5;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--mode")) mode = argv[++i];
    else if (!strcmp(argv[i], "--only")) only = argv[++i];
    else if (!strcmp(argv[i], "--iters")) iters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--reps")) reps = atoi(argv[++i]);
  }
  bool dequant = !strcmp(mode, "dequant");
  bool compute = !strcmp(mode, "dot") || dequant;

  // deterministic packed weights + x
  const int HW = ROWS * KB * HWPB;
  unsigned short* halfs; __half* x; float* buf; float* ops; float* scratch;
  cudaMalloc(&halfs, HW * sizeof(unsigned short));
  cudaMalloc(&x, K * sizeof(__half));
  cudaMalloc(&buf, (size_t)ROWS * PARTS * sizeof(float));
  cudaMalloc(&scratch, (size_t)ROWS * sizeof(float));
  cudaMalloc(&ops, (size_t)262144 * sizeof(float));   // dot uses 4096*32, dequant 1024*256 u32
  unsigned short* hh = (unsigned short*)malloc(HW * sizeof(unsigned short));
  __half* hx = (__half*)malloc(K * sizeof(__half));
  unsigned* oh = (unsigned*)malloc((size_t)262144 * sizeof(unsigned));
  union { float f; unsigned u; } fu;
  unsigned seed = 0x12345678u;
  // keep the fp16 d slots finite (mask the exponent) so the numeric spot check stays NaN-free
  for (int i = 0; i < HW; i++) { seed = seed*1664525u + 1013904223u; hh[i] = (unsigned short)((seed >> 16) & 0x7bff); }
  for (int i = 0; i < K; i++) { seed = seed*1664525u + 1013904223u; hx[i] = __float2half((float)(seed % 1000) / 500.0f - 1.0f); }
  for (int i = 0; i < 262144; i++) { seed = seed*1664525u + 1013904223u; oh[i] = seed; }
  for (int i = 0; i < 4096*32; i++) { fu.f = (float)(i % 251) * 0.01f + 1.0f; oh[i] = fu.u; }   // finite dot operands
  for (int t = 0; t < 1024; t++) {           // finite d + xr slots inside each dequant chunk
    size_t o = (size_t)t * 256;
    for (int j = 0; j < NACC; j++) {
      fu.f = 0.75f + 0.001f * (float)(j + (t & 7)); oh[o + 72 + j] = fu.u;
      for (int i = 0; i < 8; i++) {
        __half2 h2 = __floats2half2_rn(0.5f + 0.002f*(float)(2*i + (t & 15)),
                                       0.5f + 0.002f*(float)(2*i + 1 + (t & 15)));
        unsigned u2; memcpy(&u2, &h2, sizeof(u2));
        oh[o + 80 + (j*8 + i)*2] = u2;
      }
    }
  }
  cudaMemcpy(halfs, hh, HW * sizeof(unsigned short), cudaMemcpyHostToDevice);
  cudaMemcpy(x, hx, K * sizeof(__half), cudaMemcpyHostToDevice);
  cudaMemcpy(ops, oh, (size_t)262144 * sizeof(unsigned), cudaMemcpyHostToDevice);

  cudaStream_t st; cudaStreamCreate(&st);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  int dev; cudaGetDevice(&dev);
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
  printf("device=%s mode=%s iters=%d reps=%d\n", prop.name, mode, iters, reps);
  printf("%-20s %5s %3s %10s %10s %10s %10s\n", "config", "blk", "thr", "us/pass", "TB/s", "GMAC/s", "note");

  for (int ci = 0; ci < NCFGS; ci++) {
    const Cfg& c = CFGS[ci];
    if (only && strcmp(only, c.name)) continue;
    if (compute) {
      launch_compute(c, buf, ops, 1000, dequant, st); cudaStreamSynchronize(st);   // warmup
      double best = 1e30;
      for (int r = 0; r < reps; r++) {
        cudaEventRecord(e0, st);
        launch_compute(c, buf, ops, iters, dequant, st);
        cudaEventRecord(e1, st);
        cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        best = std::min(best, (double)ms);
      }
      double macs = (double)c.thr * c.blocks * iters * NACC * 16;      // MACs in the timed region
      double tm = best * 1e-3;
      double gmac = macs / tm / 1e9;
      double equiv_us = 4.19e6 / (macs / best) * 1e3;                  // us for the 1024x4096 shape's 4.19M MACs
      printf("%-20s %5d %3d %6.2f %10s %10.0f %10.2f\n", c.name, c.blocks, c.thr, equiv_us, "-",
             gmac, equiv_us);
      continue;
    }
    launch_mem(c, halfs, x, buf, 64, st); cudaStreamSynchronize(st);   // warmup
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
      cudaEventRecord(e0, st);
      launch_mem(c, halfs, x, buf, iters, st);
      cudaEventRecord(e1, st);
      cudaEventSynchronize(e1);
      float ms; cudaEventElapsedTime(&ms, e0, e1);
      best = std::min(best, (double)ms);
    }
    double us_per_pass = best * 1e3 / iters;
    double gb = (double)ROWS * KB * 210.0 / us_per_pass / 1e6;         // TB/s on weight bytes only
    double macs = (double)c.thr * c.blocks * (double)(ROWS * K) / (double)(c.thr * c.blocks);
    printf("%-20s %5d %3d %10.2f %10.2f %10.0f %s\n", c.name, c.blocks, c.thr, us_per_pass, gb,
           macs / us_per_pass / 1e3, c.legacy ? "(external_sum)" : "(in-kernel merge)");
    if (!c.legacy) {   // one-pass numerical spot check: merge total vs legacy sum-of-partials
      cudaMemset(buf, 0, (size_t)ROWS * PARTS * sizeof(float));
      k_legacy<<<dim3(PARTS, ROWS/32), 32, 0, st>>>(halfs, x, buf, 1);
      launch_mem(c, halfs, x, scratch, 1, st);
      cudaStreamSynchronize(st);
      float *h0 = (float*)malloc((size_t)ROWS * PARTS * sizeof(float));
      float *h1 = (float*)malloc((size_t)ROWS * sizeof(float));
      cudaMemcpy(h0, buf, (size_t)ROWS * PARTS * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h1, scratch, (size_t)ROWS * sizeof(float), cudaMemcpyDeviceToHost);
      double maxdiff = 0.0;
      double reldiff = 0.0;
      for (int row = 0; row < ROWS; row++) {
        float sum = 0.0f;
        for (int p = 0; p < PARTS; p++) sum += h0[row*PARTS + p];
        double d = fabs((double)(h1[row] - sum));
        maxdiff = std::max(maxdiff, d);
        reldiff = std::max(reldiff, d / std::max(fabs((double)sum), 1.0));
      }
      printf("  check: max|diff| = %.4e  max|rel| = %.3e\n", maxdiff, reldiff);
      free(h0); free(h1);
    }
  }
  cudaDeviceSynchronize();
  return 0;
}
