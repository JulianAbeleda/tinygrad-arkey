// l2_q6k_partial_sweep.cu - L2 Q6K partial single-pass diagnostic + MC2 load-pattern sweep
// (NV sm_120)
//
// Original L2 scope: wmma_peak-style characterization of the in-kernel reduce on the fixed
// parts=4 packed storage of `q6k_gen_partial_1024_4096_4` (1024 rows x 4096 k, 16 blocks/row,
// 4 blocks per part, Q6K_HALFWORDS_PER_BLOCK=105). The knob is the thread decomposition of the
// in-kernel part merge over the same packed layout (no load-time repack): rows per block x
// threads per part (split-reduce) x part lanes, covering at minimum the two recorded shapes
// (4-thread part blocks; 8-row x 4-part 32-thread blocks) plus 16-row variants and
// split-reduce variants.
//
// MC2 extension (decode-gemv-instruction-bandwidth-scope-20260803.md section 4.2): the same
// probe surface is extended to the coop-down shape (`q6k_gen_coop_4096_12288_inkernel`,
// row_tile=2) and the q4k gate/up shape (`q4k_g3_lanemap_gemv_12288_4096`), plus new load
// knobs over the recorded decomposition surface:
//   vec   - vector width of the packed-storage loads: u16 (installed scalar halfword loads),
//           u32 (4B register-window loads with byte extraction), u128 (16B register-window
//           loads). The u16/u32/u128 variants keep the installed per-use extraction math and
//           accumulation order; only the load width changes.
//   pf    - prefetch depth: blocks materialized into registers per batch (1 or 2; deeper
//           batches exceed the register budget at u128 and are dropped, see the record).
//   xsmem - x staging: installed per-row-block re-read of the activation from L2 vs one
//           smem stage per thread block reused across all rows of the block (no dtype
//           change, no quantize pass; the fp16 x bytes are staged verbatim).
//   al    - part->block map: contiguous (installed) vs interleaved, which shifts the
//           per-part window start alignment mod 16.
//   rows  - active row subset (power of two) for the L2-residency curve.
//   group - coop group-lane layout: lane = grp x row_i, each lane owns one 16-byte ql+qh
//           window and all 16 pos, x vectorized from smem, 16-component pos ladder over the
//           group lanes (the installed coop layout is lane = pos x row_i).
//   bw    - set-sized streaming read (k_bw_read) as the L2-resident bandwidth ceiling
//           control for each shape's weight set.
//
// Three probe modes, all following the wmma_peak discipline (operand setup hoisted,
// multiple independent accumulators, runtime trip count, never-taken keep-alive store,
// inspect the rendered source before believing a number):
//
//   mem    - faithful reproduction: real packed-storage loads (identical byte layout and
//            per-thread work to the installed kernel), per-decomposition launch config,
//            in-kernel XOR ladder + gated store for the merged variants, partials store
//            for the legacy external_sum shape. This is the go/no-go evidence (us per
//            pass, TB/s) and the mandatory control reproduction (12.92 us standalone /
//            17.15 in-loop partial; 34.90 in-loop coop-down).
//   dot    - pure steady-state dot chain on register-resident fp32 operands, NACC
//            independent accumulators, zero loads in the hot loop: the ALU ceiling.
//   dequant- same, but the full Q6K dequant+FMA chain runs on register-resident packed
//            bytes each iteration: the realistic ALU+dequant instruction mix, zero loads.
//
//   nvcc -O3 -arch=sm_120 -std=c++17 l2_q6k_partial_sweep.cu -o l2_q6k_partial_sweep
//   ./l2_q6k_partial_sweep --mode mem
//   ./l2_q6k_partial_sweep --mode mem --shape coop
//   ./l2_q6k_partial_sweep --mode mem --shape q4k
//   ./l2_q6k_partial_sweep --mode dot --nacc 8
//
// Verify purity (cuobjdump --dump-sass): `mem` hot loops contain the intended LDG/LDS mix
// plus the SHFL ladder + gated STG (no LDL/STL, 0 spills); `dot`/`dequant` hot loops
// contain zero LDG/LDS/STS and exactly one gated STG sentinel. The xsmem variants
// intentionally contain LDS (documented in the record); the staging STS/LDG sit outside
// the timed loop.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>

// CUDA 13.2 has no __half4; 4-halfword vector loads use this 8B-aligned struct.
struct alignas(8) H4 { __half x, y, z, w; };

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

// =========================================================================
// MC2 load-pattern sweep (scope section 4.2): coop-down + q4k gate/up shapes
// =========================================================================

// shape constants (coop-down / q4k gate-up)
#define COOP_ROWS 4096
#define COOP_K 12288
#define COOP_KB (COOP_K / 256)          // 48 blocks per row
#define Q4K_ROWS 12288
#define Q4K_K 4096
#define Q4K_KB (Q4K_K / 256)            // 16 blocks per row
#define Q4K_WORDS_PER_BLOCK 36
#define Q4K_GROUPS 8

// ---- Q6K vectorized load helpers -----------------------------------------
// A Q6_K block is 210 bytes starting at 2-byte granularity. The vec variants read
// byte j of the block from a 4B (u32) or 16B (u128) aligned window whose start is
// win = block_bytes & ~(width-1), off = block_bytes & (width-1): the window byte
// for block byte j is (off + j). ptxas CSEs the repeated wide loads across the
// unrolled (grp, pos) uses exactly as it does the installed scalar halfword loads;
// no register array is materialized, so the hot loop stays spill-free.
template<int VW>
__device__ __forceinline__ unsigned char q6k_wb(const unsigned char* win, int off, int j) {
  int k = off + j;
  if (VW == 32) {
    const unsigned* u = (const unsigned*)win;
    return (unsigned char)((u[k >> 2] >> ((k & 3) << 3)) & 0xff);
  }
  if (VW == 16) {
    const unsigned short* h = (const unsigned short*)win;
    return (unsigned char)((h[k >> 1] >> ((k & 1) << 3)) & 0xff);
  }
  const uint4* q = (const uint4*)win;
  uint4 v = q[k >> 4];
  unsigned w = (k & 4) ? ((k & 8) ? v.w : v.y) : ((k & 8) ? v.z : v.x);
  return (unsigned char)((w >> ((k & 3) << 3)) & 0xff);
}

template<int VW>
__device__ __forceinline__ float q6k_win_d(const unsigned char* win, int off) {
  int k = off + 208;
  unsigned short dv;
  if (VW == 32) {
    const unsigned* u = (const unsigned*)win;
    dv = (unsigned short)((u[k >> 2] >> ((k & 3) << 3)) & 0xffff);
  } else if (VW == 16) {
    dv = ((const unsigned short*)win)[k >> 1];
  } else {
    const uint4* q = (const uint4*)win;
    uint4 v = q[k >> 4];
    unsigned w = (k & 4) ? ((k & 8) ? v.w : v.y) : ((k & 8) ? v.z : v.x);
    dv = (unsigned short)((w >> ((k & 3) << 3)) & 0xffff);
  }
  return __half2float(__ushort_as_half(dv));
}

// byte extraction from a register-resident 16-byte window (k_coop_group uses this
// with compile-time pos, so the window arrays stay in registers)
__device__ __forceinline__ unsigned char q6k_rb(const unsigned* b, int off, int j) {
  int k = off + j;
  return (unsigned char)((b[k >> 2] >> ((k & 3) << 3)) & 0xff);
}

// pos-fixed dot over a memory window (installed accumulation order)
template<int VW>
__device__ __forceinline__ float q6k_block_dot_mem(const unsigned char* win, int off,
                                                   const __half* x, int blk, int pos) {
  float acc = 0.0f;
  float d = q6k_win_d<VW>(win, off);
#pragma unroll 4
  for (int grp = 0; grp < GRPS; grp++) {
    int half = grp >> 3, pgrp = grp & 7;
    int qlb = half*64 + (pgrp & 3)*16 + pos, qls = (pgrp >= 4) ? 4 : 0;
    int qhb = 128 + half*32 + (pgrp & 1)*16 + pos, qhs = (pgrp >> 1) * 2;
    unsigned char ql = (q6k_wb<VW>(win, off, qlb) >> qls) & 0xf;
    unsigned char qh = (q6k_wb<VW>(win, off, qhb) >> qhs) & 0x3;
    float q = (float)((int)(ql | (qh << 4)) - 32);
    float sc = (float)(signed char)q6k_wb<VW>(win, off, 192 + grp);
    acc += d * q * sc * __half2float(x[blk*256 + grp*16 + pos]);
  }
  return acc;
}

// all-16-pos dot with x from smem, grp-outer/pos-inner so the smem reads vectorize
// (partial thread structure: pos is a per-thread loop)
template<int VW>
__device__ __forceinline__ float q6k_block_dot_sx(const unsigned char* win, int off,
                                                  const __half* sx, int blk) {
  float acc = 0.0f;
  float d = q6k_win_d<VW>(win, off);
#pragma unroll 4
  for (int grp = 0; grp < GRPS; grp++) {
    const H4* p4 = (const H4*)(sx + blk*256 + grp*16);
    H4 v0 = p4[0], v1 = p4[1], v2 = p4[2], v3 = p4[3];
    float sc = (float)(signed char)q6k_wb<VW>(win, off, 192 + grp);
    int half = grp >> 3, pgrp = grp & 7;
#pragma unroll 2
  for (int pos = 0; pos < POSN; pos++) {
      int qlb = half*64 + (pgrp & 3)*16 + pos, qls = (pgrp >= 4) ? 4 : 0;
      int qhb = 128 + half*32 + (pgrp & 1)*16 + pos, qhs = (pgrp >> 1) * 2;
      unsigned char ql = (q6k_wb<VW>(win, off, qlb) >> qls) & 0xf;
      unsigned char qh = (q6k_wb<VW>(win, off, qhb) >> qhs) & 0x3;
      float q = (float)((int)(ql | (qh << 4)) - 32);
      H4 v = pos < 4 ? v0 : pos < 8 ? v1 : pos < 12 ? v2 : v3;
      int pp = pos & 3;
      float xv = __half2float(pp == 0 ? v.x : pp == 1 ? v.y : pp == 2 ? v.z : v.w);
      acc += d * q * sc * xv;
    }
  }
  return acc;
}

// pos-fixed dot with x from smem (coop lane structure; scalar LDS, coalesced across lanes)
template<int VW>
__device__ __forceinline__ float q6k_block_dot_sx_pos(const unsigned char* win, int off,
                                                      const __half* sx, int blk, int pos) {
  float acc = 0.0f;
  float d = q6k_win_d<VW>(win, off);
#pragma unroll 4
  for (int grp = 0; grp < GRPS; grp++) {
    int half = grp >> 3, pgrp = grp & 7;
    int qlb = half*64 + (pgrp & 3)*16 + pos, qls = (pgrp >= 4) ? 4 : 0;
    int qhb = 128 + half*32 + (pgrp & 1)*16 + pos, qhs = (pgrp >> 1) * 2;
    unsigned char ql = (q6k_wb<VW>(win, off, qlb) >> qls) & 0xf;
    unsigned char qh = (q6k_wb<VW>(win, off, qhb) >> qhs) & 0x3;
    float q = (float)((int)(ql | (qh << 4)) - 32);
    float sc = (float)(signed char)q6k_wb<VW>(win, off, 192 + grp);
    acc += d * q * sc * __half2float(sx[blk*256 + grp*16 + pos]);
  }
  return acc;
}

// smem x staging, hoisted once per block launch before the timed loop
template<int XHALVES>
__device__ __forceinline__ void stage_x(__half* sx, const __half* x) {
  const H4* x4 = (const H4*)x;
  H4* s4 = (H4*)sx;
  const int n4 = XHALVES / 4;   // one H4 = 4 halfwords
  for (int i = threadIdx.x; i < n4; i += blockDim.x) s4[i] = x4[i];
  __syncthreads();
}

// ---- partial: legacy block structure (32-thr, grid (4, rows/32)) with vectorized loads.
// AL interleaves the part->block map (part p handles blocks p, p+4, p+8, p+12), shifting
// the per-part window start alignment mod 16.
template<int VW, int PF, bool XSMEM, bool AL>
__global__ void __launch_bounds__(32) k_legacy_v(const unsigned short* __restrict__ halfs,
                                                 const __half* __restrict__ x,
                                                 float* __restrict__ partials, int iters, int rows) {
  const int part = blockIdx.x, rowg = blockIdx.y, rowl = threadIdx.x;
  const unsigned char* base_w = (const unsigned char*)halfs;
  __shared__ alignas(16) __half sx[4096];
  if (XSMEM) stage_x<4096>(sx, x);
  for (int t = 0; t < iters; t++) {
    int row = (rowg*32 + rowl + t) & (rows - 1);
    float acc = 0.0f;
    if (PF >= 2) {
#pragma unroll 2
      for (int b = 0; b < BPP; b++) {
        int blk = AL ? (part + b*4) : (part*BPP + b);
        const unsigned char* win = base_w + (((row*KB + blk)*HWPB) << 1 & ~((VW >> 3) - 1));
        int off = (((row*KB + blk)*HWPB) << 1) & ((VW >> 3) - 1);
        if (XSMEM) acc += q6k_block_dot_sx<VW>(win, off, sx, blk);
        else {
#pragma unroll 2
          for (int pos = 0; pos < POSN; pos++) acc += q6k_block_dot_mem<VW>(win, off, x, blk, pos);
        }
      }
    } else {
      for (int b = 0; b < BPP; b++) {
        int blk = AL ? (part + b*4) : (part*BPP + b);
        const unsigned char* win = base_w + (((row*KB + blk)*HWPB) << 1 & ~((VW >> 3) - 1));
        int off = (((row*KB + blk)*HWPB) << 1) & ((VW >> 3) - 1);
        if (XSMEM) acc += q6k_block_dot_sx<VW>(win, off, sx, blk);
        else {
#pragma unroll 4
          for (int pos = 0; pos < POSN; pos++) acc += q6k_block_dot_mem<VW>(win, off, x, blk, pos);
        }
      }
    }
    partials[row*PARTS + part] = acc;
  }
}

// ---- partial: merge family with vectorized loads (same lane map as k_merge<R,S>)
template<int R, int S, int VW, int PF, bool XSMEM>
__global__ void __launch_bounds__(R*4*S) k_merge_v(const unsigned short* __restrict__ halfs,
                                                   const __half* __restrict__ x,
                                                   float* __restrict__ out, int iters, int rows) {
  int tid = threadIdx.x;
  int row_local = tid / (4*S);
  int lane = tid % (4*S);
  int part = lane / S, split = lane % S;
  unsigned mask = (blockDim.x < 32) ? ((1u << blockDim.x) - 1u) : 0xffffffffu;
  const unsigned char* base_w = (const unsigned char*)halfs;
  __shared__ alignas(16) __half sx[4096];
  if (XSMEM) stage_x<4096>(sx, x);
  const int bps = BPP / S;
  for (int t = 0; t < iters; t++) {
    int row = (blockIdx.x*R + row_local + t) & (rows - 1);
    float acc = 0.0f;
    if (PF >= 2) {
#pragma unroll 2
      for (int b = 0; b < bps; b++) {
        int blk = part*BPP + split*bps + b;
        const unsigned char* win = base_w + (((row*KB + blk)*HWPB) << 1 & ~((VW >> 3) - 1));
        int off = (((row*KB + blk)*HWPB) << 1) & ((VW >> 3) - 1);
        if (XSMEM) acc += q6k_block_dot_sx<VW>(win, off, sx, blk);
        else {
#pragma unroll 2
          for (int pos = 0; pos < POSN; pos++) acc += q6k_block_dot_mem<VW>(win, off, x, blk, pos);
        }
      }
    } else {
      for (int b = 0; b < bps; b++) {
        int blk = part*BPP + split*bps + b;
        const unsigned char* win = base_w + (((row*KB + blk)*HWPB) << 1 & ~((VW >> 3) - 1));
        int off = (((row*KB + blk)*HWPB) << 1) & ((VW >> 3) - 1);
        if (XSMEM) acc += q6k_block_dot_sx<VW>(win, off, sx, blk);
        else {
#pragma unroll 4
          for (int pos = 0; pos < POSN; pos++) acc += q6k_block_dot_mem<VW>(win, off, x, blk, pos);
        }
      }
    }
    for (int o = S>>1; o >= 1; o >>= 1) acc += __shfl_xor_sync(mask, acc, o);
    acc += __shfl_xor_sync(mask, acc, 2*S);
    acc += __shfl_xor_sync(mask, acc, S);
    if (lane == 0) out[row] = acc;
  }
}

// ---- coop-down: installed row_tile=2 replica (control; lane = pos*2 + row_i,
// serial 48-block reduce, ladder xor 16/8/4/2)
__global__ void __launch_bounds__(32) k_coop_legacy(const unsigned short* __restrict__ halfs,
                                                    const __half* __restrict__ x,
                                                    float* __restrict__ out, int iters, int rows) {
  const int pos = threadIdx.x >> 1, row_i = threadIdx.x & 1;
  for (int t = 0; t < iters; t++) {
    int row = (blockIdx.x*2 + row_i + t) & (rows - 1);
    float acc = 0.0f;
    for (int blk = 0; blk < COOP_KB; blk++) {
      int base = (row*COOP_KB + blk) * HWPB;
      Q6KBlockCtx ctx = q6k_block_ctx(halfs, base);
      acc += q6k_block_dot(ctx, halfs, x, base, blk, pos);
    }
    acc += __shfl_xor_sync(0xffffffffu, acc, 16);
    acc += __shfl_xor_sync(0xffffffffu, acc, 8);
    acc += __shfl_xor_sync(0xffffffffu, acc, 4);
    acc += __shfl_xor_sync(0xffffffffu, acc, 2);
    if (pos == 0) out[row] = acc;   // both row_tile rows: lane 0 (row_i=0) and lane 1 (row_i=1)
  }
}

// ---- coop: S=1 (row_tile=2 lanes) vectorized, R rows per block
template<int R, int VW, int PF, bool XSMEM>
__global__ void __launch_bounds__(R*16) k_coop_v(const unsigned short* __restrict__ halfs,
                                                 const __half* __restrict__ x,
                                                 float* __restrict__ out, int iters, int rows) {
  int lane = threadIdx.x & 31;
  int pos = lane >> 1, row_i = lane & 1;
  int row_local = (threadIdx.x >> 5)*2 + row_i;
  const unsigned char* base_w = (const unsigned char*)halfs;
  __shared__ alignas(16) __half sx[COOP_K];
  if (XSMEM) stage_x<COOP_K>(sx, x);
  for (int t = 0; t < iters; t++) {
    int row = (blockIdx.x*R + row_local + t) & (rows - 1);
    float acc = 0.0f;
    if (PF >= 2) {
#pragma unroll 2
      for (int blk = 0; blk < COOP_KB; blk++) {
        const unsigned char* win = base_w + (((row*COOP_KB + blk)*HWPB) << 1 & ~((VW >> 3) - 1));
        int off = (((row*COOP_KB + blk)*HWPB) << 1) & ((VW >> 3) - 1);
        if (XSMEM) acc += q6k_block_dot_sx_pos<VW>(win, off, sx, blk, pos);
        else acc += q6k_block_dot_mem<VW>(win, off, x, blk, pos);
      }
    } else {
      for (int blk = 0; blk < COOP_KB; blk++) {
        const unsigned char* win = base_w + (((row*COOP_KB + blk)*HWPB) << 1 & ~((VW >> 3) - 1));
        int off = (((row*COOP_KB + blk)*HWPB) << 1) & ((VW >> 3) - 1);
        if (XSMEM) acc += q6k_block_dot_sx_pos<VW>(win, off, sx, blk, pos);
        else acc += q6k_block_dot_mem<VW>(win, off, x, blk, pos);
      }
    }
    acc += __shfl_xor_sync(0xffffffffu, acc, 16);
    acc += __shfl_xor_sync(0xffffffffu, acc, 8);
    acc += __shfl_xor_sync(0xffffffffu, acc, 4);
    acc += __shfl_xor_sync(0xffffffffu, acc, 2);
    if (pos == 0) out[row] = acc;   // both row_tile rows
  }
}

// ---- coop: S=2 split-reduce (32 lanes per row = 16 pos x 2 k-splits), R rows per block
template<int R, int VW, int PF, bool XSMEM>
__global__ void __launch_bounds__(R*32) k_coop_s2(const unsigned short* __restrict__ halfs,
                                                  const __half* __restrict__ x,
                                                  float* __restrict__ out, int iters, int rows) {
  int lane = threadIdx.x & 31;
  int pos = lane >> 1, split = lane & 1;
  int row_local = threadIdx.x >> 5;
  const unsigned char* base_w = (const unsigned char*)halfs;
  __shared__ alignas(16) __half sx[COOP_K];
  if (XSMEM) stage_x<COOP_K>(sx, x);
  const int bps = COOP_KB / 2;
  for (int t = 0; t < iters; t++) {
    int row = (blockIdx.x*R + row_local + t) & (rows - 1);
    float acc = 0.0f;
    if (PF >= 2) {
#pragma unroll 2
      for (int b = 0; b < bps; b++) {
        int blk = split*bps + b;
        const unsigned char* win = base_w + (((row*COOP_KB + blk)*HWPB) << 1 & ~((VW >> 3) - 1));
        int off = (((row*COOP_KB + blk)*HWPB) << 1) & ((VW >> 3) - 1);
        if (XSMEM) acc += q6k_block_dot_sx_pos<VW>(win, off, sx, blk, pos);
        else acc += q6k_block_dot_mem<VW>(win, off, x, blk, pos);
      }
    } else {
      for (int b = 0; b < bps; b++) {
        int blk = split*bps + b;
        const unsigned char* win = base_w + (((row*COOP_KB + blk)*HWPB) << 1 & ~((VW >> 3) - 1));
        int off = (((row*COOP_KB + blk)*HWPB) << 1) & ((VW >> 3) - 1);
        if (XSMEM) acc += q6k_block_dot_sx_pos<VW>(win, off, sx, blk, pos);
        else acc += q6k_block_dot_mem<VW>(win, off, x, blk, pos);
      }
    }
    acc += __shfl_xor_sync(0xffffffffu, acc, 1);   // split ladder (lane = pos*2 + split)
    acc += __shfl_xor_sync(0xffffffffu, acc, 2);   // pos ladder
    acc += __shfl_xor_sync(0xffffffffu, acc, 4);
    acc += __shfl_xor_sync(0xffffffffu, acc, 8);
    acc += __shfl_xor_sync(0xffffffffu, acc, 16);
    if (lane == 0) out[row] = acc;
  }
}

// ---- coop: S=4 split-reduce (64 lanes per row = 2 warps; cross-warp smem merge)
template<int R, int VW, bool XSMEM>
__global__ void __launch_bounds__(R*64) k_coop_s4(const unsigned short* __restrict__ halfs,
                                                  const __half* __restrict__ x,
                                                  float* __restrict__ out, int iters, int rows) {
  int tid = threadIdx.x;
  int row_local = tid >> 6;
  int l = tid & 63;
  int warp = (l >> 5) & 1;
  int pos = warp*8 + ((l & 31) >> 2), split = l & 3;
  const unsigned char* base_w = (const unsigned char*)halfs;
  __shared__ alignas(16) __half sx[COOP_K];
  __shared__ float sm[R*2];
  if (XSMEM) stage_x<COOP_K>(sx, x);
  const int bps = COOP_KB / 4;
  for (int t = 0; t < iters; t++) {
    int row = (blockIdx.x*R + row_local + t) & (rows - 1);
    float acc = 0.0f;
    for (int b0 = 0; b0 < bps; b0++) {
      int blk = split*bps + b0;
      const unsigned char* win = base_w + (((row*COOP_KB + blk)*HWPB) << 1 & ~((VW >> 3) - 1));
      int off = (((row*COOP_KB + blk)*HWPB) << 1) & ((VW >> 3) - 1);
      if (XSMEM) acc += q6k_block_dot_sx_pos<VW>(win, off, sx, blk, pos);
      else acc += q6k_block_dot_mem<VW>(win, off, x, blk, pos);
    }
    acc += __shfl_xor_sync(0xffffffffu, acc, 1);   // split within warp
    acc += __shfl_xor_sync(0xffffffffu, acc, 2);   // split bit 1 (lane = pos*4 + split)
    acc += __shfl_xor_sync(0xffffffffu, acc, 4);   // pos 0..7 within warp (lane = pos*4+split)
    acc += __shfl_xor_sync(0xffffffffu, acc, 8);
    acc += __shfl_xor_sync(0xffffffffu, acc, 16);
    if ((l & 31) == 0) sm[row_local*2 + warp] = acc;
    __syncthreads();
    if (l == 0) out[row] = sm[row_local*2] + sm[row_local*2 + 1];
    __syncthreads();
  }
}

// ---- coop: group-lane layout (lane = grp*2 + row_i; each lane owns one 16-byte ql+qh
// window and all 16 pos; x vectorized from smem; 16-component pos ladder over the group
// lanes). Q6_K blocks are 210 bytes (2 mod 16), so the ql/qh window start is runtime
// misaligned: the loads use the aligned-window extraction with runtime offset (ptxas
// legalizes to predicated LDG.64 pairs; documented in the record).
template<int R, bool XSMEM>
__global__ void __launch_bounds__(R*16) k_coop_group(const unsigned short* __restrict__ halfs,
                                                     const __half* __restrict__ x,
                                                     float* __restrict__ out, int iters, int rows) {
  int lane = threadIdx.x & 31;
  int grp = lane >> 1, row_i = lane & 1;
  int row_local = (threadIdx.x >> 5)*2 + row_i;
  __shared__ alignas(16) __half sx[COOP_K];
  if (XSMEM) stage_x<COOP_K>(sx, x);
  const int half = grp >> 3, pgrp = grp & 7;
  const int qls = (pgrp >= 4) ? 4 : 0, qhs = (pgrp >> 1) * 2;
  for (int t = 0; t < iters; t++) {
    int row = (blockIdx.x*R + row_local + t) & (rows - 1);
    float acc[POSN];
#pragma unroll
    for (int pos = 0; pos < POSN; pos++) acc[pos] = 0.0f;
    for (int blk = 0; blk < COOP_KB; blk++) {
      const unsigned char* p = (const unsigned char*)halfs + (row*COOP_KB + blk)*HWPB*2;
      const unsigned char* qlw = (const unsigned char*)(((size_t)(p + half*64 + (pgrp & 3)*16)) & ~(size_t)15);
      const unsigned char* qhw = (const unsigned char*)(((size_t)(p + 128 + half*32 + (pgrp & 1)*16)) & ~(size_t)15);
      int qlo = (int)(((size_t)(p + half*64 + (pgrp & 3)*16)) & 15);
      int qho = (int)(((size_t)(p + 128 + half*32 + (pgrp & 1)*16)) & 15);
      float d = __half2float(__ushort_as_half(*(const unsigned short*)(p + 208)));
      float sc = (float)(signed char)p[192 + grp];
      const H4* x4 = (const H4*)(sx + blk*256 + grp*16);
      H4 xv0 = x4[0], xv1 = x4[1], xv2 = x4[2], xv3 = x4[3];
#pragma unroll
      for (int pos = 0; pos < POSN; pos++) {
        unsigned char ql = (q6k_wb<128>(qlw, qlo, pos) >> qls) & 0xf;
        unsigned char qh = (q6k_wb<128>(qhw, qho, pos) >> qhs) & 0x3;
        float q = (float)((int)(ql | (qh << 4)) - 32);
        H4 v = pos < 4 ? xv0 : pos < 8 ? xv1 : pos < 12 ? xv2 : xv3;
        int pp = pos & 3;
        float xv = __half2float(pp == 0 ? v.x : pp == 1 ? v.y : pp == 2 ? v.z : v.w);
        acc[pos] += d * q * sc * xv;
      }
    }
    // 16-component ladder over the 16 group lanes (row_tile=2: xor 16/8/4/2)
#pragma unroll
    for (int pos = 0; pos < POSN; pos++) {
      float v = acc[pos];
      v += __shfl_xor_sync(0xffffffffu, v, 16);
      v += __shfl_xor_sync(0xffffffffu, v, 8);
      v += __shfl_xor_sync(0xffffffffu, v, 4);
      v += __shfl_xor_sync(0xffffffffu, v, 2);
      acc[pos] = v;
    }
    if (grp == 0) {
      float total = 0.0f;
#pragma unroll
      for (int pos = 0; pos < POSN; pos++) total += acc[pos];
      out[row] = total;
    }
  }
}

// ---- q4k helpers (byte-identical to decode_kernels.py `_q4k_group_params` +
// `_q4k_group_dot_packed_load`)
__device__ __forceinline__ float q4k_f16(unsigned w, bool hi) {
  unsigned short h = hi ? (unsigned short)(w >> 16) : (unsigned short)(w & 0xffff);
  return __half2float(__ushort_as_half(h));
}
__device__ __forceinline__ unsigned char q4k_wb(unsigned w, int idx) {
  return (unsigned char)((w >> ((idx & 3) << 3)) & 0xff);
}
__device__ __forceinline__ unsigned q4k_scale_byte(const unsigned* words, int base, int idx) {
  return q4k_wb(words[base + 1 + idx/4], idx);
}
__device__ __forceinline__ void q4k_scales(const unsigned* words, int base, int grp, float& sc, float& mn) {
  if (grp < 4) {
    sc = (float)(q4k_scale_byte(words, base, grp) & 63);
    mn = (float)(q4k_scale_byte(words, base, 4 + grp) & 63);
  } else {
    unsigned high = q4k_scale_byte(words, base, 8 + grp - 4);
    sc = (float)((high & 0xf) | (q4k_scale_byte(words, base, grp - 4) >> 6 << 4));
    mn = (float)((high >> 4) | (q4k_scale_byte(words, base, grp) >> 6 << 4));
  }
}
__device__ __forceinline__ float q4k_group_dot(const unsigned* words, const __half* x,
                                               int base, int blk, int grp, int wc) {
  unsigned w0 = words[base];
  float d = q4k_f16(w0, false), dmin = q4k_f16(w0, true);
  float sc, mn; q4k_scales(words, base, grp, sc, mn);
  unsigned qpack = (words[base + 4 + (grp/2)*8 + wc] >> ((grp & 1) << 2)) & 0x0f0f0f0fu;
  float contrib = 0.0f;
#pragma unroll
  for (int nib = 0; nib < 4; nib++) {
    int pos = wc*4 + nib;
    unsigned q = (qpack >> (nib*8)) & 0xf;
    float weight = d * sc * (float)q - dmin * mn;
    contrib += weight * __half2float(x[blk*256 + grp*32 + pos]);
  }
  return contrib;
}

// ---- q4k: installed gate/up replica (1 row per block, 32 lanes, G3 lane map)
__global__ void __launch_bounds__(32) k_q4k_legacy(const unsigned* __restrict__ words,
                                                   const __half* __restrict__ x,
                                                   float* __restrict__ out, int iters, int rows) {
  const int bg = threadIdx.x >> 3, wc = threadIdx.x & 7;
  for (int t = 0; t < iters; t++) {
    int row = (blockIdx.x + t) & (rows - 1);
    float acc = 0.0f;
    for (int l = 0; l < 4; l++) {
      int blk = bg*4 + l;
      int base = (row*Q4K_KB + blk)*Q4K_WORDS_PER_BLOCK;
      for (int grp = 0; grp < Q4K_GROUPS; grp++) acc += q4k_group_dot(words, x, base, blk, grp, wc);
    }
    for (int o = 16; o >= 1; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (threadIdx.x == 0) out[row] = acc;
  }
}

// ---- q4k: vectorized variants, R rows per block
template<int VW, bool XSMEM>
__device__ __forceinline__ void q4k_lane_dot(const unsigned* __restrict__ words,
                                             const __half* __restrict__ x,
                                             const __half* sx, int row, int bg, int wc,
                                             int b0, float& acc) {
  int blk = bg*4 + b0;
  int base = (row*Q4K_KB + blk)*Q4K_WORDS_PER_BLOCK;
  unsigned w0, w1, w2, w3;
  if (VW == 128) {
    uint4 v0 = ((const uint4*)(words + base))[0];
    w0 = v0.x; w1 = v0.y; w2 = v0.z; w3 = v0.w;
  } else if (VW == 16) {
    const unsigned short* hw = (const unsigned short*)words;
    w0 = (unsigned)hw[2*base] | ((unsigned)hw[2*base + 1] << 16);
    w1 = (unsigned)hw[2*base + 2] | ((unsigned)hw[2*base + 3] << 16);
    w2 = (unsigned)hw[2*base + 4] | ((unsigned)hw[2*base + 5] << 16);
    w3 = (unsigned)hw[2*base + 6] | ((unsigned)hw[2*base + 7] << 16);
  } else {
    w0 = words[base]; w1 = words[base + 1]; w2 = words[base + 2]; w3 = words[base + 3];
  }
  unsigned qp[4];
  if (VW == 128) {
    // one 16B-aligned uint4 per g2 (qpack rows are strided by 8 words per g2); the
    // lane extracts its 4B slot (wc & 3) like the u32 path
    const uint4* q0 = (const uint4*)(words + base + 4 + 0*8 + (wc & ~3));
    const uint4* q1 = (const uint4*)(words + base + 4 + 1*8 + (wc & ~3));
    const uint4* q2 = (const uint4*)(words + base + 4 + 2*8 + (wc & ~3));
    const uint4* q3 = (const uint4*)(words + base + 4 + 3*8 + (wc & ~3));
    uint4 g0 = q0[0], g1 = q1[0], g2 = q2[0], g3 = q3[0];
    const int c = wc & 3;
    qp[0] = c == 0 ? g0.x : c == 1 ? g0.y : c == 2 ? g0.z : g0.w;
    qp[1] = c == 0 ? g1.x : c == 1 ? g1.y : c == 2 ? g1.z : g1.w;
    qp[2] = c == 0 ? g2.x : c == 1 ? g2.y : c == 2 ? g2.z : g2.w;
    qp[3] = c == 0 ? g3.x : c == 1 ? g3.y : c == 2 ? g3.z : g3.w;
  } else if (VW == 16) {
    const unsigned short* hw = (const unsigned short*)words;
#pragma unroll
    for (int g2 = 0; g2 < 4; g2++) {
      int w = base + 4 + g2*8 + wc;
      qp[g2] = (unsigned)hw[2*w] | ((unsigned)hw[2*w + 1] << 16);
    }
  } else {
#pragma unroll
    for (int g2 = 0; g2 < 4; g2++) qp[g2] = words[base + 4 + g2*8 + wc];
  }
#pragma unroll
  for (int grp = 0; grp < Q4K_GROUPS; grp++) {
    float d = q4k_f16(w0, false), dmin = q4k_f16(w0, true);
    float sc, mn;
    if (grp < 4) {
      sc = (float)(q4k_wb(w1, grp) & 63);
      mn = (float)(q4k_wb(w2, grp) & 63);
    } else {
      unsigned hb = q4k_wb(w3, grp - 4);
      sc = (float)((hb & 0xf) | (q4k_wb(w1, grp - 4) >> 6 << 4));
      mn = (float)((hb >> 4) | (q4k_wb(w2, grp - 4) >> 6 << 4));
    }
    unsigned qpack = (qp[grp >> 1] >> ((grp & 1) << 2)) & 0x0f0f0f0fu;
#pragma unroll
    for (int nib = 0; nib < 4; nib++) {
      int pos = wc*4 + nib;
      unsigned q = (qpack >> (nib*8)) & 0xf;
      float weight = d * sc * (float)q - dmin * mn;
      float xv;
      if (XSMEM) {
        const __half2* h2 = (const __half2*)(sx + blk*256 + grp*32 + wc*4);
        __half2 a = h2[0], c2 = h2[1];
        xv = __half2float(nib < 2 ? (nib == 0 ? a.x : a.y) : (nib == 2 ? c2.x : c2.y));
      } else {
        xv = __half2float(x[blk*256 + grp*32 + pos]);
      }
      acc += weight * xv;
    }
  }
}

template<int R, int VW, int PF, bool XSMEM>
__global__ void __launch_bounds__(R*32) k_q4k_v(const unsigned* __restrict__ words,
                                                const __half* __restrict__ x,
                                                float* __restrict__ out, int iters, int rows) {
  int lane = threadIdx.x & 31;
  int bg = lane >> 3, wc = lane & 7;
  int row_local = threadIdx.x >> 5;
  __shared__ alignas(16) __half sx[4096];
  if (XSMEM) stage_x<4096>(sx, x);
  for (int t = 0; t < iters; t++) {
    int row = (blockIdx.x*R + row_local + t) & (rows - 1);
    float acc = 0.0f;
    if (PF >= 2) {
#pragma unroll 2
      for (int b0 = 0; b0 < 4; b0++) q4k_lane_dot<VW, XSMEM>(words, x, sx, row, bg, wc, b0, acc);
    } else {
      for (int b0 = 0; b0 < 4; b0++) q4k_lane_dot<VW, XSMEM>(words, x, sx, row, bg, wc, b0, acc);
    }
    for (int o = 16; o >= 1; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) out[row] = acc;
  }
}

// ---- q4k: quad-lane u128 (8 lanes per row; each thread owns 4 consecutive wc). The
// 4 qpack words of a (grp/2, wc-quad) are one 16B-aligned uint4 (Q4_K blocks are 144B,
// 0 mod 16), so the weight loads are pure LDG.128; x comes from smem as 2 LDS.128 per
// group. Ladder over the 8 group lanes (xor 4/2/1), store on lane 0 of each group.
template<int R, bool XSMEM>
__global__ void __launch_bounds__(R*8) k_q4k_qv(const unsigned* __restrict__ words,
                                                const __half* __restrict__ x,
                                                float* __restrict__ out, int iters, int rows) {
  int lane = threadIdx.x & 7;
  int row_local = threadIdx.x >> 3;
  int bg = lane >> 1, wc0 = (lane & 1) * 4;
  __shared__ alignas(16) __half sx[4096];
  if (XSMEM) stage_x<4096>(sx, x);
  for (int t = 0; t < iters; t++) {
    int row = (blockIdx.x*R + row_local + t) & (rows - 1);
    float acc = 0.0f;
    for (int b0 = 0; b0 < 4; b0++) {
      int blk = bg*4 + b0;
      int base = (row*Q4K_KB + blk)*Q4K_WORDS_PER_BLOCK;
      uint4 hdr = *(const uint4*)(words + base);
      unsigned w0 = hdr.x, w1 = hdr.y, w2 = hdr.z, w3 = hdr.w;
#pragma unroll
      for (int g2 = 0; g2 < 4; g2++) {
        uint4 q = *(const uint4*)(words + base + 4 + g2*8 + wc0);
        unsigned qp[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
        for (int gp = 0; gp < 2; gp++) {
          const int grp = 2*g2 + gp;
          const H4* x4 = (const H4*)(sx + blk*256 + grp*32 + wc0*4);
          H4 xv[4] = { x4[0], x4[1], x4[2], x4[3] };
          float d = q4k_f16(w0, false), dmin = q4k_f16(w0, true);
          float sc, mn;
          if (grp < 4) {
            sc = (float)(q4k_wb(w1, grp) & 63);
            mn = (float)(q4k_wb(w2, grp) & 63);
          } else {
            unsigned hb = q4k_wb(w3, grp - 4);
            sc = (float)((hb & 0xf) | (q4k_wb(w1, grp - 4) >> 6 << 4));
            mn = (float)((hb >> 4) | (q4k_wb(w2, grp - 4) >> 6 << 4));
          }
#pragma unroll
          for (int wc = 0; wc < 4; wc++) {
            unsigned qpack = (qp[wc] >> ((grp & 1) << 2)) & 0x0f0f0f0fu;
#pragma unroll
            for (int nib = 0; nib < 4; nib++) {
              unsigned qv = (qpack >> (nib*8)) & 0xf;
              float weight = d * sc * (float)qv - dmin * mn;
              float xvv = nib < 2 ? (nib == 0 ? __half2float(xv[wc].x) : __half2float(xv[wc].y))
                                  : (nib == 2 ? __half2float(xv[wc].z) : __half2float(xv[wc].w));
              acc += weight * xvv;
            }
          }
        }
      }
    }
    acc += __shfl_xor_sync(0xffffffffu, acc, 4);
    acc += __shfl_xor_sync(0xffffffffu, acc, 2);
    acc += __shfl_xor_sync(0xffffffffu, acc, 1);
    if (lane == 0) out[row] = acc;
  }
}

// ---- set-sized streaming read: L2-resident bandwidth ceiling control for each shape
__global__ void __launch_bounds__(256) k_bw_read(const float4* __restrict__ src,
                                                 float* __restrict__ out, int n4) {
  int i = blockIdx.x*256 + threadIdx.x;
  int stride = gridDim.x*256;
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  for (; i < n4; i += stride) {
    float4 v = src[i];
    acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
  }
  float s = acc.x + acc.y + acc.z + acc.w;
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

// ---- MC2 config table ----------------------------------------------------
// shape: 0=partial 1=coop 2=q4k; r=rows per block (32 marker for the legacy partial
// structure); s=k-splits; vec=load width; pf=prefetch depth; xsmem=x staged in smem;
// al=interleaved part map (partial); for q4k rows it marks the quad-lane u128 kernel
// (8 lanes per row, pure LDG.128 qpack windows); ctl=installed control kernel replica.
struct Cfg2 {
  const char* name;
  int shape;
  int r;
  int s;
  int vec;
  int pf;
  bool xsmem;
  bool al;
  bool ctl;
};

static const Cfg2 CFGS2[] = {
  // ---- partial: vector width / prefetch / xsmem / window alignment on the legacy
  // 32-thr structure and on the L2-best split-reduce family
  {"legacy_32_u32",            0, 32, 1,  32, 1, false, false, false},
  {"legacy_32_u128",           0, 32, 1, 128, 1, false, false, false},
  {"legacy_32_u128_pf2",       0, 32, 1, 128, 2, false, false, false},
  {"legacy_32_u128_al",        0, 32, 1, 128, 1, false, true,  false},
  {"legacy_32_u128_xsmem",     0, 32, 1, 128, 1, true,  false, false},
  {"r8_split2_64thr_u32",      0,  8, 2,  32, 1, false, false, false},
  {"r8_split2_64thr_u128",     0,  8, 2, 128, 1, false, false, false},
  {"r8_split2_64thr_u128_pf2", 0,  8, 2, 128, 2, false, false, false},
  {"r8_split4_128thr_u32",     0,  8, 4,  32, 1, false, false, false},
  {"r8_split4_128thr_u128",    0,  8, 4, 128, 1, false, false, false},
  {"r8_split4_128thr_xsmem",   0,  8, 4,  16, 1, true,  false, false},
  {"r8_split4_128thr_u128_xsmem", 0, 8, 4, 128, 1, true, false, false},
  {"r16_split4_256thr_u128_xsmem", 0, 16, 4, 128, 1, true, false, false},
  {"r32_split4_512thr_u128_xsmem", 0, 32, 4, 128, 1, true, false, false},
  // ---- coop-down: control replica, vec width, prefetch, xsmem, split-reduce,
  // block grouping, group-lane layout
  {"coop_legacy",              1,  2, 1,  16, 1, false, false, true},
  {"coop_legacy_u32",          1,  2, 1,  32, 1, false, false, false},
  {"coop_legacy_u128",         1,  2, 1, 128, 1, false, false, false},
  {"coop_legacy_u128_pf2",     1,  2, 1, 128, 2, false, false, false},
  {"coop_legacy_xsmem",        1,  2, 1,  16, 1, true,  false, false},
  {"coop_legacy_u128_xsmem",   1,  2, 1, 128, 1, true,  false, false},
  {"coop_128thr_u128_xsmem",   1,  8, 1, 128, 1, true,  false, false},
  {"coop_256thr_u128_xsmem",   1, 16, 1, 128, 1, true,  false, false},
  {"coop_512thr_u128_xsmem",   1, 32, 1, 128, 1, true,  false, false},
  {"coop_s2_u128",             1,  1, 2, 128, 1, false, false, false},
  {"coop_s2_128thr_u128_xsmem",1,  4, 2, 128, 1, true,  false, false},
  {"coop_s2_256thr_u128_xsmem",1,  8, 2, 128, 1, true,  false, false},
  {"coop_s2_512thr_u128_xsmem",1, 16, 2, 128, 1, true,  false, false},
  {"coop_s4_512thr_u128_xsmem",1,  8, 4, 128, 1, true,  false, false},
  {"coop_group_32thr_u128_xsmem", 1, 2, 1, 128, 1, true, false, false},
  {"coop_group_128thr_u128_xsmem", 1, 8, 1, 128, 1, true, false, false},
  {"coop_group_256thr_u128_xsmem", 1, 16, 1, 128, 1, true, false, false},
  {"coop_group_512thr_u128_xsmem", 1, 32, 1, 128, 1, true, false, false},
  // ---- q4k gate/up: control replica, vec width, prefetch, xsmem, block grouping
  {"q4k_legacy",               2,  1, 1,  32, 1, false, false, true},
  {"q4k_legacy_u16",           2,  1, 1,  16, 1, false, false, false},
  {"q4k_legacy_u128",          2,  1, 1, 128, 1, false, false, false},
  {"q4k_legacy_u128_pf2",      2,  1, 1, 128, 2, false, false, false},
  {"q4k_legacy_xsmem",         2,  1, 1,  32, 1, true,  false, false},
  {"q4k_4row_128thr_u32",      2,  4, 1,  32, 1, false, false, false},
  {"q4k_4row_128thr_u32_xsmem",2,  4, 1,  32, 1, true,  false, false},
  {"q4k_4row_128thr_u128_xsmem",2, 4, 1, 128, 1, true,  false, false},
  {"q4k_4row_128thr_u128_xsmem_pf2",2, 4, 1, 128, 2, true, false, false},
  {"q4k_8row_256thr_u128_xsmem",2,  8, 1, 128, 1, true,  false, false},
  {"q4k_16row_512thr_u128_xsmem",2, 16, 1, 128, 1, true,  false, false},
  {"q4k_32row_1024thr_u128_xsmem",2, 32, 1, 128, 1, true,  false, false},
  {"q4k_16row_128thr_u128_quad_xsmem",2, 16, 1, 128, 1, true, true, false},
};
static const int NCFGS2 = sizeof(CFGS2) / sizeof(CFGS2[0]);

// launch one CFGS2 config for `iters` passes. Returns the thread count (0 = no dispatch).
static int launch_mem2(const Cfg2& c, int rows, const unsigned short* halfs,
                       const unsigned* words, const __half* x, float* out, int iters,
                       cudaStream_t st) {
  if (c.shape == 0) {
    const dim3 lgrid(4, rows/32);
    if (c.s == 1 && c.r == 32) {
      if (c.vec == 32) k_legacy_v<32, 1, false, false><<<lgrid, 32, 0, st>>>(halfs, x, out, iters, rows);
      else if (c.vec == 128 && c.pf == 1 && !c.xsmem && !c.al) k_legacy_v<128, 1, false, false><<<lgrid, 32, 0, st>>>(halfs, x, out, iters, rows);
      else if (c.vec == 128 && c.pf == 2 && !c.xsmem) k_legacy_v<128, 2, false, false><<<lgrid, 32, 0, st>>>(halfs, x, out, iters, rows);
      else if (c.vec == 128 && c.pf == 1 && !c.xsmem && c.al) k_legacy_v<128, 1, false, true><<<lgrid, 32, 0, st>>>(halfs, x, out, iters, rows);
      else if (c.vec == 128 && c.pf == 1 && c.xsmem) k_legacy_v<128, 1, true, false><<<lgrid, 32, 0, st>>>(halfs, x, out, iters, rows);
      else return 0;
      return 32;
    }
    const int thr = c.r*4*c.s;
    if (c.s == 2 && c.r == 8 && c.vec == 32) k_merge_v<8, 2, 32, 1, false><<<rows/8, thr, 0, st>>>(halfs, x, out, iters, rows);
    else if (c.s == 2 && c.r == 8 && c.vec == 128 && c.pf == 1) k_merge_v<8, 2, 128, 1, false><<<rows/8, thr, 0, st>>>(halfs, x, out, iters, rows);
    else if (c.s == 2 && c.r == 8 && c.vec == 128 && c.pf == 2) k_merge_v<8, 2, 128, 2, false><<<rows/8, thr, 0, st>>>(halfs, x, out, iters, rows);
    else if (c.s == 4 && c.r == 8 && c.vec == 32) k_merge_v<8, 4, 32, 1, false><<<rows/8, thr, 0, st>>>(halfs, x, out, iters, rows);
    else if (c.s == 4 && c.r == 8 && c.vec == 128 && !c.xsmem) k_merge_v<8, 4, 128, 1, false><<<rows/8, thr, 0, st>>>(halfs, x, out, iters, rows);
    else if (c.s == 4 && c.r == 8 && c.vec == 16 && c.xsmem) k_merge_v<8, 4, 16, 1, true><<<rows/8, thr, 0, st>>>(halfs, x, out, iters, rows);
    else if (c.s == 4 && c.r == 8 && c.vec == 128 && c.xsmem) k_merge_v<8, 4, 128, 1, true><<<rows/8, thr, 0, st>>>(halfs, x, out, iters, rows);
    else if (c.s == 4 && c.r == 16 && c.vec == 128 && c.xsmem) k_merge_v<16, 4, 128, 1, true><<<rows/16, thr, 0, st>>>(halfs, x, out, iters, rows);
    else if (c.s == 4 && c.r == 32 && c.vec == 128 && c.xsmem) k_merge_v<32, 4, 128, 1, true><<<rows/32, thr, 0, st>>>(halfs, x, out, iters, rows);
    else return 0;
    return thr;
  }
  if (c.shape == 1) {
    if (c.ctl) { k_coop_legacy<<<rows/2, 32, 0, st>>>(halfs, x, out, iters, rows); return 32; }
    if (c.s == 1) {
      const int thr = c.r*16;
      if (c.r == 2 && c.vec == 32) k_coop_v<2, 32, 1, false><<<rows/2, thr, 0, st>>>(halfs, x, out, iters, rows);
      else if (c.r == 2 && c.vec == 128 && c.pf == 1 && !c.xsmem) k_coop_v<2, 128, 1, false><<<rows/2, thr, 0, st>>>(halfs, x, out, iters, rows);
      else if (c.r == 2 && c.vec == 128 && c.pf == 2) k_coop_v<2, 128, 2, false><<<rows/2, thr, 0, st>>>(halfs, x, out, iters, rows);
      else if (c.r == 2 && c.vec == 16 && c.xsmem) k_coop_v<2, 16, 1, true><<<rows/2, thr, 0, st>>>(halfs, x, out, iters, rows);
      else if (c.r == 2 && c.vec == 128 && c.xsmem) k_coop_v<2, 128, 1, true><<<rows/2, thr, 0, st>>>(halfs, x, out, iters, rows);
      else if (c.r == 8 && c.vec == 128 && c.xsmem) k_coop_v<8, 128, 1, true><<<rows/8, thr, 0, st>>>(halfs, x, out, iters, rows);
      else if (c.r == 16 && c.vec == 128 && c.xsmem) k_coop_v<16, 128, 1, true><<<rows/16, thr, 0, st>>>(halfs, x, out, iters, rows);
      else if (c.r == 32 && c.vec == 128 && c.xsmem) k_coop_v<32, 128, 1, true><<<rows/32, thr, 0, st>>>(halfs, x, out, iters, rows);
      else return 0;
      return thr;
    }
    if (c.s == 2) {
      const int thr = c.r*32;
      if (c.r == 1 && c.vec == 128) k_coop_s2<1, 128, 1, false><<<rows, thr, 0, st>>>(halfs, x, out, iters, rows);
      else if (c.r == 4 && c.vec == 128 && c.xsmem) k_coop_s2<4, 128, 1, true><<<rows/4, thr, 0, st>>>(halfs, x, out, iters, rows);
      else if (c.r == 8 && c.vec == 128 && c.xsmem) k_coop_s2<8, 128, 1, true><<<rows/8, thr, 0, st>>>(halfs, x, out, iters, rows);
      else if (c.r == 16 && c.vec == 128 && c.xsmem) k_coop_s2<16, 128, 1, true><<<rows/16, thr, 0, st>>>(halfs, x, out, iters, rows);
      else return 0;
      return thr;
    }
    if (c.s == 4) { k_coop_s4<8, 128, true><<<rows/8, 512, 0, st>>>(halfs, x, out, iters, rows); return 512; }
    // group-lane family
    const int thr = c.r*16;
    if (c.r == 2) k_coop_group<2, true><<<rows/2, thr, 0, st>>>(halfs, x, out, iters, rows);
    else if (c.r == 8) k_coop_group<8, true><<<rows/8, thr, 0, st>>>(halfs, x, out, iters, rows);
    else if (c.r == 16) k_coop_group<16, true><<<rows/16, thr, 0, st>>>(halfs, x, out, iters, rows);
    else if (c.r == 32) k_coop_group<32, true><<<rows/32, thr, 0, st>>>(halfs, x, out, iters, rows);
    else return 0;
    return thr;
  }
  // q4k
  if (c.ctl) { k_q4k_legacy<<<rows, 32, 0, st>>>(words, x, out, iters, rows); return 32; }
  const int thr = c.r*32;
  if (c.r == 1 && c.vec == 16) k_q4k_v<1, 16, 1, false><<<rows, thr, 0, st>>>(words, x, out, iters, rows);
  else if (c.r == 1 && c.vec == 128 && c.pf == 1 && !c.xsmem) k_q4k_v<1, 128, 1, false><<<rows, thr, 0, st>>>(words, x, out, iters, rows);
  else if (c.r == 1 && c.vec == 128 && c.pf == 2) k_q4k_v<1, 128, 2, false><<<rows, thr, 0, st>>>(words, x, out, iters, rows);
  else if (c.r == 1 && c.vec == 32 && c.xsmem) k_q4k_v<1, 32, 1, true><<<rows, thr, 0, st>>>(words, x, out, iters, rows);
  else if (c.r == 4 && c.vec == 32 && !c.xsmem) k_q4k_v<4, 32, 1, false><<<rows/4, thr, 0, st>>>(words, x, out, iters, rows);
  else if (c.r == 4 && c.vec == 32 && c.xsmem) k_q4k_v<4, 32, 1, true><<<rows/4, thr, 0, st>>>(words, x, out, iters, rows);
  else if (c.r == 4 && c.vec == 128 && c.pf == 1 && c.xsmem) k_q4k_v<4, 128, 1, true><<<rows/4, thr, 0, st>>>(words, x, out, iters, rows);
  else if (c.r == 4 && c.vec == 128 && c.pf == 2 && c.xsmem) k_q4k_v<4, 128, 2, true><<<rows/4, thr, 0, st>>>(words, x, out, iters, rows);
  else if (c.r == 8 && c.vec == 128 && c.xsmem) k_q4k_v<8, 128, 1, true><<<rows/8, thr, 0, st>>>(words, x, out, iters, rows);
  else if (c.r == 16 && c.vec == 128 && c.xsmem && c.al) k_q4k_qv<16, true><<<rows/16, 128, 0, st>>>(words, x, out, iters, rows);
  else if (c.r == 16 && c.vec == 128 && c.xsmem) k_q4k_v<16, 128, 1, true><<<rows/16, thr, 0, st>>>(words, x, out, iters, rows);
  else if (c.r == 32 && c.vec == 128 && c.xsmem) k_q4k_v<32, 128, 1, true><<<rows/32, thr, 0, st>>>(words, x, out, iters, rows);
  else return 0;
  return thr;
}

// set-sized L2-resident streaming read: the achievable-bandwidth ceiling control
static void run_bw_read(int shape, const void* buf, size_t bytes, cudaStream_t st,
                        cudaEvent_t e0, cudaEvent_t e1) {
  float* out;
  cudaMalloc(&out, 4);
  int n4 = (int)(bytes / 16);
  int passes = 200;
  k_bw_read<<<512, 256, 0, st>>>((const float4*)buf, out, n4);
  cudaStreamSynchronize(st);
  cudaEventRecord(e0, st);
  for (int p = 0; p < passes; p++) k_bw_read<<<512, 256, 0, st>>>((const float4*)buf, out, n4);
  cudaEventRecord(e1, st);
  cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  double us = ms * 1e3 / passes;
  const char* n = shape == 0 ? "partial" : shape == 1 ? "coop" : "q4k";
  printf("bw read %-8s set=%.2f MB -> %.2f us/pass -> %.2f TB/s (weight bytes only)\n",
         n, bytes / 1e6, us, bytes / us / 1e6);
  cudaFree(out);
}

int main(int argc, char** argv) {
  const char* mode = "mem";
  const char* only = nullptr;
  const char* shape_s = "partial";
  int iters = 2000, reps = 5;
  int rows_arg = 0;
  bool bw = false;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--mode")) mode = argv[++i];
    else if (!strcmp(argv[i], "--only")) only = argv[++i];
    else if (!strcmp(argv[i], "--iters")) iters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--reps")) reps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--shape")) shape_s = argv[++i];
    else if (!strcmp(argv[i], "--rows")) rows_arg = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--bw")) bw = true;
  }
  int shape = !strcmp(shape_s, "coop") ? 1 : !strcmp(shape_s, "q4k") ? 2 : 0;
  bool dequant = !strcmp(mode, "dequant");
  bool compute = !strcmp(mode, "dot") || dequant;

  // deterministic packed weights + x
  const int HW = ROWS * KB * HWPB;
  unsigned short* halfs; __half* x; float* buf; float* ops; float* scratch; float* partbuf;
  cudaMalloc(&halfs, HW * sizeof(unsigned short));
  cudaMalloc(&x, K * sizeof(__half));
  cudaMalloc(&buf, (size_t)ROWS * PARTS * sizeof(float));
  cudaMalloc(&scratch, (size_t)ROWS * sizeof(float));
  cudaMalloc(&partbuf, (size_t)ROWS * PARTS * sizeof(float));
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
  printf("device=%s mode=%s shape=%s iters=%d reps=%d\n", prop.name, mode, shape_s, iters, reps);

  if (shape == 0) {
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
  }

  // ---- MC2 extension: load-pattern sweep rows (mem mode only) ----
  if (!compute) {
    const int prow = rows_arg ? rows_arg : ROWS;          // partial active rows
    if (shape == 0) {
      printf("-- partial vec/xsmem sweep (rows=%d) --\n", prow);
      for (int ci = 0; ci < NCFGS2; ci++) {
        const Cfg2& c = CFGS2[ci];
        if (c.shape != 0) continue;
        if (only && strcmp(only, c.name)) continue;
        int rpb = (c.s == 1 && c.r == 32) ? 32 : c.r;
        if (prow % rpb != 0 || (rows_arg && (prow & (prow - 1)) != 0)) {
          printf("  %-20s SKIP (rows=%d not divisible by %d%s)\n", c.name, prow, rpb,
                 rows_arg && (prow & (prow - 1)) != 0 ? " or not a power of 2" : "");
          continue;
        }
        const bool leg = (c.s == 1 && c.r == 32);           // legacy family stores per-part partials
        float* obuf = leg ? partbuf : scratch;
        const size_t on = leg ? (size_t)ROWS * PARTS : (size_t)ROWS;
        launch_mem2(c, prow, halfs, nullptr, x, obuf, 64, st); cudaStreamSynchronize(st);   // warmup
        double best = 1e30;
        for (int r = 0; r < reps; r++) {
          cudaEventRecord(e0, st);
          launch_mem2(c, prow, halfs, nullptr, x, obuf, iters, st);
          cudaEventRecord(e1, st);
          cudaEventSynchronize(e1);
          float ms; cudaEventElapsedTime(&ms, e0, e1);
          best = std::min(best, (double)ms);
        }
        double us_per_pass = best * 1e3 / iters;
        double gb = (double)prow * KB * 210.0 / us_per_pass / 1e6;
        double macs = (double)prow * K;
        printf("%-20s %5d %3d %10.2f %10.2f %10.0f %s\n", c.name, prow / rpb,
               (c.s == 1 && c.r == 32) ? 32 : c.r*4*c.s, us_per_pass, gb, macs / us_per_pass / 1e3,
               c.xsmem ? "(x smem)" : c.al ? "(aligned)" : "(x l2)");
        // numerical spot check vs k_legacy sum-of-partials over the active rows
        cudaMemset(buf, 0, (size_t)ROWS * PARTS * sizeof(float));
        k_legacy<<<dim3(PARTS, prow/32), 32, 0, st>>>(halfs, x, buf, 1);
        launch_mem2(c, prow, halfs, nullptr, x, obuf, 1, st);
        cudaStreamSynchronize(st);
        float *h0 = (float*)malloc((size_t)ROWS * PARTS * sizeof(float));
        float *h1 = (float*)malloc(on * sizeof(float));
        cudaMemcpy(h0, buf, (size_t)ROWS * PARTS * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h1, obuf, on * sizeof(float), cudaMemcpyDeviceToHost);
        double maxdiff = 0.0, reldiff = 0.0;
        if (leg) {
          if (c.al) {
            // interleaved part map permutes blocks across parts; only the row total is comparable
            for (int row = 0; row < prow; row++) {
              double s1 = 0.0, s0 = 0.0;
              for (int p = 0; p < PARTS; p++) { s1 += h1[row*PARTS + p]; s0 += h0[row*PARTS + p]; }
              double d = fabs(s1 - s0);
              maxdiff = std::max(maxdiff, d);
              reldiff = std::max(reldiff, d / std::max(fabs(s0), 1.0));
            }
          } else {
            // legacy family stores per-part partials; compare per-part against k_legacy
            for (int row = 0; row < prow; row++)
              for (int p = 0; p < PARTS; p++) {
                double d = fabs((double)(h1[row*PARTS + p] - h0[row*PARTS + p]));
                maxdiff = std::max(maxdiff, d);
                reldiff = std::max(reldiff, d / std::max(fabs((double)h0[row*PARTS + p]), 1.0));
              }
          }
        } else {
          for (int row = 0; row < prow; row++) {
            float sum = 0.0f;
            for (int p = 0; p < PARTS; p++) sum += h0[row*PARTS + p];
            double d = fabs((double)(h1[row] - sum));
            maxdiff = std::max(maxdiff, d);
            reldiff = std::max(reldiff, d / std::max(fabs((double)sum), 1.0));
          }
        }
        printf("  check: max|diff| = %.4e  max|rel| = %.3e\n", maxdiff, reldiff);
        free(h0); free(h1);
      }
      if (bw) run_bw_read(0, halfs, (size_t)ROWS * KB * 210, st, e0, e1);
    }
    if (shape == 1 || shape == 2) {
      const int nrows = shape == 1 ? COOP_ROWS : Q4K_ROWS;
      const int kk = shape == 1 ? COOP_K : Q4K_K;
      const int kb = shape == 1 ? COOP_KB : Q4K_KB;
      const int wpb = shape == 1 ? HWPB : Q4K_WORDS_PER_BLOCK;
      const int qrows = rows_arg ? rows_arg : nrows;
      unsigned short* halfs2; unsigned* words2; __half* x2;
      float* out; float* ref; float* cmp;
      cudaMalloc(&halfs2, ((size_t)nrows * kb * wpb + 8) * 2);   // +16B pad for u128 over-read
      cudaMalloc(&words2, ((size_t)nrows * kb * wpb + 8) * 4);
      cudaMalloc(&x2, (size_t)kk * 2);
      cudaMalloc(&out, (size_t)nrows * 4);
      cudaMalloc(&ref, (size_t)nrows * 4);
      cudaMalloc(&cmp, (size_t)nrows * 4);
      unsigned short* hh2 = (unsigned short*)malloc(((size_t)nrows * kb * wpb + 8) * 2);
      unsigned* hw2 = (unsigned*)malloc(((size_t)nrows * kb * wpb + 8) * 4);
      __half* hx2 = (__half*)malloc((size_t)kk * 2);
      for (int i = 0; i < nrows * kb * wpb; i++) { seed = seed*1664525u + 1013904223u; hh2[i] = (unsigned short)((seed >> 16) & 0x7bff); }
      for (int i = 0; i < kk; i++) { seed = seed*1664525u + 1013904223u; hx2[i] = __float2half((float)(seed % 1000) / 500.0f - 1.0f); }
      for (int b = 0; b < nrows * kb; b++) {
        for (int w = 0; w < wpb; w++) {
          unsigned v;
          if (w == 0) {
            unsigned short lo = (unsigned short)((seed = seed*1664525u + 1013904223u) >> 16) & 0x7bff;
            unsigned short hi = (unsigned short)((seed = seed*1664525u + 1013904223u) >> 16) & 0x7bff;
            v = (unsigned)hi << 16 | lo;
          } else {
            v = (seed = seed*1664525u + 1013904223u);
          }
          hw2[(size_t)b * wpb + w] = v;
        }
      }
      cudaMemcpy(halfs2, hh2, ((size_t)nrows * kb * wpb + 8) * 2, cudaMemcpyHostToDevice);
      cudaMemcpy(words2, hw2, ((size_t)nrows * kb * wpb + 8) * 4, cudaMemcpyHostToDevice);
      cudaMemcpy(x2, hx2, (size_t)kk * 2, cudaMemcpyHostToDevice);

      printf("-- %s sweep (rows=%d) --\n", shape_s, qrows);
      for (int ci = 0; ci < NCFGS2; ci++) {
        const Cfg2& c = CFGS2[ci];
        if (c.shape != shape) continue;
        if (only && strcmp(only, c.name)) continue;
        int rpb = c.r;
        if (qrows % rpb != 0 || (rows_arg && (qrows & (qrows - 1)) != 0)) {
          printf("  %-20s SKIP (rows=%d not divisible by %d%s)\n", c.name, qrows, rpb,
                 rows_arg && (qrows & (qrows - 1)) != 0 ? " or not a power of 2" : "");
          continue;
        }
        launch_mem2(c, qrows, halfs2, words2, x2, out, 32, st); cudaStreamSynchronize(st);   // warmup
        double best = 1e30;
        for (int r = 0; r < reps; r++) {
          cudaEventRecord(e0, st);
          launch_mem2(c, qrows, halfs2, words2, x2, out, iters, st);
          cudaEventRecord(e1, st);
          cudaEventSynchronize(e1);
          float ms; cudaEventElapsedTime(&ms, e0, e1);
          best = std::min(best, (double)ms);
        }
        double us_per_pass = best * 1e3 / iters;
        double wb = (double)qrows * kb * (shape == 1 ? 210.0 : 144.0);
        double gb = wb / us_per_pass / 1e6;
        double macs = (double)qrows * kk;
        int thr = c.ctl ? 32 :
                  (shape == 1 ? (c.s == 1 ? c.r*16 : c.s == 2 ? c.r*32 : c.r*64) :
                   (c.al ? 128 : c.r*32));
        printf("%-20s %5d %3d %10.2f %10.2f %10.0f %s\n", c.name, qrows / rpb, thr, us_per_pass, gb,
               macs / us_per_pass / 1e3, c.ctl ? "(control)" : c.xsmem ? "(x smem)" : "(x l2)");
        if (!c.ctl) {   // one-pass spot check vs the installed control replica
          if (shape == 1) k_coop_legacy<<<qrows/2, 32, 0, st>>>(halfs2, x2, ref, 1, qrows);
          else k_q4k_legacy<<<qrows, 32, 0, st>>>(words2, x2, ref, 1, qrows);
          launch_mem2(c, qrows, halfs2, words2, x2, cmp, 1, st);
          cudaStreamSynchronize(st);
          float *hr = (float*)malloc((size_t)nrows * 4);
          float *hc = (float*)malloc((size_t)nrows * 4);
          cudaMemcpy(hr, ref, (size_t)nrows * 4, cudaMemcpyDeviceToHost);
          cudaMemcpy(hc, cmp, (size_t)nrows * 4, cudaMemcpyDeviceToHost);
          double maxdiff = 0.0, reldiff = 0.0;
          for (int row = 0; row < qrows; row++) {
            double d = fabs((double)(hc[row] - hr[row]));
            maxdiff = std::max(maxdiff, d);
            reldiff = std::max(reldiff, d / std::max(fabs((double)hr[row]), 1.0));
          }
          printf("  check: max|diff| = %.4e  max|rel| = %.3e\n", maxdiff, reldiff);
          free(hr); free(hc);
        }
      }
      if (bw) run_bw_read(shape, shape == 1 ? (const void*)halfs2 : (const void*)words2,
                          (size_t)nrows * kb * (shape == 1 ? 210 : 144), st, e0, e1);
      free(hh2); free(hw2); free(hx2);
    }
  }
  cudaDeviceSynchronize();
  return 0;
}
