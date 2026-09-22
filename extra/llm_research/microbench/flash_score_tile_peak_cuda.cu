// flash_score_tile_peak_cuda.cu - wmma_peak-style structural microbench for the whole-cache
// flash score tile (shared UOp builder flash_decode_attention.py:92, installed row
// flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128) at max_context=4608.
//
// Scope C of docs/task_workflow/input/decode-gemv-efficiency-forward-scope-20260803.md:
// isolate the steady-state score structure (fdot2 dots + shfl lane reduce + online softmax +
// PV merge) with hoisted operand setup, NACC independent accumulators over the score
// computation, zero loads in the hot loop where the structural variant allows, and sweep the
// tile geometry: LANES, WARPS (query_group_size), TK, and the LDS staging shape
// (DECODE_STAGE_COALESCE surface, SW), at max_context=4608. Floor: llama-class ~3.2 us per
// score kernel.
//
// Real tile facts reproduced here (rendered CUDA, Tc=4608): grid (48 splits, 8 kvh, NG query
// groups), block (LANES, WARPS), THREADS = LANES*WARPS, head = kvh*G + qg*QG + warp,
// per split window LPER aligned tokens in NB = ceil(LPER/TK) tiles (runtime nb, like the real
// kernel's runtime Ridx3), per token per lane RP = Hd/(2*LANES) half2 fdot2 dots (2 FMA each),
// log2(LANES)-step __shfl_xor_sync reduce, scale 1/sqrt(Hd), online softmax (max, exp2,
// correction/probability), PV merge over R = Hd/LANES dims, den/mx update, 2 __syncthreads
// per tile in TILE mode. Query is loaded per token inside the token loop in the real render
// (data1_4096 reads), so qhoist=0 reproduces that; qhoist=1 hoists q into registers.
//
// MODE template:
//   MODE=0 REG  - one tile's K/V operands register-resident per lane, timed loop has zero
//                 loads and zero LDS; the tile score is replayed for all NB tiles (steady-state
//                 rate, wmma_peak-style operand reuse).
//   MODE=1 LDS  - whole split window staged once into LDS before timing (cache operand setup
//                 hoisted), timed loop = NB tiles of score reading LDS only.
//   MODE=2 TILE - per-tile LDG->STS staging from the global cache (L2-resident) + 2 barriers
//                 per tile, score from LDS. Reproduces the installed-row structure; the sweep
//                 anchor at max_context=4608 is the campaign's whole-cache shape (the installed
//                 row's 7.6 us at Tc=513 is an in-loop per-launch median, reproduced in-loop at
//                 7.52 us by extra/llm_research/decode/flash_score_tile_nv_timing.py; the warm
//                 standalone d512-shape peak is 2.334 us (NACC=1) / 1.945 us (NACC=8), and the
//                 cold single-launch d512-shape pass is 6.75 us).
// SW is the staging shape (DECODE_STAGE_COALESCE surface): 0 = plain path idx = stage*THREADS+tid,
// 1..8 = CooperativeStageLaneMap width (idx = (stage*THREADS+tid)*SW + w, stages = TK*Hd/(THREADS*SW)).
// SW affects MODE=2 (per-tile staging) and MODE=1 (hoisted window staging); MODE=0 has no LDS.
//
// Purity before believing a number (cuobjdump --dump-sass, nvdisasm ships inside the triton
// package if the toolkit lacks it): MODE=0 hot loop has only FFMA/FADD/SHFL/MUFU.EX2 plus the
// gated keep-alive STG (never taken), zero LDG/LDS/STS. MODE=1 has LDS reads only. MODE=2 has
// LDG (L2-resident cache) + STS + LDS + BAR in the per-tile phase. --ptxas-options=-v: 0 spills
// in the hot loop at the anchor geometry.
//
// Build: export PATH=/usr/local/cuda-13.2/bin:$PATH
//   nvcc -O3 -arch=sm_120 -Xptxas -v flash_score_tile_peak_cuda.cu -o flash_score_tile_peak_cuda
// Run (max_context=4608 sweep): ./flash_score_tile_peak_cuda --sweep --iters 2000
// Run (d512 calibration): ./flash_score_tile_peak_cuda --lper 16 --nvalid 11 --lanes 32 --warps 4 --tk 16 --sw 1 --mode 2
// Run (single config):     ./flash_score_tile_peak_cuda --lanes 32 --warps 4 --tk 16 --sw 1 --mode 2 --qhoist 0
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HD 128
#define HKV 8
#define SPLITS 48
#define MAXC 4608
#define LPER_MAX 96                // aligned tokens per split at Tc=4608
#define G 4                        // Hq/Hkv (Hq=32)
#define LOG2E 1.4426950408889634f
#define INV_SQRT_HD 0.08838834764831843f

__device__ __forceinline__ float fdot2(float acc, __half2 a, __half2 b) {
  // matches CUDARenderer's lowering (unpacked half -> fp32 FFMA pairs)
  return acc + __half2float(__low2half(a)) * __half2float(__low2half(b))
             + __half2float(__high2half(a)) * __half2float(__high2half(b));
}

// lane reduce over LANES lanes: log2(LANES) xor-shuffle steps. LANES=64 spans two hardware
// warps and needs one cross-warp shared round trip per reduce (honest structural cost).
template<int LANES>
__device__ __forceinline__ float lane_reduce(float v, int lane, float* cross, int yslot) {
  if constexpr (LANES == 16) {
    v += __shfl_xor_sync(0xffffu, v, 8); v += __shfl_xor_sync(0xffffu, v, 4);
    v += __shfl_xor_sync(0xffffu, v, 2); v += __shfl_xor_sync(0xffffu, v, 1);
    return v;
  } else if constexpr (LANES == 32) {
    v += __shfl_xor_sync(0xffffffffu, v, 16); v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);  v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
  } else { // LANES == 64
    int half = lane >> 5, l32 = lane & 31;
    v += __shfl_xor_sync(0xffffffffu, v, 16); v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);  v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    if (l32 == 0) cross[(half << 1) | yslot] = v;
    __syncthreads();
    v += cross[((1 - half) << 1) | yslot];
    __syncthreads();
    return v;
  }
}

// NACC independent score chains cover the shfl/FMA latency; the online-softmax state
// (mx/den/acc) stays serial across tokens exactly like the real kernel.
template<int LANES, int WARPS, int TK, int SW, int MODE, int LPER, int NACC>
__global__ __launch_bounds__(LANES * WARPS) void flash_score_tile_peak(
    float* out, half* __restrict__ q, half* __restrict__ cache, int iters, int nvalid, int qhoist) {
  constexpr int THREADS = LANES * WARPS;
  constexpr int NG = G / WARPS;
  constexpr int RP = HD / (2 * LANES);
  constexpr int R = HD / LANES;
  constexpr int STAGES = (SW > 0) ? (TK * HD / (THREADS * SW)) : ((TK * HD + THREADS - 1) / THREADS);
  constexpr int WINC = LPER * HD;                 // staged-once window halves per buffer (MODE=1)
  constexpr int NB = (LPER + TK - 1) / TK;        // tiles per split window

  const int lane = threadIdx.x;                   // 0..LANES-1
  const int warp = threadIdx.y;                   // 0..WARPS-1 (QG)
  const int tid = warp * LANES + lane;
  const int head = blockIdx.y * G + blockIdx.z * WARPS + warp;
  const int kvh = blockIdx.y;
  const int split = blockIdx.x;
  const int tok_base_split = split * LPER;        // window origin of this block

  // MODE=1 holds the whole split window in dynamic shared memory (48 KB at max_context);
  // MODE=2 holds only the per-tile buffers (matches the installed 4 KB x 2 LDS shape).
  extern __shared__ __align__(16) half dyn[];
  half* winK = nullptr;
  half* winV = nullptr;
  if constexpr (MODE == 1) { winK = dyn; winV = dyn + WINC; }
  // per-tile staging buffers (only read/written in MODE=2; harmless static footprint otherwise)
  __shared__ __align__(16) half bufK[TK * HD];
  __shared__ __align__(16) half bufV[TK * HD];
  __shared__ float cross[2 * WARPS];

  // hoisted query operands (score operand setup out of the timed loop; qhoist=1)
  __half2 qp[RP];
  if (qhoist) {
    #pragma unroll
    for (int p = 0; p < RP; p++) {
      int e = p * (2 * LANES) + lane * 2;
      qp[p] = make_half2(q[head * HD + e], q[head * HD + e + 1]);
    }
  }

  // MODE=0: one tile's K/V register-resident per lane
  __half2 regK[TK][RP];
  __half regV[TK][R];

  // cache operand setup, hoisted out of the timed loop
  if constexpr (MODE == 0) {
    #pragma unroll
    for (int t = 0; t < TK; t++)
      #pragma unroll
      for (int p = 0; p < RP; p++) {
        int e = t * HD + p * (2 * LANES) + lane * 2;
        int tok = tok_base_split + t;
        half kv = (t < nvalid) ? cache[(0 * HKV + kvh) * MAXC * HD + tok * HD + e] : __float2half(0.f);
        regK[t][p] = make_half2(kv, (t < nvalid) ? cache[(0 * HKV + kvh) * MAXC * HD + tok * HD + e + 1] : __float2half(0.f));
      }
    #pragma unroll
    for (int t = 0; t < TK; t++)
      #pragma unroll
      for (int d = 0; d < R; d++) {
        int e = t * HD + lane * R + d;
        int tok = tok_base_split + t;
        regV[t][d] = (t < nvalid) ? cache[(1 * HKV + kvh) * MAXC * HD + tok * HD + e] : __float2half(0.f);
      }
  } else {
    // window staging: whole LPER window once (MODE=1) or per-tile inside the timed loop (MODE=2)
    if constexpr (MODE == 1) {
      for (int e = tid; e < WINC; e += THREADS) {
        int tok = tok_base_split + e / HD;
        int el = e % HD;
        winK[e] = (e / HD < nvalid) ? cache[(0 * HKV + kvh) * MAXC * HD + tok * HD + el] : __float2half(0.f);
        winV[e] = (e / HD < nvalid) ? cache[(1 * HKV + kvh) * MAXC * HD + tok * HD + el] : __float2half(0.f);
      }
      __syncthreads();   // window fully staged before any score read
    }
  }

  // score state (identical serial chain to the real kernel)
  float acc[R];
  #pragma unroll
  for (int d = 0; d < R; d++) acc[d] = 0.0f;
  float den = 0.0f, mx = -INFINITY;

  auto qpair = [&](int p) -> __half2 {
    if (qhoist) return qp[p];
    int e = p * (2 * LANES) + lane * 2;
    return make_half2(q[head * HD + e], q[head * HD + e + 1]);
  };
  auto kpair = [&](int b, int t, int p) -> __half2 {
    if constexpr (MODE == 0) return regK[t][p];
    else if constexpr (MODE == 1) {
      int e = b * TK * HD + t * HD + p * (2 * LANES) + lane * 2;
      return make_half2(winK[e], winK[e + 1]);
    } else {
      int e = t * HD + p * (2 * LANES) + lane * 2;
      return make_half2(bufK[e], bufK[e + 1]);
    }
  };
  auto vval = [&](int b, int t, int d) -> float {
    if constexpr (MODE == 0) return __half2float(regV[t][d]);
    else if constexpr (MODE == 1) return __half2float(winV[b * TK * HD + t * HD + lane * R + d]);
    else return __half2float(bufV[t * HD + lane * R + d]);
  };

  for (int it = 0; it < iters; it++) {
    for (int b = 0; b < NB; b++) {
      if constexpr (MODE == 2) {
        // per-tile staging: LDG (L2-resident) -> STS, shape per SW (DECODE_STAGE_COALESCE)
        if constexpr (SW == 0) {
          #pragma unroll
          for (int s = 0; s < STAGES; s++) {
            int idx = s * THREADS + tid;
            int tok = tok_base_split + b * TK + idx / HD;
            int el = idx % HD;
            half kv0 = (b * TK + idx / HD < nvalid) ? cache[(0 * HKV + kvh) * MAXC * HD + tok * HD + el] : __float2half(0.f);
            half kv1 = (b * TK + idx / HD < nvalid) ? cache[(1 * HKV + kvh) * MAXC * HD + tok * HD + el] : __float2half(0.f);
            bufK[idx] = kv0;
            bufV[idx] = kv1;
          }
        } else {
          #pragma unroll
          for (int s = 0; s < STAGES; s++)
            #pragma unroll
            for (int w = 0; w < SW; w++) {
              int idx = (s * THREADS + tid) * SW + w;
              int tok = tok_base_split + b * TK + idx / HD;
              int el = idx % HD;
              half kv0 = (b * TK + idx / HD < nvalid) ? cache[(0 * HKV + kvh) * MAXC * HD + tok * HD + el] : __float2half(0.f);
              half kv1 = (b * TK + idx / HD < nvalid) ? cache[(1 * HKV + kvh) * MAXC * HD + tok * HD + el] : __float2half(0.f);
              bufK[idx] = kv0;
              bufV[idx] = kv1;
            }
        }
        __syncthreads();
      }
      // score phase: NACC independent dot chains, then serial softmax/PV merges
      for (int t = 0; t < TK; t += NACC) {
        float score[NACC];
        #pragma unroll
        for (int j = 0; j < NACC; j++) {
          float dot = 0.0f;
          #pragma unroll
          for (int p = 0; p < RP; p++) dot = fdot2(dot, qpair(p), kpair(b, t + j, p));
          dot = lane_reduce<LANES>(dot, lane, cross, warp);
          score[j] = dot * INV_SQRT_HD;
        }
        #pragma unroll
        for (int j = 0; j < NACC; j++) {
          int gtok = b * TK + (t + j);
          bool valid = (gtok < nvalid);
          float sc = valid ? score[j] : -INFINITY;
          float nm = fmaxf(mx, sc);
          float corr = valid ? exp2f((mx - nm) * LOG2E) : 1.0f;
          float prob = valid ? exp2f((sc - nm) * LOG2E) : 0.0f;
          #pragma unroll
          for (int d = 0; d < R; d++) acc[d] = acc[d] * corr + prob * vval(b, t + j, d);
          den = den * corr + prob;
          mx = nm;
        }
      }
      if constexpr (MODE == 2) __syncthreads();
    }
  }

  float s = 0.0f;
  #pragma unroll
  for (int d = 0; d < R; d++) s += acc[d];
  s += den + mx;
  if (s == 1234.5f) out[0] = s;   // keep live, never taken
}

// --------------------------------------------------------------------------------------------

static int L[] = {16, 32, 64};
static int W[] = {1, 2, 4};
static int T[] = {8, 16, 32};
static int S[] = {0, 1, 2, 4, 8};

static int g_verbose = 0;

template<int LANES, int WARPS, int TK, int SW, int MODE, int LPER, int NACC>
static double run_cfg(half* d_q, half* d_cache, int iters, int nvalid, int qhoist, float* d_out) {
  int ng = G / WARPS;
  dim3 grid(SPLITS, HKV, ng), block(LANES, WARPS);
  size_t shmem = (MODE == 1) ? 2 * LPER * HD * sizeof(half) : 0;
  auto kfn = flash_score_tile_peak<LANES, WARPS, TK, SW, MODE, LPER, NACC>;
  cudaFuncAttributes attr; cudaFuncGetAttributes(&attr, kfn);
  cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
  cudaError_t setattr_err = cudaGetLastError();
  if (g_verbose) {
    fprintf(stderr, "DBG cfg l=%d w=%d tk=%d sw=%d m=%d lper=%d: static=%d regs=%d maxthreads=%d shmem_req=%zu setattr=%s\n",
            LANES, WARPS, TK, SW, MODE, LPER, (int)attr.sharedSizeBytes, (int)attr.numRegs, (int)attr.maxThreadsPerBlock,
            shmem, cudaGetErrorString(setattr_err));
  }
  kfn<<<grid, block, shmem, 0>>>(d_out, d_q, d_cache, 64, nvalid, qhoist);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "warmup launch error: %s (grid=(%d,%d,%d) block=(%d,%d,%d) shmem=%zu)\n",
            cudaGetErrorString(err), grid.x, grid.y, grid.z, block.x, block.y, block.z, shmem);
    exit(1);
  }
  cudaEvent_t s, e;
  cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  kfn<<<grid, block, shmem, 0>>>(d_out, d_q, d_cache, iters, nvalid, qhoist);
  cudaEventRecord(e);
  cudaError_t sync_err = cudaDeviceSynchronize();
  if (sync_err != cudaSuccess) { fprintf(stderr, "timed launch error: %s\n", cudaGetErrorString(sync_err)); exit(1); }
  float ms; cudaEventElapsedTime(&ms, s, e);
  if (g_verbose) fprintf(stderr, "DBG elapsed_ms=%.6f\n", (double)ms);
  cudaEventDestroy(s); cudaEventDestroy(e);
  return ms * 1e3f / (float)iters;   // us per score kernel
}

static int arg_int(int argc, char** argv, const char* key, int def) {
  for (int i = 1; i < argc - 1; i++) if (strcmp(argv[i], key) == 0) return atoi(argv[i + 1]);
  return def;
}
static bool arg_flag(int argc, char** argv, const char* key) {
  for (int i = 1; i < argc; i++) if (strcmp(argv[i], key) == 0) return true;
  return false;
}

// nested-template dispatch over (lanes, warps, tk, sw, mode, lper) -> <LANES,...,8>
struct BenchArgs { half* q; half* c; int it, nv, qh, nacc; float* o; double us; };
template<int L, int W, int T, int SW_, int M, int LP> static void r_d0(void* p) {
  auto* a = (BenchArgs*)p;
  a->us = run_cfg<L, W, T, SW_, M, LP, 8>(a->q, a->c, a->it, a->nv, a->qh, a->o);
}
// NACC calibration path (anchor geometry only): independent-accumulator count sweep.
template<int L, int W, int T, int SW_, int M, int LP>
static void r_nacc_lp(int nacc, BenchArgs* a) {
  switch (nacc) {
    case 1: a->us = run_cfg<L, W, T, SW_, M, LP, 1>(a->q, a->c, a->it, a->nv, a->qh, a->o); break;
    case 2: a->us = run_cfg<L, W, T, SW_, M, LP, 2>(a->q, a->c, a->it, a->nv, a->qh, a->o); break;
    case 4: a->us = run_cfg<L, W, T, SW_, M, LP, 4>(a->q, a->c, a->it, a->nv, a->qh, a->o); break;
    default: a->us = run_cfg<L, W, T, SW_, M, LP, 8>(a->q, a->c, a->it, a->nv, a->qh, a->o); break;
  }
}
template<int L, int W, int T, int SW_, int M>
static void r_nacc_anchor(int nacc, int lper, BenchArgs* a) {
  switch (lper) {
    case 16: r_nacc_lp<L, W, T, SW_, M, 16>(nacc, a); break;
    default: r_nacc_lp<L, W, T, SW_, M, 96>(nacc, a); break;
  }
}
template<int L, int W, int T, int SW_, int M> static void r_lper(int lper, void* p) {
  switch (lper) { case 16: r_d0<L, W, T, SW_, M, 16>(p); break; default: r_d0<L, W, T, SW_, M, 96>(p); break; }
}
template<int L, int W, int T, int SW_> static void r_mode(int mode, int lper, void* p) {
  switch (mode) {
    case 0: r_lper<L, W, T, SW_, 0>(lper, p); break;
    case 1: r_lper<L, W, T, SW_, 1>(lper, p); break;
    default: r_lper<L, W, T, SW_, 2>(lper, p); break;
  }
}
template<int L, int W, int T> static void r_sw(int sw, int mode, int lper, void* p) {
  switch (sw) {
    case 0: r_mode<L, W, T, 0>(mode, lper, p); break;
    case 1: r_mode<L, W, T, 1>(mode, lper, p); break;
    case 2: r_mode<L, W, T, 2>(mode, lper, p); break;
    case 4: r_mode<L, W, T, 4>(mode, lper, p); break;
    default: r_mode<L, W, T, 8>(mode, lper, p); break;
  }
}
template<int L, int W> static void r_tk(int tk, int sw, int mode, int lper, void* p) {
  switch (tk) { case 8: r_sw<L, W, 8>(sw, mode, lper, p); break; case 16: r_sw<L, W, 16>(sw, mode, lper, p); break; default: r_sw<L, W, 32>(sw, mode, lper, p); break; }
}
template<int L> static void r_warps(int warps, int tk, int sw, int mode, int lper, void* p) {
  switch (warps) { case 1: r_tk<L, 1>(tk, sw, mode, lper, p); break; case 2: r_tk<L, 2>(tk, sw, mode, lper, p); break; default: r_tk<L, 4>(tk, sw, mode, lper, p); break; }
}
static void r_lanes(int lanes, int warps, int tk, int sw, int mode, int lper, void* p) {
  switch (lanes) { case 16: r_warps<16>(warps, tk, sw, mode, lper, p); break; case 32: r_warps<32>(warps, tk, sw, mode, lper, p); break; default: r_warps<64>(warps, tk, sw, mode, lper, p); break; }
}
#undef R_D

int main(int argc, char** argv) {
  int iters = arg_int(argc, argv, "--iters", 2000);
  int lper  = arg_int(argc, argv, "--lper", LPER_MAX);
  int nvalid = arg_int(argc, argv, "--nvalid", lper);
  int qhoist = arg_int(argc, argv, "--qhoist", 1);
  int nacc   = arg_int(argc, argv, "--nacc", 8);
  bool sweep = arg_flag(argc, argv, "--sweep");
  g_verbose = arg_flag(argc, argv, "--verbose");

  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
  printf("# device %s sm_%d\n", prop.name, prop.major * 10 + prop.minor);
  printf("# iters=%d lper=%d nvalid=%d qhoist=%d\n", iters, lper, nvalid, qhoist);

  half *d_q, *d_cache; float* d_out;
  cudaMalloc(&d_q, G * HKV * HD * sizeof(half));            // Hq=32 x 128
  cudaMalloc(&d_cache, 2 * HKV * MAXC * HD * sizeof(half));  // full KV cache, L2-resident
  cudaMalloc(&d_out, 4);
  std::vector<half> hq(G * HKV * HD), hc(2 * HKV * MAXC * HD);
  for (size_t i = 0; i < hq.size(); i++) hq[i] = __float2half(0.5f + 0.001f * (i % 97));
  for (size_t i = 0; i < hc.size(); i++) hc[i] = __float2half(0.5f + 0.002f * (i % 89));
  cudaMemcpy(d_q, hq.data(), hq.size() * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cache, hc.data(), hc.size() * sizeof(half), cudaMemcpyHostToDevice);

  BenchArgs a = {d_q, d_cache, iters, nvalid, qhoist, nacc, d_out, 0.0};

  if (sweep) {
    for (int m = 0; m < 3; m++)
      for (int l : L) for (int w : W) for (int t : T)
        for (int s : S) {
          if (m != 2 && s > 0) continue;   // SW only reaches the timed loop in MODE=2
          a.us = 0; r_lanes(l, w, t, s, m, lper, &a);
          printf("RESULT lanes=%d warps=%d tk=%d sw=%d mode=%d nacc=%d us_kernel=%.3f\n", l, w, t, s, m, nacc, a.us);
        }
  } else {
    int lanes = arg_int(argc, argv, "--lanes", 32);
    int warps = arg_int(argc, argv, "--warps", 4);
    int tk    = arg_int(argc, argv, "--tk", 16);
    int sw    = arg_int(argc, argv, "--sw", 1);
    int mode  = arg_int(argc, argv, "--mode", 2);
    if (nacc != 8 && lanes == 32 && warps == 4 && tk == 16 && sw == 1 && mode == 2) {
      r_nacc_anchor<32, 4, 16, 1, 2>(nacc, lper, &a);
      printf("RESULT lanes=%d warps=%d tk=%d sw=%d mode=%d nacc=%d lper=%d us_kernel=%.3f\n", lanes, warps, tk, sw, mode, nacc, lper, a.us);
    } else {
      r_lanes(lanes, warps, tk, sw, mode, lper, &a);
      printf("RESULT lanes=%d warps=%d tk=%d sw=%d mode=%d nacc=%d lper=%d us_kernel=%.3f\n", lanes, warps, tk, sw, mode, nacc, lper, a.us);
    }
  }
  return 0;
}
