// Route B B3 — OWNED hand-authored decode-attention tile for TINYGRAD's KV layout (not vendored llama code).
// llama-style dataflow (flash decode: online softmax + PV, v_dot2 q.k), authored from scratch to tinygrad's native
// layout K/V = [Hkv, MAXC, Hd] fp16 (kv-head major) so NO repacking is needed (the B2 layout block is removed).
// Fixed shape: T=1, Hd=128, Hq=32, Hkv=8, GQA group=4. KV-split S over positions for occupancy + a combine kernel.
//
// build: hipcc --offload-arch=gfx1100 --genco -D__AMDGCN_WAVEFRONT_SIZE=32 qk_owned_flash_decode.hip -o x.co
//        then clang-offload-bundler --unbundle --targets=hipv4-amdgcn-amd-amdhsa--gfx1100 -> bare ELF for HCQ.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define Hd   128
#define MAXC 4096
#define G    4          // Hq / Hkv (GQA group)
#define Hkv  8          // number of KV heads (Qwen3-8B); K/V half stride in the whole-cache tile = Hkv*MAXC*Hd
#define LANES 32        // warp (wave32)
#define DPL  (Hd/LANES) // dims per lane = 4
#ifndef WCVEC           // Mode-B whole-cache tile LDS-staging load width (1=shipped scalar). _specialize_tile overrides.
#define WCVEC 1
#endif
#ifndef WCUNROLL        // Mode-B whole-cache tile position-loop unroll (1=shipped). _specialize_tile overrides.
#define WCUNROLL 1
#endif

// --- tile kernel: one WARP (32 threads) per (q-head x split). grid.x = Hq, grid.y = S. block = 32.
// each lane owns DPL=4 head-dims (lane*4 .. lane*4+3), as two half2. Online softmax over the split's positions;
// writes per-split partial PV (un-normalized) + (m,l) meta to global for the combine kernel.
extern "C" __global__ void owned_flash_tile(const __half* __restrict__ Q, const __half* __restrict__ K,
                                            const __half* __restrict__ V, float* __restrict__ part,
                                            float* __restrict__ meta, int n_valid, int S, float scale) {
  const int h    = blockIdx.x;           // q-head 0..Hq-1
  const int s    = blockIdx.y;           // split 0..S-1
  const int kvh  = h / G;
  const int lane = threadIdx.x;          // 0..31
  // split this warp's KV range
  const int per  = (n_valid + S - 1) / S;
  const int t0   = s * per;
  const int t1   = min(t0 + per, n_valid);

  const __half2* q2 = (const __half2*)(Q + (long)h * Hd);
  __half2 qa = q2[lane*2 + 0], qb = q2[lane*2 + 1];
  const __half* kbase = K + (long)kvh * MAXC * Hd;
  const __half* vbase = V + (long)kvh * MAXC * Hd;

  float m = -1e30f, l = 0.0f;
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
  for (int t = t0; t < t1; t++) {
    const __half2* k2 = (const __half2*)(kbase + (long)t * Hd);
    float pd = 0.f;
    pd = __builtin_amdgcn_fdot2(qa, k2[lane*2 + 0], pd, false);   // v_dot2_f32_f16
    pd = __builtin_amdgcn_fdot2(qb, k2[lane*2 + 1], pd, false);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) pd += __shfl_xor(pd, o, 32); // warp-reduce -> full score on every lane
    float sc   = pd * scale;
    float mn   = fmaxf(m, sc);
    float corr = __expf(m - mn);
    float p    = __expf(sc - mn);
    l  = l * corr + p;
    const __half* vt = vbase + (long)t * Hd;
    a0 = a0 * corr + p * (float)vt[lane*4 + 0];
    a1 = a1 * corr + p * (float)vt[lane*4 + 1];
    a2 = a2 * corr + p * (float)vt[lane*4 + 2];
    a3 = a3 * corr + p * (float)vt[lane*4 + 3];
    m  = mn;
  }
  // write un-normalized partial PV (NOT divided by l) + meta(m,l) for the combine
  float* po = part + ((long)h * S + s) * Hd;
  po[lane*4 + 0] = a0; po[lane*4 + 1] = a1; po[lane*4 + 2] = a2; po[lane*4 + 3] = a3;
  if (lane == 0) { float* mo = meta + ((long)h * S + s) * 2; mo[0] = m; mo[1] = l; }
}

// --- v2 GQA-PACKED + LDS-STAGED tile: one WORKGROUP (128 threads = 4 warps) per (kv-head x split). The 4 warps are
// the 4 q-heads sharing this kv-head, so K/V is staged into LDS ONCE and reused across all 4 (eliminates v1's 4x
// redundant K/V global reads -- the llama ncols2=4 ingredient). grid.x=Hkv, grid.y=S, block=128.
#define TK 16   // LDS tile of positions
extern "C" __global__ void owned_flash_tile_gqa(const __half* __restrict__ Q, const __half* __restrict__ K,
                                                const __half* __restrict__ V, float* __restrict__ part,
                                                float* __restrict__ meta, int n_valid, int S, float scale) {
  __shared__ __half ksh[TK * Hd];
  __shared__ __half vsh[TK * Hd];
  const int kvh  = blockIdx.x;
  const int s    = blockIdx.y;
  const int warp = threadIdx.x >> 5;     // 0..3 -> q-head in group
  const int lane = threadIdx.x & 31;
  const int h    = kvh * G + warp;
  const int per  = (n_valid + S - 1) / S;
  const int t0   = s * per, t1 = min(t0 + per, n_valid);
  const __half2* q2 = (const __half2*)(Q + (long)h * Hd);
  __half2 qa = q2[lane*2 + 0], qb = q2[lane*2 + 1];
  const __half* kbase = K + (long)kvh * MAXC * Hd;
  const __half* vbase = V + (long)kvh * MAXC * Hd;
  float m = -1e30f, l = 0.0f, a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
  for (int tb = t0; tb < t1; tb += TK) {
    int n = min(TK, t1 - tb);
    for (int i = threadIdx.x; i < n * Hd; i += 128) {       // 128 threads cooperatively stage K/V tile -> LDS
      ksh[i] = kbase[(long)tb * Hd + i];
      vsh[i] = vbase[(long)tb * Hd + i];
    }
    __syncthreads();
    for (int tt = 0; tt < n; tt++) {
      const __half2* k2 = (const __half2*)(ksh + tt * Hd);
      float pd = 0.f;
      pd = __builtin_amdgcn_fdot2(qa, k2[lane*2 + 0], pd, false);
      pd = __builtin_amdgcn_fdot2(qb, k2[lane*2 + 1], pd, false);
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) pd += __shfl_xor(pd, o, 32);
      float sc = pd * scale, mn = fmaxf(m, sc), corr = __expf(m - mn), p = __expf(sc - mn);
      l = l * corr + p;
      const __half* vt = vsh + tt * Hd;
      a0 = a0*corr + p*(float)vt[lane*4+0]; a1 = a1*corr + p*(float)vt[lane*4+1];
      a2 = a2*corr + p*(float)vt[lane*4+2]; a3 = a3*corr + p*(float)vt[lane*4+3];
      m = mn;
    }
    __syncthreads();
  }
  float* po = part + ((long)h * S + s) * Hd;
  po[lane*4 + 0] = a0; po[lane*4 + 1] = a1; po[lane*4 + 2] = a2; po[lane*4 + 3] = a3;
  if (lane == 0) { float* mo = meta + ((long)h * S + s) * 2; mo[0] = m; mo[1] = l; }
}

// --- cheaper combine (B5-lite): THREAD-PER-OUTPUT-DIM + LDS-staged meta + 2D grid for more workgroups. The base
// combine below uses ONE 32-thread warp per q-head (1024 threads total, under-occupies gfx1100) and each lane
// redundantly reloads all S meta for the gm reduction; it is latency-bound (~64 GB/s). This variant: grid=(Hq, Hd/CWD),
// block=CWD, one thread per output head-dim. Meta (m,l for all S splits) is staged into LDS ONCE per workgroup (no
// redundant global meta reloads); each thread streams its dim's S partials (coalesced across the CWD-wide warp). More
// workgroups (Hq*Hd/CWD) + threads hide the partial-read latency that caps the base combine. Same log-sum-exp math.
#ifndef CWD
#define CWD 128
#endif
extern "C" __global__ void owned_flash_combine_hd(const float* __restrict__ part, const float* __restrict__ meta,
                                                  float* __restrict__ out, int S) {
  const int h = blockIdx.x;
  const int d = blockIdx.y * CWD + threadIdx.x;          // output head-dim 0..Hd-1
  __shared__ float sm[128];                              // S <= Smax = ceildiv(MAXC, L) <= 128
  __shared__ float sl[128];
  for (int s = threadIdx.x; s < S; s += CWD) { sm[s] = meta[((long)h*S+s)*2+0]; sl[s] = meta[((long)h*S+s)*2+1]; }
  __syncthreads();
  float gm = -1e30f;
  for (int s = 0; s < S; s++) gm = fmaxf(gm, sm[s]);
  float denom = 0.f, acc = 0.f;
  for (int s = 0; s < S; s++) { float w = __expf(sm[s]-gm); denom += w*sl[s]; acc += w*part[((long)h*S+s)*Hd + d]; }
  out[(long)h*Hd + d] = acc / denom;
}

// --- cheaper combine v3 (B5-lite): thread-per-dim + LDS meta + the per-split WEIGHTS exp(m_s-gm) PRECOMPUTED ONCE in
// LDS cooperatively (the _hd variant recomputes exp per output dim -> CWD-times redundant exp). Main loop is then pure
// FMA over the part reads. grid=(Hq, Hd/CWD), block=CWD. Same log-sum-exp math.
extern "C" __global__ void owned_flash_combine_hw(const float* __restrict__ part, const float* __restrict__ meta,
                                                  float* __restrict__ out, int S) {
  const int h = blockIdx.x;
  const int d = blockIdx.y * CWD + threadIdx.x;
  __shared__ float sm[128], sl[128], sw[128];          // m, l, and precomputed weight w=exp(m-gm)
  for (int s = threadIdx.x; s < S; s += CWD) { sm[s] = meta[((long)h*S+s)*2+0]; sl[s] = meta[((long)h*S+s)*2+1]; }
  __syncthreads();
  float gm = -1e30f;
  for (int s = 0; s < S; s++) gm = fmaxf(gm, sm[s]);
  for (int s = threadIdx.x; s < S; s += CWD) sw[s] = __expf(sm[s] - gm);   // S exps total per head (not CWD*S)
  __syncthreads();
  float denom = 0.f, acc = 0.f;
  for (int s = 0; s < S; s++) { denom += sw[s]*sl[s]; acc += sw[s]*part[((long)h*S+s)*Hd + d]; }
  out[(long)h*Hd + d] = acc / denom;
}

// --- cheaper combine v2 (B5-lite): thread-per-dim BUT the S-reduction is ALSO parallelized across CSR threads, then
// reduced in LDS. The _hd variant above hit a ~10us floor because each thread serially walks all S partials (a
// dependent reduction chain of length S); CWD-count alone did not help. Here block=(CWD, CSR): thread (d, sr) sums its
// strided split subset (chain length S/CSR), then sr=0 reduces the CSR partials -> breaks the latency chain. Same math.
#ifndef CWD
#define CWD 64
#endif
#ifndef CSR
#define CSR 4
#endif
extern "C" __global__ void owned_flash_combine_sr(const float* __restrict__ part, const float* __restrict__ meta,
                                                  float* __restrict__ out, int S) {
  const int h  = blockIdx.x;
  const int d  = blockIdx.y * CWD + threadIdx.x;
  const int sr = threadIdx.y;                            // 0..CSR-1
  __shared__ float sm[128], sl[128];
  __shared__ float acc_lds[CWD * CSR];
  const int tid = threadIdx.y * CWD + threadIdx.x;
  for (int s = tid; s < S; s += CWD * CSR) { sm[s] = meta[((long)h*S+s)*2+0]; sl[s] = meta[((long)h*S+s)*2+1]; }
  __syncthreads();
  float gm = -1e30f;
  for (int s = 0; s < S; s++) gm = fmaxf(gm, sm[s]);     // per-head max from LDS (cheap)
  float acc = 0.f;
  for (int s = sr; s < S; s += CSR) acc += __expf(sm[s]-gm) * part[((long)h*S+s)*Hd + d];  // strided split subset
  acc_lds[threadIdx.x * CSR + sr] = acc;
  __syncthreads();
  if (sr == 0) {
    float a = 0.f;
    for (int j = 0; j < CSR; j++) a += acc_lds[threadIdx.x * CSR + j];   // reduce the CSR partials
    float denom = 0.f;
    for (int s = 0; s < S; s++) denom += __expf(sm[s]-gm) * sl[s];        // per-head denom from LDS
    out[(long)h*Hd + d] = a / denom;
  }
}

// --- combine kernel: one warp per q-head merges the S partials via log-sum-exp. grid.x = Hq, block = 32.
extern "C" __global__ void owned_flash_combine(const float* __restrict__ part, const float* __restrict__ meta,
                                               float* __restrict__ out, int S) {
  const int h    = blockIdx.x;
  const int lane = threadIdx.x;
  // global max over splits
  float gm = -1e30f;
  for (int s = 0; s < S; s++) gm = fmaxf(gm, meta[((long)h * S + s) * 2 + 0]);
  float denom = 0.f;
  float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
  for (int s = 0; s < S; s++) {
    float ms = meta[((long)h * S + s) * 2 + 0];
    float ls = meta[((long)h * S + s) * 2 + 1];
    float w  = __expf(ms - gm);
    denom += w * ls;
    const float* ps = part + ((long)h * S + s) * Hd;
    o0 += w * ps[lane*4 + 0]; o1 += w * ps[lane*4 + 1];
    o2 += w * ps[lane*4 + 2]; o3 += w * ps[lane*4 + 3];
  }
  float inv = 1.0f / denom;
  float* oo = out + (long)h * Hd;
  oo[lane*4 + 0] = o0 * inv; oo[lane*4 + 1] = o1 * inv;
  oo[lane*4 + 2] = o2 * inv; oo[lane*4 + 3] = o3 * inv;
}

// Whole-cache tile variant: K and V come from ONE buffer (model cache_kv [2,1,Hkv,MAXC,Hd] flattened) so callify can
// pass it with BUFFER IDENTITY (no full-MAXC slice materialization). K half at offset 0, V half at +Hkv*MAXC*Hd.
extern "C" __global__ void owned_flash_tile_gqa_whole(const __half* __restrict__ Q, const __half* __restrict__ CACHE,
                                                      float* __restrict__ part,
                                                      float* __restrict__ meta, int n_valid, int S, float scale) {
  __shared__ __half ksh[TK * Hd];
  __shared__ __half vsh[TK * Hd];
  const int kvh  = blockIdx.x;
  const int s    = blockIdx.y;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int h    = kvh * G + warp;
  const int per  = (n_valid + S - 1) / S;
  const int t0   = s * per, t1 = min(t0 + per, n_valid);
  const __half2* q2 = (const __half2*)(Q + (long)h * Hd);
  __half2 qa = q2[lane*2 + 0], qb = q2[lane*2 + 1];
  const __half* kbase = CACHE + (long)kvh * MAXC * Hd;
  const __half* vbase = CACHE + (long)(Hkv + kvh) * MAXC * Hd;   // V half at +Hkv*MAXC*Hd in the flat cache buffer
  float m = -1e30f, l = 0.0f, a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
  for (int tb = t0; tb < t1; tb += TK) {
    int n = min(TK, t1 - tb);
    // Mode-B knob VEC: LDS-staging load width (1=__half, 2=__half2/b32, 4=int2/b64). Hd=128 so n*Hd is /2 and /4;
    // kbase/vbase are 256B-aligned per position -> VEC 2/4 are safe. VEC=1 reproduces the shipped scalar staging.
#if WCVEC == 4
    for (int i = threadIdx.x; i < (n * Hd) / 4; i += 128) {
      ((int2*)ksh)[i] = ((const int2*)(kbase + (long)tb * Hd))[i];
      ((int2*)vsh)[i] = ((const int2*)(vbase + (long)tb * Hd))[i];
    }
#elif WCVEC == 2
    for (int i = threadIdx.x; i < (n * Hd) / 2; i += 128) {
      ((__half2*)ksh)[i] = ((const __half2*)(kbase + (long)tb * Hd))[i];
      ((__half2*)vsh)[i] = ((const __half2*)(vbase + (long)tb * Hd))[i];
    }
#else
    for (int i = threadIdx.x; i < n * Hd; i += 128) {
      ksh[i] = kbase[(long)tb * Hd + i];
      vsh[i] = vbase[(long)tb * Hd + i];
    }
#endif
    __syncthreads();
    #pragma unroll WCUNROLL
    for (int tt = 0; tt < n; tt++) {
      const __half2* k2 = (const __half2*)(ksh + tt * Hd);
      float pd = 0.f;
      pd = __builtin_amdgcn_fdot2(qa, k2[lane*2 + 0], pd, false);
      pd = __builtin_amdgcn_fdot2(qb, k2[lane*2 + 1], pd, false);
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) pd += __shfl_xor(pd, o, 32);
      float sc = pd * scale, mn = fmaxf(m, sc), corr = __expf(m - mn), p = __expf(sc - mn);
      l = l * corr + p;
      const __half* vt = vsh + tt * Hd;
      a0 = a0*corr + p*(float)vt[lane*4+0]; a1 = a1*corr + p*(float)vt[lane*4+1];
      a2 = a2*corr + p*(float)vt[lane*4+2]; a3 = a3*corr + p*(float)vt[lane*4+3];
      m = mn;
    }
    __syncthreads();
  }
  float* po = part + ((long)h * S + s) * Hd;
  po[lane*4 + 0] = a0; po[lane*4 + 1] = a1; po[lane*4 + 2] = a2; po[lane*4 + 3] = a3;
  if (lane == 0) { float* mo = meta + ((long)h * S + s) * 2; mo[0] = m; mo[1] = l; }
}
