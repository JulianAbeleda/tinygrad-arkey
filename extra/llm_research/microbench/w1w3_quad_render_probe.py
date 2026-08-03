#!/usr/bin/env python3
"""w1+w3 fused GEMV quad-style render + standalone timing probe (NV sm_120 / RTX 5090).

Extends the MC3 probe with the MC2-winning load pattern: `q4k_g3_lanemap_gemv_w1w3qv_12288_4096`
(768 blocks x 128 threads, 16 rows/block, 8 lanes/row, wc-quad `(lane&1)*4` in group-of-2
`lane>>1`, x staged once to shared memory, pure uint4 weight loads) against the same harness
arms as the MC3 record: gate-only, up-only, pair, fused scalar (MC3 probe shape). Renders all
four kernels through the real emitter machinery and CUDARenderer, compiles with nvcc 13.2
`-arch=sm_120 -O3 -std=c++17 --ptxas-options=-v` (0-spill gate), dumps SASS via cuobjdump +
nvdisasm, times standalone, and checks quad/scalar numerics against silu(gate[r])*up[r].

Usage (GPU run MUST be serialized; confirm 0% GPU util at lock acquisition):
  flock /tmp/nv_gpu.lock -c "PYTHONPATH=. .venv/bin/python \\
    extra/llm_research/microbench/w1w3_quad_render_probe.py [passes] [rounds]"
"""
import os, re, subprocess, sys, tempfile
from collections import Counter

sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")
from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.llm.decode_kernels import Q4K_WORDS_PER_BLOCK, q4k_g3_lanemap_gemv_kernel, \
  q4k_g3_lanemap_gemv_w1w3_kernel
from tinygrad.llm.qk_layout import Q4_K_BLOCK_ELEMS
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops, UOp

ROWS, K = 12288, 4096
W = ROWS * (K // Q4_K_BLOCK_ELEMS) * Q4K_WORDS_PER_BLOCK  # 7077888 u32 per projection
CUDA_BIN = "/usr/local/cuda-13.2/bin"
TRITON_BIN = "/home/ubuntu/tinygrad-arkey/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin"


def _render(ren, style: str) -> str:
  out = UOp.placeholder((ROWS,), dtypes.float32, 0)
  gw = UOp.placeholder((W,), dtypes.uint32, 1)
  uw = UOp.placeholder((W,), dtypes.uint32, 2)
  x = UOp.placeholder((K,), dtypes.float16, 3)
  if style == "pair":
    ast = q4k_g3_lanemap_gemv_kernel(ROWS, K)(out, gw, x)
  else:
    ast = q4k_g3_lanemap_gemv_w1w3_kernel(ROWS, K, load_style=style)(out, gw, uw, x)
  prg = to_program(ast, ren)
  return next(u.arg for u in prg.src if u.op is Ops.SOURCE)


HARNESS = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#ifndef INFINITY
#define INFINITY (__int_as_float(0x7f800000))
#endif
#ifndef NAN
#define NAN (__int_as_float(0x7fffffff))
#endif
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
struct __align__(8) half4 { half x, y, z, w; };
__SRC_PAIR__
__SRC_SCALAR__
__SRC_QUAD__

#define ROWS 12288
#define W ((size_t)ROWS * 16 * 36)
#define K 4096

static void run_pair(dim3 g, dim3 b, float* out, unsigned int* w, half* x) {
  q4k_g3_lanemap_gemv_12288_4096<<<g, b>>>(out, w, x);
}
static void run_scalar(dim3 g, dim3 b, float* out, unsigned int* gw, unsigned int* uw, half* x) {
  q4k_g3_lanemap_gemv_w1w3fused_12288_4096<<<g, b>>>(out, gw, uw, x);
}
static void run_quad(dim3 g, dim3 b, float* out, unsigned int* gw, unsigned int* uw, half* x) {
  q4k_g3_lanemap_gemv_w1w3qv_12288_4096<<<g, b>>>(out, gw, uw, x);
}

static double mn(double* a, int n) { double m = a[0]; for (int i = 1; i < n; i++) if (a[i] < m) m = a[i]; return m; }
static double md(double* a, int n) {
  double s[8]; for (int i = 0; i < n; i++) s[i] = a[i];
  for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) if (s[j] < s[i]) { double t = s[i]; s[i] = s[j]; s[j] = t; }
  return s[n / 2];
}

static double time_kernel(int mode, dim3 grid, dim3 block, float* og, float* ou, float* of,
                          unsigned int* gw, unsigned int* uw, half* x, int passes) {
  // mode 0 = gate only, 1 = up only, 2 = pair, 3 = fused scalar, 4 = fused quad
  for (int i = 0; i < 2; i++) {
    if (mode == 0) run_pair(grid, block, og, gw, x);
    else if (mode == 1) run_pair(grid, block, ou, uw, x);
    else if (mode == 2) { run_pair(grid, block, og, gw, x); run_pair(grid, block, ou, uw, x); }
    else if (mode == 3) run_scalar(grid, block, of, gw, uw, x);
    else run_quad(grid, block, of, gw, uw, x);
  }
  if (cudaGetLastError() != cudaSuccess) {
    fprintf(stderr, "launch failed: %s\n", cudaGetErrorString(cudaGetLastError()));
    exit(2);
  }
  cudaDeviceSynchronize();
  cudaEvent_t s, e;
  cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  for (int i = 0; i < passes; i++) {
    if (mode == 0) run_pair(grid, block, og, gw, x);
    else if (mode == 1) run_pair(grid, block, ou, uw, x);
    else if (mode == 2) { run_pair(grid, block, og, gw, x); run_pair(grid, block, ou, uw, x); }
    else if (mode == 3) run_scalar(grid, block, of, gw, uw, x);
    else run_quad(grid, block, of, gw, uw, x);
  }
  cudaEventRecord(e);
  cudaDeviceSynchronize();
  float ms; cudaEventElapsedTime(&ms, s, e);
  return ms * 1000.0 / passes;
}

int main(int argc, char** argv) {
  int passes = argc > 1 ? atoi(argv[1]) : 32;
  int rounds = argc > 2 ? atoi(argv[2]) : 3;
  float* og; float* ou; float* of; float* oq; unsigned int* gw; unsigned int* uw; half* x;
  if (cudaMalloc(&og, ROWS * 4) != cudaSuccess) return 2;
  if (cudaMalloc(&ou, ROWS * 4) != cudaSuccess) return 3;
  if (cudaMalloc(&of, ROWS * 4) != cudaSuccess) return 4;
  if (cudaMalloc(&oq, ROWS * 4) != cudaSuccess) return 5;
  if (cudaMalloc(&gw, W * 4) != cudaSuccess) return 6;
  if (cudaMalloc(&uw, W * 4) != cudaSuccess) return 7;
  if (cudaMalloc(&x, K * 2) != cudaSuccess) return 8;
  // Deterministic finite Q4_K fill (valid blocks so the numerics check is meaningful:
  // random raw words can dequantize to inf/NaN and make equality vacuous).
  unsigned int* hgw = (unsigned int*)malloc(W * 4);
  unsigned int* huw = (unsigned int*)malloc(W * 4);
  unsigned int seed = 0x12345678u;
  for (int t = 0; t < 2; t++) {
    unsigned int* hw = t == 0 ? hgw : huw;
    for (int row = 0; row < ROWS; row++) {
      for (int b = 0; b < 16; b++) {
        unsigned int* w = hw + (size_t)(row * 16 + b) * 36;
        float d = 0.5f + (float)((seed = seed * 1664525u + 1013904223u) % 1000) / 1000.0f;
        float dm = (float)((seed = seed * 1664525u + 1013904223u) % 2000) / 1000.0f - 1.0f;
        w[0] = ((unsigned int)__half_as_ushort(__float2half(d))) |
               (((unsigned int)__half_as_ushort(__float2half(dm))) << 16);
        for (int i = 0; i < 3; i++) {
          unsigned int word = 0;
          for (int j = 0; j < 4; j++) word |= (((seed = seed * 1664525u + 1013904223u) % 64)) << (8 * j);
          w[1 + i] = word;
        }
        for (int i = 0; i < 32; i++) w[4 + i] = (seed = seed * 1664525u + 1013904223u);
      }
    }
  }
  half* hx = (half*)malloc(K * 2);
  for (int i = 0; i < K; i++) {
    float v = (float)((i * 2654435761u) % 9973) / 9973.0f * 2.0f - 1.0f;
    if (i % 256 == 0) v = 8.0f;
    if (i % 257 == 0) v = -8.0f;
    hx[i] = __float2half(v);
  }
  cudaMemcpy(gw, hgw, W * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(uw, huw, W * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(x, hx, K * 2, cudaMemcpyHostToDevice);
  cudaMemset(og, 0, ROWS * 4);
  cudaMemset(ou, 0, ROWS * 4);
  cudaMemset(of, 0, ROWS * 4);
  cudaMemset(oq, 0, ROWS * 4);

  dim3 grid_pair(ROWS), block_pair(32);
  dim3 grid_quad(ROWS / 16), block_quad(128);
  double tg[8] = {0}, tu[8] = {0}, tp[8] = {0}, ts[8] = {0}, tq[8] = {0};
  for (int r = 0; r < rounds; r++) {
    tg[r] = time_kernel(0, grid_pair, block_pair, og, ou, of, gw, uw, x, passes);
    tu[r] = time_kernel(1, grid_pair, block_pair, og, ou, of, gw, uw, x, passes);
    tp[r] = time_kernel(2, grid_pair, block_pair, og, ou, of, gw, uw, x, passes);
    ts[r] = time_kernel(3, grid_pair, block_pair, og, ou, of, gw, uw, x, passes);
    tq[r] = time_kernel(4, grid_quad, block_quad, og, ou, oq, gw, uw, x, passes);
  }
  double pair_bytes = (double)2 * W * 4;
  printf("gate_us_min=%.2f gate_us_med=%.2f\n", mn(tg, rounds), md(tg, rounds));
  printf("up_us_min=%.2f up_us_med=%.2f\n", mn(tu, rounds), md(tu, rounds));
  printf("pair_us_min=%.2f pair_us_med=%.2f pair_TBs=%.0f\n", mn(tp, rounds), md(tp, rounds), pair_bytes / (md(tp, rounds) * 1e-6) / 1e12);
  printf("scalar_us_min=%.2f scalar_us_med=%.2f scalar_TBs=%.0f\n", mn(ts, rounds), md(ts, rounds), pair_bytes / (md(ts, rounds) * 1e-6) / 1e12);
  printf("quad_us_min=%.2f quad_us_med=%.2f quad_TBs=%.0f\n", mn(tq, rounds), md(tq, rounds), pair_bytes / (md(tq, rounds) * 1e-6) / 1e12);

  // numerics: fused[r] vs silu(gate[r]) * up[r] (host fp32, same silu expression).
  // NOTE: the timing loop leaves `of` = LAST arm's output (scalar, mode 3), `oq` = quad
  // (mode 4), so the fused copies below are unambiguous.
  float* hg = (float*)malloc(ROWS * 4);
  float* hu = (float*)malloc(ROWS * 4);
  float* hs = (float*)malloc(ROWS * 4);
  float* hq = (float*)malloc(ROWS * 4);
  cudaMemcpy(hg, og, ROWS * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(hu, ou, ROWS * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(hs, of, ROWS * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(hq, oq, ROWS * 4, cudaMemcpyDeviceToHost);
  // Host reference directly from the deterministic fill: exact installed-layout dequant.
  double* hrg = (double*)malloc(ROWS * sizeof(double));
  double* hru = (double*)malloc(ROWS * sizeof(double));
  for (int r = 0; r < ROWS; r++) {
    double sg = 0.0, su = 0.0;
    for (int b = 0; b < 16; b++) {
      unsigned int* wg = hgw + (size_t)(r * 16 + b) * 36;
      unsigned int* wu = huw + (size_t)(r * 16 + b) * 36;
      float dg = __half2float(__ushort_as_half((unsigned short)(wg[0] & 0xffffu)));
      float dmg = __half2float(__ushort_as_half((unsigned short)(wg[0] >> 16)));
      float du = __half2float(__ushort_as_half((unsigned short)(wu[0] & 0xffffu)));
      float dmu = __half2float(__ushort_as_half((unsigned short)(wu[0] >> 16)));
      for (int grp = 0; grp < 8; grp++) {
        unsigned int sc_g, mn_g, sc_u, mn_u;
        if (grp < 4) {
          sc_g = (wg[1] >> (grp * 8)) & 63; mn_g = (wg[2] >> (grp * 8)) & 63;
          sc_u = (wu[1] >> (grp * 8)) & 63; mn_u = (wu[2] >> (grp * 8)) & 63;
        } else {
          unsigned int high_g = (wg[3] >> ((grp - 4) * 8)) & 255;
          unsigned int high_u = (wu[3] >> ((grp - 4) * 8)) & 255;
          sc_g = (high_g & 15) | (((wg[1] >> ((grp - 4) * 8)) >> 6) << 4);
          mn_g = (high_g >> 4) | (((wg[2] >> ((grp - 4) * 8)) >> 6) << 4);
          sc_u = (high_u & 15) | (((wu[1] >> ((grp - 4) * 8)) >> 6) << 4);
          mn_u = (high_u >> 4) | (((wu[2] >> ((grp - 4) * 8)) >> 6) << 4);
        }
        for (int lane4 = 0; lane4 < 8; lane4++) {
          unsigned int qg = (wg[4 + (grp / 2) * 8 + lane4] >> ((grp % 2) * 4)) & 0x0F0F0F0Fu;
          unsigned int qu = (wu[4 + (grp / 2) * 8 + lane4] >> ((grp % 2) * 4)) & 0x0F0F0F0Fu;
          for (int nib = 0; nib < 4; nib++) {
            int pos = lane4 * 4 + nib;
            float xv = __half2float(hx[b * 256 + grp * 32 + pos]);
            sg += (double)((dg * (float)sc_g * (float)((qg >> (nib * 8)) & 15)) - dmg * (float)mn_g) * xv;
            su += (double)((du * (float)sc_u * (float)((qu >> (nib * 8)) & 15)) - dmu * (float)mn_u) * xv;
          }
        }
      }
    }
    hrg[r] = sg; hru[r] = su;
  }
  for (int which = 0; which < 2; which++) {
    float* hf = which == 0 ? hs : hq;
    double max_abs = 0.0, max_rel = 0.0, denom = 1e-30, max_ref_abs = 0.0;
    int nonfinite = 0, mismatches = 0, ref_mismatches = 0;
    for (int r = 0; r < ROWS; r++) {
      float silu_g = hg[r] * (1.0f / (1.0f + exp2f(hg[r] * -1.4426950408889634f)));
      float ref = silu_g * hu[r];
      if (!isfinite(hf[r]) || !isfinite(ref)) { nonfinite++; continue; }
      if (hf[r] != ref) mismatches++;
      double d = fabs((double)ref - (double)hf[r]);
      if (d > max_abs) max_abs = d;
      if (fabs((double)ref) > 1e-3 && d / fabs((double)ref) > max_rel) max_rel = d / fabs((double)ref);
      if (fabs((double)ref) > denom) denom = fabs((double)ref);
      if (fabs((double)ref) > max_ref_abs) max_ref_abs = fabs((double)ref);
      // host dequant reference (pair semantic): silu(ref_g) * ref_u
      double silu_rg = (double)hrg[r] * (1.0 / (1.0 + exp2((double)hrg[r] * -1.4426950408889634)));
      double refr = silu_rg * hru[r];
      if (fabs(refr - (double)ref) > 1e-3 * fabs(refr) + 1.0) ref_mismatches++;
    }
    printf("numerics_%s max_abs_diff=%.6f max_rel=%.3e max_abs_value=%.6f max_ref_abs=%.6f nonfinite=%d bit_mismatches=%d hostref_mismatches=%d\n",
           which == 0 ? "scalar" : "quad", max_abs, max_rel, denom, max_ref_abs, nonfinite, mismatches, ref_mismatches);
  }
  // row-level sample for the quad arm vs the pair and vs the host reference
  for (int r = 0; r < 8; r++) {
    double silu_rg = (double)hrg[r] * (1.0 / (1.0 + exp2((double)hrg[r] * -1.4426950408889634)));
    printf("row%03d pair_g=%.5f pair_u=%.5f host_g=%.5f host_u=%.5f scalar=%.5f quad=%.5f ref=%.5f\n",
           r, hg[r], hu[r], hrg[r], hru[r], hs[r], hq[r], (float)(silu_rg * hru[r]));
  }
  // quad-vs-ref mismatch pattern: largest |quad - ref| rows and quad-vs-scalar max
  int worst[8]; double worstd[8];
  for (int i = 0; i < 8; i++) { worst[i] = -1; worstd[i] = -1.0; }
  double max_qs = 0.0; int max_qs_row = -1;
  double max_rel_row = 0.0; int worst_rel_row = -1;
  for (int r = 0; r < ROWS; r++) {
    float silu_g = hg[r] * (1.0f / (1.0f + exp2f(hg[r] * -1.4426950408889634f)));
    float ref = silu_g * hu[r];
    double dq = fabs((double)ref - (double)hq[r]);
    for (int i = 0; i < 8; i++) if (dq > worstd[i]) { worstd[i] = dq; worst[i] = r; break; }
    double qs = fabs((double)hq[r] - (double)hs[r]);
    if (qs > max_qs) { max_qs = qs; max_qs_row = r; }
    if (fabs((double)ref) > 1e-3 && dq / fabs((double)ref) > max_rel_row) { max_rel_row = dq / fabs((double)ref); worst_rel_row = r; }
  }
  if (worst_rel_row >= 0) {
    int r = worst_rel_row;
    float silu_g = hg[r] * (1.0f / (1.0f + exp2f(hg[r] * -1.4426950408889634f)));
    float ref = silu_g * hu[r];
    printf("worst_rel row%05d pair_g=%.5f pair_u=%.5f scalar=%.5f quad=%.5f ref=%.5f diff=%.5f rel=%.3e\n",
           r, hg[r], hu[r], hs[r], hq[r], ref, ref - hq[r], max_rel_row);
  }
  for (int i = 0; i < 8; i++) if (worst[i] >= 0) {
    int r = worst[i];
    float silu_g = hg[r] * (1.0f / (1.0f + exp2f(hg[r] * -1.4426950408889634f)));
    float ref = silu_g * hu[r];
    printf("worst_quad row%05d pair_g=%.5f pair_u=%.5f scalar=%.5f quad=%.5f ref=%.5f diff=%.5f rel=%.3e\n",
           r, hg[r], hu[r], hs[r], hq[r], ref, ref - hq[r], fabs((double)ref - hq[r]) / (fabs((double)ref) + 1e-30));
  }
  printf("max_quad_vs_scalar=%.5f at row %d\n", max_qs, max_qs_row);
  return 0;
}
"""
def _strip_preamble(src: str) -> str:
  return src[src.index('extern "C" __global__'):]


def _ptxas_summary(compile_out: str) -> dict:
  summary = {}
  cur = None
  for line in compile_out.splitlines():
    m = re.search(r"Compiling entry function '(\S+)'", line)
    if m: cur = m.group(1)
    m = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", line)
    if m and cur: summary.setdefault(cur, {})["spill_stores"] = int(m.group(1)); summary.setdefault(cur, {})["spill_loads"] = int(m.group(2))
    m = re.search(r"Used (\d+) registers", line)
    if m and cur: summary.setdefault(cur, {})["registers"] = int(m.group(1))
  return summary


def _sass_summary(binp: str) -> dict:
  env = dict(os.environ)
  env["PATH"] = f"{CUDA_BIN}:{TRITON_BIN}:" + env.get("PATH", "")
  r = subprocess.run(["cuobjdump", "--dump-sass", binp], capture_output=True, text=True, env=env)
  if r.returncode != 0:
    raise RuntimeError(f"cuobjdump failed: {r.stderr[-500:]}")
  sections: dict[str, list[str]] = {}
  cur = None
  for line in r.stdout.splitlines():
    m = re.match(r"\s*Function : (\S+)", line)
    if m:
      cur = m.group(1); sections.setdefault(cur, [])
    elif cur is not None:
      sections[cur].append(line)
  out = {}
  for kname, lines in sections.items():
    ops = Counter(re.findall(r"/\*[0-9a-f]+\*/\s+([A-Z][A-Z0-9._]*)", "\n".join(lines)))
    out[kname] = {"total": sum(ops.values()), "ops": {o: c for o, c in sorted(ops.items())}}
  return out


def main() -> int:
  passes = int(sys.argv[1]) if len(sys.argv) > 1 else 32
  rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
  ren = CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=True)
  s_pair = _render(ren, "pair")
  s_scalar = _render(ren, "scalar")
  s_quad = _render(ren, "quad")
  assert "q4k_g3_lanemap_gemv_12288_4096(" in s_pair
  assert "q4k_g3_lanemap_gemv_w1w3fused_12288_4096(" in s_scalar
  assert "q4k_g3_lanemap_gemv_w1w3qv_12288_4096(" in s_quad
  gated = bool(re.search(r"if \(", s_quad))
  print(f"quad gated store rendered: {gated}")
  if not gated:
    print("quad store is not gated - aborting", file=sys.stderr)
    return 6
  quad_u4 = s_quad.count("uint4 val")
  print(f"quad uint4 load statements: {quad_u4}")
  harness = (HARNESS.replace("__SRC_PAIR__", _strip_preamble(s_pair))
                     .replace("__SRC_SCALAR__", _strip_preamble(s_scalar))
                     .replace("__SRC_QUAD__", _strip_preamble(s_quad)))

  with tempfile.TemporaryDirectory(prefix="mc2q_w1w3_") as td:
    cu = os.path.join(td, "w1w3_quad_probe.cu")
    with open(cu, "w") as f:
      f.write(harness)
    os.environ["PATH"] = f"{CUDA_BIN}:" + os.environ.get("PATH", "")
    binp = os.path.join(td, "w1w3_quad_probe")
    cp = subprocess.run(["nvcc", "-arch=sm_120", "-O3", "-std=c++17", "--ptxas-options=-v", cu, "-o", binp],
                        capture_output=True, text=True)
    if cp.returncode != 0:
      print(cp.stderr[-4000:], file=sys.stderr)
      return 3
    ptx = _ptxas_summary(cp.stderr)
    print("ptxas:")
    for k in sorted(ptx):
      print(f"  {k}: regs={ptx[k].get('registers')} spill_stores={ptx[k].get('spill_stores')} spill_loads={ptx[k].get('spill_loads')}")
    if any(v.get("spill_stores", 0) or v.get("spill_loads", 0) for v in ptx.values()):
      print("SPILLS PRESENT - refusing to time", file=sys.stderr)
      return 5
    sass = _sass_summary(binp)
    print("sass (instruction counts):")
    for k in sorted(sass):
      ops = sass[k]["ops"]
      prefixes = ("LDG", "STG", "FFMA", "FMUL", "FADD", "SHFL", "MUFU", "LDL", "STL", "LDS", "STS",
                  "IMAD", "IADD3", "LOP3", "ISETP", "BRA", "EXIT", "MOV", "S2R", "LDC", "BAR")
      counts = Counter()
      for o, c in ops.items():
        for p in prefixes:
          if o.startswith(p):
            counts[p] += c
            break
      print(f"  {k}: total={sass[k]['total']} " + " ".join(f"{p}={counts[p]}" for p in prefixes if counts[p]))
    r = subprocess.run([binp, str(passes), str(rounds)], capture_output=True, text=True)
    if r.returncode != 0:
      print(r.stderr[-4000:], file=sys.stderr)
      return 4
    print(r.stdout)
  return 0


if __name__ == "__main__":
  sys.exit(main())
