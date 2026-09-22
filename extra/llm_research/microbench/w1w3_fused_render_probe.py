#!/usr/bin/env python3
"""MC3 (decode-gemv-instruction-bandwidth-scope-20260803.md section 4.3): gate/up (w1+w3)
fusion viability probe on NV sm_120 / RTX 5090.

Renders three kernels through the real q4k emitter machinery and CUDARenderer:

  q4k_g3_lanemap_gemv_12288_4096           the installed gate/up kernel (both pair arms)
  q4k_g3_lanemap_gemv_w1w3fused_12288_4096 probe-only fused shape: ONE 12288-row kernel
                                           computing both projections with silu folded in,
                                           matching llama's fused w1+w3 node (grid 12288x32,
                                           no separate silu kernel)

The fused kernel is built from the same lowering pieces the installed emitter uses
(_q4k_block_dot_packed_load, LanePartition, _silu_uop, _warp_reduce_sum_staged), with the
two-accumulator loop-carried pattern already proven in production by the flash decode
kernel (flash_decode_attention.py acc/den/mx registers). It is a probe composition, not a
new emitter: the installed q4k_g3_lanemap_gemv_kernel is single-accumulator and cannot emit
the fused shape by itself (see mc3-w1w3-fusion-measurement-record-20260803.md).

Compiles with nvcc 13.2 `-arch=sm_120 -O3 -std=c++17 --ptxas-options=-v` (0-spill gate),
dumps SASS via cuobjdump + nvdisasm (triton package), and times standalone: gate-only,
up-only, pair (gate+up back-to-back), fused. Numerics: fused[r] vs silu(gate[r])*up[r].

Usage (GPU run MUST be serialized; confirm 0% GPU util at lock acquisition):
  flock /tmp/nv_gpu.lock -c "PYTHONPATH=. .venv/bin/python \\
    extra/llm_research/microbench/w1w3_fused_render_probe.py [passes] [rounds]"
"""
import os, re, subprocess, sys, tempfile
from collections import Counter

sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")
from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.codegen.late.warp_reduce import _warp_reduce_sum_staged
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Target
from tinygrad.llm.decode_kernels import (Q4KGateUpLaneMap, LanePartition, Q4K_WORDS_PER_BLOCK,
                                         _q4k_block_dot_packed_load, _silu_uop,
                                         q4k_g3_lanemap_gemv_kernel)
from tinygrad.llm.qk_layout import Q4_K_BLOCK_ELEMS
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp

ROWS, K = 12288, 4096
W = ROWS * (K // Q4_K_BLOCK_ELEMS) * Q4K_WORDS_PER_BLOCK  # 7077888 u32 per projection
CUDA_BIN = "/usr/local/cuda-13.2/bin"
TRITON_BIN = "/home/ubuntu/tinygrad-arkey/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin"


def _render_pair(ren) -> str:
  out = UOp.placeholder((ROWS,), dtypes.float32, 0)
  words = UOp.placeholder((W,), dtypes.uint32, 1)
  x = UOp.placeholder((K,), dtypes.float16, 2)
  ast = q4k_g3_lanemap_gemv_kernel(ROWS, K)(out, words, x)
  prg = to_program(ast, ren)
  return next(u.arg for u in prg.src if u.op is Ops.SOURCE)


def _render_fused(ren) -> str:
  """Probe-only fused w1+w3: one 12288-row kernel, both projections, silu folded in.

  Same lowering pieces as the installed emitter, composed twice per row with the
  two-accumulator loop pattern (flash_decode_attention.py): both loop-carried registers
  initialized in one chain, one loop, last update ends the range. Distinct REG slots for
  the two accumulators (20/21) and the two lane-reduce ladders (90-94 / 95-99).
  """
  lm = Q4KGateUpLaneMap(k=K, n=ROWS)
  lm.validate()
  name = f"q4k_g3_lanemap_gemv_w1w3fused_{ROWS}_{K}"

  def kernel(out: UOp, gate_words: UOp, up_words: UOp, x: UOp) -> UOp:
    row, lane = UOp.special(ROWS, "gidx0"), UOp.special(32, "lidx0")
    part = LanePartition(lane, lane_extent=lm.lane_extent, words_per_group=lm.words_per_group)
    acc_g = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
    acc_u = UOp.placeholder((1,), dtypes.float32, 21, addrspace=AddrSpace.REG)
    init = acc_g[0].store(0.0)
    init = acc_u.after(init)[0].store(0.0)
    acc_g, acc_u = acc_g.after(init), acc_u.after(init)
    lblk = UOp.range(lm.blocks_per_group, 0, axis_type=AxisType.REDUCE)
    blk = part.block_group * lm.blocks_per_group + lblk
    base_g = (row * lm.k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    base_u = (row * lm.k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib_g = _q4k_block_dot_packed_load(gate_words, x, base_g, blk, part.word_col)
    contrib_u = _q4k_block_dot_packed_load(up_words, x, base_u, blk, part.word_col)
    upd_g = acc_g[0].store(acc_g.after(lblk)[0] + contrib_g)
    upd_u = acc_u.after(upd_g)[0].store(acc_u.after(lblk)[0] + contrib_u).end(lblk)
    total_g = _warp_reduce_sum_staged(acc_g.after(upd_u)[0], part.lane, part.lane_extent, 90)
    total_u = _warp_reduce_sum_staged(acc_u.after(upd_u)[0], part.lane, part.lane_extent, 95)
    return out[row].store(_silu_uop(total_g) * total_u).sink(arg=KernelInfo(name=name, opts_to_apply=()))

  out = UOp.placeholder((ROWS,), dtypes.float32, 0)
  gw = UOp.placeholder((W,), dtypes.uint32, 1)
  uw = UOp.placeholder((W,), dtypes.uint32, 2)
  x = UOp.placeholder((K,), dtypes.float16, 3)
  prg = to_program(kernel(out, gw, uw, x), ren)
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
__SRC_PAIR__
__SRC_FUSED__

#define ROWS 12288
#define W ((size_t)ROWS * 16 * 36)
#define K 4096

static void run_pair(dim3 g, dim3 b, float* out, unsigned int* w, half* x) {
  q4k_g3_lanemap_gemv_12288_4096<<<g, b>>>(out, w, x);
}
static void run_fused(dim3 g, dim3 b, float* out, unsigned int* gw, unsigned int* uw, half* x) {
  q4k_g3_lanemap_gemv_w1w3fused_12288_4096<<<g, b>>>(out, gw, uw, x);
}

static double mn(double* a, int n) { double m = a[0]; for (int i = 1; i < n; i++) if (a[i] < m) m = a[i]; return m; }
static double md(double* a, int n) {
  double s[8]; for (int i = 0; i < n; i++) s[i] = a[i];
  for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) if (s[j] < s[i]) { double t = s[i]; s[i] = s[j]; s[j] = t; }
  return s[n / 2];
}

static double time_kernel(int mode, dim3 grid, dim3 block, float* og, float* ou, float* of,
                          unsigned int* gw, unsigned int* uw, half* x, int passes) {
  // mode 0 = gate only, 1 = up only, 2 = pair (gate+up back-to-back), 3 = fused
  for (int i = 0; i < 2; i++) {
    if (mode == 0) run_pair(grid, block, og, gw, x);
    else if (mode == 1) run_pair(grid, block, ou, uw, x);
    else if (mode == 2) { run_pair(grid, block, og, gw, x); run_pair(grid, block, ou, uw, x); }
    else run_fused(grid, block, of, gw, uw, x);
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
    else run_fused(grid, block, of, gw, uw, x);
  }
  cudaEventRecord(e);
  cudaDeviceSynchronize();
  float ms; cudaEventElapsedTime(&ms, s, e);
  return ms * 1000.0 / passes;
}

int main(int argc, char** argv) {
  int passes = argc > 1 ? atoi(argv[1]) : 32;
  int rounds = argc > 2 ? atoi(argv[2]) : 3;
  float* og; float* ou; float* of; unsigned int* gw; unsigned int* uw; half* x;
  if (cudaMalloc(&og, ROWS * 4) != cudaSuccess) return 2;
  if (cudaMalloc(&ou, ROWS * 4) != cudaSuccess) return 3;
  if (cudaMalloc(&of, ROWS * 4) != cudaSuccess) return 4;
  if (cudaMalloc(&gw, W * 4) != cudaSuccess) return 5;
  if (cudaMalloc(&uw, W * 4) != cudaSuccess) return 6;
  if (cudaMalloc(&x, K * 2) != cudaSuccess) return 7;
  // Deterministic finite Q4_K fill (valid blocks so the numerics check is meaningful:
  // random raw words can dequantize to inf/NaN and make equality vacuous). One row =
  // 16 blocks x 36 words; word0 = fp16(d) | fp16(dmin), words 1-3 = 12 scale bytes,
  // words 4-35 = 256 quantized nibbles.
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
        unsigned int s3 = 0;
        for (int i = 0; i < 3; i++) {
          unsigned int word = 0;
          for (int j = 0; j < 4; j++) word |= (((seed = seed * 1664525u + 1013904223u) % 64)) << (8 * j);
          w[1 + i] = word; s3 |= word;
        }
        (void)s3;
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

  dim3 grid(ROWS), block(32);
  double tg[8] = {0}, tu[8] = {0}, tp[8] = {0}, tf[8] = {0};
  for (int r = 0; r < rounds; r++) {
    tg[r] = time_kernel(0, grid, block, og, ou, of, gw, uw, x, passes);
    tu[r] = time_kernel(1, grid, block, og, ou, of, gw, uw, x, passes);
    tp[r] = time_kernel(2, grid, block, og, ou, of, gw, uw, x, passes);
    tf[r] = time_kernel(3, grid, block, og, ou, of, gw, uw, x, passes);
  }
  double pair_bytes = (double)2 * W * 4;       // weight bytes both projections (x is tiny)
  double fused_bytes = (double)2 * W * 4;
  printf("gate_us_min=%.2f gate_us_med=%.2f\n", mn(tg, rounds), md(tg, rounds));
  printf("up_us_min=%.2f up_us_med=%.2f\n", mn(tu, rounds), md(tu, rounds));
  printf("pair_us_min=%.2f pair_us_med=%.2f pair_TBs=%.0f\n", mn(tp, rounds), md(tp, rounds), pair_bytes / (md(tp, rounds) * 1e-6) / 1e12);
  printf("fused_us_min=%.2f fused_us_med=%.2f fused_TBs=%.0f\n", mn(tf, rounds), md(tf, rounds), fused_bytes / (md(tf, rounds) * 1e-6) / 1e12);

  // numerics: fused[r] vs silu(gate[r]) * up[r] (host fp32, same silu expression)
  float* hg = (float*)malloc(ROWS * 4);
  float* hu = (float*)malloc(ROWS * 4);
  float* hf = (float*)malloc(ROWS * 4);
  cudaMemcpy(hg, og, ROWS * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(hu, ou, ROWS * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(hf, of, ROWS * 4, cudaMemcpyDeviceToHost);
  double max_abs = 0.0, max_rel = 0.0, denom = 1e-30;
  int nonfinite = 0, mismatches = 0;
  for (int r = 0; r < ROWS; r++) {
    float silu_g = hg[r] * (1.0f / (1.0f + exp2f(hg[r] * -1.4426950408889634f)));
    float ref = silu_g * hu[r];
    if (!isfinite(hf[r]) || !isfinite(ref)) { nonfinite++; continue; }
    if (hf[r] != ref) mismatches++;
    double d = fabs((double)ref - (double)hf[r]);
    if (d > max_abs) max_abs = d;
    if (fabs((double)ref) > 1e-3 && d / fabs((double)ref) > max_rel) max_rel = d / fabs((double)ref);
    if (fabs((double)ref) > denom) denom = fabs((double)ref);
  }
  printf("max_abs_diff_vs_silu_gate_mul_up=%.6f max_rel=%.3e max_abs_value=%.6f nonfinite=%d bit_mismatches=%d\n",
         max_abs, max_rel, denom, nonfinite, mismatches);
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
  s_pair = _render_pair(ren)
  s_fused = _render_fused(ren)
  assert "q4k_g3_lanemap_gemv_12288_4096(" in s_pair
  assert "q4k_g3_lanemap_gemv_w1w3fused_12288_4096(" in s_fused
  harness = HARNESS.replace("__SRC_PAIR__", _strip_preamble(s_pair)).replace("__SRC_FUSED__", _strip_preamble(s_fused))

  with tempfile.TemporaryDirectory(prefix="mc3_w1w3_") as td:
    cu = os.path.join(td, "w1w3_probe.cu")
    with open(cu, "w") as f:
      f.write(harness)
    os.environ["PATH"] = f"{CUDA_BIN}:" + os.environ.get("PATH", "")
    binp = os.path.join(td, "w1w3_probe")
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
                  "IMAD", "IADD3", "LOP3", "ISETP", "BRA", "EXIT", "MOV", "S2R", "LDC")
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
