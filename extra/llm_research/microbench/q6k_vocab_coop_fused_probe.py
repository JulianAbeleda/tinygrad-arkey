#!/usr/bin/env python3
"""L4 Scope B Stage 2: fused-shape probe for the Q6_K vocab coop head (NV sm_120 / RTX 5090).

Times the REAL emitted kernels for the 151936-row vocab head at row_tile=2 - not a
hand-written replica. Both sources come from the tinygrad emitter
(Q6KGEMVRouteSpec(row_tile=2, target="NV:sm_120")) rendered through CUDARenderer:

  q6k_gen_coop_151936_4096          reduction="external_sum"  -> (N,16) partials
  q6k_gen_coop_151936_4096_inkernel reduction="in_kernel"     -> (N,)    pos-lane ladder

The in_kernel arm is the legal fused shape at NV row_tile=2 (row_tile*lane_extent=32 =
one warp, 4 x __shfl_xor_sync steps). The harness compiles the emitted sources verbatim,
launches each at its emitted geometry (grid 75968, block (2,16), one warp per block),
one launch per full vocab pass, host-looped timing under one CUDA event pair per kernel
(wmma_peak/bw_peak method: hoisted operand setup, steady-state loop, event timing).

Numerics sanity: one extra launch of each kernel, then the host reduces the external_sum
(N,16) partials over pos (the q6k_vocab_scalar_reduce output) and compares against the
in_kernel (N,) output. The two reductions use different fp32 summation orders (serial vs
shfl tree), so the check is a tolerance on max abs diff, not bit-equality.

Usage (GPU run MUST be serialized):
  flock /tmp/nv_gpu.lock -c "python3 q6k_vocab_coop_fused_probe.py [passes]"
"""
import os, subprocess, sys, tempfile

sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")
from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.llm.decode_kernels import Q6KGEMVRouteSpec, emit_q6k_gemv_kernel
from tinygrad.llm.qk_layout import Q6_K_BLOCK_ELEMS, Q6K_HALFWORDS_PER_BLOCK
from tinygrad.uop.ops import Ops, UOp

ROWS, K = 151936, 4096
RT = 2
HALFS_N = ROWS * (K // Q6_K_BLOCK_ELEMS) * Q6K_HALFWORDS_PER_BLOCK  # 255252480 u16 = 510.5 MB


def render(reduction: str) -> str:
  spec = Q6KGEMVRouteSpec(rows=ROWS, k=K, route_family="q6k_coop", row_tile=RT,
                          pos_axis="local", target="NV:sm_120", reduction=reduction)
  extent = 1 if reduction == "in_kernel" else spec.partial_axis_extent
  shape = (ROWS,) if extent == 1 else (ROWS, extent)
  partials = UOp.placeholder(shape, dtypes.float32, 0)
  halfs = UOp.placeholder((HALFS_N,), dtypes.uint16, 1)
  x = UOp.placeholder((K,), dtypes.float16, 2)
  ast = emit_q6k_gemv_kernel(spec)(partials, halfs, x)
  ren = CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=True)
  prg = to_program(ast, ren)
  return next(u.arg for u in prg.src if u.op is Ops.SOURCE)


HARNESS = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }

// ---- emitted sources (verbatim from the tinygrad emitter) ----
__SRC_EXTERNAL__
__SRC_INKERNEL__
// --------------------------------------------------------------

#define ROWS 151936
#define PARTIALS_N (ROWS * 16)
#define HALFS_N ((size_t)ROWS * 16 * 105)
#define K 4096

static void run_external(dim3 g, dim3 b, float* out, unsigned short* h, half* x) {
  q6k_gen_coop_151936_4096<<<g, b>>>(out, h, x);
}
static void run_inkernel(dim3 g, dim3 b, float* out, unsigned short* h, half* x) {
  q6k_gen_coop_151936_4096_inkernel<<<g, b>>>(out, h, x);
}

static double time_kernel(void (*fn)(dim3, dim3, float*, unsigned short*, half*),
                          dim3 grid, dim3 block, float* out, unsigned short* h,
                          half* x, int passes) {
  fn(grid, block, out, h, x);
  fn(grid, block, out, h, x);
  if (cudaGetLastError() != cudaSuccess) {
    fprintf(stderr, "launch failed: %s\n", cudaGetErrorString(cudaGetLastError()));
    exit(2);
  }
  cudaDeviceSynchronize();
  cudaEvent_t s, e;
  cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  for (int i = 0; i < passes; i++) fn(grid, block, out, h, x);
  cudaEventRecord(e);
  cudaDeviceSynchronize();
  float ms; cudaEventElapsedTime(&ms, s, e);
  return ms * 1000.0 / passes;
}

int main(int argc, char** argv) {
  int passes = argc > 1 ? atoi(argv[1]) : 32;
  float* ext_out; float* ink_out; unsigned short* h; half* x;
  if (cudaMalloc(&ext_out, PARTIALS_N * 4) != cudaSuccess) return 2;
  if (cudaMalloc(&ink_out, ROWS * 4) != cudaSuccess) return 3;
  if (cudaMalloc(&h, HALFS_N * 2) != cudaSuccess) return 4;
  if (cudaMalloc(&x, K * 2) != cudaSuccess) return 5;
  cudaMemset(h, 0x37, HALFS_N * 2);
  cudaMemset(x, 0x3f, K * 2);
  cudaMemset(ext_out, 0, PARTIALS_N * 4);
  cudaMemset(ink_out, 0, ROWS * 4);

  dim3 grid(75968), block(2, 16);
  double ext_us = time_kernel(run_external, grid, block, ext_out, h, x, passes);
  double ink_us = time_kernel(run_inkernel, grid, block, ink_out, h, x, passes);

  // Numerics sanity: external_sum (N,16) reduced over pos == in_kernel (N,)
  float* hext = (float*)malloc(PARTIALS_N * 4);
  float* hink = (float*)malloc(ROWS * 4);
  cudaMemcpy(hext, ext_out, PARTIALS_N * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(hink, ink_out, ROWS * 4, cudaMemcpyDeviceToHost);
  double max_abs = 0.0, denom = 1e-30;
  for (int r = 0; r < ROWS; r++) {
    float s = 0.0f;
    for (int p = 0; p < 16; p++) s += hext[r * 16 + p];
    double d = fabs((double)s - (double)hink[r]);
    if (d > max_abs) max_abs = d;
    if (fabs((double)hink[r]) > denom) denom = fabs((double)hink[r]);
  }
  double rel = max_abs / denom;

  double bytes = (double)HALFS_N * 2.0;
  printf("ext_us_per_pass=%.2f ink_us_per_pass=%.2f\n", ext_us, ink_us);
  printf("ext_GBps=%.0f ink_GBps=%.0f (of 1792)\n",
         bytes / (ext_us * 1e-6) / 1e9, bytes / (ink_us * 1e-6) / 1e9);
  printf("max_abs_diff_vs_reduced_partials=%.6f max_abs_value=%.6f rel=%.3e\n",
         max_abs, denom, rel);
  return 0;
}
"""


def main() -> int:
  passes = int(sys.argv[1]) if len(sys.argv) > 1 else 32
  # The emitter repeats a common preamble (INFINITY/NAN/tg_bitcast/cuda_fp16.h) in every
  # source; strip it so the harness can concatenate the two kernels into one TU.
  def strip_preamble(src: str) -> str:
    cut = src.index("extern \"C\" __global__")
    return src[cut:]
  src_ext = strip_preamble(render("external_sum"))
  src_ink = strip_preamble(render("in_kernel"))
  assert "q6k_gen_coop_151936_4096(" in src_ext and "q6k_gen_coop_151936_4096_inkernel(" in src_ink
  harness = HARNESS.replace("__SRC_EXTERNAL__", src_ext).replace("__SRC_INKERNEL__", src_ink)

  with tempfile.TemporaryDirectory(prefix="l4fused_") as td:
    cu = os.path.join(td, "fused_probe.cu")
    with open(cu, "w") as f: f.write(harness)
    os.environ["PATH"] = "/usr/local/cuda-13.2/bin:" + os.environ.get("PATH", "")
    binp = os.path.join(td, "fused_probe")
    r = subprocess.run(["nvcc", "-O3", "-arch=sm_120", cu, "-o", binp],
                       capture_output=True, text=True)
    if r.returncode != 0:
      print(r.stderr[-4000:], file=sys.stderr)
      return 3
    r = subprocess.run([binp, str(passes)], capture_output=True, text=True)
    if r.returncode != 0:
      print(r.stderr[-4000:], file=sys.stderr)
      return 4
    print(r.stdout)
  return 0


if __name__ == "__main__":
  sys.exit(main())
