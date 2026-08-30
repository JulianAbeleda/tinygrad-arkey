#!/usr/bin/env python3
"""Standalone body/cache/launch microgate for the Q/K head norms.

Renders the exact production tinygrad kernels
(``reduce_output_rmsnorm_32_128``, ``reduce_output_rmsnorm_8_128``) through
CUDARenderer, embeds a faithful PDL-off copy of llama.cpp's
``rms_norm_f32<256,true>`` (the ``q_norm`` / ``k_norm`` role from norm.cu), and
compiles all four with nvcc for sm_120a. For each kernel it measures, with
CUDA events in one clock domain:

  hot    back-to-back loop, input hot in L2
  fill   a producer writes the norm input immediately before the timed launch
  flush  a 128 MiB streaming write evicts L2 before the timed launch

The ``fill``/``flush`` measurements isolate body, producer-conditioned cache
state, and launch-after-a-predecessor. They deliberately do NOT reproduce the
HCQ submission path; production launch gaps come from the retained capture
split, not from this CUDA-runtime harness.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.codegen.late.reduce_output import emit_reduce_output
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops, ReduceOutputSpec, UOp

CUDA_BIN = "/usr/local/cuda-13.2/bin"
EPS = 1e-6
FLUSH_FLOATS = 32 * 1024 * 1024  # 128 MiB


def _render_tiny(ren: CUDARenderer) -> dict[str, str]:
  def p(name: str, shape: tuple[int, ...], dtype, slot: int) -> UOp:
    return UOp.placeholder(shape, dtype, slot)

  def src_for(rows: int) -> str:
    spec = ReduceOutputSpec(rows=rows, dim=128, eps=EPS, out_dtype=dtypes.float32,
                            affine=True, recipe="sumsq_rsqrt_affine", reduce_op=Ops.ADD,
                            warps=rows, lanes=32, per_lane=128 // 32)
    kernel = emit_reduce_output(spec, dtypes.float32, dtypes.float16)
    u = kernel(
      p("out", (rows * 128,), dtypes.float32, 0),
      p("x", (rows * 128,), dtypes.float32, 1),
      p("w", (128,), dtypes.float16, 2))
    src = next(x.arg for x in to_program(u, ren).src if x.op is Ops.SOURCE)
    marker = 'extern "C" __global__'
    return src[src.index(marker):]

  return {"q": src_for(32), "k": src_for(8)}


HARNESS = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#ifndef INFINITY
#define INFINITY (__int_as_float(0x7f800000))
#endif
#ifndef NAN
#define NAN (__int_as_float(0x7fffffff))
#endif
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }

__SRC_TQ__
__SRC_TK__

#define WARP_SIZE 32
static __device__ __forceinline__ void ggml_cuda_pdl_sync() { /* PDL-off: no grid dependency wait */ }

static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
    x += __shfl_xor_sync(0xffffffffu, x, offset, WARP_SIZE);
  return x;
}

enum class block_reduce_method { MAX, SUM };
template<block_reduce_method m, typename T> struct block_reduce_policy;
template<> struct block_reduce_policy<block_reduce_method::SUM, float> {
  static __device__ float reduce(float v) { return warp_reduce_sum(v); }
  static __device__ float sentinel() { return 0.0f; }
};
template<block_reduce_method rm, const unsigned int bs = 0, typename T>
static __device__ T block_reduce(T val, T* shared_vals) {
  val = block_reduce_policy<rm, T>::reduce(val);
  const unsigned int block_size = bs == 0 ? blockDim.x : bs;
  if (block_size > WARP_SIZE) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    if (lane_id == 0) shared_vals[warp_id] = val;
    __syncthreads();
    val = block_reduce_policy<rm, T>::sentinel();
    if (lane_id < static_cast<int>(block_size) / WARP_SIZE) val = shared_vals[lane_id];
    return block_reduce_policy<rm, T>::reduce(val);
  }
  return val;
}

// Faithful PDL-off copy of llama.cpp rms_norm_f32<256,true> for the 1D
// (nrows x 128) q/k norm layout, mul_ncols == ncols so no modulo is needed.
__global__ void llama_rms_norm_qk(const float* x, float* dst, const float* mul, const int ncols) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  x += static_cast<int64_t>(row) * ncols;
  dst += static_cast<int64_t>(row) * ncols;
  float tmp = 0.0f;
  ggml_cuda_pdl_sync();
  for (int col = tid; col < ncols; col += 256) {
    const float xi = x[col];
    tmp += xi * xi;
  }
  extern __shared__ float s_sum[];
  tmp = block_reduce<block_reduce_method::SUM, 256>(tmp, s_sum);
  const float mean = tmp / ncols;
  const float scale = rsqrtf(mean + __EPS__);
  for (int col = tid; col < ncols; col += 256) {
    dst[col] = scale * x[col] * mul[col];
  }
}

__global__ void fill_kernel(float* d, int n, float v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = v;
}
__global__ void fill_half_kernel(half* d, int n, half v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = v;
}
__global__ void flush_kernel(float* d, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = static_cast<float>(i) * 1e-6f;
}

static void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); exit(2); }
}

static float* tq_x; static float* tq_out; static half* tq_w;
static float* tk_x; static float* tk_out; static half* tk_w;
static float* lq_x; static float* lq_out; static float* lq_w;
static float* lk_x; static float* lk_out; static float* lk_w;
static float* dummy;

static void launch_norm(int which) {
  if (which == 0) {
    reduce_output_rmsnorm_32_128<<<dim3(32,1,1), dim3(4,8)>>>(tq_out, tq_x, tq_w);
  } else if (which == 1) {
    reduce_output_rmsnorm_8_128<<<dim3(8,1,1), dim3(2,16)>>>(tk_out, tk_x, tk_w);
  } else if (which == 2) {
    llama_rms_norm_qk<<<32, 256, 32 * sizeof(float)>>>(lq_x, lq_out, lq_w, 128);
  } else {
    llama_rms_norm_qk<<<8, 256, 32 * sizeof(float)>>>(lk_x, lk_out, lk_w, 128);
  }
}

static void launch_prep(int which, int prep) {
  if (prep == 0) return;
  if (prep == 1) {
    // Write the norm's own input so it is producer-hot in L2.
    if (which == 0) fill_kernel<<<(4096 + 255) / 256, 256>>>(tq_x, 4096, 0.1f);
    else if (which == 1) fill_kernel<<<(1024 + 255) / 256, 256>>>(tk_x, 1024, 0.1f);
    else if (which == 2) fill_kernel<<<(4096 + 255) / 256, 256>>>(lq_x, 4096, 0.1f);
    else fill_kernel<<<(1024 + 255) / 256, 256>>>(lk_x, 1024, 0.1f);
  } else if (prep == 2) {
    flush_kernel<<<(FLUSH_FLOATS + 255) / 256, 256>>>(dummy, FLUSH_FLOATS);
  }
}

static double hot_loop(int which, int iters) {
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  launch_norm(which); cudaDeviceSynchronize();
  cudaEventRecord(s);
  for (int i = 0; i < iters; i++) launch_norm(which);
  cudaEventRecord(e); cudaDeviceSynchronize();
  float ms = 0; cudaEventElapsedTime(&ms, s, e);
  cudaEventDestroy(s); cudaEventDestroy(e);
  return ms * 1000.0 / iters;
}

static double cond_median(int which, int prep, int iters, int reps) {
  std::vector<cudaEvent_t> s(iters), e(iters);
  for (int i = 0; i < iters; i++) { cudaEventCreate(&s[i]); cudaEventCreate(&e[i]); }
  std::vector<float> us;
  us.reserve(static_cast<size_t>(iters) * reps);
  for (int r = 0; r < reps; r++) {
    for (int i = 0; i < iters; i++) {
      launch_prep(which, prep);
      cudaEventRecord(s[i]);
      launch_norm(which);
      cudaEventRecord(e[i]);
    }
    cudaDeviceSynchronize();
    for (int i = 0; i < iters; i++) {
      float ms = 0; cudaEventElapsedTime(&ms, s[i], e[i]);
      us.push_back(ms * 1000.0f);
    }
  }
  for (int i = 0; i < iters; i++) { cudaEventDestroy(s[i]); cudaEventDestroy(e[i]); }
  std::sort(us.begin(), us.end());
  return us[us.size() / 2];
}

int main(int argc, char** argv) {
  int hot_iters = argc > 1 ? atoi(argv[1]) : 2000;
  int cond_iters = argc > 2 ? atoi(argv[2]) : 200;
  int reps = argc > 3 ? atoi(argv[3]) : 5;
  int hot_only = argc > 4 ? atoi(argv[4]) : 0;

  check(cudaMalloc(&tq_x, 4096 * sizeof(float)), "tq_x");
  check(cudaMalloc(&tq_out, 4096 * sizeof(float)), "tq_out");
  check(cudaMalloc(&tq_w, 128 * sizeof(half)), "tq_w");
  check(cudaMalloc(&tk_x, 1024 * sizeof(float)), "tk_x");
  check(cudaMalloc(&tk_out, 1024 * sizeof(float)), "tk_out");
  check(cudaMalloc(&tk_w, 128 * sizeof(half)), "tk_w");
  check(cudaMalloc(&lq_x, 4096 * sizeof(float)), "lq_x");
  check(cudaMalloc(&lq_out, 4096 * sizeof(float)), "lq_out");
  check(cudaMalloc(&lq_w, 128 * sizeof(float)), "lq_w");
  check(cudaMalloc(&lk_x, 1024 * sizeof(float)), "lk_x");
  check(cudaMalloc(&lk_out, 1024 * sizeof(float)), "lk_out");
  check(cudaMalloc(&lk_w, 128 * sizeof(float)), "lk_w");
  check(cudaMalloc(&dummy, FLUSH_FLOATS * sizeof(float)), "dummy");

  fill_kernel<<<(4096 + 255) / 256, 256>>>(tq_x, 4096, 0.1f);
  fill_kernel<<<(1024 + 255) / 256, 256>>>(tk_x, 1024, 0.1f);
  fill_kernel<<<(4096 + 255) / 256, 256>>>(lq_x, 4096, 0.1f);
  fill_kernel<<<(1024 + 255) / 256, 256>>>(lk_x, 1024, 0.1f);
  fill_half_kernel<<<(128 + 255) / 256, 256>>>(tq_w, 128, __float2half(1.0f));
  fill_half_kernel<<<(128 + 255) / 256, 256>>>(tk_w, 128, __float2half(1.0f));
  fill_kernel<<<(128 + 255) / 256, 256>>>(lq_w, 128, 1.0f);
  fill_kernel<<<(128 + 255) / 256, 256>>>(lk_w, 128, 1.0f);
  check(cudaDeviceSynchronize(), "init sync");

  for (int which = 0; which < 4; which++) {
    const char* name = which == 0 ? "tiny_q" : which == 1 ? "tiny_k" :
                       which == 2 ? "llama_q" : "llama_k";
    printf("hot_loop %s %.4f\n", name, hot_loop(which, hot_iters));
    if (hot_only) continue;
    printf("cond %s none %.4f\n", name, cond_median(which, 0, cond_iters, reps));
    printf("cond %s fill %.4f\n", name, cond_median(which, 1, cond_iters, reps));
    printf("cond %s flush %.4f\n", name, cond_median(which, 2, cond_iters, reps));
  }
  check(cudaGetLastError(), "last error");
  cudaDeviceSynchronize();
  return 0;
}
"""


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--hot-iters", type=int, default=2000)
  ap.add_argument("--cond-iters", type=int, default=200)
  ap.add_argument("--reps", type=int, default=5)
  ap.add_argument("--hot-only", action="store_true")
  ap.add_argument("--compile-only", action="store_true",
                  help="compile the harness and copy the binary to --binary-out, then exit")
  ap.add_argument("--binary-out", type=pathlib.Path, default=None)
  ap.add_argument("--source-out", type=pathlib.Path, default=None)
  ap.add_argument("--out-json", type=pathlib.Path, required=True)
  args = ap.parse_args()

  ren = CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=False)
  srcs = _render_tiny(ren)
  cu = (HARNESS
        .replace("__SRC_TQ__", srcs["q"])
        .replace("__SRC_TK__", srcs["k"])
        .replace("FLUSH_FLOATS", str(FLUSH_FLOATS))
        .replace("__EPS__", repr(EPS)))

  with tempfile.TemporaryDirectory(prefix="nv_qk_head_norm_") as td:
    cu_path = os.path.join(td, "qk_head_norm.cu")
    binp = os.path.join(td, "qk_head_norm")
    with open(cu_path, "w") as f:
      f.write(cu)
    env = dict(os.environ)
    env["PATH"] = f"{CUDA_BIN}:" + env.get("PATH", "")
    cp = subprocess.run(
      ["nvcc", "-arch=sm_120a", "-O3", "-DNDEBUG", "-use_fast_math",
       "-std=c++17", "--ptxas-options=-v",
       cu_path, "-o", binp], capture_output=True, text=True, env=env)
    if cp.returncode != 0:
      print(cp.stderr[-8000:], file=sys.stderr)
      return 3
    ptxas = [line for line in cp.stderr.strip().splitlines() if "registers" in line or "spill" in line]

    if args.source_out is not None:
      args.source_out.parent.mkdir(parents=True, exist_ok=True)
      args.source_out.write_text(cu)
    if args.compile_only:
      if args.binary_out is None:
        print("--compile-only requires --binary-out", file=sys.stderr)
        return 2
      args.binary_out.parent.mkdir(parents=True, exist_ok=True)
      args.binary_out.write_bytes(pathlib.Path(binp).read_bytes())
      print(f"compiled_binary={args.binary_out}")
      print("\n".join(ptxas))
      return 0

    r = subprocess.run([binp, str(args.hot_iters), str(args.cond_iters), str(args.reps),
                        "1" if args.hot_only else "0"],
                       capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
      print(r.stderr[-4000:], file=sys.stderr)
      return 4

  timing: dict[str, dict[str, list[float] | float]] = {}
  for line in r.stdout.splitlines():
    m = re.match(r"hot_loop (\w+) ([0-9.]+)", line)
    if m:
      timing.setdefault(m.group(1), {})["hot_loop"] = float(m.group(2))
      continue
    m = re.match(r"cond (\w+) (\w+) ([0-9.]+)", line)
    if m:
      timing.setdefault(m.group(1), {})[f"cond_{m.group(2)}"] = float(m.group(3))

  result = {
    "schema": "tinygrad.nv_qk_head_norm_microgate.v1",
    "commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                      text=True).strip(),
    "method": "render production tinygrad CUDA + faithful llama norm.cu copy -> nvcc sm_120a -> cudaEvent",
    "eps": EPS,
    "flush_mib": (FLUSH_FLOATS * 4) // (1024 * 1024),
    "hot_iters": args.hot_iters,
    "cond_iters": args.cond_iters,
    "reps": args.reps,
    "ptxas": ptxas,
    "raw_stdout": r.stdout.strip().splitlines(),
    "timing": timing,
  }
  args.out_json.parent.mkdir(parents=True, exist_ok=True)
  args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
