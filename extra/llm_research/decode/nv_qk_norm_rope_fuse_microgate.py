#!/usr/bin/env python3
"""Q/K norm+rope fusion bit-exactness microgate (measurement tooling only).

Renders the exact production tinygrad Q/K reduce-output RMSNorm bodies
(``reduce_output_rmsnorm_32_128`` / ``reduce_output_rmsnorm_8_128``) and a
research fused variant whose epilogue also applies the full-head rotary
rotation in-register.  The control path is the current two-kernel chain
(norm -> rope); the candidate is one fused kernel.  The gate is bit-exactness:
the candidate must reproduce the control output exactly (max_abs_diff == 0),
otherwise the fusion changes the fp32 association and cannot be landed.

It changes no production model/renderer/scheduler/runtime/route file.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.codegen.late.reduce_output import _NV_MULTI_ROW_ASSOC, emit_reduce_output
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, ReduceOutputSpec, UOp

CUDA_BIN = "/usr/local/cuda-13.2/bin"
EPS = 1e-6


def emit_reduce_output_rope(spec: ReduceOutputSpec, x_dtype, weight_dtype):
  """Fused reduce-output RMSNorm + full-head rotary epilogue.

  The reduction phase is a verbatim copy of ``emit_reduce_output``'s NV
  multi-row association (``_NV_MULTI_ROW_ASSOC``), so the sumsq/scale is
  bitwise-identical to the installed kernel.  Only the epilogue changes: after
  computing each normed value it applies the half-rotate
  ``(v_lo*cos - v_hi*sin, v_hi*cos + v_lo*sin)`` before storing.  per_lane is
  4, so the epilogue is unrolled over the two (lo, hi=lo+2) pairs; the partner
  element lives at the same lane id, two epi slots away, so no cross-lane
  shuffle is required.
  """
  if spec.rows not in (8, 32) or spec.dim != 128:
    raise ValueError("reduce-output rope requires rows in (8,32) and dim 128")
  if spec.recipe != "sumsq_rsqrt_affine" or not spec.affine:
    raise ValueError("reduce-output rope requires the sumsq affine recipe")
  if spec.warps != spec.rows or spec.lanes != 32 or spec.per_lane != 4:
    raise ValueError("reduce-output rope requires warps==rows, 32 lanes, 4 per_lane")
  lane, per_lane, dim = spec.lanes, spec.per_lane, spec.dim
  P, S, t_stride, s_stride = _NV_MULTI_ROW_ASSOC[(spec.rows, dim)]
  half = per_lane // 2

  def kernel(out: UOp, x: UOp, weight: UOp, freqs: UOp) -> UOp:
    laneid = UOp.range(lane, 0, AxisType.LOCAL)
    row = UOp.range(spec.rows, 0, AxisType.GLOBAL)
    partial_lane = laneid % P
    red = UOp.range(S, 2, AxisType.REDUCE)
    base = row * dim + partial_lane * t_stride + red * s_stride
    xv = x[base].cast(dtypes.float32)
    acc = UOp.placeholder((1,), dtypes.float32, 20, AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(red)[0] + xv * xv).end(red))
    smem = UOp.placeholder((P,), dtypes.float32, 230, AddrSpace.LOCAL)
    published = smem[partial_lane].store(acc[0], laneid < P)
    ready = UOp.barrier(UOp.group(published))
    total = UOp.const(dtypes.float32, 0.0)
    for ti in range(P):
      total = total + smem.after(ready)[ti]
    scale = (total / UOp.const(dtypes.float32, float(dim)) + UOp.const(dtypes.float32, spec.eps)).sqrt().reciprocal()

    epi = UOp.range(half, 2, AxisType.LOOP)
    lo_base = row * dim + laneid + epi * lane
    hi_base = row * dim + laneid + (epi + half) * lane
    w_lo = laneid + epi * lane
    w_hi = laneid + (epi + half) * lane
    v_lo = ((x[lo_base].cast(dtypes.float32) * scale).cast(x_dtype)
            * weight[w_lo].cast(x_dtype)).cast(spec.out_dtype)
    v_hi = ((x[hi_base].cast(dtypes.float32) * scale).cast(x_dtype)
            * weight[w_hi].cast(x_dtype)).cast(spec.out_dtype)
    h = laneid + epi * lane
    cosv = freqs[h].cast(dtypes.float32)
    sinv = freqs[h + dim // 2].cast(dtypes.float32)
    lo_store = out[lo_base].store(v_lo * cosv - v_hi * sinv)
    hi_store = out[hi_base].store(v_hi * cosv + v_lo * sinv)
    return UOp.group(lo_store, hi_store).end(laneid, row, epi).sink(
      arg=KernelInfo(name=f"reduce_output_rmsnorm_rope_{spec.rows}_128", opts_to_apply=()))
  return kernel


def _render(ren: CUDARenderer, rope: bool) -> dict[str, str]:
  def p(name: str, shape: tuple[int, ...], dtype, slot: int) -> UOp:
    return UOp.placeholder(shape, dtype, slot)

  out: dict[str, str] = {}
  for key, rows in (("q", 32), ("k", 8)):
    spec = ReduceOutputSpec(rows=rows, dim=128, eps=EPS, out_dtype=dtypes.float32,
                            affine=True, recipe="sumsq_rsqrt_affine", reduce_op=Ops.ADD,
                            warps=rows, lanes=32, per_lane=4)
    emit = emit_reduce_output_rope if rope else emit_reduce_output
    if rope:
      kernel = emit(spec, dtypes.float32, dtypes.float16)
      u = kernel(
        p("out", (rows * 128,), dtypes.float32, 0),
        p("x", (rows * 128,), dtypes.float32, 1),
        p("w", (128,), dtypes.float16, 2),
        p("freqs", (128,), dtypes.float32, 3))
    else:
      kernel = emit(spec, dtypes.float32, dtypes.float16)
      u = kernel(
        p("out", (rows * 128,), dtypes.float32, 0),
        p("x", (rows * 128,), dtypes.float32, 1),
        p("w", (128,), dtypes.float16, 2))
    src = next(x.arg for x in to_program(u, ren).src if x.op is Ops.SOURCE)
    marker = 'extern "C" __global__'
    out[key] = src[src.index(marker):]
  return out


HARNESS = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#ifndef INFINITY
#define INFINITY (__int_as_float(0x7f800000))
#endif
#ifndef NAN
#define NAN (__int_as_float(0x7fffffff))
#endif
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }

__SRC_NQ__
__SRC_NK__
__SRC_FQ__
__SRC_FK__

// Faithful copy of the half-rotate rope applied by tinygrad apply_rope:
// out[0:64] = x[0:64]*cos - x[64:128]*sin ; out[64:128] = x[64:128]*cos + x[0:64]*sin
// freqs is (128,) = [cos(64), sin(64)].
__global__ void rope_128(const float* x, float* out, const float* freqs) {
  int row = blockIdx.x;
  int e = threadIdx.x;
  int h = e & 63;
  float cosv = freqs[h];
  float sinv = freqs[h + 64];
  float xi = x[row * 128 + e];
  if (e < 64) out[row * 128 + e] = xi * cosv - x[row * 128 + e + 64] * sinv;
  else        out[row * 128 + e] = xi * cosv + x[row * 128 + e - 64] * sinv;
}

static void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); exit(2); }
}

static void launch_norm(int which, float* out, float* x, half* w) {
  if (which == 0) reduce_output_rmsnorm_32_128<<<dim3(32,1,1), dim3(4,8)>>>(out, x, w);
  else            reduce_output_rmsnorm_8_128<<<dim3(8,1,1), dim3(2,16)>>>(out, x, w);
}
static void launch_fused(int which, float* out, float* x, half* w, float* freqs) {
  if (which == 0) reduce_output_rmsnorm_rope_32_128<<<dim3(32,1,1), dim3(4,8)>>>(out, x, w, freqs);
  else            reduce_output_rmsnorm_rope_8_128<<<dim3(8,1,1), dim3(2,16)>>>(out, x, w, freqs);
}

static double hot_loop(int which, int rows, int iters, float* out, float* x, half* w, float* freqs, int fused) {
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  if (fused) launch_fused(which, out, x, w, freqs); else { launch_norm(which, out, x, w); rope_128<<<rows, 128>>>(out, out, freqs); }
  cudaDeviceSynchronize();
  cudaEventRecord(s);
  for (int i = 0; i < iters; i++) {
    if (fused) launch_fused(which, out, x, w, freqs);
    else { launch_norm(which, out, x, w); rope_128<<<rows, 128>>>(out, out, freqs); }
  }
  cudaEventRecord(e); cudaDeviceSynchronize();
  float ms = 0; cudaEventElapsedTime(&ms, s, e);
  cudaEventDestroy(s); cudaEventDestroy(e);
  return ms * 1000.0 / iters;
}

static double max_abs_diff(const float* a, const float* b, int n) {
  double m = 0.0;
  for (int i = 0; i < n; i++) m = fmax(m, fabs((double)a[i] - (double)b[i]));
  return m;
}

static unsigned int f2i(float f) { unsigned int u; memcpy(&u, &f, 4); return u; }

int main(int argc, char** argv) {
  int iters = argc > 1 ? atoi(argv[1]) : 2000;
  int rows = argc > 2 ? atoi(argv[2]) : 32;  // 32 = Q, 8 = K
  int which = rows == 32 ? 0 : 1;
  int n = rows * 128;

  float* x; float* ctrl; float* cand; float* freqs; half* w;
  check(cudaMalloc(&x, n * sizeof(float)), "x");
  check(cudaMalloc(&ctrl, n * sizeof(float)), "ctrl");
  check(cudaMalloc(&cand, n * sizeof(float)), "cand");
  check(cudaMalloc(&freqs, 128 * sizeof(float)), "freqs");
  check(cudaMalloc(&w, 128 * sizeof(half)), "w");

  float* hx = (float*)malloc(n * sizeof(float));
  float* hf = (float*)malloc(128 * sizeof(float));
  half* hw = (half*)malloc(128 * sizeof(half));
  for (int i = 0; i < n; i++) hx[i] = 0.1f + 0.001f * (float)(i % 17);
  for (int i = 0; i < 128; i++) { hf[i] = (i < 64) ? 0.9f + 0.001f * (float)i : 0.4f - 0.001f * (float)(i - 64); }
  for (int i = 0; i < 128; i++) hw[i] = __float2half(1.0f + 0.001f * (float)(i % 5));
  check(cudaMemcpy(x, hx, n * sizeof(float), cudaMemcpyHostToDevice), "x h2d");
  check(cudaMemcpy(freqs, hf, 128 * sizeof(float), cudaMemcpyHostToDevice), "f h2d");
  check(cudaMemcpy(w, hw, 128 * sizeof(half), cudaMemcpyHostToDevice), "w h2d");

  // Control: norm then rope.  Candidate: fused.
  launch_norm(which, ctrl, x, w); rope_128<<<rows, 128>>>(ctrl, ctrl, freqs);
  launch_fused(which, cand, x, w, freqs);
  check(cudaDeviceSynchronize(), "sync");

  float* hc = (float*)malloc(n * sizeof(float));
  float* hd = (float*)malloc(n * sizeof(float));
  check(cudaMemcpy(hc, ctrl, n * sizeof(float), cudaMemcpyDeviceToHost), "ctrl d2h");
  check(cudaMemcpy(hd, cand, n * sizeof(float), cudaMemcpyDeviceToHost), "cand d2h");
  printf("rows=%d max_abs_diff=%.9g\n", rows, max_abs_diff(hc, hd, n));
  for (int i = 0, printed = 0; i < n && printed < 8; i++) {
    if (f2i(hc[i]) != f2i(hd[i])) {
      printf("  idx=%d ctrl=%08x (%.9g) cand=%08x (%.9g)\n",
             i, f2i(hc[i]), hc[i], f2i(hd[i]), hd[i]);
      printed++;
    }
  }
  printf("hot control us=%.4f\n", hot_loop(which, rows, iters, ctrl, x, w, freqs, 0));
  printf("hot candidate us=%.4f\n", hot_loop(which, rows, iters, cand, x, w, freqs, 1));
  check(cudaGetLastError(), "last error");
  return 0;
}
"""


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--iters", type=int, default=2000)
  ap.add_argument("--out-json", type=pathlib.Path, required=True)
  ap.add_argument("--source-out", type=pathlib.Path, default=None)
  args = ap.parse_args()

  ren = CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=False)
  ctrl = _render(ren, rope=False)
  cand = _render(ren, rope=True)
  cu = (HARNESS
        .replace("__SRC_NQ__", ctrl["q"]).replace("__SRC_NK__", ctrl["k"])
        .replace("__SRC_FQ__", cand["q"]).replace("__SRC_FK__", cand["k"]))

  with tempfile.TemporaryDirectory(prefix="nv_qk_rope_fuse_") as td:
    cu_path = os.path.join(td, "qk_rope_fuse.cu")
    binp = os.path.join(td, "qk_rope_fuse")
    with open(cu_path, "w") as f:
      f.write(cu)
    env = dict(os.environ)
    env["PATH"] = f"{CUDA_BIN}:" + env.get("PATH", "")
    cp = subprocess.run(
      ["nvcc", "-arch=sm_120a", "-O3", "-std=c++17", "--ptxas-options=-v",
       cu_path, "-o", binp],
      capture_output=True, text=True, env=env)
    if cp.returncode != 0:
      print(cp.stderr[-8000:], file=sys.stderr)
      return 3
    ptxas = [line for line in cp.stderr.strip().splitlines() if "registers" in line or "spill" in line]

    if args.source_out is not None:
      args.source_out.parent.mkdir(parents=True, exist_ok=True)
      args.source_out.write_text(cu)

    results = {}
    for rows in (32, 8):
      r = subprocess.run([binp, str(args.iters), str(rows)], capture_output=True, text=True)
      if r.returncode != 0:
        print(r.stderr[-4000:], file=sys.stderr)
        return 4
      results[str(rows)] = r.stdout.strip().splitlines()
      print(r.stdout.strip())

  parsed = {}
  for rows, lines in results.items():
    d = {}
    for line in lines:
      m = re.match(r"rows=(\d+) max_abs_diff=([0-9.e+-]+)", line)
      if m: d["max_abs_diff"] = float(m.group(2))
      m = re.match(r"hot (control|candidate) us=([0-9.]+)", line)
      if m: d[f"{m.group(1)}_us"] = float(m.group(2))
    parsed[rows] = d

  out = {
    "schema": "tinygrad.nv_qk_norm_rope_fuse_microgate.v1",
    "commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
    "method": "render production reduce_output + fused rope variant -> nvcc sm_120a -> bit-exact + cudaEvent",
    "eps": EPS,
    "iters": args.iters,
    "ptxas": ptxas,
    "results": parsed,
    "verdicts": {
      f"Q_rows_32_bit_exact": parsed.get("32", {}).get("max_abs_diff") == 0.0,
      f"K_rows_8_bit_exact": parsed.get("8", {}).get("max_abs_diff") == 0.0,
    },
  }
  args.out_json.parent.mkdir(parents=True, exist_ok=True)
  args.out_json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
  print(json.dumps(out["verdicts"], indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
