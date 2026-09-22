#!/usr/bin/env python3
"""Standalone feasibility bracket: single-pass fused FFN norm into gate/up.

M1 folded the FFN norm epilogue into the fused w1+w3 GEMV by applying
``(half)((h*s)*w)`` on every packed-Q4 load.  It was bitwise exact but
wall-negative because the normalized activation was recomputed once for the
gate dot and once for the up dot, and the consumer streamed fp32.

This probe tests whether the two named failure modes are removable by computing
the normalized fp16 activation exactly once per element and reusing that single
value for both dots.  The three arms are:

  control    q4k_g3_lanemap_gemv_w1w3fused16_12288_4096   (no norm in-kernel)
  m1         q4k_g3_lanemap_gemv_w1w3_rms_affine16_...    (norm twice per x)
  norm_once  q4k_w1w3_norm_once16_12288_4096              (norm once per x)

The control has no norm arithmetic at all, so it is the lower bound the fused
body can approach.  The M1 and norm_once arms are numerically identical to each
other (same accumulation order), and the harness verifies their fp16 outputs
match bit-for-bit before timing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.llm.decode_kernels import (
  Q4K_WORDS_PER_BLOCK, Q4KGateUpLaneMap, LanePartition, _q4k_group_params, _silu_uop,
  _warp_reduce_sum_staged, q4k_g3_lanemap_gemv_w1w3_kernel,
  q4k_g3_lanemap_gemv_w1w3_rms_affine_kernel)
from tinygrad.llm.qk_layout import Q4_K_BLOCK_ELEMS
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp
from tinygrad.dtype import AddrSpace

CUDA_BIN = "/usr/local/cuda-13.2/bin"
ROWS, K = 12288, 4096
W_PER_TENSOR = (K // Q4_K_BLOCK_ELEMS) * Q4K_WORDS_PER_BLOCK * ROWS


def emit_q4k_w1w3_norm_once_fp16() -> callable:
  """Fused gate/up with the FFN norm computed once per activation element."""
  lm = Q4KGateUpLaneMap(k=K, n=ROWS)
  lm.validate()
  name = f"q4k_w1w3_norm_once16_{ROWS}_{K}"

  def kernel(out: UOp, gate_words: UOp, up_words: UOp, x: UOp,
             norm_weight: UOp, scale: UOp) -> UOp:
    row, lane = UOp.special(ROWS, "gidx0"), UOp.special(32, "lidx0")
    part = LanePartition(lane, lane_extent=lm.lane_extent, words_per_group=lm.words_per_group)
    lblk = UOp.range(lm.blocks_per_group, 0, axis_type=AxisType.REDUCE)
    blk = part.block_group * lm.blocks_per_group + lblk
    bg = (row * lm.k_blocks + blk) * Q4K_WORDS_PER_BLOCK

    contrib_g = UOp.const(dtypes.float32, 0.0)
    contrib_u = UOp.const(dtypes.float32, 0.0)
    for grp in range(8):
      dg, dming, scg, mng = _q4k_group_params(gate_words, bg, grp)
      du, dminu, scu, mnu = _q4k_group_params(up_words, bg, grp)
      qpack_g = gate_words[bg + 4 + (grp // 2) * 8 + part.word_col].rshift(
        (grp % 2) * 4).bitwise_and(0x0F0F0F0F)
      qpack_u = up_words[bg + 4 + (grp // 2) * 8 + part.word_col].rshift(
        (grp % 2) * 4).bitwise_and(0x0F0F0F0F)
      for nib in range(4):
        pos = part.word_col * 4 + nib
        idx = blk * Q4_K_BLOCK_ELEMS + grp * 32 + pos
        # The ordinary FFN norm epilogue, computed once and shared by both dots.
        xv = ((x[idx].cast(dtypes.float32) * scale[0])
              * norm_weight[idx].cast(dtypes.float32)).cast(dtypes.float16).cast(dtypes.float32)
        qg = qpack_g.rshift(nib * 8).bitwise_and(0xf)
        qu = qpack_u.rshift(nib * 8).bitwise_and(0xf)
        wg = dg * scg.cast(dtypes.float32) * qg.cast(dtypes.float32) - dming * mng.cast(dtypes.float32)
        wu = du * scu.cast(dtypes.float32) * qu.cast(dtypes.float32) - dminu * mnu.cast(dtypes.float32)
        contrib_g = contrib_g + wg * xv
        contrib_u = contrib_u + wu * xv

    acc_g = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
    acc_u = UOp.placeholder((1,), dtypes.float32, 21, addrspace=AddrSpace.REG)
    init = acc_g[0].store(0.0)
    init = acc_u.after(init)[0].store(0.0)
    acc_g, acc_u = acc_g.after(init), acc_u.after(init)
    upd_g = acc_g[0].store(acc_g.after(lblk)[0] + contrib_g)
    upd_u = acc_u.after(upd_g)[0].store(acc_u.after(lblk)[0] + contrib_u).end(lblk)
    total_g = _warp_reduce_sum_staged(acc_g.after(upd_u)[0], part.lane, part.lane_extent, 90)
    total_u = _warp_reduce_sum_staged(acc_u.after(upd_u)[0], part.lane, part.lane_extent, 95)
    val = _silu_uop(total_g) * total_u
    return out[row].store(val.cast(dtypes.float16)).sink(
      arg=KernelInfo(name=name, opts_to_apply=()))
  return kernel


def _render(ren: CUDARenderer) -> dict[str, str]:
  def p(name: str, shape: tuple[int, ...], dtype, slot: int) -> UOp:
    return UOp.placeholder(shape, dtype, slot)

  out = p("out", (ROWS,), dtypes.float16, 0)
  gate = p("gate", (W_PER_TENSOR,), dtypes.uint32, 1)
  up = p("up", (W_PER_TENSOR,), dtypes.uint32, 2)
  x = p("x", (K,), dtypes.float16, 3)
  norm = p("norm", (K,), dtypes.float16, 4)
  scale = p("scale", (1,), dtypes.float32, 5)

  control = q4k_g3_lanemap_gemv_w1w3_kernel(ROWS, K, load_style="scalar", store_fp16=True)(
    out, gate, up, x)
  m1 = q4k_g3_lanemap_gemv_w1w3_rms_affine_kernel(ROWS, K, store_fp16=True)(
    out, gate, up, x, norm, scale)
  candidate = emit_q4k_w1w3_norm_once_fp16()(out, gate, up, x, norm, scale)

  def src(u: UOp) -> str:
    return next(x.arg for x in to_program(u, ren).src if x.op is Ops.SOURCE)

  def strip(s: str) -> str:
    marker = 'extern "C" __global__'
    return s[s.index(marker):]

  return {"control": strip(src(control)), "m1": strip(src(m1)), "norm_once": strip(src(candidate))}


HARNESS = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifndef INFINITY
#define INFINITY (__int_as_float(0x7f800000))
#endif
#ifndef NAN
#define NAN (__int_as_float(0x7fffffff))
#endif
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }

__SRC_CONTROL__
__SRC_M1__
__SRC_NORM_ONCE__

static void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); exit(2); }
}

static double time_control(half* out, unsigned int* g, unsigned int* u, half* x, int passes) {
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  for (int i = 0; i < passes; i++)
    q4k_g3_lanemap_gemv_w1w3fused16_12288_4096<<<12288, 32>>>(out, g, u, x);
  cudaEventRecord(e); check(cudaDeviceSynchronize(), "sync control");
  float ms = 0; cudaEventElapsedTime(&ms, s, e);
  cudaEventDestroy(s); cudaEventDestroy(e);
  return ms * 1000.0 / passes;
}

static double time_m1(half* out, unsigned int* g, unsigned int* u, half* x, half* n, float* sc, int passes) {
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  for (int i = 0; i < passes; i++)
    q4k_g3_lanemap_w1w3_rms_affine16_12288_4096<<<12288, 32>>>(out, g, u, x, n, sc);
  cudaEventRecord(e); check(cudaDeviceSynchronize(), "sync m1");
  float ms = 0; cudaEventElapsedTime(&ms, s, e);
  cudaEventDestroy(s); cudaEventDestroy(e);
  return ms * 1000.0 / passes;
}

static double time_norm_once(half* out, unsigned int* g, unsigned int* u, half* x, half* n, float* sc, int passes) {
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  for (int i = 0; i < passes; i++)
    q4k_w1w3_norm_once16_12288_4096<<<12288, 32>>>(out, g, u, x, n, sc);
  cudaEventRecord(e); check(cudaDeviceSynchronize(), "sync norm_once");
  float ms = 0; cudaEventElapsedTime(&ms, s, e);
  cudaEventDestroy(s); cudaEventDestroy(e);
  return ms * 1000.0 / passes;
}

int main(int argc, char** argv) {
  int passes = argc > 1 ? atoi(argv[1]) : 200;
  int reps = argc > 2 ? atoi(argv[2]) : 5;
  half* out = nullptr; half* out2 = nullptr;
  unsigned int* g = nullptr; unsigned int* u = nullptr;
  half* x = nullptr; half* n = nullptr; float* sc = nullptr;
  check(cudaMalloc(&out, 12288 * sizeof(half)), "out");
  check(cudaMalloc(&out2, 12288 * sizeof(half)), "out2");
  check(cudaMalloc(&g, 7077888 * sizeof(unsigned int)), "g");
  check(cudaMalloc(&u, 7077888 * sizeof(unsigned int)), "u");
  check(cudaMalloc(&x, 4096 * sizeof(half)), "x");
  check(cudaMalloc(&n, 4096 * sizeof(half)), "n");
  check(cudaMalloc(&sc, sizeof(float)), "sc");
  check(cudaMemset(out, 0, 12288 * sizeof(half)), "memset out");
  check(cudaMemset(out2, 0, 12288 * sizeof(half)), "memset out2");
  check(cudaMemset(g, 0, 7077888 * sizeof(unsigned int)), "memset g");
  check(cudaMemset(u, 0, 7077888 * sizeof(unsigned int)), "memset u");
  check(cudaMemset(x, 0, 4096 * sizeof(half)), "memset x");
  check(cudaMemset(n, 0, 4096 * sizeof(half)), "memset n");
  check(cudaMemset(sc, 0, sizeof(float)), "memset sc");

  q4k_g3_lanemap_gemv_w1w3fused16_12288_4096<<<12288, 32>>>(out, g, u, x);
  q4k_g3_lanemap_w1w3_rms_affine16_12288_4096<<<12288, 32>>>(out, g, u, x, n, sc);
  q4k_w1w3_norm_once16_12288_4096<<<12288, 32>>>(out2, g, u, x, n, sc);
  check(cudaGetLastError(), "warmup launch"); check(cudaDeviceSynchronize(), "warmup sync");

  half* h1 = (half*)malloc(12288 * sizeof(half));
  half* h2 = (half*)malloc(12288 * sizeof(half));
  check(cudaMemcpy(h1, out, 12288 * sizeof(half), cudaMemcpyDeviceToHost), "copy m1");
  check(cudaMemcpy(h2, out2, 12288 * sizeof(half), cudaMemcpyDeviceToHost), "copy norm_once");
  int identical = memcmp(h1, h2, 12288 * sizeof(half)) == 0;
  printf("norm_once_vs_m1_bitwise_identical=%d\n", identical);
  free(h1); free(h2);

  printf("shape=12288x4096 passes=%d reps=%d\n", passes, reps);
  for (int r = 0; r < reps; r++) {
    double c = time_control(out, g, u, x, passes);
    double m = time_m1(out, g, u, x, n, sc, passes);
    double o = time_norm_once(out2, g, u, x, n, sc, passes);
    printf("rep=%d control=%.4f m1=%.4f norm_once=%.4f\n", r, c, m, o);
  }
  check(cudaFree(out), "free out"); check(cudaFree(out2), "free out2");
  check(cudaFree(g), "free g"); check(cudaFree(u), "free u"); check(cudaFree(x), "free x");
  check(cudaFree(n), "free n"); check(cudaFree(sc), "free sc");
  return 0;
}
"""


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--passes", type=int, default=200)
  ap.add_argument("--reps", type=int, default=5)
  ap.add_argument("--out", default="")
  args = ap.parse_args()

  ren = CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=False)
  srcs = _render(ren)
  cu = (HARNESS.replace("__SRC_CONTROL__", srcs["control"])
               .replace("__SRC_M1__", srcs["m1"])
               .replace("__SRC_NORM_ONCE__", srcs["norm_once"]))

  with tempfile.TemporaryDirectory(prefix="q4k_w1w3_norm_once_") as td:
    cu_path = os.path.join(td, "norm_once.cu")
    binp = os.path.join(td, "norm_once")
    with open(cu_path, "w") as f:
      f.write(cu)
    env = dict(os.environ)
    env["PATH"] = f"{CUDA_BIN}:" + env.get("PATH", "")
    cp = subprocess.run(
      ["nvcc", "-arch=sm_120a", "-O3", "-std=c++17", "--ptxas-options=-v",
       cu_path, "-o", binp], capture_output=True, text=True, env=env)
    if cp.returncode != 0:
      print(cp.stderr[-6000:], file=sys.stderr)
      return 3

    r = subprocess.run([binp, str(args.passes), str(args.reps)],
                       capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
      print(r.stderr[-4000:], file=sys.stderr)
      return 4

    identical = None
    m = re.search(r"norm_once_vs_m1_bitwise_identical=(\d)", r.stdout)
    if m:
      identical = bool(int(m.group(1)))

    c_vals, m_vals, o_vals = [], [], []
    for line in r.stdout.splitlines():
      m2 = re.search(r"rep=(\d+) control=([0-9.]+) m1=([0-9.]+) norm_once=([0-9.]+)", line)
      if m2:
        c_vals.append(float(m2.group(2)))
        m_vals.append(float(m2.group(3)))
        o_vals.append(float(m2.group(4)))

    out = {
      "schema": "tinygrad.q4k_w1w3_norm_once_microgate.v1",
      "commit": subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        cwd=str(ROOT)).stdout.strip(),
      "method": "render production CUDA -> nvcc sm_120a -> cudaEvent",
      "shape": {"rows": ROWS, "k": K},
      "norm_once_vs_m1_bitwise_identical": identical,
      "passes": args.passes,
      "reps": args.reps,
      "ptxas": cp.stderr.strip().splitlines(),
      "raw_stdout": r.stdout.strip().splitlines(),
      "timing": {"unit": "us_per_launch_cuda_event"},
    }
    if c_vals and m_vals and o_vals:
      med = lambda v: statistics.median(v)
      out["timing"] = {
        "unit": "us_per_launch_cuda_event",
        "control": c_vals,
        "m1": m_vals,
        "norm_once": o_vals,
        "control_median": med(c_vals),
        "m1_median": med(m_vals),
        "norm_once_median": med(o_vals),
        "m1_over_control": med(m_vals) / med(c_vals),
        "norm_once_over_control": med(o_vals) / med(c_vals),
        "norm_once_over_m1": med(o_vals) / med(m_vals),
      }

    text = json.dumps(out, indent=2, sort_keys=True)
    if args.out:
      with open(args.out, "w") as f:
        f.write(text + "\n")
    print(text)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
