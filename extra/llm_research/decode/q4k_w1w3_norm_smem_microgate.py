#!/usr/bin/env python3
"""Standalone feasibility bracket: shared-memory-staged FFN norm before the K-loop.

M1 (``q4k_g3_lanemap_gemv_w1w3_rms_affine16``) and ``norm_once`` both lost
~1.48x vs the no-norm control because the FFN norm arithmetic stays inside the
memory-bound dot loop (80 vs 74 registers) and the fused consumer streams the
activation from global.  The corpus's one remaining unrefuted construction is
to stage the normalized fp16 activation into shared memory ONCE, before the
K-loop, so the reduce/scale runs once per element outside the hot loop and the
dot loop reads fp16 from LDS instead of recomputing the norm per nibble.

The arms are:

  control     q4k_g3_lanemap_gemv_w1w3fused16_...      (no norm in-kernel)
  m1          q4k_g3_lanemap_gemv_w1w3_rms_affine16_...(norm twice per x)
  norm_once   q4k_w1w3_norm_once16_...                 (norm once per x, in-loop)
  smem_norm   q4k_w1w3_norm_smem16_...                 (norm once, staged to smem)

``smem_norm`` is bitwise-identical to ``norm_once`` (same formula, same
accumulation order; the only difference is where the normalized value lives
between the one-time staging pass and the dot loop).
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
  """Fused gate/up with the FFN norm computed once per activation element (in-loop)."""
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


def emit_q4k_w1w3_norm_smem_fp16() -> callable:
  """Fused gate/up with the FFN norm staged to shared memory before the K-loop.

  Scalar spelling: each of the 32 lanes normalizes K/threads elements with one
  scalar fp16 global load and one scalar fp16 smem store per element."""
  lm = Q4KGateUpLaneMap(k=K, n=ROWS)
  lm.validate()
  name = f"q4k_w1w3_norm_smem16_{ROWS}_{K}"
  threads = 32
  stage_iters = K // threads

  def kernel(out: UOp, gate_words: UOp, up_words: UOp, x: UOp,
             norm_weight: UOp, scale: UOp) -> UOp:
    row, lane = UOp.special(ROWS, "gidx0"), UOp.special(threads, "lidx0")
    part = LanePartition(lane, lane_extent=lm.lane_extent, words_per_group=lm.words_per_group)

    # One-time staging pass: each lane normalizes K/threads elements and writes
    # fp16 into shared memory.  This is the SAME scalar formula norm_once uses,
    # so the stored bytes are bitwise-identical.
    xsh = UOp.placeholder((K,), dtypes.float16, 22, addrspace=AddrSpace.LOCAL)
    stage = UOp.range(stage_iters, 0, axis_type=AxisType.REDUCE)
    sidx = stage * UOp.const(dtypes.weakint, threads) + lane
    sval = ((x[sidx].cast(dtypes.float32) * scale[0])
            * norm_weight[sidx].cast(dtypes.float32)).cast(dtypes.float16)
    xstore = xsh[sidx].store(sval)
    barrier = UOp.barrier(UOp.group(xstore.end(stage)))

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
        xv = xsh.after(barrier)[idx].cast(dtypes.float32)
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


def emit_q4k_w1w3_norm_smem_v4_fp16() -> callable:
  """Vectorized (half4) spelling of the smem-staged norm.

  Identical construction to ``emit_q4k_w1w3_norm_smem_fp16`` except the staging
  pass loads/stores fp16.vec(4) (8-byte vector), matching the quad-style x
  staging pattern.  This isolates the staging instruction count from the
  construction itself."""
  lm = Q4KGateUpLaneMap(k=K, n=ROWS)
  lm.validate()
  name = f"q4k_w1w3_norm_smemv4_16_{ROWS}_{K}"
  threads = 32
  stage_iters = K // (threads * 4)

  def kernel(out: UOp, gate_words: UOp, up_words: UOp, x: UOp,
             norm_weight: UOp, scale: UOp) -> UOp:
    row, lane = UOp.special(ROWS, "gidx0"), UOp.special(threads, "lidx0")
    part = LanePartition(lane, lane_extent=lm.lane_extent, words_per_group=lm.words_per_group)

    xsh = UOp.placeholder((K,), dtypes.float16, 22, addrspace=AddrSpace.LOCAL)
    stage = UOp.range(stage_iters, 0, axis_type=AxisType.REDUCE)
    sidx = (stage * UOp.const(dtypes.weakint, threads) + lane).mul(UOp.const(dtypes.weakint, 4))
    xvec = x[sidx].load(dtype=dtypes.float16.vec(4))
    nvec = norm_weight[sidx].load(dtype=dtypes.float16.vec(4))
    sval = ((xvec.cast(dtypes.float32) * scale[0].broadcast(4))
            * nvec.cast(dtypes.float32)).cast(dtypes.float16)
    xstore = xsh[sidx].store(sval)
    barrier = UOp.barrier(UOp.group(xstore.end(stage)))

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
        xv = xsh.after(barrier)[idx].cast(dtypes.float32)
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
  norm_once = emit_q4k_w1w3_norm_once_fp16()(out, gate, up, x, norm, scale)
  smem_norm = emit_q4k_w1w3_norm_smem_fp16()(out, gate, up, x, norm, scale)
  smem_v4 = emit_q4k_w1w3_norm_smem_v4_fp16()(out, gate, up, x, norm, scale)

  def src(u: UOp) -> str:
    return next(x.arg for x in to_program(u, ren).src if x.op is Ops.SOURCE)

  def strip(s: str) -> str:
    marker = 'extern "C" __global__'
    return s[s.index(marker):]

  return {
    "control": strip(src(control)),
    "m1": strip(src(m1)),
    "norm_once": strip(src(norm_once)),
    "smem_norm": strip(src(smem_norm)),
    "smem_v4": strip(src(smem_v4)),
  }


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
struct __align__(8) half4 { half x, y, z, w; };
__device__ half4 make_half4(half x, half y, half z, half w) { half4 r={x,y,z,w}; return r; }

__SRC_CONTROL__
__SRC_M1__
__SRC_NORM_ONCE__
__SRC_SMEM_NORM__
__SRC_SMEM_V4__

static void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); exit(2); }
}

static double time_control(half* out, unsigned int* g, unsigned int* u, half* x, int passes) {
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  for (int i = 0; i < passes; i++)
    q4k_g3_lanemap_gemv_w1w3fused16_12288_4096<<<12288, 32>>>(out, g, u, x);
  cudaEventRecord(e); check(cudaDeviceSynchronize(), "sync");
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

static double time_smem_norm(half* out, unsigned int* g, unsigned int* u, half* x, half* n, float* sc, int passes) {
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  for (int i = 0; i < passes; i++)
    q4k_w1w3_norm_smem16_12288_4096<<<12288, 32>>>(out, g, u, x, n, sc);
  cudaEventRecord(e); check(cudaDeviceSynchronize(), "sync smem_norm");
  float ms = 0; cudaEventElapsedTime(&ms, s, e);
  cudaEventDestroy(s); cudaEventDestroy(e);
  return ms * 1000.0 / passes;
}

static double time_smem_v4(half* out, unsigned int* g, unsigned int* u, half* x, half* n, float* sc, int passes) {
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  for (int i = 0; i < passes; i++)
    q4k_w1w3_norm_smemv4_16_12288_4096<<<12288, 32>>>(out, g, u, x, n, sc);
  cudaEventRecord(e); check(cudaDeviceSynchronize(), "sync smem_v4");
  float ms = 0; cudaEventElapsedTime(&ms, s, e);
  cudaEventDestroy(s); cudaEventDestroy(e);
  return ms * 1000.0 / passes;
}

int main(int argc, char** argv) {
  int passes = argc > 1 ? atoi(argv[1]) : 200;
  int reps = argc > 2 ? atoi(argv[2]) : 5;
  half* out = nullptr; half* out2 = nullptr; half* out3 = nullptr; half* out4 = nullptr;
  unsigned int* g = nullptr; unsigned int* u = nullptr;
  half* x = nullptr; half* n = nullptr; float* sc = nullptr;
  check(cudaMalloc(&out, 12288 * sizeof(half)), "out");
  check(cudaMalloc(&out2, 12288 * sizeof(half)), "out2");
  check(cudaMalloc(&out3, 12288 * sizeof(half)), "out3");
  check(cudaMalloc(&out4, 12288 * sizeof(half)), "out4");
  check(cudaMalloc(&g, 7077888 * sizeof(unsigned int)), "g");
  check(cudaMalloc(&u, 7077888 * sizeof(unsigned int)), "u");
  check(cudaMalloc(&x, 4096 * sizeof(half)), "x");
  check(cudaMalloc(&n, 4096 * sizeof(half)), "n");
  check(cudaMalloc(&sc, sizeof(float)), "sc");
  check(cudaMemset(out, 0, 12288 * sizeof(half)), "memset out");
  check(cudaMemset(out2, 0, 12288 * sizeof(half)), "memset out2");
  check(cudaMemset(out3, 0, 12288 * sizeof(half)), "memset out3");
  check(cudaMemset(out4, 0, 12288 * sizeof(half)), "memset out4");
  check(cudaMemset(g, 0, 7077888 * sizeof(unsigned int)), "memset g");
  check(cudaMemset(u, 0, 7077888 * sizeof(unsigned int)), "memset u");
  check(cudaMemset(x, 0, 4096 * sizeof(half)), "memset x");
  check(cudaMemset(n, 0, 4096 * sizeof(half)), "memset n");
  check(cudaMemset(sc, 0, sizeof(float)), "memset sc");

  q4k_g3_lanemap_gemv_w1w3fused16_12288_4096<<<12288, 32>>>(out, g, u, x);
  q4k_g3_lanemap_w1w3_rms_affine16_12288_4096<<<12288, 32>>>(out, g, u, x, n, sc);
  q4k_w1w3_norm_once16_12288_4096<<<12288, 32>>>(out2, g, u, x, n, sc);
  q4k_w1w3_norm_smem16_12288_4096<<<12288, 32>>>(out3, g, u, x, n, sc);
  q4k_w1w3_norm_smemv4_16_12288_4096<<<12288, 32>>>(out4, g, u, x, n, sc);
  check(cudaGetLastError(), "warmup launch"); check(cudaDeviceSynchronize(), "warmup sync");

  half* h1 = (half*)malloc(12288 * sizeof(half));
  half* h2 = (half*)malloc(12288 * sizeof(half));
  half* h3 = (half*)malloc(12288 * sizeof(half));
  half* h4 = (half*)malloc(12288 * sizeof(half));
  check(cudaMemcpy(h1, out, 12288 * sizeof(half), cudaMemcpyDeviceToHost), "copy m1");
  check(cudaMemcpy(h2, out2, 12288 * sizeof(half), cudaMemcpyDeviceToHost), "copy norm_once");
  check(cudaMemcpy(h3, out3, 12288 * sizeof(half), cudaMemcpyDeviceToHost), "copy smem_norm");
  check(cudaMemcpy(h4, out4, 12288 * sizeof(half), cudaMemcpyDeviceToHost), "copy smem_v4");
  int smem_vs_norm_once = memcmp(h2, h3, 12288 * sizeof(half)) == 0;
  int smemv4_vs_norm_once = memcmp(h2, h4, 12288 * sizeof(half)) == 0;
  int norm_once_vs_m1 = memcmp(h1, h2, 12288 * sizeof(half)) == 0;
  printf("smem_vs_norm_once_bitwise_identical=%d\n", smem_vs_norm_once);
  printf("smemv4_vs_norm_once_bitwise_identical=%d\n", smemv4_vs_norm_once);
  printf("norm_once_vs_m1_bitwise_identical=%d\n", norm_once_vs_m1);
  free(h1); free(h2); free(h3); free(h4);

  printf("shape=12288x4096 passes=%d reps=%d\n", passes, reps);
  for (int r = 0; r < reps; r++) {
    double c = time_control(out, g, u, x, passes);
    double m = time_m1(out, g, u, x, n, sc, passes);
    double o = time_norm_once(out2, g, u, x, n, sc, passes);
    double s = time_smem_norm(out3, g, u, x, n, sc, passes);
    double v = time_smem_v4(out4, g, u, x, n, sc, passes);
    printf("rep=%d control=%.4f m1=%.4f norm_once=%.4f smem_norm=%.4f smem_v4=%.4f\n", r, c, m, o, s, v);
  }
  check(cudaFree(out), "free out"); check(cudaFree(out2), "free out2"); check(cudaFree(out3), "free out3");
  check(cudaFree(out4), "free out4");
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
               .replace("__SRC_NORM_ONCE__", srcs["norm_once"])
               .replace("__SRC_SMEM_NORM__", srcs["smem_norm"])
               .replace("__SRC_SMEM_V4__", srcs["smem_v4"]))

  with tempfile.TemporaryDirectory(prefix="q4k_w1w3_norm_smem_") as td:
    cu_path = os.path.join(td, "norm_smem.cu")
    binp = os.path.join(td, "norm_smem")
    with open(cu_path, "w") as f:
      f.write(cu)
    env = dict(os.environ)
    env["PATH"] = f"{CUDA_BIN}:" + env.get("PATH", "")
    cp = subprocess.run(
      ["nvcc", "-arch=sm_120a", "-O3", "-std=c++17", "--ptxas-options=-v",
       cu_path, "-o", binp], capture_output=True, text=True, env=env)
    if cp.returncode != 0:
      print(cp.stderr[-8000:], file=sys.stderr)
      return 3

    r = subprocess.run([binp, str(args.passes), str(args.reps)],
                       capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
      print(r.stderr[-4000:], file=sys.stderr)
      return 4

    smem_id = None
    m = re.search(r"smem_vs_norm_once_bitwise_identical=(\d)", r.stdout)
    if m: smem_id = bool(int(m.group(1)))
    smemv4_id = None
    m = re.search(r"smemv4_vs_norm_once_bitwise_identical=(\d)", r.stdout)
    if m: smemv4_id = bool(int(m.group(1)))
    n1_id = None
    m = re.search(r"norm_once_vs_m1_bitwise_identical=(\d)", r.stdout)
    if m: n1_id = bool(int(m.group(1)))

    c_vals, m_vals, o_vals, s_vals, v_vals = [], [], [], [], []
    for line in r.stdout.splitlines():
      m2 = re.search(
        r"rep=(\d+) control=([0-9.]+) m1=([0-9.]+) norm_once=([0-9.]+) smem_norm=([0-9.]+) smem_v4=([0-9.]+)", line)
      if m2:
        c_vals.append(float(m2.group(2)))
        m_vals.append(float(m2.group(3)))
        o_vals.append(float(m2.group(4)))
        s_vals.append(float(m2.group(5)))
        v_vals.append(float(m2.group(6)))

    out = {
      "schema": "tinygrad.q4k_w1w3_norm_smem_microgate.v1",
      "commit": subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        cwd=str(ROOT)).stdout.strip(),
      "method": "render production CUDA -> nvcc sm_120a -> cudaEvent",
      "shape": {"rows": ROWS, "k": K},
      "smem_vs_norm_once_bitwise_identical": smem_id,
      "smemv4_vs_norm_once_bitwise_identical": smemv4_id,
      "norm_once_vs_m1_bitwise_identical": n1_id,
      "passes": args.passes,
      "reps": args.reps,
      "ptxas": cp.stderr.strip().splitlines(),
      "raw_stdout": r.stdout.strip().splitlines(),
      "timing": {"unit": "us_per_launch_cuda_event"},
    }
    if c_vals and m_vals and o_vals and s_vals and v_vals:
      med = lambda v: statistics.median(v)
      out["timing"] = {
        "unit": "us_per_launch_cuda_event",
        "control": c_vals,
        "m1": m_vals,
        "norm_once": o_vals,
        "smem_norm": s_vals,
        "smem_v4": v_vals,
        "control_median": med(c_vals),
        "m1_median": med(m_vals),
        "norm_once_median": med(o_vals),
        "smem_norm_median": med(s_vals),
        "smem_v4_median": med(v_vals),
        "m1_over_control": med(m_vals) / med(c_vals),
        "norm_once_over_control": med(o_vals) / med(c_vals),
        "smem_norm_over_control": med(s_vals) / med(c_vals),
        "smem_norm_over_norm_once": med(s_vals) / med(o_vals),
        "smem_v4_over_control": med(v_vals) / med(c_vals),
        "smem_v4_over_norm_once": med(v_vals) / med(o_vals),
      }

    text = json.dumps(out, indent=2, sort_keys=True)
    if args.out:
      with open(args.out, "w") as f:
        f.write(text + "\n")
    print(text)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
