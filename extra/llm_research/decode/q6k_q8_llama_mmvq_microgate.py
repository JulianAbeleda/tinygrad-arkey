#!/usr/bin/env python3
"""Q6_K/Q8_1 llama-exact MMVQ microgate (vectorized unpack, strided warps).

This is the "is the Q6 attention-V wall real" probe.  The installed shared-Q8
Q6 consumer decodes one Q6 byte at a time and assigns four contiguous Q6
blocks to each warp.  llama.cpp's ``vec_dot_q6_K_q8_1_impl_mmvq`` instead:

  * one output row per CTA, 32 lanes x 4 warps,
  * each lane loads two 4-byte ints (ql / qh) and produces two packed int8x4
    values with pure bit ops,
  * each lane owns 8 values per Q6 block, and the warp strides its four blocks
    across the row (block = warp + 4*block_rel), not contiguously,
  * the 32-bias is applied as ``dp4a(w, x) - 32*dp4a(0x01010101, x)``, which
    is algebraically identical to llama's ``__vsubss4(w, 0x20)``.

This file is research-only.  It does not change any production route.
"""
from __future__ import annotations
import argparse, hashlib, json, statistics, subprocess, time
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.codegen.late.int8_dot import int8x4_dot
from tinygrad.codegen.late.warp_reduce import _staged_shfl
from tinygrad.dtype import AddrSpace
from tinygrad.llm.decode_kernels import (Q6K_HALFWORDS_PER_BLOCK, _f16_half, _q6k_byte,
  emit_q6k_gemv_kernel, q6k_spec_for_role)
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec, execute_research_program
from tinygrad.llm.shared_q8_attention import _emit_q8_provider
from tinygrad.uop.ops import AxisType, KernelInfo, UOp
from extra.llm_research.decode.route_class_numerics import _make_q6k_halfs
from extra.llm_research.layout import q6_k_reference

ROWS, K, K_BLOCKS = 1024, 4096, 16
WARP, WARPS = 32, 4
BLOCKS_PER_WARP = K_BLOCKS // WARPS
Q8_PACKS, Q8_GROUPS = K // 4, K // 32


def _q8_d(packed: UOp, group: UOp) -> UOp:
  return packed[Q8_PACKS + group].bitwise_and(0xffff).cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)


def _q6_scale_i32(halfs: UOp, base: UOp, scale_idx: UOp) -> UOp:
  return _q6k_byte(halfs, base, 192 + scale_idx).cast(dtypes.uint8).bitcast(dtypes.int8).cast(dtypes.int32)


def emit_q6k_q8_llama_mmvq(interleaved: bool = True):
  """One row per CTA, llama's strided four-warp Q6 MMVQ ownership."""
  def kernel(out: UOp, halfs: UOp, packed: UOp) -> UOp:
    row = UOp.special(ROWS, "gidx0")
    lid = UOp.special(WARP * WARPS, "lidx0")
    warp, lane = lid // WARP, lid % WARP
    block_rel = UOp.range(BLOCKS_PER_WARP, 0, axis_type=AxisType.REDUCE)
    block = warp + WARPS * block_rel if interleaved else warp * BLOCKS_PER_WARP + block_rel
    base = (row * K_BLOCKS + block) * Q6K_HALFWORDS_PER_BLOCK

    # get_int_b2(bq6_K->ql, lane): two halfwords, low halfword first.
    vl = halfs[base + lane * 2].cast(dtypes.uint32).bitwise_or(
      halfs[base + lane * 2 + 1].cast(dtypes.uint32).lshift(16))
    # get_int_b2(bq6_K->qh, 8*(lane//16) + lane%8) >> 2*((lane%16)//8).
    qh_half = 16 * (lane // 16) + 2 * (lane % 8)
    vh = halfs[base + 64 + qh_half].cast(dtypes.uint32).bitwise_or(
      halfs[base + 65 + qh_half].cast(dtypes.uint32).lshift(16))
    vh = vh.rshift(2 * ((lane % 16) // 8))

    scale_idx = 8 * (lane // 16) + (lane % 16) // 4
    q8_group0 = 4 * (lane // 16) + (lane % 16) // 8
    contrib = UOp.const(dtypes.float32, 0.0)
    for i in range(2):
      low = vl.rshift(4 * i).bitwise_and(0x0F0F0F0F)
      high = vh.rshift(4 * i).lshift(4).bitwise_and(0x30303030)
      qword = low.bitwise_or(high)
      q8_group = q8_group0 + 2 * i
      xword = packed[block * 64 + q8_group * 8 + lane % 8]
      # scales[4*i] relative to scales + scale_offset, so i=1 advances four bytes.
      scale = _q6_scale_i32(halfs, base, scale_idx + 4 * i)
      dot = int8x4_dot(UOp.const(dtypes.int32, 0), qword, xword)
      xsum = int8x4_dot(UOp.const(dtypes.int32, 0), UOp.const(dtypes.uint32, 0x01010101), xword)
      # dot - 32*sum(x) is byte-for-byte llama's (qword - 0x20) then dp4a.
      term = (dot - UOp.const(dtypes.int32, 32) * xsum) * scale
      contrib = contrib + term.cast(dtypes.float32) * _q8_d(packed, block * 8 + q8_group)
    # vec_dot_q6_K_q8_1_impl_mmvq returns d * sumf.
    contrib = contrib * _f16_half(halfs[base + 104])

    acc = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(block_rel)[0] + contrib).end(block_rel))
    total = acc[0]
    for slot, off in enumerate((16, 8, 4, 2, 1), 90):
      total = total + _staged_shfl(total, off, lane, slot)
    smem = UOp.placeholder((WARPS,), dtypes.float32, 230, addrspace=AddrSpace.LOCAL)
    published = smem[warp].store(total, lane.eq(0))
    ready = UOp.barrier(UOp.group(published))
    merged = UOp.const(dtypes.float32, 0.0)
    for wi in range(WARPS):
      merged = merged + smem.after(ready)[wi]
    name = f"q6k_q8_llama_mmvq_{ROWS}_{K}" + ("" if interleaved else "_contig")
    return out[row].store(merged, lid.eq(0)).sink(arg=KernelInfo(name=name, opts_to_apply=()))
  return kernel


def _program(name, emitter, output_shape):
  return KernelProgram("research.q6k_q8_llama_mmvq", name, KernelProgramProvenance.RESEARCH_ONLY, emitter,
                       output_spec=OutputSpec(output_shape, dtypes.float32))


def _decode_q8(packed: np.ndarray) -> np.ndarray:
  packed = np.asarray(packed, dtype=np.uint32)
  payload = packed[:Q8_PACKS].view(np.uint8).reshape(Q8_GROUPS, 32).astype(np.int8).astype(np.float32)
  meta = packed[Q8_PACKS:]
  d = (meta & np.uint32(0xffff)).astype(np.uint16).view(np.float16).astype(np.float32)
  return (payload * d[:, None]).reshape(K)


def run(replays: int = 300, reps: int = 7, interleaved: bool = True) -> dict:
  dev = Device.DEFAULT
  if not str(dev).startswith("NV"):
    raise RuntimeError(f"native NV required, got {dev}")
  halfs_np = _make_q6k_halfs(ROWS, K, 20260805)
  x_np = np.random.default_rng(20260805).normal(0, .2, K).astype(np.float16)
  w = Tensor(halfs_np.copy(), dtype=dtypes.uint16, device=dev).contiguous().realize()
  x = Tensor(x_np.copy(), dtype=dtypes.float16, device=dev).contiguous().realize()

  provider_program = KernelProgram("research.q6k_q8_llama_mmvq", "q8_provider",
    KernelProgramProvenance.RESEARCH_ONLY, _emit_q8_provider(), output_spec=OutputSpec((Q8_PACKS + Q8_GROUPS,), dtypes.uint32))
  packed_t = execute_research_program(Tensor.empty((Q8_PACKS + Q8_GROUPS,), dtype=dtypes.uint32, device=dev),
    x, program=provider_program).realize()
  Device[dev].synchronize()

  candidate_program = _program(f"llama_mmvq_{'interleaved' if interleaved else 'contig'}",
    emit_q6k_q8_llama_mmvq(interleaved), (ROWS,))
  bs = q6k_spec_for_role(ROWS, K, role="attn_kv", parts=4, use_coop=False, reduction="external_sum")
  baseline_program = _program(bs.kernel_name, emit_q6k_gemv_kernel(bs), (ROWS, 4))

  @TinyJit
  def candidate(ww, xp):
    return execute_research_program(Tensor.empty((ROWS,), dtype=dtypes.float32, device=dev), ww, xp,
      program=candidate_program)

  @TinyJit
  def baseline(ww, xx):
    p = execute_research_program(Tensor.empty((ROWS, 4), dtype=dtypes.float32, device=dev), ww, xx,
      program=baseline_program)
    return p.sum(axis=1).contiguous()

  candidate(w, packed_t).realize()
  candidate_out = candidate(w, packed_t).realize()
  baseline_out = baseline(w, x).realize()
  Device[dev].synchronize()

  raw = halfs_np.view(np.uint8)
  weights = q6_k_reference(Tensor(raw.copy(), dtype=dtypes.uint8), ROWS * K).numpy().astype(np.float32).reshape(ROWS, K)
  q8_values = _decode_q8(packed_t.numpy().astype(np.uint32))
  ref = weights @ q8_values
  got = candidate_out.numpy().astype(np.float32)
  base = baseline_out.numpy().astype(np.float32)
  atol = max(.02, float(np.max(np.abs(ref))) * .015)
  cand_err = float(np.max(np.abs(got - ref)))
  baseline_err = float(np.max(np.abs(base - weights @ x_np.astype(np.float32))))

  for _ in range(1000):
    baseline(w, x).realize()
    candidate(w, packed_t).realize()
  Device[dev].synchronize()

  def timed(fn):
    vals = []
    for _ in range(reps):
      Device[dev].synchronize()
      st = time.perf_counter_ns()
      for _ in range(replays):
        fn().realize()
      Device[dev].synchronize()
      vals.append((time.perf_counter_ns() - st) / 1e3 / replays)
    return vals

  a, b, c = timed(lambda: baseline(w, x)), timed(lambda: candidate(w, packed_t)), timed(lambda: baseline(w, x))
  mid = (statistics.median(a) + statistics.median(c)) / 2
  return {
    "schema": "tinygrad.q6k_q8_llama_mmvq_microgate.v1",
    "device": str(dev),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "shape": {"rows": ROWS, "k": K, "warps": WARPS, "lanes": WARP, "interleaved_blocks": interleaved},
    "payload": {"q6_sha256": hashlib.sha256(raw).hexdigest(), "x_sha256": hashlib.sha256(x_np.tobytes()).hexdigest()},
    "correctness": {
      "atol": atol,
      "candidate_max_abs_q8_ref": cand_err,
      "baseline_max_abs_fp16_ref": baseline_err,
      "candidate_vs_baseline_rel_l2": float(
        np.linalg.norm((got - base).astype(np.float64)) / max(np.linalg.norm(base.astype(np.float64)), 1e-30)),
      "pass": bool(cand_err <= atol),
    },
    "timing": {"unit": "us_per_graph_replay", "replays": replays, "reps": reps,
      "control_a": a, "candidate_b": b, "control_c": c, "control_midpoint_median": mid,
      "candidate_median": statistics.median(b), "delta_us": statistics.median(b) - mid},
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--replays", type=int, default=300)
  ap.add_argument("--reps", type=int, default=7)
  ap.add_argument("--interleaved", type=int, default=1)
  ap.add_argument("--out")
  a = ap.parse_args()
  report = run(a.replays, a.reps, bool(a.interleaved))
  text = json.dumps(report, indent=2, sort_keys=True)
  if a.out:
    open(a.out, "w").write(text + "\n")
  print(text)
  return 0 if report["correctness"]["pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
