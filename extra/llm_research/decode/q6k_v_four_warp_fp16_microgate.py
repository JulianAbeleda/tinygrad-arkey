#!/usr/bin/env python3
"""Research-only microgate: Q6_K attention-V 1024x4096 four-warp fp16 direct.

The control is the production ``q6k_gen_partial_1024_4096_4`` route (parts=4,
external_sum, one warp per row-part) with the parts reduce folded into the
decode kv-store.  The candidate mirrors the already-landed Q4/Q6 four-warp fp16
geometry onto the V shape: 128 threads per row across four warps, fp16
activation consumed directly, no Q8 provider, no parts buffer, one pass to a
contiguous fp32[1024] output.  Each lane owns two of the row's 16 K-blocks
across a (sub, pos) partition, and the 32-lane partials reduce with the staged
XOR ladder plus a shared-memory warp combine.

This is a measurement gate, not a route selector: the candidate emitter lives
only in this file and is never reachable from production decode.
"""
from __future__ import annotations

import argparse, hashlib, json, statistics, subprocess, time
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.codegen.late.warp_reduce import _staged_shfl
from tinygrad.dtype import AddrSpace
from tinygrad.llm.decode_kernels import (
  Q6_K_BLOCK_ELEMS, Q6K_HALFWORDS_PER_BLOCK, _q6k_block_dot, emit_q6k_gemv_kernel, q6k_spec_for_role)
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, execute_research_program
from tinygrad.uop.ops import AxisType, KernelInfo, UOp
from extra.llm_research.decode.route_class_numerics import _make_q6k_halfs
from extra.llm_research.layout import q6_k_reference

ROWS, K = 1024, 4096
WARP, WARPS_PER_ROW, POS = 32, 4, 16
K_BLOCKS = K // Q6_K_BLOCK_ELEMS          # 16
BLOCKS_PER_WARP = K_BLOCKS // WARPS_PER_ROW  # 4
BLOCKS_PER_SUB = BLOCKS_PER_WARP // 2    # 2


def emit_q6k_v_four_warp_fp16_direct() -> callable:
  """Q6_K attention-V four-warp fp16-direct consumer (128 threads/row, no Q8)."""
  def kernel(out:UOp, halfs:UOp, x:UOp) -> UOp:
    row = UOp.special(ROWS, "gidx0")
    lid = UOp.special(WARP * WARPS_PER_ROW, "lidx0")
    warp, lane = lid // WARP, lid % WARP
    sub, pos = lane // POS, lane % POS

    acc = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    blk = UOp.range(BLOCKS_PER_SUB, 0, axis_type=AxisType.REDUCE)
    block = warp * BLOCKS_PER_WARP + sub * BLOCKS_PER_SUB + blk
    base = (row * K_BLOCKS + block) * Q6K_HALFWORDS_PER_BLOCK
    contrib = _q6k_block_dot(halfs, x, base, block, pos)
    acc = acc.after(acc[0].store(acc.after(blk)[0] + contrib).end(blk))
    total = acc[0]

    for slot, offset in enumerate((16, 8, 4, 2, 1), 90):
      total = total + _staged_shfl(total, offset, lane, slot)
    smem = UOp.placeholder((WARPS_PER_ROW,), dtypes.float32, 40, addrspace=AddrSpace.LOCAL)
    published = smem[warp].store(total, lane.eq(0))
    ready = UOp.barrier(UOp.group(published))
    merged = UOp.const(dtypes.float32, 0.0)
    for wi in range(WARPS_PER_ROW):
      merged = merged + smem.after(ready)[wi]
    return out[row].store(merged, lid.eq(0)).sink(
      arg=KernelInfo(name="q6k_v_four_warp_fp16_direct_1024_4096", opts_to_apply=()))
  return kernel


def _program(name:str, fn:callable) -> KernelProgram:
  return KernelProgram("research.q6k_v_four_warp_fp16", name,
    KernelProgramProvenance.RESEARCH_ONLY, fn)


def run(replays:int, reps:int) -> dict:
  dev = Device.DEFAULT
  if not str(dev).startswith("NV"):
    raise RuntimeError(f"native NV required, got {dev}")
  halfs_np = _make_q6k_halfs(ROWS, K, 20260805)
  x_np = np.random.default_rng(20260805).normal(0, 0.2, K).astype(np.float16)
  halfs = Tensor(halfs_np.copy(), dtype=dtypes.uint16, device=dev).contiguous().realize()
  x = Tensor(x_np.copy(), dtype=dtypes.float16, device=dev).contiguous().realize()

  partial_spec = q6k_spec_for_role(ROWS, K, role="attn_kv", parts=4, use_coop=False,
    reduction="external_sum")
  partial_prog = _program(partial_spec.kernel_name, emit_q6k_gemv_kernel(partial_spec))
  candidate = _program("q6k_v_four_warp_fp16_direct_1024_4096",
    emit_q6k_v_four_warp_fp16_direct())

  @TinyJit
  def partial_only(w:Tensor, a:Tensor):
    return execute_research_program(Tensor.empty((ROWS, 4), dtype=dtypes.float32, device=dev),
      w, a, program=partial_prog)

  @TinyJit
  def control_fn(w:Tensor, a:Tensor):
    parts = execute_research_program(Tensor.empty((ROWS, 4), dtype=dtypes.float32, device=dev),
      w, a, program=partial_prog)
    return parts.sum(axis=1).contiguous()

  @TinyJit
  def candidate_fn(w:Tensor, a:Tensor):
    return execute_research_program(Tensor.empty((ROWS,), dtype=dtypes.float32, device=dev),
      w, a, program=candidate)

  control_fn(halfs, x).realize()
  got_control = control_fn(halfs, x).realize().numpy().astype(np.float32)
  candidate_fn(halfs, x).realize()
  got_candidate = candidate_fn(halfs, x).realize().numpy().astype(np.float32)
  Device[dev].synchronize()

  raw = halfs_np.view(np.uint8)
  weights = q6_k_reference(Tensor(raw.copy(), dtype=dtypes.uint8), ROWS * K).numpy().astype(np.float32).reshape(ROWS, K)
  ref = weights @ x_np.astype(np.float32)
  err = np.abs(got_candidate - ref)
  base_err = np.abs(got_control - ref)
  cross = np.abs(got_candidate - got_control)
  atol = max(2e-2, float(np.max(np.abs(ref))) * 2e-4)

  def timed(fn) -> list[float]:
    samples = []
    for _ in range(reps):
      Device[dev].synchronize()
      st = time.perf_counter_ns()
      for _ in range(replays):
        fn(halfs, x).realize()
      Device[dev].synchronize()
      samples.append((time.perf_counter_ns() - st) / 1e3 / replays)
    return samples

  a, b, c = timed(control_fn), timed(candidate_fn), timed(control_fn)
  control_mid = (statistics.median(a) + statistics.median(c)) / 2
  # The production V path pays the partial kernel alone (the parts reduce is
  # folded into the kv-store), so the honest device comparison is candidate vs
  # partial_only. The full-arithmetic control is reported for reference.
  d0 = timed(partial_only)
  return {
    "schema": "tinygrad.q6k_v_four_warp_fp16_microgate.v1",
    "device": str(dev),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "payload": {"q6_sha256": hashlib.sha256(raw).hexdigest(),
      "x_sha256": hashlib.sha256(x_np.tobytes()).hexdigest()},
    "candidate": {"id": "q6k_v_four_warp_fp16_direct_1024_4096",
      "threads_per_row": WARP * WARPS_PER_ROW, "warps_per_row": WARPS_PER_ROW,
      "blocks_per_warp": BLOCKS_PER_WARP, "blocks_per_sub": BLOCKS_PER_SUB,
      "provider_nodes": 0, "parts": 1},
    "control": {"id": partial_spec.kernel_name, "spec": partial_spec.to_json()},
    "correctness": {"atol": atol,
      "candidate_max_abs_ref": float(err.max()), "control_max_abs_ref": float(base_err.max()),
      "candidate_vs_control_max_abs": float(cross.max()), "pass": bool(err.max() <= atol)},
    "timing": {"unit": "us_per_graph_replay", "replays": replays, "reps": reps,
      "control_a": a, "candidate_b": b, "control_c": c,
      "control_midpoint_median": control_mid, "candidate_median": statistics.median(b),
      "delta": statistics.median(b) - control_mid,
      "partial_kernel_only_median": statistics.median(d0),
      "candidate_vs_partial_kernel_only_delta": statistics.median(b) - statistics.median(d0)},
  }


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--replays", type=int, default=300)
  ap.add_argument("--reps", type=int, default=7)
  ap.add_argument("--out")
  a = ap.parse_args()
  result = run(a.replays, a.reps)
  text = json.dumps(result, indent=2, sort_keys=True)
  if a.out:
    with open(a.out, "w") as f:
      f.write(text + "\n")
  print(text)
  return 0 if result["correctness"]["pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
