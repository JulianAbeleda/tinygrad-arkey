#!/usr/bin/env python3
"""Research-only microgate: Q6_K FFN-down four-warp fp16 direct vs installed coop.

The control is the production ``q6k_gen_coop_4096_12288_inkernel_epi_ffnresadd``
route (row_tile=2, 16 pos lanes, 32 threads per two rows).  The candidate mirrors
the already-landed Q4 four-warp fp16 geometry spelling onto Q6: 128 threads per
row across four warps, fp16 activation consumed directly, no Q8 provider node,
and the same Q6 dequant/dequant-fma arithmetic as the control.  Each lane owns
six of the row's 48 K-blocks across a (pos, sub-group) partition, and the
32-lane partials reduce with the standard staged XOR ladder.

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

ROWS, K = 4096, 12288
WARP, WARPS_PER_ROW, POS = 32, 4, 16
K_BLOCKS = K // Q6_K_BLOCK_ELEMS          # 48
BLOCKS_PER_WARP = K_BLOCKS // WARPS_PER_ROW  # 12
BLOCKS_PER_SUB = BLOCKS_PER_WARP // 2    # 6


def emit_q6k_four_warp_fp16_direct() -> callable:
  """Q6_K FFN-down four-warp fp16-direct consumer (128 threads/row, no Q8)."""
  def kernel(out:UOp, halfs:UOp, x:UOp, h:UOp) -> UOp:
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
    result = merged + h[row].cast(dtypes.float32)
    return out[row].store(result, lid.eq(0)).sink(
      arg=KernelInfo(name="q6k_four_warp_fp16_direct_4096_12288_epi_ffnresadd", opts_to_apply=()))
  return kernel


def _program(name:str, fn:callable) -> KernelProgram:
  return KernelProgram("research.q6k_ffn_down_four_warp_fp16", name,
    KernelProgramProvenance.RESEARCH_ONLY, fn)


def run(replays:int, reps:int) -> dict:
  dev = Device.DEFAULT
  if not str(dev).startswith("NV"):
    raise RuntimeError(f"native NV required, got {dev}")
  halfs_np = _make_q6k_halfs(ROWS, K, 20260805)
  x_np = np.random.default_rng(20260805).normal(0, 0.2, K).astype(np.float16)
  h_np = np.random.default_rng(20260806).normal(0, 0.05, ROWS).astype(np.float32)
  halfs = Tensor(halfs_np.copy(), dtype=dtypes.uint16, device=dev).contiguous().realize()
  x = Tensor(x_np.copy(), dtype=dtypes.float16, device=dev).contiguous().realize()
  h = Tensor(h_np.copy(), dtype=dtypes.float32, device=dev).contiguous().realize()

  control_spec = q6k_spec_for_role(ROWS, K, role="ffn_down", parts=1, row_tile=2,
    use_coop=True, reduction="in_kernel", epilogue="ffn_down_resadd")
  control = _program(control_spec.kernel_name, emit_q6k_gemv_kernel(control_spec))
  candidate = _program("q6k_four_warp_fp16_direct_4096_12288_epi_ffnresadd",
    emit_q6k_four_warp_fp16_direct())

  @TinyJit
  def control_fn(w:Tensor, a:Tensor, res:Tensor):
    return execute_research_program(Tensor.empty((ROWS,), dtype=dtypes.float32, device=dev),
      w, a, res, program=control)

  @TinyJit
  def candidate_fn(w:Tensor, a:Tensor, res:Tensor):
    return execute_research_program(Tensor.empty((ROWS,), dtype=dtypes.float32, device=dev),
      w, a, res, program=candidate)

  control_fn(halfs, x, h).realize()
  got_control = control_fn(halfs, x, h).realize().numpy().astype(np.float32)
  candidate_fn(halfs, x, h).realize()
  got_candidate = candidate_fn(halfs, x, h).realize().numpy().astype(np.float32)
  Device[dev].synchronize()

  raw = halfs_np.view(np.uint8)
  weights = q6_k_reference(Tensor(raw.copy(), dtype=dtypes.uint8), ROWS * K).numpy().astype(np.float32).reshape(ROWS, K)
  ref = weights @ x_np.astype(np.float32) + h_np.astype(np.float32)
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
        fn(halfs, x, h).realize()
      Device[dev].synchronize()
      samples.append((time.perf_counter_ns() - st) / 1e3 / replays)
    return samples

  a, b, c = timed(control_fn), timed(candidate_fn), timed(control_fn)
  control_mid = (statistics.median(a) + statistics.median(c)) / 2
  return {
    "schema": "tinygrad.q6k_ffn_down_four_warp_fp16_microgate.v1",
    "device": str(dev),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "payload": {"q6_sha256": hashlib.sha256(raw).hexdigest(),
      "x_sha256": hashlib.sha256(x_np.tobytes()).hexdigest()},
    "candidate": {"id": "q6k_four_warp_fp16_direct_4096_12288_epi_ffnresadd",
      "threads_per_row": WARP * WARPS_PER_ROW, "warps_per_row": WARPS_PER_ROW,
      "provider_nodes": 0, "epilogue": "ffn_down_resadd"},
    "control": {"id": control_spec.kernel_name, "spec": control_spec.to_json()},
    "correctness": {"atol": atol,
      "candidate_max_abs_ref": float(err.max()), "control_max_abs_ref": float(base_err.max()),
      "candidate_vs_control_max_abs": float(cross.max()), "pass": bool(err.max() <= atol)},
    "timing": {"unit": "us_per_graph_replay", "replays": replays, "reps": reps,
      "control_a": a, "candidate_b": b, "control_c": c,
      "control_midpoint_median": control_mid, "candidate_median": statistics.median(b),
      "delta": statistics.median(b) - control_mid},
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
