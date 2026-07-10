#!/usr/bin/env python3
"""Executable Q4_K x Q8_1 MMQ atom body for bounded 14B prefill gates.

This is the first runnable backend atom behind the hybrid MMQ boundary. It is
intentionally not wired into whole-prefill selection and it does not claim GPU
performance. The value of this slice is that the atom API is now executable,
spec-validated, lifecycle-attributed, and usable by the bounded harness before
the AMD kernel body replaces the reference execution core.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp

from extra.qk.amd_warp_reduce import warp_reduce_sum
from extra.qk.layout import Q4K_WORDS_PER_BLOCK, Q4_K_BLOCK_ELEMS, Q8_1_BLOCK_ELEMS
from extra.qk.mmq_atom_boundary import (
  PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_CLASSIFICATION,
  PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_ROUTE_ID,
)
from extra.qk.mmq_lifecycle import MMQLifecycleRow, zero_counters
from extra.qk.mmq_q4k_q8_reference import Q4KQ81MMQTileSpec, q4k_q8_1_mmq_tile_reference
from extra.qk.quant.q4_k_gemv_primitive import _q4k_group_params, _q4k_quant

BACKEND_ATOM_ID = "q4k_q8_1_mmq_reference_backed_atom_v0"
AMD_BACKEND_ATOM_ID = "q4k_q8_1_mmq_amd_uop_atom_v0"
AMD_WARP_BACKEND_ATOM_ID = "q4k_q8_1_mmq_amd_warp_atom_v0"


@dataclass(frozen=True)
class Q4KQ8MMQAtomResult:
  output: np.ndarray
  lifecycle: MMQLifecycleRow
  backend_atom_id: str = BACKEND_ATOM_ID

  def to_json(self) -> dict[str, Any]:
    return {
      "backend_atom_id": self.backend_atom_id,
      "route_id": PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_ROUTE_ID,
      "classification": PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_CLASSIFICATION,
      "output_shape": list(self.output.shape),
      "lifecycle": self.lifecycle.to_json(),
    }


def _tile_id(spec: Q4KQ81MMQTileSpec) -> str:
  return f"m{spec.m0}_n{spec.n0}_k{spec.k0}_kg{spec.effective_k_groups}"


def _lifecycle_for_spec(spec: Q4KQ81MMQTileSpec) -> MMQLifecycleRow:
  counters = zero_counters(
    activation_quant_epochs=1,
    activation_q8_1_global_writes=spec.tile_m * spec.effective_k_groups,
    activation_q8_1_reads=spec.tile_m * spec.effective_k_groups,
    packed_weight_global_loads=spec.tile_n * (spec.effective_k_groups * 32 // 256),
    scale_min_metadata_loads=spec.tile_n * (spec.effective_k_groups * 32 // 256),
    dot_accumulation_epochs=1,
    dot_ops_or_packed_dot_insts=spec.tile_m * spec.tile_n * spec.effective_k_groups,
    intermediate_global_writes=0,
    output_store_epochs=1,
    output_stores=spec.tile_m * spec.tile_n,
  )
  return MMQLifecycleRow(role=spec.role, tile_id=_tile_id(spec), counters=counters)


def _as_u32_words(q4k_bytes: np.ndarray) -> np.ndarray:
  raw = np.ascontiguousarray(np.asarray(q4k_bytes, dtype=np.uint8).reshape(-1))
  if raw.size % 4:
    raise ValueError(f"Q4_K byte input must be uint32 aligned, got {raw.size} bytes")
  return raw.view("<u4").astype(np.uint32, copy=False)


def _validate_amd_spec(spec: Q4KQ81MMQTileSpec) -> None:
  spec.validate()
  if spec.k0 % Q4_K_BLOCK_ELEMS:
    raise ValueError(f"AMD MMQ atom requires k0 to be Q4_K block aligned, got {spec.k0}")
  if spec.effective_k_groups % (Q4_K_BLOCK_ELEMS // Q8_1_BLOCK_ELEMS):
    raise ValueError(f"AMD MMQ atom requires k_groups to cover whole Q4_K blocks, got {spec.effective_k_groups}")


def _q4k_q8_1_tile_kernel(spec: Q4KQ81MMQTileSpec):
  _validate_amd_spec(spec)
  tile_m, tile_n = spec.tile_m, spec.tile_n
  k_blocks = spec.effective_k_groups // (Q4_K_BLOCK_ELEMS // Q8_1_BLOCK_ELEMS)
  full_k_blocks = spec.k // Q4_K_BLOCK_ELEMS
  full_q8_groups = spec.k // Q8_1_BLOCK_ELEMS
  first_blk = spec.k0 // Q4_K_BLOCK_ELEMS
  name = f"q4k_q8_1_mmq_atom_{spec.role}_{tile_m}_{tile_n}_{spec.k0}_{spec.effective_k_groups}"

  def kernel(out: UOp, words: UOp, xq: UOp, xscales: UOp) -> UOp:
    bb = UOp.range(tile_m, 0)
    row_i = UOp.range(tile_n, 1)
    blk_i = UOp.range(k_blocks, 2, axis_type=AxisType.REDUCE)
    pos = UOp.range(32, 3, axis_type=AxisType.REDUCE)
    row = spec.n0 + row_i
    tok = spec.m0 + bb
    blk = first_blk + blk_i
    base = (row * full_k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = UOp.const(dtypes.float32, 0.0)
    for grp in range(8):
      d, dmin, sc, mn = _q4k_group_params(words, base, grp)
      q = _q4k_quant(words, base, grp, pos).cast(dtypes.float32)
      w = d * sc.cast(dtypes.float32) * q - dmin * mn.cast(dtypes.float32)
      q8_idx = tok * spec.k + blk * Q4_K_BLOCK_ELEMS + grp * Q8_1_BLOCK_ELEMS + pos
      scale_idx = tok * full_q8_groups + blk * (Q4_K_BLOCK_ELEMS // Q8_1_BLOCK_ELEMS) + grp
      x = xq[q8_idx].cast(dtypes.float32) * xscales[scale_idx].cast(dtypes.float32)
      contrib = contrib + w * x
    return out[bb, row_i].store(contrib.reduce(blk_i, pos, arg=Ops.ADD)).end(bb, row_i).sink(
      arg=KernelInfo(name=name, opts_to_apply=()))

  return kernel


def _q4k_q8_1_tile_warp_kernel(spec: Q4KQ81MMQTileSpec):
  _validate_amd_spec(spec)
  tile_m, tile_n = spec.tile_m, spec.tile_n
  k_blocks = spec.effective_k_groups // (Q4_K_BLOCK_ELEMS // Q8_1_BLOCK_ELEMS)
  full_k_blocks = spec.k // Q4_K_BLOCK_ELEMS
  full_q8_groups = spec.k // Q8_1_BLOCK_ELEMS
  first_blk = spec.k0 // Q4_K_BLOCK_ELEMS
  name = f"q4k_q8_1_mmq_warp_atom_{spec.role}_{tile_m}_{tile_n}_{spec.k0}_{spec.effective_k_groups}"

  def kernel(out: UOp, words: UOp, xq: UOp, xscales: UOp) -> UOp:
    row_i = UOp.special(tile_n, "gidx0")
    bb = UOp.special(tile_m, "gidx1")
    lane = UOp.special(32, "lidx0")
    blk_i = UOp.range(k_blocks, 0, axis_type=AxisType.REDUCE)
    row = spec.n0 + row_i
    tok = spec.m0 + bb
    blk = first_blk + blk_i
    base = (row * full_k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = UOp.const(dtypes.float32, 0.0)
    for grp in range(8):
      d, dmin, sc, mn = _q4k_group_params(words, base, grp)
      q = _q4k_quant(words, base, grp, lane).cast(dtypes.float32)
      w = d * sc.cast(dtypes.float32) * q - dmin * mn.cast(dtypes.float32)
      q8_idx = tok * spec.k + blk * Q4_K_BLOCK_ELEMS + grp * Q8_1_BLOCK_ELEMS + lane
      scale_idx = tok * full_q8_groups + blk * (Q4_K_BLOCK_ELEMS // Q8_1_BLOCK_ELEMS) + grp
      x = xq[q8_idx].cast(dtypes.float32) * xscales[scale_idx].cast(dtypes.float32)
      contrib = contrib + w * x
    acc = UOp.placeholder((1,), dtypes.float32, 70, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(blk_i)[0] + contrib).end(blk_i))
    total = warp_reduce_sum(acc[0], lane, 32)
    return out[bb, row_i].store(total).sink(arg=KernelInfo(name=name, opts_to_apply=()))

  return kernel


def amd_atom_source_hash(spec: Q4KQ81MMQTileSpec) -> str:
  # Stable evidence for the generated UOp atom identity. This is not a binary hash.
  payload = repr(_q4k_q8_1_tile_kernel(spec)(UOp.placeholder((spec.tile_m, spec.tile_n), dtypes.float32, 0),
                                             UOp.placeholder((spec.n * (spec.k // Q4_K_BLOCK_ELEMS) * Q4K_WORDS_PER_BLOCK,), dtypes.uint32, 1),
                                             UOp.placeholder((spec.m * spec.k,), dtypes.int8, 2),
                                             UOp.placeholder((spec.m * (spec.k // Q8_1_BLOCK_ELEMS),), dtypes.float32, 3)))
  return hashlib.sha256(payload.encode()).hexdigest()[:16]


def amd_warp_atom_source_hash(spec: Q4KQ81MMQTileSpec) -> str:
  payload = repr(_q4k_q8_1_tile_warp_kernel(spec)(UOp.placeholder((spec.tile_m, spec.tile_n), dtypes.float32, 0),
                                                  UOp.placeholder((spec.n * (spec.k // Q4_K_BLOCK_ELEMS) * Q4K_WORDS_PER_BLOCK,), dtypes.uint32, 1),
                                                  UOp.placeholder((spec.m * spec.k,), dtypes.int8, 2),
                                                  UOp.placeholder((spec.m * (spec.k // Q8_1_BLOCK_ELEMS),), dtypes.float32, 3)))
  return hashlib.sha256(payload.encode()).hexdigest()[:16]


def run_q4k_q8_1_mmq_tile_amd(q4k_bytes: np.ndarray, xq: np.ndarray, xscales: np.ndarray,
                              spec: Q4KQ81MMQTileSpec, *, device: str = "AMD") -> Q4KQ8MMQAtomResult:
  _validate_amd_spec(spec)
  words_np = _as_u32_words(q4k_bytes)
  words = Tensor(words_np, dtype=dtypes.uint32, device=device).realize()
  xq_t = Tensor(np.ascontiguousarray(np.asarray(xq, dtype=np.int8).reshape(-1)), dtype=dtypes.int8, device=device).realize()
  xs_t = Tensor(np.ascontiguousarray(np.asarray(xscales, dtype=np.float32).reshape(-1)), dtype=dtypes.float32, device=device).realize()
  out = Tensor.empty(spec.tile_m, spec.tile_n, dtype=dtypes.float32, device=device).custom_kernel(
    words, xq_t, xs_t, fxn=_q4k_q8_1_tile_kernel(spec))[0].realize()
  return Q4KQ8MMQAtomResult(output=out.numpy().astype(np.float32), lifecycle=_lifecycle_for_spec(spec),
                            backend_atom_id=AMD_BACKEND_ATOM_ID)


def run_q4k_q8_1_mmq_tile_amd_warp(q4k_bytes: np.ndarray, xq: np.ndarray, xscales: np.ndarray,
                                   spec: Q4KQ81MMQTileSpec, *, device: str = "AMD") -> Q4KQ8MMQAtomResult:
  _validate_amd_spec(spec)
  words_np = _as_u32_words(q4k_bytes)
  words = Tensor(words_np, dtype=dtypes.uint32, device=device).realize()
  xq_t = Tensor(np.ascontiguousarray(np.asarray(xq, dtype=np.int8).reshape(-1)), dtype=dtypes.int8, device=device).realize()
  xs_t = Tensor(np.ascontiguousarray(np.asarray(xscales, dtype=np.float32).reshape(-1)), dtype=dtypes.float32, device=device).realize()
  out = Tensor.empty(spec.tile_m, spec.tile_n, dtype=dtypes.float32, device=device).custom_kernel(
    words, xq_t, xs_t, fxn=_q4k_q8_1_tile_warp_kernel(spec))[0].realize()
  return Q4KQ8MMQAtomResult(output=out.numpy().astype(np.float32), lifecycle=_lifecycle_for_spec(spec),
                            backend_atom_id=AMD_WARP_BACKEND_ATOM_ID)


def run_q4k_q8_1_mmq_tile_with_lifecycle(q4k_bytes: np.ndarray, xq: np.ndarray, xscales: np.ndarray,
                                         spec: Q4KQ81MMQTileSpec) -> Q4KQ8MMQAtomResult:
  spec.validate()
  output = q4k_q8_1_mmq_tile_reference(q4k_bytes, xq, xscales, spec)
  return Q4KQ8MMQAtomResult(output=np.asarray(output, dtype=np.float32), lifecycle=_lifecycle_for_spec(spec))


def run_q4k_q8_1_mmq_tile(q4k_bytes: np.ndarray, xq: np.ndarray, xscales: np.ndarray,
                          spec: Q4KQ81MMQTileSpec) -> np.ndarray:
  return run_q4k_q8_1_mmq_tile_with_lifecycle(q4k_bytes, xq, xscales, spec).output
