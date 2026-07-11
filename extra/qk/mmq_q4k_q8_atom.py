#!/usr/bin/env python3
"""Executable Q4_K x Q8_1 MMQ atom body for bounded 14B prefill gates.

This is the first runnable backend atom behind the hybrid MMQ boundary. It is
intentionally not wired into whole-prefill selection and it does not claim GPU
performance. The value of this slice is that the atom API is now executable,
spec-validated, lifecycle-attributed, and usable by the bounded harness before
the AMD kernel body replaces the reference execution core.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp

from extra.qk.amd_warp_reduce import warp_reduce_sum
from extra.qk.layout import Q4K_WORDS_PER_BLOCK, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, Q8_1_BLOCK_ELEMS
from extra.qk.mmq_atom_boundary import (
  PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_CLASSIFICATION,
  PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_ROUTE_ID,
)
from extra.qk.mmq_lifecycle import MMQLifecycleRow, zero_counters
from extra.qk.mmq_q4k_q8_reference import (
  Q81MMQDS4Activation, Q4KQ81MMQTileSpec, Q8_1_MMQ_DS4_LAYOUT, q4k_q8_1_mmq_tile_reference,
  q8_1_mmq_ds4_from_row_major_reference,
)
from extra.qk.quant.q4_k_gemv_primitive import _q4k_group_params, _q4k_group_qpack_lane4, _q4k_quant

BACKEND_ATOM_ID = "q4k_q8_1_mmq_reference_backed_atom_v0"
AMD_BACKEND_ATOM_ID = "q4k_q8_1_mmq_amd_uop_atom_v0"
AMD_WARP_BACKEND_ATOM_ID = "q4k_q8_1_mmq_amd_warp_atom_v0"
AMD_WARP_BATCHED_BACKEND_ATOM_ID = "q4k_q8_1_mmq_amd_warp_batched_atom_v0"
AMD_DOT4_BATCHED_BACKEND_ATOM_ID = "q4k_q8_1_mmq_amd_dot4_batched_atom_v0"
AMD_DOT4X4_BATCHED_BACKEND_ATOM_ID = "q4k_q8_1_mmq_amd_dot4x4_batched_atom_v0"
AMD_STAGED_DS4_BACKEND_ATOM_ID = "q4k_q8_1_mmq_amd_staged_ds4_atom_v0"
AMD_DS4_WARP_BACKEND_ATOM_ID = "q4k_q8_1_mmq_amd_ds4_warp_atom_v0"
AMD_DS4_DOT4X4_BACKEND_ATOM_ID = "q4k_q8_1_mmq_amd_ds4_dot4x4_atom_v0"
AMD_DS4_LDS_SKELETON_BACKEND_ATOM_ID = "q4k_q8_1_mmq_amd_ds4_lds_skeleton_atom_v0"
AMD_DS4_COOP_TILE_BACKEND_ATOM_ID = "q4k_q8_1_mmq_amd_ds4_coop_tile_atom_v0"
AMD_DS4_COOP_TILE_BLOCKER = (
  "16x16x256 DS4 coop numeric atom emits and passes bounded correctness only after omitting store_owner metadata "
  "from the Tensor custom_kernel graph; attaching tuple owner metadata still fails in "
  "tinygrad/codegen/late/linearizer.py while sorting the tagged store graph. R4 owner proof remains separate, "
  "and no production route promotion is claimed."
)
DS4_ACTIVATION_LAYOUT = Q8_1_MMQ_DS4_LAYOUT


@dataclass(frozen=True)
class Q4KQ8MMQAtomResult:
  output: np.ndarray
  lifecycle: MMQLifecycleRow
  backend_atom_id: str = BACKEND_ATOM_ID
  lifecycle_detail: dict[str, Any] = field(default_factory=dict)

  def to_json(self) -> dict[str, Any]:
    row = {
      "backend_atom_id": self.backend_atom_id,
      "route_id": PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_ROUTE_ID,
      "classification": PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_CLASSIFICATION,
      "output_shape": list(self.output.shape),
      "lifecycle": self.lifecycle.to_json(),
    }
    if self.lifecycle_detail:
      row["lifecycle_detail"] = self.lifecycle_detail
    return row


def q8_1_mmq_ds4_from_row_major(xq: np.ndarray, xscales: np.ndarray) -> Q81MMQDS4Activation:
  return q8_1_mmq_ds4_from_row_major_reference(xq, xscales)


def _q4k_f16_pair(block: np.ndarray) -> tuple[np.float32, np.float32]:
  vals = block[:4].view("<f2").astype(np.float32)
  return np.float32(vals[0]), np.float32(vals[1])


def _q4k_scale_min(block: np.ndarray, grp: int) -> tuple[int, int]:
  qs = block[4:16]
  if grp < 4:
    return int(qs[grp] & 63), int(qs[4 + grp] & 63)
  high = int(qs[8 + grp - 4])
  sc = (high & 0x0f) | ((int(qs[grp - 4]) >> 6) << 4)
  mn = (high >> 4) | ((int(qs[4 + grp - 4]) >> 6) << 4)
  return sc, mn


def _q4k_unsigned_nibbles(block: np.ndarray, grp: int) -> np.ndarray:
  qbytes = block[16:].reshape(4, 32)
  packed = qbytes[grp // 2]
  return ((packed >> ((grp % 2) * 4)) & 0x0f).astype(np.float32)


def _validate_staged_ds4_inputs(q4k_bytes: np.ndarray, ds4: Q81MMQDS4Activation,
                                spec: Q4KQ81MMQTileSpec) -> np.ndarray:
  spec.validate()
  ds4.spec.validate()
  if spec.k != ds4.spec.k or spec.m != ds4.spec.m:
    raise ValueError(f"DS4 activation shape {(ds4.spec.m, ds4.spec.k)} does not match spec {(spec.m, spec.k)}")
  if spec.k0 % 128 or spec.effective_k_groups % 4:
    raise ValueError("staged DS4 atom requires k0 and k_groups to be aligned to 128-value DS4 blocks")
  raw = np.asarray(q4k_bytes, dtype=np.uint8)
  expected = (spec.n, spec.k // Q4_K_BLOCK_ELEMS, Q4_K_BLOCK_BYTES)
  if raw.reshape(-1).size != spec.n * (spec.k // Q4_K_BLOCK_ELEMS) * Q4_K_BLOCK_BYTES:
    raise ValueError(f"expected Q4_K bytes for shape {expected}, got {raw.shape}")
  return np.ascontiguousarray(raw.reshape(expected))


def _staged_ds4_lifecycle_for_spec(spec: Q4KQ81MMQTileSpec) -> tuple[MMQLifecycleRow, dict[str, Any]]:
  k_blocks = spec.effective_k_groups // 8
  ds4_blocks = spec.effective_k_groups // 4
  staged_activation_loads = spec.tile_m * ds4_blocks
  staged_weight_loads = spec.tile_n * k_blocks
  metadata_loads = spec.tile_n * k_blocks * 8
  counters = zero_counters(
    activation_quant_epochs=1,
    activation_q8_1_reads=staged_activation_loads,
    packed_weight_global_loads=staged_weight_loads,
    scale_min_metadata_loads=metadata_loads,
    dot_accumulation_epochs=spec.tile_m * spec.tile_n * spec.effective_k_groups,
    dot_ops_or_packed_dot_insts=spec.tile_m * spec.tile_n * spec.effective_k_groups * 8,
    barriers=2 * max(ds4_blocks, 1),
    intermediate_global_writes=0,
    output_store_epochs=1,
    output_stores=spec.tile_m * spec.tile_n,
    duplicate_quant_work=0,
    duplicate_dequant_or_scale_work=0,
    split_k_reductions=0,
  )
  detail = {
    "backend_stage": "reference_backed_staged_ds4_probe",
    "promotion_claim": False,
    "global_activation_ds4_loads": staged_activation_loads,
    "global_q4k_tile_loads": staged_weight_loads,
    "staged_activation_tile_loads": staged_activation_loads,
    "staged_q4k_tile_loads": staged_weight_loads,
    "barrier_epochs": counters["barriers"],
    "dot_epochs": counters["dot_accumulation_epochs"],
    "output_store_epochs": counters["output_store_epochs"],
    "uses_precomputed_activation_sums": True,
  }
  return MMQLifecycleRow(role=spec.role, tile_id=f"staged_ds4_{_tile_id(spec)}", counters=counters), detail


def q4k_q8_1_mmq_staged_ds4_tile_reference(q4k_bytes: np.ndarray, ds4: Q81MMQDS4Activation,
                                           spec: Q4KQ81MMQTileSpec) -> np.ndarray:
  q4 = _validate_staged_ds4_inputs(q4k_bytes, ds4, spec)
  out = np.zeros((spec.tile_m, spec.tile_n), dtype=np.float32)
  first_block = spec.k0 // Q4_K_BLOCK_ELEMS
  k_blocks = spec.effective_k_groups // 8
  for mi, mrow in enumerate(range(spec.m0, spec.m0 + spec.tile_m)):
    for ni, nrow in enumerate(range(spec.n0, spec.n0 + spec.tile_n)):
      acc = np.float32(0.0)
      for blk_i in range(k_blocks):
        q4_blk = q4[nrow, first_block + blk_i]
        d, dmin = _q4k_f16_pair(q4_blk)
        for grp in range(8):
          ds4_block = (spec.k0 // 128) + blk_i * 2 + grp // 4
          ds4_group = grp % 4
          xvals = ds4.values[ds4_block, mrow].reshape(4, Q8_1_BLOCK_ELEMS)[ds4_group].astype(np.float32)
          xscale = np.float32(ds4.scales[ds4_block, mrow, ds4_group])
          xsum = np.float32(ds4.sums[ds4_block, mrow, ds4_group])
          sc, mn = _q4k_scale_min(q4_blk, grp)
          q = _q4k_unsigned_nibbles(q4_blk, grp)
          dot_term = np.float32(np.dot(q, xvals))
          acc += xscale * (d * np.float32(sc) * dot_term) - (dmin * np.float32(mn) * xsum)
      out[mi, ni] = acc
  return out


def staged_ds4_atom_source_hash(spec: Q4KQ81MMQTileSpec) -> str:
  payload = "|".join((
    AMD_STAGED_DS4_BACKEND_ATOM_ID,
    DS4_ACTIVATION_LAYOUT,
    str(spec.to_json()),
    "q8_sums=precomputed_dequantized_group_sums",
    "reference_backed_no_gpu_kernel",
  ))
  return hashlib.sha256(payload.encode()).hexdigest()[:16]


def run_q4k_q8_1_mmq_staged_ds4_atom(q4k_bytes: np.ndarray, ds4: Q81MMQDS4Activation,
                                     spec: Q4KQ81MMQTileSpec) -> Q4KQ8MMQAtomResult:
  output = q4k_q8_1_mmq_staged_ds4_tile_reference(q4k_bytes, ds4, spec)
  lifecycle, detail = _staged_ds4_lifecycle_for_spec(spec)
  return Q4KQ8MMQAtomResult(output=np.asarray(output, dtype=np.float32), lifecycle=lifecycle,
                            backend_atom_id=AMD_STAGED_DS4_BACKEND_ATOM_ID, lifecycle_detail=detail)


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


def _q4k_q8_1_bounded_warp_kernel(m:int, n:int, k:int, role:str):
  if k % Q4_K_BLOCK_ELEMS:
    raise ValueError(f"AMD batched MMQ atom requires k to be Q4_K block aligned, got {k}")
  k_blocks = k // Q4_K_BLOCK_ELEMS
  full_q8_groups = k // Q8_1_BLOCK_ELEMS
  name = f"q4k_q8_1_mmq_warp_batched_atom_{role}_{m}_{n}_{k}"

  def kernel(out: UOp, words: UOp, xq: UOp, xscales: UOp) -> UOp:
    row = UOp.special(n, "gidx0")
    bb = UOp.special(m, "gidx1")
    lane = UOp.special(32, "lidx0")
    blk = UOp.range(k_blocks, 0, axis_type=AxisType.REDUCE)
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = UOp.const(dtypes.float32, 0.0)
    for grp in range(8):
      d, dmin, sc, mn = _q4k_group_params(words, base, grp)
      q = _q4k_quant(words, base, grp, lane).cast(dtypes.float32)
      w = d * sc.cast(dtypes.float32) * q - dmin * mn.cast(dtypes.float32)
      q8_idx = bb * k + blk * Q4_K_BLOCK_ELEMS + grp * Q8_1_BLOCK_ELEMS + lane
      scale_idx = bb * full_q8_groups + blk * (Q4_K_BLOCK_ELEMS // Q8_1_BLOCK_ELEMS) + grp
      x = xq[q8_idx].cast(dtypes.float32) * xscales[scale_idx].cast(dtypes.float32)
      contrib = contrib + w * x
    acc = UOp.placeholder((1,), dtypes.float32, 70, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(blk)[0] + contrib).end(blk))
    total = warp_reduce_sum(acc[0], lane, 32)
    return out[bb, row].store(total).sink(arg=KernelInfo(name=name, opts_to_apply=()))

  return kernel


def _pack_q8x4(xq: UOp, base_idx: UOp) -> UOp:
  packed = UOp.const(dtypes.uint32, 0)
  for i in range(4):
    byte = xq[base_idx + i].cast(dtypes.uint8).cast(dtypes.uint32)
    packed = packed.bitwise_or(byte.lshift(i * 8))
  return packed


def _sudot4(q_unsigned_bytes: UOp, x_signed_bytes: UOp) -> UOp:
  zero = UOp.const(dtypes.int32, 0)
  return UOp(Ops.CUSTOMI, dtypes.int32, (zero, q_unsigned_bytes, x_signed_bytes),
             arg="__builtin_amdgcn_sudot4(true, {1}, true, {2}, {0}, false)")


def _q4k_q8_1_bounded_dot4_kernel(m:int, n:int, k:int, role:str):
  if k % Q4_K_BLOCK_ELEMS:
    raise ValueError(f"AMD dot4 MMQ atom requires k to be Q4_K block aligned, got {k}")
  k_blocks = k // Q4_K_BLOCK_ELEMS
  full_q8_groups = k // Q8_1_BLOCK_ELEMS
  name = f"q4k_q8_1_mmq_dot4_batched_atom_{role}_{m}_{n}_{k}"

  def kernel(out: UOp, words: UOp, xq: UOp, xscales: UOp) -> UOp:
    row = UOp.special(n, "gidx0")
    bb = UOp.special(m, "gidx1")
    lane = UOp.special(32, "lidx0")
    lane4 = lane % UOp.const(dtypes.int32, 8)
    active = lane < UOp.const(dtypes.int32, 8)
    blk = UOp.range(k_blocks, 0, axis_type=AxisType.REDUCE)
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = UOp.const(dtypes.float32, 0.0)
    ones = UOp.const(dtypes.uint32, 0x01010101)
    for grp in range(8):
      d, dmin, sc, mn = _q4k_group_params(words, base, grp)
      qword = words[base + 4 + (grp//2)*8 + lane4]
      qpack = qword.rshift((grp % 2) * 4).bitwise_and(0x0F0F0F0F)
      q8_idx = bb * k + blk * Q4_K_BLOCK_ELEMS + grp * Q8_1_BLOCK_ELEMS + lane4 * 4
      scale_idx = bb * full_q8_groups + blk * (Q4_K_BLOCK_ELEMS // Q8_1_BLOCK_ELEMS) + grp
      xpack = _pack_q8x4(xq, q8_idx)
      dot_q = _sudot4(qpack, xpack).cast(dtypes.float32)
      dot_sum = _sudot4(ones, xpack).cast(dtypes.float32)
      scale = xscales[scale_idx].cast(dtypes.float32)
      contrib = contrib + scale * (d * sc.cast(dtypes.float32) * dot_q - dmin * mn.cast(dtypes.float32) * dot_sum)
    contrib = active.where(contrib, UOp.const(dtypes.float32, 0.0))
    acc = UOp.placeholder((1,), dtypes.float32, 70, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(blk)[0] + contrib).end(blk))
    total = warp_reduce_sum(acc[0], lane, 32)
    return out[bb, row].store(total).sink(arg=KernelInfo(name=name, opts_to_apply=()))

  return kernel


def _q4k_q8_1_bounded_dot4x4_kernel(m:int, n:int, k:int, role:str):
  if k % Q4_K_BLOCK_ELEMS:
    raise ValueError(f"AMD dot4x4 MMQ atom requires k to be Q4_K block aligned, got {k}")
  if m % 4:
    raise ValueError(f"AMD dot4x4 MMQ atom requires M to be a multiple of 4, got {m}")
  k_blocks = k // Q4_K_BLOCK_ELEMS
  full_q8_groups = k // Q8_1_BLOCK_ELEMS
  name = f"q4k_q8_1_mmq_dot4x4_batched_atom_{role}_{m}_{n}_{k}"

  def kernel(out: UOp, words: UOp, xq: UOp, xscales: UOp) -> UOp:
    row = UOp.special(n, "gidx0")
    bb4 = UOp.special(m // 4, "gidx1")
    lane = UOp.special(32, "lidx0")
    subtok = lane // UOp.const(dtypes.int32, 8)
    lane4 = lane % UOp.const(dtypes.int32, 8)
    bb = bb4 * 4 + subtok
    blk = UOp.range(k_blocks, 0, axis_type=AxisType.REDUCE)
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = UOp.const(dtypes.float32, 0.0)
    ones = UOp.const(dtypes.uint32, 0x01010101)
    for grp in range(8):
      d, dmin, sc, mn = _q4k_group_params(words, base, grp)
      qword = words[base + 4 + (grp//2)*8 + lane4]
      qpack = qword.rshift((grp % 2) * 4).bitwise_and(0x0F0F0F0F)
      q8_idx = bb * k + blk * Q4_K_BLOCK_ELEMS + grp * Q8_1_BLOCK_ELEMS + lane4 * 4
      scale_idx = bb * full_q8_groups + blk * (Q4_K_BLOCK_ELEMS // Q8_1_BLOCK_ELEMS) + grp
      xpack = _pack_q8x4(xq, q8_idx)
      dot_q = _sudot4(qpack, xpack).cast(dtypes.float32)
      dot_sum = _sudot4(ones, xpack).cast(dtypes.float32)
      scale = xscales[scale_idx].cast(dtypes.float32)
      contrib = contrib + scale * (d * sc.cast(dtypes.float32) * dot_q - dmin * mn.cast(dtypes.float32) * dot_sum)
    acc = UOp.placeholder((1,), dtypes.float32, 70, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(blk)[0] + contrib).end(blk))
    total = warp_reduce_sum(acc[0], lane, 8)
    return out[bb, row].store(total).sink(arg=KernelInfo(name=name, opts_to_apply=()))

  return kernel


def _q4k_q8_1_bounded_ds4_dot4x4_kernel(m:int, n:int, k:int, role:str):
  if k % Q4_K_BLOCK_ELEMS:
    raise ValueError(f"AMD DS4 dot4x4 MMQ atom requires k to be Q4_K block aligned, got {k}")
  if m % 4:
    raise ValueError(f"AMD DS4 dot4x4 MMQ atom requires M to be a multiple of 4, got {m}")
  k_blocks = k // Q4_K_BLOCK_ELEMS
  name = f"q4k_q8_1_mmq_ds4_dot4x4_atom_{role}_{m}_{n}_{k}"

  def kernel(out: UOp, words: UOp, q8_values: UOp, q8_scales: UOp, q8_sums: UOp) -> UOp:
    row = UOp.special(n, "gidx0")
    bb4 = UOp.special(m // 4, "gidx1")
    lane = UOp.special(32, "lidx0")
    subtok = lane // UOp.const(dtypes.int32, 8)
    lane4 = lane % UOp.const(dtypes.int32, 8)
    bb = bb4 * 4 + subtok
    blk = UOp.range(k_blocks, 0, axis_type=AxisType.REDUCE)
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = UOp.const(dtypes.float32, 0.0)
    for grp in range(8):
      d, dmin, sc, mn = _q4k_group_params(words, base, grp)
      qpack = _q4k_group_qpack_lane4(words, base, grp, lane4)
      ds4_block = blk * 2 + (grp // 4)
      ds4_group = grp % 4
      q8_idx = (ds4_block * m + bb) * 128 + ds4_group * Q8_1_BLOCK_ELEMS + lane4 * 4
      meta_idx = (ds4_block * m + bb) * 4 + ds4_group
      xpack = _pack_q8x4(q8_values, q8_idx)
      dot_q = _sudot4(qpack, xpack).cast(dtypes.float32)
      scale = q8_scales[meta_idx].cast(dtypes.float32)
      xsum = q8_sums[meta_idx].cast(dtypes.float32)
      min_term = lane4.eq(0).where(dmin * mn.cast(dtypes.float32) * xsum, UOp.const(dtypes.float32, 0.0))
      contrib = contrib + scale * d * sc.cast(dtypes.float32) * dot_q - min_term
    acc = UOp.placeholder((1,), dtypes.float32, 70, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(blk)[0] + contrib).end(blk))
    total = warp_reduce_sum(acc[0], lane, 8)
    return out[bb, row].store(total).sink(arg=KernelInfo(name=name, opts_to_apply=()))

  return kernel


def _q4k_q8_1_bounded_ds4_warp_kernel(m:int, n:int, k:int, role:str):
  if k % Q4_K_BLOCK_ELEMS:
    raise ValueError(f"AMD DS4 warp MMQ atom requires k to be Q4_K block aligned, got {k}")
  k_blocks = k // Q4_K_BLOCK_ELEMS
  name = f"q4k_q8_1_mmq_ds4_warp_atom_{role}_{m}_{n}_{k}"

  def kernel(out: UOp, words: UOp, q8_values: UOp, q8_scales: UOp, q8_sums: UOp) -> UOp:
    row = UOp.special(n, "gidx0")
    bb = UOp.special(m, "gidx1")
    lane = UOp.special(32, "lidx0")
    blk = UOp.range(k_blocks, 0, axis_type=AxisType.REDUCE)
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = UOp.const(dtypes.float32, 0.0)
    for grp in range(8):
      d, dmin, sc, mn = _q4k_group_params(words, base, grp)
      ds4_block = blk * 2 + (grp // 4)
      ds4_group = grp % 4
      q8_idx = (ds4_block * m + bb) * 128 + ds4_group * Q8_1_BLOCK_ELEMS + lane
      meta_idx = (ds4_block * m + bb) * 4 + ds4_group
      q = _q4k_quant(words, base, grp, lane).cast(dtypes.float32)
      scale = q8_scales[meta_idx].cast(dtypes.float32)
      x = q8_values[q8_idx].cast(dtypes.float32) * scale
      xsum = q8_sums[meta_idx].cast(dtypes.float32)
      min_term = lane.eq(0).where(dmin * mn.cast(dtypes.float32) * xsum, UOp.const(dtypes.float32, 0.0))
      contrib = contrib + d * sc.cast(dtypes.float32) * q * x - min_term
    acc = UOp.placeholder((1,), dtypes.float32, 70, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(blk)[0] + contrib).end(blk))
    total = warp_reduce_sum(acc[0], lane, 32)
    return out[bb, row].store(total).sink(arg=KernelInfo(name=name, opts_to_apply=()))

  return kernel


def _q4k_q8_1_bounded_ds4_lds_skeleton_kernel(m:int, n:int, k:int, role:str):
  if k % Q4_K_BLOCK_ELEMS:
    raise ValueError(f"AMD DS4 LDS skeleton MMQ atom requires k to be Q4_K block aligned, got {k}")
  k_blocks = k // Q4_K_BLOCK_ELEMS
  name = f"q4k_q8_1_mmq_ds4_lds_skeleton_atom_{role}_{m}_{n}_{k}"

  def kernel(out: UOp, words: UOp, q8_values: UOp, q8_scales: UOp, q8_sums: UOp) -> UOp:
    row = UOp.special(n, "gidx0")
    bb = UOp.special(m, "gidx1")
    lane = UOp.special(32, "lidx0")
    st_blk = UOp.range(k_blocks, 0)
    st_grp = UOp.range(8, 1)
    lds_q8 = UOp.placeholder((k,), dtypes.int8, 206, addrspace=AddrSpace.LOCAL)
    st_ds4_block = st_blk * 2 + (st_grp // UOp.const(dtypes.int32, 4))
    st_ds4_group = st_grp % UOp.const(dtypes.int32, 4)
    st_global = (st_ds4_block * m + bb) * 128 + st_ds4_group * Q8_1_BLOCK_ELEMS + lane
    st_local = st_blk * Q4_K_BLOCK_ELEMS + st_grp * Q8_1_BLOCK_ELEMS + lane
    stage = lds_q8[st_local].store(q8_values[st_global]).end(st_grp).end(st_blk)
    bar = UOp.barrier(UOp.group(stage))

    blk = UOp.range(k_blocks, 2, axis_type=AxisType.REDUCE)
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = UOp.const(dtypes.float32, 0.0)
    for grp in range(8):
      d, dmin, sc, mn = _q4k_group_params(words, base, grp)
      ds4_block = blk * 2 + (grp // 4)
      ds4_group = grp % 4
      lds_idx = blk * Q4_K_BLOCK_ELEMS + grp * Q8_1_BLOCK_ELEMS + lane
      meta_idx = (ds4_block * m + bb) * 4 + ds4_group
      q = _q4k_quant(words, base, grp, lane).cast(dtypes.float32)
      scale = q8_scales[meta_idx].cast(dtypes.float32)
      x = lds_q8.after(bar)[lds_idx].cast(dtypes.float32) * scale
      xsum = q8_sums[meta_idx].cast(dtypes.float32)
      min_term = lane.eq(0).where(dmin * mn.cast(dtypes.float32) * xsum, UOp.const(dtypes.float32, 0.0))
      contrib = contrib + d * sc.cast(dtypes.float32) * q * x - min_term
    acc = UOp.placeholder((1,), dtypes.float32, 70, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(blk)[0] + contrib).end(blk))
    total = warp_reduce_sum(acc[0], lane, 32)
    return out[bb, row].store(total).sink(arg=KernelInfo(name=name, opts_to_apply=()))

  return kernel


def _q4k_q8_1_bounded_ds4_coop_tile_kernel(m:int, n:int, k:int, role:str):
  if (m, n, k) != (16, 16, 256):
    raise ValueError(f"AMD DS4 coop tile atom is bounded to 16x16x256, got {m}x{n}x{k}")
  k_blocks = k // Q4_K_BLOCK_ELEMS
  name = f"q4k_q8_1_mmq_ds4_coop_tile_atom_{role}_{m}_{n}_{k}"

  def kernel(out: UOp, words: UOp, q8_values: UOp, q8_scales: UOp, q8_sums: UOp) -> UOp:
    row = UOp.special(n, "gidx0")
    bb = UOp.special(m, "gidx1")
    lane = UOp.special(32, "lidx0")
    st_blk = UOp.range(k_blocks, 0)
    st_grp = UOp.range(8, 1)
    lds_q8 = UOp.placeholder((k,), dtypes.int8, 206, addrspace=AddrSpace.LOCAL)
    st_ds4_block = st_blk * 2 + (st_grp // UOp.const(dtypes.int32, 4))
    st_ds4_group = st_grp % UOp.const(dtypes.int32, 4)
    st_global = (st_ds4_block * m + bb) * 128 + st_ds4_group * Q8_1_BLOCK_ELEMS + lane
    st_local = st_blk * Q4_K_BLOCK_ELEMS + st_grp * Q8_1_BLOCK_ELEMS + lane
    stage = lds_q8[st_local].store(q8_values[st_global]).end(st_grp).end(st_blk)
    bar = UOp.barrier(UOp.group(stage))

    blk = UOp.range(k_blocks, 2, axis_type=AxisType.REDUCE)
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = UOp.const(dtypes.float32, 0.0)
    for grp in range(8):
      d, dmin, sc, mn = _q4k_group_params(words, base, grp)
      ds4_block = blk * 2 + (grp // 4)
      ds4_group = grp % 4
      lds_idx = blk * Q4_K_BLOCK_ELEMS + grp * Q8_1_BLOCK_ELEMS + lane
      meta_idx = (ds4_block * m + bb) * 4 + ds4_group
      q = _q4k_quant(words, base, grp, lane).cast(dtypes.float32)
      scale = q8_scales[meta_idx].cast(dtypes.float32)
      x = lds_q8.after(bar)[lds_idx].cast(dtypes.float32) * scale
      xsum = q8_sums[meta_idx].cast(dtypes.float32)
      min_term = lane.eq(0).where(dmin * mn.cast(dtypes.float32) * xsum, UOp.const(dtypes.float32, 0.0))
      contrib = contrib + d * sc.cast(dtypes.float32) * q * x - min_term
    acc = UOp.placeholder((1,), dtypes.float32, 70, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(blk)[0] + contrib).end(blk))
    total = warp_reduce_sum(acc[0], lane, 32)
    stores = []
    for mi in range(16):
      for ni in range(16):
        stores.append(out[mi, ni].store(total, gate=bb.eq(mi) & row.eq(ni)))
    return UOp.group(*stores).sink(arg=KernelInfo(name=name, opts_to_apply=()))

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


def amd_warp_batched_atom_source_hash(m:int, n:int, k:int, role:str) -> str:
  payload = repr(_q4k_q8_1_bounded_warp_kernel(m, n, k, role)(
    UOp.placeholder((m, n), dtypes.float32, 0),
    UOp.placeholder((n * (k // Q4_K_BLOCK_ELEMS) * Q4K_WORDS_PER_BLOCK,), dtypes.uint32, 1),
    UOp.placeholder((m * k,), dtypes.int8, 2),
    UOp.placeholder((m * (k // Q8_1_BLOCK_ELEMS),), dtypes.float32, 3)))
  return hashlib.sha256(payload.encode()).hexdigest()[:16]


def amd_dot4_batched_atom_source_hash(m:int, n:int, k:int, role:str) -> str:
  payload = repr(_q4k_q8_1_bounded_dot4_kernel(m, n, k, role)(
    UOp.placeholder((m, n), dtypes.float32, 0),
    UOp.placeholder((n * (k // Q4_K_BLOCK_ELEMS) * Q4K_WORDS_PER_BLOCK,), dtypes.uint32, 1),
    UOp.placeholder((m * k,), dtypes.int8, 2),
    UOp.placeholder((m * (k // Q8_1_BLOCK_ELEMS),), dtypes.float32, 3)))
  return hashlib.sha256(payload.encode()).hexdigest()[:16]


def amd_dot4x4_batched_atom_source_hash(m:int, n:int, k:int, role:str) -> str:
  payload = repr(_q4k_q8_1_bounded_dot4x4_kernel(m, n, k, role)(
    UOp.placeholder((m, n), dtypes.float32, 0),
    UOp.placeholder((n * (k // Q4_K_BLOCK_ELEMS) * Q4K_WORDS_PER_BLOCK,), dtypes.uint32, 1),
    UOp.placeholder((m * k,), dtypes.int8, 2),
    UOp.placeholder((m * (k // Q8_1_BLOCK_ELEMS),), dtypes.float32, 3)))
  return hashlib.sha256(payload.encode()).hexdigest()[:16]


def amd_ds4_dot4x4_atom_source_hash(m:int, n:int, k:int, role:str) -> str:
  payload = repr(_q4k_q8_1_bounded_ds4_dot4x4_kernel(m, n, k, role)(
    UOp.placeholder((m, n), dtypes.float32, 0),
    UOp.placeholder((n * (k // Q4_K_BLOCK_ELEMS) * Q4K_WORDS_PER_BLOCK,), dtypes.uint32, 1),
    UOp.placeholder(((k // 128) * m * 128,), dtypes.int8, 2),
    UOp.placeholder(((k // 128) * m * 4,), dtypes.float32, 3),
    UOp.placeholder(((k // 128) * m * 4,), dtypes.float32, 4)))
  return hashlib.sha256(payload.encode()).hexdigest()[:16]


def amd_ds4_warp_atom_source_hash(m:int, n:int, k:int, role:str) -> str:
  payload = repr(_q4k_q8_1_bounded_ds4_warp_kernel(m, n, k, role)(
    UOp.placeholder((m, n), dtypes.float32, 0),
    UOp.placeholder((n * (k // Q4_K_BLOCK_ELEMS) * Q4K_WORDS_PER_BLOCK,), dtypes.uint32, 1),
    UOp.placeholder(((k // 128) * m * 128,), dtypes.int8, 2),
    UOp.placeholder(((k // 128) * m * 4,), dtypes.float32, 3),
    UOp.placeholder(((k // 128) * m * 4,), dtypes.float32, 4)))
  return hashlib.sha256(payload.encode()).hexdigest()[:16]


def amd_ds4_lds_skeleton_atom_source_hash(m:int, n:int, k:int, role:str) -> str:
  payload = repr(_q4k_q8_1_bounded_ds4_lds_skeleton_kernel(m, n, k, role)(
    UOp.placeholder((m, n), dtypes.float32, 0),
    UOp.placeholder((n * (k // Q4_K_BLOCK_ELEMS) * Q4K_WORDS_PER_BLOCK,), dtypes.uint32, 1),
    UOp.placeholder(((k // 128) * m * 128,), dtypes.int8, 2),
    UOp.placeholder(((k // 128) * m * 4,), dtypes.float32, 3),
    UOp.placeholder(((k // 128) * m * 4,), dtypes.float32, 4)))
  return hashlib.sha256(payload.encode()).hexdigest()[:16]


def amd_ds4_coop_tile_atom_source_hash(m:int, n:int, k:int, role:str) -> str:
  payload = repr(_q4k_q8_1_bounded_ds4_coop_tile_kernel(m, n, k, role)(
    UOp.placeholder((m, n), dtypes.float32, 0),
    UOp.placeholder((n * (k // Q4_K_BLOCK_ELEMS) * Q4K_WORDS_PER_BLOCK,), dtypes.uint32, 1),
    UOp.placeholder(((k // 128) * m * 128,), dtypes.int8, 2),
    UOp.placeholder(((k // 128) * m * 4,), dtypes.float32, 3),
    UOp.placeholder(((k // 128) * m * 4,), dtypes.float32, 4)))
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


def run_q4k_q8_1_mmq_bounded_amd_warp(q4k_bytes: np.ndarray, xq: np.ndarray, xscales: np.ndarray, *,
                                      role: str, device: str = "AMD") -> Q4KQ8MMQAtomResult:
  q4 = np.asarray(q4k_bytes, dtype=np.uint8)
  if q4.ndim != 3:
    raise ValueError(f"q4k_bytes must have shape [N,K/256,144], got {q4.shape}")
  n, k_blocks, _ = q4.shape
  k = k_blocks * Q4_K_BLOCK_ELEMS
  xq_arr = np.asarray(xq, dtype=np.int8)
  if xq_arr.ndim != 2 or xq_arr.shape[1] != k:
    raise ValueError(f"xq must have shape [M,{k}], got {xq_arr.shape}")
  m = xq_arr.shape[0]
  words = Tensor(_as_u32_words(q4), dtype=dtypes.uint32, device=device).realize()
  xq_t = Tensor(np.ascontiguousarray(xq_arr.reshape(-1)), dtype=dtypes.int8, device=device).realize()
  xs_t = Tensor(np.ascontiguousarray(np.asarray(xscales, dtype=np.float32).reshape(-1)), dtype=dtypes.float32, device=device).realize()
  out = Tensor.empty(m, n, dtype=dtypes.float32, device=device).custom_kernel(
    words, xq_t, xs_t, fxn=_q4k_q8_1_bounded_warp_kernel(m, n, k, role))[0].realize()
  return Q4KQ8MMQAtomResult(output=out.numpy().astype(np.float32), lifecycle=MMQLifecycleRow(role=role, tile_id=f"bounded_{m}x{n}x{k}",
                            counters=zero_counters(dot_accumulation_epochs=1, output_store_epochs=1, output_stores=m*n)),
                            backend_atom_id=AMD_WARP_BATCHED_BACKEND_ATOM_ID)


def run_q4k_q8_1_mmq_bounded_amd_dot4(q4k_bytes: np.ndarray, xq: np.ndarray, xscales: np.ndarray, *,
                                      role: str, device: str = "AMD") -> Q4KQ8MMQAtomResult:
  q4 = np.asarray(q4k_bytes, dtype=np.uint8)
  if q4.ndim != 3:
    raise ValueError(f"q4k_bytes must have shape [N,K/256,144], got {q4.shape}")
  n, k_blocks, _ = q4.shape
  k = k_blocks * Q4_K_BLOCK_ELEMS
  xq_arr = np.asarray(xq, dtype=np.int8)
  if xq_arr.ndim != 2 or xq_arr.shape[1] != k:
    raise ValueError(f"xq must have shape [M,{k}], got {xq_arr.shape}")
  m = xq_arr.shape[0]
  words = Tensor(_as_u32_words(q4), dtype=dtypes.uint32, device=device).realize()
  xq_t = Tensor(np.ascontiguousarray(xq_arr.reshape(-1)), dtype=dtypes.int8, device=device).realize()
  xs_t = Tensor(np.ascontiguousarray(np.asarray(xscales, dtype=np.float32).reshape(-1)), dtype=dtypes.float32, device=device).realize()
  out = Tensor.empty(m, n, dtype=dtypes.float32, device=device).custom_kernel(
    words, xq_t, xs_t, fxn=_q4k_q8_1_bounded_dot4_kernel(m, n, k, role))[0].realize()
  return Q4KQ8MMQAtomResult(output=out.numpy().astype(np.float32), lifecycle=MMQLifecycleRow(role=role, tile_id=f"bounded_dot4_{m}x{n}x{k}",
                            counters=zero_counters(dot_accumulation_epochs=1, output_store_epochs=1, output_stores=m*n)),
                            backend_atom_id=AMD_DOT4_BATCHED_BACKEND_ATOM_ID)


def run_q4k_q8_1_mmq_bounded_amd_dot4x4(q4k_bytes: np.ndarray, xq: np.ndarray, xscales: np.ndarray, *,
                                        role: str, device: str = "AMD") -> Q4KQ8MMQAtomResult:
  q4 = np.asarray(q4k_bytes, dtype=np.uint8)
  if q4.ndim != 3:
    raise ValueError(f"q4k_bytes must have shape [N,K/256,144], got {q4.shape}")
  n, k_blocks, _ = q4.shape
  k = k_blocks * Q4_K_BLOCK_ELEMS
  xq_arr = np.asarray(xq, dtype=np.int8)
  if xq_arr.ndim != 2 or xq_arr.shape[1] != k:
    raise ValueError(f"xq must have shape [M,{k}], got {xq_arr.shape}")
  m = xq_arr.shape[0]
  words = Tensor(_as_u32_words(q4), dtype=dtypes.uint32, device=device).realize()
  xq_t = Tensor(np.ascontiguousarray(xq_arr.reshape(-1)), dtype=dtypes.int8, device=device).realize()
  xs_t = Tensor(np.ascontiguousarray(np.asarray(xscales, dtype=np.float32).reshape(-1)), dtype=dtypes.float32, device=device).realize()
  out = Tensor.empty(m, n, dtype=dtypes.float32, device=device).custom_kernel(
    words, xq_t, xs_t, fxn=_q4k_q8_1_bounded_dot4x4_kernel(m, n, k, role))[0].realize()
  return Q4KQ8MMQAtomResult(output=out.numpy().astype(np.float32), lifecycle=MMQLifecycleRow(role=role, tile_id=f"bounded_dot4x4_{m}x{n}x{k}",
                            counters=zero_counters(dot_accumulation_epochs=1, output_store_epochs=1, output_stores=m*n)),
                            backend_atom_id=AMD_DOT4X4_BATCHED_BACKEND_ATOM_ID)


def _ds4_tensors(ds4: Q81MMQDS4Activation, device: str) -> tuple[Tensor, Tensor, Tensor]:
  values_t = Tensor(np.ascontiguousarray(np.asarray(ds4.values, dtype=np.int8).reshape(-1)), dtype=dtypes.int8, device=device).realize()
  scales_t = Tensor(np.ascontiguousarray(np.asarray(ds4.scales, dtype=np.float32).reshape(-1)), dtype=dtypes.float32, device=device).realize()
  sums_t = Tensor(np.ascontiguousarray(np.asarray(ds4.sums, dtype=np.float32).reshape(-1)), dtype=dtypes.float32, device=device).realize()
  return values_t, scales_t, sums_t


def run_q4k_q8_1_mmq_bounded_amd_ds4_warp(q4k_bytes: np.ndarray, ds4: Q81MMQDS4Activation, *,
                                          role: str, device: str = "AMD") -> Q4KQ8MMQAtomResult:
  q4 = np.asarray(q4k_bytes, dtype=np.uint8)
  if q4.ndim != 3:
    raise ValueError(f"q4k_bytes must have shape [N,K/256,144], got {q4.shape}")
  ds4.spec.validate()
  n, k_blocks, _ = q4.shape
  k = k_blocks * Q4_K_BLOCK_ELEMS
  if ds4.spec.k != k:
    raise ValueError(f"DS4 K={ds4.spec.k} does not match Q4_K K={k}")
  m = ds4.spec.m
  words = Tensor(_as_u32_words(q4), dtype=dtypes.uint32, device=device).realize()
  values_t, scales_t, sums_t = _ds4_tensors(ds4, device)
  out = Tensor.empty(m, n, dtype=dtypes.float32, device=device).custom_kernel(
    words, values_t, scales_t, sums_t, fxn=_q4k_q8_1_bounded_ds4_warp_kernel(m, n, k, role))[0].realize()
  lifecycle, detail = _staged_ds4_lifecycle_for_spec(
    Q4KQ81MMQTileSpec(role=role, m=m, n=n, k=k, m_tile=m, n_tile=n, activation_layout=Q8_1_MMQ_DS4_LAYOUT))
  detail = {**detail, "backend_stage": "amd_ds4_warp_direct_gpu", "gpu_kernel_emitted": True,
            "uses_precomputed_activation_sums": True, "shared_memory_staging": False}
  return Q4KQ8MMQAtomResult(output=out.numpy().astype(np.float32), lifecycle=lifecycle,
                            backend_atom_id=AMD_DS4_WARP_BACKEND_ATOM_ID, lifecycle_detail=detail)


def run_q4k_q8_1_mmq_bounded_amd_ds4_dot4x4(q4k_bytes: np.ndarray, ds4: Q81MMQDS4Activation, *,
                                            role: str, device: str = "AMD") -> Q4KQ8MMQAtomResult:
  q4 = np.asarray(q4k_bytes, dtype=np.uint8)
  if q4.ndim != 3:
    raise ValueError(f"q4k_bytes must have shape [N,K/256,144], got {q4.shape}")
  ds4.spec.validate()
  n, k_blocks, _ = q4.shape
  k = k_blocks * Q4_K_BLOCK_ELEMS
  if ds4.spec.k != k:
    raise ValueError(f"DS4 K={ds4.spec.k} does not match Q4_K K={k}")
  m = ds4.spec.m
  if m % 4:
    raise ValueError(f"AMD DS4 dot4x4 MMQ atom requires M to be a multiple of 4, got {m}")
  words = Tensor(_as_u32_words(q4), dtype=dtypes.uint32, device=device).realize()
  values_t, scales_t, sums_t = _ds4_tensors(ds4, device)
  out = Tensor.empty(m, n, dtype=dtypes.float32, device=device).custom_kernel(
    words, values_t, scales_t, sums_t, fxn=_q4k_q8_1_bounded_ds4_dot4x4_kernel(m, n, k, role))[0].realize()
  lifecycle, detail = _staged_ds4_lifecycle_for_spec(
    Q4KQ81MMQTileSpec(role=role, m=m, n=n, k=k, m_tile=m, n_tile=n, activation_layout=Q8_1_MMQ_DS4_LAYOUT))
  detail = {**detail, "backend_stage": "amd_ds4_dot4x4_direct_gpu", "gpu_kernel_emitted": True,
            "uses_precomputed_activation_sums": True, "shared_memory_staging": False}
  return Q4KQ8MMQAtomResult(output=out.numpy().astype(np.float32), lifecycle=lifecycle,
                            backend_atom_id=AMD_DS4_DOT4X4_BACKEND_ATOM_ID, lifecycle_detail=detail)


def run_q4k_q8_1_mmq_bounded_amd_ds4_lds_skeleton(q4k_bytes: np.ndarray, ds4: Q81MMQDS4Activation, *,
                                                  role: str, device: str = "AMD") -> Q4KQ8MMQAtomResult:
  q4 = np.asarray(q4k_bytes, dtype=np.uint8)
  if q4.ndim != 3:
    raise ValueError(f"q4k_bytes must have shape [N,K/256,144], got {q4.shape}")
  ds4.spec.validate()
  n, k_blocks, _ = q4.shape
  k = k_blocks * Q4_K_BLOCK_ELEMS
  if ds4.spec.k != k:
    raise ValueError(f"DS4 K={ds4.spec.k} does not match Q4_K K={k}")
  m = ds4.spec.m
  words = Tensor(_as_u32_words(q4), dtype=dtypes.uint32, device=device).realize()
  values_t, scales_t, sums_t = _ds4_tensors(ds4, device)
  out = Tensor.empty(m, n, dtype=dtypes.float32, device=device).custom_kernel(
    words, values_t, scales_t, sums_t, fxn=_q4k_q8_1_bounded_ds4_lds_skeleton_kernel(m, n, k, role))[0].realize()
  lifecycle, detail = _staged_ds4_lifecycle_for_spec(
    Q4KQ81MMQTileSpec(role=role, m=m, n=n, k=k, m_tile=m, n_tile=n, activation_layout=Q8_1_MMQ_DS4_LAYOUT))
  staged_activation_values = m * k
  detail = {**detail, "backend_stage": "amd_ds4_lds_skeleton_gpu", "gpu_kernel_emitted": True,
            "uses_precomputed_activation_sums": True, "shared_memory_staging": True,
            "lds_layout": "per_output_token_q8_values_linear_k",
            "local_activation_q8_stores": staged_activation_values,
            "local_activation_q8_loads": m * n * k,
            "global_q4k_tile_loads": m * n * k_blocks,
            "global_activation_ds4_loads": staged_activation_values,
            "output_stores": m * n,
            "bounded_only": True,
            "promotion_eligible": False,
            "production_dispatch_changed": False,
            "default_route": "direct_packed"}
  return Q4KQ8MMQAtomResult(output=out.numpy().astype(np.float32), lifecycle=lifecycle,
                            backend_atom_id=AMD_DS4_LDS_SKELETON_BACKEND_ATOM_ID, lifecycle_detail=detail)


def run_q4k_q8_1_mmq_bounded_amd_ds4_coop_tile(q4k_bytes: np.ndarray, ds4: Q81MMQDS4Activation, *,
                                               role: str, device: str = "AMD") -> Q4KQ8MMQAtomResult:
  q4 = np.asarray(q4k_bytes, dtype=np.uint8)
  if q4.ndim != 3:
    raise ValueError(f"q4k_bytes must have shape [N,K/256,144], got {q4.shape}")
  ds4.spec.validate()
  n, k_blocks, _ = q4.shape
  k = k_blocks * Q4_K_BLOCK_ELEMS
  m = ds4.spec.m
  if (m, n, k) != (16, 16, 256):
    raise ValueError(f"AMD DS4 coop tile atom is bounded to 16x16x256, got {m}x{n}x{k}")
  if ds4.spec.k != k:
    raise ValueError(f"DS4 K={ds4.spec.k} does not match Q4_K K={k}")
  words = Tensor(_as_u32_words(q4), dtype=dtypes.uint32, device=device).realize()
  values_t, scales_t, sums_t = _ds4_tensors(ds4, device)
  try:
    out = Tensor.empty(m, n, dtype=dtypes.float32, device=device).custom_kernel(
      words, values_t, scales_t, sums_t, fxn=_q4k_q8_1_bounded_ds4_coop_tile_kernel(m, n, k, role))[0].realize()
  except TypeError as exc:
    if "'tuple' and 'NoneType'" in str(exc):
      raise RuntimeError(AMD_DS4_COOP_TILE_BLOCKER) from exc
    raise
  lifecycle, detail = _staged_ds4_lifecycle_for_spec(
    Q4KQ81MMQTileSpec(role=role, m=m, n=n, k=k, m_tile=m, n_tile=n, activation_layout=Q8_1_MMQ_DS4_LAYOUT))
  detail = {**detail, "backend_stage": "amd_ds4_coop_tile_gpu", "gpu_kernel_emitted": True,
            "uses_precomputed_activation_sums": True, "shared_memory_staging": True,
            "store_owner_metadata": False, "store_owner_count": 0,
            "store_owner_proof": "separate_r4_lowered_isa_trace",
            "bounded_only": True, "promotion_eligible": False,
            "production_dispatch_changed": False, "default_route": "direct_packed"}
  return Q4KQ8MMQAtomResult(output=out.numpy().astype(np.float32), lifecycle=lifecycle,
                            backend_atom_id=AMD_DS4_COOP_TILE_BACKEND_ATOM_ID, lifecycle_detail=detail)


def run_q4k_q8_1_mmq_tile_with_lifecycle(q4k_bytes: np.ndarray, xq: np.ndarray, xscales: np.ndarray,
                                         spec: Q4KQ81MMQTileSpec) -> Q4KQ8MMQAtomResult:
  spec.validate()
  output = q4k_q8_1_mmq_tile_reference(q4k_bytes, xq, xscales, spec)
  return Q4KQ8MMQAtomResult(output=np.asarray(output, dtype=np.float32), lifecycle=_lifecycle_for_spec(spec))


def run_q4k_q8_1_mmq_tile(q4k_bytes: np.ndarray, xq: np.ndarray, xscales: np.ndarray,
                          spec: Q4KQ81MMQTileSpec) -> np.ndarray:
  return run_q4k_q8_1_mmq_tile_with_lifecycle(q4k_bytes, xq, xscales, spec).output
