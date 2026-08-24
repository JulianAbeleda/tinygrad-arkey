"""Closed-default native Q4_K gate/up four-warp fp16 MMVQ admission candidate.

The installed gate/up decode route emits one
``q4k_g3_lanemap_gemv_w1w3fused16_12288_4096`` kernel per block with one warp
per 12288-row output (grid [12288,1,1], threads [32,1,1]).  This candidate
swaps it 1:1 for a 128-thread, four-warp-per-row consumer with the same Q4_K
packed dequant arithmetic and the same fused silu(gate)*up epilogue.  Each warp
owns four of the row's 16 Q4_K blocks; the four warp partials rendezvous in
shared memory before the silu multiply.  The output is stored fp16 in-kernel,
matching the installed fused16 spelling, so the downstream FFN-down consumer
sees the same fp16 ABI.

The route is deliberately unreachable unless a research harness attaches an
explicit ``Q4KGateUpFourWarpAdmission`` to the concrete Q4_K ffn_gate linear.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tinygrad import Tensor, dtypes
from tinygrad.codegen.late.warp_reduce import _warp_reduce_sum_staged
from tinygrad.dtype import AddrSpace
from tinygrad.llm.decode_kernels import (
  LanePartition, Q4KGateUpLaneMap, Q4K_WORDS_PER_BLOCK, _q4k_block_dot_packed_load, _silu_uop)
from tinygrad.llm.kernel_program import (
  KernelProgram, KernelProgramProvenance, OutputSpec, execute_promoted_program)
from tinygrad.uop.ops import AxisType, KernelInfo, UOp

ROWS, K = 12288, 4096
WARP, WARPS_PER_ROW = 32, 4
K_BLOCKS = 16
BLOCKS_PER_WARP = K_BLOCKS // WARPS_PER_ROW


@dataclass(frozen=True)
class Q4KGateUpFourWarpAdmission:
  block_index: int
  def __post_init__(self):
    if not isinstance(self.block_index, int) or isinstance(self.block_index, bool) or self.block_index < 0:
      raise ValueError("Q4_K gate/up four-warp block index must be a non-negative integer")


def emit_q4k_gate_up_four_warp_fp16() -> callable:
  """Four-warp fp16-direct Q4_K gate/up consumer (128 threads/row, fused silu)."""
  lm = Q4KGateUpLaneMap(k=K, n=ROWS)
  lm.validate()
  name = f"q4k_gate_up_four_warp_fp16_{ROWS}_{K}"

  def kernel(out:UOp, gate_words:UOp, up_words:UOp, x:UOp) -> UOp:
    row = UOp.special(ROWS, "gidx0")
    lid = UOp.special(WARP * WARPS_PER_ROW, "lidx0")
    warp, lane = lid // WARP, lid % WARP
    part = LanePartition(lane, lane_extent=lm.lane_extent, words_per_group=lm.words_per_group)
    block = warp * BLOCKS_PER_WARP + part.block_group
    base = (row * K_BLOCKS + block) * Q4K_WORDS_PER_BLOCK

    contrib_g = _q4k_block_dot_packed_load(gate_words, x, base, block, part.word_col)
    contrib_u = _q4k_block_dot_packed_load(up_words, x, base, block, part.word_col)
    total_g = _warp_reduce_sum_staged(contrib_g, lane, WARP, 90)
    total_u = _warp_reduce_sum_staged(contrib_u, lane, WARP, 95)

    smem_g = UOp.placeholder((WARPS_PER_ROW,), dtypes.float32, 40, addrspace=AddrSpace.LOCAL)
    smem_u = UOp.placeholder((WARPS_PER_ROW,), dtypes.float32, 41, addrspace=AddrSpace.LOCAL)
    published_g = smem_g[warp].store(total_g, lane.eq(0))
    published_u = smem_u.after(published_g)[warp].store(total_u, lane.eq(0))
    ready = UOp.barrier(UOp.group(published_u))

    merged_g = UOp.const(dtypes.float32, 0.0)
    merged_u = UOp.const(dtypes.float32, 0.0)
    for wi in range(WARPS_PER_ROW):
      merged_g = merged_g + smem_g.after(ready)[wi]
      merged_u = merged_u + smem_u.after(ready)[wi]
    z = _silu_uop(merged_g) * merged_u
    return out[row].store(z.cast(dtypes.float16), lid.eq(0)).sink(
      arg=KernelInfo(name=name, opts_to_apply=()))
  return kernel


def q4k_gate_up_four_warp_call(admission:object, gate:Any, up:Any, x:Tensor) -> Tensor|None:
  """Return the leased four-warp gate/up consumer, or None without changing the installed path."""
  if not isinstance(admission, Q4KGateUpFourWarpAdmission): return None
  capability = getattr(getattr(gate, "route_admission", None), "capability", None)
  if (getattr(capability, "backend", None), getattr(capability, "architecture", None)) != ("NV", "sm_120"):
    return None
  if (getattr(gate, "route_role", None), getattr(gate, "out_features", None),
      getattr(gate, "in_features", None)) != ("ffn_gate_up", ROWS, K):
    return None
  if (getattr(up, "route_role", None), getattr(up, "out_features", None),
      getattr(up, "in_features", None)) != ("ffn_gate_up", ROWS, K):
    return None
  if getattr(gate, "bias", None) is not None or getattr(up, "bias", None) is not None: return None
  if not str(x.device).startswith("NV"): return None

  gw = gate.q4k_storage.words.to(x.device).contiguous() if gate.q4k_storage.mode == "q4_ondemand" else gate.q4k_storage.words.to(x.device)
  uw = up.q4k_storage.words.to(x.device).contiguous() if up.q4k_storage.mode == "q4_ondemand" else up.q4k_storage.words.to(x.device)
  xv = x[:, 0, :].reshape(K).cast(dtypes.float16).contiguous()
  consumer = KernelProgram("decode_q4k_gate_up_four_warp", f"blk{admission.block_index}.gate_up",
    KernelProgramProvenance.MACHINE_SEARCH_GENERATED, emit_q4k_gate_up_four_warp_fp16(),
    output_spec=OutputSpec((ROWS,), dtypes.float16))
  out = execute_promoted_program(Tensor.empty((ROWS,), dtype=dtypes.float16, device=x.device),
    gw, uw, xv, program=consumer)
  return out.reshape(1, 1, ROWS)


__all__ = ["Q4KGateUpFourWarpAdmission", "ROWS", "K", "emit_q4k_gate_up_four_warp_fp16",
           "q4k_gate_up_four_warp_call"]
