"""Closed-default Q4_K K/V dual-output decode producer.

The route replaces two exact 1024x4096 vector Q4_K projection launches with
one launch that keeps two accumulators and writes a contiguous [K,V] pair.
Normal model loads never attach the admission object.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tinygrad import Tensor, dtypes
from tinygrad.codegen.late.warp_reduce import _warp_reduce_sum_staged
from tinygrad.dtype import AddrSpace
from tinygrad.llm.decode_kernels import (LanePartition, Q4KGateUpLaneMap, Q4K_WORDS_PER_BLOCK,
  _q4k_block_dot_packed_load_vec)
from tinygrad.llm.kernel_program import (KernelProgram, KernelProgramProvenance, OutputSpec,
  execute_promoted_program_outputs)
from tinygrad.uop.ops import AxisType, KernelInfo, UOp

ROWS, K, WARP = 1024, 4096, 32


@dataclass(frozen=True)
class Q4KKVPairAdmission:
  block_index:int

  def __post_init__(self):
    if not isinstance(self.block_index, int) or isinstance(self.block_index, bool) or self.block_index < 0:
      raise ValueError("Q4_K K/V pair block index must be a non-negative integer")


def emit_q4k_kv_pair_vector(rows:int=ROWS, k:int=K):
  """Two vector Q4_K dots with the installed per-lane association."""
  if (rows, k) != (ROWS, K): raise ValueError("Q4_K K/V pair requires the exact 1024x4096 shape")
  lm = Q4KGateUpLaneMap(k=k, n=rows)
  lm.validate()
  name = f"q4k_g3_lanemap_gemv_pair_vec_{rows}_{k}"

  def kernel(k_out:UOp, v_out:UOp, k_words:UOp, v_words:UOp, x:UOp) -> UOp:
    if k_out.shape != (rows,) or v_out.shape != (rows,) or k_out.dtype.base != dtypes.float32 or v_out.dtype.base != dtypes.float32:
      raise ValueError("Q4_K K/V pair output ABI mismatch")
    row, lane = UOp.special(rows, "gidx0"), UOp.special(WARP, "lidx0")
    part = LanePartition(lane, lane_extent=lm.lane_extent, words_per_group=lm.words_per_group)
    lblk = UOp.range(lm.blocks_per_group, 0, axis_type=AxisType.REDUCE)
    blk = part.block_group * lm.blocks_per_group + lblk
    base = (row * lm.k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib_k = _q4k_block_dot_packed_load_vec(k_words, x, base, blk, part.word_col)
    contrib_v = _q4k_block_dot_packed_load_vec(v_words, x, base, blk, part.word_col)
    acc_k = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
    acc_v = UOp.placeholder((1,), dtypes.float32, 21, addrspace=AddrSpace.REG)
    init = acc_k[0].store(0.0)
    init = acc_v.after(init)[0].store(0.0)
    acc_k, acc_v = acc_k.after(init), acc_v.after(init)
    upd_k = acc_k[0].store(acc_k.after(lblk)[0] + contrib_k)
    upd_v = acc_v.after(upd_k)[0].store(acc_v.after(lblk)[0] + contrib_v).end(lblk)
    total_k = _warp_reduce_sum_staged(acc_k.after(upd_v)[0], part.lane, part.lane_extent, 90)
    total_v = _warp_reduce_sum_staged(acc_v.after(upd_v)[0], part.lane, part.lane_extent, 95)
    return UOp.group(k_out[row].store(total_k), v_out[row].store(total_v)).sink(
      arg=KernelInfo(name=name, opts_to_apply=()))
  return kernel


def q4k_kv_pair_call(admission:object, k_linear:Any, v_linear:Any, x:Tensor) -> tuple[Tensor, Tensor]|None:
  """Return exact K/V tensors from the leased dual producer, or None."""
  if not isinstance(admission, Q4KKVPairAdmission): return None
  for linear in (k_linear, v_linear):
    capability = getattr(getattr(linear, "route_admission", None), "capability", None)
    if (getattr(capability, "backend", None), getattr(capability, "architecture", None)) != ("NV", "sm_120"): return None
    if (getattr(linear, "route_role", None), getattr(linear, "out_features", None),
        getattr(linear, "in_features", None)) != ("attn_kv", ROWS, K): return None
    if getattr(linear, "bias", None) is not None or not hasattr(linear, "q4k_storage"): return None
  if x.shape != (1, 1, K) or not str(x.device).startswith("NV"): return None
  kw = k_linear.q4k_storage.words.to(x.device).contiguous() if k_linear.q4k_storage.mode == "q4_ondemand" \
    else k_linear.q4k_storage.words.to(x.device)
  vw = v_linear.q4k_storage.words.to(x.device).contiguous() if v_linear.q4k_storage.mode == "q4_ondemand" \
    else v_linear.q4k_storage.words.to(x.device)
  xv = x[:, 0, :].reshape(K).cast(dtypes.float16).contiguous()
  program = KernelProgram("decode_q4k_kv_pair", f"blk{admission.block_index}.kv_pair",
    KernelProgramProvenance.TINYGRAD_SCHEDULER_GENERATED, emit_q4k_kv_pair_vector(),
    output_spec=OutputSpec((ROWS,), dtypes.float32))
  k_out=Tensor.empty((ROWS,),dtype=dtypes.float32,device=x.device)
  v_out=Tensor.empty((ROWS,),dtype=dtypes.float32,device=x.device)
  outputs=execute_promoted_program_outputs(k_out,v_out,kw,vw,xv,program=program)
  return outputs[0].reshape(1,1,ROWS),outputs[1].reshape(1,1,ROWS)


__all__ = ["Q4KKVPairAdmission", "emit_q4k_kv_pair_vector", "q4k_kv_pair_call"]
