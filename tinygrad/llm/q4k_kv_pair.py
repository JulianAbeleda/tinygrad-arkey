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
  _q4k_block_dot_packed_load, _q4k_block_dot_packed_load_vec)
from tinygrad.llm.kernel_program import (KernelProgram, KernelProgramProvenance, OutputSpec,
  execute_promoted_program_outputs)
from tinygrad.uop.ops import AxisType, KernelInfo, UOp

ROWS, K, WARP = 1024, 4096, 32
Q_ROWS = 4096


@dataclass(frozen=True)
class Q4KKVPairAdmission:
  block_index:int

  def __post_init__(self):
    if not isinstance(self.block_index, int) or isinstance(self.block_index, bool) or self.block_index < 0:
      raise ValueError("Q4_K K/V pair block index must be a non-negative integer")


@dataclass(frozen=True)
class Q4KQKVAdmission:
  block_index:int

  def __post_init__(self):
    if not isinstance(self.block_index, int) or isinstance(self.block_index, bool) or self.block_index < 0:
      raise ValueError("Q4_K Q/K/V producer block index must be a non-negative integer")


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


def emit_q4k_qkv_full(q_rows:int=Q_ROWS, kv_rows:int=ROWS, k:int=K):
  """Exact full-grid ordinary Q4/Q4/Q4 producer.

  Every CTA retains the installed vector-load Q dot. The first 2*kv_rows CTAs
  also execute one vector-load K-or-V dot from packed K-then-V row storage. The
  global CTA predicate is uniform for the whole workgroup, so the inactive Q
  tail skips the K/V body entirely.
  """
  if (q_rows, kv_rows, k) != (Q_ROWS, ROWS, K):
    raise ValueError("Q4_K Q/K/V producer requires exact 4096/1024x4096 shapes")
  lm=Q4KGateUpLaneMap(k=k,n=q_rows); lm.validate()
  name=f"q4k_g3_lanemap_gemv_qkv_full_{q_rows}_{kv_rows}_{k}"

  def kernel(q_out:UOp, k_out:UOp, v_out:UOp, q_words:UOp, kv_words:UOp, x:UOp) -> UOp:
    if q_out.shape != (q_rows,) or k_out.shape != (kv_rows,) or v_out.shape != (kv_rows,):
      raise ValueError("Q4_K Q/K/V output ABI mismatch")
    if any(out.dtype.base != dtypes.float32 for out in (q_out,k_out,v_out)):
      raise ValueError("Q4_K Q/K/V outputs must be fp32")
    row,lane=UOp.special(q_rows,"gidx0"),UOp.special(WARP,"lidx0")
    part=LanePartition(lane,lane_extent=lm.lane_extent,words_per_group=lm.words_per_group)

    qblk=UOp.range(lm.blocks_per_group,0,axis_type=AxisType.REDUCE)
    qblock=part.block_group*lm.blocks_per_group+qblk
    qbase=(row*lm.k_blocks+qblock)*Q4K_WORDS_PER_BLOCK
    qc=_q4k_block_dot_packed_load_vec(q_words,x,qbase,qblock,part.word_col)
    qa=UOp.placeholder((1,),dtypes.float32,20,addrspace=AddrSpace.REG)
    qa=qa.after(qa[0].store(0.0))
    qa=qa.after(qa[0].store(qa.after(qblk)[0]+qc).end(qblk))
    qt=_warp_reduce_sum_staged(qa[0],part.lane,part.lane_extent,90)
    qstore=q_out[row].store(qt)

    # Anchor the typed uniform region after the exact Q store. Packed K/V rows
    # map directly from CTA ids 0..2047, so the optional body needs no pointer
    # selection and writes into separate caller-owned outputs.
    anchor=UOp.barrier(UOp.group(qstore))
    region=anchor.post_barrier_region(row<kv_rows*2,workgroup_uniform=True)
    kblk=UOp.range(lm.blocks_per_group,1,axis_type=AxisType.REDUCE)
    kblock=part.block_group*lm.blocks_per_group+kblk
    kbase=(row*lm.k_blocks+kblock)*Q4K_WORDS_PER_BLOCK
    kc=_q4k_block_dot_packed_load_vec(kv_words.after(region),x.after(region),kbase,kblock,part.word_col)
    ka=UOp.placeholder((1,),dtypes.float32,21,addrspace=AddrSpace.REG)
    ka=ka.after(ka[0].store(0.0))
    ka=ka.after(ka[0].store(ka.after(kblk)[0]+kc).end(kblk))
    kt=_warp_reduce_sum_staged(ka[0],part.lane,part.lane_extent,100)
    k_idx=(row<kv_rows).where(row,UOp.const(dtypes.weakint,0))
    v_idx=(row>=kv_rows).where(row-kv_rows,UOp.const(dtypes.weakint,0))
    stores=UOp.group(k_out[k_idx].store(kt,row<kv_rows),v_out[v_idx].store(kt,row>=kv_rows))
    return region.end_region(stores).sink(arg=KernelInfo(name=name,opts_to_apply=()))
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


def q4k_qkv_call(admission:object, q_linear:Any, k_linear:Any, v_linear:Any, x:Tensor) -> tuple[Tensor,Tensor,Tensor]|None:
  """Return exact ordinary Q/K/V tensors from the leased full-grid producer."""
  if not isinstance(admission,Q4KQKVAdmission): return None
  for linear,role,rows in ((q_linear,"attn_qo",Q_ROWS),(k_linear,"attn_kv",ROWS),(v_linear,"attn_kv",ROWS)):
    capability=getattr(getattr(linear,"route_admission",None),"capability",None)
    if (getattr(capability,"backend",None),getattr(capability,"architecture",None)) != ("NV","sm_120"): return None
    if (getattr(linear,"route_role",None),getattr(linear,"out_features",None),getattr(linear,"in_features",None)) != (role,rows,K): return None
    if getattr(linear,"bias",None) is not None or not hasattr(linear,"q4k_storage"): return None
  if x.shape != (1,1,K) or not str(x.device).startswith("NV"): return None
  packed_words=getattr(q_linear,"_q4k_qkv_words",None)
  expected=2*ROWS*(K//256)*Q4K_WORDS_PER_BLOCK
  if packed_words is None or packed_words.shape != (expected,): return None
  qw=q_linear.q4k_storage.words.to(x.device).contiguous() if q_linear.q4k_storage.mode == "q4_ondemand" else q_linear.q4k_storage.words.to(x.device)
  xv=x[:,0,:].reshape(K).cast(dtypes.float16).contiguous()
  program=KernelProgram("decode_q4k_qkv",f"blk{admission.block_index}.qkv_full",
    KernelProgramProvenance.TINYGRAD_SCHEDULER_GENERATED,emit_q4k_qkv_full(),output_spec=OutputSpec((Q_ROWS,),dtypes.float32))
  q_out=Tensor.empty((Q_ROWS,),dtype=dtypes.float32,device=x.device)
  k_out=Tensor.empty((ROWS,),dtype=dtypes.float32,device=x.device)
  v_out=Tensor.empty((ROWS,),dtype=dtypes.float32,device=x.device)
  outputs=execute_promoted_program_outputs(q_out,k_out,v_out,qw,packed_words.to(x.device),xv,program=program)
  return tuple(out.reshape(1,1,rows) for out,rows in zip(outputs,(Q_ROWS,ROWS,ROWS)))


__all__ = ["Q4KKVPairAdmission", "Q4KQKVAdmission", "emit_q4k_kv_pair_vector", "emit_q4k_qkv_full",
           "q4k_kv_pair_call", "q4k_qkv_call"]
