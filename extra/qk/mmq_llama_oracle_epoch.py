"""Construction of the source-pinned llama gfx1100 Q4_K/Q8_1 K256 stage.

This owns one exact epoch graph. Runtime K looping, tails, writeback, routing,
and performance admission remain separate and cannot be inferred from it.
"""
from __future__ import annotations

from tinygrad import dtypes
from tinygrad.codegen.opt.kernel_lds import (HierarchicalPackedRecordStage, HierarchicalPackedRecordStageDescriptor,
  PackedRecordLDSRegionBinding, PrecontractContractSpec, PrecontractThreadAxes,
  build_hierarchical_packed_record_stage, prove_hierarchical_packed_record_stage)
from tinygrad.dtype import AddrSpace, PtrDType
from tinygrad.uop.ops import AxisType, UOp

from extra.qk.mmq_llama_candidate_plan import llama_mmq_candidate_plan
from extra.qk.mmq_llama_record_producers import build_q4_k_record_template, build_q8_ds4_record_template


def _contracts(tc) -> tuple[PrecontractContractSpec, PrecontractContractSpec]:
  result = []
  for operand, role in enumerate(("A", "B")):
    axes = tuple(UOp.range(2, 1810+operand*4+i, AxisType.UPCAST) for i in range(4))
    element = ((axes[0]*2+axes[1])*2+axes[2])*2+axes[3]
    result.append(PrecontractContractSpec(role, axes, tuple((x.arg[0], 2) for x in axes), element,
      tuple(tc.lane_map.remaps()[operand].items())))
  return tuple(result)  # type: ignore[return-value]


def build_llama_oracle_epoch_stage(q4_source: UOp, q8_source: UOp, *, q4_word_offset:UOp|int=0,
                                   q8_byte_offset:UOp|int=0) -> HierarchicalPackedRecordStage:
  """Build the exact cooperative LDS stage for one K256 epoch and one 128x128 output tile."""
  if not isinstance(q4_source.dtype, PtrDType) or q4_source.dtype.base != dtypes.uint32 or q4_source.dtype.size < 128*36:
    raise TypeError("Q4 source must cover 128 physical uint32[36] blocks")
  if not isinstance(q8_source.dtype, PtrDType) or q8_source.dtype.base != dtypes.uint8 or q8_source.dtype.size < 2*128*144:
    raise TypeError("Q8 source must cover two physical [128,144] DS4 panels")
  if not isinstance(q4_word_offset, UOp): q4_word_offset = UOp.const(dtypes.weakint, q4_word_offset)
  if not isinstance(q8_byte_offset, UOp): q8_byte_offset = UOp.const(dtypes.weakint, q8_byte_offset)
  q4_source = q4_source.index(q4_word_offset, ptr=True)
  q8_source = q8_source.index(q8_byte_offset, ptr=True)
  plan = llama_mmq_candidate_plan()
  geometry, tc = plan.geometry, plan.tensor_core
  row_a, k_a = UOp.range(128, 1800, AxisType.LOOP), UOp.range(256, 1801, AxisType.REDUCE)
  row_b, k_b = UOp.range(128, 1802, AxisType.LOOP), UOp.range(256, 1803, AxisType.REDUCE)
  zero = UOp.const(dtypes.weakint, 0)
  templates = (build_q4_k_record_template("A", q4_source, row_a, k_a, zero),
               build_q8_ds4_record_template("B", q8_source, row_b, k_b, zero))
  threads = PrecontractThreadAxes(UOp.range(8, 1804, AxisType.LOCAL), UOp.range(1, 1805, AxisType.LOCAL),
                                  UOp.range(32, -1, AxisType.WARP))
  subtile_m, subtile_n = UOp.range(1, 1806, AxisType.UPCAST), UOp.range(8, 1807, AxisType.UPCAST)
  allocation = UOp.placeholder((geometry.lds_bytes,), dtypes.uint8, 1808, addrspace=AddrSpace.LOCAL)
  descriptor = HierarchicalPackedRecordStageDescriptor(plan.lifecycle, 256, 128, 32)
  stage = build_hierarchical_packed_record_stage(geometry, allocation=allocation, descriptor=descriptor,
    templates=templates, regions=(PackedRecordLDSRegionBinding("A", "q4"), PackedRecordLDSRegionBinding("B", "q8")),
    threads=threads, subtile_m=subtile_m, subtile_n=subtile_n, tc=tc, contracts=_contracts(tc))
  proof = prove_hierarchical_packed_record_stage(stage)
  if not proof.passed: raise ValueError("invalid exact llama epoch stage: " + "; ".join(proof.errors))
  return stage


__all__ = ["build_llama_oracle_epoch_stage"]
