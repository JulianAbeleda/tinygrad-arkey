"""Construction of the source-pinned llama gfx1100 Q4_K/Q8_1 K256 stage.

This owns one exact epoch graph. Runtime K looping, tails, writeback, routing,
and performance admission remain separate and cannot be inferred from it.
"""
from __future__ import annotations

from tinygrad import dtypes
from extra.qk.kernel_lds import (HierarchicalPackedRecordStage, HierarchicalPackedRecordStageDescriptor,
  PackedRecordLDSRegionBinding, PrecontractContractSpec, PrecontractThreadAxes,
  build_hierarchical_packed_record_stage, prove_hierarchical_packed_record_stage)
from tinygrad.dtype import AddrSpace, PtrDType
from tinygrad.uop.ops import AxisType, UOp

from extra.qk.mmq_llama_candidate_plan import llama_mmq_candidate_plan
from extra.qk.mmq_llama_record_producers import (build_llama_activation_fp16_group_record_template,
  build_llama_q4_k_fp16_group_record_template, build_q4_k_record_template, build_q8_ds4_record_template,
  build_split_q8_ds4_record_template)


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


def build_llama_oracle_epoch_stage_five_buffer(q4_words: UOp, values: UOp, scales: UOp, sums: UOp, *,
                                                q4_word_offset:UOp|int=0, values_offset:UOp|int=0,
                                                scales_offset:UOp|int=0, sums_offset:UOp|int=0,
                                                q4_row_stride_words:int=36,
                                                q8_record_rows:int|None=None,
                                                ) -> HierarchicalPackedRecordStage:
  """Build the same exact epoch while adapting split five-buffer Q8 inputs."""
  if not isinstance(q4_words.dtype, PtrDType) or q4_words.dtype.base != dtypes.uint32 or q4_words.dtype.size < 128*36:
    raise TypeError("Q4 source must cover 128 physical uint32[36] blocks")
  if not isinstance(q4_row_stride_words, int) or isinstance(q4_row_stride_words, bool) or q4_row_stride_words < 36:
    raise ValueError("Q4 source row stride must be at least 36 uint32 words")
  if q8_record_rows is not None and \
     (not isinstance(q8_record_rows, int) or isinstance(q8_record_rows, bool) or q8_record_rows <= 0):
    raise ValueError("split Q8 record row count must be a positive integer")
  offsets = []
  for offset in (q4_word_offset, values_offset, scales_offset, sums_offset):
    offsets.append(offset if isinstance(offset, UOp) else UOp.const(dtypes.weakint, offset))
  q4_words, values, scales, sums = (source.index(offset, ptr=True) for source, offset in
    zip((q4_words, values, scales, sums), offsets))
  plan = llama_mmq_candidate_plan()
  geometry, tc = plan.geometry, plan.tensor_core
  row_a, k_a = UOp.range(128, 1800, AxisType.LOOP), UOp.range(256, 1801, AxisType.REDUCE)
  row_b, k_b = UOp.range(128, 1802, AxisType.LOOP), UOp.range(256, 1803, AxisType.REDUCE)
  zero = UOp.const(dtypes.weakint, 0)
  templates = (build_q4_k_record_template(
                 "A", q4_words, row_a, k_a, zero, row_stride_words=q4_row_stride_words),
               build_split_q8_ds4_record_template(
                 "B", values, scales, sums, row_b, k_b, zero, record_rows=q8_record_rows))
  # The generated full-grid kernel has one real 256-thread workgroup.  Keep
  # the source-pinned dense oracle above range-based, but bind this five-buffer
  # seam to the hardware local ID so every producer and WMMA fragment is
  # genuinely lane/wave owned.  A bare AxisType.WARP range is only a serial
  # loop in a one-workgroup kernel and would duplicate the complete stage per
  # thread.
  local = UOp.special(geometry.threads, "lidx0")
  linear_wave = local // geometry.wave_size
  threads = PrecontractThreadAxes(linear_wave // geometry.waves[1], linear_wave % geometry.waves[1],
                                  local % geometry.wave_size)
  subtile_m, subtile_n = UOp.range(1, 1806, AxisType.UPCAST), UOp.range(8, 1807, AxisType.UPCAST)
  allocation = UOp.placeholder((geometry.lds_bytes,), dtypes.uint8, 1808, addrspace=AddrSpace.LOCAL)
  descriptor = HierarchicalPackedRecordStageDescriptor(plan.lifecycle, 256, 128, 32)
  stage = build_hierarchical_packed_record_stage(geometry, allocation=allocation, descriptor=descriptor,
    templates=templates, regions=(PackedRecordLDSRegionBinding("A", "q4"), PackedRecordLDSRegionBinding("B", "q8")),
    threads=threads, subtile_m=subtile_m, subtile_n=subtile_n, tc=tc, contracts=_contracts(tc))
  proof = prove_hierarchical_packed_record_stage(stage)
  if not proof.passed: raise ValueError("invalid exact llama five-buffer epoch stage: " + "; ".join(proof.errors))
  return stage


def build_llama_oracle_group_stage(q4_words: UOp, activation: UOp, *, q4_word_offset:UOp|int=0,
                                   q4_row_stride_words:int=36, group_index:int=0, k_base:int=0,
                                   activation_element_offset:UOp|int=0, k_total:int) -> HierarchicalPackedRecordStage:
  """Build one K32-group fp16-dequant-in-register stage (implementation plan I.2/I.3, II.3/II.6).

  One call is exactly one hand-kernel ``decode_group``/``compute0`` step: Q4_K is
  dequantized straight into the fp16 A LDS row (no int8/Q8_1 activation split
  any more -- ``activation`` is a plain ``[M,K]`` fp16 buffer, cooperatively
  copied into the fp16 B LDS row).  A full K256 epoch is 8 calls
  (``group_index`` 0..7); an outer M/N/K loop is expected to invoke this 8x per
  K256 and chain the resulting recurrences (see ``mmq_llama_group_chain.py``).
  """
  if not isinstance(q4_words.dtype, PtrDType) or q4_words.dtype.base != dtypes.uint32 or q4_words.dtype.size < 128*36:
    raise TypeError("Q4 source must cover 128 physical uint32[36] blocks")
  if not isinstance(q4_row_stride_words, int) or isinstance(q4_row_stride_words, bool) or q4_row_stride_words < 36:
    raise ValueError("Q4 source row stride must be at least 36 uint32 words")
  if not isinstance(activation.dtype, PtrDType) or activation.dtype.base != dtypes.half:
    raise TypeError("activation source must be a physical half[M,K] array")
  if not 0 <= group_index < 8: raise ValueError("group_index must be in [0, 8)")
  if not isinstance(k_base, int) or isinstance(k_base, bool) or k_base < 0: raise ValueError("k_base must be a non-negative int")
  if not isinstance(k_total, int) or isinstance(k_total, bool) or k_total <= 0: raise ValueError("k_total must be a positive int")
  q4_word_offset = q4_word_offset if isinstance(q4_word_offset, UOp) else UOp.const(dtypes.weakint, q4_word_offset)
  activation_element_offset = activation_element_offset if isinstance(activation_element_offset, UOp) else \
    UOp.const(dtypes.weakint, activation_element_offset)
  q4_words = q4_words.index(q4_word_offset, ptr=True)
  activation = activation.index(activation_element_offset, ptr=True)
  plan = llama_mmq_candidate_plan()
  geometry, tc = plan.geometry, plan.tensor_core
  row_a, k_a = UOp.range(128, 1800, AxisType.LOOP), UOp.range(32, 1801, AxisType.REDUCE)
  row_b, k_b = UOp.range(128, 1802, AxisType.LOOP), UOp.range(32, 1803, AxisType.REDUCE)
  zero = UOp.const(dtypes.weakint, 0)
  templates = (build_llama_q4_k_fp16_group_record_template(
                 "A", q4_words, row_a, k_a, zero, group_index=group_index, row_stride_words=q4_row_stride_words),
               build_llama_activation_fp16_group_record_template(
                 "B", activation, row_b, k_b, zero, k_total=k_total, k_base=k_base))
  # One real 256-thread workgroup: bind to the hardware local ID (see
  # build_llama_oracle_epoch_stage_five_buffer) rather than a bare serial range.
  local = UOp.special(geometry.threads, "lidx0")
  linear_wave = local // geometry.wave_size
  threads = PrecontractThreadAxes(linear_wave // geometry.waves[1], linear_wave % geometry.waves[1],
                                  local % geometry.wave_size)
  subtile_m, subtile_n = UOp.range(1, 1806, AxisType.UPCAST), UOp.range(8, 1807, AxisType.UPCAST)
  allocation = UOp.placeholder((geometry.lds_bytes,), dtypes.uint8, 1808, addrspace=AddrSpace.LOCAL)
  descriptor = HierarchicalPackedRecordStageDescriptor(plan.lifecycle, 32, 32, 32)
  stage = build_hierarchical_packed_record_stage(geometry, allocation=allocation, descriptor=descriptor,
    templates=templates, regions=(PackedRecordLDSRegionBinding("A", "A"), PackedRecordLDSRegionBinding("B", "B")),
    threads=threads, subtile_m=subtile_m, subtile_n=subtile_n, tc=tc, contracts=_contracts(tc))
  proof = prove_hierarchical_packed_record_stage(stage)
  if not proof.passed: raise ValueError("invalid llama fp16 K32-group stage: " + "; ".join(proof.errors))
  return stage


__all__ = ["build_llama_oracle_epoch_stage", "build_llama_oracle_epoch_stage_five_buffer", "build_llama_oracle_group_stage"]
