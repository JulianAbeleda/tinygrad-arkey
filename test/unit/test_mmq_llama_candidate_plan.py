from dataclasses import replace
from copy import deepcopy

from extra.qk.mmq_llama_candidate_plan import llama_mmq_candidate_plan
from extra.qk.mmq_llama_differential import LLAMA_SOURCE_COMMIT, compare_structures, llama_q4k_q8_structural_descriptor
from tinygrad import dtypes
from tinygrad.codegen.opt.kernel_pipeline import DotUpdateRecurrencePlan, HierarchicalKernelPipelinePlan
from tinygrad.codegen.opt.kernel_writeback import WMMAWritebackDescriptor
from tinygrad.codegen.opt.packed_weight import PackedOperandRecordTransform
from tinygrad.uop.ops import KernelCandidateContext, KernelTileGeometry


def test_source_anchored_typed_plan_and_formulas():
  p = llama_mmq_candidate_plan()
  assert p.source_commit == LLAMA_SOURCE_COMMIT and len(LLAMA_SOURCE_COMMIT) == 40
  assert isinstance(p.geometry, KernelTileGeometry) and p.geometry.tile == (128, 128, 256)
  assert (p.geometry.waves, p.geometry.threads, p.geometry.wave_size, p.geometry.lds_bytes) == ((8, 1), 256, 32, 57856)
  assert tuple(x.role for x in p.geometry.lds_windows) == ("B", "A")
  assert isinstance(p.q8_transform, PackedOperandRecordTransform) and p.q8_transform.source == p.q8_transform.produced
  assert isinstance(p.q4_transform, PackedOperandRecordTransform) and p.q4_transform.source != p.q4_transform.produced
  assert p.q4_transform.source.components[-1].size_bytes == 128 and sum(x.size_bytes for x in p.q4_transform.source.components) == 144
  q8, q4 = p.geometry.lds_region("q8"), p.geometry.lds_region("q4")
  assert (q8.base, q8.end, q8.records.rows, q8.records.stride_bytes) == (512, 18944, 128, 144)
  assert [(x.component, x.dtype, x.size_bytes) for x in q8.records.components] == [("ds", dtypes.half, 16), ("qs", dtypes.char, 128)]
  assert (q4.base, q4.end, q4.records.rows, q4.records.stride_bytes) == (18944, 57856, 128, 304)
  assert [(x.component, x.dtype, x.size_bytes) for x in q4.records.components] == [
    ("qs", dtypes.int, 256), ("dm", dtypes.half, 32), ("padding", dtypes.int, 16)]
  assert isinstance(p.lifecycle, HierarchicalKernelPipelinePlan) and p.lifecycle.phase_count == 2
  assert isinstance(p.recurrence, DotUpdateRecurrencePlan)
  assert (p.recurrence.persistent_dtype, p.recurrence.dot_dtype) == (dtypes.float.vec(8), dtypes.int.vec(8))
  assert (p.recurrence.group_count, p.recurrence.total_dot_count) == (2*4, 2*4*2)
  assert isinstance(p.writeback, WMMAWritebackDescriptor) and p.writeback.layout.identified_axis == "col"
  assert p.writeback.accumulator_dtype == dtypes.float and p.writeback.accumulator_count == 8
  assert p.tensor_core.dtype_in == dtypes.char and p.tensor_core.dtype_out == dtypes.int


def test_stable_context_and_representation_only_json():
  a, b = llama_mmq_candidate_plan(), llama_mmq_candidate_plan()
  assert a.identity() == b.identity()
  ctx = a.context()
  assert isinstance(ctx, KernelCandidateContext) and ctx.canonical_identity == a.identity()
  assert ctx.packed_operand_a is a.q4_transform and ctx.packed_operand_b is a.q8_transform
  row = a.to_json()
  assert row["classification"] == "representation_plan_only" and row["emitted"] is False
  assert row["structural_plan"] == llama_q4k_q8_structural_descriptor()


def test_only_full_typed_projection_passes_and_mutations_fail():
  plan = llama_mmq_candidate_plan()
  projected = plan.structural_descriptor()
  assert compare_structures(projected).equivalent
  for dimension, mutate in (
    ("tile_k", lambda x: x + 1),
    ("lds_q4_panel", lambda x: {**x, "bytes": x["bytes"] - 4}),
    ("k_lifecycle", lambda x: {**x, "q8_half_loads_per_step": 1}),
    ("dot_primitive", lambda x: {**x, "wmma_per_scale_group": 1}),
    ("writeback", lambda x: {**x, "owner": "wrong"}),
  ):
    bad = deepcopy(projected)
    bad["dimensions"][dimension] = mutate(bad["dimensions"][dimension])
    result = compare_structures(bad)
    assert not result.equivalent and any(g["dimension"] == dimension for g in result.gaps)
  assert not compare_structures({"dimensions": projected["dimensions"] | {"barriers": None}}).equivalent


def test_typed_plan_rejects_equivalence_claim():
  plan = llama_mmq_candidate_plan()
  import pytest
  with pytest.raises(ValueError, match="cannot claim"):
    replace(plan, emitted=True)
