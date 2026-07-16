import pytest

from extra.qk.kernel_vocabulary import KernelLDSArenaRegion
from extra.qk.q4k_q8_cooperative_operands import (CORRECTION_IDENTITY, DS4_Q8_1_COOPERATIVE_RECORD,
  Q4_K_COOPERATIVE_RECORD, Q4KQ8CooperativeOperands)
from tinygrad.codegen.opt.packed_weight import PackedOperandComponent, PackedOperandTransform
from tinygrad.dtype import dtypes


def test_cooperative_operand_layout_and_identity():
  spec = Q4KQ8CooperativeOperands(512, a_rows=3, b_rows=2)
  assert spec.epochs == 2
  assert tuple(x.name for x in Q4_K_COOPERATIVE_RECORD.components) == ("codes", "d", "dmin", "sc", "mn")
  assert tuple(x.name for x in DS4_Q8_1_COOPERATIVE_RECORD.components) == ("values", "scale", "weighted_sum")
  b, a = spec.lds_regions
  assert (b.base, b.end, b.records.stride_bytes) == (0, 560, 280)
  assert (a.base, a.end, a.records.stride_bytes) == (560, 1520, 320)
  assert b.row_slice(1, "mn") == (552, 560)
  assert a.row_slice(2, "weighted_sum") == (1488, 1520)
  assert CORRECTION_IDENTITY == "scale*d*sc*dot - dmin*mn*weighted_sum"
  spec.validate_regions((b, a))


def _replace_component(transform, index, component):
  parts = list(transform.components); parts[index] = component
  return PackedOperandTransform(transform.name+".bad", tuple(parts))


def test_rejects_bad_component_order_and_dtype():
  c = Q4_K_COOPERATIVE_RECORD.components
  reordered = PackedOperandTransform("bad.order", (c[1], c[0], *c[2:]))
  with pytest.raises(ValueError, match="order"): Q4KQ8CooperativeOperands(256, 1, 1, b_record=reordered)
  bad_dtype = _replace_component(DS4_Q8_1_COOPERATIVE_RECORD, 0,
    PackedOperandComponent("values", dtypes.uint8, 0, 256, "physical_ds4_signed_int8", 256, 16))
  with pytest.raises(TypeError, match="dtype"): Q4KQ8CooperativeOperands(256, 1, 1, a_record=bad_dtype)


@pytest.mark.parametrize("kwargs", ({"k": 255}, {"k": 512, "k_epoch": 128}, {"k": 256, "k_epoch": 512}))
def test_rejects_bad_k(kwargs):
  with pytest.raises(ValueError, match="K"): Q4KQ8CooperativeOperands(a_rows=1, b_rows=1, **kwargs)


def test_rejects_bad_region_order_overlap_and_overflow():
  spec = Q4KQ8CooperativeOperands(256, 1, 1)
  b, a = spec.lds_regions
  with pytest.raises(ValueError, match="ordered"): spec.validate_regions((a, b))
  overlapping = KernelLDSArenaRegion("A_ds4", b.end-8, b.end-8+320)
  with pytest.raises(ValueError, match="overlap"): spec.validate_regions((b, overlapping))
  with pytest.raises(ValueError, match="overflow"): Q4KQ8CooperativeOperands(256, 1, 1, arena_bytes=a.end-1)
