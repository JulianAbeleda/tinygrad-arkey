import pytest

from tinygrad.codegen.opt.packed_weight import PackedOperandComponent, PackedOperandTransform, PackedWeightTransform
from tinygrad.dtype import dtypes
from extra.qk.kernel_lds import lds_arena_bytes, lds_component_view, lds_component_views
from tinygrad.uop.ops import KernelCandidateContext, KernelLDSComponentWindow, KernelLDSWindow, KernelTileGeometry


def _geometry():
  return KernelTileGeometry((128, 128, 32), (2, 2), 128, 32,
    (KernelLDSWindow("A", 0, 8192, 64), KernelLDSWindow("B", 8192, 16384, 64)))


def test_legacy_single_packed_weight_is_the_b_transform():
  weight = PackedWeightTransform("Q4_K", 16, 256)
  context = KernelCandidateContext("boltbeam.full_kernel_candidate.v1", "1" * 64, _geometry(), packed_weight=weight)
  assert context.packed_weight is weight and context.packed_operand_a is None and context.packed_operand_b is weight


def test_a_and_b_transform_identity_is_independent():
  a = PackedOperandTransform("oracle-a", (PackedOperandComponent("values", dtypes.int8, 0, 128, alignment=16),))
  b = PackedOperandTransform("oracle-b", (PackedOperandComponent("values", dtypes.uint8, 0, 256, alignment=32),))
  context = KernelCandidateContext("boltbeam.full_kernel_candidate.v1", "2" * 64, _geometry(), packed_operand_a=a, packed_operand_b=b)
  assert context.packed_operand_a is a and context.packed_operand_b is b and context.packed_weight is None


def test_named_typed_components_preserve_layout_stride_alignment_and_identity():
  transform = PackedOperandTransform("oracle-components", (
    PackedOperandComponent("values", dtypes.int8, 0, 256, "k-major", 32, 32),
    PackedOperandComponent("scales", dtypes.half, 256, 32, "row-major", 2, 16),
  ))
  assert transform.component("scales").dtype is dtypes.half
  assert transform.component("values").layout == "k-major" and transform.component("values").stride_bytes == 32
  assert hash(transform) == hash(transform) and transform.identity == transform.identity
  assert transform.to_json() == transform.to_json() and isinstance(transform.to_json()["components"], tuple)
  with pytest.raises(Exception): transform.components += ()  # type: ignore[misc]


@pytest.mark.parametrize("components,match", [
  ((PackedOperandComponent("x", dtypes.uint8, 0, 8), PackedOperandComponent("x", dtypes.uint8, 8, 8)), "duplicate"),
  ((PackedOperandComponent("x", dtypes.uint8, 0, 9), PackedOperandComponent("y", dtypes.uint8, 8, 8)), "overlap"),
])
def test_duplicate_and_overlapping_components_are_rejected(components, match):
  with pytest.raises(ValueError, match=match): PackedOperandTransform("bad", components)


@pytest.mark.parametrize("kwargs,match", [
  ({"name":"", "dtype":dtypes.uint8, "offset_bytes":0, "size_bytes":1}, "name"),
  ({"name":"x", "dtype":dtypes.uint8, "offset_bytes":-1, "size_bytes":1}, "offset"),
  ({"name":"x", "dtype":dtypes.half, "offset_bytes":0, "size_bytes":1}, "whole"),
  ({"name":"x", "dtype":dtypes.uint8, "offset_bytes":1, "size_bytes":1, "alignment":2}, "alignment"),
  ({"name":"x", "dtype":dtypes.uint8, "offset_bytes":0, "size_bytes":1, "stride_bytes":0}, "stride"),
])
def test_invalid_components_are_rejected(kwargs, match):
  with pytest.raises((TypeError, ValueError), match=match): PackedOperandComponent(**kwargs)


def _oracle_shaped_geometry():
  # Generic compiler vocabulary matching an oracle-derived 512 + 18,432 + 38,912 byte arena.
  return KernelTileGeometry((128, 128, 32), (2, 2), 128, 32,
    (KernelLDSWindow("A", 0, 18_944, 64), KernelLDSWindow("B", 18_944, 57_856, 64)),
    (KernelLDSComponentWindow("A", "ids", dtypes.uint16, 0, 512, 16),
     KernelLDSComponentWindow("A", "q8_values", dtypes.int8, 512, 16_896, 32, 32),
     KernelLDSComponentWindow("A", "q8_ds4", dtypes.float, 16_896, 18_944, 16, 16),
     KernelLDSComponentWindow("B", "q4_values", dtypes.uint8, 18_944, 51_712, 32, 32),
     KernelLDSComponentWindow("B", "q4_scales", dtypes.half, 51_712, 57_856, 16, 16)))


def test_oracle_shaped_named_lds_components_preserve_legacy_arena():
  geometry = _oracle_shaped_geometry()
  assert tuple(x.role for x in geometry.lds_windows) == ("A", "B")
  assert geometry.lds_windows[-1].end == geometry.lds_bytes == lds_arena_bytes(geometry) == 57_856
  assert lds_component_view(geometry, "A", "ids").elements == 256
  assert lds_component_view(geometry, "A", "q8_ds4").dtype is dtypes.float
  assert tuple(x.component for x in lds_component_views(geometry, "B")) == ("q4_values", "q4_scales")


@pytest.mark.parametrize("components,match", [
  ((KernelLDSComponentWindow("A", "x", dtypes.uint8, 0, 32, 16),
    KernelLDSComponentWindow("A", "x", dtypes.uint8, 32, 64, 16)), "duplicate"),
  ((KernelLDSComponentWindow("A", "x", dtypes.uint8, 0, 32, 16),
    KernelLDSComponentWindow("A", "y", dtypes.uint8, 16, 48, 16)), "overlap"),
  ((KernelLDSComponentWindow("A", "x", dtypes.uint8, 0, 32, 16),
    KernelLDSComponentWindow("B", "y", dtypes.uint8, 16, 48, 16)), "bounds"),
])
def test_geometry_rejects_duplicate_overlap_and_out_of_bounds_components(components, match):
  with pytest.raises(ValueError, match=match):
    KernelTileGeometry((128, 128, 32), (2, 2), 128, 32,
      (KernelLDSWindow("A", 0, 64, 16), KernelLDSWindow("B", 64, 128, 16)), components)


@pytest.mark.parametrize("make,match", [
  (lambda: KernelLDSComponentWindow("A", "x", dtypes.uint8, 1, 16, 16), "alignment"),
  (lambda: KernelLDSComponentWindow("A", "x", dtypes.float, 0, 10, 4), "whole"),
  (lambda: KernelLDSComponentWindow("A", "x", dtypes.uint8, 0, 16, 3), "power of two"),
  (lambda: KernelLDSComponentWindow("A", "x", dtypes.half, 0, 16, 16, 8), "stride"),
])
def test_invalid_typed_lds_component_is_rejected(make, match):
  with pytest.raises((TypeError, ValueError), match=match): make()
