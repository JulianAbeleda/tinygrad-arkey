import pytest

from tinygrad.codegen.opt.packed_weight import PackedOperandComponent, PackedOperandRecordTransform, PackedOperandTransform
from tinygrad.dtype import dtypes
from extra.qk.kernel_vocabulary import (KernelCandidateContext, KernelLDSArenaRegion, KernelLDSRecordComponent, KernelLDSRecordLayout,
                                        KernelLDSWindow, KernelTileGeometry)


def _geometry():
  q8 = KernelLDSRecordLayout(128, 144, (
    KernelLDSRecordComponent("ds", dtypes.half, 0, 16, 16),
    KernelLDSRecordComponent("qs", dtypes.int8, 16, 128, 16)))
  q4 = KernelLDSRecordLayout(128, 304, (
    KernelLDSRecordComponent("qs", dtypes.uint8, 0, 256, 16),
    KernelLDSRecordComponent("dm", dtypes.half, 256, 32, 16),
    KernelLDSRecordComponent("padding", dtypes.uint8, 288, 16, 16)))
  return KernelTileGeometry((128, 128, 32), (2, 2), 128, 32,
    (KernelLDSWindow("A", 512, 18_944, 144), KernelLDSWindow("B", 18_944, 57_856, 304)), (),
    (KernelLDSArenaRegion("ids", 0, 512), KernelLDSArenaRegion("q8", 512, 18_944, records=q8),
     KernelLDSArenaRegion("q4", 18_944, 57_856, records=q4)))


def test_exact_arena_and_actual_interleaved_row_slices_are_nonoverlapping():
  geometry = _geometry()
  assert geometry.lds_bytes == 57_856
  assert [(x.name, x.base, x.end) for x in geometry.lds_regions] == [("ids", 0, 512), ("q8", 512, 18_944), ("q4", 18_944, 57_856)]
  q8, q4 = geometry.lds_region("q8"), geometry.lds_region("q4")
  assert q8.row_slice(0, "ds") == (512, 528) and q8.row_slice(0, "qs") == (528, 656)
  assert q8.row_slice(127) == (18_800, 18_944)
  assert q4.row_slice(0, "qs") == (18_944, 19_200) and q4.row_slice(0, "dm") == (19_200, 19_232)
  assert q4.row_slice(0, "padding") == (19_232, 19_248) and q4.row_slice(127) == (57_552, 57_856)
  slices = [region.row_slice(row, component.component) for region in (q8, q4) for row in range(region.records.rows)
            for component in region.records.components]
  assert all(left[1] <= right[0] for left, right in zip(sorted(slices), sorted(slices)[1:]))
  assert sum(end-start for start, end in slices) == (18_944-512) + (57_856-18_944)


def test_record_component_dtype_alignment_and_bounds_are_owned_by_layout():
  geometry = _geometry()
  assert geometry.lds_region("q8").records.component("ds").dtype is dtypes.half
  assert geometry.lds_region("q4").records.component("dm").alignment == 16
  with pytest.raises(IndexError, match="out of bounds"): geometry.lds_region("q8").row_slice(128)


@pytest.mark.parametrize("make, match", [
  (lambda: KernelLDSRecordLayout(1, 16, (KernelLDSRecordComponent("x", dtypes.uint8, 0, 12, 4),
                                        KernelLDSRecordComponent("y", dtypes.uint8, 8, 8, 4))), "overlap"),
  (lambda: KernelLDSRecordLayout(1, 16, (KernelLDSRecordComponent("x", dtypes.uint8, 0, 8, 4),
                                        KernelLDSRecordComponent("y", dtypes.uint8, 12, 4, 4))), "gap"),
  (lambda: KernelLDSRecordLayout(1, 20, (KernelLDSRecordComponent("x", dtypes.uint8, 0, 16, 4),)), "stride"),
])
def test_record_layout_rejects_overlap_gap_and_wrong_stride(make, match):
  with pytest.raises(ValueError, match=match): make()


def test_arena_rejects_overlap_gap_wrong_window_and_record_extent():
  with pytest.raises(ValueError, match="record layout size"):
    KernelLDSArenaRegion("bad", 0, 32, records=KernelLDSRecordLayout(1, 16, (KernelLDSRecordComponent("x", dtypes.uint8, 0, 16, 1),)))
  good = _geometry()
  for regions, match in (((KernelLDSArenaRegion("ids", 0, 512), KernelLDSArenaRegion("q8", 528, 18_944)), "gap"),
                         ((KernelLDSArenaRegion("ids", 0, 528), KernelLDSArenaRegion("q8", 512, 18_944)), "overlap")):
    with pytest.raises(ValueError, match=match):
      KernelTileGeometry(good.tile, good.waves, good.threads, good.wave_size, good.lds_windows, (), regions)


def test_source_and_produced_record_transforms_are_distinct_and_generic():
  source = PackedOperandTransform("packed-source", (PackedOperandComponent("record", dtypes.uint8, 0, 144, alignment=16),))
  produced = PackedOperandTransform("decoded-lds", (
    PackedOperandComponent("qs", dtypes.uint8, 0, 256, "record", 304, 16),
    PackedOperandComponent("dm", dtypes.half, 256, 32, "record", 304, 16),
    PackedOperandComponent("padding", dtypes.uint8, 288, 16, "record", 304, 16)))
  transform = PackedOperandRecordTransform("decode-record", source, produced)
  assert transform.source is source and transform.produced is produced and transform.identity[1] != transform.identity[2]
  context = KernelCandidateContext("boltbeam.full_kernel_candidate.v1", "a"*64, _geometry(), packed_operand_b=transform)
  assert context.packed_operand_b is transform and context.packed_weight is None
