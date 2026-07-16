"""Format-owned cooperative Q4_K and physical DS4/Q8_1 operand descriptors."""
from __future__ import annotations

from dataclasses import dataclass

from extra.qk.kernel_vocabulary import KernelLDSArenaRegion, KernelLDSRecordComponent, KernelLDSRecordLayout
from tinygrad.codegen.opt.packed_weight import PackedOperandComponent, PackedOperandTransform
from tinygrad.dtype import dtypes

Q4_K_BLOCK_ELEMS, Q8_1_GROUP_ELEMS = 256, 32

# One decoded Q4_K block.  These names also name the five independent input buffers.
Q4_K_COOPERATIVE_RECORD = PackedOperandTransform("q4_k.cooperative.split_buffers.v1", (
  PackedOperandComponent("codes", dtypes.int8, 0, 256, "decoded_q4_codes", 256, 16),
  PackedOperandComponent("d", dtypes.float32, 256, 4, "block_scale", 4, 4),
  PackedOperandComponent("dmin", dtypes.float32, 260, 4, "block_min_scale", 4, 4),
  PackedOperandComponent("sc", dtypes.int8, 264, 8, "eight_group_scales", 8, 4),
  PackedOperandComponent("mn", dtypes.int8, 272, 8, "eight_group_mins", 8, 4),
))

# One physical DS4 epoch: eight Q8_1 groups and their pre-quantization sums.
DS4_Q8_1_COOPERATIVE_RECORD = PackedOperandTransform("q8_1.physical_ds4.split_buffers.v1", (
  PackedOperandComponent("values", dtypes.int8, 0, 256, "physical_ds4_signed_int8", 256, 16),
  PackedOperandComponent("scale", dtypes.float32, 256, 32, "eight_q8_1_scales", 4, 16),
  PackedOperandComponent("weighted_sum", dtypes.float32, 288, 32, "eight_original_fp_sums", 4, 16),
))

CORRECTION_IDENTITY = "scale*d*sc*dot - dmin*mn*weighted_sum"


def _record_layout(transform:PackedOperandTransform, rows:int) -> KernelLDSRecordLayout:
  if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0: raise ValueError("operand rows must be a positive int")
  stride = max(x.end_bytes for x in transform.components)
  return KernelLDSRecordLayout(rows, stride, tuple(KernelLDSRecordComponent(
    x.name, x.dtype, x.offset_bytes, x.size_bytes, x.alignment) for x in transform.components))


@dataclass(frozen=True)
class Q4KQ8CooperativeOperands:
  """CPU-only ABI adapter for one or more cooperative K epochs."""
  k: int
  a_rows: int
  b_rows: int
  k_epoch: int = Q4_K_BLOCK_ELEMS
  arena_bytes: int | None = None
  b_record: PackedOperandTransform = Q4_K_COOPERATIVE_RECORD
  a_record: PackedOperandTransform = DS4_Q8_1_COOPERATIVE_RECORD

  def __post_init__(self) -> None:
    if not isinstance(self.k, int) or isinstance(self.k, bool) or self.k <= 0 or self.k % Q4_K_BLOCK_ELEMS:
      raise ValueError("K must be a positive Q4_K-block-aligned int")
    if self.k_epoch != Q4_K_BLOCK_ELEMS or self.k % self.k_epoch:
      raise ValueError("K epoch must be 256 and divide K")
    _validate_transform(self.b_record, Q4_K_COOPERATIVE_RECORD, "B/Q4")
    _validate_transform(self.a_record, DS4_Q8_1_COOPERATIVE_RECORD, "A/DS4")
    # Constructing layouts validates row counts, component order, gaps, and overlap.
    regions = self.lds_regions
    if self.arena_bytes is not None:
      if not isinstance(self.arena_bytes, int) or isinstance(self.arena_bytes, bool) or self.arena_bytes <= 0:
        raise ValueError("arena_bytes must be a positive int")
      if regions[-1].end > self.arena_bytes: raise ValueError("cooperative operand LDS regions overflow arena_bytes")

  @property
  def epochs(self) -> int: return self.k // self.k_epoch

  @property
  def lds_regions(self) -> tuple[KernelLDSArenaRegion, KernelLDSArenaRegion]:
    b = _record_layout(self.b_record, self.b_rows)
    a = _record_layout(self.a_record, self.a_rows)
    a_base = (b.size_bytes + 15) & -16
    return (KernelLDSArenaRegion("B_q4", 0, b.size_bytes, 16, b),
            KernelLDSArenaRegion("A_ds4", a_base, a_base+a.size_bytes, 16, a))

  def validate_regions(self, regions:tuple[KernelLDSArenaRegion, ...]) -> None:
    if not isinstance(regions, tuple) or len(regions) != 2: raise ValueError("LDS regions must be ordered (B_q4, A_ds4)")
    expected = self.lds_regions
    if tuple(x.name for x in regions) != ("B_q4", "A_ds4"): raise ValueError("LDS regions must be ordered (B_q4, A_ds4)")
    if regions[1].base < regions[0].end: raise ValueError("cooperative operand LDS regions overlap")
    if regions != expected: raise ValueError("LDS regions do not match cooperative operand record layout")


def _validate_transform(actual:PackedOperandTransform, expected:PackedOperandTransform, label:str) -> None:
  if not isinstance(actual, PackedOperandTransform): raise TypeError(f"{label} record must be a PackedOperandTransform")
  an, en = tuple(x.name for x in actual.components), tuple(x.name for x in expected.components)
  if an != en: raise ValueError(f"{label} component order must be {en!r}")
  for got, want in zip(actual.components, expected.components):
    if got.dtype != want.dtype: raise TypeError(f"{label} component {got.name!r} must have dtype {want.dtype.name}")
    if got.identity != want.identity: raise ValueError(f"{label} component {got.name!r} has an invalid physical layout")


__all__ = ["CORRECTION_IDENTITY", "DS4_Q8_1_COOPERATIVE_RECORD", "Q4_K_COOPERATIVE_RECORD", "Q4KQ8CooperativeOperands"]
