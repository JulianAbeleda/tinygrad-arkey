from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, TypeAlias

from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp

from tinygrad.llm.qk_layout import (Q4K_WORDS_PER_BLOCK, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS,
                                    Q6K_HALFWORDS_PER_BLOCK, Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMS)

PackedFormat: TypeAlias = Literal["Q4_K", "Q6_K"]
ScalarIndex: TypeAlias = int | UOp
LoadSource: TypeAlias = UOp | Callable[[ScalarIndex], UOp]


@dataclass(frozen=True)
class PackedWeightAddress:
  """Addresses touched while producing one logical weight (all offsets are from the tensor base)."""
  block: ScalarIndex
  block_byte: ScalarIndex
  payload_byte: ScalarIndex
  scale_byte: ScalarIndex
  d_byte: ScalarIndex
  min_scale_byte: ScalarIndex | None = None
  auxiliary_bytes: tuple[ScalarIndex, ...] = ()

  def byte_offsets(self) -> tuple[ScalarIndex, ...]:
    return tuple(x for x in (self.payload_byte, self.scale_byte, self.d_byte, self.min_scale_byte) if x is not None) + self.auxiliary_bytes

  def unit_offsets(self, width:int) -> tuple[ScalarIndex, ...]:
    if width not in (1, 2, 4): raise ValueError(f"packed unit width must be 1, 2, or 4 bytes, got {width}")
    return tuple(x // width for x in self.byte_offsets())


@dataclass(frozen=True)
class PackedWeightTransform:
  """Model-independent GGML K-quant layout and scalar fp16 producer.

  ``k`` is the row stride in logical elements. Q4_K sources are uint32 words and
  Q6_K sources are uint16 halfwords, matching the aligned GGUF loaders.
  """
  quant_format: PackedFormat
  rows: int
  k: int
  block_elems: int = 256
  block_bytes: int | None = None

  def __post_init__(self) -> None:
    if self.quant_format not in ("Q4_K", "Q6_K"):
      raise ValueError(f"quant_format must be Q4_K or Q6_K, got {self.quant_format!r}")
    canonical_elems = Q4_K_BLOCK_ELEMS if self.quant_format == "Q4_K" else Q6_K_BLOCK_ELEMS
    canonical_bytes = Q4_K_BLOCK_BYTES if self.quant_format == "Q4_K" else Q6_K_BLOCK_BYTES
    if self.block_elems != canonical_elems:
      raise ValueError(f"{self.quant_format} block_elems must be {canonical_elems}, got {self.block_elems}")
    if self.block_bytes is None: object.__setattr__(self, "block_bytes", canonical_bytes)
    elif self.block_bytes != canonical_bytes:
      raise ValueError(f"{self.quant_format} block_bytes must be {canonical_bytes}, got {self.block_bytes}")
    if not isinstance(self.rows, int) or self.rows <= 0: raise ValueError(f"rows must be a positive integer, got {self.rows!r}")
    if not isinstance(self.k, int) or self.k <= 0: raise ValueError(f"k must be a positive integer, got {self.k!r}")
    if self.k % canonical_elems: raise ValueError(f"k must be {self.quant_format} block aligned ({canonical_elems}), got {self.k}")

  @property
  def blocks_per_row(self) -> int: return self.k // self.block_elems

  @property
  def storage_dtype(self): return dtypes.uint32 if self.quant_format == "Q4_K" else dtypes.uint16

  @property
  def storage_width(self) -> int: return 4 if self.quant_format == "Q4_K" else 2

  @property
  def units_per_block(self) -> int:
    return Q4K_WORDS_PER_BLOCK if self.quant_format == "Q4_K" else Q6K_HALFWORDS_PER_BLOCK

  @property
  def packed_bytes(self) -> int: return self.rows * self.blocks_per_row * int(self.block_bytes)

  def _check_coords(self, row:ScalarIndex, k:ScalarIndex) -> None:
    if isinstance(k, int) and not 0 <= k < self.k: raise IndexError(f"k={k} is outside [0, {self.k})")
    if not isinstance(k, (int, UOp)): raise TypeError("logical k must be an integer or UOp")
    if isinstance(row, int) and not 0 <= row < self.rows: raise IndexError(f"row={row} is outside [0, {self.rows})")
    if not isinstance(row, (int, UOp)): raise TypeError("row must be an integer or UOp")

  def address(self, row:ScalarIndex, k:int) -> PackedWeightAddress:
    self._check_coords(row, k)
    block = row * self.blocks_per_row + k // self.block_elems
    base = block * int(self.block_bytes)
    if self.quant_format == "Q4_K":
      group, pos = (k % 256) // 32, k % 32
      payload = base + 16 + (group // 2) * 32 + pos
      # Groups 4..7 combine low bits from bytes 4..11 with high bits from bytes 12..15.
      low_group = group if group < 4 else group - 4
      scale, minimum = base + 4 + low_group, base + 8 + low_group
      high = () if group < 4 else (base + 12 + group - 4,)
      return PackedWeightAddress(block, base, payload, scale, base, minimum, high)
    group, pos = (k % 256) // 16, k % 16
    half, pgroup = group // 8, group % 8
    ql = base + half * 64 + (pgroup % 4) * 16 + pos
    # payload_byte names ql; qh is derivable and included through min_scale_byte.
    qh = base + 128 + half * 32 + (pgroup % 2) * 16 + pos
    return PackedWeightAddress(block, base, ql, base + 192 + group, base + 208, qh)

  @staticmethod
  def _load(source:LoadSource, index:ScalarIndex) -> UOp:
    return source(index) if callable(source) else source[index]

  def dequant(self, source:LoadSource, row:ScalarIndex, k:ScalarIndex) -> UOp:
    """Return a pure scalar UOp expression, rounded to fp16, for ``weight[row, k]``."""
    self._check_coords(row, k)
    block = row * self.blocks_per_row + k // self.block_elems
    unit_base = block * self.units_per_block
    if self.quant_format == "Q4_K": return self._dequant_q4(source, unit_base, k % 256)
    return self._dequant_q6(source, unit_base, k % 256)

  def _dequant_q4(self, source:LoadSource, base:ScalarIndex, within:ScalarIndex) -> UOp:
    group, pos = within // 32, within % 32
    word0 = self._load(source, base)
    d = word0.bitwise_and(0xffff).cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)
    dmin = word0.rshift(16).bitwise_and(0xffff).cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)
    def byte(i:ScalarIndex) -> UOp: return self._load(source, base + 1 + i//4).rshift((i%4)*8).bitwise_and(0xff)
    if isinstance(group, int):
      if group < 4: scale, minimum = byte(group).bitwise_and(63), byte(4+group).bitwise_and(63)
      else:
        high = byte(8+group-4)
        scale = high.bitwise_and(15).bitwise_or(byte(group-4).rshift(6).lshift(4))
        minimum = high.rshift(4).bitwise_or(byte(4+group-4).rshift(6).lshift(4))
    else:
      # WHERE is branchless on the GPU. Clamp the high-half group before constructing loads so groups 0..3 never
      # speculatively address byte(group-4) before the packed block.
      high_group = (group < 4).where(0, group-4)
      low_scale, low_minimum = byte(group).bitwise_and(63), byte(4+group).bitwise_and(63)
      high = byte(8+high_group)
      high_scale = high.bitwise_and(15).bitwise_or(byte(high_group).rshift(6).lshift(4))
      high_minimum = high.rshift(4).bitwise_or(byte(4+high_group).rshift(6).lshift(4))
      scale, minimum = (group < 4).where(low_scale, high_scale), (group < 4).where(low_minimum, high_minimum)
    qword = self._load(source, base + 4 + (group//2)*8 + pos//4)
    q = qword.rshift((pos%4)*8 + (group%2)*4).bitwise_and(15)
    return (d * scale.cast(dtypes.float32) * q.cast(dtypes.float32) - dmin * minimum.cast(dtypes.float32)).cast(dtypes.float16)

  def _dequant_q6(self, source:LoadSource, base:ScalarIndex, within:ScalarIndex) -> UOp:
    group, pos = within // 16, within % 16
    def byte(i:ScalarIndex) -> UOp: return self._load(source, base + i//2).rshift((i%2)*8).bitwise_and(0xff)
    half, pgroup = group // 8, group % 8
    ql_shift = 4 if isinstance(pgroup, int) and pgroup >= 4 else 0 if isinstance(pgroup, int) else (pgroup >= 4).where(4, 0)
    ql = byte(half*64 + (pgroup%4)*16 + pos).rshift(ql_shift).bitwise_and(15)
    qh = byte(128 + half*32 + (pgroup%2)*16 + pos).rshift((pgroup//2)*2).bitwise_and(3).lshift(4)
    q = ql.bitwise_or(qh).cast(dtypes.float32) - UOp.const(dtypes.float32, 32.0)
    scale = byte(192+group).cast(dtypes.uint8).bitcast(dtypes.int8).cast(dtypes.float32)
    d = self._load(source, base+104).cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)
    return (d * q * scale).cast(dtypes.float16)

  def to_json(self) -> dict[str, int | str]:
    return {"quant_format": self.quant_format, "rows": self.rows, "k": self.k, "block_elems": self.block_elems,
            "block_bytes": int(self.block_bytes), "storage_dtype": self.storage_dtype.name,
            "storage_width": self.storage_width, "units_per_block": self.units_per_block, "packed_bytes": self.packed_bytes}

  @classmethod
  def from_json(cls, obj:dict[str, object]) -> PackedWeightTransform:
    required = {"quant_format", "rows", "k"}
    if missing := required - obj.keys(): raise ValueError(f"missing packed-weight fields: {', '.join(sorted(missing))}")
    allowed = required | {"block_elems", "block_bytes", "storage_dtype", "storage_width", "units_per_block", "packed_bytes"}
    if extra := obj.keys() - allowed: raise ValueError(f"unknown packed-weight fields: {', '.join(sorted(extra))}")
    ret = cls(obj["quant_format"], obj["rows"], obj["k"], obj.get("block_elems", 256), obj.get("block_bytes"))  # type: ignore[arg-type]
    for key in ("storage_dtype", "storage_width", "units_per_block", "packed_bytes"):
      if key in obj and obj[key] != ret.to_json()[key]: raise ValueError(f"{key} does not match {ret.quant_format} geometry")
    return ret


PackedWeightDescriptor = PackedWeightTransform
