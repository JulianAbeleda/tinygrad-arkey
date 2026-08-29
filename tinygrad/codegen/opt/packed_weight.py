from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, TypeAlias

from tinygrad.dtype import DType, dtypes
from tinygrad.uop.ops import Ops, UOp

from tinygrad.llm.qk_layout import Q4K_WORDS_PER_BLOCK, Q6K_HALFWORDS_PER_BLOCK
from tinygrad.llm.qk_layout import Q4_K, Q6_K, QUANT_FORMATS, QuantFormat

ScalarIndex: TypeAlias = int | UOp
LoadSource: TypeAlias = UOp | Callable[[ScalarIndex], UOp]


def _check_k32_fragment_boundary(k_base:ScalarIndex, width:int, *, label:str) -> None:
  """Fail closed unless every represented fragment stays inside one K32 metadata group."""
  within = k_base % 32
  maximum = within if isinstance(within, int) else within.simplify().vmax
  if maximum + width > 32:
    raise ValueError(f"{label} fragment crosses a K32 metadata boundary: k_base%32 vmax={maximum}, width={width}")


@dataclass(frozen=True)
class PackedOperandComponent:
  """One named, typed byte range produced or consumed by a packed operand transform."""
  name: str
  dtype: DType
  offset_bytes: int
  size_bytes: int
  layout: str = "contiguous"
  stride_bytes: int | None = None
  alignment: int = 1

  def __post_init__(self) -> None:
    if not isinstance(self.name, str) or not self.name or not self.name.isidentifier():
      raise ValueError(f"packed component name must be a non-empty identifier, got {self.name!r}")
    if not isinstance(self.dtype, DType): raise TypeError("packed component dtype must be a DType")
    if not isinstance(self.layout, str) or not self.layout: raise ValueError("packed component layout must be a non-empty string")
    if not isinstance(self.offset_bytes, int) or isinstance(self.offset_bytes, bool) or self.offset_bytes < 0:
      raise ValueError("packed component offset_bytes must be non-negative")
    for field, value in (("size_bytes", self.size_bytes), ("alignment", self.alignment)):
      if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"packed component {field} must be positive")
    if self.stride_bytes is not None and (not isinstance(self.stride_bytes, int) or isinstance(self.stride_bytes, bool) or self.stride_bytes <= 0):
      raise ValueError("packed component stride_bytes must be positive when present")
    if self.offset_bytes % self.alignment: raise ValueError("packed component offset_bytes must satisfy alignment")
    if self.size_bytes % self.dtype.itemsize: raise ValueError("packed component size_bytes must contain whole dtype values")

  @property
  def end_bytes(self) -> int: return self.offset_bytes + self.size_bytes

  @property
  def identity(self) -> tuple[str, str, int, int, str, int|None, int]:
    return (self.name, self.dtype.name, self.offset_bytes, self.size_bytes, self.layout, self.stride_bytes, self.alignment)

  def to_json(self) -> dict[str, str|int|None]:
    return dict(zip(("name", "dtype", "offset_bytes", "size_bytes", "layout", "stride_bytes", "alignment"), self.identity))


@dataclass(frozen=True)
class PackedOperandTransform:
  """Generic packed-operand vocabulary; lowering is deliberately owned elsewhere."""
  name: str
  components: tuple[PackedOperandComponent, ...]

  def __post_init__(self) -> None:
    if not isinstance(self.name, str) or not self.name: raise ValueError("packed transform name must be a non-empty string")
    if not isinstance(self.components, tuple) or not self.components or not all(isinstance(x, PackedOperandComponent) for x in self.components):
      raise TypeError("packed transform components must be a non-empty tuple of PackedOperandComponent values")
    names: set[str] = set()
    for component in self.components:
      if component.name in names: raise ValueError(f"duplicate packed component name {component.name!r}")
      names.add(component.name)
    ordered = sorted(self.components, key=lambda x: (x.offset_bytes, x.end_bytes))
    for left, right in zip(ordered, ordered[1:]):
      if right.offset_bytes < left.end_bytes:
        raise ValueError(f"packed components {left.name!r} and {right.name!r} overlap")

  @property
  def identity(self) -> tuple[str, tuple[tuple[str, str, int, int, str, int|None, int], ...]]:
    return self.name, tuple(x.identity for x in self.components)

  def component(self, name:str) -> PackedOperandComponent:
    try: return next(x for x in self.components if x.name == name)
    except StopIteration as exc: raise KeyError(name) from exc

  def to_json(self) -> dict[str, object]:
    return {"name": self.name, "components": tuple(x.to_json() for x in self.components)}


@dataclass(frozen=True)
class PackedOperandRecordTransform:
  """Generic source-record to produced-record transform vocabulary."""
  name: str
  source: PackedOperandTransform
  produced: PackedOperandTransform

  def __post_init__(self) -> None:
    if not isinstance(self.name, str) or not self.name: raise ValueError("record transform name must be a non-empty string")
    if not isinstance(self.source, PackedOperandTransform) or not isinstance(self.produced, PackedOperandTransform):
      raise TypeError("record transform source and produced must be PackedOperandTransform values")

  @property
  def identity(self) -> tuple[str, tuple, tuple]: return self.name, self.source.identity, self.produced.identity

  def to_json(self) -> dict[str, object]:
    return {"name": self.name, "source": self.source.to_json(), "produced": self.produced.to_json()}


@dataclass(frozen=True)
class PackedWeightTile:
  """A decoded packed-weight tile and the logical interval it represents."""
  value: UOp
  row: ScalarIndex
  k_base: ScalarIndex
  width: Literal[8, 16]

  def __post_init__(self) -> None:
    if self.width not in (8, 16): raise ValueError(f"packed tile width must be 8 or 16, got {self.width}")
    if self.value.dtype != dtypes.half.vec(self.width):
      raise TypeError(f"packed tile value must be half.vec({self.width}), got {self.value.dtype}")


@dataclass(frozen=True)
class PackedInt8Fragment:
  """One direct-packed int8 tensor-core fragment with logical row/K ownership.

  Unlike :class:`PackedWeightTile`, this carrier is not a dequantized weight.
  It contains the unsigned Q4 code (0..15) in a signed-int8 carrier, exactly as
  consumed by NVIDIA's ``s8 x s8 -> s32`` IMMA instruction.  ``row`` and
  ``k_base`` deliberately survive on the typed value: a tensor-core range
  permutation may choose physical lanes only *after* these logical coordinates
  have selected the Q4 low/high nibble.
  """
  value: UOp
  row: ScalarIndex
  k_base: ScalarIndex
  width: Literal[8, 16]
  logical_shape: tuple[int, int]
  block_scale: UOp
  block_minimum: UOp
  group_scale: UOp
  group_minimum: UOp
  logical_group: ScalarIndex
  abi: str = "q4_k.logical_nk_to_s8.v1"

  def __post_init__(self) -> None:
    if self.width not in (8, 16): raise ValueError(f"packed int8 fragment width must be 8 or 16, got {self.width}")
    if self.value.dtype != dtypes.char.vec(self.width):
      raise TypeError(f"packed int8 fragment value must be char.vec({self.width}), got {self.value.dtype}")
    if (not isinstance(self.logical_shape, tuple) or len(self.logical_shape) != 2 or
        not all(isinstance(x, int) and x > 0 for x in self.logical_shape)):
      raise ValueError("packed int8 fragment logical_shape must be a positive (N,K) pair")
    if self.block_scale.dtype != dtypes.float or self.block_minimum.dtype != dtypes.float:
      raise TypeError("packed int8 fragment block scale/minimum must be float32")
    if self.group_scale.dtype.scalar() not in (dtypes.uint, dtypes.int) or self.group_minimum.dtype.scalar() not in (dtypes.uint, dtypes.int):
      raise TypeError("packed int8 fragment group scale/minimum must be integer scalars")


@dataclass(frozen=True)
class Q4KInt8FragmentProvider:
  """Typed logical ``(N,K) -> int8`` provider for canonical Q4_K storage.

  This is intentionally narrower than arbitrary ALU-DAG substitution.  The
  provider owns one exact :class:`PackedWeightTransform`, accepts logical row
  and K coordinates, and only then extracts the corresponding nibble.  That
  ordering is the contract which prevents the observed postrange bug where a
  permuted K range changed low/high-nibble parity.

  The produced values are *codes*, not complete affine Q4_K weights.  Q4_K
  scale/min correction is a separate typed accumulator contract; callers must
  not treat a raw IMMA reduction across multiple 32-value groups as a complete
  Q4_K projection.
  """
  transform: "PackedWeightTransform"
  abi: str = "q4_k.logical_nk_to_s8.v1"

  def __post_init__(self) -> None:
    if not isinstance(self.transform, PackedWeightTransform):
      raise TypeError("Q4_K int8 fragment provider requires a PackedWeightTransform")
    if self.transform.quant_format is not Q4_K:
      raise ValueError("Q4_K int8 fragment provider requires canonical Q4_K storage")

  @property
  def logical_shape(self) -> tuple[int, int]: return self.transform.rows, self.transform.k

  def _code(self, source:LoadSource, row:ScalarIndex, k:ScalarIndex) -> UOp:
    self.transform._check_coords(row, k)
    within = k % self.transform.block_elems
    group, pos = within // 32, within % 32
    block = row * self.transform.blocks_per_row + k // self.transform.block_elems
    base = block * self.transform.units_per_block
    qword = self.transform._load(source, base + 4 + (group//2)*8 + pos//4)
    # 0..15 is representable by signed int8.  Cast only after logical group
    # parity has selected the low/high nibble.
    return qword.rshift((pos%4)*8 + (group%2)*4).bitwise_and(15).cast(dtypes.char)

  def fragment(self, source:LoadSource, row:ScalarIndex, k_base:ScalarIndex,
               width:Literal[8, 16]=16) -> PackedInt8Fragment:
    if width not in (8, 16): raise ValueError(f"packed int8 fragment width must be 8 or 16, got {width}")
    self.transform._check_coords(row, k_base)
    if isinstance(k_base, int) and k_base + width > self.transform.k:
      raise IndexError(f"fragment [{k_base}, {k_base+width}) is outside [0, {self.transform.k})")
    _check_k32_fragment_boundary(k_base, width, label="Q4_K")
    requested: list[ScalarIndex] = []
    def collect(index:ScalarIndex) -> UOp:
      requested.append(index.simplify() if isinstance(index, UOp) else index)
      return UOp.const(self.transform.storage_dtype, 0)
    for i in range(width): self._code(collect, row, k_base+i)
    block = row * self.transform.blocks_per_row + k_base // self.transform.block_elems
    unit_base, logical_group = block*self.transform.units_per_block, (k_base%self.transform.block_elems)//32
    self.transform._q4_params(collect, unit_base, logical_group)
    cached_load = self.transform._tile_loads(source, tuple(requested))
    lanes = tuple(self._code(cached_load, row, k_base+i) for i in range(width))
    block_scale, block_minimum, group_scale, group_minimum = self.transform._q4_params(cached_load, unit_base, logical_group)
    return PackedInt8Fragment(UOp(Ops.STACK, dtypes.char.vec(width), lanes), row, k_base, width, self.logical_shape,
      block_scale, block_minimum, group_scale, group_minimum, logical_group, self.abi)

  def metadata(self, source:LoadSource, row:ScalarIndex, k_base:ScalarIndex) -> tuple[UOp, UOp, UOp, UOp, ScalarIndex]:
    """Return the affine Q4_K metadata owned by the logical K32 group.

    This is the accumulator-side half of :meth:`fragment`: callers can obtain
    the correction factors without constructing a throwaway code vector.  The
    logical ``k_base`` is interpreted before any tensor-core permutation, just
    like fragment nibble selection.
    """
    self.transform._check_coords(row, k_base)
    block = row * self.transform.blocks_per_row + k_base // self.transform.block_elems
    logical_group = (k_base % self.transform.block_elems) // 32
    d, dmin, scale, minimum = self.transform._q4_params(source, block*self.transform.units_per_block, logical_group)
    return d, dmin, scale, minimum, logical_group

  @property
  def identity(self) -> tuple[str, tuple[int, int], tuple[tuple[str, object], ...]]:
    return self.abi, self.logical_shape, tuple(sorted(self.transform.to_json().items()))


@dataclass(frozen=True)
class Q6KInt8Fragment:
  """One signed Q6_K code fragment with exact K16 scale ownership."""
  value: UOp
  row: ScalarIndex
  k_base: ScalarIndex
  width: Literal[8, 16]
  logical_shape: tuple[int, int]
  block_scale: UOp
  subgroup_scale: UOp
  logical_subgroup: ScalarIndex
  abi: str = "q6_k.logical_nk_to_s8_k16.v1"

  def __post_init__(self) -> None:
    if self.width not in (8, 16) or self.value.dtype != dtypes.char.vec(self.width):
      raise TypeError("Q6_K int8 fragment must be a char vector of width 8 or 16")
    if self.block_scale.dtype != dtypes.float:
      raise TypeError("Q6_K block scale must be float32")
    if self.subgroup_scale.dtype.scalar() not in (dtypes.char, dtypes.int):
      raise TypeError("Q6_K K16 subgroup scale must be a signed integer scalar")


@dataclass(frozen=True)
class Q6KInt8FragmentProvider:
  """Typed logical ``(N,K) -> signed int8`` provider for canonical Q6_K.

  Q6_K's scale ownership is K16, narrower than NVIDIA signed IMMA's K32.
  Fragments therefore fail closed at K16 boundaries and retain their exact
  subgroup identity.  The paired-subtotal accumulator below is the only
  admissible way to combine two such halves into one K32 tensor-core step.
  """
  transform: "PackedWeightTransform"
  abi: str = "q6_k.logical_nk_to_s8_k16.v1"

  def __post_init__(self) -> None:
    if not isinstance(self.transform, PackedWeightTransform):
      raise TypeError("Q6_K int8 fragment provider requires a PackedWeightTransform")
    if self.transform.quant_format is not Q6_K:
      raise ValueError("Q6_K int8 fragment provider requires canonical Q6_K storage")

  @property
  def logical_shape(self) -> tuple[int, int]: return self.transform.rows, self.transform.k

  def _code(self, source:LoadSource, row:ScalarIndex, k:ScalarIndex) -> UOp:
    self.transform._check_coords(row, k)
    within = k % self.transform.block_elems
    subgroup, pos = within // 16, within % 16
    half, payload_group = subgroup // 8, subgroup % 8
    block = row*self.transform.blocks_per_row+k//self.transform.block_elems
    base = block*self.transform.units_per_block
    byte = self.transform._byte_loader(source, base, 2)
    ql_shift = 4 if isinstance(payload_group, int) and payload_group >= 4 else 0 if isinstance(payload_group, int) else \
      (payload_group >= 4).where(4, 0)
    ql = byte(half*64+(payload_group%4)*16+pos).rshift(ql_shift).bitwise_and(15)
    qh = byte(128+half*32+(payload_group%2)*16+pos).rshift((payload_group//2)*2).bitwise_and(3).lshift(4)
    return (ql.bitwise_or(qh).cast(dtypes.int)-32).cast(dtypes.char)

  def metadata(self, source:LoadSource, row:ScalarIndex, k_base:ScalarIndex) -> tuple[UOp, UOp, ScalarIndex]:
    self.transform._check_coords(row, k_base)
    block = row*self.transform.blocks_per_row+k_base//self.transform.block_elems
    base = block*self.transform.units_per_block
    subgroup = (k_base%self.transform.block_elems)//16
    byte = self.transform._byte_loader(source, base, 2)
    scale = byte(192+subgroup).cast(dtypes.uint8).bitcast(dtypes.char).cast(dtypes.int)
    d = self.transform._load(source, base+104).cast(dtypes.uint16).bitcast(dtypes.half).cast(dtypes.float)
    return d, scale, subgroup

  def k32_metadata(self, source:LoadSource, row:ScalarIndex, k_base:ScalarIndex) -> tuple[UOp, UOp, UOp, ScalarIndex]:
    """Return D and both independently-owned K16 scales for one K32 step."""
    within = k_base % 32
    if (within if isinstance(within, int) else within.simplify().vmax) != 0:
      raise ValueError("Q6_K paired metadata requires a K32-aligned base")
    d0, scale0, subgroup0 = self.metadata(source, row, k_base)
    d1, scale1, subgroup1 = self.metadata(source, row, k_base+16)
    if isinstance(subgroup0, int) and isinstance(subgroup1, int) and subgroup1 != subgroup0+1:
      raise ValueError("Q6_K paired metadata does not own adjacent K16 subgroups")
    # Both halves are in one K32 and therefore one K256 block.
    return d0, scale0, scale1, subgroup0

  def fragment(self, source:LoadSource, row:ScalarIndex, k_base:ScalarIndex,
               width:Literal[8, 16]=16) -> Q6KInt8Fragment:
    if width not in (8, 16): raise ValueError(f"Q6_K int8 fragment width must be 8 or 16, got {width}")
    self.transform._check_coords(row, k_base)
    if isinstance(k_base, int) and k_base+width > self.transform.k:
      raise IndexError(f"fragment [{k_base}, {k_base+width}) is outside [0, {self.transform.k})")
    within = k_base % 16
    maximum = within if isinstance(within, int) else within.simplify().vmax
    if maximum+width > 16:
      raise ValueError(f"Q6_K fragment crosses a K16 scale boundary: k_base%16 vmax={maximum}, width={width}")
    requested:list[ScalarIndex] = []
    def collect(index:ScalarIndex) -> UOp:
      requested.append(index.simplify() if isinstance(index, UOp) else index)
      return UOp.const(self.transform.storage_dtype, 0)
    for i in range(width): self._code(collect, row, k_base+i)
    self.metadata(collect, row, k_base)
    cached_load = self.transform._tile_loads(source, tuple(requested))
    values = tuple(self._code(cached_load, row, k_base+i) for i in range(width))
    d, scale, subgroup = self.metadata(cached_load, row, k_base)
    return Q6KInt8Fragment(UOp(Ops.STACK, dtypes.char.vec(width), values), row, k_base, width,
                          self.logical_shape, d, scale, subgroup, self.abi)

  @property
  def identity(self) -> tuple[str, tuple[int, int], tuple[tuple[str, object], ...]]:
    return self.abi, self.logical_shape, tuple(sorted(self.transform.to_json().items()))


@dataclass(frozen=True)
class Q8ActivationRecordTransform:
  """Compact compiler-owned Q8 activation record used by grouped IMMA.

  One uint32 backing buffer owns row-major signed-int8 values followed by FP32
  K32 scales and raw sums.  Keeping those components in one typed PARAM lets an
  ordinary two-operand matmul retain every correction input without a global
  ``[groups,M,N]`` intermediate.
  """
  rows: int
  k: int
  group_elems: int = 32

  def __post_init__(self) -> None:
    if not isinstance(self.rows, int) or self.rows <= 0 or not isinstance(self.k, int) or self.k <= 0:
      raise ValueError("Q8 activation rows and K must be positive integers")
    if self.group_elems != 32 or self.k % self.group_elems:
      raise ValueError("Q8 activation K must be aligned to 32-value groups")

  @property
  def storage_dtype(self): return dtypes.uint32
  @property
  def storage_width(self) -> int: return 4
  @property
  def groups_per_row(self) -> int: return self.k//self.group_elems
  @property
  def values_bytes(self) -> int: return self.rows*self.k
  @property
  def metadata_elems(self) -> int: return self.rows*self.groups_per_row
  @property
  def scales_offset_words(self) -> int: return self.values_bytes//4
  @property
  def sums_offset_words(self) -> int: return self.scales_offset_words+self.metadata_elems
  @property
  def packed_bytes(self) -> int: return self.values_bytes+2*self.metadata_elems*4
  @property
  def storage_units(self) -> int: return self.packed_bytes//4
  @property
  def logical_shape(self) -> tuple[int, int]: return self.rows, self.k

  def _check_coords(self, row:ScalarIndex, k:ScalarIndex) -> None:
    if isinstance(row, int) and not 0 <= row < self.rows: raise IndexError(f"row={row} is outside [0, {self.rows})")
    if isinstance(k, int) and not 0 <= k < self.k: raise IndexError(f"k={k} is outside [0, {self.k})")
    if not isinstance(row, (int, UOp)) or not isinstance(k, (int, UOp)):
      raise TypeError("Q8 logical row/K must be integers or UOps")

  @staticmethod
  def _load(source:LoadSource, index:ScalarIndex) -> UOp:
    return source(index) if callable(source) else source[index]

  def value(self, source:LoadSource, row:ScalarIndex, k:ScalarIndex) -> UOp:
    self._check_coords(row, k)
    byte_index = row*self.k+k
    word = self._load(source, byte_index//4)
    return word.rshift((byte_index%4)*8).bitwise_and(0xff).cast(dtypes.uint8).bitcast(dtypes.char)

  def metadata(self, source:LoadSource, row:ScalarIndex, k_base:ScalarIndex) -> tuple[UOp, UOp, ScalarIndex]:
    self._check_coords(row, k_base)
    group = k_base//self.group_elems
    index = row*self.groups_per_row+group
    scale = self._load(source, self.scales_offset_words+index).bitcast(dtypes.float)
    raw_sum = self._load(source, self.sums_offset_words+index).bitcast(dtypes.float)
    return scale, raw_sum, group

  def to_json(self) -> dict[str, int|str]:
    return {"abi":"q8_1.compact_record.v1", "rows":self.rows, "k":self.k, "group_elems":self.group_elems,
            "values_bytes":self.values_bytes, "metadata_elems":self.metadata_elems, "packed_bytes":self.packed_bytes,
            "storage_dtype":self.storage_dtype.name}


@dataclass(frozen=True)
class Q8Int8Fragment:
  value: UOp
  row: ScalarIndex
  k_base: ScalarIndex
  width: Literal[8, 16]
  scale: UOp
  raw_sum: UOp
  logical_group: ScalarIndex
  abi: str = "q8_1.logical_mk_to_s8.v1"

  def __post_init__(self) -> None:
    if self.width not in (8, 16) or self.value.dtype != dtypes.char.vec(self.width):
      raise TypeError("Q8 int8 fragment must be a char vector of width 8 or 16")
    if self.scale.dtype != dtypes.float or self.raw_sum.dtype != dtypes.float:
      raise TypeError("Q8 int8 fragment scale/raw_sum must be float32")


@dataclass(frozen=True)
class Q8Int8FragmentProvider:
  transform: Q8ActivationRecordTransform
  abi: str = "q8_1.logical_mk_to_s8.v1"

  def __post_init__(self) -> None:
    if not isinstance(self.transform, Q8ActivationRecordTransform):
      raise TypeError("Q8 int8 fragment provider requires a Q8ActivationRecordTransform")

  @property
  def logical_shape(self) -> tuple[int, int]: return self.transform.logical_shape

  def fragment(self, source:LoadSource, row:ScalarIndex, k_base:ScalarIndex,
               width:Literal[8, 16]=16) -> Q8Int8Fragment:
    if width not in (8, 16): raise ValueError(f"Q8 int8 fragment width must be 8 or 16, got {width}")
    self.transform._check_coords(row, k_base)
    if isinstance(k_base, int) and k_base+width > self.transform.k:
      raise IndexError(f"Q8 fragment [{k_base}, {k_base+width}) is outside [0, {self.transform.k})")
    _check_k32_fragment_boundary(k_base, width, label="Q8_1")
    values = tuple(self.transform.value(source, row, k_base+i) for i in range(width))
    scale, raw_sum, group = self.transform.metadata(source, row, k_base)
    return Q8Int8Fragment(UOp(Ops.STACK, dtypes.char.vec(width), values), row, k_base, width, scale, raw_sum, group, self.abi)

  @property
  def identity(self) -> tuple[str, tuple[tuple[str, object], ...]]:
    return self.abi, tuple(sorted(self.transform.to_json().items()))


@dataclass(frozen=True)
class Q4KQ8GroupAccumulatorContract:
  """Typed K32 correction ABI consumed after one signed-int8 IMMA."""
  weight: Q4KInt8FragmentProvider
  activation: Q8Int8FragmentProvider
  abi: str = "q4_k_q8_1.k32_fp32_accumulator.v1"

  def __post_init__(self) -> None:
    if not isinstance(self.weight, Q4KInt8FragmentProvider) or not isinstance(self.activation, Q8Int8FragmentProvider):
      raise TypeError("group accumulator requires typed Q4_K and Q8 fragment providers")
    if self.weight.transform.k != self.activation.transform.k:
      raise ValueError("Q4_K/Q8 group accumulator K ownership must match")

  def correct(self, weight_source:LoadSource, activation_source:LoadSource, *, row:ScalarIndex, column:ScalarIndex,
              k_base:ScalarIndex, integer_dot:UOp) -> UOp:
    """Turn one exact K32 ``s8*s8->s32`` dot into its Q4_K/Q8_1 contribution.

    The half round-trips are part of the ABI, not an optimization accident:
    they match the proven composite emitter's packed ``half2`` metadata staging
    before its FP32 accumulator update.
    """
    if integer_dot.dtype.scalar() not in (dtypes.int, dtypes.uint) or integer_dot.dtype.count != 1:
      raise TypeError(f"group accumulator requires one integer scalar dot, got {integer_dot.dtype}")
    d, dmin, scale, minimum, weight_group = self.weight.metadata(weight_source, column, k_base)
    activation_scale, raw_sum, activation_group = self.activation.transform.metadata(activation_source, row, k_base)
    # Concrete mismatches are programming errors. Symbolic equality is guarded
    # by the shared logical k_base and the providers' identical K32 grouping.
    if isinstance(weight_group, int) and isinstance(activation_group, int) and weight_group != activation_group % 8:
      raise ValueError("Q4_K and Q8 metadata do not own the same logical K32 group")
    weight_code_scale = (d * scale.cast(dtypes.float)).cast(dtypes.half).cast(dtypes.float)
    weight_minimum = (-dmin * minimum.cast(dtypes.float)).cast(dtypes.half).cast(dtypes.float)
    activation_scale = activation_scale.cast(dtypes.half).cast(dtypes.float)
    raw_sum = raw_sum.cast(dtypes.half).cast(dtypes.float)
    return weight_code_scale * activation_scale * integer_dot.cast(dtypes.float) + weight_minimum * raw_sum

  @staticmethod
  def combine_staged(integer_dot:UOp, weight_metadata:UOp, activation_metadata:UOp) -> UOp:
    """Consume the proven ``half2`` metadata packets after cooperative LDS staging."""
    if integer_dot.dtype.scalar() not in (dtypes.int, dtypes.uint) or integer_dot.dtype.count != 1:
      raise TypeError(f"group accumulator requires one integer scalar dot, got {integer_dot.dtype}")
    if weight_metadata.dtype != dtypes.half.vec(2) or activation_metadata.dtype != dtypes.half.vec(2):
      raise TypeError("staged group metadata must be two exact half2 packets")
    wc0, wc1 = weight_metadata.gep(0).cast(dtypes.float), weight_metadata.gep(1).cast(dtypes.float)
    yc0, yc1 = activation_metadata.gep(0).cast(dtypes.float), activation_metadata.gep(1).cast(dtypes.float)
    return wc0 * yc0 * integer_dot.cast(dtypes.float) + wc1 * yc1


@dataclass(frozen=True)
class Q6KQ8SubgroupAccumulatorContract:
  """Typed paired-K16 correction ABI for one NVIDIA K32 IMMA step.

  ``integer_dots`` must contain independently masked low/high K16 subtotals.
  Accepting a single K32 dot would lose Q6_K's two distinct scales and is
  deliberately not part of this ABI.
  """
  weight: Q6KInt8FragmentProvider
  activation: Q8Int8FragmentProvider
  abi: str = "q6_k_q8_1.k16_pair_fp32_accumulator.v1"

  def __post_init__(self) -> None:
    if not isinstance(self.weight, Q6KInt8FragmentProvider) or not isinstance(self.activation, Q8Int8FragmentProvider):
      raise TypeError("subgroup accumulator requires typed Q6_K and Q8 fragment providers")
    if self.weight.transform.k != self.activation.transform.k:
      raise ValueError("Q6_K/Q8 subgroup accumulator K ownership must match")

  @staticmethod
  def _dots(integer_dots:tuple[UOp, UOp]) -> tuple[UOp, UOp]:
    if not isinstance(integer_dots, tuple) or len(integer_dots) != 2:
      raise TypeError("Q6_K subgroup accumulator requires separate low/high K16 integer dots")
    if any(x.dtype.scalar() not in (dtypes.int, dtypes.uint) or x.dtype.count != 1 for x in integer_dots):
      raise TypeError("Q6_K subgroup accumulator requires two integer scalar dots")
    return integer_dots

  def correct(self, weight_source:LoadSource, activation_source:LoadSource, *, row:ScalarIndex, column:ScalarIndex,
              k_base:ScalarIndex, integer_dots:tuple[UOp, UOp]) -> UOp:
    dot0, dot1 = self._dots(integer_dots)
    d, scale0, scale1, _ = self.weight.k32_metadata(weight_source, column, k_base)
    activation_scale, _, _ = self.activation.transform.metadata(activation_source, row, k_base)
    wc0 = (d*scale0.cast(dtypes.float)).cast(dtypes.half).cast(dtypes.float)
    wc1 = (d*scale1.cast(dtypes.float)).cast(dtypes.half).cast(dtypes.float)
    yc = activation_scale.cast(dtypes.half).cast(dtypes.float)
    return yc*(wc0*dot0.cast(dtypes.float)+wc1*dot1.cast(dtypes.float))

  @classmethod
  def combine_staged(cls, integer_dots:tuple[UOp, UOp], weight_metadata:UOp, activation_metadata:UOp) -> UOp:
    dot0, dot1 = cls._dots(integer_dots)
    if weight_metadata.dtype != dtypes.half.vec(2) or activation_metadata.dtype != dtypes.half.vec(2):
      raise TypeError("staged subgroup metadata must be two exact half2 packets")
    wc0, wc1 = weight_metadata.gep(0).cast(dtypes.float), weight_metadata.gep(1).cast(dtypes.float)
    yc = activation_metadata.gep(0).cast(dtypes.float)
    return yc*(wc0*dot0.cast(dtypes.float)+wc1*dot1.cast(dtypes.float))


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
  quant_format: QuantFormat
  rows: int
  k: int
  block_elems: int = 256
  block_bytes: int | None = None

  def __post_init__(self) -> None:
    quant_format = self.quant_format
    if isinstance(quant_format, str):
      try: quant_format = QUANT_FORMATS[quant_format]
      except KeyError as exc: raise ValueError(f"quant_format must be Q4_K or Q6_K, got {quant_format!r}") from exc
      object.__setattr__(self, "quant_format", quant_format)
    if quant_format is not Q4_K and quant_format is not Q6_K:
      raise ValueError(f"quant_format must be Q4_K or Q6_K, got {quant_format!r}")
    canonical_elems = quant_format.block_elems
    canonical_bytes = quant_format.block_bytes
    if self.block_elems != canonical_elems:
      raise ValueError(f"{quant_format.name} block_elems must be {canonical_elems}, got {self.block_elems}")
    if self.block_bytes is None: object.__setattr__(self, "block_bytes", canonical_bytes)
    elif self.block_bytes != canonical_bytes:
      raise ValueError(f"{quant_format.name} block_bytes must be {canonical_bytes}, got {self.block_bytes}")
    if not isinstance(self.rows, int) or self.rows <= 0: raise ValueError(f"rows must be a positive integer, got {self.rows!r}")
    if not isinstance(self.k, int) or self.k <= 0: raise ValueError(f"k must be a positive integer, got {self.k!r}")
    if self.k % canonical_elems: raise ValueError(f"k must be {quant_format.name} block aligned ({canonical_elems}), got {self.k}")

  @property
  def blocks_per_row(self) -> int: return self.k // self.block_elems

  @property
  def storage_dtype(self): return dtypes.uint32 if self.quant_format is Q4_K else dtypes.uint16

  @property
  def storage_width(self) -> int: return 4 if self.quant_format is Q4_K else 2

  @property
  def units_per_block(self) -> int:
    return Q4K_WORDS_PER_BLOCK if self.quant_format is Q4_K else Q6K_HALFWORDS_PER_BLOCK

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
    if self.quant_format is Q4_K:
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

  @classmethod
  def _byte_loader(cls, source:LoadSource, base:ScalarIndex, unit_bytes:int) -> Callable[[ScalarIndex], UOp]:
    def byte(index:ScalarIndex) -> UOp:
      return cls._load(source, base + index//unit_bytes).rshift((index%unit_bytes)*8).bitwise_and(0xff)
    return byte

  def _tile_loads(self, source:LoadSource, indices:tuple[ScalarIndex, ...]) -> Callable[[ScalarIndex], UOp]:
    """Cache native units, preserving proven adjacent units as b128/b64 LOADs.

    Direct-buffer offsets are unconditional and have exact descriptor bounds.
    Symbolic root-plus-constant runs widen only with alignment/range proofs;
    callable or otherwise unprovable sources keep the scalar path.
    """
    cache: dict[ScalarIndex, UOp] = {}
    if not callable(source):
      limit = self.packed_bytes // self.storage_width
      groups: dict[UOp|None, dict[int, ScalarIndex]] = {}
      for key in dict.fromkeys(indices):
        if isinstance(key, int): root, offset = None, key
        elif key.op is Ops.CONST and isinstance(key.arg, int): root, offset = None, key.arg
        else:
          root, offset = key.pop_const()
          if not isinstance(offset, int): continue
        groups.setdefault(root, {})[offset] = key
      for width in (16 // self.storage_width, 8 // self.storage_width):
        for root, offsets in groups.items():
          for start in sorted(offsets):
            run = tuple(range(start, start+width))
            if not all(x in offsets and offsets[x] not in cache for x in run): continue
            start_index = UOp.const(dtypes.weakint, start) if root is None else (root + start).simplify()
            if start_index.divides(width) is None or start_index.vmin < 0 or start_index.vmax+width > limit: continue
            carrier = source.index(start_index, dtype=self.storage_dtype.vec(width)).load()
            for lane, offset in enumerate(run): cache[offsets[offset]] = carrier.gep(lane)
    def load(index:ScalarIndex) -> UOp:
      key = index.simplify() if isinstance(index, UOp) else index
      if key not in cache: cache[key] = self._load(source, key)
      return cache[key]
    return load

  def dequant(self, source:LoadSource, row:ScalarIndex, k:ScalarIndex) -> UOp:
    """Return a pure scalar UOp expression, rounded to fp16, for ``weight[row, k]``."""
    self._check_coords(row, k)
    block = row * self.blocks_per_row + k // self.block_elems
    unit_base = block * self.units_per_block
    if self.quant_format is Q4_K: return self._dequant_q4(source, unit_base, k % self.block_elems)
    return self._dequant_q6(source, unit_base, k % self.block_elems)

  def dequant_tile(self, source:LoadSource, row:ScalarIndex, k_base:ScalarIndex, width:Literal[8, 16]=8) -> PackedWeightTile:
    """Decode 8 or 16 adjacent weights while sharing loads of their native packed units.

    Each lane retains scalar decoder semantics.  A per-tile native-unit cache hoists
    block/group metadata and naturally splits tiles which cross quant groups or
    blocks without introducing speculative out-of-range accesses.
    """
    if width not in (8, 16): raise ValueError(f"packed tile width must be 8 or 16, got {width}")
    self._check_coords(row, k_base)
    if isinstance(k_base, int) and k_base + width > self.k:
      raise IndexError(f"tile [{k_base}, {k_base+width}) is outside [0, {self.k})")
    requested: list[ScalarIndex] = []
    def collect(index:ScalarIndex) -> UOp:
      requested.append(index.simplify() if isinstance(index, UOp) else index)
      return UOp.const(self.storage_dtype, 0)
    # Discover the exact native-unit set before constructing lane expressions.
    for i in range(width): self.dequant(collect, row, k_base+i)
    cached_load = self._tile_loads(source, tuple(requested))
    lanes = tuple(self.dequant(cached_load, row, k_base+i) for i in range(width))
    return PackedWeightTile(UOp(Ops.STACK, dtypes.half.vec(width), lanes), row, k_base, width)

  def _dequant_q4(self, source:LoadSource, base:ScalarIndex, within:ScalarIndex) -> UOp:
    group, pos = within // 32, within % 32
    d, dmin, scale, minimum = self._q4_params(source, base, group)
    qword = self._load(source, base + 4 + (group//2)*8 + pos//4)
    q = qword.rshift((pos%4)*8 + (group%2)*4).bitwise_and(15)
    return (d * scale.cast(dtypes.float32) * q.cast(dtypes.float32) - dmin * minimum.cast(dtypes.float32)).cast(dtypes.float16)

  def _q4_params(self, source:LoadSource, base:ScalarIndex, group:ScalarIndex) -> tuple[UOp, UOp, UOp, UOp]:
    """Return canonical Q4_K ``D,Dmin,scale,min`` for one logical K32 group."""
    word0 = self._load(source, base)
    d = word0.bitwise_and(0xffff).cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)
    dmin = word0.rshift(16).bitwise_and(0xffff).cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)
    byte = self._byte_loader(source, base + 1, 4)
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
    return d, dmin, scale, minimum

  def _dequant_q6(self, source:LoadSource, base:ScalarIndex, within:ScalarIndex) -> UOp:
    group, pos = within // 16, within % 16
    byte = self._byte_loader(source, base, 2)
    half, pgroup = group // 8, group % 8
    ql_shift = 4 if isinstance(pgroup, int) and pgroup >= 4 else 0 if isinstance(pgroup, int) else (pgroup >= 4).where(4, 0)
    ql = byte(half*64 + (pgroup%4)*16 + pos).rshift(ql_shift).bitwise_and(15)
    qh = byte(128 + half*32 + (pgroup%2)*16 + pos).rshift((pgroup//2)*2).bitwise_and(3).lshift(4)
    q = ql.bitwise_or(qh).cast(dtypes.float32) - UOp.const(dtypes.float32, 32.0)
    scale = byte(192+group).cast(dtypes.uint8).bitcast(dtypes.int8).cast(dtypes.float32)
    d = self._load(source, base+104).cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)
    return (d * q * scale).cast(dtypes.float16)

  def to_json(self) -> dict[str, int | str]:
    return {"quant_format": self.quant_format.name, "rows": self.rows, "k": self.k, "block_elems": self.block_elems,
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
      if key in obj and obj[key] != ret.to_json()[key]: raise ValueError(f"{key} does not match {ret.quant_format.name} geometry")
    return ret


PackedWeightDescriptor = PackedWeightTransform
