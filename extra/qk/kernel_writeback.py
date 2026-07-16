"""Typed, fail-closed WMMA accumulator writeback construction."""
from __future__ import annotations

from dataclasses import dataclass

from extra.qk.kernel_lds import rdna3_wmma_output_coord, wmma_output_owners
from tinygrad.codegen.opt.kernel_lds import validate_rdna3_wmma_descriptor
from tinygrad.dtype import AddrSpace, DType, PtrDType, dtypes
from tinygrad.uop.ops import KernelTileGeometry, Ops, UOp


@dataclass(frozen=True)
class WMMAWritebackLayout:
  """Address ``identified_axis[id] * stride + direct_axis``.

  Axis names are output-coordinate names (``row`` and ``col``), not operand or
  quantization-format roles.
  """
  identified_axis: str
  direct_axis: str
  stride: int

  def __post_init__(self) -> None:
    if (self.identified_axis, self.direct_axis) not in (("row", "col"), ("col", "row")):
      raise ValueError("writeback layout must identify exactly one of row/col")
    if not isinstance(self.stride, int) or isinstance(self.stride, bool) or self.stride <= 0:
      raise ValueError("writeback stride must be a positive int")


@dataclass(frozen=True)
class WMMAWritebackDescriptor:
  geometry: KernelTileGeometry
  tc: object
  accumulator_dtype: DType
  accumulator_count: int
  layout: WMMAWritebackLayout
  ids_region: str | None = None
  exact_tile: bool = True

  def __post_init__(self) -> None:
    validate_rdna3_wmma_descriptor(self.tc)
    if not isinstance(self.exact_tile, bool): raise ValueError("exact_tile must be boolean")
    if self.geometry.wave_size != 32 or self.geometry.waves[0] * self.geometry.waves[1] * 32 != self.geometry.threads:
      raise ValueError("writeback requires the descriptor's exact wave32 geometry")
    # The lane map belongs to tc, but the persistent drain carrier need not: a
    # WMMA result may be corrected/converted immediately into another register
    # type before writeback.  Keep that boundary explicit and independently typed.
    if (not isinstance(self.accumulator_dtype, DType) or isinstance(self.accumulator_dtype, PtrDType) or
        self.accumulator_dtype.vcount != 1 or
        not (dtypes.is_float(self.accumulator_dtype) or dtypes.is_int(self.accumulator_dtype))):
      raise ValueError("accumulator_dtype must be an explicit scalar numeric output DType")
    sm = self.geometry.tile[0] // (self.geometry.waves[0] * 16)
    sn = self.geometry.tile[1] // (self.geometry.waves[1] * 16)
    if sm <= 0 or sn <= 0 or self.geometry.tile[:2] != (sm*self.geometry.waves[0]*16, sn*self.geometry.waves[1]*16):
      raise ValueError("wrong wave geometry: output tile does not exactly divide into 16x16 wave subtiles")
    if self.accumulator_count != sm * sn:
      raise ValueError("accumulator count drifted from per-wave subtile ownership")
    if self.ids_region is not None and (not isinstance(self.ids_region, str) or not self.ids_region.isidentifier()):
      raise ValueError("ids_region must be a named persistent LDS arena region")


@dataclass(frozen=True)
class WMMAWritebackProof:
  descriptor: WMMAWritebackDescriptor
  owner_count: int
  coordinates: frozenset[tuple[int, int]]

  @classmethod
  def prove(cls, descriptor: WMMAWritebackDescriptor) -> "WMMAWritebackProof":
    owners = wmma_output_owners(descriptor.geometry, tc=descriptor.tc)
    coords = [(x.row, x.col) for x in owners]
    expected = {(r, c) for r in range(descriptor.geometry.tile[0]) for c in range(descriptor.geometry.tile[1])}
    if len(coords) != len(set(coords)): raise ValueError("duplicate WMMA output owner coverage")
    if set(coords) != expected: raise ValueError("missing WMMA output owner coverage")
    return cls(descriptor, len(coords), frozenset(coords))


@dataclass(frozen=True)
class WMMAIDsReady:
  allocation: UOp
  ready: UOp
  region_name: str


@dataclass(frozen=True)
class WMMAWritebackTileMapping:
  """Runtime logical output bounds and this physical tile's destination origin.

  ``m`` is the writeback ``row`` axis and ``n`` is its ``col`` axis.  Bounds
  are full destination extents, not tail lengths; an element is valid when
  ``m_offset + row < m_extent`` and ``n_offset + col < n_extent``.
  """
  m_extent: UOp
  n_extent: UOp
  m_offset: UOp
  n_offset: UOp

  def __post_init__(self) -> None:
    for name in ("m_extent", "n_extent", "m_offset", "n_offset"):
      value = getattr(self, name)
      if not isinstance(value, UOp) or value.dtype.vcount != 1 or value.dtype.scalar() not in (dtypes.int, dtypes.weakint):
        raise ValueError(f"{name} must be a scalar integer UOp")
      if value.vmin < 0: raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class WMMAWriteback:
  proof: WMMAWritebackProof
  stores: tuple[UOp, ...]
  sink: UOp


def _validate_ids(proof:WMMAWritebackProof, ids:WMMAIDsReady|None) -> tuple[UOp, object]|None:
  desc = proof.descriptor
  if desc.ids_region is None:
    if ids is not None: raise ValueError("IDs binding supplied to an identity writeback")
    return None
  if ids is None or ids.region_name != desc.ids_region: raise ValueError("missing required named IDs LDS region")
  try: region = desc.geometry.lds_region(desc.ids_region)
  except KeyError as exc: raise ValueError("missing required named IDs LDS region") from exc
  extent = desc.geometry.tile[0 if desc.layout.identified_axis == "row" else 1]
  if region.records is None or region.records.rows != extent or region.records.stride_bytes != dtypes.int.itemsize or \
     len(region.records.components) != 1 or region.records.components[0].dtype != dtypes.int or \
     region.records.components[0].size_bytes != dtypes.int.itemsize:
    raise ValueError("IDs LDS region has the wrong integer dtype or size")
  alloc = ids.allocation
  if (alloc.op is not Ops.DEFINE_LOCAL or not isinstance(alloc.dtype, PtrDType) or alloc.ptrdtype.addrspace is not AddrSpace.LOCAL or
      alloc.ptrdtype.base != dtypes.uint8 or alloc.ptrdtype.size != desc.geometry.lds_bytes):
    raise ValueError("IDs allocation must be the exact byte-addressed LDS arena")
  if ids.ready.op not in (Ops.GROUP, Ops.END, Ops.BARRIER) or alloc not in ids.ready.backward_slice_with_self:
    raise ValueError("IDs readiness is detached from its LDS allocation")
  return alloc.after(ids.ready), region


def build_wmma_writeback(proof:WMMAWritebackProof, *, destination:UOp, accumulators:tuple[UOp, ...],
                         wave_m:UOp, wave_n:UOp, lane:UOp, ids:WMMAIDsReady|None=None,
                         mapping:WMMAWritebackTileMapping|None=None) -> WMMAWriteback:
  """Build one thread's scalar drains; the immutable proof covers the whole workgroup."""
  desc = proof.descriptor
  if proof != WMMAWritebackProof.prove(desc): raise ValueError("writeback proof is stale or detached from its descriptor")
  if desc.exact_tile and mapping is not None: raise ValueError("exact-tile writeback does not accept runtime edge mapping")
  if not desc.exact_tile and mapping is None: raise ValueError("edge writeback requires runtime M/N destination mapping")
  expected_vec = desc.accumulator_dtype.vec(8)
  if len(accumulators) != desc.accumulator_count or any(x.dtype != expected_vec for x in accumulators):
    raise ValueError("accumulator dtype/count drift")
  if not isinstance(destination.dtype, PtrDType) or destination.ptrdtype.base != desc.accumulator_dtype:
    raise ValueError("writeback destination has the wrong pointer dtype")
  for name, axis, bound in (("wave_m", wave_m, desc.geometry.waves[0]), ("wave_n", wave_n, desc.geometry.waves[1]), ("lane", lane, 32)):
    if axis.dtype.scalar() not in (dtypes.int, dtypes.weakint) or axis.vmin != 0 or axis.vmax != bound-1:
      raise ValueError(f"wrong wave geometry for {name}")
  ids_view = _validate_ids(proof, ids)
  sm = desc.geometry.tile[0] // (desc.geometry.waves[0] * 16)
  sn = desc.geometry.tile[1] // (desc.geometry.waves[1] * 16)
  stores: list[UOp] = []
  for subtile_m in range(sm):
    for subtile_n in range(sn):
      acc = accumulators[subtile_m*sn + subtile_n]
      for element in range(8):
        lr, lc = rdna3_wmma_output_coord(0, element, tc=desc.tc)
        # lane%16 and lane//16 are the lane-dependent portions of the same authoritative map.
        row = (wave_m*sm + subtile_m)*16 + lane % 16 + lr
        col = (wave_n*sn + subtile_n)*16 + lane // 16 + lc
        destination_row = row if mapping is None else mapping.m_offset + row
        destination_col = col if mapping is None else mapping.n_offset + col
        valid = None if mapping is None else (destination_row < mapping.m_extent) & (destination_col < mapping.n_extent)
        coord = row if desc.layout.identified_axis == "row" else col
        if ids_view is not None:
          arena, region = ids_view
          byte_index = region.base + coord * region.records.stride_bytes  # type: ignore[union-attr]
          ids_index = arena.index(byte_index, dtype=dtypes.int)
          identified = ids_index.load() if valid is None else ids_index.load(UOp.const(dtypes.int, 0), valid)
        else: identified = coord
        if desc.layout.identified_axis == "row":
          mapped_row, mapped_col = identified if mapping is None else mapping.m_offset + identified, destination_col
        else:
          mapped_row, mapped_col = destination_row, identified if mapping is None else mapping.n_offset + identified
        identified = mapped_row if desc.layout.identified_axis == "row" else mapped_col
        direct = mapped_col if desc.layout.direct_axis == "col" else mapped_row
        address = identified * desc.layout.stride + direct
        stores.append(destination.index(address).store(acc.gep(element), valid).replace(
          tag=("wmma_writeback", subtile_m, subtile_n, element, desc.layout.identified_axis)))
  return WMMAWriteback(proof, tuple(stores), UOp.sink(*stores))
