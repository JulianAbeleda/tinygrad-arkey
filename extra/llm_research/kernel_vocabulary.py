"""Frozen research vocabulary for qk kernel candidates and hierarchical LDS layouts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tinygrad.dtype import DType

@dataclass(frozen=True)
class KernelLDSWindow:
  role: str
  base: int
  end: int
  stride_bytes: int
  def __post_init__(self):
    if self.role not in ("A", "B"): raise ValueError(f"kernel LDS window role must be A or B, got {self.role!r}")
    for name, value in (("base", self.base), ("end", self.end), ("stride_bytes", self.stride_bytes)):
      if not isinstance(value, int) or isinstance(value, bool): raise ValueError(f"kernel LDS window {name} must be an int")
    if self.base < 0 or self.end <= self.base: raise ValueError("kernel LDS window must have a non-empty non-negative interval")
    if self.stride_bytes <= 0: raise ValueError("kernel LDS window stride_bytes must be positive")
    if self.base % 16 or self.end % 16 or self.stride_bytes % 16: raise ValueError("kernel LDS window interval and stride must be b128 aligned")

@dataclass(frozen=True)
class KernelLDSComponentWindow:
  role: str
  component: str
  dtype: DType
  base: int
  end: int
  alignment: int
  stride_bytes: int|None = None
  def __post_init__(self):
    if self.role not in ("A", "B"): raise ValueError(f"kernel LDS component role must be A or B, got {self.role!r}")
    _name(self.component, "component")
    _dtype(self.dtype, "component")
    _ints("component", base=self.base, end=self.end, alignment=self.alignment)
    if self.base < 0 or self.end <= self.base: raise ValueError("kernel LDS component must have a non-empty non-negative interval")
    _alignment(self.alignment, self.dtype, self.base, "component")
    if (self.end-self.base) % self.dtype.itemsize: raise ValueError("kernel LDS component interval must contain whole dtype values")
    if self.stride_bytes is not None and (not isinstance(self.stride_bytes, int) or isinstance(self.stride_bytes, bool) or
                                          self.stride_bytes <= 0 or self.stride_bytes % self.alignment):
      raise ValueError("kernel LDS component stride_bytes must be positive and satisfy alignment when present")
  @property
  def size_bytes(self): return self.end-self.base
  @property
  def elements(self): return self.size_bytes//self.dtype.itemsize

@dataclass(frozen=True)
class KernelLDSRecordComponent:
  component: str
  dtype: DType
  offset_bytes: int
  size_bytes: int
  alignment: int
  def __post_init__(self):
    _name(self.component, "record component")
    _dtype(self.dtype, "record component")
    _ints("record component", offset_bytes=self.offset_bytes, size_bytes=self.size_bytes, alignment=self.alignment)
    if self.offset_bytes < 0 or self.size_bytes <= 0: raise ValueError("kernel LDS record component slice must be non-empty and non-negative")
    _alignment(self.alignment, self.dtype, self.offset_bytes, "record component")
    if self.size_bytes % self.dtype.itemsize: raise ValueError("kernel LDS record component slice must contain whole dtype values")
  @property
  def end_bytes(self): return self.offset_bytes+self.size_bytes

@dataclass(frozen=True)
class KernelLDSRecordLayout:
  rows: int
  stride_bytes: int
  components: tuple[KernelLDSRecordComponent, ...]
  def __post_init__(self):
    if not isinstance(self.rows, int) or isinstance(self.rows, bool) or self.rows <= 0: raise ValueError("kernel LDS record rows must be a positive int")
    if not isinstance(self.stride_bytes, int) or isinstance(self.stride_bytes, bool) or self.stride_bytes <= 0: raise ValueError("kernel LDS record stride_bytes must be a positive int")
    if not isinstance(self.components, tuple) or not self.components or not all(isinstance(x, KernelLDSRecordComponent) for x in self.components):
      raise ValueError("kernel LDS record components must be a non-empty tuple of KernelLDSRecordComponent values")
    if len({x.component for x in self.components}) != len(self.components): raise ValueError("duplicate kernel LDS record component")
    cursor = 0
    for component in sorted(self.components, key=lambda x: (x.offset_bytes, x.end_bytes)):
      if component.offset_bytes < cursor: raise ValueError("kernel LDS record component slices overlap")
      if component.offset_bytes > cursor: raise ValueError("kernel LDS record component slices have a gap")
      cursor = component.end_bytes
    if cursor != self.stride_bytes: raise ValueError("kernel LDS record components do not exactly cover stride_bytes")
  @property
  def size_bytes(self): return self.rows*self.stride_bytes
  def component(self, component:str):
    try: return next(x for x in self.components if x.component == component)
    except StopIteration as exc: raise KeyError(component) from exc
  def row_slice(self, row:int, component:str|None=None):
    if not isinstance(row, int) or isinstance(row, bool) or not 0 <= row < self.rows: raise IndexError(f"record row {row!r} is out of bounds")
    base = row*self.stride_bytes
    if component is None: return base, base+self.stride_bytes
    view = self.component(component)
    return base+view.offset_bytes, base+view.end_bytes

@dataclass(frozen=True)
class KernelLDSArenaRegion:
  name: str
  base: int
  end: int
  alignment: int = 16
  records: KernelLDSRecordLayout|None = None
  def __post_init__(self):
    _name(self.name, "arena region")
    _ints("arena region", base=self.base, end=self.end, alignment=self.alignment)
    if self.base < 0 or self.end <= self.base: raise ValueError("kernel LDS arena region must have a non-empty non-negative interval")
    if self.alignment <= 0 or self.alignment & (self.alignment-1) or self.base % self.alignment:
      raise ValueError("kernel LDS arena region base must satisfy a positive power-of-two alignment")
    if self.records is not None and not isinstance(self.records, KernelLDSRecordLayout): raise TypeError("records must be a KernelLDSRecordLayout")
    if self.records is not None and self.records.size_bytes != self.end-self.base: raise ValueError("kernel LDS arena region bounds do not match record layout size")
  def row_slice(self, row:int, component:str|None=None):
    if self.records is None: raise TypeError("kernel LDS arena region has no record layout")
    start, end = self.records.row_slice(row, component)
    return self.base+start, self.base+end

@dataclass(frozen=True)
class KernelTileGeometry:
  tile: tuple[int, int, int]
  waves: tuple[int, int]
  threads: int
  wave_size: int
  lds_windows: tuple[KernelLDSWindow, ...]
  lds_components: tuple[KernelLDSComponentWindow, ...] = ()
  lds_regions: tuple[KernelLDSArenaRegion, ...] = ()
  def __post_init__(self):
    if not isinstance(self.tile, tuple) or len(self.tile) != 3 or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in self.tile):
      raise ValueError("kernel tile geometry tile must contain three positive ints")
    if not isinstance(self.waves, tuple) or len(self.waves) != 2 or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in self.waves):
      raise ValueError("kernel tile geometry waves must contain two positive ints")
    if not isinstance(self.threads, int) or isinstance(self.threads, bool) or self.threads <= 0: raise ValueError("kernel tile geometry threads must be a positive int")
    if not isinstance(self.wave_size, int) or isinstance(self.wave_size, bool) or self.wave_size <= 0: raise ValueError("kernel tile geometry wave_size must be a positive int")
    if self.threads != self.waves[0]*self.waves[1]*self.wave_size: raise ValueError("kernel tile geometry waves do not account for threads")
    if not isinstance(self.lds_windows, tuple) or not all(isinstance(w, KernelLDSWindow) for w in self.lds_windows): raise ValueError("kernel tile geometry lds_windows must contain frozen KernelLDSWindow values")
    if len(self.lds_windows) != 2 or {w.role for w in self.lds_windows} != {"A", "B"}: raise ValueError("kernel tile geometry requires exactly one A and one B LDS window")
    if self.lds_windows[0].end != self.lds_windows[1].base: raise ValueError("kernel tile geometry LDS windows must be contiguous")
    if not isinstance(self.lds_components, tuple) or not all(isinstance(w, KernelLDSComponentWindow) for w in self.lds_components): raise ValueError("kernel tile geometry lds_components must contain frozen KernelLDSComponentWindow values")
    parents, keys = {w.role:w for w in self.lds_windows}, set()
    for component in self.lds_components:
      key = component.role, component.component
      if key in keys: raise ValueError(f"duplicate kernel LDS component {key!r}")
      keys.add(key); parent = parents[component.role]
      if component.base < parent.base or component.end > parent.end: raise ValueError(f"kernel LDS component {key!r} is outside its {component.role} window bounds")
    for left, right in zip(sorted(self.lds_components, key=lambda x:(x.base,x.end)), sorted(self.lds_components, key=lambda x:(x.base,x.end))[1:]):
      if right.base < left.end: raise ValueError(f"kernel LDS components {(left.role,left.component)!r} and {(right.role,right.component)!r} overlap")
    if not isinstance(self.lds_regions, tuple) or not all(isinstance(x, KernelLDSArenaRegion) for x in self.lds_regions): raise ValueError("kernel tile geometry lds_regions must contain frozen KernelLDSArenaRegion values")
    if not self.lds_regions:
      if self.lds_windows[0].base != 0: raise ValueError("kernel tile geometry LDS windows must be contiguous from byte zero")
    else:
      if len({x.name for x in self.lds_regions}) != len(self.lds_regions): raise ValueError("duplicate kernel LDS arena region")
      cursor = 0
      for region in self.lds_regions:
        if region.base < cursor: raise ValueError("kernel LDS arena regions overlap")
        if region.base > cursor: raise ValueError("kernel LDS arena regions have a gap")
        cursor = region.end
      if cursor != self.lds_windows[-1].end: raise ValueError("kernel LDS arena regions do not exactly cover the LDS arena")
      for window in self.lds_windows:
        if not any(x.base == window.base and x.end == window.end for x in self.lds_regions): raise ValueError(f"kernel LDS {window.role} window must exactly match an arena region")
  @property
  def lds_bytes(self): return max(x.end for x in self.lds_windows)
  def lds_component(self, role:str, component:str):
    try: return next(x for x in self.lds_components if (x.role,x.component) == (role,component))
    except StopIteration as exc: raise KeyError((role,component)) from exc
  def lds_component_views(self, role:str):
    if role not in ("A", "B"): raise ValueError(f"kernel LDS component role must be A or B, got {role!r}")
    return tuple(x for x in self.lds_components if x.role == role)
  def lds_region(self, name:str):
    try: return next(x for x in self.lds_regions if x.name == name)
    except StopIteration as exc: raise KeyError(name) from exc

@dataclass(frozen=True)
class KernelCandidateContext:
  schema_version: str
  canonical_identity: str
  geometry: KernelTileGeometry|None = None
  pipeline: Any|None = None
  packed_weight: Any|None = None
  packed_operand_a: Any|None = None
  packed_operand_b: Any|None = None
  def __post_init__(self):
    if self.schema_version != "boltbeam.full_kernel_candidate.v1": raise ValueError(f"unsupported kernel candidate context schema {self.schema_version!r}")
    if len(self.canonical_identity) != 64 or any(c not in "0123456789abcdef" for c in self.canonical_identity): raise ValueError("kernel candidate canonical_identity must be a lowercase SHA-256 hex digest")
    if self.packed_weight is not None or self.packed_operand_a is not None or self.packed_operand_b is not None:
      from tinygrad.codegen.opt.packed_weight import PackedOperandRecordTransform, PackedOperandTransform, PackedWeightTransform
      if self.packed_weight is not None and not isinstance(self.packed_weight, PackedWeightTransform): raise TypeError("kernel candidate packed_weight must be a PackedWeightTransform")
      valid = PackedOperandRecordTransform, PackedOperandTransform, PackedWeightTransform
      if self.packed_operand_a is not None and not isinstance(self.packed_operand_a, valid): raise TypeError("kernel candidate packed_operand_a must be a packed operand transform")
      if self.packed_operand_b is not None and not isinstance(self.packed_operand_b, valid): raise TypeError("kernel candidate packed_operand_b must be a packed operand transform")
      if self.packed_weight is not None and self.packed_operand_b is not None and self.packed_weight != self.packed_operand_b: raise ValueError("packed_weight and packed_operand_b describe different B transforms")
      if self.packed_weight is not None and self.packed_operand_b is None: object.__setattr__(self, "packed_operand_b", self.packed_weight)
      if self.geometry is None: raise ValueError("packed-weight candidates require explicit kernel tile geometry")

def _name(value, kind):
  if not isinstance(value, str) or not value or not value.isidentifier(): raise ValueError(f"kernel LDS {kind} name must be a non-empty identifier, got {value!r}")
def _dtype(value, kind):
  if not isinstance(value, DType): raise TypeError(f"kernel LDS {kind} dtype must be a DType")
def _ints(kind, **values):
  for name, value in values.items():
    if not isinstance(value, int) or isinstance(value, bool): raise ValueError(f"kernel LDS {kind} {name} must be an int")
def _alignment(alignment, dtype, offset, kind):
  if alignment <= 0 or alignment & (alignment-1): raise ValueError(f"kernel LDS {kind} alignment must be a positive power of two")
  if alignment < dtype.itemsize or offset % alignment: raise ValueError(f"kernel LDS {kind} offset must satisfy dtype alignment")
