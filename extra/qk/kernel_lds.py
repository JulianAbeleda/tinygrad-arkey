"""Pure cooperative LDS ownership math for compiler-bound kernel geometry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeAlias

from extra.qk.kernel_pipeline import (HierarchicalKernelPipelinePlan, HierarchicalLifecycleEvent,
                                      hierarchical_lifecycle_events, prove_hierarchical_lifecycle)
from tinygrad.codegen.opt.kernel_lds import (CooperativeLDSStore, PackedPrecontractOperandTemplate, PrecontractContractSpec,
  PrecontractFactors, PrecontractFragmentInstance, PrecontractKAxis, PrecontractLDSStage, PrecontractOperand,
  PrecontractOperandTemplate, PrecontractPipelineTemplate, PrecontractProducerInstance, PrecontractThreadAxes, WMMAFragmentLoad,
  WMMAOutputOwner, _rdna3_wmma_output_coord, _window, build_precontract_lds_stage, contract_symbolic_upcast,
  cooperative_lds_padding_offsets, cooperative_lds_stores, derive_precontract_factors, derive_precontract_shape_factors,
  instantiate_precontract_fragments, instantiate_precontract_producer, lower_symbolic_barrier_dependencies, rdna3_wmma_output_coord,
  semantic_wave_coords, validate_precontract_carriers, validate_precontract_contracts, validate_precontract_operand_templates,
  validate_precontract_thread_axes, validate_precontract_wmma_abi, validate_rdna3_wmma_descriptor, wmma_fragment_loads,
  wmma_output_owners)
from tinygrad.codegen.opt.packed_weight import PackedOperandRecordTransform, PackedOperandTransform, PackedWeightTransform
from tinygrad.dtype import AddrSpace, DType, PtrDType, dtypes
from tinygrad.uop.ops import (AxisType, KernelLDSArenaRegion, KernelLDSComponentWindow, KernelLDSWindow, KernelTileGeometry,
                              Ops, UOp)





def lds_arena_bytes(geometry:KernelTileGeometry) -> int:
  """Total bytes in one LDS arena, for legacy and component-aware geometries."""
  return geometry.lds_bytes


def lds_component_view(geometry:KernelTileGeometry, role:str, component:str) -> KernelLDSComponentWindow:
  """Return one typed named LDS view without changing legacy A/B window ownership."""
  return geometry.lds_component(role, component)


def lds_component_views(geometry:KernelTileGeometry, role:str) -> tuple[KernelLDSComponentWindow, ...]:
  return geometry.lds_component_views(role)














PackedComponentVectorProducer: TypeAlias = UOp | Callable[[UOp, UOp, UOp, int], UOp]


@dataclass(frozen=True)
class PackedComponentLDSBinding:
  """Bind one generic transform component to one typed LDS component window."""
  role: str
  component: str
  source: UOp
  row_axis: UOp
  k_axis: UOp
  row_tile_base: UOp
  producer: PackedComponentVectorProducer
  vector_bytes: int = 16

  def __post_init__(self) -> None:
    if self.role not in ("A", "B"): raise ValueError(f"packed component binding role must be A or B, got {self.role!r}")
    if not isinstance(self.component, str) or not self.component: raise ValueError("packed component binding requires a component name")
    if not isinstance(self.source.dtype, PtrDType): raise TypeError("packed component binding source must be a pointer")
    if self.row_axis.op is not Ops.RANGE or self.k_axis.op is not Ops.RANGE:
      raise ValueError("packed component binding must retain row/K RANGE ownership")
    if self.row_tile_base.dtype.scalar() not in (dtypes.int, dtypes.weakint):
      raise ValueError("packed component binding row tile base must be integer")
    if not isinstance(self.producer, UOp) and not callable(self.producer):
      raise TypeError("packed component producer must be a UOp expression or callback")
    if not isinstance(self.vector_bytes, int) or isinstance(self.vector_bytes, bool) or self.vector_bytes <= 0:
      raise ValueError("packed component vector_bytes must be positive")


@dataclass(frozen=True)
class PackedComponentOperandTemplate:
  """One fragment-producing component and its epilogue-visible sidecars."""
  role: str
  transform: PackedOperandTransform
  value: PackedComponentLDSBinding
  sidecars: tuple[PackedComponentLDSBinding, ...] = ()

  def __post_init__(self) -> None:
    if self.role not in ("A", "B"): raise ValueError(f"packed component template role must be A or B, got {self.role!r}")
    if not isinstance(self.transform, PackedOperandTransform): raise TypeError("packed component template requires a PackedOperandTransform")
    if not isinstance(self.value, PackedComponentLDSBinding) or not isinstance(self.sidecars, tuple) or \
       not all(isinstance(x, PackedComponentLDSBinding) for x in self.sidecars):
      raise TypeError("packed component template bindings must be frozen PackedComponentLDSBinding values")
    if any(x.role != self.role for x in self.bindings): raise ValueError("packed component binding role does not match its template")
    names = tuple(x.component for x in self.bindings)
    if len(set(names)) != len(names): raise ValueError("packed component bindings require unique component ownership")
    transform_names = tuple(x.name for x in self.transform.components)
    for name in names:
      if name not in transform_names: raise ValueError(f"packed component {name!r} does not exist in the transform")
    if set(names) != set(transform_names): raise ValueError("packed component binding byte sum must cover the transform exactly")
    source_bytes = 0
    for binding in self.bindings:
      component = self.transform.component(binding.component)
      if binding.source.ptrdtype.base != component.dtype or binding.source.ptrdtype.size < 0:
        raise ValueError(f"packed component {binding.component!r} source dtype/size does not match the transform")
      source_bytes += binding.source.ptrdtype.size * component.dtype.itemsize
      if binding.source.ptrdtype.size * component.dtype.itemsize != component.size_bytes:
        raise ValueError(f"packed component {binding.component!r} source dtype/size does not match the transform")
      if component.size_bytes % binding.vector_bytes or binding.vector_bytes % component.dtype.itemsize:
        raise ValueError(f"packed component {binding.component!r} requires exact whole aligned vectors")
      if isinstance(binding.producer, UOp):
        expected = component.dtype.vec(binding.vector_bytes // component.dtype.itemsize)
        if binding.producer.dtype != expected:
          raise ValueError(f"packed component {binding.component!r} producer has the wrong dtype/vector width")
        owned = binding.producer.backward_slice_with_self
        if binding.source not in owned or binding.row_axis not in owned or binding.k_axis not in owned:
          raise ValueError(f"packed component {binding.component!r} producer is detached from source/row/K ownership")
    if source_bytes != sum(x.size_bytes for x in self.transform.components):
      raise ValueError("packed component binding byte sum does not match the transform")

  @property
  def bindings(self) -> tuple[PackedComponentLDSBinding, ...]: return (self.value,)+self.sidecars






@dataclass(frozen=True)
class PackedComponentSidecarView:
  role: str
  component: str
  vectors: tuple[UOp, ...]


@dataclass(frozen=True)
class PackedComponentLDSStage:
  allocation: UOp
  producer: UOp
  barrier: UOp
  fragment_a: UOp
  fragment_b: UOp
  sidecars: tuple[PackedComponentSidecarView, ...]


PackedRecordVectorProducer: TypeAlias = UOp | Callable[[tuple[UOp, ...], UOp, UOp, int], UOp]
PackedRecordAddressRemap: TypeAlias = Callable[[UOp], UOp]
PackedRecordLoadValidity: TypeAlias = Callable[[UOp], UOp]


@dataclass(frozen=True)
class PackedRecordCooperativeStore:
  """One callback-elected store from a packed source coordinate to a record field vector."""
  field: str
  iteration: int
  source_row: UOp
  source_k: UOp
  destination_row: UOp
  destination_vector: UOp
  value: UOp

  def __post_init__(self) -> None:
    if not isinstance(self.field, str) or not self.field: raise ValueError("cooperative store requires a field")
    if not isinstance(self.iteration, int) or isinstance(self.iteration, bool) or self.iteration < 0:
      raise ValueError("cooperative store iteration must be a non-negative int")
    if not all(isinstance(x, UOp) for x in (self.source_row, self.source_k, self.destination_row,
                                             self.destination_vector, self.value)):
      raise TypeError("cooperative store coordinates and value must be UOps")


PackedRecordScheduleCallback: TypeAlias = Callable[["PackedRecordOperandTemplate", PrecontractThreadAxes, int],
                                                    tuple[PackedRecordCooperativeStore, ...]]


@dataclass(frozen=True)
class PackedRecordCooperativeSchedule:
  """Typed per-template vocabulary for arbitrary cooperative source/destination ownership."""
  name: str
  callback: PackedRecordScheduleCallback
  owner_axes: tuple[str, ...]

  def __post_init__(self) -> None:
    if not isinstance(self.name, str) or not self.name: raise ValueError("cooperative schedule requires a name")
    if not callable(self.callback): raise TypeError("cooperative schedule requires a callback")
    if not isinstance(self.owner_axes, tuple) or not self.owner_axes or not set(self.owner_axes) <= {"wave_m", "wave_n", "lane"} or \
       len(set(self.owner_axes)) != len(self.owner_axes) or "lane" not in self.owner_axes:
      raise ValueError("cooperative schedule owner_axes must be unique wave_m/wave_n/lane names including lane")

  def axes(self, threads:PrecontractThreadAxes) -> tuple[UOp, ...]: return tuple(getattr(threads, x) for x in self.owner_axes)


@dataclass(frozen=True)
class PackedRecordSource:
  """One packed source with optional logical-address remapping and tail validity."""
  name: str
  pointer: UOp
  address_remap: PackedRecordAddressRemap | None = None
  load_validity: PackedRecordLoadValidity | None = None

  def __post_init__(self) -> None:
    if not isinstance(self.name, str) or not self.name: raise ValueError("packed record source requires a name")
    if not isinstance(self.pointer.dtype, PtrDType): raise TypeError("packed record source must be a pointer")
    if self.address_remap is not None and not callable(self.address_remap):
      raise TypeError("packed record source address_remap must be callable when present")
    if self.load_validity is not None and not callable(self.load_validity):
      raise TypeError("packed record source load_validity must be callable when present")

  def load(self, logical_index:UOp, *, dtype:DType|None=None) -> UOp:
    """Load one logical address; invalid lanes access element zero and produce neutral zero."""
    if not isinstance(logical_index, UOp) or not dtypes.is_int(logical_index.dtype):
      raise TypeError("packed record source logical load index must be an integer UOp")
    physical_index = logical_index if self.address_remap is None else self.address_remap(logical_index)
    if not isinstance(physical_index, UOp) or not dtypes.is_int(physical_index.dtype):
      raise TypeError("packed record source address_remap must return an integer UOp")
    load_dtype = dtype or self.pointer.ptrdtype.base
    if not isinstance(load_dtype, DType): raise TypeError("packed record source load dtype must be a DType")
    if self.load_validity is None: return self.pointer.index(physical_index, dtype=load_dtype).load()
    valid = self.load_validity(logical_index)
    if not isinstance(valid, UOp) or valid.dtype.scalar() != dtypes.bool:
      raise TypeError("packed record source load_validity must return a bool UOp")
    safe_index = valid.where(physical_index, UOp.const(physical_index.dtype.scalar(), 0))
    return self.pointer.index(safe_index, dtype=load_dtype).load(UOp.const(load_dtype, 0), valid)


@dataclass(frozen=True)
class PackedRecordFieldProducer:
  """Produce one non-reserved field from one or more declared packed sources."""
  field: str
  sources: tuple[str, ...]
  producer: PackedRecordVectorProducer
  vector_bytes: int = 16

  def __post_init__(self) -> None:
    if not isinstance(self.field, str) or not self.field: raise ValueError("packed record producer requires a field name")
    if not isinstance(self.sources, tuple) or not self.sources or not all(isinstance(x, str) and x for x in self.sources):
      raise ValueError("packed record producer requires declared source names")
    if len(set(self.sources)) != len(self.sources): raise ValueError("packed record producer has duplicate sources")
    if not isinstance(self.producer, UOp) and not callable(self.producer):
      raise TypeError("packed record producer must be a UOp expression or callback")
    if not isinstance(self.vector_bytes, int) or isinstance(self.vector_bytes, bool) or self.vector_bytes <= 0:
      raise ValueError("packed record vector_bytes must be positive")


@dataclass(frozen=True)
class PackedRecordOperandTemplate:
  """A source-record transform staged into one typed interleaved LDS record region."""
  role: str
  transform: PackedOperandRecordTransform
  sources: tuple[PackedRecordSource, ...]
  fields: tuple[PackedRecordFieldProducer, ...]
  reserved_fields: tuple[str, ...]
  primary_field: str
  row_axis: UOp
  k_axis: UOp
  row_tile_base: UOp
  fragment_dtype: DType | None = None
  cooperative_schedule: PackedRecordCooperativeSchedule | None = None

  def __post_init__(self) -> None:
    if self.role not in ("A", "B"): raise ValueError(f"packed record template role must be A or B, got {self.role!r}")
    if not isinstance(self.transform, PackedOperandRecordTransform):
      raise TypeError("packed record template requires a PackedOperandRecordTransform")
    if not isinstance(self.sources, tuple) or not self.sources or not all(isinstance(x, PackedRecordSource) for x in self.sources):
      raise TypeError("packed record template requires declared sources")
    if len({x.name for x in self.sources}) != len(self.sources): raise ValueError("duplicate packed record source declaration")
    source_names = {x.name for x in self.sources}
    transform_sources = {x.name for x in self.transform.source.components}
    if source_names != transform_sources: raise ValueError("packed record sources must exactly match transform source declarations")
    if not isinstance(self.fields, tuple) or not all(isinstance(x, PackedRecordFieldProducer) for x in self.fields):
      raise TypeError("packed record fields must be frozen PackedRecordFieldProducer values")
    names = tuple(x.field for x in self.fields)
    if len(set(names)) != len(names): raise ValueError("duplicate packed record field producer")
    if not isinstance(self.reserved_fields, tuple) or not all(isinstance(x, str) for x in self.reserved_fields):
      raise TypeError("packed record reserved_fields must be a tuple of names")
    if len(set(self.reserved_fields)) != len(self.reserved_fields): raise ValueError("duplicate packed record reserved field")
    if set(names) & set(self.reserved_fields): raise ValueError("reserved packed record fields cannot be produced")
    produced_names = {x.name for x in self.transform.produced.components}
    if set(names) | set(self.reserved_fields) != produced_names:
      raise ValueError("packed record fields require exactly one producer or an explicit reserved declaration")
    if self.primary_field not in names: raise ValueError("packed record primary field must be produced")
    for field in self.fields:
      if not set(field.sources) <= source_names: raise ValueError(f"packed record field {field.field!r} uses an undeclared source")
    if self.row_axis.op is not Ops.RANGE or self.k_axis.op is not Ops.RANGE:
      raise ValueError("packed record template must retain row/K RANGE ownership")
    if self.row_tile_base.dtype.scalar() not in (dtypes.int, dtypes.weakint):
      raise ValueError("packed record row tile base must be integer")
    if self.fragment_dtype is not None and not isinstance(self.fragment_dtype, DType):
      raise TypeError("packed record fragment_dtype must be a DType when present")
    if self.cooperative_schedule is not None and not isinstance(self.cooperative_schedule, PackedRecordCooperativeSchedule):
      raise TypeError("packed record cooperative_schedule must be typed when present")

  def source(self, name:str) -> UOp:
    try: return next(x.pointer for x in self.sources if x.name == name)
    except StopIteration as exc: raise KeyError(name) from exc

  def source_load(self, name:str, logical_index:UOp, *, dtype:DType|None=None) -> UOp:
    try: return next(x for x in self.sources if x.name == name).load(logical_index, dtype=dtype)
    except StopIteration as exc: raise KeyError(name) from exc


@dataclass(frozen=True)
class PackedRecordLDSRegionBinding:
  role: str
  region: str

  def __post_init__(self) -> None:
    if self.role not in ("A", "B"): raise ValueError("packed record region role must be A or B")
    if not isinstance(self.region, str) or not self.region: raise ValueError("packed record region requires a name")


@dataclass(frozen=True)
class PackedRecordSidecarView:
  role: str
  field: str
  vectors: tuple[UOp, ...]


@dataclass(frozen=True)
class PackedRecordLDSStage:
  allocation: UOp
  producer: UOp
  barrier: UOp
  fragment_a: UOp
  fragment_b: UOp
  sidecars: tuple[PackedRecordSidecarView, ...]


@dataclass(frozen=True)
class HierarchicalPackedRecordStageDescriptor:
  """K decomposition for an asymmetric, record-backed hierarchical stage."""
  plan: HierarchicalKernelPipelinePlan
  outer_k: int
  phase_k: int
  group_k: int

  def __post_init__(self) -> None:
    if not isinstance(self.plan, HierarchicalKernelPipelinePlan): raise TypeError("hierarchical record stage requires a pipeline plan")
    for name, value in (("outer_k", self.outer_k), ("phase_k", self.phase_k), ("group_k", self.group_k)):
      if not isinstance(value, int) or isinstance(value, bool) or value <= 0: raise ValueError(f"{name} must be a positive int")
    if self.outer_k != self.plan.phase_count * self.phase_k:
      raise ValueError("outer_k must equal phase_count * phase_k")
    if self.phase_k % self.group_k: raise ValueError("phase_k must be divisible by group_k")

  @property
  def groups_per_phase(self) -> int: return self.phase_k // self.group_k


@dataclass(frozen=True)
class HierarchicalPackedRecordSidecar:
  role: str
  field: str
  byte_address: UOp
  byte_size: int
  value: UOp


@dataclass(frozen=True)
class HierarchicalPackedRecordGroup:
  phase: int
  group: int
  persistent_k: int
  overwriteable_k: int
  persistent_row: UOp
  overwriteable_row: UOp
  persistent_byte_address: UOp
  overwriteable_byte_address: UOp
  persistent_fragment: UOp
  overwriteable_fragment: UOp
  sidecars: tuple[HierarchicalPackedRecordSidecar, ...]


@dataclass(frozen=True)
class HierarchicalPackedRecordPhase:
  phase: int
  producer: UOp
  publish: UOp
  groups: tuple[HierarchicalPackedRecordGroup, ...]
  release: UOp


@dataclass(frozen=True)
class HierarchicalPackedRecordStage:
  descriptor: HierarchicalPackedRecordStageDescriptor
  geometry: KernelTileGeometry
  tc: object
  contracts: tuple[PrecontractContractSpec, ...]
  templates: tuple[PackedRecordOperandTemplate, ...]
  regions: tuple[PackedRecordLDSRegionBinding, ...]
  threads: PrecontractThreadAxes
  subtile_m: UOp
  subtile_n: UOp
  allocation: UOp
  persistent_producer: UOp
  phases: tuple[HierarchicalPackedRecordPhase, ...]
  events: tuple[HierarchicalLifecycleEvent, ...]

  @property
  def groups(self) -> tuple[HierarchicalPackedRecordGroup, ...]: return tuple(x for phase in self.phases for x in phase.groups)
  @property
  def barriers(self) -> tuple[UOp, ...]: return tuple(x for phase in self.phases for x in (phase.publish, phase.release))


@dataclass(frozen=True)
class HierarchicalPackedRecordStageProof:
  passed: bool
  errors: tuple[str, ...]























def validate_packed_component_templates(geometry:KernelTileGeometry, tc,
                                        templates:tuple[PackedComponentOperandTemplate, ...]) -> PrecontractFactors:
  """Validate generic transform-to-LDS component ownership without defining a packed format."""
  factors = derive_precontract_shape_factors(geometry, tc)
  if (not isinstance(templates, tuple) or not all(isinstance(x, PackedComponentOperandTemplate) for x in templates) or
      tuple(x.role for x in templates) != ("A", "B")):
    raise ValueError("packed component templates must be exactly ordered A and B")
  owned: set[tuple[str, str]] = set()
  for template in templates:
    parent = _window(geometry, template.role)
    geometry_names = {x.component for x in geometry.lds_component_views(template.role)}
    binding_names = {x.component for x in template.bindings}
    if geometry_names != binding_names:
      raise ValueError(f"packed component {template.role} binding byte sum must cover its LDS window components exactly")
    lds_bytes = 0
    rows = geometry.tile[0] if template.role == "A" else geometry.tile[1]
    for binding in template.bindings:
      key = (binding.role, binding.component)
      if key in owned: raise ValueError(f"packed component {key!r} has duplicate ownership")
      owned.add(key)
      component = template.transform.component(binding.component)
      try: lds = geometry.lds_component(binding.role, binding.component)
      except KeyError as exc: raise ValueError(f"packed component {key!r} has no matching LDS component window") from exc
      if lds.dtype != component.dtype: raise ValueError(f"packed component {key!r} LDS dtype does not match the transform")
      if lds.size_bytes != component.size_bytes: raise ValueError(f"packed component {key!r} LDS range does not match the transform")
      if lds.stride_bytes is None or lds.size_bytes != rows * lds.stride_bytes:
        raise ValueError(f"packed component {key!r} LDS range/stride does not exactly cover operand rows")
      if component.stride_bytes is not None and component.stride_bytes != lds.stride_bytes:
        raise ValueError(f"packed component {key!r} LDS stride does not match the transform")
      required_alignment = max(component.alignment, binding.vector_bytes)
      if (binding.vector_bytes != 16 or lds.alignment < required_alignment or lds.base % binding.vector_bytes or
          lds.stride_bytes % binding.vector_bytes):
        raise ValueError(f"packed component {key!r} LDS vector width/alignment is invalid")
      lds_bytes += lds.size_bytes
    if lds_bytes != parent.end - parent.base:
      raise ValueError(f"packed component {template.role} binding byte sum does not match its LDS arena window")
    value_component = template.transform.component(template.value.component)
    value_lds = geometry.lds_component(template.role, template.value.component)
    if value_component.dtype != tc.dtype_in or value_lds.stride_bytes < geometry.tile[2] * tc.dtype_in.itemsize:
      raise ValueError(f"packed component {template.role} value component does not cover the tensor-core K row")
  return factors

















def build_packed_component_lds_stage(geometry:KernelTileGeometry, *, tc, allocation:UOp,
                                     templates:tuple[PackedComponentOperandTemplate, ...], threads:PrecontractThreadAxes,
                                     k_axis:PrecontractKAxis, subtile_m:UOp, subtile_n:UOp,
                                     contracts:tuple[PrecontractContractSpec, ...]) -> PackedComponentLDSStage:
  """Stage generic named components through a byte-addressed heterogeneous LDS arena."""
  factors = validate_packed_component_templates(geometry, tc, templates)
  validate_precontract_thread_axes(geometry, factors, threads, subtile_m, subtile_n, context="packed component")
  validate_precontract_contracts(tc, contracts, context="packed component",
                                 mismatch="does not match actual descriptor operand mapping")
  if (k_axis.tile_owner.op is not Ops.RANGE or k_axis.tile_owner.arg[-1] is not AxisType.REDUCE or
      k_axis.tile_owner not in k_axis.tile_base.backward_slice_with_self):
    raise ValueError("packed component K tile owner must be a live REDUCE range in tile base")
  if (k_axis.substep_owner.op is not Ops.RANGE or k_axis.substep_owner.arg[-1] is not AxisType.UNROLL or
      k_axis.substep_owner.vmax+1 != factors.k_substeps or k_axis.substep_owner not in k_axis.substep.backward_slice_with_self):
    raise ValueError("packed component K substep owner must be a live derived-size UNROLL range in substep")
  if (allocation.op is not Ops.DEFINE_LOCAL or allocation.ptrdtype.addrspace is not AddrSpace.LOCAL or
      allocation.ptrdtype.base != dtypes.uint8 or allocation.ptrdtype.size != geometry.lds_bytes):
    raise ValueError("packed component allocation must be one exact byte-addressed LDS arena")

  thread = (threads.wave_m * geometry.waves[1] + threads.wave_n) * geometry.wave_size + threads.lane
  stores: list[UOp] = []
  for template in templates:
    rows = geometry.tile[0] if template.role == "A" else geometry.tile[1]
    for binding in template.bindings:
      component = template.transform.component(binding.component)
      window = geometry.lds_component(binding.role, binding.component)
      vector_elements = binding.vector_bytes // component.dtype.itemsize
      vectors_per_row = window.stride_bytes // binding.vector_bytes  # type: ignore[operator]
      vector_count = rows * vectors_per_row
      if vector_count % geometry.threads:
        raise ValueError(f"packed component {(binding.role, binding.component)!r} vectors do not divide cooperative threads")
      for iteration in range(vector_count // geometry.threads):
        linear_vector = thread + iteration * geometry.threads
        row, vector = linear_vector // vectors_per_row, linear_vector % vectors_per_row
        logical_row, logical_k = binding.row_tile_base + row, k_axis.tile_base + vector * vector_elements
        if isinstance(binding.producer, UOp):
          value = binding.producer.substitute({binding.row_axis:logical_row, binding.k_axis:logical_k})
        else: value = binding.producer(binding.source, logical_row, logical_k, vector_elements)
        if not isinstance(value, UOp) or value.dtype != component.dtype.vec(vector_elements):
          raise ValueError(f"packed component {(binding.role, binding.component)!r} producer has the wrong dtype/vector width")
        owned = value.backward_slice_with_self
        if binding.source not in owned or logical_row not in owned or logical_k not in owned:
          raise ValueError(f"packed component {(binding.role, binding.component)!r} producer is detached from source/row/K ownership")
        byte_index = window.base + row * window.stride_bytes + vector * binding.vector_bytes  # type: ignore[operator]
        tag = ("packed_component_store", binding.role, binding.component, iteration)
        idx = allocation.index(byte_index, dtype=component.dtype.vec(vector_elements)).replace(tag=tag)
        stores.append(idx.store(value).replace(tag=tag).end())

  producer = UOp.group(*stores)
  barrier = UOp.barrier(producer)
  ordered, lane = allocation.after(barrier), threads.lane
  fragment_rows: dict[str, UOp] = {}
  fragments: list[UOp] = []
  for operand_idx, (template, subtile, wave, subtiles, contract) in enumerate(zip(
      templates, (subtile_m, subtile_n), (threads.wave_m, threads.wave_n), (factors.subtiles_m, factors.subtiles_n), contracts)):
    binding = template.value
    window = geometry.lds_component(template.role, binding.component)
    row = (wave * subtiles + subtile) * tc.dims[1-operand_idx] + lane % tc.dims[1-operand_idx]
    fragment_rows[template.role] = row
    logical_k = k_axis.substep * tc.dims[2] + contract.element
    byte_index = window.base + row * window.stride_bytes + logical_k * tc.dtype_in.itemsize  # type: ignore[operator]
    load = ordered.index(byte_index, dtype=tc.dtype_in).replace(
      tag=("packed_component_fragment_load", template.role, binding.component)).load()
    fragments.append(UOp(Ops.CONTRACT, tc.dtype_in.vec(tc.elements_per_thread[operand_idx]), (load,), contract.arg,
                         tag=("packed_component_fragment", template.role, binding.component)))

  sidecar_views: list[PackedComponentSidecarView] = []
  for template in templates:
    row = fragment_rows[template.role]
    for binding in template.sidecars:
      component, window = template.transform.component(binding.component), geometry.lds_component(binding.role, binding.component)
      vector_elements = binding.vector_bytes // component.dtype.itemsize
      vectors = tuple(ordered.index(window.base + row * window.stride_bytes + vector * binding.vector_bytes,
                                    dtype=component.dtype.vec(vector_elements)).replace(
                        tag=("packed_component_sidecar_load", binding.role, binding.component, vector)).load()
                      for vector in range(window.stride_bytes // binding.vector_bytes))  # type: ignore[operator]
      sidecar_views.append(PackedComponentSidecarView(binding.role, binding.component, vectors))
  return PackedComponentLDSStage(allocation, producer, barrier, fragments[0], fragments[1], tuple(sidecar_views))


def build_packed_record_lds_stage(geometry:KernelTileGeometry, *, tc, allocation:UOp,
                                  templates:tuple[PackedRecordOperandTemplate, ...],
                                  regions:tuple[PackedRecordLDSRegionBinding, ...], threads:PrecontractThreadAxes,
                                  k_axis:PrecontractKAxis, subtile_m:UOp, subtile_n:UOp,
                                  contracts:tuple[PrecontractContractSpec, ...]) -> PackedRecordLDSStage:
  """Stage typed fields into AoS/interleaved records without imposing source-field storage equivalence."""
  factors = derive_precontract_shape_factors(geometry, tc)
  if (not isinstance(templates, tuple) or tuple(x.role for x in templates) != ("A", "B") or
      not all(isinstance(x, PackedRecordOperandTemplate) for x in templates)):
    raise ValueError("packed record templates must be exactly ordered A and B")
  if (not isinstance(regions, tuple) or tuple(x.role for x in regions) != ("A", "B") or
      not all(isinstance(x, PackedRecordLDSRegionBinding) for x in regions)):
    raise ValueError("packed record regions must map exactly ordered A and B roles")
  validate_precontract_thread_axes(geometry, factors, threads, subtile_m, subtile_n, context="packed record")
  validate_precontract_contracts(tc, contracts, context="packed record",
                                 mismatch="does not match actual descriptor operand mapping")
  if (k_axis.tile_owner.op is not Ops.RANGE or k_axis.tile_owner.arg[-1] is not AxisType.REDUCE or
      k_axis.tile_owner not in k_axis.tile_base.backward_slice_with_self):
    raise ValueError("packed record K tile owner must be a live REDUCE range in tile base")
  if (k_axis.substep_owner.op is not Ops.RANGE or k_axis.substep_owner.arg[-1] is not AxisType.UNROLL or
      k_axis.substep_owner.vmax+1 != factors.k_substeps or k_axis.substep_owner not in k_axis.substep.backward_slice_with_self):
    raise ValueError("packed record K substep owner must be a live derived-size UNROLL range in substep")
  if (allocation.op is not Ops.DEFINE_LOCAL or allocation.ptrdtype.addrspace is not AddrSpace.LOCAL or
      allocation.ptrdtype.base != dtypes.uint8 or allocation.ptrdtype.size != geometry.lds_bytes):
    raise ValueError("packed record allocation must be one exact byte-addressed LDS arena")

  role_regions: dict[str, KernelLDSArenaRegion] = {}
  for binding in regions:
    try: region = geometry.lds_region(binding.region)
    except KeyError as exc: raise ValueError(f"packed record role {binding.role} names an unknown LDS region") from exc
    if region.records is None: raise ValueError(f"packed record role {binding.role} region has no record layout")
    role_regions[binding.role] = region

  for template in templates:
    region, layout = role_regions[template.role], role_regions[template.role].records
    assert layout is not None
    rows = geometry.tile[0] if template.role == "A" else geometry.tile[1]
    if layout.rows != rows: raise ValueError(f"packed record {template.role} region row count does not match tile ownership")
    transform_fields = {x.name:x for x in template.transform.produced.components}
    layout_fields = {x.component:x for x in layout.components}
    if set(transform_fields) != set(layout_fields):
      raise ValueError(f"packed record {template.role} produced layout fields do not exactly match the LDS record layout")
    for name, component in transform_fields.items():
      field = layout_fields[name]
      if (component.offset_bytes, component.size_bytes) != (field.offset_bytes, field.size_bytes):
        raise ValueError(f"packed record {template.role} field {name!r} offset/size does not match the LDS record layout")
      if component.dtype != field.dtype:
        raise ValueError(f"packed record {template.role} field {name!r} dtype does not match the LDS record layout")
    primary = transform_fields[template.primary_field]
    fragment_dtype = primary.dtype if template.fragment_dtype is None else template.fragment_dtype
    if fragment_dtype != tc.dtype_in or primary.size_bytes < geometry.tile[2] * fragment_dtype.itemsize:
      raise ValueError(f"packed record {template.role} primary fragment view must match and cover tensor-core dtype_in")
    if region.base < 0 or region.end > geometry.lds_bytes or region.end-region.base != layout.size_bytes:
      raise ValueError(f"packed record {template.role} addresses escape the LDS region")
    for binding in template.fields:
      field = layout_fields[binding.field]
      if (binding.vector_bytes != 16 or field.size_bytes % binding.vector_bytes or
          binding.vector_bytes % field.dtype.itemsize or (region.base+field.offset_bytes) % binding.vector_bytes or
          layout.stride_bytes % binding.vector_bytes):
        raise ValueError(f"packed record {(template.role, binding.field)!r} vector width/alignment is invalid")
      if region.base + (layout.rows-1)*layout.stride_bytes + field.end_bytes > region.end:
        raise ValueError(f"packed record {(template.role, binding.field)!r} addresses escape the LDS region")

  thread = (threads.wave_m * geometry.waves[1] + threads.wave_n) * geometry.wave_size + threads.lane
  stores: list[UOp] = []
  for template in templates:
    region, layout = role_regions[template.role], role_regions[template.role].records
    assert layout is not None
    for binding in template.fields:
      field = layout.component(binding.field)
      vector_elements = binding.vector_bytes // field.dtype.itemsize
      vectors_per_row = field.size_bytes // binding.vector_bytes
      vector_count = layout.rows * vectors_per_row
      if vector_count % geometry.threads:
        raise ValueError(f"packed record {(template.role, binding.field)!r} vectors do not divide cooperative threads")
      source_ptrs = tuple(template.source(x) for x in binding.sources)
      for iteration in range(vector_count // geometry.threads):
        linear_vector = thread + iteration * geometry.threads
        row, vector = linear_vector // vectors_per_row, linear_vector % vectors_per_row
        logical_row, logical_k = template.row_tile_base + row, k_axis.tile_base + vector * vector_elements
        value = binding.producer.substitute({template.row_axis:logical_row, template.k_axis:logical_k}) \
          if isinstance(binding.producer, UOp) else binding.producer(source_ptrs, logical_row, logical_k, vector_elements)
        expected = field.dtype.vec(vector_elements)
        if not isinstance(value, UOp) or value.dtype != expected:
          raise ValueError(f"packed record {(template.role, binding.field)!r} producer has the wrong dtype/vector width")
        owned = set(value.backward_slice_with_self)
        declared = {x.pointer for x in template.sources}
        used_ptrs = {x for x in owned if x.op is Ops.PARAM and isinstance(x.dtype, PtrDType)}
        if not set(source_ptrs) <= owned or logical_row not in owned or logical_k not in owned:
          raise ValueError(f"packed record {(template.role, binding.field)!r} producer is detached from source/row/K ownership")
        if not used_ptrs <= declared:
          raise ValueError(f"packed record {(template.role, binding.field)!r} producer uses an undeclared source")
        byte_index = region.base + row*layout.stride_bytes + field.offset_bytes + vector*binding.vector_bytes
        tag = ("packed_record_store", template.role, binding.field, iteration)
        stores.append(allocation.index(byte_index, dtype=value.dtype).replace(tag=tag).store(value).replace(tag=tag).end())

  producer = UOp.group(*stores)
  barrier = UOp.barrier(producer)
  ordered, lane = allocation.after(barrier), threads.lane
  fragment_rows: dict[str, UOp] = {}
  fragments: list[UOp] = []
  for operand_idx, (template, subtile, wave, subtiles, contract) in enumerate(zip(
      templates, (subtile_m, subtile_n), (threads.wave_m, threads.wave_n), (factors.subtiles_m, factors.subtiles_n), contracts)):
    region, layout = role_regions[template.role], role_regions[template.role].records
    assert layout is not None
    field = layout.component(template.primary_field)
    row = (wave * subtiles + subtile) * tc.dims[1-operand_idx] + lane % tc.dims[1-operand_idx]
    fragment_rows[template.role] = row
    logical_k = k_axis.substep * tc.dims[2] + contract.element
    fragment_dtype = field.dtype if template.fragment_dtype is None else template.fragment_dtype
    byte_index = region.base + row*layout.stride_bytes + field.offset_bytes + logical_k*fragment_dtype.itemsize
    load = ordered.index(byte_index, dtype=fragment_dtype).replace(
      tag=("packed_record_fragment_load", template.role, template.primary_field)).load()
    fragments.append(UOp(Ops.CONTRACT, tc.dtype_in.vec(tc.elements_per_thread[operand_idx]), (load,), contract.arg,
                         tag=("packed_record_fragment", template.role, template.primary_field)))

  sidecars: list[PackedRecordSidecarView] = []
  for template in templates:
    region, layout, row = role_regions[template.role], role_regions[template.role].records, fragment_rows[template.role]
    assert layout is not None
    producer_by_name = {x.field:x for x in template.fields}
    for name in (x.field for x in template.fields if x.field != template.primary_field):
      field, binding = layout.component(name), producer_by_name[name]
      vector_elements = binding.vector_bytes // field.dtype.itemsize
      vectors = tuple(ordered.index(region.base + row*layout.stride_bytes + field.offset_bytes + vector*binding.vector_bytes,
                                    dtype=field.dtype.vec(vector_elements)).replace(
                        tag=("packed_record_sidecar_load", template.role, name, vector)).load()
                      for vector in range(field.size_bytes // binding.vector_bytes))
      sidecars.append(PackedRecordSidecarView(template.role, name, vectors))
  return PackedRecordLDSStage(allocation, producer, barrier, fragments[0], fragments[1], tuple(sidecars))


def _hierarchical_record_roles(geometry:KernelTileGeometry, descriptor:HierarchicalPackedRecordStageDescriptor,
                               templates:tuple[PackedRecordOperandTemplate, ...],
                               regions:tuple[PackedRecordLDSRegionBinding, ...]):
  if not isinstance(templates, tuple) or len(templates) != 2 or not all(isinstance(x, PackedRecordOperandTemplate) for x in templates):
    raise ValueError("hierarchical record stage requires exactly two packed record templates")
  if not isinstance(regions, tuple) or len(regions) != 2 or not all(isinstance(x, PackedRecordLDSRegionBinding) for x in regions):
    raise ValueError("hierarchical record stage requires exactly two record region bindings")
  by_role = {x.role:x for x in templates}
  region_names = {x.role:x.region for x in regions}
  names = (descriptor.plan.persistent.name, descriptor.plan.overwriteable.name)
  if set(by_role) != set(names) or set(region_names) != set(names):
    raise ValueError("hierarchical plan roles must exactly match template and region roles")
  result = []
  for role, expected_k in zip(names, (descriptor.outer_k, descriptor.phase_k)):
    template = by_role[role]
    try: region = geometry.lds_region(region_names[role])
    except KeyError as exc: raise ValueError(f"hierarchical role {role!r} names an unknown LDS region") from exc
    if region.records is None: raise ValueError(f"hierarchical role {role!r} region has no record layout")
    layout = region.records
    fields = {x.name:x for x in template.transform.produced.components}
    if set(fields) != {x.component for x in layout.components}:
      raise ValueError(f"hierarchical role {role!r} transform fields do not exactly match its record layout")
    for name, component in fields.items():
      field = layout.component(name)
      if (component.dtype, component.offset_bytes, component.size_bytes) != (field.dtype, field.offset_bytes, field.size_bytes):
        raise ValueError(f"hierarchical role {role!r} field {name!r} does not match its record layout")
    primary = layout.component(template.primary_field)
    fragment_dtype = primary.dtype if template.fragment_dtype is None else template.fragment_dtype
    if fragment_dtype != dtypes.char: raise ValueError("hierarchical primary fragment view must be char")
    if primary.size_bytes != expected_k * fragment_dtype.itemsize:
      raise ValueError(f"hierarchical role {role!r} K extent mismatch")
    if descriptor.group_k < 16: raise ValueError("group_k must cover one char.vec16 primary fragment")
    if region.base < 0 or region.end > geometry.lds_bytes or region.end-region.base != layout.size_bytes:
      raise ValueError(f"hierarchical role {role!r} region addresses escape LDS")
    if region.base + (layout.rows-1)*layout.stride_bytes + layout.stride_bytes > region.end:
      raise ValueError(f"hierarchical role {role!r} record addresses escape LDS")
    result.append((template, region, layout, expected_k))
  return tuple(result)


def build_hierarchical_packed_record_stage(geometry:KernelTileGeometry, *, allocation:UOp,
                                           descriptor:HierarchicalPackedRecordStageDescriptor,
                                           templates:tuple[PackedRecordOperandTemplate, ...],
                                           regions:tuple[PackedRecordLDSRegionBinding, ...], threads:PrecontractThreadAxes,
                                           subtile_m:UOp, subtile_n:UOp, tc,
                                           contracts:tuple[PrecontractContractSpec, ...], verify:bool=True) -> HierarchicalPackedRecordStage:
  """Build the exact publish/consume/release protocol for two record-backed role lifetimes."""
  if not isinstance(descriptor, HierarchicalPackedRecordStageDescriptor): raise TypeError("expected hierarchical record descriptor")
  if (allocation.op is not Ops.DEFINE_LOCAL or allocation.ptrdtype.addrspace is not AddrSpace.LOCAL or
      allocation.ptrdtype.base != dtypes.uint8 or allocation.ptrdtype.size != geometry.lds_bytes):
    raise ValueError("hierarchical record allocation must be one exact byte-addressed LDS arena")
  persistent, overwriteable = _hierarchical_record_roles(geometry, descriptor, templates, regions)
  validate_rdna3_wmma_descriptor(tc)
  if tc.dtype_in != dtypes.char: raise ValueError("hierarchical record stage requires the int8 RDNA3 descriptor")
  validate_precontract_contracts(tc, contracts, context="hierarchical packed record",
                                 mismatch="does not match actual descriptor operand mapping")
  contract_by_role = {x.role:x for x in contracts}
  subtiles_m, rem_m = divmod(geometry.tile[0], geometry.waves[0]*16)
  subtiles_n, rem_n = divmod(geometry.tile[1], geometry.waves[1]*16)
  if rem_m or rem_n or subtiles_m <= 0 or subtiles_n <= 0: raise ValueError("hierarchical tile does not divide wave/subtile ownership")
  if ((threads.wave_m.op, threads.wave_m.vmax+1, threads.wave_m.arg[-1]) != (Ops.RANGE, geometry.waves[0], AxisType.LOCAL) or
      (threads.wave_n.op, threads.wave_n.vmax+1, threads.wave_n.arg[-1]) != (Ops.RANGE, geometry.waves[1], AxisType.LOCAL) or
      (threads.lane.op, threads.lane.vmax+1, threads.lane.arg[-1]) != (Ops.RANGE, geometry.wave_size, AxisType.WARP)):
    raise ValueError("hierarchical record thread axes do not match wave geometry")
  if (subtile_m.op is not Ops.RANGE or subtile_m.vmax+1 != subtiles_m or
      subtile_n.op is not Ops.RANGE or subtile_n.vmax+1 != subtiles_n):
    raise ValueError("hierarchical record subtile axes do not match geometry")
  thread = (threads.wave_m*geometry.waves[1]+threads.wave_n)*geometry.wave_size+threads.lane

  def order_value_sources(value:UOp, sources:tuple[UOp, ...], dependency:UOp) -> UOp:
    # UOp.substitute recursively visits replacement graphs.  Here those graphs intentionally contain the prior
    # STORE (and therefore the same source pointer), so rebuild only the original value slice to avoid a false cycle.
    replaced = {source:source.after(dependency) for source in sources}
    for node in value.toposort():
      if node in replaced: continue
      src = tuple(replaced.get(x, x) for x in node.src)
      replaced[node] = node if src == node.src else node.replace(src=src)
    return replaced[value]

  def produce(role_data, source_k:int, dependency:UOp|None, phase:int|None) -> UOp:
    template, region, layout, _ = role_data
    stores, transaction_dependency = [], dependency
    if template.cooperative_schedule is None:
      raise ValueError(f"hierarchical role {template.role!r} requires an explicit cooperative schedule")
    emissions = template.cooperative_schedule.callback(template, threads, source_k)
    if not isinstance(emissions, tuple) or not all(isinstance(x, PackedRecordCooperativeStore) for x in emissions):
      raise TypeError(f"hierarchical role {template.role!r} cooperative schedule returned invalid stores")
    bindings = {x.field:x for x in template.fields}
    covered: dict[str, set[tuple[int, int]]] = {x.field:set() for x in template.fields}
    for emission in emissions:
      if emission.field not in bindings: raise ValueError(f"hierarchical schedule emitted undeclared field {emission.field!r}")
      binding = bindings[emission.field]
      field = layout.component(binding.field)
      if (field.size_bytes % binding.vector_bytes or binding.vector_bytes % field.dtype.itemsize or
          (region.base+field.offset_bytes) % binding.vector_bytes or layout.stride_bytes % binding.vector_bytes):
        raise ValueError(f"hierarchical field {(template.role, binding.field)!r} vector width is invalid")
      source_ptrs = tuple(template.source(x) for x in binding.sources)
      width = binding.vector_bytes // field.dtype.itemsize
      value = emission.value
      if value.dtype != field.dtype.vec(width):
        raise ValueError(f"hierarchical field {(template.role, binding.field)!r} producer has the wrong dtype/vector width")
      owned = set(value.backward_slice_with_self)
      if not set(source_ptrs) <= owned or emission.source_row not in owned or \
         not set(template.cooperative_schedule.axes(threads)) <= owned:
        raise ValueError(f"hierarchical field {(template.role, binding.field)!r} producer is detached from source/row/K/thread ownership")
      vectors_per_row = field.size_bytes//binding.vector_bytes
      if (emission.destination_row.vmin < 0 or emission.destination_row.vmax >= layout.rows or
          emission.destination_vector.vmin < 0 or emission.destination_vector.vmax >= vectors_per_row):
        raise ValueError(f"hierarchical field {(template.role, binding.field)!r} schedule destination escapes field")
      for row in range(emission.destination_row.vmin, emission.destination_row.vmax+1):
        for vector in range(emission.destination_vector.vmin, emission.destination_vector.vmax+1):
          # Ranges used by this vocabulary are affine/elective; cardinality is checked again by the proof's exact addresses.
          covered[binding.field].add((row, vector))
      address = UOp.const(dtypes.weakint, region.base+field.offset_bytes)+emission.destination_row*layout.stride_bytes+\
                emission.destination_vector*binding.vector_bytes
      # Keep each cooperative load/decode/address transaction live only until its LDS store.  GROUP remains the
      # producer's final effect join, while this ordered chain prevents independent transactions from overlapping.
      ordered_allocation = allocation
      if transaction_dependency is not None:
        # Order both sides of the transaction through pointers, which are stable legal AFTER carriers through
        # symbolic rewriting.  Ordering an ALU value directly is invalid, and a CONTIGUOUS wrapper is simplified
        # away before program verification.  Substitution retains the original source pointer under each AFTER.
        ordered_allocation = allocation.after(transaction_dependency)
        value = order_value_sources(value, source_ptrs, transaction_dependency)
      tag = ("hierarchical_record_store", template.role, binding.field, phase, emission.iteration,
             template.cooperative_schedule.name)
      store = ordered_allocation.index(address, dtype=value.dtype).replace(tag=tag).store(value).replace(tag=tag).end()
      stores.append(store)
      transaction_dependency = store
    for binding in template.fields:
      field = layout.component(binding.field)
      expected = {(row, vector) for row in range(layout.rows) for vector in range(field.size_bytes//binding.vector_bytes)}
      if covered[binding.field] != expected:
        raise ValueError(f"hierarchical field {(template.role, binding.field)!r} cooperative schedule does not cover exact destinations")
    return UOp.group(*stores).replace(tag=("hierarchical_record_producer", template.role, phase))

  persistent_producer = produce(persistent, 0, None, None)
  previous_release, phases = None, []
  for phase in range(descriptor.plan.phase_count):
    phase_producer = produce(overwriteable, phase*descriptor.phase_k, previous_release, phase)
    publish = UOp.barrier(UOp.group(persistent_producer, phase_producer)).replace(tag=("hierarchical_publish", phase))
    ordered = allocation.after(publish)
    groups = []
    for group in range(descriptor.groups_per_phase):
      views, consumed = [], []
      for role_data, logical_k in ((persistent, phase*descriptor.phase_k+group*descriptor.group_k),
                                   (overwriteable, group*descriptor.group_k)):
        template, region, layout, role_k = role_data
        primary = layout.component(template.primary_field)
        contract = contract_by_role[template.role]
        if template.role == "A": row = (threads.wave_m*subtiles_m+subtile_m)*16+threads.lane%16
        else: row = (threads.wave_n*subtiles_n+subtile_n)*16+threads.lane%16
        address = UOp.const(dtypes.weakint, region.base+primary.offset_bytes+logical_k)+row*layout.stride_bytes+contract.element
        if logical_k < 0 or logical_k+16 > role_k:
          raise ValueError(f"hierarchical role {template.role!r} group primary address escapes its K extent")
        load = ordered.index(address, dtype=dtypes.char).replace(
          tag=("hierarchical_record_fragment_load", template.role, phase, group, logical_k)).load()
        fragment = UOp(Ops.CONTRACT, dtypes.char.vec(16), (load,), contract.arg,
                       tag=("hierarchical_record_fragment", template.role, phase, group, logical_k))
        consumed.append(fragment)
        sidecars = []
        for binding in template.fields:
          if binding.field == template.primary_field: continue
          field = layout.component(binding.field)
          start_num, size_num = logical_k*field.size_bytes, descriptor.group_k*field.size_bytes
          if start_num % role_k or size_num % role_k: raise ValueError(f"hierarchical sidecar {binding.field!r} does not divide group K")
          offset, size = start_num//role_k, size_num//role_k
          if size <= 0 or size % field.dtype.itemsize: raise ValueError(f"hierarchical sidecar {binding.field!r} has an invalid typed group extent")
          side_address = UOp.const(dtypes.weakint, region.base+field.offset_bytes+offset)+row*layout.stride_bytes
          if offset < 0 or offset+size > field.size_bytes:
            raise ValueError(f"hierarchical sidecar {binding.field!r} group address escapes its field")
          value = ordered.index(side_address, dtype=field.dtype.vec(size//field.dtype.itemsize)).replace(
            tag=("hierarchical_record_sidecar", template.role, binding.field, phase, group, logical_k)).load()
          sidecars.append(HierarchicalPackedRecordSidecar(template.role, binding.field, side_address, size, value))
          consumed.append(value)
        views.append((row, address, fragment, tuple(sidecars)))
      groups.append(HierarchicalPackedRecordGroup(phase, group, phase*descriptor.phase_k+group*descriptor.group_k,
        group*descriptor.group_k, views[0][0], views[1][0], views[0][1], views[1][1], views[0][2], views[1][2], views[0][3]+views[1][3]))
    release = UOp.barrier(UOp.group(*(x for group_record in groups for x in
      (group_record.persistent_fragment, group_record.overwriteable_fragment, *(side.value for side in group_record.sidecars))))).replace(
        tag=("hierarchical_release", phase))
    phases.append(HierarchicalPackedRecordPhase(phase, phase_producer, publish, tuple(groups), release))
    previous_release = release
  stage = HierarchicalPackedRecordStage(descriptor, geometry, tc, contracts, templates, regions, threads, subtile_m, subtile_n,
                                        allocation, persistent_producer, tuple(phases),
                                        hierarchical_lifecycle_events(descriptor.plan))
  if verify and not (proof := prove_hierarchical_packed_record_stage(stage)).passed:
    raise ValueError("invalid hierarchical packed record stage: " + "; ".join(proof.errors))
  return stage


def prove_hierarchical_packed_record_stage(stage:HierarchicalPackedRecordStage) -> HierarchicalPackedRecordStageProof:
  """Fail closed over lifecycle order, role extents, and every group-specific view."""
  if not isinstance(stage, HierarchicalPackedRecordStage): raise TypeError("expected HierarchicalPackedRecordStage")
  errors = list(prove_hierarchical_lifecycle(stage.descriptor.plan, stage.events).errors)
  descriptor = stage.descriptor
  try:
    validate_rdna3_wmma_descriptor(stage.tc)
    validate_precontract_contracts(stage.tc, stage.contracts, context="hierarchical packed record proof")
  except (TypeError, ValueError) as exc: errors.append(str(exc))
  contract_by_role = {x.role:x for x in stage.contracts if isinstance(x, PrecontractContractSpec)}
  template_by_role = {x.role:x for x in stage.templates if isinstance(x, PackedRecordOperandTemplate)}
  region_by_role = {}
  for binding in stage.regions:
    try: region_by_role[binding.role] = stage.geometry.lds_region(binding.region)
    except (AttributeError, KeyError): errors.append(f"role {getattr(binding, 'role', '?')}: invalid region binding")
  def load_address(value:UOp) -> UOp|None:
    try:
      load = value.src[0] if value.op is Ops.CONTRACT else value
      index = load.src[0]
      return index.src[1] if index.op is Ops.INDEX else None
    except (AttributeError, IndexError): return None
  if len(stage.phases) != descriptor.plan.phase_count: errors.append("phase count mismatch")
  if getattr(stage.persistent_producer, "tag", None) != ("hierarchical_record_producer", descriptor.plan.persistent.name, None):
    errors.append("persistent role must be produced exactly once")
  for role, producer in ((descriptor.plan.persistent.name, stage.persistent_producer),
                         *((descriptor.plan.overwriteable.name, x.producer) for x in stage.phases)):
    stores = [x for x in producer.src if x.op is Ops.STORE and
              isinstance(x.tag, tuple) and x.tag[:2] == ("hierarchical_record_store", role)]
    template = template_by_role.get(role)
    required_thread_axes = set(template.cooperative_schedule.axes(stage.threads)) if template is not None and \
      template.cooperative_schedule is not None else {stage.threads.lane}
    if not stores or any(not required_thread_axes <= set(x.backward_slice_with_self) for x in stores):
      errors.append(f"role {role}: detached or noncooperative thread ownership")
    region, template = region_by_role.get(role), template_by_role.get(role)
    if region is None or region.records is None or template is None:
      errors.append(f"role {role}: record ownership metadata mismatch")
    elif template.cooperative_schedule is None:
      errors.append(f"role {role}: missing explicit cooperative schedule")
    else:
      phase = producer.tag[2]
      source_k = 0 if phase is None else phase*descriptor.phase_k
      try: emissions = template.cooperative_schedule.callback(template, stage.threads, source_k)
      except Exception as exc: errors.append(f"role {role}: cooperative schedule callback failed: {exc}")
      else:
        expected = []
        bindings = {x.field:x for x in template.fields}
        for emission in emissions:
          binding = bindings.get(emission.field)
          if binding is None: continue
          field = region.records.component(emission.field)
          address = UOp.const(dtypes.weakint, region.base+field.offset_bytes)+emission.destination_row*region.records.stride_bytes+\
                    emission.destination_vector*binding.vector_bytes
          expected.append((emission.field, emission.iteration, address))
        if len(stores) != len(expected): errors.append(f"role {role}: cooperative store count mismatch")
        for store, (field, iteration, address) in zip(stores, expected):
          if store.tag != ("hierarchical_record_store", role, field, phase, iteration, template.cooperative_schedule.name):
            errors.append(f"role {role}: cooperative schedule identity mismatch")
          try: actual_address = store.src[0].src[1]
          except (AttributeError, IndexError): actual_address = None
          if actual_address is not None and actual_address.op is Ops.AFTER: actual_address = actual_address.src[0]
          if actual_address is not address: errors.append(f"role {role} field {field}: noncooperative store address")
  for phase_index, phase in enumerate(stage.phases):
    if phase.phase != phase_index: errors.append(f"phase {phase_index}: phase ordinal mismatch")
    if getattr(phase.producer, "tag", None) != ("hierarchical_record_producer", descriptor.plan.overwriteable.name, phase_index):
      errors.append(f"phase {phase_index}: overwriteable production mismatch")
    if phase_index and stage.phases[phase_index-1].release not in phase.producer.backward_slice:
      errors.append(f"phase {phase_index}: overwrite occurs before prior release")
    if phase.producer not in phase.publish.backward_slice or stage.persistent_producer not in phase.publish.backward_slice:
      errors.append(f"phase {phase_index}: missing publish barrier")
    if len(phase.groups) != descriptor.groups_per_phase: errors.append(f"phase {phase_index}: group count mismatch")
    for group_index, group in enumerate(phase.groups):
      expected_p = phase_index*descriptor.phase_k + group_index*descriptor.group_k
      expected_o = group_index*descriptor.group_k
      if (group.phase, group.group, group.persistent_k, group.overwriteable_k) != (phase_index, group_index, expected_p, expected_o):
        errors.append(f"phase {phase_index} group {group_index}: K address mismatch")
      if group.persistent_fragment.dtype != dtypes.char.vec(16) or group.overwriteable_fragment.dtype != dtypes.char.vec(16):
        errors.append(f"phase {phase_index} group {group_index}: primary fragment carrier mismatch")
      for role, fragment, row in ((descriptor.plan.persistent.name, group.persistent_fragment, group.persistent_row),
                                  (descriptor.plan.overwriteable.name, group.overwriteable_fragment, group.overwriteable_row)):
        contract = contract_by_role.get(role)
        if fragment.op is not Ops.CONTRACT or contract is None or fragment.arg != contract.arg:
          errors.append(f"phase {phase_index} group {group_index}: {role} descriptor contract mismatch")
        subtiles_m = stage.geometry.tile[0]//(stage.geometry.waves[0]*16)
        subtiles_n = stage.geometry.tile[1]//(stage.geometry.waves[1]*16)
        expected_row = ((stage.threads.wave_m*subtiles_m+stage.subtile_m)*16+stage.threads.lane%16 if role == "A" else
                        (stage.threads.wave_n*subtiles_n+stage.subtile_n)*16+stage.threads.lane%16)
        expected_axes = ({stage.threads.wave_m, stage.subtile_m, stage.threads.lane} if role == "A" else
                         {stage.threads.wave_n, stage.subtile_n, stage.threads.lane})
        if row is not expected_row or not expected_axes <= set(row.backward_slice_with_self) or not expected_axes <= set(fragment.backward_slice_with_self):
          errors.append(f"phase {phase_index} group {group_index}: {role} detached wave/subtile ownership")
      if load_address(group.persistent_fragment) is not group.persistent_byte_address or \
         load_address(group.overwriteable_fragment) is not group.overwriteable_byte_address:
        errors.append(f"phase {phase_index} group {group_index}: primary group address escape")
      for role, address in ((descriptor.plan.persistent.name, group.persistent_byte_address),
                            (descriptor.plan.overwriteable.name, group.overwriteable_byte_address)):
        region = region_by_role.get(role)
        if region is None or address.vmin < region.base or address.vmax+1 > region.end:
          errors.append(f"phase {phase_index} group {group_index}: primary address escapes role region")
      for side in group.sidecars:
        if side.byte_size != side.value.dtype.itemsize: errors.append(f"phase {phase_index} group {group_index}: sidecar typed extent mismatch")
        if load_address(side.value) is not side.byte_address: errors.append(f"phase {phase_index} group {group_index}: sidecar group address escape")
        region = region_by_role.get(side.role)
        if region is None or side.byte_address.vmin < region.base or side.byte_address.vmax+side.byte_size > region.end:
          errors.append(f"phase {phase_index} group {group_index}: sidecar address escapes role region")
        side_row = group.persistent_row if side.role == descriptor.plan.persistent.name else group.overwriteable_row
        expected_axes = ({stage.threads.wave_m, stage.subtile_m, stage.threads.lane} if side.role == "A" else
                         {stage.threads.wave_n, stage.subtile_n, stage.threads.lane})
        if not expected_axes <= set(side_row.backward_slice_with_self) or not expected_axes <= set(side.value.backward_slice_with_self):
          errors.append(f"phase {phase_index} group {group_index}: sidecar detached row ownership")
        if phase.publish not in side.value.backward_slice: errors.append(f"phase {phase_index} group {group_index}: sidecar missing publish dependency")
    consumed = (x for group in phase.groups for x in (group.persistent_fragment, group.overwriteable_fragment,
                *(side.value for side in group.sidecars)))
    if any(x not in phase.release.backward_slice for x in consumed): errors.append(f"phase {phase_index}: missing release barrier")
  if len(stage.barriers) != descriptor.plan.phase_count*2: errors.append("barrier count mismatch")
  return HierarchicalPackedRecordStageProof(not errors, tuple(errors))
