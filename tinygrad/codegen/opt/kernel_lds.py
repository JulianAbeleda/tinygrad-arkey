"""Pure cooperative LDS ownership math for compiler-bound kernel geometry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeAlias, TYPE_CHECKING

from tinygrad.codegen.opt.packed_weight import PackedWeightTransform
from tinygrad.dtype import AddrSpace, PtrDType, dtypes
from tinygrad.uop.ops import AxisType, Ops, UOp
if TYPE_CHECKING: from tinygrad.uop.ops import KernelLDSWindow, KernelTileGeometry

_RDNA3_DIMS = (16, 16, 16)
_RDNA3_ELEMENTS = (16, 16, 8)
_RDNA3_OPTS = ("l0", "l0", "l0", "l0", "l1", "u1", "u1", "u1")
_RDNA3_SWIZZLE = ((('l4', 'u0', 'u1', 'u2', 'l0'), ('r1', 'r2', 'r3'), ('l1', 'l2', 'l3', 'r0')),
                  (('l0', 'l1', 'l2', 'l3', 'l4'), ('r1', 'r2', 'r3'), ('u0', 'u1', 'u2', 'r0')))
_RDNA3_REMAPS = ({'l0': 'l4', 'l1': 'u0', 'l2': 'u1', 'l3': 'u2', 'l4': 'l0', 'u0': 'r1', 'u1': 'r2', 'u2': 'r3',
                   'r0': 'l1', 'r1': 'l2', 'r2': 'l3', 'r3': 'r0'},
                  {'l0': 'l0', 'l1': 'l1', 'l2': 'l2', 'l3': 'l3', 'l4': 'l4', 'u0': 'r1', 'u1': 'r2', 'u2': 'r3',
                   'r0': 'u0', 'r1': 'u1', 'r2': 'u2', 'r3': 'r0'})


def validate_rdna3_wmma_descriptor(tc) -> None:
  """Admit only the exact fp16->fp32 and int8->int32 descriptors this mapping proves."""
  fields = (("dims", _RDNA3_DIMS), ("threads", 32), ("elements_per_thread", _RDNA3_ELEMENTS),
            ("opts", _RDNA3_OPTS), ("swizzle", _RDNA3_SWIZZLE))
  for name, expected in fields:
    if getattr(tc, name, None) != expected: raise ValueError(f"RDNA3 WMMA descriptor {name} drifted")
  dtype_in, dtype_out = getattr(tc, "dtype_in", None), getattr(tc, "dtype_out", None)
  if dtype_in not in (dtypes.half, dtypes.char): raise ValueError("RDNA3 WMMA descriptor dtype_in drifted")
  if dtype_out != (dtypes.float if dtype_in == dtypes.half else dtypes.int):
    raise ValueError("RDNA3 WMMA descriptor dtype_out drifted")
  try: remaps = tuple(tc.lane_map.remaps())
  except (AttributeError, AssertionError, TypeError, ValueError) as exc:
    raise ValueError("RDNA3 WMMA descriptor remaps are unavailable or invalid") from exc
  if remaps != _RDNA3_REMAPS: raise ValueError("RDNA3 WMMA descriptor remaps drifted")


def contract_symbolic_upcast(value:UOp, axis:UOp) -> UOp:
  """Materialize one scalar value over its owned UPCAST axis as a legal vector carrier."""
  if axis.op is not Ops.RANGE or axis.arg[-1] is not AxisType.UPCAST or axis.vmin != 0:
    raise ValueError("symbolic contraction requires a zero-based UPCAST range")
  if value.dtype is dtypes.void or value.dtype.count != 1: raise ValueError("symbolic contraction requires a non-void scalar value")
  if axis not in value.backward_slice_with_self: raise ValueError("symbolic contraction value does not own the requested axis")
  width = axis.vmax+1
  return UOp(Ops.CONTRACT, value.dtype.vec(width), (value,), ((axis.arg[0], width),))


def lower_symbolic_barrier_dependencies(root:UOp, axis:UOp) -> UOp:
  """Contract scalar UPCAST values before they cross an effect barrier.

  Effect barriers preserve ordering, but must not retain scalar-shaped values over
  an upcast axis: those otherwise survive expansion as illegal program UNROLLs.
  """
  if axis.op is not Ops.RANGE or axis.arg[-1] is not AxisType.UPCAST or axis.vmin != 0:
    raise ValueError("symbolic barrier lowering requires a zero-based UPCAST range")
  lowered: dict[UOp, UOp] = {}
  for node in root.toposort():
    src = tuple(lowered[x] for x in node.src)
    if node.op is Ops.BARRIER:
      src = tuple(contract_symbolic_upcast(x, axis) if x.dtype is not dtypes.void and x.dtype.count == 1 and
                  axis in x.backward_slice_with_self else x for x in src)
    lowered[node] = node if src == node.src else node.replace(src=src)
  return lowered[root]


@dataclass(frozen=True)
class PrecontractOperandTemplate:
  role: str
  source: UOp
  row_axis: UOp
  k_axis: UOp
  row_tile_base: UOp


@dataclass(frozen=True)
class PackedPrecontractOperandTemplate:
  """Packed B source decoded to fp16 at cooperative tile-production coordinates."""
  role: str
  source: UOp
  transform: PackedWeightTransform
  row_axis: UOp
  k_axis: UOp
  row_tile_base: UOp


PrecontractOperand: TypeAlias = PrecontractOperandTemplate | PackedPrecontractOperandTemplate

@dataclass(frozen=True)
class PrecontractThreadAxes:
  wave_m: UOp
  wave_n: UOp
  lane: UOp

@dataclass(frozen=True)
class PrecontractKAxis:
  tile_owner: UOp
  substep_owner: UOp
  tile_base: UOp
  substep: UOp

@dataclass(frozen=True)
class PrecontractContractSpec:
  role: str
  axes: tuple[UOp, ...]
  arg: tuple[tuple[int, int], ...]
  element: UOp
  descriptor_remap: tuple[tuple[str, str], ...]

@dataclass(frozen=True)
class PrecontractLDSStage:
  allocation: UOp
  producer: UOp
  barrier: UOp
  fragment_a: UOp
  fragment_b: UOp






@dataclass(frozen=True)
class PrecontractProducerInstance:
  epoch: UOp
  slot: UOp
  role_nodes: tuple[UOp, UOp]

@dataclass(frozen=True)
class PrecontractFragmentInstance:
  epoch: UOp
  slot: UOp
  ready: UOp
  fragments: tuple[UOp, UOp]

@dataclass(frozen=True)
class PrecontractFactors:
  subtiles_m: int
  subtiles_n: int
  waves_m: int
  waves_n: int
  k_substeps: int
  vectors_per_row: int
  loads_a: int
  loads_b: int


@dataclass(frozen=True)
class PrecontractCandidateContract:
  """Single owner of tensor-core candidate geometry, storage, operand, and CONTRACT assembly."""
  context: object; tc: object
  factors: PrecontractFactors; register_mode: bool

  @property
  def pipeline(self): return getattr(self.context, "pipeline", None)

  @classmethod
  def create(cls, context:object, tc) -> PrecontractCandidateContract:
    geometry = getattr(context, "geometry", None)
    if geometry is None: raise ValueError("precontract candidate requires explicit geometry")
    pipeline = getattr(context, "pipeline", None)
    register_hint = getattr(getattr(pipeline, "storage", None), "kind", None) == "global_register_resident"
    factors = derive_precontract_shape_factors(geometry, tc) if register_hint else derive_precontract_factors(geometry, tc)
    register_mode = False
    if pipeline is not None:
      from tinygrad.codegen.opt.kernel_pipeline import pipeline_policy_from_candidate
      policy = pipeline_policy_from_candidate(pipeline)
      if policy.storage_kind != "lds":
        coverage = getattr(pipeline, "wait_coverage", None)
        if coverage is None or not coverage.passed: raise ValueError("register-resident candidate lacks proven wait dependency coverage")
      register_mode = policy.storage_kind == "global_register_resident"
      if register_mode != register_hint: raise ValueError("candidate storage policy disagrees with geometry contract")
    return cls(context, tc, factors, register_mode)

  def assemble(self, *, in0:UOp, in1:UOp, original_axes:tuple[UOp, UOp, UOp], outer_n:UOp, outer_m:UOp,
               wave_m:UOp, wave_n:UOp, lane:UOp, tc_upcast_axes:tuple[tuple[tuple[int, int], ...], ...],
               range_by_id:dict[int, UOp], allocation_id:Callable[[], int]|None
               ) -> tuple[tuple[PrecontractOperand, ...], PrecontractThreadAxes, tuple[PrecontractContractSpec, ...], UOp|None]:
    geometry, tc = self.context.geometry, self.tc
    contracts = []
    for operand_idx, role in enumerate(("A", "B")):
      axes = tuple(range_by_id[a] for a, size in tc_upcast_axes[operand_idx] if size == 2)
      if len(axes) != 4: raise ValueError(f"candidate {role} contract does not have four binary axes")
      element = ((axes[0]*2+axes[1])*2+axes[2])*2+axes[3]
      contracts.append(PrecontractContractSpec(role, axes, tc_upcast_axes[operand_idx], element,
                                               tuple(tc.lane_map.remaps()[operand_idx].items())))
    contracts = tuple(contracts)
    validate_precontract_contracts(tc, contracts, context="candidate", mismatch="does not match actual descriptor operand mapping")

    # Preserve descriptor, allocation, then operand UOp creation order: linearization tie-breaks follow graph insertion order.
    allocation = None
    if not self.register_mode:
      total_bytes = geometry.lds_windows[-1].end if self.pipeline is None else self.pipeline.active_lds_bytes
      if allocation_id is None: raise ValueError("LDS candidate requires an allocation ID owner")
      tag = ("kernel_tile_lds", geometry) if self.pipeline is None else ("kernel_tile_lds", geometry, self.pipeline)
      allocation = UOp.placeholder((total_bytes//tc.dtype_in.itemsize,), tc.dtype_in, allocation_id(), addrspace=AddrSpace.LOCAL).replace(tag=tag)

    operand_a = PrecontractOperandTemplate("A", in0, original_axes[1], original_axes[2], outer_m*geometry.tile[0])
    packed_weight = getattr(self.context, "packed_weight", None)
    if packed_weight is None:
      operand_b:PrecontractOperand = PrecontractOperandTemplate("B", in1, original_axes[0], original_axes[2], outer_n*geometry.tile[1])
    else:
      if self.register_mode: raise ValueError("packed-weight candidate requires LDS tile storage")
      if (original_axes[0].vmax+1, original_axes[2].vmax+1) != (packed_weight.rows, packed_weight.k): raise ValueError(
        "packed-weight candidate row/K ownership does not match admitted transform")
      packed_params = [u for u in in1.toposort() if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType) and
                       u.ptrdtype.base == packed_weight.storage_dtype]
      if len(packed_params) != 1: raise ValueError(f"packed-weight B carrier must reach exactly one canonical packed PARAM, found {len(packed_params)}")
      if getattr(packed_params[0].arg, "slot", packed_params[0].arg) != 2: raise ValueError(
        f"packed-weight B carrier must own ABI slot 2, got PARAM {packed_params[0].arg!r}")
      if any(u.op is Ops.PARAM and isinstance(u.dtype, PtrDType) and u.ptrdtype.base == dtypes.half for u in in1.toposort()): raise ValueError(
        "packed-weight B carrier unexpectedly reaches a dense fp16 PARAM")
      operand_b = PackedPrecontractOperandTemplate("B", packed_params[0], packed_weight, original_axes[0], original_axes[2], outer_n*geometry.tile[1])
    operands:tuple[PrecontractOperand, ...] = (operand_a, operand_b)
    validate_precontract_operand_templates(operands, dtype_in=tc.dtype_in, context="candidate")
    return operands, PrecontractThreadAxes(wave_m, wave_n, lane), contracts, allocation


def derive_precontract_shape_factors(geometry:KernelTileGeometry, tc) -> PrecontractFactors:
  """Derive WMMA tile factors without consulting any storage allocation.

  This is the shared geometry contract for LDS and register-resident producers.
  ``derive_precontract_factors`` below adds the LDS-window checks needed by the
  legacy staged implementation.
  """
  validate_rdna3_wmma_descriptor(tc)
  tm, tn, tk = geometry.tile
  if (tm % (geometry.waves[0] * tc.dims[1]) or tn % (geometry.waves[1] * tc.dims[0]) or
      tk % tc.dims[2]):
    raise ValueError("tile must divide into whole per-wave tensor-core subtiles and K steps")
  sm = tm // (geometry.waves[0] * tc.dims[1])
  sn = tn // (geometry.waves[1] * tc.dims[0])
  ks = tk // tc.dims[2]
  if ks < 2:
    raise ValueError("current atomic staging requires at least two tensor-core K steps")
  vectors_per_row = tk * tc.dtype_in.itemsize // 16
  if vectors_per_row <= 0 or tk * tc.dtype_in.itemsize % 16:
    raise ValueError("K row must contain whole b128 vectors")
  rows = (tm, tn)
  loads = tuple(row * vectors_per_row // geometry.threads for row in rows)
  if any(row * vectors_per_row % geometry.threads for row in rows) or any(x <= 0 for x in loads):
    raise ValueError("operand vectors must divide evenly across cooperative threads")
  return PrecontractFactors(sm, sn, *geometry.waves, ks, vectors_per_row, *loads)


def validate_precontract_operand_templates(operands:tuple[PrecontractOperand, ...], *, dtype_in=dtypes.half,
                                           context:str="precontract") -> None:
  """Validate source dtype and live row/K ownership independent of storage."""
  if tuple(x.role for x in operands) != ("A", "B"):
    raise ValueError(f"{context} operands must be exactly ordered A and B")
  for operand in operands:
    if operand.row_axis.op is not Ops.RANGE or operand.k_axis.op is not Ops.RANGE:
      raise ValueError(f"{context} {operand.role} template does not retain row/K ownership")
    if isinstance(operand, PackedPrecontractOperandTemplate):
      if dtype_in != dtypes.half:
        raise ValueError(f"{context} packed templates currently produce only scalar fp16 values")
      if (operand.role != "B" or not isinstance(operand.source.dtype, PtrDType) or
          operand.source.ptrdtype.base != operand.transform.storage_dtype):
        raise ValueError(f"{context} packed template must be a B operand with canonical packed storage dtype")
      # The packed carrier no longer contains the dense source expression, so
      # these two ranges are the only remaining proof of logical ownership.
      # Keep the transform and carrier bounds in the same contract as the
      # producer: accepting a detached/partial domain would silently decode a
      # different row or read past the packed allocation.
      if (operand.row_axis.vmax + 1 != operand.transform.rows or
          operand.k_axis.vmax + 1 != operand.transform.k):
        raise ValueError(f"{context} packed B row/K ownership does not match the transform")
      packed_units = operand.transform.packed_bytes // operand.transform.storage_width
      if operand.source.ptrdtype.size != packed_units:
        raise ValueError(f"{context} packed B carrier does not exactly cover the transform")
    elif (operand.row_axis not in operand.source.backward_slice_with_self or
          operand.k_axis not in operand.source.backward_slice_with_self or
          operand.source.dtype.scalar() != dtype_in):
      raise ValueError(f"{context} {operand.role} template does not retain scalar {dtype_in.name} row/K ownership")


def validate_precontract_contracts(tc, contracts:tuple[PrecontractContractSpec, ...], *,
                                   context:str="precontract", mismatch:str="does not match the descriptor") -> None:
  """Validate A/B CONTRACT axes, folded element identity, and descriptor remaps."""
  if tuple(c.role for c in contracts) != ("A", "B"):
    raise ValueError(f"{context} contracts must be exactly ordered A and B")
  descriptor_remaps = tuple(tuple(x.items()) for x in tc.lane_map.remaps())
  for operand_idx, contract in enumerate(contracts):
    folded = ((contract.axes[0] * 2 + contract.axes[1]) * 2 + contract.axes[2]) * 2 + contract.axes[3] \
      if len(contract.axes) == 4 else None
    if (len(contract.axes) != 4 or any(a.op is not Ops.RANGE or a.vmax + 1 != 2 for a in contract.axes) or
        contract.arg != tuple((a.arg[0], 2) for a in contract.axes) or contract.element is not folded or
        contract.descriptor_remap != descriptor_remaps[operand_idx]):
      raise ValueError(f"{context} {contract.role} contract {mismatch}")


def validate_precontract_carriers(fragment_dtype, accumulator_dtype, *, tc=None, context:str="precontract") -> None:
  """Validate the stable WMMA fragment and accumulator carrier ABI."""
  if tc is not None: validate_rdna3_wmma_descriptor(tc)
  dtype_in, dtype_out, elements = (dtypes.half, dtypes.float, _RDNA3_ELEMENTS) if tc is None else \
    (tc.dtype_in, tc.dtype_out, tc.elements_per_thread)
  expected_fragments = (dtype_in.vec(elements[0]), dtype_in.vec(elements[1]))
  if fragment_dtype not in expected_fragments:
    raise ValueError(f"{context} fragment carrier must match the tensor-core input carrier")
  if accumulator_dtype != dtype_out.vec(elements[2]):
    raise ValueError(f"{context} accumulator carrier must match the tensor-core output carrier")


def validate_precontract_wmma_abi(node: UOp, *, context: str = "precontract") -> None:
  """Validate the WMMA node ABI before a backend/devectorizer sees it.

  The tensor-core matcher accepts two descriptor-sized input fragments and one
  descriptor-sized accumulator, producing the same output carrier.  The argument carries the corresponding four binary A/B axes and
  three binary C axes.  Keep this check storage-independent so LDS and
  register-resident templates cannot drift into different ABI rules.
  """
  if not isinstance(node, UOp) or node.op is not Ops.WMMA:
    raise ValueError(f"{context} WMMA ABI validator requires an Ops.WMMA node")
  if len(node.src) != 3:
    raise ValueError(f"{context} WMMA ABI requires A, B, and C inputs")
  arg = node.arg
  if not isinstance(arg, tuple) or len(arg) < 8:
    raise ValueError(f"{context} WMMA descriptor argument is incomplete")
  try: dims = tuple(arg[1])
  except (TypeError, ValueError) as exc:
    raise ValueError(f"{context} WMMA descriptor dimensions are invalid") from exc
  dtype_in, dtype_out = arg[2], arg[3]
  if dims != _RDNA3_DIMS or dtype_in not in (dtypes.half, dtypes.char) or \
     dtype_out != (dtypes.float if dtype_in == dtypes.half else dtypes.int) or arg[5] != 32:
    raise ValueError(f"{context} WMMA descriptor carrier ABI drifted")
  expected_a, expected_b = (dtype_in.vec(x) for x in _RDNA3_ELEMENTS[:2])
  expected_out = dtype_out.vec(_RDNA3_ELEMENTS[2])
  if node.src[0].dtype != expected_a: raise ValueError(f"{context} A fragment carrier does not match the descriptor")
  if node.src[1].dtype != expected_b: raise ValueError(f"{context} B fragment carrier does not match the descriptor")
  if node.src[2].dtype != expected_out: raise ValueError(f"{context} accumulator carrier does not match the descriptor")
  if node.dtype != expected_out: raise ValueError(f"{context} WMMA result carrier does not match the descriptor")
  axes = arg[6]
  if not isinstance(axes, tuple) or len(axes) != 3:
    raise ValueError(f"{context} WMMA descriptor requires A/B/C axis groups")
  for role, count, group in (("A", 4, axes[0]), ("B", 4, axes[1]), ("C", 3, axes[2])):
    if not isinstance(group, tuple) or len(group) != count or any(not isinstance(x, tuple) or len(x) != 2 or x[1] != 2 for x in group):
      raise ValueError(f"{context} {role} WMMA contract requires {count} binary axes")


def validate_precontract_thread_axes(geometry:KernelTileGeometry, factors:PrecontractFactors,
                                     threads:PrecontractThreadAxes, subtile_m:UOp, subtile_n:UOp,
                                     *, context:str="precontract") -> None:
  """Validate live wave/lane and subtile RANGE ownership against tile factors."""
  if ((threads.wave_m.op, threads.wave_m.vmax + 1, threads.wave_m.arg[-1]) !=
      (Ops.RANGE, factors.waves_m, AxisType.LOCAL) or
      (threads.wave_n.op, threads.wave_n.vmax + 1, threads.wave_n.arg[-1]) !=
      (Ops.RANGE, factors.waves_n, AxisType.LOCAL) or
      (threads.lane.op, threads.lane.vmax + 1, threads.lane.arg[-1]) !=
      (Ops.RANGE, geometry.wave_size, AxisType.WARP)):
    raise ValueError(f"{context} thread axes do not match derived wave geometry")
  if (subtile_m.op is not Ops.RANGE or subtile_m.vmax + 1 != factors.subtiles_m or
      subtile_n.op is not Ops.RANGE or subtile_n.vmax + 1 != factors.subtiles_n):
    raise ValueError(f"{context} subtile axes do not match derived geometry")

@dataclass(frozen=True)
class PrecontractPipelineTemplate:
  """Validated immutable inputs for every epoch of a precontract LDS pipeline."""
  geometry: KernelTileGeometry
  tc: object
  allocation: UOp
  operands: tuple[PrecontractOperand, ...]
  threads: PrecontractThreadAxes
  subtile_m: UOp
  subtile_n: UOp
  contracts: tuple[PrecontractContractSpec, ...]
  pipeline_plan: object

  def __post_init__(self) -> None:
    factors = derive_precontract_factors(self.geometry, self.tc)
    validate_precontract_operand_templates(self.operands, dtype_in=self.tc.dtype_in, context="precontract pipeline")
    validate_precontract_thread_axes(self.geometry, factors, self.threads, self.subtile_m, self.subtile_n,
                                     context="precontract pipeline")
    validate_precontract_contracts(self.tc, self.contracts, context="precontract pipeline")
    slot_bytes = self.geometry.lds_windows[-1].end
    if (getattr(self.pipeline_plan, "slot_bytes", None) != slot_bytes or
        self.allocation.op is not Ops.DEFINE_LOCAL or self.allocation.ptrdtype.addrspace is not AddrSpace.LOCAL or
        self.allocation.ptrdtype.base != self.tc.dtype_in or
        self.allocation.ptrdtype.size*self.tc.dtype_in.itemsize != self.pipeline_plan.active_lds_bytes):
      raise ValueError("precontract pipeline allocation does not exactly cover its active LDS slots")

  @property
  def factors(self) -> PrecontractFactors: return derive_precontract_factors(self.geometry, self.tc)

  def producer(self, epoch:UOp, slot:UOp) -> PrecontractProducerInstance:
    return instantiate_precontract_producer(self.geometry, tc=self.tc, allocation=self.allocation,
      operands=self.operands, threads=self.threads, epoch=epoch, slot=slot)

  def fragments(self, epoch:UOp, slot:UOp, ready:UOp, k_substep:int) -> PrecontractFragmentInstance:
    if not 0 <= k_substep < self.factors.k_substeps: raise ValueError("precontract K substep is out of range")
    return instantiate_precontract_fragments(self.geometry, tc=self.tc, allocation=self.allocation, threads=self.threads,
      k_substep=UOp.const(dtypes.weakint,k_substep), subtile_m=self.subtile_m, subtile_n=self.subtile_n,
      contracts=self.contracts, epoch=epoch, slot=slot, ready=ready)

def derive_precontract_factors(geometry:KernelTileGeometry, tc) -> PrecontractFactors:
  factors = derive_precontract_shape_factors(geometry, tc)
  tm, tn, tk = geometry.tile
  rows = (tm, tn)
  for window,row in zip(geometry.lds_windows, rows):
    if window.stride_bytes < tk*tc.dtype_in.itemsize or window.end-window.base != row*window.stride_bytes:
      raise ValueError("LDS windows must exactly cover padded operand rows")
  return factors


def _window(geometry:KernelTileGeometry, role:str) -> KernelLDSWindow:
  if role not in ("A", "B"): raise ValueError(f"cooperative LDS role must be A or B, got {role!r}")
  return next(w for w in geometry.lds_windows if w.role == role)




def instantiate_precontract_producer(geometry:KernelTileGeometry, *, tc, allocation:UOp,
                                     operands:tuple[PrecontractOperand,...], threads:PrecontractThreadAxes,
                                     epoch:UOp, slot:UOp) -> PrecontractProducerInstance:
  factors=derive_precontract_factors(geometry,tc)
  item_bytes, vector_bytes = tc.dtype_in.itemsize, 16
  vector_elements = vector_bytes // item_bytes
  slot_base=slot*(geometry.lds_windows[-1].end//item_bytes)
  thread=(threads.wave_m*geometry.waves[1]+threads.wave_n)*geometry.wave_size+threads.lane
  role_nodes=[]
  for operand in operands:
    stores=[]; window=_window(geometry,operand.role); loads=factors.loads_a if operand.role == "A" else factors.loads_b
    for row_iteration in range(loads):
      linear_vector=thread+row_iteration*geometry.threads
      row,vector=linear_vector//factors.vectors_per_row,linear_vector%factors.vectors_per_row
      logical_k=vector*vector_elements
      logical_row = operand.row_tile_base + row
      value = operand.transform.dequant_tile(operand.source, logical_row, epoch*geometry.tile[2]+logical_k, vector_elements).value \
        if isinstance(operand, PackedPrecontractOperandTemplate) else UOp(Ops.STACK,tc.dtype_in.vec(vector_elements),tuple(operand.source.substitute({
          operand.row_axis:logical_row, operand.k_axis:epoch*geometry.tile[2]+logical_k+elem}) for elem in range(vector_elements)))
      tag=("kernel_tile_store",operand.role,row_iteration,epoch,slot)
      idx=allocation.index(slot_base+(window.base+row*window.stride_bytes+logical_k*item_bytes)//item_bytes,
                           dtype=tc.dtype_in.vec(vector_elements)).replace(tag=tag)
      stores.append(idx.store(value).replace(tag=tag).end())
    role_nodes.append(UOp.group(*stores))
  return PrecontractProducerInstance(epoch,slot,(role_nodes[0],role_nodes[1]))

def instantiate_precontract_fragments(geometry:KernelTileGeometry, *, tc, allocation:UOp, threads:PrecontractThreadAxes,
                                      k_substep:UOp, subtile_m:UOp, subtile_n:UOp,
                                      contracts:tuple[PrecontractContractSpec,...], epoch:UOp, slot:UOp,
                                      ready:UOp) -> PrecontractFragmentInstance:
  factors=derive_precontract_factors(geometry,tc); item_bytes=tc.dtype_in.itemsize
  slot_base=slot*(geometry.lds_windows[-1].end//item_bytes)
  ordered=allocation.after(ready); lane=threads.lane
  def fragment(role,subtile,wave,subtiles,contract):
    window=_window(geometry,role); row=(wave*subtiles+subtile)*16+lane%16
    logical_k=k_substep*tc.dims[2]+contract.element
    idx=slot_base+(window.base+row*window.stride_bytes+logical_k*item_bytes)//item_bytes
    semantic=(role,epoch,slot,k_substep,subtile)
    load=ordered.index(idx,dtype=tc.dtype_in).replace(tag=("kernel_tile_fragment_load",*semantic)).load()
    operand_idx = 0 if role == "A" else 1
    return UOp(Ops.CONTRACT,tc.dtype_in.vec(tc.elements_per_thread[operand_idx]),(load,),contract.arg,
               tag=("kernel_tile_fragment",*semantic))
  frags=(fragment("A",subtile_m,threads.wave_m,factors.subtiles_m,contracts[0]),
         fragment("B",subtile_n,threads.wave_n,factors.subtiles_n,contracts[1]))
  return PrecontractFragmentInstance(epoch,slot,ready,frags)

def build_precontract_lds_stage(geometry:KernelTileGeometry, *, tc, allocation:UOp,
                                operands:tuple[PrecontractOperand, ...], threads:PrecontractThreadAxes,
                                k_axis:PrecontractKAxis, subtile_m:UOp, subtile_n:UOp,
                                contracts:tuple[PrecontractContractSpec, ...], pipeline_plan=None) -> PrecontractLDSStage:
  """Build an unwired vector cooperative stage while full operand index templates still exist."""
  factors = derive_precontract_factors(geometry, tc)
  validate_precontract_operand_templates(operands, dtype_in=tc.dtype_in, context="precontract")
  for operand in operands:
    if operand.row_tile_base.dtype.scalar() not in (dtypes.int, dtypes.weakint): raise ValueError("precontract row tile base must be integer")
  validate_precontract_thread_axes(geometry, factors, threads, subtile_m, subtile_n, context="precontract")
  if (k_axis.tile_owner.op is not Ops.RANGE or k_axis.tile_owner.arg[-1] is not AxisType.REDUCE or
      k_axis.tile_owner not in k_axis.tile_base.backward_slice_with_self):
    raise ValueError("precontract K tile owner must be a live REDUCE range in tile base")
  if (k_axis.substep_owner.op is not Ops.RANGE or k_axis.substep_owner.arg[-1] is not AxisType.UNROLL or
      k_axis.substep_owner.vmax+1 != factors.k_substeps or k_axis.substep_owner not in k_axis.substep.backward_slice_with_self):
    raise ValueError("precontract K substep owner must be a live derived-size UNROLL range in substep")
  if (subtile_m.op is not Ops.RANGE or subtile_m.vmax+1 != factors.subtiles_m or
      subtile_n.op is not Ops.RANGE or subtile_n.vmax+1 != factors.subtiles_n):
    raise ValueError("precontract K/subtile axes are invalid")
  validate_precontract_contracts(tc, contracts, context="precontract", mismatch="does not match actual descriptor operand mapping")
  slot_bytes = geometry.lds_windows[-1].end
  total_bytes = slot_bytes if pipeline_plan is None else pipeline_plan.active_lds_bytes
  if (allocation.op is not Ops.DEFINE_LOCAL or allocation.ptrdtype.addrspace is not AddrSpace.LOCAL or
      allocation.ptrdtype.base != tc.dtype_in or allocation.ptrdtype.size * tc.dtype_in.itemsize != total_bytes):
    raise ValueError("precontract caller allocation must be one exact dtype_in LDS window")
  item_bytes, vector_bytes = tc.dtype_in.itemsize, 16
  vector_elements = vector_bytes // item_bytes
  stores = []
  slot_base = UOp.const(dtypes.weakint, 0) if pipeline_plan is None else \
    (k_axis.tile_owner % pipeline_plan.buffer_count) * (slot_bytes // item_bytes)
  thread = (threads.wave_m * geometry.waves[1] + threads.wave_n) * geometry.wave_size + threads.lane
  for operand in operands:
    window = _window(geometry, operand.role)
    loads = factors.loads_a if operand.role == "A" else factors.loads_b
    for row_iteration in range(loads):
      linear_vector = thread + row_iteration*geometry.threads
      row, vector = linear_vector//factors.vectors_per_row, linear_vector%factors.vectors_per_row
      logical_k = vector * vector_elements
      logical_row = operand.row_tile_base + row
      value = operand.transform.dequant_tile(operand.source, logical_row, k_axis.tile_base + logical_k, vector_elements).value \
        if isinstance(operand, PackedPrecontractOperandTemplate) else UOp(Ops.STACK, tc.dtype_in.vec(vector_elements), tuple(operand.source.substitute({
          operand.row_axis: logical_row, operand.k_axis: k_axis.tile_base + logical_k + elem}) for elem in range(vector_elements)))
      index = slot_base + (window.base + row * window.stride_bytes + logical_k * item_bytes) // item_bytes
      store_tag = ("kernel_tile_store", operand.role, row_iteration)
      store_idx = allocation.index(index, dtype=tc.dtype_in.vec(vector_elements)).replace(tag=store_tag)
      stores.append(store_idx.store(value).replace(tag=store_tag).end())
  producer = UOp.group(*stores)
  barrier = UOp.barrier(producer)
  wave_m, wave_n, lane = threads.wave_m, threads.wave_n, threads.lane
  ordered = allocation.after(barrier)
  def _fragment(role:str, subtile:UOp, wave:UOp, subtiles:int, contract:PrecontractContractSpec) -> UOp:
    window = _window(geometry, role)
    row = (wave * subtiles + subtile) * 16 + lane % 16
    logical_k = k_axis.substep * tc.dims[2] + contract.element
    index = slot_base + (window.base + row * window.stride_bytes + logical_k * item_bytes) // item_bytes
    load = ordered.index(index, dtype=tc.dtype_in).replace(tag=("kernel_tile_fragment_load", role)).load()
    operand_idx = 0 if role == "A" else 1
    return UOp(Ops.CONTRACT, tc.dtype_in.vec(tc.elements_per_thread[operand_idx]), (load,), contract.arg,
               tag=("kernel_tile_fragment", role))
  return PrecontractLDSStage(allocation, producer, barrier, _fragment("A", subtile_m, wave_m, factors.subtiles_m, contracts[0]),
                             _fragment("B", subtile_n, wave_n, factors.subtiles_n, contracts[1]))
