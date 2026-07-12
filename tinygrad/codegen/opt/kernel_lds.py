"""Pure cooperative LDS ownership math for compiler-bound kernel geometry."""
from __future__ import annotations

from dataclasses import dataclass

from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import AxisType, KernelLDSWindow, KernelTileGeometry, Ops, UOp

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
  """Admit only the exact tensor-core descriptor this mapping proves."""
  fields = (("dims", _RDNA3_DIMS), ("threads", 32), ("elements_per_thread", _RDNA3_ELEMENTS),
            ("opts", _RDNA3_OPTS), ("swizzle", _RDNA3_SWIZZLE))
  for name, expected in fields:
    if getattr(tc, name, None) != expected: raise ValueError(f"RDNA3 WMMA descriptor {name} drifted")
  if getattr(getattr(tc, "dtype_in", None), "name", None) != "half": raise ValueError("RDNA3 WMMA descriptor dtype_in drifted")
  if getattr(getattr(tc, "dtype_out", None), "name", None) != "float": raise ValueError("RDNA3 WMMA descriptor dtype_out drifted")
  try: remaps = tuple(tc.lane_map.remaps())
  except (AttributeError, AssertionError, TypeError, ValueError) as exc:
    raise ValueError("RDNA3 WMMA descriptor remaps are unavailable or invalid") from exc
  if remaps != _RDNA3_REMAPS: raise ValueError("RDNA3 WMMA descriptor remaps drifted")


@dataclass(frozen=True)
class CooperativeLDSStore:
  role: str
  thread: int
  iteration: int
  row: int
  vector: int
  byte_offset: int
  vector_bytes: int

@dataclass(frozen=True)
class WMMAFragmentLoad:
  role: str
  thread: int
  wave_m: int
  wave_n: int
  subtile: int
  k_substep: int
  element: int
  logical_row: int
  logical_k: int
  byte_offset: int

@dataclass(frozen=True)
class WMMAOutputOwner:
  thread: int
  wave_m: int
  wave_n: int
  subtile_m: int
  subtile_n: int
  element: int
  row: int
  col: int

@dataclass(frozen=True)
class PrecontractOperandTemplate:
  role: str
  source: UOp
  row_axis: UOp
  k_axis: UOp
  row_tile_base: UOp

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
class PrecontractPipelineTemplate:
  """Validated immutable inputs for every epoch of a precontract LDS pipeline."""
  geometry: KernelTileGeometry
  tc: object
  allocation: UOp
  operands: tuple[PrecontractOperandTemplate, ...]
  threads: PrecontractThreadAxes
  subtile_m: UOp
  subtile_n: UOp
  contracts: tuple[PrecontractContractSpec, ...]
  pipeline_plan: object

  def __post_init__(self) -> None:
    factors = derive_precontract_factors(self.geometry, self.tc)
    if tuple(x.role for x in self.operands) != ("A", "B"):
      raise ValueError("precontract pipeline operands must be exactly ordered A and B")
    for operand in self.operands:
      if (operand.row_axis.op is not Ops.RANGE or operand.k_axis.op is not Ops.RANGE or
          operand.row_axis not in operand.source.backward_slice_with_self or
          operand.k_axis not in operand.source.backward_slice_with_self or operand.source.dtype.scalar() != dtypes.half):
        raise ValueError(f"precontract pipeline {operand.role} template does not retain scalar fp16 row/K ownership")
    if ((self.threads.wave_m.op, self.threads.wave_m.vmax+1, self.threads.wave_m.arg[-1]) !=
        (Ops.RANGE, factors.waves_m, AxisType.LOCAL) or
        (self.threads.wave_n.op, self.threads.wave_n.vmax+1, self.threads.wave_n.arg[-1]) !=
        (Ops.RANGE, factors.waves_n, AxisType.LOCAL) or
        (self.threads.lane.op, self.threads.lane.vmax+1, self.threads.lane.arg[-1]) !=
        (Ops.RANGE, self.geometry.wave_size, AxisType.WARP)):
      raise ValueError("precontract pipeline thread axes do not match derived wave geometry")
    if (self.subtile_m.op is not Ops.RANGE or self.subtile_m.vmax+1 != factors.subtiles_m or
        self.subtile_n.op is not Ops.RANGE or self.subtile_n.vmax+1 != factors.subtiles_n):
      raise ValueError("precontract pipeline subtile axes do not match derived geometry")
    if tuple(c.role for c in self.contracts) != ("A", "B"):
      raise ValueError("precontract pipeline contracts must be exactly ordered A and B")
    descriptor_remaps = tuple(tuple(x.items()) for x in self.tc.lane_map.remaps())
    for operand_idx, contract in enumerate(self.contracts):
      folded = ((contract.axes[0]*2+contract.axes[1])*2+contract.axes[2])*2+contract.axes[3] if len(contract.axes) == 4 else None
      if (len(contract.axes) != 4 or any(a.op is not Ops.RANGE or a.vmax+1 != 2 for a in contract.axes) or
          contract.arg != tuple((a.arg[0], 2) for a in contract.axes) or contract.element is not folded or
          contract.descriptor_remap != descriptor_remaps[operand_idx]):
        raise ValueError(f"precontract pipeline {contract.role} contract does not match the descriptor")
    slot_bytes = self.geometry.lds_windows[-1].end
    if (getattr(self.pipeline_plan, "slot_bytes", None) != slot_bytes or
        self.allocation.op is not Ops.DEFINE_LOCAL or self.allocation.ptrdtype.addrspace is not AddrSpace.LOCAL or
        self.allocation.ptrdtype.base != dtypes.half or
        self.allocation.ptrdtype.size*dtypes.half.itemsize != self.pipeline_plan.active_lds_bytes):
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
  validate_rdna3_wmma_descriptor(tc)
  tm,tn,tk = geometry.tile
  if tm % (geometry.waves[0]*tc.dims[1]) or tn % (geometry.waves[1]*tc.dims[0]) or tk % tc.dims[2]:
    raise ValueError("tile must divide into whole per-wave tensor-core subtiles and K steps")
  sm,sn,ks = tm//(geometry.waves[0]*tc.dims[1]), tn//(geometry.waves[1]*tc.dims[0]), tk//tc.dims[2]
  if ks < 2: raise ValueError("current atomic staging requires at least two tensor-core K steps")
  vectors_per_row = tk*dtypes.half.itemsize//16
  if vectors_per_row <= 0 or tk*dtypes.half.itemsize % 16: raise ValueError("K row must contain whole b128 vectors")
  rows = (tm,tn)
  loads = tuple(row*vectors_per_row//geometry.threads for row in rows)
  if any(row*vectors_per_row % geometry.threads for row in rows) or any(x <= 0 for x in loads):
    raise ValueError("operand vectors must divide evenly across cooperative threads")
  for window,row in zip(geometry.lds_windows, rows):
    if window.stride_bytes < tk*2 or window.end-window.base != row*window.stride_bytes:
      raise ValueError("LDS windows must exactly cover padded operand rows")
  return PrecontractFactors(sm,sn,*geometry.waves,ks,vectors_per_row,*loads)


def semantic_wave_coords(geometry:KernelTileGeometry, thread:int) -> tuple[int, int, int]:
  """Return (wave_m, wave_n, lane) using row-major wave ownership."""
  if not isinstance(thread, int) or isinstance(thread, bool) or not 0 <= thread < geometry.threads:
    raise ValueError(f"thread must be in [0, {geometry.threads})")
  wave_id, lane = divmod(thread, geometry.wave_size)
  wave_m, wave_n = divmod(wave_id, geometry.waves[1])
  return wave_m, wave_n, lane


def _window(geometry:KernelTileGeometry, role:str) -> KernelLDSWindow:
  if role not in ("A", "B"): raise ValueError(f"cooperative LDS role must be A or B, got {role!r}")
  return next(w for w in geometry.lds_windows if w.role == role)


def cooperative_lds_stores(geometry:KernelTileGeometry, role:str, *, element_bytes:int=2,
                           vector_bytes:int=16) -> tuple[CooperativeLDSStore, ...]:
  """Elect one thread for every non-padding vector in an A or B tile window."""
  if not isinstance(element_bytes, int) or isinstance(element_bytes, bool) or element_bytes <= 0:
    raise ValueError("element_bytes must be a positive int")
  if not isinstance(vector_bytes, int) or isinstance(vector_bytes, bool) or vector_bytes <= 0:
    raise ValueError("vector_bytes must be a positive int")
  window = _window(geometry, role)
  rows = geometry.tile[0] if role == "A" else geometry.tile[1]
  row_data_bytes = geometry.tile[2] * element_bytes
  if row_data_bytes % vector_bytes: raise ValueError("tile K row bytes must be divisible by vector_bytes")
  if window.stride_bytes < row_data_bytes or window.stride_bytes % vector_bytes:
    raise ValueError("LDS stride must contain the data row and be vector aligned")
  if window.end - window.base != rows * window.stride_bytes:
    raise ValueError("LDS window size must exactly equal rows * stride")
  vectors_per_row = row_data_bytes // vector_bytes
  vector_count = rows * vectors_per_row
  if vector_count % geometry.threads:
    raise ValueError("cooperative tile vectors must divide evenly across threads")
  stores = []
  for linear in range(vector_count):
    thread, iteration = linear % geometry.threads, linear // geometry.threads
    row, vector = divmod(linear, vectors_per_row)
    byte_offset = window.base + row * window.stride_bytes + vector * vector_bytes
    stores.append(CooperativeLDSStore(role, thread, iteration, row, vector, byte_offset, vector_bytes))
  return tuple(stores)


def cooperative_lds_padding_offsets(geometry:KernelTileGeometry, role:str, *, element_bytes:int=2,
                                    vector_bytes:int=16) -> tuple[int, ...]:
  """Return vector-aligned padding slots, which intentionally have no store owner."""
  window = _window(geometry, role)
  rows = geometry.tile[0] if role == "A" else geometry.tile[1]
  row_data_bytes = geometry.tile[2] * element_bytes
  if row_data_bytes % vector_bytes or window.stride_bytes < row_data_bytes or window.stride_bytes % vector_bytes:
    raise ValueError("LDS row data and stride must be valid vector-aligned intervals")
  return tuple(window.base + row * window.stride_bytes + offset
               for row in range(rows) for offset in range(row_data_bytes, window.stride_bytes, vector_bytes))


def _rdna3_wmma_output_coord(lane:int, element:int) -> tuple[int, int]:
  if not isinstance(lane, int) or isinstance(lane, bool) or not 0 <= lane < 32: raise ValueError("lane must be in [0, 32)")
  if not isinstance(element, int) or isinstance(element, bool) or not 0 <= element < 8: raise ValueError("element must be in [0, 8)")
  return lane % 16, lane // 16 + element * 2

def rdna3_wmma_output_coord(lane:int, element:int, *, tc) -> tuple[int, int]:
  """RDNA3 fp32 16x16x16 output map used by the Python WMMA interpreter."""
  validate_rdna3_wmma_descriptor(tc)
  return _rdna3_wmma_output_coord(lane, element)


def wmma_fragment_loads(geometry:KernelTileGeometry, role:str, *, tc, element_bytes:int=2) -> tuple[WMMAFragmentLoad, ...]:
  """Enumerate per-wave RDNA3 A/B fragment loads from the staged tile windows."""
  validate_rdna3_wmma_descriptor(tc)
  window = _window(geometry, role)
  if geometry.wave_size != 32 or geometry.tile[2] % 16:
    raise ValueError("RDNA3 fragment mapping requires wave32 and K divisible by 16")
  if element_bytes != 2: raise ValueError("RDNA3 fp16 fragment mapping requires element_bytes=2")
  subtiles = geometry.tile[0] // (geometry.waves[0] * 16) if role == "A" else \
             geometry.tile[1] // (geometry.waves[1] * 16)
  if subtiles <= 0 or (geometry.tile[0] if role == "A" else geometry.tile[1]) != \
     subtiles * (geometry.waves[0] if role == "A" else geometry.waves[1]) * 16:
    raise ValueError("tile extent must divide exactly into wave 16x16 subtiles")
  loads = []
  for thread in range(geometry.threads):
    wave_m, wave_n, lane = semantic_wave_coords(geometry, thread)
    for subtile in range(subtiles):
      logical_row = (wave_m * subtiles + subtile) * 16 + lane % 16 if role == "A" else \
                    (wave_n * subtiles + subtile) * 16 + lane % 16
      for k_substep in range(geometry.tile[2] // 16):
        for element in range(16):
          logical_k = k_substep * 16 + element
          byte_offset = window.base + logical_row * window.stride_bytes + logical_k * element_bytes
          if not window.base <= byte_offset or byte_offset + element_bytes > window.end:
            raise ValueError("RDNA3 fragment load is outside its LDS window")
          loads.append(WMMAFragmentLoad(role, thread, wave_m, wave_n, subtile, k_substep, element,
                                        logical_row, logical_k, byte_offset))
  return tuple(loads)


def wmma_output_owners(geometry:KernelTileGeometry, *, tc) -> tuple[WMMAOutputOwner, ...]:
  """Enumerate RDNA3 output ownership for every wave and its 2-D WMMA subtile grid."""
  validate_rdna3_wmma_descriptor(tc)
  if geometry.wave_size != 32: raise ValueError("RDNA3 output mapping requires wave32")
  subtiles_m = geometry.tile[0] // (geometry.waves[0] * 16)
  subtiles_n = geometry.tile[1] // (geometry.waves[1] * 16)
  if (subtiles_m <= 0 or subtiles_n <= 0 or geometry.tile[0] != subtiles_m * geometry.waves[0] * 16 or
      geometry.tile[1] != subtiles_n * geometry.waves[1] * 16):
    raise ValueError("output tile must divide exactly into wave 16x16 subtiles")
  owners = []
  for thread in range(geometry.threads):
    wave_m, wave_n, lane = semantic_wave_coords(geometry, thread)
    for subtile_m in range(subtiles_m):
      for subtile_n in range(subtiles_n):
        for element in range(8):
          local_row, local_col = _rdna3_wmma_output_coord(lane, element)
          row = (wave_m * subtiles_m + subtile_m) * 16 + local_row
          col = (wave_n * subtiles_n + subtile_n) * 16 + local_col
          owners.append(WMMAOutputOwner(thread, wave_m, wave_n, subtile_m, subtile_n, element, row, col))
  return tuple(owners)


def instantiate_precontract_producer(geometry:KernelTileGeometry, *, tc, allocation:UOp,
                                     operands:tuple[PrecontractOperandTemplate,...], threads:PrecontractThreadAxes,
                                     epoch:UOp, slot:UOp) -> PrecontractProducerInstance:
  factors=derive_precontract_factors(geometry,tc)
  slot_base=slot*(geometry.lds_windows[-1].end//2)
  thread=(threads.wave_m*geometry.waves[1]+threads.wave_n)*geometry.wave_size+threads.lane
  role_nodes=[]
  for operand in operands:
    stores=[]; window=_window(geometry,operand.role); loads=factors.loads_a if operand.role == "A" else factors.loads_b
    for row_iteration in range(loads):
      linear_vector=thread+row_iteration*geometry.threads
      row,vector=linear_vector//factors.vectors_per_row,linear_vector%factors.vectors_per_row
      logical_k=vector*8
      values=tuple(operand.source.substitute({operand.row_axis:operand.row_tile_base+row,
        operand.k_axis:epoch*geometry.tile[2]+logical_k+elem}) for elem in range(8))
      tag=("kernel_tile_store",operand.role,row_iteration,epoch,slot)
      idx=allocation.index(slot_base+(window.base+row*window.stride_bytes+logical_k*2)//2,dtype=dtypes.half.vec(8)).replace(tag=tag)
      stores.append(idx.store(UOp(Ops.STACK,dtypes.half.vec(8),values)).replace(tag=tag).end())
    role_nodes.append(UOp.group(*stores))
  return PrecontractProducerInstance(epoch,slot,(role_nodes[0],role_nodes[1]))

def instantiate_precontract_fragments(geometry:KernelTileGeometry, *, tc, allocation:UOp, threads:PrecontractThreadAxes,
                                      k_substep:UOp, subtile_m:UOp, subtile_n:UOp,
                                      contracts:tuple[PrecontractContractSpec,...], epoch:UOp, slot:UOp,
                                      ready:UOp) -> PrecontractFragmentInstance:
  factors=derive_precontract_factors(geometry,tc); slot_base=slot*(geometry.lds_windows[-1].end//2)
  ordered=allocation.after(ready); lane=threads.lane
  def fragment(role,subtile,wave,subtiles,contract):
    window=_window(geometry,role); row=(wave*subtiles+subtile)*16+lane%16
    logical_k=k_substep*tc.dims[2]+contract.element
    idx=slot_base+(window.base+row*window.stride_bytes+logical_k*2)//2
    semantic=(role,epoch,slot,k_substep,subtile)
    load=ordered.index(idx,dtype=dtypes.half).replace(tag=("kernel_tile_fragment_load",*semantic)).load()
    return UOp(Ops.CONTRACT,dtypes.half.vec(16),(load,),contract.arg,tag=("kernel_tile_fragment",*semantic))
  frags=(fragment("A",subtile_m,threads.wave_m,factors.subtiles_m,contracts[0]),
         fragment("B",subtile_n,threads.wave_n,factors.subtiles_n,contracts[1]))
  return PrecontractFragmentInstance(epoch,slot,ready,frags)

def build_precontract_lds_stage(geometry:KernelTileGeometry, *, tc, allocation:UOp,
                                operands:tuple[PrecontractOperandTemplate, ...], threads:PrecontractThreadAxes,
                                k_axis:PrecontractKAxis, subtile_m:UOp, subtile_n:UOp,
                                contracts:tuple[PrecontractContractSpec, ...], pipeline_plan=None) -> PrecontractLDSStage:
  """Build an unwired vector cooperative stage while full operand index templates still exist."""
  factors = derive_precontract_factors(geometry, tc)
  if tuple(x.role for x in operands) != ("A", "B"): raise ValueError("precontract operands must be exactly ordered A and B")
  for operand in operands:
    if operand.row_axis.op is not Ops.RANGE or operand.k_axis.op is not Ops.RANGE:
      raise ValueError(f"precontract {operand.role} template axes must be RANGE UOps")
    if operand.row_axis not in operand.source.backward_slice_with_self or operand.k_axis not in operand.source.backward_slice_with_self:
      raise ValueError(f"precontract {operand.role} template does not retain row and K axes")
    if operand.source.dtype.scalar() != dtypes.half: raise ValueError("precontract operands must be fp16 scalar templates")
    if operand.row_tile_base.dtype.scalar() not in (dtypes.int, dtypes.weakint): raise ValueError("precontract row tile base must be integer")
  if ((threads.wave_m.op, threads.wave_m.vmax+1, threads.wave_m.arg[-1]) != (Ops.RANGE, factors.waves_m, AxisType.LOCAL) or
      (threads.wave_n.op, threads.wave_n.vmax+1, threads.wave_n.arg[-1]) != (Ops.RANGE, factors.waves_n, AxisType.LOCAL) or
      (threads.lane.op, threads.lane.vmax+1, threads.lane.arg[-1]) != (Ops.RANGE, geometry.wave_size, AxisType.WARP)):
    raise ValueError("precontract thread axes do not match derived wave geometry")
  if (k_axis.tile_owner.op is not Ops.RANGE or k_axis.tile_owner.arg[-1] is not AxisType.REDUCE or
      k_axis.tile_owner not in k_axis.tile_base.backward_slice_with_self):
    raise ValueError("precontract K tile owner must be a live REDUCE range in tile base")
  if (k_axis.substep_owner.op is not Ops.RANGE or k_axis.substep_owner.arg[-1] is not AxisType.UNROLL or
      k_axis.substep_owner.vmax+1 != factors.k_substeps or k_axis.substep_owner not in k_axis.substep.backward_slice_with_self):
    raise ValueError("precontract K substep owner must be a live derived-size UNROLL range in substep")
  if (subtile_m.op is not Ops.RANGE or subtile_m.vmax+1 != factors.subtiles_m or
      subtile_n.op is not Ops.RANGE or subtile_n.vmax+1 != factors.subtiles_n):
    raise ValueError("precontract K/subtile axes are invalid")
  if tuple(c.role for c in contracts) != ("A", "B"): raise ValueError("precontract contracts must be exactly ordered A and B")
  descriptor_remaps = tuple(tuple(x.items()) for x in tc.lane_map.remaps())
  for operand_idx, contract in enumerate(contracts):
    folded = ((contract.axes[0]*2+contract.axes[1])*2+contract.axes[2])*2+contract.axes[3] if len(contract.axes) == 4 else None
    if (len(contract.axes) != 4 or any(a.op is not Ops.RANGE or a.vmax+1 != 2 for a in contract.axes) or
        contract.arg != tuple((a.arg[0], 2) for a in contract.axes) or contract.element is not folded or
        contract.descriptor_remap != descriptor_remaps[operand_idx]):
      raise ValueError(f"precontract {contract.role} contract does not match actual descriptor operand mapping")
  slot_bytes = geometry.lds_windows[-1].end
  total_bytes = slot_bytes if pipeline_plan is None else pipeline_plan.active_lds_bytes
  if (allocation.op is not Ops.DEFINE_LOCAL or allocation.ptrdtype.addrspace is not AddrSpace.LOCAL or
      allocation.ptrdtype.base != dtypes.half or allocation.ptrdtype.size * dtypes.half.itemsize != total_bytes):
    raise ValueError("precontract caller allocation must be one exact fp16 LDS window")
  stores = []
  slot_base = UOp.const(dtypes.weakint, 0) if pipeline_plan is None else \
    (k_axis.tile_owner % pipeline_plan.buffer_count) * (slot_bytes // 2)
  thread = (threads.wave_m * geometry.waves[1] + threads.wave_n) * geometry.wave_size + threads.lane
  for operand in operands:
    window = _window(geometry, operand.role)
    loads = factors.loads_a if operand.role == "A" else factors.loads_b
    for row_iteration in range(loads):
      linear_vector = thread + row_iteration*geometry.threads
      row, vector = linear_vector//factors.vectors_per_row, linear_vector%factors.vectors_per_row
      logical_k = vector * 8
      values = tuple(operand.source.substitute({operand.row_axis: operand.row_tile_base + row,
                                                operand.k_axis: k_axis.tile_base + logical_k + elem}) for elem in range(8))
      value = UOp(Ops.STACK, dtypes.half.vec(8), values)
      index = slot_base + (window.base + row * window.stride_bytes + logical_k * 2) // 2
      store_tag = ("kernel_tile_store", operand.role, row_iteration)
      store_idx = allocation.index(index, dtype=dtypes.half.vec(8)).replace(tag=store_tag)
      stores.append(store_idx.store(value).replace(tag=store_tag).end())
  producer = UOp.group(*stores)
  barrier = UOp.barrier(producer)
  wave_m, wave_n, lane = threads.wave_m, threads.wave_n, threads.lane
  ordered = allocation.after(barrier)
  def _fragment(role:str, subtile:UOp, wave:UOp, subtiles:int, contract:PrecontractContractSpec) -> UOp:
    window = _window(geometry, role)
    row = (wave * subtiles + subtile) * 16 + lane % 16
    logical_k = k_axis.substep * tc.dims[2] + contract.element
    index = slot_base + (window.base + row * window.stride_bytes + logical_k * 2) // 2
    load = ordered.index(index, dtype=dtypes.half).replace(tag=("kernel_tile_fragment_load", role)).load()
    return UOp(Ops.CONTRACT, dtypes.half.vec(16), (load,), contract.arg, tag=("kernel_tile_fragment", role))
  return PrecontractLDSStage(allocation, producer, barrier, _fragment("A", subtile_m, wave_m, factors.subtiles_m, contracts[0]),
                             _fragment("B", subtile_n, wave_n, factors.subtiles_n, contracts[1]))
