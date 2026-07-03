from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from tinygrad.dtype import AddrSpace, DType
from tinygrad.renderer.amd.dsl import FixedBitField, Inst, Reg
from tinygrad.uop.ops import Ops, UOp, GroupOp


@dataclass(frozen=True)
class AMDScheduleMeta:
  idx: int
  source: str
  op: str
  dtype: str | None
  latency_class: str
  memory_space: str | None = None
  vector_width: int | None = None
  wait_group: str | None = None
  barrier_scope: str | None = None
  live_range_boundary: str | None = None
  issue_cluster: str | None = None
  prefetch_stage: int | None = None
  lds_stage: int | None = None
  register_pressure_budget: str | None = None

  def to_dict(self) -> dict[str, Any]: return asdict(self)


@dataclass(frozen=True)
class AMDScheduleAction:
  idx: int
  action: str
  reason: str
  op: str | None = None
  wait_group: str | None = None

  def to_dict(self) -> dict[str, Any]: return asdict(self)


@dataclass(frozen=True)
class AMDPipelineStageMeta:
  idx: int
  pipeline_id: str
  phase: str
  stage_id: int
  stage_count: int
  producer_distance: int
  k_axis: str
  buffer_role: str
  lds_slot: int | None
  dependency_group: str
  semantic_order: int
  resource_budget: str | None = None
  source: str = "structured"
  op: str | None = None
  uop_idx: int | None = None

  def to_dict(self) -> dict[str, Any]: return asdict(self)


def pipeline_stage_metadata_from_records(records: Iterable[dict[str, Any]]) -> list[AMDPipelineStageMeta]:
  rows: list[AMDPipelineStageMeta] = []
  for idx, record in enumerate(records):
    rows.append(AMDPipelineStageMeta(
      idx=idx,
      pipeline_id=str(record["pipeline_id"]),
      phase=str(record["phase"]),
      stage_id=int(record["stage_id"]),
      stage_count=int(record["stage_count"]),
      producer_distance=int(record["producer_distance"]),
      k_axis=str(record["k_axis"]),
      buffer_role=str(record["buffer_role"]),
      lds_slot=None if record.get("lds_slot") is None else int(record["lds_slot"]),
      dependency_group=str(record["dependency_group"]),
      semantic_order=int(record["semantic_order"]),
      resource_budget=None if record.get("resource_budget") is None else str(record["resource_budget"]),
      source=str(record.get("source", "structured")),
      op=None if record.get("op") is None else str(record["op"]),
      uop_idx=None if record.get("uop_idx") is None else int(record["uop_idx"])))
  return rows


def pipeline_stage_summary(rows: Iterable[AMDPipelineStageMeta]) -> dict[str, Any]:
  data = list(rows)
  def counts(attr: str) -> dict[str, int]:
    ret: dict[str, int] = {}
    for row in data:
      val = getattr(row, attr)
      if val is not None: ret[str(val)] = ret.get(str(val), 0) + 1
    return ret
  phases, roles = counts("phase"), counts("buffer_role")
  lds_slots = sorted({row.lds_slot for row in data if row.lds_slot is not None})
  stage_counts = sorted({row.stage_count for row in data})
  producer_distances = sorted({row.producer_distance for row in data})
  dependency_groups = sorted({row.dependency_group for row in data})
  return {
    "row_count": len(data),
    "counts": {
      "phase": phases,
      "buffer_role": roles,
      "lds_slot": {str(slot): sum(1 for row in data if row.lds_slot == slot) for slot in lds_slots},
      "pipeline_id": counts("pipeline_id"),
    },
    "stage_counts": stage_counts,
    "producer_distances": producer_distances,
    "lds_slots": lds_slots,
    "dependency_group_count": len(dependency_groups),
    "semantic_order_monotonic": all(a.semantic_order <= b.semantic_order for a, b in zip(data, data[1:])),
    "has_two_stage_pipeline": 2 in stage_counts and {0, 1}.issubset(set(lds_slots)),
    "has_required_roles": {"global_load", "lds_store", "lds_load", "wmma_consume"}.issubset(set(roles)),
    "has_required_phases": {"prologue", "steady"}.issubset(set(phases)),
    "has_steady_producer_distance_1": any(row.phase == "steady" and row.producer_distance == 1 for row in data),
  }


def pipeline_stage_dump(rows: Iterable[AMDPipelineStageMeta]) -> dict[str, Any]:
  data = list(rows)
  return {"rows": [row.to_dict() for row in data], "summary": pipeline_stage_summary(data)}


@dataclass(frozen=True)
class AMDLDSStagePlan:
  pipeline_id: str
  stage_count: int
  slots: tuple[int, ...]
  slot_roles: dict[int, tuple[str, ...]]
  slot_offsets: dict[int, int]
  dependency_groups: tuple[str, ...]
  required_local_bytes: int
  alias_safe: bool
  lowering_status: str = "planned"

  def to_dict(self) -> dict[str, Any]:
    ret = asdict(self)
    ret["slots"] = list(self.slots)
    ret["slot_roles"] = {str(k): list(v) for k, v in self.slot_roles.items()}
    ret["slot_offsets"] = {str(k): v for k, v in self.slot_offsets.items()}
    ret["dependency_groups"] = list(self.dependency_groups)
    return ret


def lds_stage_plan_from_pipeline(rows: Iterable[AMDPipelineStageMeta], slot_bytes: int) -> AMDLDSStagePlan:
  data = list(rows)
  if not data: raise ValueError("pipeline rows are required for LDS stage planning")
  if slot_bytes <= 0: raise ValueError("slot_bytes must be positive")
  pipeline_ids = {row.pipeline_id for row in data}
  if len(pipeline_ids) != 1: raise ValueError(f"expected one pipeline_id, got {sorted(pipeline_ids)}")
  stage_counts = {row.stage_count for row in data}
  if len(stage_counts) != 1: raise ValueError(f"expected one stage_count, got {sorted(stage_counts)}")
  slots = tuple(sorted({row.lds_slot for row in data if row.lds_slot is not None}))
  slot_roles = {slot: tuple(sorted({row.buffer_role for row in data if row.lds_slot == slot})) for slot in slots}
  slot_offsets = {slot: i * slot_bytes for i, slot in enumerate(slots)}
  dependency_groups = tuple(sorted({row.dependency_group for row in data}))
  offset_ranges = [(off, off + slot_bytes) for off in slot_offsets.values()]
  alias_safe = len(offset_ranges) == len(set(offset_ranges)) and all(a[1] <= b[0] or b[1] <= a[0] for i, a in enumerate(offset_ranges) for b in offset_ranges[i+1:])
  return AMDLDSStagePlan(
    pipeline_id=next(iter(pipeline_ids)), stage_count=next(iter(stage_counts)), slots=slots, slot_roles=slot_roles,
    slot_offsets=slot_offsets, dependency_groups=dependency_groups, required_local_bytes=slot_bytes * len(slots),
    alias_safe=alias_safe)


def lds_stage_plan_dump(plan: AMDLDSStagePlan) -> dict[str, Any]:
  return {
    "plan": plan.to_dict(),
    "summary": {
      "slot_count": len(plan.slots),
      "stage_count": plan.stage_count,
      "required_local_bytes": plan.required_local_bytes,
      "alias_safe": plan.alias_safe,
      "dependency_group_count": len(plan.dependency_groups),
      "lowering_status": plan.lowering_status,
      "has_two_slots": {0, 1}.issubset(set(plan.slots)),
      "has_deterministic_offsets": list(plan.slot_offsets.values()) == sorted(plan.slot_offsets.values()),
    },
  }


@dataclass(frozen=True)
class AMDLDSLoweredSlot:
  slot: int
  define_local_slot: int
  offset_bytes: int
  size_bytes: int
  element_count: int
  dtype: str
  addrspace: str
  roles: tuple[str, ...]
  lowering_status: str = "lowered_define_local"

  def to_dict(self) -> dict[str, Any]:
    ret = asdict(self)
    ret["roles"] = list(self.roles)
    return ret


def lower_lds_stage_plan_to_define_locals(plan: AMDLDSStagePlan, dtype: DType, base_slot: int = 9000) -> tuple[list[UOp], list[AMDLDSLoweredSlot]]:
  if not plan.alias_safe: raise ValueError("cannot lower an alias-unsafe LDS stage plan")
  if not plan.slots: raise ValueError("cannot lower an empty LDS stage plan")
  if plan.required_local_bytes % len(plan.slots) != 0:
    raise ValueError("required_local_bytes must divide evenly across planned LDS slots")
  slot_bytes = plan.required_local_bytes // len(plan.slots)
  itemsize = dtype.itemsize
  if itemsize <= 0 or slot_bytes % itemsize != 0:
    raise ValueError(f"slot_bytes={slot_bytes} must be a multiple of dtype itemsize={itemsize}")
  locals_: list[UOp] = []
  records: list[AMDLDSLoweredSlot] = []
  for slot in plan.slots:
    define_slot = base_slot + slot
    element_count = slot_bytes // itemsize
    locals_.append(UOp.placeholder((element_count,), dtype, define_slot, AddrSpace.LOCAL))
    records.append(AMDLDSLoweredSlot(
      slot=slot, define_local_slot=define_slot, offset_bytes=plan.slot_offsets[slot], size_bytes=slot_bytes,
      element_count=element_count, dtype=str(dtype), addrspace="lds", roles=plan.slot_roles.get(slot, ())))
  return locals_, records


def lds_lowering_dump(plan: AMDLDSStagePlan, lowered: Iterable[AMDLDSLoweredSlot]) -> dict[str, Any]:
  data = list(lowered)
  return {
    "lowered_slots": [row.to_dict() for row in data],
    "summary": {
      "lowered_slot_count": len(data),
      "slots": [row.slot for row in data],
      "define_local_slots": [row.define_local_slot for row in data],
      "offsets": [row.offset_bytes for row in data],
      "required_local_bytes": plan.required_local_bytes,
      "lowered_local_bytes": sum(row.size_bytes for row in data),
      "addrspaces": sorted({row.addrspace for row in data}),
      "lowering_status": "lowered_define_local" if data and all(row.lowering_status == "lowered_define_local" for row in data) else "incomplete",
    },
  }


def _uop_mem_space(u: UOp) -> str | None:
  if u.op is Ops.LOAD:
    target = u.src[0]
    return _addrspace_name(target.addrspace)
  if u.op is Ops.STORE:
    target = u.src[0]
    return _addrspace_name(target.addrspace)
  if u.op in {Ops.DEFINE_LOCAL, Ops.DEFINE_REG, Ops.BUFFER, Ops.PARAM}: return _addrspace_name(u.addrspace)
  return None


def _addrspace_name(addrspace: AddrSpace | None) -> str | None:
  if addrspace is AddrSpace.GLOBAL: return "global"
  if addrspace is AddrSpace.LOCAL: return "lds"
  if addrspace is AddrSpace.REG: return "register"
  return None


def _uop_latency_class(u: UOp) -> str:
  if u.op is Ops.WMMA: return "wmma"
  if u.op is Ops.BARRIER: return "barrier"
  if u.op is Ops.LOAD:
    space = _uop_mem_space(u)
    return {"global": "global_memory", "lds": "lds_memory", "register": "register"}.get(space or "", "memory")
  if u.op is Ops.STORE:
    space = _uop_mem_space(u)
    return {"global": "global_store", "lds": "lds_store", "register": "register"}.get(space or "", "store")
  if u.op in GroupOp.ALU: return "valu"
  if u.op in {Ops.RANGE, Ops.END, Ops.IF, Ops.ENDIF, Ops.SPECIAL}: return "control"
  if u.op in {Ops.DEFINE_LOCAL, Ops.DEFINE_REG, Ops.BUFFER, Ops.PARAM}: return "resource"
  return "other"


def _uop_wait_group(u: UOp) -> str | None:
  if u.op is Ops.LOAD:
    return {"global": "vmcnt", "lds": "lgkmcnt"}.get(_uop_mem_space(u) or "")
  if u.op is Ops.STORE:
    return {"global": "vscnt", "lds": "lgkmcnt"}.get(_uop_mem_space(u) or "")
  if u.op is Ops.WMMA: return "wmma_dependency"
  if u.op is Ops.BARRIER: return "barrier"
  return None


def _uop_issue_cluster(u: UOp) -> str | None:
  if u.op is Ops.WMMA: return "wmma"
  if u.op in GroupOp.ALU: return "valu"
  if u.op in {Ops.LOAD, Ops.STORE}: return {"global": "vmem", "lds": "lds", "register": "valu"}.get(_uop_mem_space(u) or "")
  if u.op in {Ops.RANGE, Ops.END, Ops.IF, Ops.ENDIF, Ops.SPECIAL}: return "salu"
  return None


def _dtype_name(u: UOp) -> str | None:
  try: return str(u.dtype)
  except Exception: return None


def metadata_from_uops(uops: Iterable[UOp]) -> list[AMDScheduleMeta]:
  rows: list[AMDScheduleMeta] = []
  for idx, u in enumerate(uops):
    if u.op in {Ops.NOOP, Ops.SINK}: continue
    space = _uop_mem_space(u)
    rows.append(AMDScheduleMeta(
      idx=idx, source="uop", op=u.op.name, dtype=_dtype_name(u), latency_class=_uop_latency_class(u),
      memory_space=space, vector_width=getattr(u.dtype, "count", None), wait_group=_uop_wait_group(u),
      barrier_scope="workgroup" if u.op is Ops.BARRIER else None,
      live_range_boundary="define" if u.op in {Ops.DEFINE_LOCAL, Ops.DEFINE_REG, Ops.PARAM, Ops.BUFFER} else "use" if u.op in {Ops.LOAD, Ops.STORE, Ops.WMMA} else None,
      issue_cluster=_uop_issue_cluster(u),
      prefetch_stage=0 if u.op is Ops.LOAD and space == "global" else None,
      lds_stage=0 if space == "lds" else None,
      register_pressure_budget="tracked_later" if u.op in {Ops.DEFINE_REG, Ops.WMMA} or u.op in GroupOp.ALU else None))
  return rows


def _inst_name(inst: Inst) -> str:
  op = getattr(inst, "op", None)
  return getattr(op, "name", None) or getattr(inst, "op_name", None) or type(inst).__name__


def _inst_family(inst: Inst) -> str:
  return type(inst).__name__.split("_", 1)[0].lower()


def _inst_latency_class(name: str, family: str) -> str:
  lname = name.lower()
  if "wmma" in lname or "mfma" in lname: return "wmma"
  if lname.startswith(("global_store", "flat_store", "scratch_store")): return "global_store"
  if lname.startswith(("global_", "flat_", "scratch_")) or family in {"global", "flat", "scratch"}: return "global_memory"
  if lname.startswith("ds_") or family == "ds": return "lds_memory"
  if lname.startswith("s_waitcnt"): return "wait"
  if lname.startswith("s_barrier"): return "barrier"
  if lname.startswith("s_"): return "salu"
  if lname.startswith("v_"): return "valu"
  return family or "other"


def _inst_memory_space(name: str, family: str) -> str | None:
  lname = name.lower()
  if lname.startswith(("global_", "flat_", "scratch_")) or family in {"global", "flat", "scratch"}: return "global"
  if lname.startswith("ds_") or family == "ds": return "lds"
  return None


def _inst_wait_group(name: str, space: str | None) -> str | None:
  lname = name.lower()
  if lname.startswith("s_waitcnt"):
    if "vmcnt" in lname: return "vmcnt"
    if "lgkmcnt" in lname: return "lgkmcnt"
    return "waitcnt"
  if space == "global": return "vmcnt"
  if space == "lds": return "lgkmcnt"
  if "wmma" in lname or "mfma" in lname: return "wmma_dependency"
  return None


def _inst_issue_cluster(name: str, family: str, space: str | None) -> str | None:
  lname = name.lower()
  if "wmma" in lname or "mfma" in lname: return "wmma"
  if space == "global": return "vmem"
  if space == "lds": return "lds"
  if lname.startswith("s_") or family.startswith("sop"): return "salu"
  if lname.startswith("v_") or family.startswith("vop"): return "valu"
  return family or None


def metadata_from_instructions(instructions: Iterable[Inst]) -> list[AMDScheduleMeta]:
  rows: list[AMDScheduleMeta] = []
  for idx, inst in enumerate(instructions):
    name, family = _inst_name(inst), _inst_family(inst)
    space = _inst_memory_space(name, family)
    rows.append(AMDScheduleMeta(
      idx=idx, source="instruction", op=name, dtype=None, latency_class=_inst_latency_class(name, family),
      memory_space=space, vector_width=None, wait_group=_inst_wait_group(name, space),
      barrier_scope="workgroup" if name.lower().startswith("s_barrier") else None,
      live_range_boundary="use" if space is not None or "wmma" in name.lower() or "mfma" in name.lower() else None,
      issue_cluster=_inst_issue_cluster(name, family, space),
      prefetch_stage=0 if space == "global" else None, lds_stage=0 if space == "lds" else None,
      register_pressure_budget="tracked_later" if name.lower().startswith("v_") or "wmma" in name.lower() or "mfma" in name.lower() else None))
  return rows


def schedule_metadata_summary(rows: Iterable[AMDScheduleMeta]) -> dict[str, Any]:
  data = list(rows)
  def counts(attr: str) -> dict[str, int]:
    ret: dict[str, int] = {}
    for row in data:
      val = getattr(row, attr)
      if val is not None: ret[str(val)] = ret.get(str(val), 0) + 1
    return ret
  fields = ("latency_class", "memory_space", "wait_group", "barrier_scope", "live_range_boundary", "issue_cluster")
  return {
    "row_count": len(data),
    "field_coverage": {field: sum(1 for row in data if getattr(row, field) is not None) for field in fields},
    "counts": {field: counts(field) for field in fields},
  }


def schedule_metadata_dump(rows: Iterable[AMDScheduleMeta]) -> dict[str, Any]:
  data = list(rows)
  return {"rows": [row.to_dict() for row in data], "summary": schedule_metadata_summary(data)}


def plan_schedule_actions(rows: Iterable[AMDScheduleMeta]) -> list[AMDScheduleAction]:
  data = list(rows)
  actions: list[AMDScheduleAction] = []
  in_global_clause = False
  pending_wait: str | None = None
  for row in data:
    if row.memory_space == "global" and row.latency_class in {"global_memory", "global_store"}:
      if not in_global_clause:
        actions.append(AMDScheduleAction(row.idx, "insert_s_clause", "begin contiguous global-memory clause", row.op, row.wait_group))
      in_global_clause = True
      pending_wait = row.wait_group or pending_wait
      continue
    in_global_clause = False
    if pending_wait is not None and row.issue_cluster in {"valu", "wmma", "lds"}:
      actions.append(AMDScheduleAction(row.idx, "ensure_s_waitcnt", "consumer uses prior memory result", row.op, pending_wait))
      pending_wait = None
    if row.issue_cluster in {"valu", "wmma"}:
      actions.append(AMDScheduleAction(row.idx + 1, "insert_s_delay_alu", "space dependent ALU/WMMA issue", row.op, row.wait_group))
  return actions


def apply_instruction_schedule(instructions: list[Inst]) -> tuple[list[Inst], list[AMDScheduleAction]]:
  from tinygrad.runtime.autogen.amd.rdna3.ins import s_clause, s_delay_alu, s_waitcnt

  rows = metadata_from_instructions(instructions)
  actions = plan_schedule_actions(rows)
  before: dict[int, list[Inst]] = {}
  after: dict[int, list[Inst]] = {}
  for action in actions:
    if action.action == "insert_s_clause": before.setdefault(action.idx, []).append(s_clause(simm16=0))
    elif action.action == "ensure_s_waitcnt": before.setdefault(action.idx, []).append(s_waitcnt(simm16=0))
    elif action.action == "insert_s_delay_alu": after.setdefault(action.idx - 1, []).append(s_delay_alu(simm16=0))
  out: list[Inst] = []
  for idx, inst in enumerate(instructions):
    out.extend(before.get(idx, ()))
    out.append(inst)
    out.extend(after.get(idx, ()))
  out.extend(before.get(len(instructions), ()))
  return out, actions


def _inst_registers(inst: Inst) -> list[Reg]:
  regs: list[Reg] = []
  for name, field in getattr(type(inst), "_fields", ()):
    if isinstance(field, FixedBitField): continue
    try: val = getattr(inst, name)
    except Exception: continue
    if isinstance(val, Reg): regs.append(val)
  return regs


def resource_summary_from_instructions(instructions: Iterable[Inst]) -> dict[str, Any]:
  insts = list(instructions)
  regs = [r for inst in insts for r in _inst_registers(inst)]
  vgprs = [(r.offset - 256, r.offset - 256 + r.sz - 1) for r in regs if 256 <= r.offset < 512]
  sgprs = [(r.offset, r.offset + r.sz - 1) for r in regs if 0 <= r.offset < 128]
  def span(items: list[tuple[int, int]]) -> dict[str, int | None]:
    if not items: return {"min": None, "max": None, "span": 0}
    lo, hi = min(x[0] for x in items), max(x[1] for x in items)
    return {"min": lo, "max": hi, "span": hi - lo + 1}
  return {
    "vgpr": span(vgprs),
    "sgpr": span(sgprs),
    "instruction_count": len(insts),
    "register_operand_count": len(regs),
    "occupancy_policy": "measure_after_scheduler_changes",
    "spill_risk": "unknown_until_allocator",
  }


def resource_summary_from_metadata(rows: Iterable[AMDScheduleMeta]) -> dict[str, Any]:
  data = list(rows)
  counts = schedule_metadata_summary(data)["counts"]
  return {
    "row_count": len(data),
    "register_rows": counts["memory_space"].get("register", 0),
    "wmma_rows": counts["latency_class"].get("wmma", 0),
    "valu_rows": counts["issue_cluster"].get("valu", 0),
    "global_rows": counts["memory_space"].get("global", 0),
    "lds_rows": counts["memory_space"].get("lds", 0),
    "register_pressure_budget_rows": sum(1 for row in data if row.register_pressure_budget is not None),
    "occupancy_policy": "metadata_only_no_allocator_change",
    "spill_risk": "high" if counts["latency_class"].get("wmma", 0) and counts["memory_space"].get("register", 0) > 128 else "unknown",
  }


# ---------------------------------------------------------------------------
# First-class AMD GEMM schedule object (structural, UNWIRED).
#
# This represents a shape-specialized, LDS-staged, software-pipelined GEMM as a
# single object: shape contract + LDS layout + ordered pipeline stages +
# resource gate + ISA-evidence ledger, plus a structural gate evaluated BEFORE
# any timing. It is the surface the Tensile transfer table named as missing
# ("a Tensile-class GEMM schedule object", not the WMMA atom and not "add LDS").
#
# It changes NO default behavior and is not wired into the live compile path.
# It carries structure and gates only; it makes NO performance claim and does
# not lower to ISA. The fields encode the selected rocBLAS ffn_gate/up contract
# (M=512,N=12288,K=4096, MT128x128x16, TT4_64, WG32x4x1, DepthU=16, WGM8, PGR1
# double-buffer, LDS=25088, scratch=0). Performance remains a separate, later
# gate; this object only makes the contract first-class so it can be inspected.
# ---------------------------------------------------------------------------

# The named pipeline stages a Tensile-class GEMM schedule must express, in order.
GEMM_PIPELINE_STAGES: tuple[str, ...] = (
  "global_load_A", "global_load_B",
  "wait_global_before_lds",
  "lds_store_A", "lds_store_B",
  "barrier_after_lds_store",
  "lds_read_A", "lds_read_B",
  "wait_lds_before_wmma",
  "wmma_consume",
  "store_output",
  "buffer_swap",
)


@dataclass(frozen=True)
class AMDGemmShapeContract:
  role: str
  m: int
  n: int
  k: int
  dtype_in: str
  dtype_acc: str
  macro_tile: tuple[int, int, int]   # [M, N, unroll] e.g. [128, 128, 16]
  thread_tile: tuple[int, int]       # [4, 64]
  work_group: tuple[int, int, int]   # [32, 4, 1]
  depth_u: int                       # 16
  workgroup_mapping: int             # WGM8
  grid: tuple[int, int, int]         # [512, 96, 1]
  workgroup: tuple[int, int, int]    # [128, 1, 1]

  def to_dict(self) -> dict[str, Any]:
    ret = asdict(self)
    for key in ("macro_tile", "thread_tile", "work_group", "grid", "workgroup"):
      ret[key] = list(getattr(self, key))
    return ret

  @property
  def flops(self) -> int: return 2 * self.m * self.n * self.k

  def structural_checks(self) -> dict[str, bool]:
    wg_threads = self.work_group[0] * self.work_group[1] * self.work_group[2]
    return {
      "shape_positive": self.m > 0 and self.n > 0 and self.k > 0,
      "macro_tile_3d": len(self.macro_tile) == 3,
      "workgroup_threads_match_launch": wg_threads == (self.workgroup[0] * self.workgroup[1] * self.workgroup[2]),
      "k_divisible_by_depth_u": self.depth_u > 0 and self.k % self.depth_u == 0,
      "grid_tiles_cover_mn": self.grid[0] * self.workgroup[0] >= self.m and self.grid[1] * self.macro_tile[1] >= self.n,
    }


@dataclass(frozen=True)
class AMDGemmLDSRegion:
  name: str          # A0, B0, A1, B1
  operand: str       # A or B
  buffer_slot: int   # 0 lower / 1 second (PGR1 double buffer)
  byte_base: int
  byte_span: int
  pad_bytes: int     # per-region LDS padding (B uses LdsPadB)
  role: str          # "current/prefetch A tile" etc.

  def to_dict(self) -> dict[str, Any]: return asdict(self)

  @property
  def byte_end(self) -> int: return self.byte_base + self.byte_span


@dataclass(frozen=True)
class AMDGemmLDSLayout:
  regions: tuple[AMDGemmLDSRegion, ...]
  total_bytes: int                    # group_segment_fixed_size, e.g. 25088
  alignment_gap: tuple[int, int]      # (gap_byte_base, gap_byte_span); PGR1 pow2 second-buffer gap

  def to_dict(self) -> dict[str, Any]:
    return {
      "regions": [r.to_dict() for r in self.regions],
      "total_bytes": self.total_bytes,
      "alignment_gap": list(self.alignment_gap),
      "structural_checks": self.structural_checks(),
    }

  def operand_slots(self) -> dict[tuple[str, int], AMDGemmLDSRegion]:
    return {(r.operand, r.buffer_slot): r for r in self.regions}

  def _alias_safe(self) -> bool:
    ranges = sorted((r.byte_base, r.byte_end) for r in self.regions)
    return all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:]))

  def structural_checks(self) -> dict[str, bool]:
    slots = self.operand_slots()
    gap_base, gap_span = self.alignment_gap
    covered = sum(r.byte_span for r in self.regions) + gap_span
    max_end = max((r.byte_end for r in self.regions), default=0)
    return {
      "nonzero_lds": self.total_bytes > 0 and bool(self.regions),
      "double_buffer_present": {("A", 0), ("A", 1), ("B", 0), ("B", 1)}.issubset(set(slots)),
      "alias_safe": self._alias_safe(),
      "regions_within_total": max_end <= self.total_bytes,
      "spans_plus_gap_equal_total": covered == self.total_bytes,
      "gap_after_lower_buffers": gap_base >= 0 and gap_span >= 0,
    }


@dataclass(frozen=True)
class AMDGemmPipelineStage:
  order: int
  stage: str                    # one of GEMM_PIPELINE_STAGES
  phase: str                    # prologue / steady / epilogue
  op_class: str                 # global_load / wait / lds_store / barrier / lds_load / wmma / global_store / swap
  operand: str | None           # A / B / None
  buffer_slot: int | None       # which LDS slot this stage touches
  produces_for: str | None      # downstream stage name (dependency edge)
  wait_group: str | None        # vmcnt / lgkmcnt / barrier / wmma_dependency
  isa_evidence: str | None      # buffer_load_b64 / ds_store_b64 / ds_load_b128 / v_wmma / s_waitcnt / s_barrier

  def to_dict(self) -> dict[str, Any]: return asdict(self)


@dataclass(frozen=True)
class AMDGemmResourceGate:
  lds_bytes_target: int         # 25088
  lds_bytes_actual: int
  private_scratch_required: int  # 0
  private_scratch_actual: int
  vgpr_budget: int              # 256
  sgpr_budget: int              # 58

  def to_dict(self) -> dict[str, Any]:
    ret = asdict(self)
    ret["evaluation"] = self.evaluate()
    return ret

  def evaluate(self) -> dict[str, bool]:
    return {
      "lds_within_target": 0 < self.lds_bytes_actual <= self.lds_bytes_target,
      "no_private_scratch": self.private_scratch_actual == self.private_scratch_required == 0,
      "vgpr_budget_set": 0 < self.vgpr_budget <= 256,
      "sgpr_budget_set": 0 < self.sgpr_budget <= 128,
    }


@dataclass(frozen=True)
class AMDGemmISAEvidence:
  # Counts/flags from the audited selected-kernel disassembly. The schedule object
  # carries them so the structural gate can be evaluated without re-disassembling.
  global_load: int
  ds_store: int
  ds_load_b128: int
  v_wmma: int
  s_waitcnt: int
  s_barrier: int
  lds_store_reuses_global_regs: bool   # global-load regs feed ds_store (handoff)
  wmma_operands_from_lds: bool         # v_wmma consumes ds_load_b128 dest VGPRs

  def to_dict(self) -> dict[str, Any]: return asdict(self)

  def structural_checks(self) -> dict[str, bool]:
    return {
      "visible_global_load": self.global_load > 0,
      "visible_ds_store": self.ds_store > 0,
      "visible_ds_load_b128": self.ds_load_b128 > 0,
      "visible_v_wmma": self.v_wmma > 0,
      "waits_present": self.s_waitcnt > 0,
      "barriers_present": self.s_barrier > 0,
      "wmma_fed_from_lds": self.wmma_operands_from_lds,
      "global_to_lds_handoff": self.lds_store_reuses_global_regs,
    }


@dataclass(frozen=True)
class AMDGemmScheduleObject:
  shape: AMDGemmShapeContract
  lds: AMDGemmLDSLayout
  pipeline: tuple[AMDGemmPipelineStage, ...]
  resource_gate: AMDGemmResourceGate
  isa_evidence: AMDGemmISAEvidence | None
  blocked_unknown: tuple[str, ...]   # explicit non-bitexact / unreconstructed rows
  lowering_status: str = "structural_unwired"
  performance_claim: bool = False

  def to_dict(self) -> dict[str, Any]:
    return {
      "shape": self.shape.to_dict(),
      "lds": self.lds.to_dict(),
      "pipeline": [s.to_dict() for s in self.pipeline],
      "resource_gate": self.resource_gate.to_dict(),
      "isa_evidence": self.isa_evidence.to_dict() if self.isa_evidence is not None else None,
      "blocked_unknown": list(self.blocked_unknown),
      "lowering_status": self.lowering_status,
      "performance_claim": self.performance_claim,
      "structural_gate": self.structural_gate(),
    }

  def _pipeline_checks(self) -> dict[str, bool]:
    present = {s.stage for s in self.pipeline}
    orders = [s.order for s in self.pipeline]
    return {
      "all_named_stages_present": set(GEMM_PIPELINE_STAGES).issubset(present),
      "order_monotonic": orders == sorted(orders),
      "has_prologue_and_steady": {"prologue", "steady"}.issubset({s.phase for s in self.pipeline}),
      "global_before_wmma": (min((s.order for s in self.pipeline if s.op_class == "global_load"), default=1)
                             < min((s.order for s in self.pipeline if s.op_class == "wmma"), default=0)),
      "has_buffer_swap": any(s.op_class == "swap" for s in self.pipeline),
    }

  def structural_gate(self) -> dict[str, Any]:
    """All structural gates that must pass BEFORE any timing. No performance is claimed here."""
    checks: dict[str, bool] = {}
    for prefix, sub in (("shape", self.shape.structural_checks()), ("lds", self.lds.structural_checks()),
                        ("resource", self.resource_gate.evaluate()), ("pipeline", self._pipeline_checks())):
      for name, ok in sub.items(): checks[f"{prefix}.{name}"] = bool(ok)
    if self.isa_evidence is not None:
      for name, ok in self.isa_evidence.structural_checks().items(): checks[f"isa.{name}"] = bool(ok)
    return {"checks": checks, "passed": all(checks.values()), "performance_claim": self.performance_claim}


def gemm_schedule_object_summary(obj: AMDGemmScheduleObject) -> dict[str, Any]:
  gate = obj.structural_gate()
  failed = [k for k, v in gate["checks"].items() if not v]
  return {
    "role": obj.shape.role,
    "shape": [obj.shape.m, obj.shape.n, obj.shape.k],
    "flops": obj.shape.flops,
    "lds_total_bytes": obj.lds.total_bytes,
    "lds_region_count": len(obj.lds.regions),
    "pipeline_stage_count": len(obj.pipeline),
    "blocked_unknown_count": len(obj.blocked_unknown),
    "lowering_status": obj.lowering_status,
    "performance_claim": obj.performance_claim,
    "structural_gate_passed": gate["passed"],
    "failed_checks": failed,
  }


# ---------------------------------------------------------------------------
# First-class AMD decode MMVQ schedule object (structural, UNWIRED).
#
# This is the decode analogue of AMDGemmScheduleObject, but the primitive is not
# dense GEMM. It represents a small-batch quantized matvec lifecycle: generated q8_1
# activation production/reuse, packed Q4_K/Q6_K weight load, packed extract or
# dequant-dot, reduction/output, route policy, and evidence labels.
#
# It changes NO default behavior and does not lower to ISA. It is metadata plus
# gates only, so existing shipped Q4/Q6 decode paths can be inspected against
# one primitive contract before any native renderer or search work starts.
# ---------------------------------------------------------------------------

DECODE_MMVQ_STAGES: tuple[str, ...] = (
  "activation_prepare",
  "activation_q8_producer",
  "activation_reuse",
  "packed_weight_load",
  "packed_extract",
  "dot_or_dequant_dot",
  "scale_apply",
  "partial_reduce",
  "output_store",
  "route_policy",
)


@dataclass(frozen=True)
class DecodeMMVQRoleContract:
  role: str
  quant_format: str
  out_features: int
  in_features: int
  batch_max: int
  activation_format: str
  rows_per_work_item: str
  reduction: str
  route_status: str

  def to_dict(self) -> dict[str, Any]: return asdict(self)

  def structural_checks(self) -> dict[str, bool]:
    return {
      "shape_positive": self.out_features > 0 and self.in_features > 0,
      "decode_small_batch": 1 <= self.batch_max <= 8,
      "quant_format_known": self.quant_format in {"Q4_K", "Q6_K"},
      "activation_format_known": self.activation_format in {"fp16", "q8_1"},
      "reduction_named": bool(self.reduction),
      "route_status_named": bool(self.route_status),
    }


@dataclass(frozen=True)
class DecodeMMVQStage:
  order: int
  stage: str
  op_class: str
  operand: str | None
  evidence: str
  authority: str

  def to_dict(self) -> dict[str, Any]: return asdict(self)


@dataclass(frozen=True)
class DecodeMMVQResourceGate:
  max_batch: int
  requires_q8_activation_buffer: bool
  lossy_activation: bool
  default_on_allowed: bool
  native_renderer_owned: bool

  def to_dict(self) -> dict[str, Any]:
    ret = asdict(self)
    ret["evaluation"] = self.evaluate()
    return ret

  def evaluate(self) -> dict[str, bool]:
    return {
      "small_batch_policy": 1 <= self.max_batch <= 8,
      "lossy_default_guarded": (not self.lossy_activation) or (not self.default_on_allowed),
      "ownership_explicit": isinstance(self.native_renderer_owned, bool),
      "q8_lifecycle_explicit": isinstance(self.requires_q8_activation_buffer, bool),
    }


@dataclass(frozen=True)
class DecodeMMVQEvidence:
  llama_source_contract: bool
  tinygrad_route_present: bool
  correctness_or_quality_gate: bool
  timing_grade_native_feature: bool
  imported_or_artifact_route: bool
  q8_role_joined_body: bool

  def to_dict(self) -> dict[str, Any]: return asdict(self)

  def structural_checks(self) -> dict[str, bool]:
    return {
      "llama_source_contract_present": self.llama_source_contract,
      "tinygrad_route_present": self.tinygrad_route_present,
      "correctness_or_quality_gate_present": self.correctness_or_quality_gate,
      "native_feature_gate_labeled": isinstance(self.timing_grade_native_feature, bool),
      "route_ownership_labeled": isinstance(self.imported_or_artifact_route, bool),
      "role_join_gap_labeled": isinstance(self.q8_role_joined_body, bool),
    }


@dataclass(frozen=True)
class DecodeMMVQScheduleObject:
  contract: DecodeMMVQRoleContract
  stages: tuple[DecodeMMVQStage, ...]
  resource_gate: DecodeMMVQResourceGate
  evidence: DecodeMMVQEvidence
  blocked_unknown: tuple[str, ...]
  lowering_status: str = "structural_unwired"
  performance_claim: bool = False

  def to_dict(self) -> dict[str, Any]:
    return {
      "contract": self.contract.to_dict(),
      "stages": [s.to_dict() for s in self.stages],
      "resource_gate": self.resource_gate.to_dict(),
      "evidence": self.evidence.to_dict(),
      "blocked_unknown": list(self.blocked_unknown),
      "lowering_status": self.lowering_status,
      "performance_claim": self.performance_claim,
      "structural_gate": self.structural_gate(),
    }

  def _stage_checks(self) -> dict[str, bool]:
    present = {s.stage for s in self.stages}
    orders = [s.order for s in self.stages]
    authorities = {s.authority for s in self.stages}
    return {
      "all_named_stages_present": set(DECODE_MMVQ_STAGES).issubset(present),
      "order_monotonic": orders == sorted(orders),
      "has_source_or_runtime_authority": bool(authorities & {"source", "runtime", "quality", "timing", "static"}),
      "packed_weight_before_dot": (
        min((s.order for s in self.stages if s.stage == "packed_weight_load"), default=1) <
        min((s.order for s in self.stages if s.stage == "dot_or_dequant_dot"), default=0)),
      "output_after_reduce": (
        min((s.order for s in self.stages if s.stage == "partial_reduce"), default=1) <
        min((s.order for s in self.stages if s.stage == "output_store"), default=0)),
    }

  def structural_gate(self) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for prefix, sub in (("contract", self.contract.structural_checks()), ("stage", self._stage_checks()),
                        ("resource", self.resource_gate.evaluate()), ("evidence", self.evidence.structural_checks())):
      for name, ok in sub.items(): checks[f"{prefix}.{name}"] = bool(ok)
    return {"checks": checks, "passed": all(checks.values()), "performance_claim": self.performance_claim}


def decode_mmvq_schedule_object_summary(obj: DecodeMMVQScheduleObject) -> dict[str, Any]:
  gate = obj.structural_gate()
  failed = [k for k, v in gate["checks"].items() if not v]
  return {
    "role": obj.contract.role,
    "quant_format": obj.contract.quant_format,
    "shape": [obj.contract.out_features, obj.contract.in_features],
    "stage_count": len(obj.stages),
    "blocked_unknown_count": len(obj.blocked_unknown),
    "lowering_status": obj.lowering_status,
    "performance_claim": obj.performance_claim,
    "native_renderer_owned": obj.resource_gate.native_renderer_owned,
    "structural_gate_passed": gate["passed"],
    "failed_checks": failed,
  }


@dataclass(frozen=True)
class DecodeMMVQArtifactLaunchContract:
  runtime_name: str
  global_size: tuple[int, int, int]
  local_size: tuple[int, int, int]
  kernarg_size: int
  group_segment_size: int
  private_segment_size: int

  def to_dict(self) -> dict[str, Any]:
    ret = asdict(self)
    ret["global_size"] = list(self.global_size)
    ret["local_size"] = list(self.local_size)
    return ret

  def structural_checks(self, *, expected_global: tuple[int, int, int], expected_local: tuple[int, int, int],
                        expected_kernarg: int, expected_group_segment: int) -> dict[str, bool]:
    return {
      "runtime_named": bool(self.runtime_name),
      "global_size_matches_oracle": self.global_size == expected_global,
      "local_size_matches_oracle": self.local_size == expected_local,
      "kernarg_size_matches_oracle": self.kernarg_size == expected_kernarg,
      "group_segment_matches_oracle": self.group_segment_size == expected_group_segment,
      "private_segment_zero": self.private_segment_size == 0,
    }


@dataclass(frozen=True)
class DecodeMMVQInstructionContract:
  dot4: int
  fma: int
  convert: int
  valu: int
  salu: int
  ds: int
  barrier: int
  global_load: int
  global_store: int
  shuffle: int
  branch: int
  waitcnt: int

  def to_dict(self) -> dict[str, Any]: return asdict(self)

  def structural_checks(self) -> dict[str, bool]:
    return {
      "dot4_matches_oracle": self.dot4 == 16,
      "single_output_store": self.global_store == 1,
      "single_barrier": self.barrier == 1,
      "shuffle_topology_matches_oracle": self.shuffle == 5,
      "has_wait_policy": self.waitcnt > 0,
      "has_branch_policy": self.branch > 0,
      "global_load_budget_oracle": self.global_load <= 11,
      "ds_budget_oracle": self.ds <= 7,
      "valu_present": self.valu > 0,
      "salu_present": self.salu > 0,
    }


@dataclass(frozen=True)
class DecodeMMVQArtifactOracleBinding:
  producer: DecodeMMVQArtifactLaunchContract
  gateup: DecodeMMVQArtifactLaunchContract
  work_decomposition: str
  instruction_contract: DecodeMMVQInstructionContract
  correctness_passed: bool
  default_changed: bool
  route_status: str = "hardened_opt_in_oracle"
  lowering_status: str = "structural_oracle_binding_unwired"
  performance_claim: bool = False

  def to_dict(self) -> dict[str, Any]:
    return {
      "producer": self.producer.to_dict(),
      "gateup": self.gateup.to_dict(),
      "work_decomposition": self.work_decomposition,
      "instruction_contract": self.instruction_contract.to_dict(),
      "correctness_passed": self.correctness_passed,
      "default_changed": self.default_changed,
      "route_status": self.route_status,
      "lowering_status": self.lowering_status,
      "performance_claim": self.performance_claim,
      "structural_gate": self.structural_gate(),
    }

  def structural_gate(self) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for prefix, sub in (
      ("producer", self.producer.structural_checks(expected_global=(1, 1, 1), expected_local=(1024, 1, 1),
                                                   expected_kernarg=32, expected_group_segment=4096)),
      ("gateup", self.gateup.structural_checks(expected_global=(12288, 2, 1), expected_local=(32, 4, 1),
                                               expected_kernarg=40, expected_group_segment=16)),
      ("instruction", self.instruction_contract.structural_checks()),
    ):
      for name, ok in sub.items(): checks[f"{prefix}.{name}"] = bool(ok)
    checks["work_decomposition_names_lane_mapping"] = all(x in self.work_decomposition for x in ("128 threads per row", "sub=tid&7", "kb=tid/8"))
    checks["correctness_passed"] = self.correctness_passed
    checks["default_unchanged"] = not self.default_changed
    checks["route_status_named"] = bool(self.route_status)
    checks["lowering_unwired"] = self.lowering_status == "structural_oracle_binding_unwired"
    checks["no_performance_claim"] = not self.performance_claim
    return {"checks": checks, "passed": all(checks.values()), "performance_claim": self.performance_claim}


def decode_mmvq_artifact_oracle_binding_summary(binding: DecodeMMVQArtifactOracleBinding) -> dict[str, Any]:
  gate = binding.structural_gate()
  failed = [k for k, v in gate["checks"].items() if not v]
  return {
    "producer_runtime": binding.producer.runtime_name,
    "gateup_runtime": binding.gateup.runtime_name,
    "gateup_global": list(binding.gateup.global_size),
    "gateup_local": list(binding.gateup.local_size),
    "gateup_group_segment_size": binding.gateup.group_segment_size,
    "gateup_private_segment_size": binding.gateup.private_segment_size,
    "dot4": binding.instruction_contract.dot4,
    "global_load": binding.instruction_contract.global_load,
    "ds": binding.instruction_contract.ds,
    "structural_gate_passed": gate["passed"],
    "failed_checks": failed,
    "lowering_status": binding.lowering_status,
    "performance_claim": binding.performance_claim,
  }


@dataclass(frozen=True)
class DecodeMMVQSchedulerFeature:
  name: str
  category: str
  native_state: str
  oracle_state: str
  standalone_movement_us: float | None
  required_policy: str
  implementation_status: str

  def to_dict(self) -> dict[str, Any]: return asdict(self)

  def structural_checks(self) -> dict[str, bool]:
    return {
      "name_present": bool(self.name),
      "category_present": bool(self.category),
      "native_state_present": bool(self.native_state),
      "oracle_state_present": bool(self.oracle_state),
      "required_policy_present": bool(self.required_policy),
      "implementation_status_present": bool(self.implementation_status),
    }


@dataclass(frozen=True)
class DecodeMMVQResourceLedger:
  native_time_us: float
  oracle_time_us: float
  native_group_segment_size: int
  oracle_group_segment_size: int
  native_private_segment_size: int
  oracle_private_segment_size: int
  native_global_loads: int
  oracle_global_loads: int
  native_ds_ops: int
  oracle_ds_ops: int
  native_waitcnt: int
  oracle_waitcnt: int
  native_branch: int
  oracle_branch: int
  native_s_clause: int
  oracle_s_clause: int
  native_s_delay_alu: int
  oracle_s_delay_alu: int

  def to_dict(self) -> dict[str, Any]:
    ret = asdict(self)
    ret["native_minus_oracle_us"] = self.native_time_us - self.oracle_time_us
    return ret

  def structural_checks(self) -> dict[str, bool]:
    return {
      "native_time_positive": self.native_time_us > 0,
      "oracle_time_positive": self.oracle_time_us > 0,
      "native_slower_than_oracle": self.native_time_us > self.oracle_time_us,
      "no_private_segments": self.native_private_segment_size == 0 and self.oracle_private_segment_size == 0,
      "lds_known": self.native_group_segment_size >= 0 and self.oracle_group_segment_size >= 0,
      "load_delta_visible": self.native_global_loads > self.oracle_global_loads,
      "branch_delta_visible": self.native_branch < self.oracle_branch,
      "scheduler_marker_delta_visible": self.native_s_clause < self.oracle_s_clause and self.native_s_delay_alu < self.oracle_s_delay_alu,
    }


@dataclass(frozen=True)
class DecodeMMVQSchedulerResourcePlan:
  role: str
  quant_format: str
  resource_ledger: DecodeMMVQResourceLedger
  features: tuple[DecodeMMVQSchedulerFeature, ...]
  required_capabilities: tuple[str, ...]
  closed_standalone_features: tuple[str, ...]
  hardware_attribution_status: str
  lowering_status: str = "scheduler_resource_plan_unwired"
  performance_claim: bool = False

  def to_dict(self) -> dict[str, Any]:
    return {
      "role": self.role,
      "quant_format": self.quant_format,
      "resource_ledger": self.resource_ledger.to_dict(),
      "features": [f.to_dict() for f in self.features],
      "required_capabilities": list(self.required_capabilities),
      "closed_standalone_features": list(self.closed_standalone_features),
      "hardware_attribution_status": self.hardware_attribution_status,
      "lowering_status": self.lowering_status,
      "performance_claim": self.performance_claim,
      "structural_gate": self.structural_gate(),
    }

  def structural_gate(self) -> dict[str, Any]:
    checks: dict[str, bool] = {
      "role_present": bool(self.role),
      "quant_format_q4k_q8": self.quant_format == "Q4_K x q8_1",
      "has_features": len(self.features) >= 5,
      "has_required_capabilities": len(self.required_capabilities) >= 5,
      "closed_standalone_features_recorded": {"dot4", "global_load_shape", "waitcnt", "reduction"}.issubset(set(self.closed_standalone_features)),
      "hardware_attribution_status_recorded": bool(self.hardware_attribution_status),
      "lowering_unwired": self.lowering_status == "scheduler_resource_plan_unwired",
      "no_performance_claim": not self.performance_claim,
    }
    for name, ok in self.resource_ledger.structural_checks().items():
      checks[f"resource.{name}"] = bool(ok)
    for i, feature in enumerate(self.features):
      for name, ok in feature.structural_checks().items():
        checks[f"feature{i}.{name}"] = bool(ok)
    return {"checks": checks, "passed": all(checks.values()), "performance_claim": self.performance_claim}


def decode_mmvq_scheduler_resource_plan_summary(plan: DecodeMMVQSchedulerResourcePlan) -> dict[str, Any]:
  gate = plan.structural_gate()
  failed = [k for k, v in gate["checks"].items() if not v]
  return {
    "role": plan.role,
    "quant_format": plan.quant_format,
    "native_minus_oracle_us": plan.resource_ledger.native_time_us - plan.resource_ledger.oracle_time_us,
    "feature_count": len(plan.features),
    "required_capability_count": len(plan.required_capabilities),
    "closed_standalone_features": list(plan.closed_standalone_features),
    "hardware_attribution_status": plan.hardware_attribution_status,
    "lowering_status": plan.lowering_status,
    "performance_claim": plan.performance_claim,
    "structural_gate_passed": gate["passed"],
    "failed_checks": failed,
  }
