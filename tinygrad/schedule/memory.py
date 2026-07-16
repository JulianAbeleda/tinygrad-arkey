from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Iterator
from tinygrad.device import Device
from tinygrad.helpers import NO_MEMORY_PLANNER, DEBUG, round_up
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes
from tinygrad.runtime.support.memory import TLSFAllocator

def _collect_bufs(u:UOp) -> list[UOp]:
  if u.op is Ops.BUFFER: return [u]
  if u.op is Ops.MEMORY_SEMANTIC and len(u.src) == 1: return _collect_bufs(u.src[0])
  if u.op in {Ops.MSELECT, Ops.MSTACK}: return [b for s in u.src for b in _collect_bufs(s)]
  return []

def _can_plan(b:UOp, held_bufs:set[UOp]) -> bool:
  if b in held_bufs: return False
  devs = (b.device,) if isinstance(b.device, str) else b.device
  return all(not d.startswith(("DISK", "TINYFS")) and hasattr(Device[d].allocator, "_offset") for d in devs)

LaneKey = tuple[str, int]

@dataclass(frozen=True)
class ScheduleMemoryBuffer:
  identity: str
  device: str
  lane: int
  rounded_bytes: int
  first_index: int
  last_index: int
  arena_identity: str
  arena_size: int
  offset: int
  byte_range: tuple[int, int]
  semantic_owner: Any

@dataclass(frozen=True)
class ScheduleMemoryArena:
  identity: str
  device: str
  lane: int
  size: int
  backing_uop: UOp|None = None

@dataclass(frozen=True)
class ScheduleMemoryIndex:
  index: int
  physical_byte_union: int
  # (arena identity, merged live byte ranges)
  arena_ranges: tuple[tuple[str, tuple[tuple[int, int], ...]], ...]

@dataclass(frozen=True)
class ScheduleMemoryManifest:
  buffers: tuple[ScheduleMemoryBuffer, ...]
  arenas: tuple[ScheduleMemoryArena, ...]
  indices: tuple[ScheduleMemoryIndex, ...]
  peak_physical_bytes: int

_memory_manifest_collectors:ContextVar[tuple[Callable[[ScheduleMemoryManifest], None], ...]] = ContextVar("memory_manifest_collectors", default=())

@contextmanager
def collect_memory_plan_manifests(on_manifest:Callable[[ScheduleMemoryManifest], None]|None=None) -> Iterator[list[ScheduleMemoryManifest]]:
  """Collect manifests produced by memory-plan rewrites in this context.

  Scopes compose: a rewrite in a nested scope is appended to both the inner and outer
  collections. Context-local storage keeps concurrent threads/tasks isolated.
  """
  manifests:list[ScheduleMemoryManifest] = []
  if on_manifest is not None and not callable(on_manifest): raise TypeError("on_manifest must be callable")
  def collect(manifest:ScheduleMemoryManifest) -> None:
    # The callback runs synchronously before the rewritten schedule can dispatch, allowing evidence consumers to
    # bind the exact backing UOps while preserving the list-returning inspection API.
    if on_manifest is not None: on_manifest(manifest)
    manifests.append(manifest)
  token = _memory_manifest_collectors.set(_memory_manifest_collectors.get() + (collect,))
  try: yield manifests
  finally: _memory_manifest_collectors.reset(token)

@dataclass(frozen=True)
class _MemoryPlan:
  first: dict[UOp, int]
  last: dict[UOp, int]
  planner_last: dict[UOp, int]
  copy_bufs: set[UOp]
  nbytes: dict[UOp, int]
  offsets: dict[UOp, int]
  arena_sizes: dict[LaneKey, int]
  written: set[UOp]
  copy_sources: dict[UOp, tuple[UOp, ...]]

def _call_write_slots(call:UOp) -> set[int]:
  function = call.src[0]
  if function.op in {Ops.COPY, Ops.SLICE}: return {0}
  if function.op is Ops.PROGRAM: return set(function.arg.outs)
  slots:set[int] = set()
  for node in function.toposort():
    if node.op is not Ops.STORE: continue
    target = node.src[0].buf_uop
    if target.op is Ops.PARAM and hasattr(target.arg, "slot"): slots.add(target.arg.slot)
  return slots

def _lane_key(b:UOp, copy_bufs:set[UOp]) -> LaneKey:
  if not isinstance(b.device, str): raise ValueError(f"memory manifest requires a single resolved buffer device, got {b.device!r}")
  return (b.device, 1 if b in copy_bufs else 0)

def _rounded_nbytes(b:UOp, block_size:int) -> int:
  if not isinstance(b.arg, int): raise ValueError(f"memory manifest incomplete: unresolved size for buffer {b.key.hex()}")
  return round_up(b.arg * b.dtype.itemsize, block_size)

def _make_plan(linear:UOp, held_bufs:set[UOp]) -> _MemoryPlan:
  # Track every logical schedule buffer for evidence. Only plannable, non-held buffers enter shared arenas.
  all_first:dict[UOp, int] = {}
  all_last:dict[UOp, int] = {}
  first_appearance:dict[UOp, int] = {}
  last_appearance:dict[UOp, int] = {}
  copy_bufs:set[UOp] = set()
  written:set[UOp] = set()
  copy_sources:dict[UOp, tuple[UOp, ...]] = {}
  for i, si in enumerate(linear.src):
    all_bufs = [b for src in si.src[1:] for b in _collect_bufs(src)]
    for b in all_bufs:
      all_first.setdefault(b, i)
      all_last[b] = i
    si_bufs = [b for b in all_bufs if _can_plan(b, held_bufs)]
    for b in si_bufs:
      first_appearance.setdefault(b, i)
      last_appearance[b] = i
    args = tuple(x for x in si.src[1:] if x.op is not Ops.BIND)
    if si.src[0].op is Ops.COPY:
      copy_bufs.update(all_bufs)
      if len(args) >= 2:
        sources = tuple(_collect_bufs(args[1]))
        for dest in _collect_bufs(args[0]): copy_sources[dest] = sources
    for slot in _call_write_slots(si):
      if slot < len(args): written.update(_collect_bufs(args[slot]))

  block_size = 256
  nbytes = {b: _rounded_nbytes(b, block_size) for b in all_first}
  buf_hold = {b: last_appearance[b] - first_appearance[b] + 1 for b in first_appearance if b in copy_bufs}
  planner_last = {b: last_appearance[b] + buf_hold.get(b, 0) for b in first_appearance}
  events = sorted([(first_appearance[b], True, b) for b in first_appearance] +
                  [(last_appearance[b] + 1 + buf_hold.get(b, 0), False, b) for b in first_appearance], key=lambda x: (x[0], x[1]))
  # Capacity intentionally matches the historical rewrite calculation exactly; evidence-only dedicated buffers do not perturb placement.
  total_memory = sum(nbytes[b] for b in first_appearance) * 2
  offsets:dict[UOp, int] = {}
  peaks:dict[LaneKey, tuple[int, TLSFAllocator]] = defaultdict(lambda: (0, TLSFAllocator(total_memory, block_size=block_size, lv2_cnt=32)))
  for _, is_open, buf in events:
    key = _lane_key(buf, copy_bufs)
    if is_open: offsets[buf] = peaks[key][1].alloc(nbytes[buf])
    else: peaks[key][1].free(offsets[buf])
    peaks[key] = (max(peaks[key][0], offsets[buf] + buf.arg * buf.dtype.itemsize), peaks[key][1])
  arena_sizes = {key: round_up(peak, block_size) for key, (peak, _) in peaks.items()}
  return _MemoryPlan(all_first, all_last, planner_last, copy_bufs, nbytes, offsets, arena_sizes, written, copy_sources)

def _buffer_identity(b:UOp) -> str:
  # BUFFER keys include their UNIQUE source and are stable across manifest requests.
  return f"buffer:{b.key.hex()}"

def _semantic_owner(b:UOp) -> Any:
  from tinygrad.uop.ops import memory_semantic_owner
  if (owner:=memory_semantic_owner(b)) is not None: return owner
  # Read-only compatibility for manifests produced by callers that predate the
  # explicit vocabulary. New annotations never use tag (callify owns it).
  def freeze(value:Any) -> Any:
    if isinstance(value, dict): return tuple(sorted((freeze(k), freeze(v)) for k,v in value.items()))
    if isinstance(value, (list, tuple)): return tuple(freeze(x) for x in value)
    if isinstance(value, set): return tuple(sorted(freeze(x) for x in value))
    return value
  if isinstance(b.tag, dict):
    for key in ("semantic_owner", "semantic_ownership", "owner", "ownership"):
      if key in b.tag: return freeze(b.tag[key])
  if isinstance(b.tag, tuple) and len(b.tag) == 2 and b.tag[0] in ("semantic_owner", "semantic_ownership", "owner", "ownership"):
    return freeze(b.tag[1])
  return "unknown"

def _call_bound_owners(root:UOp) -> dict[UOp, Any]:
  """Resolve semantic ownership through the arguments of this invocation.

  Callified PARAMs are deliberately cache-normalized.  Consequently their
  ownership must never be stored on the PARAM itself: the only authoritative
  binding is the concrete argument on the CALL currently being collected.
  This map is local to one manifest request and is therefore safe when the
  same cached function is invoked with differently owned buffers.
  """
  owners:dict[UOp, Any] = {}
  conflicts:set[UOp] = set()

  def visit(u:UOp, bindings:dict[int, UOp]|None = None) -> None:
    if u.op is Ops.MEMORY_SEMANTIC and len(u.src) == 1:
      for b in _collect_bufs(u.src[0]):
        if b in owners and owners[b] != u.arg: conflicts.add(b)
        else: owners[b] = u.arg
      visit(u.src[0], bindings)
      return
    if u.op is Ops.PARAM and bindings is not None:
      slot = u.arg.slot if hasattr(u.arg, "slot") else u.arg
      arg = bindings.get(slot)
      if arg is None: return
      for b in _collect_bufs(arg):
        owner = _semantic_owner(arg)
        if owner != "unknown":
          if b in owners and owners[b] != owner: conflicts.add(b)
          else: owners[b] = owner
      return
    if u.op in {Ops.CALL, Ops.FUNCTION} and len(u.src) > 0:
      args = {i: x for i, x in enumerate(u.src[1:])}
      function_info = getattr(u.src[0], "arg", None)
      call_info = getattr(u, "arg", None)
      semantic_slots = dict(getattr(function_info, "memory_semantic_slots", ()))
      semantic_slots.update(getattr(call_info, "memory_semantic_slots", ()))
      for slot, owner in semantic_slots.items():
        arg = args.get(slot)
        if arg is None: continue
        for b in _collect_bufs(arg):
          if b in owners and owners[b] != owner: conflicts.add(b)
          else: owners[b] = owner
      visit(u.src[0], args)
      # Arguments can themselves contain nested calls.
      for x in u.src[1:]: visit(x, bindings)
      return
    for x in u.src: visit(x, bindings)

  visit(root)
  for b in conflicts: owners[b] = "unknown"
  return owners

def _arena_identity(key:LaneKey) -> str: return f"arena:{key[0]}:{'copy' if key[1] else 'compute'}"

def _merge_ranges(ranges:list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
  merged:list[tuple[int, int]] = []
  for start, end in sorted(ranges):
    if merged and start <= merged[-1][1]: merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
    else: merged.append((start, end))
  return tuple(merged)

def _memory_plan_manifest(linear:UOp, plan:_MemoryPlan, rewritten_arenas:dict[LaneKey, UOp]|None=None) -> ScheduleMemoryManifest:
  rows:list[ScheduleMemoryBuffer] = []
  arena_meta:dict[str, ScheduleMemoryArena] = {}
  bound_owners = _call_bound_owners(linear)
  explicit_owners = {b:bound_owners.get(b, _semantic_owner(b)) for b in plan.first}
  # COPY's destination is the same logical value on another device. Propagate
  # a unique explicit source role to its written destination; this covers host
  # request staging introduced by lowering without inferring from device/size.
  for dest, sources in plan.copy_sources.items():
    if explicit_owners.get(dest) != "unknown": continue
    source_owners = {explicit_owners.get(source, "unknown") for source in sources}
    source_owners.discard("unknown")
    if len(source_owners) == 1: explicit_owners[dest] = source_owners.pop()
  from tinygrad.uop import MemorySemanticOwner, MemorySemanticClass, PREFILL_SCRATCH, RUNTIME_SCRATCH
  # A replay can read a previous phase's output as its input (prefill sample ->
  # decode feedback).  Only output buffers written by this schedule establish
  # the current execution phase; read-only historical outputs cannot make the
  # phase ambiguous.
  execution_outputs = {owner.semantic_class for b,owner in explicit_owners.items() if b in plan.written and
                       isinstance(owner, MemorySemanticOwner) and
                       owner.semantic_class in {MemorySemanticClass.PREFILL_OUTPUT, MemorySemanticClass.RUNTIME_OUTPUT}}
  execution_inputs = {owner.semantic_class for owner in explicit_owners.values() if isinstance(owner, MemorySemanticOwner) and
                      owner.semantic_class is MemorySemanticClass.RUNTIME_INPUT}
  default_scratch = PREFILL_SCRATCH if execution_outputs == {MemorySemanticClass.PREFILL_OUTPUT} else \
                    RUNTIME_SCRATCH if execution_outputs == {MemorySemanticClass.RUNTIME_OUTPUT} or \
                    (not execution_outputs and execution_inputs) else None
  for b in sorted(plan.first, key=lambda x: (plan.first[x], _buffer_identity(x))):
    lane_key = _lane_key(b, plan.copy_bufs)
    if b in plan.offsets:
      aid, offset, arena_size = _arena_identity(lane_key), plan.offsets[b], plan.arena_sizes[lane_key]
      backing_uop = None if rewritten_arenas is None else rewritten_arenas[lane_key]
    else:
      aid, offset, arena_size = f"dedicated:{_buffer_identity(b)}", 0, plan.nbytes[b]
      backing_uop = b
    arena_meta.setdefault(aid, ScheduleMemoryArena(aid, lane_key[0], lane_key[1], arena_size, backing_uop))
    owner = explicit_owners[b]
    # A written value whose first appearance is after graph entry is
    # structurally produced by this execution.  It is execution-local scratch
    # whether memory planning places it in a shared arena or keeps it dedicated
    # across an opaque call boundary.  Graph-entry buffers remain fail-closed:
    # they may be caller inputs or persistent model/runtime state and require
    # explicit ownership.
    internally_introduced = plan.first[b] > 0
    if owner == "unknown" and default_scratch is not None and (b in plan.written or internally_introduced):
      owner = default_scratch
    rows.append(ScheduleMemoryBuffer(_buffer_identity(b), lane_key[0], lane_key[1], plan.nbytes[b], plan.first[b],
                                     plan.planner_last.get(b, plan.last[b]),
                                     aid, arena_size, offset, (offset, offset + plan.nbytes[b]),
                                     owner))

  indices:list[ScheduleMemoryIndex] = []
  for i in range(len(linear.src)):
    by_arena:dict[str, list[tuple[int, int, Any]]] = defaultdict(list)
    for row in rows:
      if row.first_index <= i <= row.last_index: by_arena[row.arena_identity].append((*row.byte_range, row.semantic_owner))
    arena_ranges:list[tuple[str, tuple[tuple[int, int], ...]]] = []
    physical = 0
    for aid, live in sorted(by_arena.items()):
      # Conflicting known semantic owners may never claim the same live physical byte.
      for n, (start, end, owner) in enumerate(live):
        for ostart, oend, other in live[:n]:
          if max(start, ostart) < min(end, oend) and owner != "unknown" and other != "unknown" and owner != other:
            raise ValueError(f"conflicting semantic ownership of live physical bytes in {aid} at schedule index {i}")
      merged = _merge_ranges([(x[0], x[1]) for x in live])
      arena_ranges.append((aid, merged))
      physical += sum(end-start for start, end in merged)
    indices.append(ScheduleMemoryIndex(i, physical, tuple(arena_ranges)))
  return ScheduleMemoryManifest(tuple(rows), tuple(sorted(arena_meta.values(), key=lambda x:x.identity)), tuple(indices),
                                max((x.physical_byte_union for x in indices), default=0))

def memory_plan_manifest(linear:UOp, held_bufs:set[UOp]|None=None) -> ScheduleMemoryManifest:
  """Return immutable exact physical-memory evidence without rewriting *linear*."""
  return _memory_plan_manifest(linear, _make_plan(linear, set() if held_bufs is None else held_bufs))

def memory_plan_rewrite(linear:UOp, held_bufs:set[UOp]|None=None) -> UOp:
  if held_bufs is None: held_bufs = set()
  collectors = _memory_manifest_collectors.get()
  if NO_MEMORY_PLANNER and not collectors: return linear
  plan = _make_plan(linear, held_bufs)
  # These are the exact arena UOps installed by the rewrite and, when requested, referenced by the manifest.
  arenas = {} if NO_MEMORY_PLANNER else {key: UOp.new_buffer(key[0], sz, dtypes.int8) for key, sz in plan.arena_sizes.items()}
  if collectors:
    # Build evidence from this exact plan before rewriting. Any completeness failure propagates,
    # so requesting collection can never silently dispatch without its manifest.
    manifest = _memory_plan_manifest(linear, plan, arenas if not NO_MEMORY_PLANNER else None)
    for collect in collectors: collect(manifest)
  if NO_MEMORY_PLANNER: return linear
  first_appearance, copy_bufs, nbytes, offsets, arena_sizes = plan.first, plan.copy_bufs, plan.nbytes, plan.offsets, plan.arena_sizes
  if not offsets: return linear

  # build replace_map: each buffer becomes a SLICE into a shared per-device-lane arena
  replace_map:dict[UOp, UOp] = {}
  for buf_uop, offset in offsets.items():
    replace_map[buf_uop] = UOp(Ops.SLICE, buf_uop.dtype, (arenas[_lane_key(buf_uop, copy_bufs)], UOp.const(dtypes.weakint, offset)), buf_uop.arg)

  if DEBUG >= 1 and (omem:=sum(nbytes[b] for b in offsets) / 1e6) != (nmem:=sum(arena_sizes.values()) / 1e6):
    print(f"memory reduced from {omem:.2f} MB -> {nmem:.2f} MB, {len(first_appearance)} -> {len(arenas)} bufs")

  return linear.substitute(replace_map, name="memory plan", walk=True)
