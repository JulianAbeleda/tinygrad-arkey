from typing import TypeVar, Generic, Callable, Any
import contextlib, contextvars, functools, collections, hashlib, json, os
from enum import Enum
from tinygrad.tensor import Tensor
from tinygrad.helpers import flatten, merge_dicts, DEBUG, getenv, JIT, JIT_BATCH_SIZE, dedup, pluralize, VIZ, Metadata
from tinygrad.device import Buffer, Compiled, Device, MultiBuffer
from tinygrad.dtype import DType, dtypes
from tinygrad.uop.ops import UOp, UPat, PatternMatcher, Variable, sym_infer, Ops, buffers, track_rewrites, graph_rewrite
from tinygrad.engine.realize import capturing, Estimates, compile_linear, run_linear, graph_cache, estimate_uop, get_runtime
from tinygrad.engine.realize import unwrap_multi, resolve_params, get_call_arg_uops, get_call_outs_ins
from tinygrad.engine.metadata import PROGRAM_IDENTITY_FIELDS
from tinygrad.schedule.memory import memory_plan_rewrite, _collect_bufs
from tinygrad.nn.state import get_parameters
from tinygrad.schedule.rangeify import mop_cleanup
from dataclasses import dataclass, field, replace

class GraphAdmissionReason(str, Enum):
  ADMITTED = "admitted"
  NO_GRAPH_BACKEND = "no_graph_backend"
  UNSUPPORTED_CALL_OP = "unsupported_call_op"
  MIXED_DEVICE = "mixed_device"
  BACKEND_BUFFER_OFFSET_WIDTH = "backend_buffer_offset_width"
  BACKEND_RESOURCE_LIMIT = "backend_resource_limit"
  EXPLICIT_GRAPH_BARRIER = "explicit_graph_barrier"
  BATCH_SIZE_LIMIT = "batch_size_limit"
  GRAPH_CONSTRUCTOR_FAILURE = "graph_constructor_failure"
  IGNORED_SLICE_NODE = "ignored_slice_node"
  SINGLETON_GRAPH_ELIDED = "singleton_graph_elided"
  UNKNOWN = "unknown"

class GraphAdmissionDecision(str, Enum):
  ADMITTED = "admitted"
  REJECTED = "rejected"
  BATCH_BOUNDARY = "batch_boundary"
  IGNORED = "ignored"

@dataclass(frozen=True)
class GraphAdmissionResource:
  buffer_arg_index: int
  base_allocation_id: int
  byte_offset: int
  byte_span: int | None = None

@dataclass(frozen=True)
class GraphAdmission:
  supported: bool
  reason: GraphAdmissionReason
  capability: str | None = None
  limit: int | None = None
  observed: int | None = None
  resources: tuple[GraphAdmissionResource, ...] = ()
  def __bool__(self) -> bool: return self.supported

@dataclass(frozen=True)
class GraphAdmissionObservation:
  call_index: int
  decision: GraphAdmissionDecision
  admission: GraphAdmission
  batch_boundary_reason: GraphAdmissionReason | None = None
  program_hash: str | None = None
  program_name: str | None = None
  metadata: tuple[Metadata, ...] = ()
  source_sha256: str | None = None
  binary_sha256: str | None = None
  assignment: str = "unassigned"
  assignment_reason: GraphAdmissionReason | None = None
  batch_index: int | None = None
  batch_member_index: int | None = None
  batch_size: int | None = None
  direct_call_index: int | None = None

@dataclass(frozen=True)
class GraphConstructorFailureObservation:
  error_type: str
  message: str

GraphAdmissionEvent = GraphAdmissionObservation | GraphConstructorFailureObservation
GraphAdmissionObserver = Callable[[GraphAdmissionEvent], None]

@dataclass
class GraphAdmissionCensus:
  """In-memory deterministic census. No files or JSON are produced unless explicitly requested."""
  records: list[GraphAdmissionObservation] = field(default_factory=list)
  constructor_failures: list[GraphConstructorFailureObservation] = field(default_factory=list)
  # Runtime-only capture bindings for exact research measurement. They are
  # deliberately absent from to_dict/deterministic_json and never become
  # serialized graph-admission policy.
  calls: dict[int, UOp] = field(default_factory=dict, repr=False, compare=False)
  execution_linear: UOp|None = field(default=None, repr=False, compare=False)
  execution_inputs: tuple[UOp, ...] = field(default=(), repr=False, compare=False)
  execution_var_vals: dict[str, int] = field(default_factory=dict, repr=False, compare=False)

  def __call__(self, event:GraphAdmissionEvent) -> None:
    (self.records if isinstance(event, GraphAdmissionObservation) else self.constructor_failures).append(event)

  def bind_call(self, call_index:int, call:UOp) -> None:
    if call_index in self.calls and self.calls[call_index] is not call: raise ValueError("graph-admission call binding changed")
    self.calls[call_index] = call

  def bind_execution(self, linear:UOp, input_uops:tuple[UOp, ...], var_vals:dict[str, int]) -> None:
    self.execution_linear, self.execution_inputs, self.execution_var_vals = linear, tuple(input_uops), dict(var_vals)

  def to_dict(self) -> dict[str, Any]:
    records = sorted(self.records, key=lambda record: record.call_index)
    if len({record.call_index for record in records}) != len(records): raise ValueError("duplicate graph-admission call index")
    if [record.call_index for record in records] != list(range(len(records))): raise ValueError("graph-admission call indexes are not contiguous")
    batches: dict[int, int] = {}
    batch_members: dict[int, list[int]] = collections.defaultdict(list)
    direct_indexes = []
    for record in records:
      if record.assignment == "graph":
        if record.batch_index is None or record.batch_member_index is None or record.batch_size is None:
          raise ValueError("graph assignment missing batch identity")
        if record.batch_index in batches and batches[record.batch_index] != record.batch_size: raise ValueError("inconsistent graph batch size")
        batches[record.batch_index] = record.batch_size
        batch_members[record.batch_index].append(record.batch_member_index)
      elif record.assignment == "direct":
        if record.direct_call_index is None: raise ValueError("direct assignment missing index")
        direct_indexes.append(record.direct_call_index)
      elif record.assignment != "ignored": raise ValueError("unreconciled graph-admission record")
    if sorted(batches) != list(range(len(batches))): raise ValueError("graph batch indexes are not contiguous")
    if any(sorted(batch_members[index]) != list(range(size)) for index, size in batches.items()):
      raise ValueError("graph batch members do not reconcile with batch size")
    if sorted(direct_indexes) != list(range(len(direct_indexes))): raise ValueError("direct-call indexes are not contiguous")
    graph_members = sum(record.assignment == "graph" for record in records)
    direct_calls = sum(record.assignment == "direct" for record in records)
    ignored = sum(record.assignment == "ignored" for record in records)
    if graph_members + direct_calls + ignored != len(records): raise ValueError("graph-admission census does not reconcile")
    reasons = collections.Counter((record.assignment_reason or record.admission.reason).value for record in records)
    admission_reasons = collections.Counter(record.admission.reason.value for record in records)
    boundaries = collections.Counter(record.batch_boundary_reason.value for record in records if record.batch_boundary_reason is not None)
    serialized_records = [_graph_admission_record(record) for record in records]
    semantic_calls = sum(record["metadata_status"] == "semantic" for record in serialized_records)
    workload_roles = collections.Counter(role for record in serialized_records for role in record["workload_roles"])
    return {"schema":"tinygrad.graph_admission_census.v1", "counts":{"logical_calls":len(records), "graph_members":graph_members,
      "direct_calls":direct_calls, "ignored_slice_nodes":ignored, "graph_batches":len(batches),
      "constructor_failures":len(self.constructor_failures), "semantic_calls":semantic_calls,
      "generic_calls":len(records)-semantic_calls}, "reason_histogram":dict(sorted(reasons.items())),
      "admission_reason_histogram":dict(sorted(admission_reasons.items())),
      "batch_boundary_histogram":dict(sorted(boundaries.items())),
      "workload_role_histogram":dict(sorted(workload_roles.items())),
      "batches":[{"batch_index":index, "size":batches[index]} for index in sorted(batches)],
      "records":serialized_records,
      "constructor_failures":[{"error_type":failure.error_type, "message":failure.message} for failure in self.constructor_failures]}

  def deterministic_json(self) -> str: return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"

_GRAPH_ADMISSION_OBSERVER: contextvars.ContextVar[GraphAdmissionObserver|None] = contextvars.ContextVar(
  "tinygrad_graph_admission_observer", default=None)

@contextlib.contextmanager
def observe_graph_admissions(census:GraphAdmissionCensus|None=None):
  """Install one context-local admission collector for JIT lowering and graph construction."""
  collector = census if census is not None else GraphAdmissionCensus()
  token = _GRAPH_ADMISSION_OBSERVER.set(collector)
  try: yield collector
  finally: _GRAPH_ADMISSION_OBSERVER.reset(token)

def _graph_admission_record(record:GraphAdmissionObservation) -> dict[str, Any]:
  admission = record.admission
  semantic = []
  for item in record.metadata:
    if all(hasattr(item, field) for field in PROGRAM_IDENTITY_FIELDS):
      semantic.append({field:getattr(item, field) for field in PROGRAM_IDENTITY_FIELDS})
  return {"call_index":record.call_index, "program_hash":record.program_hash, "program_name":record.program_name,
    "source_sha256":record.source_sha256, "binary_sha256":record.binary_sha256,
    "metadata":[{"name":item.name, "caller":item.caller, "backward":item.backward} for item in record.metadata],
    "semantic_identities":semantic, "metadata_status":"semantic" if semantic else ("generic" if record.metadata else "unavailable"),
    "metadata_unavailable":not bool(semantic), "workload_roles":list(dict.fromkeys(item["role"] for item in semantic)) if semantic else ["generic"],
    "decision":record.decision.value, "supported":admission.supported,
    "reason":(record.assignment_reason or admission.reason).value,
    "admission_reason":admission.reason.value, "batch_boundary_reason":
    record.batch_boundary_reason.value if record.batch_boundary_reason is not None else None, "assignment":record.assignment,
    "batch_index":record.batch_index, "batch_member_index":record.batch_member_index, "batch_size":record.batch_size,
    "direct_call_index":record.direct_call_index, "capability":admission.capability, "limit":admission.limit,
    "observed":admission.observed, "resources":[{"buffer_arg_index":resource.buffer_arg_index,
      "base_allocation_id":resource.base_allocation_id, "byte_offset":resource.byte_offset, "byte_span":resource.byte_span}
      for resource in admission.resources]}

def prune_linear(linear:UOp, needed:set[UOp]) -> tuple[UOp, UOp]:
  kept, onetime = [], []
  for si in linear.src:
    si_bufs = {b for src in si.src[1:] for b in _collect_bufs(src)}
    if not si_bufs.isdisjoint(needed):
      kept.append(si)
      needed |= si_bufs
    else: onetime.append(si)
  return linear.replace(src=tuple(kept)), linear.replace(src=tuple(onetime))

def create_graph_call(batch:list[UOp]) -> UOp:
  # all external inputs are PARAMs
  input_list = dedup(u for si in batch for b in si.src[1:] for u in b.toposort() if u.op is Ops.PARAM)
  cf = UOp(Ops.CUSTOM_FUNCTION, dtypes.void, src=(UOp(Ops.LINEAR, src=tuple(batch)),), arg="graph")
  return cf.call(*input_list, metadata=tuple(m for si in batch for m in si.arg.metadata))

def _jit_no_graph_kernel_prefixes() -> tuple[str, ...]:
  # Read fresh: helpers.getenv is cached, while route-local graph barriers are installed after process start.
  prefixes = os.environ.get("JIT_NO_GRAPH_KERNEL_PREFIXES", "")
  return tuple(p for p in str(prefixes).split(",") if p)

def _should_skip_graph_for_prefix(si:UOp) -> bool:
  if si.src[0].op is not Ops.PROGRAM: return False
  prefixes = _jit_no_graph_kernel_prefixes()
  return bool(prefixes and si.src[0].arg.name.startswith(prefixes))

def _typed_graph_admission(graph_t, batch_devs:list[Compiled], new_call:UOp) -> GraphAdmission:
  # Legacy graph backends may override only supports_uop during migration. Their
  # bool remains authoritative until that backend adds a typed implementation.
  if "supports_uop" in graph_t.__dict__ and "admission" not in graph_t.__dict__:
    supported = bool(graph_t.supports_uop(batch_devs, new_call))
    return GraphAdmission(supported, GraphAdmissionReason.ADMITTED if supported else GraphAdmissionReason.UNKNOWN)
  result = graph_t.admission(batch_devs, new_call)
  return result if isinstance(result, GraphAdmission) else GraphAdmission(bool(result), GraphAdmissionReason.UNKNOWN)

def _admission_observation(call_index:int, call, decision:GraphAdmissionDecision, admission:GraphAdmission,
                           boundary_reason:GraphAdmissionReason|None=None) -> GraphAdmissionObservation:
  program = call.src[0]
  key = getattr(program, "key", None)
  program_hash = key.hex() if isinstance(key, bytes) else None
  name = getattr(getattr(program, "arg", None), "name", None)
  program_src = getattr(program, "src", ())
  source = next((item.arg for item in program_src if item.op is Ops.SOURCE and isinstance(item.arg, str)), None)
  binary = next((item.arg for item in program_src if item.op is Ops.BINARY and isinstance(item.arg, bytes)), None)
  metadata = getattr(getattr(call, "arg", None), "metadata", ())
  # Concrete program arguments are the truthful authority for packed QK route
  # attribution. This registry is side data and is queried only after compile.
  from tinygrad.engine.metadata import resolve_call_metadata
  registry_metadata = resolve_call_metadata(call)
  metadata = tuple(dedup((*metadata, *registry_metadata))) if isinstance(metadata, tuple) else registry_metadata
  return GraphAdmissionObservation(call_index, decision, admission, boundary_reason, program_hash,
    name if isinstance(name, str) else None,
    tuple(item for item in metadata if isinstance(item, Metadata)) if isinstance(metadata, tuple) else (),
    hashlib.sha256(source.encode()).hexdigest() if source is not None else None,
    hashlib.sha256(binary).hexdigest() if binary is not None else None)

def graph_split_rewrite(linear:UOp, max_batch_size:int=0, observer:GraphAdmissionObserver|None=None) -> UOp:
  new_src: list[UOp] = []
  current_batch: list[UOp] = []
  current_batch_indexes: list[int] = []
  current_batch_devs: list[Compiled] = []
  pending: dict[int, GraphAdmissionObservation] = {}
  batch_index = direct_call_index = 0

  def flush_batch():
    nonlocal current_batch, current_batch_indexes, current_batch_devs, max_batch_size, new_src, batch_index, direct_call_index
    if len(current_batch) <= 1 and not getenv("GRAPH_ONE_KERNEL"):
      new_src.extend(current_batch)
      if observer is not None:
        for index in current_batch_indexes:
          pending[index] = replace(pending[index], assignment="direct", assignment_reason=GraphAdmissionReason.SINGLETON_GRAPH_ELIDED,
                                   direct_call_index=direct_call_index)
          direct_call_index += 1
    else:
      new_src.append(create_graph_call(current_batch))
      if observer is not None:
        for member_index, index in enumerate(current_batch_indexes):
          pending[index] = replace(pending[index], assignment="graph", assignment_reason=GraphAdmissionReason.ADMITTED,
                                   batch_index=batch_index, batch_member_index=member_index, batch_size=len(current_batch))
        batch_index += 1
      max_batch_size *= 2
      if DEBUG >= 2: print(f"JIT GRAPHing batch with {len(current_batch)} kernels")
    current_batch, current_batch_indexes, current_batch_devs = [], [], []

  def append_direct(call_index:int, call, observation:GraphAdmissionObservation):
    nonlocal direct_call_index
    new_src.append(call)
    if observer is not None:
      pending[call_index] = replace(observation, assignment="direct", assignment_reason=observation.admission.reason,
                                    direct_call_index=direct_call_index)
      direct_call_index += 1

  for call_index, si in enumerate(linear.src):
    if observer is not None and hasattr(observer, "bind_call"): observer.bind_call(call_index, si)
    if si.src[0].op is Ops.SLICE:
      if observer is not None:
        pending[call_index] = replace(_admission_observation(call_index, si, GraphAdmissionDecision.IGNORED,
          GraphAdmission(False, GraphAdmissionReason.IGNORED_SLICE_NODE)), assignment="ignored",
          assignment_reason=GraphAdmissionReason.IGNORED_SLICE_NODE)
      continue
    if _should_skip_graph_for_prefix(si):
      flush_batch()
      append_direct(call_index, si, _admission_observation(call_index, si, GraphAdmissionDecision.BATCH_BOUNDARY,
                    GraphAdmission(False, GraphAdmissionReason.EXPLICIT_GRAPH_BARRIER)))
      current_batch_devs = []
      continue

    devs = dedup([Device[x] for b in si.src[1:] if b.op is not Ops.BIND for x in (b.device if isinstance(b.device, tuple) else (b.device,))])
    graph_t = graph_class(devs[0]) if devs[0].graph is not None else None

    if observer is None:
      can_graph = graph_t is not None and graph_t.supports_uop(devs, si)
      can_extend = can_graph and graph_t is not None and (not current_batch_devs or graph_t.supports_uop(current_batch_devs, si)) \
        and (max_batch_size == 0 or len(current_batch) < max_batch_size)
    else:
      admission = _typed_graph_admission(graph_t, devs, si) if graph_t is not None else \
        GraphAdmission(False, GraphAdmissionReason.NO_GRAPH_BACKEND)
      can_graph = bool(admission)
      extension = _typed_graph_admission(graph_t, current_batch_devs, si) if can_graph and current_batch_devs else admission
      batch_limited = can_graph and max_batch_size != 0 and len(current_batch) >= max_batch_size
      can_extend = can_graph and bool(extension) and not batch_limited
      boundary_reason = (GraphAdmissionReason.BATCH_SIZE_LIMIT if batch_limited else
                         extension.reason if can_graph and current_batch and not extension else None)
      pending[call_index] = _admission_observation(call_index, si,
        GraphAdmissionDecision.ADMITTED if can_graph else GraphAdmissionDecision.REJECTED, admission, boundary_reason)
    if not can_extend and current_batch: flush_batch()

    # append this si and update devs
    if can_graph:
      current_batch.append(si)
      if observer is not None: current_batch_indexes.append(call_index)
      current_batch_devs = dedup(current_batch_devs + devs)
    else:
      if observer is None: new_src.append(si)
      else: append_direct(call_index, si, pending[call_index])
      current_batch_devs = []
  if current_batch: flush_batch()
  if observer is not None:
    for call_index in sorted(pending): observer(pending[call_index])
  return linear.replace(src=tuple(new_src))

def _copy_input(u:UOp) -> UOp:
  run_linear(UOp(Ops.LINEAR, src=(u.copy_to_device(u.device).call(new:=UOp.new_buffer(u.device, u.arg, u.dtype), u, metadata=()),)))
  return new

@track_rewrites(lambda linear,held_bufs,input_uops,ret=(): f"JIT {pluralize('call', len(linear.src))}")
def jit_lower(linear:UOp, held_bufs:set[UOp], input_uops:list[UOp]) -> UOp:
  if VIZ: graph_rewrite(linear, PatternMatcher([]), name="View captured linear")

  # parametrize input buffers: map each input buffer UOp to a PARAM with the correct slot index
  linear = linear.substitute({u: UOp.param(i, u.dtype, u.shape, u.device) for i,u in enumerate(input_uops)}, walk=True)
  linear = memory_plan_rewrite(linear, held_bufs)
  linear = compile_linear(linear)
  if JIT < 2: linear = graph_split_rewrite(linear, max_batch_size=JIT_BATCH_SIZE.value, observer=_GRAPH_ADMISSION_OBSERVER.get())
  if VIZ: graph_rewrite(linear, PatternMatcher([]), name="View graphed linear")
  return linear

class GraphException(Exception): pass
class JitError(Exception): pass

# The input contract of a captured JIT has two distinct parts.  Concrete
# buffers and variable values are invocation-owned, while the view graph used
# to describe their shape/device contract is normally stable across decode
# tokens.  Keep a deliberately small *identity* cache for the latter.  This
# is not structural memoization: a new UOp, even one which happens to compare
# equal, takes the conservative full path.
_JIT_INPUT_DESCRIPTOR_CACHE: collections.OrderedDict[tuple[int, ...], tuple[tuple[UOp, ...], tuple]] = collections.OrderedDict()
_JIT_INPUT_DESCRIPTOR_CACHE_LIMIT = 64

def _jit_input_descriptors(input_uops:list[UOp]) -> tuple:
  if not getenv("JIT_INPUT_DESCRIPTOR_CACHE", 1):
    return tuple((*(graph_rewrite(u.substitute({u.base:UOp(Ops.NOOP)}, extra_pm=mop_cleanup), pm_jit_input_metadata).unbind_all()),
                  u.dtype, u.device) for u in input_uops)
  key = tuple(id(u) for u in input_uops)
  cached = _JIT_INPUT_DESCRIPTOR_CACHE.get(key)
  if cached is not None and len(cached[0]) == len(input_uops) and all(a is b for a,b in zip(cached[0], input_uops)):
    _JIT_INPUT_DESCRIPTOR_CACHE.move_to_end(key)
    return cached[1]
  descriptors = tuple((*(graph_rewrite(u.substitute({u.base:UOp(Ops.NOOP)}, extra_pm=mop_cleanup), pm_jit_input_metadata).unbind_all()),
                       u.dtype, u.device) for u in input_uops)
  _JIT_INPUT_DESCRIPTOR_CACHE[key] = (tuple(input_uops), descriptors)
  _JIT_INPUT_DESCRIPTOR_CACHE.move_to_end(key)
  while len(_JIT_INPUT_DESCRIPTOR_CACHE) > _JIT_INPUT_DESCRIPTOR_CACHE_LIMIT: _JIT_INPUT_DESCRIPTOR_CACHE.popitem(last=False)
  return descriptors

pm_jit_input_metadata = PatternMatcher([
  (UPat(Ops.MEMORY_SEMANTIC, src=(UPat(),), name="m"), lambda m: m.src[0]),
])

def _check_no_non_tensor_return(ret):
  if ret is None or isinstance(ret, Tensor): return
  if isinstance(ret, (tuple, list, dict)):
    for item in (ret.values() if isinstance(ret, dict) else ret): _check_no_non_tensor_return(item)
    return
  raise JitError(f"JIT return contains non-Tensor value of type {type(ret).__name__}")

def graph_class(dev): return dev.graph.func if isinstance(dev.graph, functools.partial) else dev.graph

class DepsTracker:
  def __init__(self):
    # tracks (offset, end, dep) ranges per base buffer id to handle suballocated buffers correctly.
    self.w_dependency_map: dict[int, list[tuple[int, int, Any]]] = collections.defaultdict(list)
    self.r_dependency_map: dict[int, list[tuple[int, int, Any]]] = collections.defaultdict(list)

  @staticmethod
  def _buf_key(buf:Buffer) -> int: return id(buf.base)

  def access_resources(self, bufs:list[Buffer], write:list[int], new_dependency:Any):
    wait_nodes = []
    for i,buf in enumerate(bufs):
      key, s, e = self._buf_key(buf), buf.offset, buf.offset + buf.nbytes
      wait_nodes += [dep for st,en,dep in self.w_dependency_map[key] if st < e and s < en]
      if i in write: wait_nodes += [dep for st,en,dep in self.r_dependency_map[key] if st < e and s < en]
    for i,buf in enumerate(bufs):
      key, s, e = self._buf_key(buf), buf.offset, buf.offset + buf.nbytes
      if i in write:
        for dmap in [self.w_dependency_map, self.r_dependency_map]:
          kept = []
          for st,en,dep in dmap[key]:
            if st < min(s, en): kept.append((st, min(s, en), dep))
            if max(e, st) < en: kept.append((max(e, st), en, dep))
          dmap[key] = kept
        self.w_dependency_map[key].append((s, e, new_dependency))
      else: self.r_dependency_map[key].append((s, e, new_dependency))
    return list({id(x):x for x in wait_nodes}.values())

class GraphRunner:
  def __init__(self, linear:UOp, input_uops:tuple[UOp, ...]=()):
    self.linear = linear.src[0]
    self.calls: list[tuple[int, UOp, list[Buffer], dict[str, int]]] = []
    self.call_metadata: list[tuple[Metadata, ...]] = []
    self.runtimes: list[Any|None] = []
    self.uop_replace: list[list[tuple[int, int]]] = []
    for call in self.linear.src:
      replace = [(p, b.arg.slot) for p, b in enumerate(get_call_arg_uops(call)) if b.op is Ops.PARAM]
      for dev_idx, (bufs, device_vars) in enumerate(unwrap_multi(call, resolve_params(call, input_uops))):
        self.calls.append((dev_idx, call.src[0], [b.ensure_allocated() for b in bufs], device_vars))
        self.call_metadata.append(call.arg.metadata)
        self.runtimes.append(get_runtime(bufs[0].device, call.src[0]) if call.src[0].op is Ops.PROGRAM else None)
        self.uop_replace.append(replace)

    self.var_vals_replace:dict[int, list[tuple[int, int]]] = {}
    self.launch_dims_replace:dict[int, tuple[int|None, int|None]] = {}
    self.launch_dims_base:dict[int, tuple[tuple[int|float, ...], tuple[int, ...]]] = {}

    def is_sym_dim(dim) -> bool: return not all(isinstance(d, (int, float)) for d in dim)

    crs = [(j, self.calls[j][1].arg, self.calls[j][3]) for j in range(len(self.calls)) if self.calls[j][1].op is Ops.PROGRAM]
    self.vars = sorted({v.expr for _,p,dv in crs for v in p.vars if v.expr not in dv | p.runtimevars})
    self.symbolic_dims = dedup(tuple(d) for _,p,_ in crs for d in (p.local_size, p.global_size) if d and is_sym_dim(d))

    def find_symbolic_dim(dim): return self.symbolic_dims.index(tuple(dim)) if dim is not None and tuple(dim) in self.symbolic_dims else None

    for j,p,dv in crs:
      if (replace:=[(i, self.vars.index(v.expr)) for i, v in enumerate(p.vars) if v.expr not in dv | p.runtimevars]):
        self.var_vals_replace[j] = replace
      global_dim_idx, local_dim_idx = find_symbolic_dim(p.global_size), find_symbolic_dim(p.local_size)
      if global_dim_idx is not None or local_dim_idx is not None:
        self.launch_dims_replace[j] = (global_dim_idx, local_dim_idx)
        assert p.local_size is not None
        self.launch_dims_base[j] = (tuple(p.global_size), tuple(p.local_size))

    estimates = sum((estimate_uop(call) for call in self.linear.src), Estimates())

    # used in MultiGraphRunner
    self.deps = DepsTracker()

    self.device, self.estimates = self.calls[0][2][0].device.split(":")[0], estimates.simplify()

  def __call__(self, input_uops:tuple[UOp, ...], var_vals:dict[str, int], wait=False) -> float|None: raise NotImplementedError("override this")

  def updated_vars(self, var_vals: dict[str, int]):
    vals = [var_vals[v] for v in self.vars]
    for j, vidxs in self.var_vals_replace.items():
      for i, v in vidxs: yield j, i, vals[v]

  def updated_launch_dims(self, var_vals: dict[str, int]):
    dims = [tuple(sym_infer(s, var_vals) for s in dim) for dim in self.symbolic_dims]
    for j, (gl, lc) in self.launch_dims_replace.items():
      yield j, (dims[gl] if gl is not None else self.launch_dims_base[j][0]), (dims[lc] if lc is not None else self.launch_dims_base[j][1])

  def _access_resources(self, bufs:list[Buffer], write:list[int], new_dependency:Any):
    return self.deps.access_resources(bufs, write, new_dependency)

  @staticmethod
  def _all_devs(batch_devs:list[Compiled], new_call:UOp) -> list[Compiled]:
    return dedup(batch_devs + [Device[x] for b in get_call_arg_uops(new_call)
                 for x in (b.device if isinstance(b.device, tuple) else (b.device,))])

  @staticmethod
  def admission(batch_devs:list[Compiled], new_call:UOp) -> GraphAdmission:
    if new_call.src[0].op is not Ops.PROGRAM: return GraphAdmission(False, GraphAdmissionReason.UNSUPPORTED_CALL_OP)
    if len(GraphRunner._all_devs(batch_devs, new_call)) != 1: return GraphAdmission(False, GraphAdmissionReason.MIXED_DEVICE)
    return GraphAdmission(True, GraphAdmissionReason.ADMITTED)

  @classmethod
  def supports_uop(cls, batch_devs:list[Compiled], new_call:UOp) -> bool: return bool(cls.admission(batch_devs, new_call))

# a marker for your graph supporting multiple devices of the same type
class MultiGraphRunner(GraphRunner):
  @staticmethod
  def admission(batch_devs:list[Compiled], new_call:UOp) -> GraphAdmission:
    # Devices must be the same type
    if new_call.src[0].op not in (Ops.PROGRAM, Ops.COPY): return GraphAdmission(False, GraphAdmissionReason.UNSUPPORTED_CALL_OP)
    if len(dedup([type(d) for d in GraphRunner._all_devs(batch_devs, new_call)])) != 1:
      return GraphAdmission(False, GraphAdmissionReason.MIXED_DEVICE)
    return GraphAdmission(True, GraphAdmissionReason.ADMITTED)

ReturnType = TypeVar('ReturnType')
@dataclass
class CapturedJit(Generic[ReturnType]):
  ret: Any  # includes the Tensors or any other returned object
  linear: UOp
  expected_names: list[int|str]
  expected_input_info: list[tuple[UOp, tuple[Variable, ...], DType, str]]  # (view, variables, dtype, device) per input

  # Kept out of the constructor and pickle contract: shadows are concrete
  # runtime allocations, valid only for this captured linear and input slot.
  _written_input_shadows: dict[int, UOp] = field(default_factory=dict, init=False, repr=False, compare=False)

  def __reduce__(self): return self.__class__, (self.ret, self.linear, self.expected_names, self.expected_input_info)

  @functools.cached_property
  def _written_uops(self) -> set[UOp]:
    out: set[UOp] = set()
    for call in self.linear.toposort():
      if call.op is not Ops.CALL: continue
      arg_uops = get_call_arg_uops(call)
      outs, ins = get_call_outs_ins(call)
      out |= {arg_uops[k] for k in set(outs) - set(ins) if arg_uops[k].op in (Ops.BUFFER, Ops.SLICE)}
    return out

  def __call__(self, input_uops:list[UOp], var_vals:dict[str, int]) -> ReturnType:
    if not getenv("JIT_REUSE_WRITTEN_INPUT_SHADOWS", 1):
      concrete = tuple(_copy_input(u) if u in self._written_uops else u for u in input_uops)
    else:
      concrete = []
      for index, u in enumerate(input_uops):
        if u not in self._written_uops:
          concrete.append(u)
          continue
        shadow = self._written_input_shadows.get(index)
        # A shadow is an alias firewall, not a conversion route.  If the
        # captured slot's concrete contract ever changes, fail closed instead
        # of reusing storage with an ambiguous layout or device binding.
        if shadow is None:
          shadow = UOp.new_buffer(u.device, u.arg, u.dtype)
          self._written_input_shadows[index] = shadow
        elif shadow.device != u.device or shadow.dtype != u.dtype or shadow.arg != u.arg:
          raise JitError(f"written JIT input contract changed at slot {index}")
        run_linear(UOp(Ops.LINEAR, src=(u.copy_to_device(u.device).call(shadow, u, metadata=()),)))
        concrete.append(shadow)
      concrete = tuple(concrete)
    if DEBUG >= 1 and len(self.linear.src) >= 10: print(f"jit execs {len(self.linear.src)} calls")
    if (observer:=_GRAPH_ADMISSION_OBSERVER.get()) is not None and hasattr(observer, "bind_execution"):
      observer.bind_execution(self.linear, concrete, var_vals)
    try: run_linear(self.linear, var_vals, input_uops=concrete, jit=True)
    except GraphException as exc:
      if (observer:=_GRAPH_ADMISSION_OBSERVER.get()) is not None:
        try: observer(GraphConstructorFailureObservation(type(exc).__name__, str(exc)))
        finally: raise
      raise
    return self.ret

  def free_intermediates(self):
    # drop graph runners
    for call in self.linear.src:
      if call.src[0].op is Ops.CUSTOM_FUNCTION and call.src[0].arg == "graph": graph_cache.pop(call.src[0], None)
    for u in self._written_uops:
      if (buf:=buffers.get(u)) is None: continue
      for b in (buf.bufs if isinstance(buf, MultiBuffer) else (buf,)):
        if b.is_initialized(): b.deallocate()
        if (base:=b._base) is not None and base.allocated_views == 0 and base.is_allocated(): base.deallocate()

def _prepare_jit_inputs(args, kwargs):
  input_tensors: list[tuple[int|str, Tensor]] = [(name,t) for name,t in list(enumerate(args))+sorted(kwargs.items()) if t.__class__ is Tensor]
  names, tensors = [name for name,_ in input_tensors], [t for _,t in input_tensors]
  # extract tensors from containers (shallow, not recursive to avoid grabbing model weights)
  for x in args + tuple(kwargs.values()):
    it = x if isinstance(x, (tuple,list)) else x.values() if isinstance(x, dict) else []
    tensors += [t for t in it if t.__class__ is Tensor and not any(t is y for y in tensors)]
  def get_input_uops() -> list[UOp]: return flatten([t.uop.src if t.uop.op is Ops.MULTI else [t.uop] for t in tensors])
  # TODO: drop the CONST branch once all CONST are deviceless
  if any(u.device is None or u.base.op is Ops.CONST for u in get_input_uops()): raise JitError("JIT inputs must be real buffers; use .clone()")
  if len(unrealized_tensors := [x for x in tensors if not x.uop.is_realized]): Tensor.realize(*unrealized_tensors)
  input_uops = get_input_uops()
  # collect buffer UOps (including MultiBuffer)
  input_buf_uops: list[UOp] = [u.base for u in input_uops if u.base.realized is not None]
  if len(set(input_buf_uops)) != len(input_buf_uops): raise JitError("duplicate inputs to JIT")
  # Semantic ownership describes the concrete allocation at each invocation, not
  # the shape/device contract used to reuse compiled JIT programs. Keep the real
  # input UOps above for call binding, but erase the tensor-only carrier from the
  # structural signature just like the substituted buffer identity.
  inputs = _jit_input_descriptors(input_uops)
  _var_vals = merge_dicts([x[1] for x in inputs] + [dict(v.unbind() for v in (args + tuple(kwargs.values())) if isinstance(v, UOp))])
  var_vals = {k.expr:v for k,v in _var_vals.items()}
  expected_input_info = [(x[0], tuple(sorted(x[1].keys(), key=lambda v: v.expr)), x[2], x[3]) for x in inputs]
  return input_buf_uops, var_vals, names, expected_input_info

class TinyJit(Generic[ReturnType]):
  def __init__(self, fxn:Callable[..., ReturnType]|None, captured:CapturedJit|None=None, prune=False):
    assert fxn or captured, "need either a function or a CapturedJit"
    self.fxn = fxn
    self.captured: CapturedJit|None = captured
    self.cnt: int = 2 if self.fxn is None else 0
    self.prune = prune

  def add_linear(self, linear:UOp, var_vals:dict[str, int]): self._linears.append(linear)

  def reset(self):
    assert self.fxn is not None, "can't reset without function"
    self.cnt = 0
    self.captured = None

  def __reduce__(self):
    assert self.captured is not None, "can't pickle an uncaptured JIT"
    return self.__class__, (None, self.captured)

  def __get__(self, obj, objtype): return functools.partial(self.__call__, obj) # add support for instance methods

  def __call__(self, *args, **kwargs) -> ReturnType:
    input_buf_uops, var_vals, names, expected_input_info = _prepare_jit_inputs(args, kwargs)
    if not JIT or self.cnt == 0:
      # jit ignore
      assert self.fxn is not None
      ret = self.fxn(*args, **kwargs)
      if len(params:=get_parameters(ret)): Tensor.realize(*params)
    elif self.cnt == 1:
      # jit capture
      assert self.fxn is not None
      if capturing: raise RuntimeError(f"having TinyJit inside another TinyJit is not supported {len(capturing)=} {capturing=}")
      self._linears: list[UOp] = []
      capturing.append(self)
      try:
        ret = self.fxn(*args, **kwargs)
        if len(params:=get_parameters(ret)): Tensor.realize(*params)
      finally: capturing.clear()
      if not len(self._linears): raise JitError("didn't JIT anything!")
      _check_no_non_tensor_return(ret)
      if DEBUG >= 1: print(f"JIT captured {len(self._linears)} linears with {len(input_buf_uops)} inputs")

      # combine all captured linears into one, memory plan, and graph split
      big_linear = UOp(Ops.LINEAR, src=tuple(flatten([l.src for l in self._linears])))
      del self._linears

      if self.prune:
        big_linear, onetime_linear = prune_linear(big_linear, set(input_buf_uops))
        if DEBUG >= 1: print(f"pruned from {len(big_linear.src) + len(onetime_linear.src)} -> {len(big_linear.src)} kernels")
        run_linear(onetime_linear, var_vals)

      held_bufs = set(buffers) | {t.uop.buf_uop for t in get_parameters(ret) if t.uop.buf_uop.op is Ops.BUFFER}
      linear = jit_lower(big_linear, held_bufs, input_buf_uops)
      self.captured = CapturedJit(ret, linear, names, expected_input_info)
      ret = self.captured(input_buf_uops, var_vals)
    elif self.cnt >= 2:
      # jit exec
      assert self.captured is not None
      if self.captured.expected_names != names: raise JitError(f"args mismatch in JIT: {self.captured.expected_names=} != {names}")
      if self.captured.expected_input_info != expected_input_info:
        raise JitError(f"args mismatch in JIT: {self.captured.expected_input_info=} != {expected_input_info=}")
      ret = self.captured(input_buf_uops, var_vals)

    self.cnt += 1
    return ret
