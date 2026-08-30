"""Default-off compiler-owned Qwen3-8B pp512 Q6_K V/down binding.

The route consumes canonical Q6_K halfwords and a scheduler-owned compact-Q8
record.  It admits only the model's Q6 population: 18 attention V and 18 FFN
down projections.  Q4 V/down tensors are deliberately outside the contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from types import MappingProxyType
from typing import Mapping, Any
import time

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.packed_weight import (PackedWeightTransform, Q6KInt8FragmentProvider,
  Q8ActivationRecordTransform, Q8Int8FragmentProvider, Q6KQ8SubgroupAccumulatorContract)
from tinygrad.codegen.opt.postrange import warmstart_key
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import Ops, UOp
from extra.llm_research.kernel_vocabulary import KernelLDSWindow, KernelTileGeometry
from extra.llm_research.prefill.nv_native_program_uop import native_nv_program
from extra.llm_research.prefill.nv_q8_compact_producer_gate import SRC_FP16
from extra.llm_research.prefill.nv_compiler_q6_schedule_config import CompilerQ6ScheduleConfig

M, TILE_K = 512, 64
ROLE_SHAPES = MappingProxyType({"attn_v": (1024, 4096), "ffn_down": (4096, 12288)})
ROLE_COUNTS = MappingProxyType({"attn_v": 18, "ffn_down": 18})
PROJECTIONS_PER_MODEL = sum(ROLE_COUNTS.values())
_BINDINGS: dict[str, "CompilerQ6PP512Binding"] = {}

# Packet-local diagnostic ABI.  This is deliberately not part of the route ABI:
# it describes only work that was actually present in the captured PROGRAM list.
BOUNDARY_NAMES = ("compact_q8_producer", "q6_main", "output_publication", "residual_epilogue")
BOUNDARY_MARKER_ABI = "tinygrad.nv_compiler_q6k_boundary_marker.v1"
FORCED_CUTS = ("producer", "main", "publication", "residual")

def forced_cut_identity(cut: str) -> str:
  """Return the stable identity for one explicitly requested cumulative cut."""
  if cut not in FORCED_CUTS: raise ValueError(f"unknown Q6-down forced cut: {cut}")
  return f"{BOUNDARY_MARKER_ABI}:{cut}:v1"

def forced_cut_from_env() -> str | None:
  """Read the opt-in cut selector; unset means ordinary route behavior."""
  cut = __import__("os").environ.get("NV_Q6DOWN_FORCED_CUT")
  if cut is not None and cut not in FORCED_CUTS: raise ValueError(f"unknown Q6-down forced cut: {cut}")
  return cut

@dataclass
class Q6DownBoundaryReplay:
  """Diagnostic-only wrapper for one exact 18-role Q6-down replay.

  The wrapper owns no route state and never changes submitted PROGRAMs.  A
  runner supplies the captured calls plus optional submission observations;
  missing device-side observations remain explicitly unavailable.
  """
  marker_identities: Mapping[str, str]
  role_counts: Mapping[str, int]

  def capture(self, calls, *, observations:Mapping[str, Mapping[str, Any]]|None=None) -> dict[str, Any]:
    observations = observations or {}
    names = [getattr(getattr(c, "src", (None,))[0], "arg", None) for c in calls]
    names = [getattr(a, "name", "") for a in names]
    producer_names = {f"q8_compact_record_fp16_q6_{role}" for role in self.role_counts}
    producers = [n for n in names if n in producer_names]
    records = []
    for boundary in BOUNDARY_NAMES:
      obs = dict(observations.get(boundary, {}))
      required = ("device_begin_ns", "device_end_ns", "queue_ready_ns", "dependency_wait_ns",
                  "allocations", "copies", "materializations")
      available = all(key in obs and obs[key] is not None for key in required)
      records.append({"boundary": boundary,
        "status": "OBSERVED" if available else "UNAVAILABLE",
        "marker_abi": self.marker_identities[boundary],
        "role_records": len(producers) if boundary in ("compact_q8_producer", "q6_main") else 0,
        "device_begin_ns": obs.get("device_begin_ns"), "device_end_ns": obs.get("device_end_ns"),
        "queue_ready_ns": obs.get("queue_ready_ns"), "dependency_wait_ns": obs.get("dependency_wait_ns"),
        "host_observed_ns": obs.get("host_observed_ns", time.monotonic_ns()),
        "allocations": obs.get("allocations", {"count": 0, "bytes": 0}),
        "copies": obs.get("copies", {"count": 0, "bytes": 0}),
        "materializations": obs.get("materializations", {"count": 0, "bytes": 0})})
    role_ok = len(producers) == sum(self.role_counts.values())
    complete = role_ok and all(r["status"] == "OBSERVED" for r in records)
    return {"schema": "tinygrad.nv_compiler_q6k_boundary_capture.v1", "status": "PASS" if complete else "BLOCKED",
      "required_boundaries": list(BOUNDARY_NAMES), "marker_abi": BOUNDARY_MARKER_ABI,
      "marker_identities": dict(self.marker_identities), "records": records,
      "exact_role_counts": dict(self.role_counts), "observed_producer_records": len(producers),
      "exact_role_census": role_ok,
      "limitation": None if complete else "device timestamps or allocation/copy/materialization observations unavailable"}

def new_boundary_replay(dev, *, role_counts:Mapping[str, int]=ROLE_COUNTS) -> Q6DownBoundaryReplay:
  """Create the explicit D0 wrapper; marker compilation remains opt-in."""
  return Q6DownBoundaryReplay(boundary_marker_identities(dev), MappingProxyType(dict(role_counts)))

def compile_boundary_markers(dev) -> Mapping[str, UOp]:
  """Compile the opt-in, no-op device marker ABI used by D0 capture.

  Markers deliberately have their own names/cache keys and a single diagnostic
  uint32 sink argument.  They are never installed by ``binding_for`` and do
  not participate in the model graph; callers must explicitly launch them
  around the four captured submissions.
  """
  source = "extern \"C\" __global__ void %s(unsigned int* marker) { if (threadIdx.x == 0) marker[0] = marker[0] + 1; }"
  out = {}
  for boundary in BOUNDARY_NAMES:
    name = f"nv_q6k_boundary_marker_{boundary}_v1"
    lib = NVRTCCompiler(dev.arch, ptx=False, cache_key=f"{name}_abi_{BOUNDARY_MARKER_ABI}").compile(source % name)
    out[boundary] = native_nv_program(name, lib, global_size=(1, 1, 1), local_size=(32, 1, 1),
                                      globals=(0,), ins=(0,))
  return MappingProxyType(out)

def boundary_marker_identities(dev) -> Mapping[str, str]:
  """Return stable diagnostic identities without changing route compilation."""
  return MappingProxyType({boundary: f"{BOUNDARY_MARKER_ABI}:{boundary}:v1" for boundary in BOUNDARY_NAMES})

def boundary_capture_from_calls(calls, *, role_counts:Mapping[str,int]=ROLE_COUNTS) -> dict[str,Any]:
  """Build a strict, non-inferential D0 capture record from submitted calls.

  The current graph exposes producer and Q6-main PROGRAMs only.  Publication and
  residual work are intentionally reported as unavailable rather than guessed.
  This keeps the diagnostic ABI distinct from promotion and makes D1 impossible
  to start on a partial capture.
  """
  now_ns = time.monotonic_ns()
  names = [getattr(getattr(c, "src", (None,))[0], "arg", None) for c in calls]
  names = [getattr(a, "name", "") for a in names]
  producer = [n for n in names if n.startswith("q8_compact_record_fp16_q6_")]
  main = [n for n in names if "candidate_context" in n]  # never matches: explicit fail-closed sentinel
  records = []
  for boundary, observed in (("compact_q8_producer", producer), ("q6_main", main),
                             ("output_publication", []), ("residual_epilogue", [])):
    records.append({"boundary": boundary, "status": "OBSERVED" if observed else "UNAVAILABLE",
      "role_records": len(observed), "device_begin_ns": None, "device_end_ns": None,
      "queue_ready_ns": None, "dependency_wait_ns": None, "host_observed_ns": now_ns,
      "allocations": {"count": 0, "bytes": 0}, "copies": {"count": 0, "bytes": 0},
      "materializations": {"count": 0, "bytes": 0}})
  return {"schema":"tinygrad.nv_compiler_q6k_boundary_capture.v1",
    "status":"BLOCKED", "required_boundaries":list(BOUNDARY_NAMES),
    "marker_abi":BOUNDARY_MARKER_ABI,
    "marker_identities":{boundary:f"{BOUNDARY_MARKER_ABI}:{boundary}:v1" for boundary in BOUNDARY_NAMES},
    "records":records, "exact_role_counts":dict(role_counts),
    "limitation":"current PROGRAM capture has no device-marker ABI and does not identify publication or residual epilogue"}


def supports(*, model_family:str, role:str, weight_type:str, m:int, n:int, k:int, device:str) -> bool:
  return (model_family == "qwen3_8b" and weight_type == "Q6_K" and m == M and
          ROLE_SHAPES.get(role) == (n, k) and device == "NV")


@dataclass(frozen=True)
class _Context:
  schema_version: str
  canonical_identity: str
  geometry: KernelTileGeometry
  packed_weight: PackedWeightTransform
  packed_fragment_provider: Q6KInt8FragmentProvider
  packed_activation: Q8ActivationRecordTransform
  packed_activation_provider: Q8Int8FragmentProvider
  group_accumulator: Q6KQ8SubgroupAccumulatorContract
  pipeline: None = None
  output_dtype: object = dtypes.float32


def _weight_carrier(halfs:Tensor, transform:PackedWeightTransform) -> Tensor:
  blocks = transform.rows*transform.blocks_per_row
  return halfs.reshape(blocks, 105).pad(((0, 0), (0, 23))).reshape(blocks, 128, 1).expand(blocks, 128, 2) \
    .reshape(transform.rows, transform.k).bitcast(dtypes.float16).cast(dtypes.int8)


def _activation_carrier(record:Tensor, transform:Q8ActivationRecordTransform) -> Tensor:
  return record.bitcast(dtypes.uint16)[:transform.values_bytes//2].reshape(transform.rows, transform.k//2, 1) \
    .expand(transform.rows, transform.k//2, 2).reshape(transform.rows, transform.k).bitcast(dtypes.float16).cast(dtypes.int8)


def _record_source(k:int, name:str) -> str:
  old = ("void q8_compact_fp16(const half* __restrict__ x, signed char* __restrict__ q,\n"
         " float* __restrict__ scales,float* __restrict__ sums) {")
  new = (f"void {name}(const half* __restrict__ x, unsigned int* __restrict__ record) {{\n"
         " signed char* __restrict__ q=(signed char*)record;\n"
         f" float* __restrict__ scales=(float*)(q+{M*k});\n"
         f" float* __restrict__ sums=scales+{M*(k//32)};")
  if old not in SRC_FP16: raise RuntimeError("Q8 producer source ABI changed")
  src = SRC_FP16.replace(old, new)
  if "base=row*4096+i;" not in src or "int g=row*128+seg*16+t/8;" not in src:
    raise RuntimeError("Q8 producer shape spelling changed")
  return src.replace("base=row*4096+i;", f"base=row*{k}+i;") \
    .replace("int g=row*128+seg*16+t/8;", f"int g=row*{k//32}+seg*16+t/8;")

def _build_role_expression(record:Tensor, halfs:Tensor, context:_Context) -> Tensor:
  """Construct the exact packed Q6 carrier expression before compiler capture."""
  return _activation_carrier(record, context.packed_activation).matmul(
    _weight_carrier(halfs, context.packed_weight).transpose(), dtype=dtypes.int) \
    .cast(context.output_dtype).contiguous()


@dataclass(frozen=True)
class _RoleAsset:
  role: str
  producer: object
  main_program: object
  transform: PackedWeightTransform
  activation: Q8ActivationRecordTransform
  context: _Context
  warmstart_key: tuple

  @property
  def candidate_identity(self) -> str: return self.context.canonical_identity


def _compile_role(dev, role:str, output_dtype=dtypes.float32, schedule:CompilerQ6ScheduleConfig|None=None) -> _RoleAsset:
  n, k = ROLE_SHAPES[role]
  wt, at = PackedWeightTransform("Q6_K", n, k), Q8ActivationRecordTransform(M, k)
  wp, ap = Q6KInt8FragmentProvider(wt), Q8Int8FragmentProvider(at)
  accumulator = Q6KQ8SubgroupAccumulatorContract(wp, ap)
  schedule = schedule or CompilerQ6ScheduleConfig()
  schedule.validate(M, n, k)
  (a_off, a_bytes, stride), (b_off, b_bytes, _) = schedule.lds_windows()
  geometry = KernelTileGeometry((schedule.tile_m, schedule.tile_n, schedule.tile_k), (schedule.warp_m, schedule.warp_n), schedule.threads, 32,
    (KernelLDSWindow("A", a_off, a_off+a_bytes, stride), KernelLDSWindow("B", b_off, b_off+b_bytes, stride)))
  identity = hashlib.sha256(repr((geometry, wp.identity, ap.identity, accumulator.abi, str(output_dtype))).encode()).hexdigest()
  context = _Context("boltbeam.full_kernel_candidate.v1", identity, geometry, wt, wp, at, ap, accumulator, output_dtype=output_dtype)
  key = warmstart_key({M, n}, k, {wt.storage_dtype, at.storage_dtype})
  name = f"q8_compact_record_fp16_q6_{role}"
  lib = NVRTCCompiler(dev.arch, ptx=False, cache_key=f"{name}_v1").compile(_record_source(k, name))
  producer = native_nv_program(name, lib, global_size=(M, k//512, 1), local_size=(128, 1, 1),
                               globals=(0, 1), outs=(1,), ins=(0,))

  from tinygrad.codegen import to_program_cache
  from tinygrad.codegen.opt.postrange import warmstart_candidate_state
  record_probe = Tensor.empty((M*k + 2*M*(k//32)*4)//4, dtype=dtypes.uint32, device="NV").realize()
  halfs_probe = Tensor.empty(wt.packed_bytes//2, dtype=dtypes.uint16, device="NV").realize()
  with warmstart_candidate_state({key:(Opt(OptOps.TC, 0, (-1, 2, 1)),)}, {key:context}):
      _build_role_expression(record_probe, halfs_probe, context).realize()
  matching = [program for program in to_program_cache.values() if program.op is Ops.PROGRAM and program.src and
              getattr(program.src[0].arg, "candidate_context", None) is not None and
              program.src[0].arg.candidate_context.canonical_identity == identity]
  if len(set(matching)) != 1: raise RuntimeError(f"expected one compiler Q6 {role} PROGRAM, found {len(set(matching))}")
  compiled = matching[0]
  main_program = compiled.replace(src=(UOp(Ops.SINK, arg=compiled.src[0].arg), compiled.src[1], UOp(Ops.LINEAR), *compiled.src[3:]))
  expected_global = (n//32, M//64, 1)
  if main_program.arg.outs != (0,) or main_program.arg.ins != (1, 2):
    raise RuntimeError(f"compiler Q6 {role} PROGRAM has unexpected ABI {main_program.arg}")
  persistent_global = (n//32, 1, 1)
  allowed_global = persistent_global if __import__('os').environ.get("NV_Q6_PERSISTENT_RESEARCH") == "1" else expected_global
  if (main_program.arg.global_size, main_program.arg.local_size) != (allowed_global, (32, 2, 2)):
    raise RuntimeError(f"compiler Q6 {role} PROGRAM lost qualified geometry: {main_program.arg}")
  return _RoleAsset(role, producer, main_program, wt, at, context, key)


@dataclass(frozen=True)
class CompilerQ6PP512Binding:
  roles: Mapping[str, _RoleAsset]
  warmstart: Mapping
  warmstart_contexts: Mapping

  @classmethod
  def compile(cls, dev) -> "CompilerQ6PP512Binding":
    schedule = CompilerQ6ScheduleConfig()
    roles = {role:_compile_role(dev, role, dtypes.float32, schedule) for role in ROLE_SHAPES}
    warmstart = {asset.warmstart_key:(Opt(OptOps.TC, schedule.tc_axis, (schedule.tc_select, schedule.tc_opt, schedule.use_tc)),) for asset in roles.values()}
    contexts = {asset.warmstart_key:asset.context for asset in roles.values()}
    if len(warmstart) != len(roles): raise RuntimeError("Q6 V/down warmstart keys collide")
    return cls(MappingProxyType(roles), MappingProxyType(warmstart), MappingProxyType(contexts))

  @classmethod
  def compile_research_output_dtype(cls, dev, output_dtype) -> "CompilerQ6PP512Binding":
    """Research-only asset constructor; production binding remains fp32."""
    if output_dtype not in (dtypes.float16, dtypes.float32): raise ValueError("unsupported Q6 output dtype")
    roles = {role:_compile_role(dev, role, output_dtype) for role in ROLE_SHAPES}
    warmstart = {asset.warmstart_key:(Opt(OptOps.TC, 0, (-1, 2, 1)),) for asset in roles.values()}
    contexts = {asset.warmstart_key:asset.context for asset in roles.values()}
    return cls(MappingProxyType(roles), MappingProxyType(warmstart), MappingProxyType(contexts))

  @classmethod
  def compile_research_schedule(cls, dev, schedule:CompilerQ6ScheduleConfig) -> "CompilerQ6PP512Binding":
    schedule.validate()
    roles = {role:_compile_role(dev, role, dtypes.float32, schedule) for role in ROLE_SHAPES}
    warmstart = {asset.warmstart_key:(Opt(OptOps.TC, schedule.tc_axis, (schedule.tc_select, schedule.tc_opt, schedule.use_tc)),) for asset in roles.values()}
    contexts = {asset.warmstart_key:asset.context for asset in roles.values()}
    return cls(MappingProxyType(roles), MappingProxyType(warmstart), MappingProxyType(contexts))

  @property
  def candidate_identities(self) -> Mapping[str,str]:
    return MappingProxyType({role:asset.candidate_identity for role,asset in self.roles.items()})

  def install_warmstart(self, model) -> None:
    opts, contexts = dict(model._packed_wmma_warmstart or {}), dict(model._packed_wmma_warmstart_contexts or {})
    for key, value in self.warmstart.items():
      if key in opts and opts[key] != value: raise RuntimeError("compiler Q6 warmstart key collides with another route")
      opts[key] = value
    for key, value in self.warmstart_contexts.items():
      if key in contexts and contexts[key] != value: raise RuntimeError("compiler Q6 context key collides with another route")
      contexts[key] = value
    model._packed_wmma_warmstart, model._packed_wmma_warmstart_contexts = opts, contexts

  def new_capture(self) -> "CompilerQ6PP512Capture": return CompilerQ6PP512Capture(self)

  def prepare_records(self, count:int) -> None:
    if count != PROJECTIONS_PER_MODEL: raise ValueError(f"exact Q6 route requires {PROJECTIONS_PER_MODEL} projections")


@dataclass
class CompilerQ6PP512Capture:
  asset: CompilerQ6PP512Binding
  trace_epoch: int = 0
  cursors: dict[str,int] | None = None

  def prepare_records(self, count:int) -> None: self.asset.prepare_records(count)
  def begin_trace(self) -> None: self.trace_epoch, self.cursors = self.trace_epoch+1, {role:0 for role in ROLE_SHAPES}

  def project(self, x:Tensor, halfs:Tensor, *, model_family:str, role:str, weight_type:str="Q6_K", wait:bool=False,
              output_dtype=dtypes.float32) -> Tensor:
    if self.trace_epoch == 0 or self.cursors is None: raise RuntimeError("begin_trace must establish a Q6 capture-local epoch")
    if role not in ROLE_COUNTS or self.cursors[role] >= ROLE_COUNTS[role]:
      raise RuntimeError(f"compiler Q6 trace exceeded exact {role} census")
    self.cursors[role] += 1
    return _project(self.asset, x, halfs, model_family=model_family, role=role, weight_type=weight_type, wait=wait, output_dtype=output_dtype)


def _project(binding:CompilerQ6PP512Binding, x:Tensor, halfs:Tensor, *, model_family:str, role:str,
             weight_type:str="Q6_K", wait:bool=False, output_dtype=dtypes.float32) -> Tensor:
  del wait
  n, k = ROLE_SHAPES.get(role, (0, 0))
  if not supports(model_family=model_family, role=role, weight_type=weight_type,
                  m=x.shape[0], n=n, k=x.shape[1], device=x.device):
    raise ValueError("unsupported compiler Q6 V/down research route")
  if x.dtype != dtypes.float16 or halfs.dtype != dtypes.uint16:
    raise ValueError("compiler Q6 route requires fp16 activation and canonical uint16 Q6_K halfwords")
  asset = binding.roles[role]
  record_u32 = (M*k + 2*M*(k//32)*4)//4
  record = Tensor.empty(record_u32, dtype=dtypes.uint32, device=x.device)
  _, record = x.uop_program(record, fxn=lambda *_:asset.producer)
  # Both roles use the pinned compiler PROGRAM.  In particular, V must not
  # fall back to a carrier matmul: that would discard the candidate context,
  # permit an epilogue rewrite, and make V depend on a different compiler path
  # than FFN-down.  The records, output, and canonical weight remain lazy and
  # scheduler-owned in both cases.
  out = Tensor.empty(M*n, dtype=binding.roles[role].context.output_dtype, device=x.device)
  out, record, halfs = out.uop_program(record, halfs, fxn=lambda *_:asset.main_program)
  result = out.reshape(M, n)
  if output_dtype not in (dtypes.float32, dtypes.float16):
    raise ValueError("research Q6 output_dtype must be float32 or float16")
  # Research-only ABI probe: this cast is intentionally after the captured
  # main PROGRAM. A static census can therefore prove whether it introduces a
  # second E kernel; the production/default fp32 path is unchanged.
  return result if output_dtype == dtypes.float32 else result.cast(dtypes.float16)


def binding_for(device:str="NV") -> CompilerQ6PP512Binding:
  if device != "NV": raise ValueError("compiler Q6 research binding is NV-only")
  if device not in _BINDINGS: _BINDINGS[device] = CompilerQ6PP512Binding.compile(Device[device])
  return _BINDINGS[device]
