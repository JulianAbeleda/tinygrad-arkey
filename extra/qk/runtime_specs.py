from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Any

OP_FAMILIES = ("QuantizedLinear", "DenseLinear", "FlashAttention", "KVCache", "ActivationFusion")
PHASES = ("prefill", "decode")
ROLES = ("ffn_gate_up", "ffn_down", "attn_qo", "attn_kv", "lm_head", "attention", "unknown")
QUANT_FORMATS = ("Q4_K", "Q6_K", "fp16", "fp8", "int8", "unknown")
ACTIVATION_FORMATS = ("fp16", "fp32", "Q8_1", "none")
LOWERING_STRATEGIES = (
  "packed_dequant_dot", "grouped_int_dot_correction", "iu8_wmma_grouped_dot", "iu8_wmma_tiled_grouped_dot",
  "dequant_once_matmul", "fused_dequant_wmma", "online_softmax_flash", "tinygrad_scheduler", "unknown",
)
PROVENANCE = ("machine_authored_generated", "tinygrad_scheduler_generated", "banned", "unknown")
GENERATED_PROVENANCE = ("machine_authored_generated", "tinygrad_scheduler_generated")
FULL_KERNEL_CANDIDATE_SCHEMA = "boltbeam.full_kernel_candidate.v1"
FULL_KERNEL_CANDIDATE_SET_SCHEMA = "boltbeam.full_kernel_candidate_set.v1"
PACKED_SCALAR_DECODER_VERSION = "ggml_k_quant_v1"
Q4K_Q8_1_FIVE_BUFFER_ABI = "q4k_q8_1_five_buffer_v1"
Q4K_Q8_1_EMITTER_FAMILY = "q4k_q8_1_mmq"
ANCHOR_SINGLE_BUFFER_CANDIDATE_HASH = "579b909f9d9b3ed89eab2129fca41baaa35c94b8eab040ccb0cbcee7a340fa0c"

class FullKernelAdmissionError(ValueError):
  def __init__(self, code:str, message:str): self.code = code; super().__init__(f"{code}: {message}")

@dataclass(frozen=True)
class FullKernelCapability:
  capability_id: str = "amd.gfx1100.prefill.wmma_lds.single_buffer.v1"
  backend: str = "AMD"
  arch: str = "gfx1100"
  wave_size: int = 32
  max_lds_bytes: int = 65536
  buffer_count: int = 1
  stage_count: int = 1
  vector_bytes: int = 16
  instruction_family: str = "wmma_f32_16x16x16_f16"
  fragment_layout: str = "rdna3_wmma_f32_16x16x16_f16_lds2_static"
  transport: str = "lds"

GFX1100_SINGLE_BUFFER_CAPABILITY = FullKernelCapability()
GFX1100_TWO_BUFFER_STAGE1_CAPABILITY = FullKernelCapability(
  capability_id="amd.gfx1100.prefill.wmma_lds.two_buffer_stage1.v1", buffer_count=2, stage_count=1)
GFX1100_REGISTER_RESIDENT_CAPABILITY = FullKernelCapability(
  capability_id="amd.gfx1100.prefill.wmma_register.two_stage.v1", buffer_count=1, stage_count=2,
  fragment_layout="rdna3_wmma_f32_16x16x16_f16_register_static", transport="direct_l2")
GFX1100_Q4K_Q8_FIVE_BUFFER_CAPABILITY = FullKernelCapability(
  capability_id="amd.gfx1100.prefill.q4k_q8.direct_physical_ds4.v1", max_lds_bytes=0,
  buffer_count=0, stage_count=0, vector_bytes=16, instruction_family="wmma_i32_16x16x16_iu8",
  fragment_layout="rdna3_wave32_signed_i8_direct_global", transport="direct_global")

@dataclass(frozen=True)
class Q4KQ8FiveBufferEmitterPlan:
  tile: tuple[int,int,int] = (16,16,256)
  waves: tuple[int,int] = (1,1)
  threads: int = 32
  transport: str = "direct_global"
  active_lds_bytes: int = 0
  instruction_family: str = "wmma_i32_16x16x16_iu8"
  activation_layout: str = "q8_1_mmq_ds4_transposed_blocks"
  tail_policy: str = "aligned_only_no_tails"

def candidate_storage_kind(payload: dict[str, Any]) -> str:
  """Resolve typed stage storage while keeping legacy payloads on LDS."""
  if payload.get("kernel_abi", {}).get("family") == Q4K_Q8_1_FIVE_BUFFER_ABI: return "direct_global"
  residency = payload.get("schedule", {}).get("residency", {}) if isinstance(payload, dict) else {}
  resident = residency.get("resident", ()) if isinstance(residency, dict) else ()
  return "global_register_resident" if isinstance(resident, (list, tuple)) and "stage_ab_register" in resident else "lds"

def capability_transport(capability: "FullKernelCapability") -> str:
  """Typed transport carried by an already-admitted capability.

  The transport is read from the typed capability lattice element, never
  inferred from residency marker strings.  Register-resident admission is the
  direct-L2 transport; the five-buffer physical-DS4 capability is direct-global.
  """
  return capability.transport

def full_kernel_candidate_capability(payload:dict[str,Any]) -> "FullKernelCapability":
  """Resolve the frozen hardware capability from typed schedule facts in one place."""
  if payload.get("kernel_abi", {}).get("family") == Q4K_Q8_1_FIVE_BUFFER_ABI: return GFX1100_Q4K_Q8_FIVE_BUFFER_CAPABILITY
  if candidate_storage_kind(payload) == "global_register_resident": return GFX1100_REGISTER_RESIDENT_CAPABILITY
  pipeline = payload.get("schedule", {}).get("pipeline", {})
  return GFX1100_TWO_BUFFER_STAGE1_CAPABILITY if \
    (pipeline.get("buffer_count"), pipeline.get("stage_count")) == (2, 1) else GFX1100_SINGLE_BUFFER_CAPABILITY

@dataclass(frozen=True)
class FullKernelAdmission:
  canonical_identity: str
  normalized_payload: dict[str, Any]
  geometry: Any
  plan: Any
  pipeline_plan: Any
  active_lds_bytes: int
  capability: FullKernelCapability
  context: Any
  operand_plan: Any = None

FullKernelExactKey = tuple[str, int, int, int, str, str, int]
LegacyFullKernelExactKey = tuple[str, str, int, int, int, str, str, int]
FullKernelWarmstartKey = tuple[frozenset[int], int]

@dataclass(frozen=True)
class FullKernelWorkload:
  """Typed model-independent workload identity carried by a full-kernel candidate."""
  profile: str
  role: str
  shape: tuple[int, int, int]
  target: dict[str, Any]

  @property
  def target_id(self) -> str:
    return f"{self.target['backend']}:{self.target['arch']}:wave{self.target['wave_size']}"

  @property
  def exact_key(self) -> FullKernelExactKey:
    return (self.role, *self.shape, self.target["backend"], self.target["arch"], self.target["wave_size"])

  @property
  def legacy_exact_key(self) -> LegacyFullKernelExactKey:
    """Profile-bearing lookup alias for legacy artifacts; never a semantic key."""
    return (self.profile, *self.exact_key)


def full_kernel_workload(payload:dict[str,Any]) -> FullKernelWorkload:
  """Parse the canonical workload identity without making model or route decisions."""
  try:
    workload, shape, target = payload["workload"], payload["workload"]["shape"], payload["workload"]["target"]
    profile, role = workload["profile"], workload["role"]
    mnk = tuple(shape[x] for x in ("m", "n", "k"))
    target_row = {x:target[x] for x in ("backend", "arch", "wave_size")}
  except (KeyError, TypeError) as exc:
    raise FullKernelAdmissionError("workload_schema", "candidate workload is malformed") from exc
  if not isinstance(profile, str) or not profile or not isinstance(role, str) or not role:
    raise FullKernelAdmissionError("workload_schema", "candidate profile and role must be non-empty strings")
  if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in mnk):
    raise FullKernelAdmissionError("workload_schema", "candidate M/N/K must be positive integers")
  if any(not isinstance(target_row[x], str) or not target_row[x] for x in ("backend", "arch")) or \
     not isinstance(target_row["wave_size"], int) or isinstance(target_row["wave_size"], bool) or target_row["wave_size"] <= 0:
    raise FullKernelAdmissionError("workload_schema", "candidate target is malformed")
  return FullKernelWorkload(profile, role, mnk, target_row)

class _FrozenDict(dict):
  def _immutable(self,*_args,**_kwargs): raise TypeError("candidate-set payload is immutable")
  __setitem__=__delitem__=clear=pop=popitem=setdefault=update=_immutable

def _freeze_json(value:Any) -> Any:
  if isinstance(value,dict): return _FrozenDict({k:_freeze_json(v) for k,v in value.items()})
  if isinstance(value,list): return tuple(_freeze_json(v) for v in value)
  return value

def _semantic_full_kernel_payload(payload:dict[str,Any]) -> dict[str,Any]:
  """Strip provenance-only profile labels from otherwise exact candidate content."""
  semantic = json.loads(json.dumps(payload, allow_nan=False))
  semantic.get("workload", {}).pop("profile", None)
  semantic.get("applicability", {}).pop("profiles", None)
  return semantic

def _legacy_full_kernel_identity(payload:dict[str,Any]) -> str:
  encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
  return hashlib.sha256(encoded).hexdigest()

def _canonical_full_kernel_identity(payload:dict[str,Any]) -> str:
  encoded = json.dumps(_semantic_full_kernel_payload(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
  return hashlib.sha256(encoded).hexdigest()

def _full_kernel_exact_key(payload:dict[str,Any]) -> FullKernelExactKey:
  return full_kernel_workload(payload).exact_key

def _full_kernel_warmstart_key(payload:dict[str,Any]) -> FullKernelWarmstartKey:
  shape=payload["workload"]["shape"]
  return frozenset((shape["m"],shape["n"])),shape["k"]

@dataclass(frozen=True)
class FullKernelCandidateSetEntry:
  canonical_identity: str
  payload: dict[str,Any]

  def __post_init__(self) -> None:
    try: payload=json.loads(json.dumps(self.payload,allow_nan=False))
    except (TypeError,ValueError) as exc: raise FullKernelAdmissionError("payload_json",str(exc)) from exc
    semantic_identity, legacy_identity = _canonical_full_kernel_identity(payload), _legacy_full_kernel_identity(payload)
    if self.canonical_identity not in (semantic_identity, legacy_identity):
      raise FullKernelAdmissionError("identity_mismatch","candidate-set entry identity differs from canonical payload")
    object.__setattr__(self,"canonical_identity",semantic_identity)
    object.__setattr__(self,"payload",_freeze_json(payload))

  @property
  def exact_key(self) -> FullKernelExactKey: return _full_kernel_exact_key(self.payload)
  @property
  def legacy_exact_key(self) -> LegacyFullKernelExactKey: return full_kernel_workload(self.payload).legacy_exact_key
  @property
  def legacy_identity_alias(self) -> str: return _legacy_full_kernel_identity(self.payload)
  @property
  def warmstart_key(self) -> FullKernelWarmstartKey: return _full_kernel_warmstart_key(self.payload)
  def to_json(self) -> dict[str,Any]:
    return {"canonical_identity":self.canonical_identity,"payload":json.loads(json.dumps(self.payload))}


def derive_packed_weight_candidate(payload:dict[str,Any], quant_format:str) -> FullKernelCandidateSetEntry:
  """Return a canonical full-kernel candidate whose ABI slot 2 is a packed Q4_K/Q6_K weight.

  This is the single construction authority shared by compile gates, canaries, and future model routing; callers do
  not hand-maintain packed block geometry or canonical identities.
  """
  normalized = json.loads(json.dumps(payload, allow_nan=False))
  shape = normalized.get("workload", {}).get("shape", {})
  try: rows, k = int(shape["n"]), int(shape["k"])
  except (KeyError, TypeError, ValueError) as exc: raise ValueError("packed candidate requires workload shape n/k") from exc
  from tinygrad.codegen.opt.packed_weight import PackedWeightTransform
  transform = PackedWeightTransform(quant_format, rows, k)
  normalized["operand_sources"] = {
    "a":{"kind":"dense", "logical_dtype":"fp16", "storage_dtype":"fp16", "abi_slot":1},
    "b":{"kind":"packed_scalar_decoder", "logical_dtype":"fp16",
         "storage_dtype":"uint32" if quant_format == "Q4_K" else "uint16", "abi_slot":2,
         "quant_format":quant_format, "rows":rows, "k":k, "block_elems":transform.block_elems,
         "block_bytes":transform.block_bytes, "decoder_version":PACKED_SCALAR_DECODER_VERSION}}
  identity = _canonical_full_kernel_identity(normalized)
  return FullKernelCandidateSetEntry(identity, normalized)

def q4k_q8_1_five_buffer_abi_plan() -> dict[str,Any]:
  """Return the canonical JSON ABI descriptor consumed by admission and compile adapters."""
  from extra.qk.layout import (Q4K_WORDS_PER_BLOCK, Q4_K_BLOCK_ELEMS, Q8_1_BLOCK_ELEMS,
                               Q8_1_MMQ_BLOCK_ELEMS, Q8_1_MMQ_GROUPS_PER_BLOCK)
  from extra.qk.mmq_q4k_q8_reference import Q8_1_MMQ_DS4_LAYOUT
  return {"family":Q4K_Q8_1_FIVE_BUFFER_ABI, "quant_format":"Q4_K", "activation_format":"Q8_1",
    "activation_layout":Q8_1_MMQ_DS4_LAYOUT, "output_layout":"tokens_rows", "emitter_family":Q4K_Q8_1_EMITTER_FAMILY,
    "block_geometry":{"q4_block_elems":Q4_K_BLOCK_ELEMS, "q4_words_per_block":Q4K_WORDS_PER_BLOCK,
      "q8_group_elems":Q8_1_BLOCK_ELEMS, "q8_ds4_block_elems":Q8_1_MMQ_BLOCK_ELEMS,
      "q8_groups_per_ds4_block":Q8_1_MMQ_GROUPS_PER_BLOCK},
    "buffers":{
      "output":{"abi_slot":0,"direction":"out","storage_dtype":"float32","logical_axes":["m","n"],
        "axis_extents":[["workload","m"],["workload","n"]],"access":"logical"},
      "q4_packed_words":{"abi_slot":1,"direction":"in","storage_dtype":"uint32",
        "logical_axes":["n","q4_blocks","q4_words"],"axis_extents":[["workload","n"],
          ["quotient",["workload","k"],["block_geometry","q4_block_elems"]],
          ["block_geometry","q4_words_per_block"]],"access":"flat"},
      "q8_ds4_values":{"abi_slot":2,"direction":"in","storage_dtype":"int8","signed":True,
        "logical_axes":["ds4_blocks","m","ds4_block_elems"],"axis_extents":[
          ["quotient",["workload","k"],["block_geometry","q8_ds4_block_elems"]],
          ["workload","m"],["block_geometry","q8_ds4_block_elems"]],"access":"flat"},
      "q8_scales":{"abi_slot":3,"direction":"in","storage_dtype":"float32",
        "logical_axes":["ds4_blocks","m","q8_groups_per_ds4_block"],"axis_extents":[
          ["quotient",["workload","k"],["block_geometry","q8_ds4_block_elems"]],
          ["workload","m"],["block_geometry","q8_groups_per_ds4_block"]],"access":"flat"},
      "q8_weighted_sums":{"abi_slot":4,"direction":"in","storage_dtype":"float32",
        "logical_axes":["ds4_blocks","m","q8_groups_per_ds4_block"],"axis_extents":[
          ["quotient",["workload","k"],["block_geometry","q8_ds4_block_elems"]],
          ["workload","m"],["block_geometry","q8_groups_per_ds4_block"]],"access":"flat"}}}

def _q4k_q8_1_direct_emitter_schedule() -> dict[str,Any]:
  """Exact model-independent schedule facts of the physical-DS4 direct UOp emitter."""
  return {"variant":"q4k_q8_1_physical_ds4_direct_v1",
    "tile":{"m":16,"n":16,"k":256}, "waves":{"m":1,"n":1}, "threads":32,
    "transport":"direct_global", "lane_ownership":"rdna3_wave32_direct_wmma_output_tile",
    "operands":{"q4_packed_words":{"source":"global","alignment":16},
      "q8_ds4_values":{"source":"global","alignment":16,"signed":True},
      "q8_scales":{"source":"global","alignment":4},"q8_weighted_sums":{"source":"global","alignment":4}},
    "lds_bytes":0, "pipeline":{"buffer_count":0,"stage_count":0},
    "wmma":{"instruction_family":"wmma_i32_16x16x16_iu8","fragment_layout":"rdna3_wave32_signed_i8_direct_global",
      "accumulator_ownership":"int32_group_dot_fp32_scale_min_correction"},
    "epilogue":{"lane_mapping":"wmma_accumulator_scalar_f32","vector_width":1},
    "tail_policy":"aligned_only_no_tails",
    "compile_environment":{"REGALLOC_END_NO_SOURCE_LIVE":1,"REGALLOC_ADDR_REMAT":1},
    "numerical_mode":"signed_i8_dot_int32_fp32_scale_min_correction"}

def derive_q4k_q8_1_five_buffer_candidate(payload:dict[str,Any]) -> FullKernelCandidateSetEntry:
  """Retain the model-independent Q4_K/Q8_1 DS4 full-kernel ABI in a v1 candidate."""
  normalized = json.loads(json.dumps(payload, allow_nan=False))
  if "operand_sources" in normalized: raise ValueError("five-buffer ABI is ambiguous with operand_sources")
  normalized["kernel_abi"] = q4k_q8_1_five_buffer_abi_plan()
  normalized["workload"]["dtypes"] = {"a":"Q8_1","b":"Q4_K","c":"fp32","accumulator":"int32_fp32"}
  normalized["workload"]["layout"] = {"a":"physical_ds4","b":"q4_k_packed_words","c":"tokens_rows"}
  normalized["schedule"] = _q4k_q8_1_direct_emitter_schedule()
  normalized["static_constraints"] = {"max_lds_bytes":0,"max_vgpr_per_thread":256,"allow_spill":False}
  _validate_full_kernel_payload(normalized)
  return FullKernelCandidateSetEntry(_canonical_full_kernel_identity(normalized), normalized)


def rebind_full_kernel_workload(payload:dict[str,Any], *, profile:str, role:str, shape:tuple[int,int,int],
                                target:dict[str,Any]|None=None) -> FullKernelCandidateSetEntry:
  """Rebind a schedule template to an exact workload and return its new canonical identity.

  This changes workload/applicability data only. Admission remains responsible for proving that the retained schedule is
  legal for the new shape and target.
  """
  normalized = json.loads(json.dumps(payload, allow_nan=False))
  if len(shape) != 3 or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in shape):
    raise ValueError("rebound workload shape must contain positive integer M/N/K")
  if not isinstance(profile, str) or not profile or not isinstance(role, str) or not role:
    raise ValueError("rebound workload profile and role must be non-empty strings")
  workload = normalized.get("workload")
  if not isinstance(workload, dict): raise ValueError("schedule template has no workload")
  target_row = dict(workload.get("target", {}) if target is None else target)
  if set(target_row) != {"backend", "arch", "wave_size"}: raise ValueError("rebound workload target is malformed")
  workload.update(profile=profile, role=role, shape=dict(zip(("m", "n", "k"), shape)), target=target_row)
  normalized["applicability"] = {"exact_shape":True, "profiles":[profile], "roles":[role],
    "targets":[f"{target_row['backend']}:{target_row['arch']}:wave{target_row['wave_size']}"]}
  normalized.pop("operand_sources", None)
  _validate_full_kernel_payload(normalized)
  return FullKernelCandidateSetEntry(_canonical_full_kernel_identity(normalized), normalized)

@dataclass(frozen=True)
class FullKernelCandidateSet:
  entries: tuple[FullKernelCandidateSetEntry,...]
  schema: str = FULL_KERNEL_CANDIDATE_SET_SCHEMA

  def __post_init__(self) -> None:
    if self.schema != FULL_KERNEL_CANDIDATE_SET_SCHEMA:
      raise FullKernelAdmissionError("candidate_set_schema",f"unsupported candidate-set schema {self.schema!r}")
    object.__setattr__(self,"entries",tuple(self.entries))

  def to_json(self) -> dict[str,Any]: return {"schema":self.schema,"entries":[x.to_json() for x in self.entries]}

  @classmethod
  def from_json(cls,row:dict[str,Any]) -> "FullKernelCandidateSet":
    if not isinstance(row,dict) or set(row) != {"schema","entries"} or not isinstance(row["entries"],list):
      raise FullKernelAdmissionError("candidate_set_schema","candidate set requires exactly schema and entries")
    return cls(tuple(FullKernelCandidateSetEntry(x["canonical_identity"],x["payload"]) for x in row["entries"]),row["schema"])

@dataclass(frozen=True)
class AdmittedFullKernelCandidateSet:
  candidate_set: FullKernelCandidateSet
  admissions: tuple[FullKernelAdmission,...]
  exact_index: Any = field(init=False,repr=False)

  def __post_init__(self) -> None:
    if len(self.candidate_set.entries) != len(self.admissions): raise ValueError("candidate-set admission count mismatch")
    exact:dict[FullKernelExactKey,FullKernelAdmission]={}; weak:dict[FullKernelWarmstartKey,tuple[FullKernelExactKey,str]]={}
    for entry,admission in zip(self.candidate_set.entries,self.admissions):
      key=entry.exact_key
      if key in exact: raise FullKernelAdmissionError("duplicate_exact_key",f"duplicate candidate exact key {key!r}")
      prior=weak.get(entry.warmstart_key)
      if prior is not None and prior != (key,entry.canonical_identity):
        raise FullKernelAdmissionError("warmstart_key_collision",
          f"weak warmstart key {entry.warmstart_key!r} aliases {prior[0]!r} and {key!r}")
      exact[key]=admission; weak[entry.warmstart_key]=(key,entry.canonical_identity)
    object.__setattr__(self,"exact_index",MappingProxyType(exact))

  def get(self,role:str,shape:tuple[int,int,int],target:dict[str,Any]) -> FullKernelAdmission|None:
    return self.exact_index.get((role,*shape,target["backend"],target["arch"],target["wave_size"]))

  def legacy_get(self,profile:str,role:str,shape:tuple[int,int,int],target:dict[str,Any]) -> FullKernelAdmission|None:
    """Read a profile-bearing legacy binding without making profile a selector."""
    del profile
    return self.get(role,shape,target)

def admit_full_kernel_candidate_set(candidate_set:FullKernelCandidateSet) -> AdmittedFullKernelCandidateSet:
  admissions=[]
  for entry in candidate_set.entries:
    role,m,n,k,backend,arch,wave_size=entry.exact_key
    admissions.append(admit_full_kernel_candidate(entry.payload,entry.canonical_identity,profile=full_kernel_workload(entry.payload).profile,role=role,
      shape=(m,n,k),target={"backend":backend,"arch":arch,"wave_size":wave_size},
      capability=full_kernel_candidate_capability(entry.payload)))
  return AdmittedFullKernelCandidateSet(candidate_set,tuple(admissions))

def full_kernel_candidate_set_from_legacy(payload:dict[str,Any],canonical_identity:str) -> FullKernelCandidateSet:
  """Adapt the current JSON/hash environment pair without changing its individual candidate identity."""
  return FullKernelCandidateSet((FullKernelCandidateSetEntry(canonical_identity,payload),))

def admit_full_kernel_candidate(payload:dict[str, Any], canonical_identity:str, *, profile:str, role:str,
                                shape:tuple[int,int,int], target:dict[str,Any],
                                capability:FullKernelCapability=GFX1100_SINGLE_BUFFER_CAPABILITY) -> FullKernelAdmission:
  try: normalized = json.loads(json.dumps(payload, allow_nan=False))
  except (TypeError, ValueError) as exc: raise FullKernelAdmissionError("payload_json", str(exc)) from exc
  try: _validate_full_kernel_payload(normalized)
  except ValueError as exc: raise FullKernelAdmissionError("payload_schema", str(exc)) from exc
  actual_identity = _canonical_full_kernel_identity(normalized)
  if canonical_identity not in (actual_identity, _legacy_full_kernel_identity(normalized)):
    raise FullKernelAdmissionError("identity_mismatch", "canonical SHA-256 differs from semantic payload or its legacy alias")
  workload,schedule,applicability = normalized["workload"],normalized["schedule"],normalized["applicability"]
  storage_kind = candidate_storage_kind(normalized)
  # Preserve the public default while resolving the register transport to its
  # own frozen capability. Explicit non-default capabilities still remain
  # authoritative and are validated below.
  if storage_kind == "global_register_resident" and capability is GFX1100_SINGLE_BUFFER_CAPABILITY:
    capability = GFX1100_REGISTER_RESIDENT_CAPABILITY
  if storage_kind == "direct_global" and capability is GFX1100_SINGLE_BUFFER_CAPABILITY:
    capability = GFX1100_Q4K_Q8_FIVE_BUFFER_CAPABILITY
  target_id = f"{target['backend']}:{target['arch']}:wave{target['wave_size']}"
  # Profile is retained in legacy payloads and call signatures solely as provenance.
  if workload["role"] != role or role not in applicability["roles"]: raise FullKernelAdmissionError("workload_role", "role is not exact/applicable")
  if tuple(workload["shape"][x] for x in ("m","n","k")) != shape or not applicability["exact_shape"]:
    raise FullKernelAdmissionError("workload_shape", "shape is not exact")
  if workload["target"] != target or target_id not in applicability["targets"]: raise FullKernelAdmissionError("workload_target", "target is not exact/applicable")
  if target != {"backend":capability.backend,"arch":capability.arch,"wave_size":capability.wave_size}:
    raise FullKernelAdmissionError("capability_target", "target is outside frozen capability")
  if "kernel_abi" in normalized:
    if capability is not GFX1100_Q4K_Q8_FIVE_BUFFER_CAPABILITY:
      raise FullKernelAdmissionError("capability_five_buffer", "five-buffer ABI requires its typed direct-global capability")
    if any(shape[i] % (16,16,256)[i] for i in range(3)):
      raise FullKernelAdmissionError("geometry_divisibility", "direct five-buffer workload is not 16x16x256 aligned (no tails)")
    direct_plan = Q4KQ8FiveBufferEmitterPlan()
    from tinygrad.uop.ops import KernelCandidateContext
    context = KernelCandidateContext(schema_version=normalized["schema_version"], canonical_identity=actual_identity,
      geometry=None, pipeline=direct_plan)
    operand_plan = _freeze_json(normalized["kernel_abi"])
    return FullKernelAdmission(actual_identity,normalized,None,direct_plan,direct_plan,0,capability,context,operand_plan)
  if workload["dtypes"] != {"a":"fp16","b":"fp16","c":"fp16","accumulator":"fp32"}:
    raise FullKernelAdmissionError("capability_dtype", "only fp16/fp32 accumulation is supported")
  if storage_kind == "global_register_resident":
    if (schedule["pipeline"]["buffer_count"], schedule["pipeline"]["stage_count"]) != (1, 2):
      raise FullKernelAdmissionError("capability_register_pipeline", "register candidates require one static slot and two logical stages")
    if role != "attn_qo" or shape != (512, 4096, 4096):
      raise FullKernelAdmissionError("capability_register_shape", "the current register template is only proved for attn_qo 512x4096x4096")
  elif schedule["pipeline"]["buffer_count"] != capability.buffer_count or schedule["pipeline"]["stage_count"] != capability.stage_count:
    raise FullKernelAdmissionError("capability_pipeline", "only single-buffer stage1 is supported")
  if schedule["wmma"]["instruction_family"] != capability.instruction_family:
    raise FullKernelAdmissionError("capability_tc", "tensor-core descriptor is unsupported")
  if schedule["wmma"]["fragment_layout"] != capability.fragment_layout:
    raise FullKernelAdmissionError("capability_tc", "tensor-core descriptor is unsupported")
  if (storage_kind != "global_register_resident" and
      any(schedule["lds"][x] != 8 for x in ("store_vector_width","load_vector_width"))) or \
     any(schedule["cooperative_load"][r]["vector_width"]*2 != capability.vector_bytes or
         schedule["cooperative_load"][r]["alignment"] != capability.vector_bytes for r in ("a","b")):
    raise FullKernelAdmissionError("capability_vector", "only aligned b128 fp16 transport is supported")
  expected_lane_mapping = "wave_contiguous_b128" if storage_kind == "global_register_resident" else "cooperative_row_stride_64_b128"
  if any(schedule["cooperative_load"][r]["lane_mapping"] != expected_lane_mapping for r in ("a","b")):
    raise FullKernelAdmissionError("capability_lane_map", f"{storage_kind} requires {expected_lane_mapping}")
  from tinygrad.uop.ops import KernelCandidateContext, KernelLDSWindow, KernelTileGeometry
  # KernelTileGeometry predates non-LDS transport and still carries mandatory
  # compatibility windows. Register admission uses inert aligned sentinels;
  # no register authority is derived from payload LDS layout fields.
  windows = ((KernelLDSWindow("A",0,16,16),KernelLDSWindow("B",16,32,16)) if storage_kind == "global_register_resident" else
             tuple(KernelLDSWindow(r.upper(),*schedule["lds"]["windows"][r],schedule["lds"]["strides"][r]) for r in ("a","b")))
  try: geometry = KernelTileGeometry(tuple(schedule["tile"][x] for x in ("m","n","k")),
    tuple(schedule["waves"][x] for x in ("m","n")),schedule["threads"],target["wave_size"],windows)
  except ValueError as exc: raise FullKernelAdmissionError("geometry_invalid", str(exc)) from exc
  if any(shape[i] % geometry.tile[i] for i in range(3)): raise FullKernelAdmissionError("geometry_divisibility", "workload is not tile divisible")
  from tinygrad.codegen.opt.kernel_pipeline import KernelStage1PipelinePlan
  if storage_kind == "global_register_resident":
    from tinygrad.codegen.opt.compiler_policies import RegisterPipePlan
    pipeline_plan = RegisterPipePlan()
    active_lds = 0
  else:
    pipeline_plan = KernelStage1PipelinePlan(capability.buffer_count, geometry.lds_windows[-1].end, capability.stage_count)
    active_lds = pipeline_plan.active_lds_bytes
  if active_lds > capability.max_lds_bytes or active_lds > normalized["static_constraints"]["max_lds_bytes"]:
    raise FullKernelAdmissionError("capability_lds", "active LDS exceeds a declared limit")
  try:
    from tinygrad.codegen.opt.kernel_lds import derive_precontract_factors, derive_precontract_shape_factors
    from tinygrad.codegen.opt.tc import amd_rdna3
    from tinygrad.dtype import dtypes
    tc = next(x for x in amd_rdna3 if x.dtype_in == dtypes.half and x.dtype_out == dtypes.float)
    plan = (derive_precontract_shape_factors(geometry, tc) if storage_kind == "global_register_resident" else
            derive_precontract_factors(geometry, tc))
  except ValueError as exc: raise FullKernelAdmissionError("capability_geometry", str(exc)) from exc
  # Keep the established buffer1 context and binary identity unchanged.
  packed_weight = None
  if "operand_sources" in normalized:
    b_source = normalized["operand_sources"]["b"]
    if b_source["kind"] == "packed_scalar_decoder":
      if storage_kind == "global_register_resident":
        raise FullKernelAdmissionError("capability_storage", "packed-weight candidates require LDS tile storage")
      from tinygrad.codegen.opt.packed_weight import PackedWeightTransform
      packed_weight = PackedWeightTransform(b_source["quant_format"], b_source["rows"], b_source["k"],
                                            b_source["block_elems"], b_source["block_bytes"])
  context = KernelCandidateContext(schema_version=normalized["schema_version"], canonical_identity=actual_identity, geometry=geometry,
    pipeline=pipeline_plan if (capability.buffer_count > 1 or storage_kind == "global_register_resident") else None,
    packed_weight=packed_weight)
  operand_plan = _freeze_json(normalized["kernel_abi"]) if "kernel_abi" in normalized else None
  return FullKernelAdmission(actual_identity,normalized,geometry,plan,pipeline_plan,active_lds,capability,context,operand_plan)


def bind_full_kernel_candidate(payload:dict[str, Any], canonical_identity:str, *, profile:str, role:str,
                               shape:tuple[int, int, int], target:dict[str, Any], tile:tuple[int, int, int]|None=None,
                               waves:tuple[int, int]|None=None, threads:int|None=None, buffer_count:int|None=None,
                               stage_count:int|None=None, lds_windows:dict[str, list[int]]|None=None,
                               lds_strides:dict[str, int]|None=None, lds_padding:int|None=None, lds_bytes:int|None=None):
  """Compatibility wrapper. Schedule authority comes exclusively from the canonical payload."""
  return admit_full_kernel_candidate(payload,canonical_identity,profile=profile,role=role,shape=shape,target=target,
                                     capability=full_kernel_candidate_capability(payload)).context


def _check(name:str, value:str, allowed:tuple[str, ...]) -> str:
  if value not in allowed: raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
  return value


def _shape_json(shape:dict[str, int | str]) -> dict[str, int | str]:
  out: dict[str, int | str] = {}
  for k, v in shape.items():
    if not isinstance(k, str): raise ValueError(f"shape key must be str, got {k!r}")
    if not isinstance(v, (int, str)): raise ValueError(f"shape[{k!r}] must be int|str, got {type(v).__name__}")
    out[k] = v
  return out


def _strict_keys(row:dict[str, Any], required:set[str], label:str) -> None:
  if not isinstance(row, dict): raise ValueError(f"{label} must be an object")
  missing, unknown = required - set(row), set(row) - required
  if missing: raise ValueError(f"{label} missing fields {sorted(missing)}")
  if unknown: raise ValueError(f"{label} has unknown fields {sorted(unknown)}")


def _positive_int(value:Any, label:str) -> None:
  if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
    raise ValueError(f"{label} must be a positive int, got {value!r}")


def _nonempty_str(value:Any, label:str) -> None:
  if not isinstance(value, str) or not value: raise ValueError(f"{label} must be a non-empty string")


def _validate_full_kernel_payload(payload:dict[str, Any]) -> None:
  required = {"schema_version", "workload", "schedule", "static_constraints", "applicability"}
  if not isinstance(payload, dict): raise ValueError("full_kernel_candidate must be an object")
  missing, unknown = required - set(payload), set(payload) - required - {"operand_sources", "kernel_abi"}
  if missing: raise ValueError(f"full_kernel_candidate missing fields {sorted(missing)}")
  if unknown: raise ValueError(f"full_kernel_candidate has unknown fields {sorted(unknown)}")
  if payload["schema_version"] != FULL_KERNEL_CANDIDATE_SCHEMA:
    raise ValueError(f"unsupported full-kernel candidate schema_version {payload['schema_version']!r}")
  workload = payload["workload"]
  _strict_keys(workload, {"profile", "role", "shape", "dtypes", "layout", "target"}, "workload")
  _nonempty_str(workload["profile"], "workload.profile")
  _nonempty_str(workload["role"], "workload.role")
  groups = {"shape": {"m", "n", "k"}, "dtypes": {"a", "b", "c", "accumulator"},
            "layout": {"a", "b", "c"}, "target": {"backend", "arch", "wave_size"}}
  for name, keys in groups.items(): _strict_keys(workload[name], keys, f"workload.{name}")
  for dim in ("m", "n", "k"): _positive_int(workload["shape"][dim], f"workload.shape.{dim}")
  for group in ("dtypes", "layout"):
    for key, value in workload[group].items(): _nonempty_str(value, f"workload.{group}.{key}")
  for key in ("backend", "arch"): _nonempty_str(workload["target"][key], f"workload.target.{key}")
  _positive_int(workload["target"]["wave_size"], "workload.target.wave_size")

  if "operand_sources" in payload and "kernel_abi" in payload:
    raise ValueError("full-kernel candidate cannot combine kernel_abi with operand_sources")
  direct_five_buffer = "kernel_abi" in payload
  if "kernel_abi" in payload:
    if payload["kernel_abi"] != q4k_q8_1_five_buffer_abi_plan():
      raise ValueError("kernel_abi must exactly describe the retained Q4_K/Q8_1 five-buffer format, slots, dtypes, layouts, geometry, and emitter family")

  if "operand_sources" in payload:
    sources = payload["operand_sources"]
    _strict_keys(sources, {"a", "b"}, "operand_sources")
    dense_keys = {"kind", "logical_dtype", "storage_dtype", "abi_slot"}
    _strict_keys(sources["a"], dense_keys, "operand_sources.a")
    if sources["a"] != {"kind":"dense", "logical_dtype":"fp16", "storage_dtype":"fp16", "abi_slot":1}:
      raise ValueError("operand_sources.a must be dense fp16 at ABI slot 1")
    b = sources["b"]
    if not isinstance(b, dict): raise ValueError("operand_sources.b must be an object")
    if b.get("kind") == "dense":
      _strict_keys(b, dense_keys, "operand_sources.b")
      if b != {"kind":"dense", "logical_dtype":"fp16", "storage_dtype":"fp16", "abi_slot":2}:
        raise ValueError("dense operand_sources.b must be logical/storage fp16 at ABI slot 2")
    elif b.get("kind") == "packed_scalar_decoder":
      packed_keys = dense_keys | {"quant_format", "rows", "k", "block_elems", "block_bytes", "decoder_version"}
      _strict_keys(b, packed_keys, "operand_sources.b")
      if b["logical_dtype"] != "fp16": raise ValueError("packed operand_sources.b logical_dtype must be fp16")
      if b["abi_slot"] != 2: raise ValueError("packed operand_sources.b must use ABI slot 2")
      if b["decoder_version"] != PACKED_SCALAR_DECODER_VERSION:
        raise ValueError(f"packed operand_sources.b decoder_version must be {PACKED_SCALAR_DECODER_VERSION!r}")
      try:
        from tinygrad.codegen.opt.packed_weight import PackedWeightTransform
        transform = PackedWeightTransform(b["quant_format"], b["rows"], b["k"], b["block_elems"], b["block_bytes"])
      except (TypeError, ValueError) as exc: raise ValueError(f"invalid packed operand_sources.b: {exc}") from exc
      expected_storage_dtype = "uint32" if transform.quant_format == "Q4_K" else "uint16"
      if b["storage_dtype"] != expected_storage_dtype:
        raise ValueError(f"packed operand_sources.b storage_dtype must be {expected_storage_dtype}")
      if workload["dtypes"]["b"] != "fp16":
        raise ValueError("packed operand_sources.b logical fp16 dtype must match workload.dtypes.b")
      if (transform.rows, transform.k) != (workload["shape"]["n"], workload["shape"]["k"]):
        raise ValueError("packed operand_sources.b rows/k must exactly match workload N/K")
    else: raise ValueError("operand_sources.b kind must be dense or packed_scalar_decoder")

  if direct_five_buffer:
    if workload["dtypes"] != {"a":"Q8_1","b":"Q4_K","c":"fp32","accumulator":"int32_fp32"}:
      raise ValueError("five-buffer workload dtypes must describe Q8_1, Q4_K, fp32 output, and int32/fp32 accumulation")
    if workload["layout"] != {"a":"physical_ds4","b":"q4_k_packed_words","c":"tokens_rows"}:
      raise ValueError("five-buffer workload layouts must describe physical DS4, packed Q4_K words, and token-row output")
    if payload["schedule"] != _q4k_q8_1_direct_emitter_schedule():
      raise ValueError("five-buffer schedule must exactly describe the direct-global wave32 signed-i8 WMMA emitter with zero LDS and no tails")
    if payload["static_constraints"] != {"max_lds_bytes":0,"max_vgpr_per_thread":256,"allow_spill":False}:
      raise ValueError("five-buffer resources must require zero LDS, bounded VGPRs, and no spills")
    applicability = payload["applicability"]
    _strict_keys(applicability, {"exact_shape", "profiles", "roles", "targets"}, "applicability")
    if applicability["exact_shape"] is not True: raise ValueError("full-kernel applicability.exact_shape must be true")
    for key in ("profiles", "roles", "targets"):
      values = applicability[key]
      if not isinstance(values, list) or not values or any(not isinstance(x, str) or not x for x in values):
        raise ValueError(f"applicability.{key} must be a non-empty list of strings")
    return

  schedule = payload["schedule"]
  schedule_groups = {"tile", "waves", "threads", "lane_ownership", "cooperative_load", "lds", "pipeline", "wmma",
                     "dependency_policy", "residency", "epilogue", "numerical_mode"}
  _strict_keys(schedule, schedule_groups, "schedule")
  _strict_keys(schedule["tile"], {"m", "n", "k"}, "schedule.tile")
  _strict_keys(schedule["waves"], {"m", "n"}, "schedule.waves")
  for group in ("tile", "waves"):
    for key, value in schedule[group].items(): _positive_int(value, f"schedule.{group}.{key}")
  _positive_int(schedule["threads"], "schedule.threads")
  _nonempty_str(schedule["lane_ownership"], "schedule.lane_ownership")
  _nonempty_str(schedule["numerical_mode"], "schedule.numerical_mode")
  _strict_keys(schedule["cooperative_load"], {"a", "b"}, "schedule.cooperative_load")
  for operand in ("a", "b"):
    load = schedule["cooperative_load"][operand]
    _strict_keys(load, {"lane_mapping", "vector_width", "alignment"}, f"schedule.cooperative_load.{operand}")
    _nonempty_str(load["lane_mapping"], f"schedule.cooperative_load.{operand}.lane_mapping")
    for key in ("vector_width", "alignment"): _positive_int(load[key], f"schedule.cooperative_load.{operand}.{key}")
  nested = {"lds": {"windows", "strides", "padding", "banks", "store_vector_width", "load_vector_width"},
            "pipeline": {"buffer_count", "stage_count", "epoch_graph"},
            "wmma": {"instruction_family", "fragment_layout", "accumulator_ownership"},
            "dependency_policy": {"waitcnt", "barriers"}, "residency": {"preload", "resident", "reuse"},
            "epilogue": {"lane_mapping", "vector_width"}}
  for name, keys in nested.items(): _strict_keys(schedule[name], keys, f"schedule.{name}")
  for group in ("lds", "pipeline", "wmma", "dependency_policy", "residency", "epilogue"):
    for key, value in schedule[group].items():
      if key in {"buffer_count", "stage_count", "vector_width", "padding", "banks", "store_vector_width", "load_vector_width"}:
        _positive_int(value, f"schedule.{group}.{key}")
      elif key in {"windows", "strides", "epoch_graph", "waitcnt", "barriers", "preload", "resident", "reuse"}:
        if not isinstance(value, (dict, list)): raise ValueError(f"schedule.{group}.{key} must be an object or list")
      else: _nonempty_str(value, f"schedule.{group}.{key}")

  constraints = payload["static_constraints"]
  _strict_keys(constraints, {"max_lds_bytes", "max_vgpr_per_thread", "allow_spill"}, "static_constraints")
  for key in ("max_lds_bytes", "max_vgpr_per_thread"): _positive_int(constraints[key], f"static_constraints.{key}")
  if not isinstance(constraints["allow_spill"], bool): raise ValueError("static_constraints.allow_spill must be bool")
  applicability = payload["applicability"]
  _strict_keys(applicability, {"exact_shape", "profiles", "roles", "targets"}, "applicability")
  if applicability["exact_shape"] is not True: raise ValueError("full-kernel applicability.exact_shape must be true")
  for key in ("profiles", "roles", "targets"):
    values = applicability[key]
    if not isinstance(values, list) or not values or any(not isinstance(x, str) or not x for x in values):
      raise ValueError(f"applicability.{key} must be a non-empty list of strings")


@dataclass(frozen=True)
class QuantizedTensorSpec:
  format: str
  block_size: int | None = None
  group_size: int | None = None
  scale_layout: str = ""
  min_layout: str = ""
  signed: bool | None = None

  def __post_init__(self):
    _check("format", self.format, QUANT_FORMATS)

  def to_json(self) -> dict[str, Any]:
    return {"format": self.format, "block_size": self.block_size, "group_size": self.group_size,
            "scale_layout": self.scale_layout, "min_layout": self.min_layout, "signed": self.signed}

  @classmethod
  def from_json(cls, row:dict[str, Any]) -> "QuantizedTensorSpec":
    return cls(format=str(row.get("format", "unknown")), block_size=row.get("block_size"),
               group_size=row.get("group_size"), scale_layout=str(row.get("scale_layout", "")),
               min_layout=str(row.get("min_layout", "")), signed=row.get("signed"))


@dataclass(frozen=True)
class ActivationQuantSpec:
  format: str
  block_size: int | None = None
  signed: bool | None = None
  scale_layout: str = ""

  def __post_init__(self):
    _check("format", self.format, ACTIVATION_FORMATS)

  def to_json(self) -> dict[str, Any]:
    return {"format": self.format, "block_size": self.block_size, "signed": self.signed, "scale_layout": self.scale_layout}

  @classmethod
  def from_json(cls, row:dict[str, Any]) -> "ActivationQuantSpec":
    return cls(format=str(row.get("format", "none")), block_size=row.get("block_size"),
               signed=row.get("signed"), scale_layout=str(row.get("scale_layout", "")))


@dataclass(frozen=True)
class CandidateAdmissionFacts:
  """Runtime facts used only to rank/admit generated primitives, never to bind them."""
  memory_budget_bytes: int | None = None
  dequant_buffer_bytes: int | None = None
  scheduler_owned: bool = False
  dequant_once_admitted: bool = False
  fused_wmma_admitted: bool = False

  def __post_init__(self):
    for name in ("memory_budget_bytes", "dequant_buffer_bytes"):
      value = getattr(self, name)
      if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
        raise ValueError(f"{name} must be a non-negative integer or None")

  @property
  def dequant_buffer_fits(self) -> bool:
    return (self.memory_budget_bytes is not None and self.dequant_buffer_bytes is not None and
            self.dequant_buffer_bytes <= self.memory_budget_bytes)

  def to_json(self) -> dict[str, Any]:
    return {"memory_budget_bytes": self.memory_budget_bytes, "dequant_buffer_bytes": self.dequant_buffer_bytes,
            "scheduler_owned": self.scheduler_owned, "dequant_once_admitted": self.dequant_once_admitted,
            "fused_wmma_admitted": self.fused_wmma_admitted}

  @classmethod
  def from_json(cls, row:dict[str, Any]) -> "CandidateAdmissionFacts": return cls(**row)


@dataclass(frozen=True)
class RuntimeOpSpec:
  family: str
  phase: str
  role: str
  shape: dict[str, int | str]
  weight: QuantizedTensorSpec
  activation: ActivationQuantSpec = field(default_factory=lambda: ActivationQuantSpec("fp16"))
  lowering_strategy: str = "unknown"
  device: str = "unknown"
  route_id: str = ""
  codegen_features: tuple[str, ...] = ()
  profile: str = ""
  target: dict[str, Any] = field(default_factory=dict)
  admission: CandidateAdmissionFacts = field(default_factory=CandidateAdmissionFacts)

  def __post_init__(self):
    _check("family", self.family, OP_FAMILIES)
    _check("phase", self.phase, PHASES)
    _check("role", self.role, ROLES)
    _check("lowering_strategy", self.lowering_strategy, LOWERING_STRATEGIES)
    _shape_json(self.shape)

  def to_json(self) -> dict[str, Any]:
    return {"family": self.family, "phase": self.phase, "role": self.role, "shape": _shape_json(self.shape),
            "weight": self.weight.to_json(), "activation": self.activation.to_json(),
            "lowering_strategy": self.lowering_strategy, "device": self.device, "route_id": self.route_id,
            "codegen_features": list(self.codegen_features), "profile": self.profile, "target": dict(self.target),
            "admission": self.admission.to_json()}

  @classmethod
  def from_json(cls, row:dict[str, Any]) -> "RuntimeOpSpec":
    return cls(family=str(row["family"]), phase=str(row["phase"]), role=str(row.get("role", "unknown")),
               shape=dict(row.get("shape", {})), weight=QuantizedTensorSpec.from_json(dict(row["weight"])),
               activation=ActivationQuantSpec.from_json(dict(row.get("activation", {"format": "none"}))),
               lowering_strategy=str(row.get("lowering_strategy", "unknown")),
               device=str(row.get("device", "unknown")), route_id=str(row.get("route_id", "")),
               codegen_features=tuple(str(x) for x in row.get("codegen_features", ())),
               profile=str(row.get("profile", "")), target=dict(row.get("target", {})),
               admission=CandidateAdmissionFacts.from_json(dict(row.get("admission", {}))))


@dataclass(frozen=True)
class GeneratedCandidate:
  candidate_id: str
  op_family: str
  supported_quant_formats: tuple[str, ...]
  supported_activation_formats: tuple[str, ...]
  phases: tuple[str, ...]
  roles: tuple[str, ...]
  lowering_strategy: str
  provenance: str
  route_id: str = ""
  shape_constraints: tuple[dict[str, Any], ...] = ()
  device_constraints: tuple[str, ...] = ()
  target_constraints: tuple[dict[str, Any], ...] = ()
  required_codegen_features: tuple[str, ...] = ()
  search_space_id: str = ""
  rollback_behavior: dict[str, str] = field(default_factory=dict)
  authority_gates: tuple[str, ...] = ()
  full_kernel_candidate: dict[str, Any] | None = None
  candidate_class: str = "performance"
  lifecycle: str = "candidate"
  priority: int = 100
  required_admission_facts: tuple[str, ...] = ()

  def __post_init__(self):
    _check("op_family", self.op_family, OP_FAMILIES)
    _check("lowering_strategy", self.lowering_strategy, LOWERING_STRATEGIES)
    _check("provenance", self.provenance, PROVENANCE)
    for q in self.supported_quant_formats: _check("supported_quant_format", q, QUANT_FORMATS)
    for a in self.supported_activation_formats: _check("supported_activation_format", a, ACTIVATION_FORMATS)
    for p in self.phases: _check("phase", p, PHASES)
    for r in self.roles: _check("role", r, ROLES)
    if self.candidate_class not in ("performance", "rollback"):
      raise ValueError("candidate_class must be 'performance' or 'rollback'")
    if self.lifecycle not in ("diagnostic", "candidate", "shipped", "refuted", "deferred"):
      raise ValueError("lifecycle must be diagnostic, candidate, shipped, refuted, or deferred")
    if not isinstance(self.priority, int) or isinstance(self.priority, bool): raise ValueError("priority must be an integer")
    known_facts = {"scheduler_owned", "dequant_once_admitted", "dequant_buffer_fits", "fused_wmma_admitted"}
    if unknown := set(self.required_admission_facts) - known_facts:
      raise ValueError(f"unknown required admission facts: {sorted(unknown)!r}")
    if self.full_kernel_candidate is not None:
      try: payload = json.loads(json.dumps(self.full_kernel_candidate, allow_nan=False))
      except (TypeError, ValueError) as exc: raise ValueError(f"full_kernel_candidate must be JSON data: {exc}") from exc
      _validate_full_kernel_payload(payload)
      object.__setattr__(self, "full_kernel_candidate", payload)

  @property
  def is_generated_only(self) -> bool:
    return self.provenance in GENERATED_PROVENANCE

  @property
  def is_full_kernel_candidate(self) -> bool:
    return self.full_kernel_candidate is not None

  def _registry_json(self) -> dict[str, Any]:
    return ({"candidate_id": self.candidate_id, "op_family": self.op_family,
            "supported_quant_formats": list(self.supported_quant_formats),
            "supported_activation_formats": list(self.supported_activation_formats), "phases": list(self.phases),
            "roles": list(self.roles), "lowering_strategy": self.lowering_strategy, "provenance": self.provenance,
            "route_id": self.route_id, "shape_constraints": list(self.shape_constraints),
            "device_constraints": list(self.device_constraints),
            "target_constraints": list(self.target_constraints),
            "required_codegen_features": list(self.required_codegen_features), "search_space_id": self.search_space_id,
            "rollback_behavior": dict(self.rollback_behavior), "authority_gates": list(self.authority_gates)}
            | {"candidate_class": self.candidate_class, "lifecycle": self.lifecycle, "priority": self.priority,
               "required_admission_facts": list(self.required_admission_facts)})

  @property
  def canonical_identity(self) -> str:
    if not self.is_full_kernel_candidate: return ""
    return _canonical_full_kernel_identity(self.full_kernel_candidate)

  @property
  def legacy_identity_alias(self) -> str:
    if not self.is_full_kernel_candidate: return ""
    return _legacy_full_kernel_identity(self.full_kernel_candidate)

  def kernel_candidate_context(self):
    if not self.is_full_kernel_candidate: raise ValueError("legacy candidate has no full-kernel candidate context")
    from tinygrad.uop.ops import KernelCandidateContext
    assert self.full_kernel_candidate is not None
    return KernelCandidateContext(self.full_kernel_candidate["schema_version"], self.canonical_identity)

  def supports(self, op:RuntimeOpSpec) -> bool:
    if self.op_family != op.family: return False
    if op.phase not in self.phases: return False
    if op.role not in self.roles and "unknown" not in self.roles: return False
    if op.weight.format not in self.supported_quant_formats: return False
    if op.activation.format not in self.supported_activation_formats: return False
    if self.device_constraints and op.device not in self.device_constraints: return False
    if self.target_constraints and not any(all(op.target.get(k) == v for k, v in target.items())
                                           for target in self.target_constraints): return False
    if self.shape_constraints and not any(all(constraint.get(dim, "*") == "*" or op.shape.get(dim) == constraint[dim]
                                              for dim in ("M", "N", "K"))
                                          for constraint in self.shape_constraints): return False
    if any(not bool(getattr(op.admission, fact)) for fact in self.required_admission_facts): return False
    if self.is_full_kernel_candidate:
      assert self.full_kernel_candidate is not None
      workload, applicability = self.full_kernel_candidate["workload"], self.full_kernel_candidate["applicability"]
      try: op_shape = tuple(op.shape[k] for k in ("M", "N", "K"))
      except KeyError: return False
      shape = workload["shape"]
      if op_shape != (shape["m"], shape["n"], shape["k"]): return False
      if op.role != workload["role"] or op.role not in applicability["roles"]: return False
      if op.target != workload["target"]: return False
      target = workload["target"]
      if f"{target['backend']}:{target['arch']}:wave{target['wave_size']}" not in applicability["targets"]: return False
      required_feature = self.full_kernel_candidate["schedule"]["wmma"]["instruction_family"]
      if required_feature not in op.codegen_features: return False
    return self.lowering_strategy == op.lowering_strategy or op.lowering_strategy == "unknown"

  def to_json(self) -> dict[str, Any]:
    row = self._registry_json()
    if self.is_full_kernel_candidate:
      row["full_kernel_candidate"] = json.loads(json.dumps(self.full_kernel_candidate))
      row["canonical_identity"] = self.canonical_identity
    return row

  @classmethod
  def from_json(cls, row:dict[str, Any]) -> "GeneratedCandidate":
    candidate = cls(candidate_id=str(row["candidate_id"]), op_family=str(row["op_family"]),
               supported_quant_formats=tuple(row.get("supported_quant_formats", ())),
               supported_activation_formats=tuple(row.get("supported_activation_formats", ())),
               phases=tuple(row.get("phases", ())), roles=tuple(row.get("roles", ())),
               lowering_strategy=str(row["lowering_strategy"]), provenance=str(row.get("provenance", "unknown")),
               route_id=str(row.get("route_id", "")),
               shape_constraints=tuple(dict(x) for x in row.get("shape_constraints", ())),
               device_constraints=tuple(row.get("device_constraints", ())),
               target_constraints=tuple(dict(x) for x in row.get("target_constraints", ())),
               required_codegen_features=tuple(row.get("required_codegen_features", ())),
               search_space_id=str(row.get("search_space_id", "")),
               rollback_behavior=dict(row.get("rollback_behavior", {})),
               authority_gates=tuple(row.get("authority_gates", ())),
               full_kernel_candidate=None if row.get("full_kernel_candidate") is None else dict(row["full_kernel_candidate"]),
               candidate_class=str(row.get("candidate_class", "performance")), lifecycle=str(row.get("lifecycle", "candidate")),
               priority=int(row.get("priority", 100)),
               required_admission_facts=tuple(row.get("required_admission_facts", ())))
    if candidate.is_full_kernel_candidate:
      identity = row.get("canonical_identity")
      aliases = (candidate.canonical_identity, _legacy_full_kernel_identity(candidate.full_kernel_candidate))
      if not isinstance(identity, str) or identity not in aliases:
        raise ValueError("strict full-kernel candidate canonical_identity is missing or does not match canonical payload")
    return candidate
