from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

OP_FAMILIES = ("QuantizedLinear", "DenseLinear", "FlashAttention", "KVCache", "ActivationFusion")
PHASES = ("prefill", "decode")
ROLES = ("ffn_gate_up", "ffn_down", "attn_qo", "attn_kv", "lm_head", "attention", "unknown")
QUANT_FORMATS = ("Q4_K", "Q6_K", "fp16", "fp8", "int8", "unknown")
ACTIVATION_FORMATS = ("fp16", "fp32", "Q8_1", "none")
LOWERING_STRATEGIES = (
  "packed_dequant_dot", "grouped_int_dot_correction", "iu8_wmma_grouped_dot", "iu8_wmma_tiled_grouped_dot",
  "online_softmax_flash", "tinygrad_scheduler", "unknown",
)
PROVENANCE = ("machine_authored_generated", "tinygrad_scheduler_generated", "banned", "unknown")
GENERATED_PROVENANCE = ("machine_authored_generated", "tinygrad_scheduler_generated")
FULL_KERNEL_CANDIDATE_SCHEMA = "boltbeam.full_kernel_candidate.v1"
ANCHOR_SINGLE_BUFFER_CANDIDATE_HASH = "81c27275d1aad1bb8147c5c5cdaa8000e9375e81f3d085b49d62064a731313d6"

class FullKernelAdmissionError(ValueError):
  def __init__(self, code:str, message:str): self.code = code; super().__init__(f"{code}: {message}")

@dataclass(frozen=True)
class FullKernelCapability:
  backend: str = "AMD"
  arch: str = "gfx1100"
  wave_size: int = 32
  max_lds_bytes: int = 65536
  buffer_count: int = 1
  stage_count: int = 1
  vector_bytes: int = 16
  instruction_family: str = "wmma_f32_16x16x16_f16"
  fragment_layout: str = "rdna3_wmma_f32_16x16x16_f16_lds2_static"

GFX1100_SINGLE_BUFFER_CAPABILITY = FullKernelCapability()

@dataclass(frozen=True)
class FullKernelAdmission:
  canonical_identity: str
  normalized_payload: dict[str, Any]
  geometry: Any
  plan: Any
  active_lds_bytes: int
  capability: FullKernelCapability
  context: Any

def admit_full_kernel_candidate(payload:dict[str, Any], canonical_identity:str, *, profile:str, role:str,
                                shape:tuple[int,int,int], target:dict[str,Any],
                                capability:FullKernelCapability=GFX1100_SINGLE_BUFFER_CAPABILITY) -> FullKernelAdmission:
  try: normalized = json.loads(json.dumps(payload, allow_nan=False))
  except (TypeError, ValueError) as exc: raise FullKernelAdmissionError("payload_json", str(exc)) from exc
  try: _validate_full_kernel_payload(normalized)
  except ValueError as exc: raise FullKernelAdmissionError("payload_schema", str(exc)) from exc
  encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
  actual_identity = hashlib.sha256(encoded).hexdigest()
  if canonical_identity != actual_identity: raise FullKernelAdmissionError("identity_mismatch", "canonical SHA-256 differs from payload")
  workload,schedule,applicability = normalized["workload"],normalized["schedule"],normalized["applicability"]
  target_id = f"{target['backend']}:{target['arch']}:wave{target['wave_size']}"
  if workload["profile"] != profile or profile not in applicability["profiles"]: raise FullKernelAdmissionError("workload_profile", "profile is not exact/applicable")
  if workload["role"] != role or role not in applicability["roles"]: raise FullKernelAdmissionError("workload_role", "role is not exact/applicable")
  if tuple(workload["shape"][x] for x in ("m","n","k")) != shape or not applicability["exact_shape"]:
    raise FullKernelAdmissionError("workload_shape", "shape is not exact")
  if workload["target"] != target or target_id not in applicability["targets"]: raise FullKernelAdmissionError("workload_target", "target is not exact/applicable")
  if target != {"backend":capability.backend,"arch":capability.arch,"wave_size":capability.wave_size}:
    raise FullKernelAdmissionError("capability_target", "target is outside frozen capability")
  if workload["dtypes"] != {"a":"fp16","b":"fp16","c":"fp16","accumulator":"fp32"}:
    raise FullKernelAdmissionError("capability_dtype", "only fp16/fp32 accumulation is supported")
  if schedule["pipeline"]["buffer_count"] != capability.buffer_count or schedule["pipeline"]["stage_count"] != capability.stage_count:
    raise FullKernelAdmissionError("capability_pipeline", "only single-buffer stage1 is supported")
  if schedule["wmma"]["instruction_family"] != capability.instruction_family or schedule["wmma"]["fragment_layout"] != capability.fragment_layout:
    raise FullKernelAdmissionError("capability_tc", "tensor-core descriptor is unsupported")
  if any(schedule["lds"][x] != 8 for x in ("store_vector_width","load_vector_width")) or \
     any(schedule["cooperative_load"][r]["vector_width"]*2 != capability.vector_bytes or
         schedule["cooperative_load"][r]["alignment"] != capability.vector_bytes for r in ("a","b")):
    raise FullKernelAdmissionError("capability_vector", "only aligned b128 fp16 transport is supported")
  if any(schedule["cooperative_load"][r]["lane_mapping"] != "cooperative_row_stride_64_b128" for r in ("a","b")):
    raise FullKernelAdmissionError("capability_lane_map", "cooperative lane mapping is unsupported")
  from tinygrad.uop.ops import KernelCandidateContext, KernelLDSWindow, KernelTileGeometry
  try: geometry = KernelTileGeometry(tuple(schedule["tile"][x] for x in ("m","n","k")),
    tuple(schedule["waves"][x] for x in ("m","n")),schedule["threads"],target["wave_size"],
    tuple(KernelLDSWindow(r.upper(),*schedule["lds"]["windows"][r],schedule["lds"]["strides"][r]) for r in ("a","b")))
  except ValueError as exc: raise FullKernelAdmissionError("geometry_invalid", str(exc)) from exc
  if any(shape[i] % geometry.tile[i] for i in range(3)): raise FullKernelAdmissionError("geometry_divisibility", "workload is not tile divisible")
  active_lds = geometry.lds_windows[-1].end
  if active_lds > capability.max_lds_bytes or active_lds > normalized["static_constraints"]["max_lds_bytes"]:
    raise FullKernelAdmissionError("capability_lds", "active LDS exceeds a declared limit")
  try:
    from tinygrad.codegen.opt.kernel_lds import derive_precontract_factors
    from tinygrad.codegen.opt.tc import amd_rdna3
    from tinygrad.dtype import dtypes
    tc = next(x for x in amd_rdna3 if x.dtype_in == dtypes.half and x.dtype_out == dtypes.float)
    plan = derive_precontract_factors(geometry, tc)
  except ValueError as exc: raise FullKernelAdmissionError("capability_geometry", str(exc)) from exc
  context = KernelCandidateContext(normalized["schema_version"],actual_identity,geometry)
  return FullKernelAdmission(actual_identity,normalized,geometry,plan,active_lds,capability,context)


def bind_full_kernel_candidate(payload:dict[str, Any], canonical_identity:str, *, profile:str, role:str,
                               shape:tuple[int, int, int], target:dict[str, Any], tile:tuple[int, int, int]|None=None,
                               waves:tuple[int, int]|None=None, threads:int|None=None, buffer_count:int|None=None,
                               stage_count:int|None=None, lds_windows:dict[str, list[int]]|None=None,
                               lds_strides:dict[str, int]|None=None, lds_padding:int|None=None, lds_bytes:int|None=None):
  """Compatibility wrapper. Schedule authority comes exclusively from the canonical payload."""
  return admit_full_kernel_candidate(payload,canonical_identity,profile=profile,role=role,shape=shape,target=target).context


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
  _strict_keys(payload, {"schema_version", "workload", "schedule", "static_constraints", "applicability"},
               "full_kernel_candidate")
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
            "codegen_features": list(self.codegen_features), "profile": self.profile, "target": dict(self.target)}

  @classmethod
  def from_json(cls, row:dict[str, Any]) -> "RuntimeOpSpec":
    return cls(family=str(row["family"]), phase=str(row["phase"]), role=str(row.get("role", "unknown")),
               shape=dict(row.get("shape", {})), weight=QuantizedTensorSpec.from_json(dict(row["weight"])),
               activation=ActivationQuantSpec.from_json(dict(row.get("activation", {"format": "none"}))),
               lowering_strategy=str(row.get("lowering_strategy", "unknown")),
               device=str(row.get("device", "unknown")), route_id=str(row.get("route_id", "")),
               codegen_features=tuple(str(x) for x in row.get("codegen_features", ())),
               profile=str(row.get("profile", "")), target=dict(row.get("target", {})))


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
  required_codegen_features: tuple[str, ...] = ()
  search_space_id: str = ""
  rollback_behavior: dict[str, str] = field(default_factory=dict)
  authority_gates: tuple[str, ...] = ()
  full_kernel_candidate: dict[str, Any] | None = None

  def __post_init__(self):
    _check("op_family", self.op_family, OP_FAMILIES)
    _check("lowering_strategy", self.lowering_strategy, LOWERING_STRATEGIES)
    _check("provenance", self.provenance, PROVENANCE)
    for q in self.supported_quant_formats: _check("supported_quant_format", q, QUANT_FORMATS)
    for a in self.supported_activation_formats: _check("supported_activation_format", a, ACTIVATION_FORMATS)
    for p in self.phases: _check("phase", p, PHASES)
    for r in self.roles: _check("role", r, ROLES)
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
    return {"candidate_id": self.candidate_id, "op_family": self.op_family,
            "supported_quant_formats": list(self.supported_quant_formats),
            "supported_activation_formats": list(self.supported_activation_formats), "phases": list(self.phases),
            "roles": list(self.roles), "lowering_strategy": self.lowering_strategy, "provenance": self.provenance,
            "route_id": self.route_id, "shape_constraints": list(self.shape_constraints),
            "device_constraints": list(self.device_constraints),
            "required_codegen_features": list(self.required_codegen_features), "search_space_id": self.search_space_id,
            "rollback_behavior": dict(self.rollback_behavior), "authority_gates": list(self.authority_gates)}

  @property
  def canonical_identity(self) -> str:
    if not self.is_full_kernel_candidate: return ""
    encoded = json.dumps(self.full_kernel_candidate, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()

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
    if self.is_full_kernel_candidate:
      assert self.full_kernel_candidate is not None
      workload, applicability = self.full_kernel_candidate["workload"], self.full_kernel_candidate["applicability"]
      try: op_shape = tuple(op.shape[k] for k in ("M", "N", "K"))
      except KeyError: return False
      shape = workload["shape"]
      if op_shape != (shape["m"], shape["n"], shape["k"]): return False
      if op.profile != workload["profile"] or op.profile not in applicability["profiles"]: return False
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
               required_codegen_features=tuple(row.get("required_codegen_features", ())),
               search_space_id=str(row.get("search_space_id", "")),
               rollback_behavior=dict(row.get("rollback_behavior", {})),
               authority_gates=tuple(row.get("authority_gates", ())),
               full_kernel_candidate=None if row.get("full_kernel_candidate") is None else dict(row["full_kernel_candidate"]))
    if candidate.is_full_kernel_candidate:
      identity = row.get("canonical_identity")
      if not isinstance(identity, str) or identity != candidate.canonical_identity:
        raise ValueError("strict full-kernel candidate canonical_identity is missing or does not match canonical payload")
    return candidate
