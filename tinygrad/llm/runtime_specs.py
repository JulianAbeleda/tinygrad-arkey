from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

OP_FAMILIES = ("QuantizedLinear", "DenseLinear", "FlashAttention", "KVCache", "ActivationFusion")
PHASES = ("prefill", "decode")
ROLES = ("ffn_gate_up", "ffn_down", "attn_qo", "attn_kv", "lm_head", "attention", "unknown")
QUANT_FORMATS = ("Q4_K", "Q6_K", "fp16", "fp8", "int8", "unknown")
ACTIVATION_FORMATS = ("fp16", "fp32", "Q8_1", "none")
LOWERING_STRATEGIES = (
  "packed_dequant_dot", "grouped_int_dot_correction", "iu8_wmma_grouped_dot",
  "online_softmax_flash", "tinygrad_scheduler", "unknown",
)
PROVENANCE = ("machine_authored_generated", "tinygrad_scheduler_generated", "banned", "unknown")
GENERATED_PROVENANCE = ("machine_authored_generated", "tinygrad_scheduler_generated")


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

  def __post_init__(self):
    _check("family", self.family, OP_FAMILIES)
    _check("phase", self.phase, PHASES)
    _check("role", self.role, ROLES)
    _check("lowering_strategy", self.lowering_strategy, LOWERING_STRATEGIES)
    _shape_json(self.shape)

  def to_json(self) -> dict[str, Any]:
    return {"family": self.family, "phase": self.phase, "role": self.role, "shape": _shape_json(self.shape),
            "weight": self.weight.to_json(), "activation": self.activation.to_json(),
            "lowering_strategy": self.lowering_strategy, "device": self.device, "route_id": self.route_id}

  @classmethod
  def from_json(cls, row:dict[str, Any]) -> "RuntimeOpSpec":
    return cls(family=str(row["family"]), phase=str(row["phase"]), role=str(row.get("role", "unknown")),
               shape=dict(row.get("shape", {})), weight=QuantizedTensorSpec.from_json(dict(row["weight"])),
               activation=ActivationQuantSpec.from_json(dict(row.get("activation", {"format": "none"}))),
               lowering_strategy=str(row.get("lowering_strategy", "unknown")),
               device=str(row.get("device", "unknown")), route_id=str(row.get("route_id", "")))


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

  def __post_init__(self):
    _check("op_family", self.op_family, OP_FAMILIES)
    _check("lowering_strategy", self.lowering_strategy, LOWERING_STRATEGIES)
    _check("provenance", self.provenance, PROVENANCE)
    for q in self.supported_quant_formats: _check("supported_quant_format", q, QUANT_FORMATS)
    for a in self.supported_activation_formats: _check("supported_activation_format", a, ACTIVATION_FORMATS)
    for p in self.phases: _check("phase", p, PHASES)
    for r in self.roles: _check("role", r, ROLES)

  @property
  def is_generated_only(self) -> bool:
    return self.provenance in GENERATED_PROVENANCE

  def supports(self, op:RuntimeOpSpec) -> bool:
    if self.op_family != op.family: return False
    if op.phase not in self.phases: return False
    if op.role not in self.roles and "unknown" not in self.roles: return False
    if op.weight.format not in self.supported_quant_formats: return False
    if op.activation.format not in self.supported_activation_formats: return False
    return self.lowering_strategy == op.lowering_strategy or op.lowering_strategy == "unknown"

  def to_json(self) -> dict[str, Any]:
    return {"candidate_id": self.candidate_id, "op_family": self.op_family,
            "supported_quant_formats": list(self.supported_quant_formats),
            "supported_activation_formats": list(self.supported_activation_formats), "phases": list(self.phases),
            "roles": list(self.roles), "lowering_strategy": self.lowering_strategy, "provenance": self.provenance,
            "route_id": self.route_id, "shape_constraints": list(self.shape_constraints),
            "device_constraints": list(self.device_constraints),
            "required_codegen_features": list(self.required_codegen_features), "search_space_id": self.search_space_id,
            "rollback_behavior": dict(self.rollback_behavior), "authority_gates": list(self.authority_gates)}

  @classmethod
  def from_json(cls, row:dict[str, Any]) -> "GeneratedCandidate":
    return cls(candidate_id=str(row["candidate_id"]), op_family=str(row["op_family"]),
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
               authority_gates=tuple(row.get("authority_gates", ())))
