"""Hardware-independent, serializable execution-bridge contracts."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import hashlib, json
from typing import Any, ClassVar, Mapping


def _text(value: Any, name: str) -> str:
  if not isinstance(value, str) or not value.strip(): raise ValueError(f"{name} must be a non-empty string")
  return value

def _mapping(value: Any, name: str) -> dict[str, Any]:
  if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value): raise ValueError(f"{name} must be a string-keyed mapping")
  try: json.dumps(value, sort_keys=True, separators=(",", ":"))
  except (TypeError, ValueError) as exc: raise ValueError(f"{name} must be JSON serializable") from exc
  return dict(value)


class Contract:
  schema: ClassVar[str]
  def to_dict(self) -> dict[str, Any]: return {"schema": self.schema, **{f.name: getattr(self, f.name) for f in fields(self)}}
  def to_json(self) -> str: return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
  @property
  def digest(self) -> str: return hashlib.sha256(self.to_json().encode()).hexdigest()


@dataclass(frozen=True)
class WorkloadIdentity(Contract):
  schema: ClassVar[str] = "execution_bridge.workload_identity.v1"
  workload_id: str; role: str; shape: tuple[int, ...]; dtypes: tuple[str, ...]; layout: Mapping[str, Any] = field(default_factory=dict)
  def __post_init__(self):
    _text(self.workload_id, "workload_id"); _text(self.role, "role")
    if not self.shape or any(not isinstance(x, int) or x <= 0 for x in self.shape): raise ValueError("shape must contain positive integers")
    if not self.dtypes or any(not isinstance(x, str) or not x for x in self.dtypes): raise ValueError("dtypes must be non-empty")
    _mapping(self.layout, "layout")


@dataclass(frozen=True)
class SemanticScheduleIdentity(Contract):
  schema: ClassVar[str] = "execution_bridge.semantic_schedule_identity.v1"
  workload_digest: str; schedule_digest: str; buffer_roles: tuple[str, ...]; abi_digest: str
  def __post_init__(self):
    for name in ("workload_digest", "schedule_digest", "abi_digest"): _text(getattr(self, name), name)
    if not self.buffer_roles or any(not isinstance(x, str) or not x for x in self.buffer_roles): raise ValueError("buffer_roles must be non-empty")


@dataclass(frozen=True)
class TransportPlan(Contract):
  schema: ClassVar[str] = "execution_bridge.transport_plan.v1"
  transport: str; schedule_digest: str; requirements: Mapping[str, Any] = field(default_factory=dict); fallback_eligible: bool = False
  def __post_init__(self):
    _text(self.transport, "transport"); _text(self.schedule_digest, "schedule_digest"); _mapping(self.requirements, "requirements")
    if not isinstance(self.fallback_eligible, bool): raise ValueError("fallback_eligible must be bool")


@dataclass(frozen=True)
class CompileArtifactMetadata(Contract):
  schema: ClassVar[str] = "execution_bridge.compile_artifact.v1"
  candidate_digest: str; schedule_digest: str; transport: str; target: str; abi_digest: str; binary_sha256: str; source_sha256: str
  resources: Mapping[str, Any] = field(default_factory=dict); dispatch_allowed: bool = False
  def __post_init__(self):
    for name in ("candidate_digest", "schedule_digest", "transport", "target", "abi_digest", "binary_sha256", "source_sha256"): _text(getattr(self, name), name)
    _mapping(self.resources, "resources")
    if not isinstance(self.dispatch_allowed, bool): raise ValueError("dispatch_allowed must be bool")


@dataclass(frozen=True)
class ExecutableArtifactMetadata(Contract):
  schema: ClassVar[str] = "execution_bridge.executable_artifact.v1"
  compile_digest: str; binary_sha256: str; runtime_cache_key: str; target: str
  def __post_init__(self):
    for name in ("compile_digest", "binary_sha256", "runtime_cache_key", "target"): _text(getattr(self, name), name)


@dataclass(frozen=True)
class DispatchEvidence(Contract):
  schema: ClassVar[str] = "execution_bridge.dispatch_evidence.v1"
  executable_digest: str; workload_digest: str; transport: str; status: str; elapsed_ns: int | None = None; output_digest: str | None = None; errors: tuple[str, ...] = ()
  def __post_init__(self):
    for name in ("executable_digest", "workload_digest", "transport"): _text(getattr(self, name), name)
    if self.status not in ("passed", "failed", "timed_out"): raise ValueError("unsupported dispatch status")
    if self.elapsed_ns is not None and (not isinstance(self.elapsed_ns, int) or self.elapsed_ns < 0): raise ValueError("elapsed_ns must be non-negative")
    if self.output_digest is not None: _text(self.output_digest, "output_digest")


@dataclass(frozen=True)
class SafetyAdmission(Contract):
  schema: ClassVar[str] = "execution_bridge.safety_admission.v1"
  workload_digest: str; executable_digest: str; authorized: bool; opt_in: bool; health_token: str; revocation_reason: str | None = None
  def __post_init__(self):
    for name in ("workload_digest", "executable_digest", "health_token"): _text(getattr(self, name), name)
    if not isinstance(self.authorized, bool) or not isinstance(self.opt_in, bool): raise ValueError("authorization flags must be bool")
    if self.authorized and not self.opt_in: raise ValueError("authorized admission requires opt_in")
    if self.authorized and self.revocation_reason is not None: raise ValueError("authorized admission cannot be revoked")


__all__ = ["Contract", "WorkloadIdentity", "SemanticScheduleIdentity", "TransportPlan", "CompileArtifactMetadata", "ExecutableArtifactMetadata", "DispatchEvidence", "SafetyAdmission"]
