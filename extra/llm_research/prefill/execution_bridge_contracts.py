"""Research-only, hardware-independent, serializable execution-bridge contracts."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import hashlib, json, math
from typing import Any, ClassVar, Mapping, TypeVar


def _text(value: Any, name: str) -> str:
  if not isinstance(value, str) or not value.strip(): raise ValueError(f"{name} must be a non-empty string")
  return value

def _mapping(value: Any, name: str) -> dict[str, Any]:
  if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value): raise ValueError(f"{name} must be a string-keyed mapping")
  try: json.dumps(value, sort_keys=True, separators=(",", ":"))
  except (TypeError, ValueError) as exc: raise ValueError(f"{name} must be JSON serializable") from exc
  return dict(value)

def _tuple(value: Any, name: str) -> tuple[Any, ...]:
  if not isinstance(value, (list, tuple)): raise ValueError(f"{name} must be a list or tuple")
  return tuple(value)

def _bool(value: Any, name: str) -> bool:
  if not isinstance(value, bool): raise ValueError(f"{name} must be bool")
  return value

def _non_negative_number(value: Any, name: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
    raise ValueError(f"{name} must be a finite non-negative number")
  return float(value)

def _json_value(value: Any) -> Any:
  if isinstance(value, Contract): return value.to_dict()
  if isinstance(value, Mapping): return {k: _json_value(v) for k, v in value.items()}
  if isinstance(value, (tuple, list)): return [_json_value(v) for v in value]
  return value


# --- Typed lifecycle/outcome vocabularies (P0-1, P1-5) -----------------------
# The dispatch lifecycle STATE is truthful evidence of what the executor did.
# It is deliberately distinct from every downstream outcome, so a record can
# never conflate execution with correctness, benchmark validity, research
# verdict, or shipping decision.
DISPATCH_STATES: tuple[str, ...] = ("not_attempted", "attempted", "submitted", "completed", "failed", "timed_out", "device_lost")
RESEARCH_VERDICTS: tuple[str, ...] = ("direct_l2_wins", "retain_lds", "measurement_inconclusive", "blocked", "failed")
SHIPPING_DECISIONS: tuple[str, ...] = ("retain_lds", "promote_direct_l2")
OPERAND_STRATEGIES: tuple[str, ...] = ("register_resident", "lds_staged", "cache_streamed", "reloaded", "unknown")
RESULT_STATUSES: tuple[str, ...] = ("not_attempted", "passed", "failed", "timed_out", "unsupported", "blocked")

def dispatch_state(value: Any, name: str = "dispatch_state") -> str:
  if value not in DISPATCH_STATES: raise ValueError(f"{name} must be one of {DISPATCH_STATES}")
  return value

def canonical_json(payload: Mapping[str, Any], name: str = "payload") -> str:
  return json.dumps(_mapping(payload, name), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

def canonical_digest(payload: Mapping[str, Any], name: str = "payload") -> str:
  return hashlib.sha256(canonical_json(payload, name).encode()).hexdigest()

def reject_synthetic(record: Mapping[str, Any], *, production: bool, name: str = "evidence") -> None:
  # P0-4: synthetic fixtures declare synthetic=True; production decisions reject them.
  if production and record.get("synthetic") is True:
    raise ValueError(f"{name} is synthetic and cannot back a production decision")


ContractT = TypeVar("ContractT", bound="Contract")

class Contract:
  schema: ClassVar[str]
  def to_dict(self) -> dict[str, Any]: return {"schema": self.schema, **{f.name: _json_value(getattr(self, f.name)) for f in fields(self)}}
  def to_json(self) -> str: return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
  @property
  def digest(self) -> str: return hashlib.sha256(self.to_json().encode()).hexdigest()
  @classmethod
  def from_dict(cls: type[ContractT], payload: Mapping[str, Any]) -> ContractT:
    row = _mapping(payload, cls.__name__)
    schema = row.pop("schema", cls.schema)
    if schema != cls.schema: raise ValueError(f"schema must be {cls.schema}")
    known = {f.name for f in fields(cls)}
    unknown = set(row) - known
    if unknown: raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**cls._coerce(row))
  @classmethod
  def _coerce(cls, payload: dict[str, Any]) -> dict[str, Any]: return payload


@dataclass(frozen=True)
class WorkloadIdentity(Contract):
  schema: ClassVar[str] = "execution_bridge.workload_identity.v1"
  workload_id: str; role: str; shape: tuple[int, ...]; dtypes: tuple[str, ...]; layout: Mapping[str, Any] = field(default_factory=dict)
  def __post_init__(self):
    _text(self.workload_id, "workload_id"); _text(self.role, "role")
    if not self.shape or any(not isinstance(x, int) or x <= 0 for x in self.shape): raise ValueError("shape must contain positive integers")
    if not self.dtypes or any(not isinstance(x, str) or not x for x in self.dtypes): raise ValueError("dtypes must be non-empty")
    _mapping(self.layout, "layout")
  @classmethod
  def _coerce(cls, payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("shape", "dtypes"):
      if key in payload: payload[key] = _tuple(payload[key], key)
    return payload


@dataclass(frozen=True)
class SemanticScheduleIdentity(Contract):
  schema: ClassVar[str] = "execution_bridge.semantic_schedule_identity.v1"
  workload_digest: str; schedule_digest: str; buffer_roles: tuple[str, ...]; abi_digest: str
  def __post_init__(self):
    for name in ("workload_digest", "schedule_digest", "abi_digest"): _text(getattr(self, name), name)
    if not self.buffer_roles or any(not isinstance(x, str) or not x for x in self.buffer_roles): raise ValueError("buffer_roles must be non-empty")
  @classmethod
  def _coerce(cls, payload: dict[str, Any]) -> dict[str, Any]:
    if "buffer_roles" in payload: payload["buffer_roles"] = _tuple(payload["buffer_roles"], "buffer_roles")
    return payload


@dataclass(frozen=True)
class SemanticOperandPlan(Contract):
  schema: ClassVar[str] = "execution_bridge.semantic_operand_plan.v1"
  operand_id: str; semantic_role: str; abi_argument: str; declared_strategy: str
  requirements: Mapping[str, Any] = field(default_factory=dict); fallback_eligible: bool = False
  def __post_init__(self):
    for name in ("operand_id", "semantic_role", "abi_argument"): _text(getattr(self, name), name)
    if self.declared_strategy not in OPERAND_STRATEGIES: raise ValueError(f"declared_strategy must be one of {OPERAND_STRATEGIES}")
    _mapping(self.requirements, "requirements"); _bool(self.fallback_eligible, "fallback_eligible")


@dataclass(frozen=True)
class TransportPlan(Contract):
  schema: ClassVar[str] = "execution_bridge.transport_plan.v1"
  transport: str; schedule_digest: str; requirements: Mapping[str, Any] = field(default_factory=dict); fallback_eligible: bool = False
  operands: tuple[SemanticOperandPlan, ...] = ()
  def __post_init__(self):
    _text(self.transport, "transport"); _text(self.schedule_digest, "schedule_digest"); _mapping(self.requirements, "requirements")
    if not isinstance(self.fallback_eligible, bool): raise ValueError("fallback_eligible must be bool")
    if any(not isinstance(x, SemanticOperandPlan) for x in self.operands): raise ValueError("operands must contain SemanticOperandPlan values")
    if len({x.operand_id for x in self.operands}) != len(self.operands): raise ValueError("operand ids must be unique")
    if len({x.abi_argument for x in self.operands}) != len(self.operands): raise ValueError("operand ABI arguments must be unique")
  @classmethod
  def _coerce(cls, payload: dict[str, Any]) -> dict[str, Any]:
    if "operands" in payload: payload["operands"] = tuple(SemanticOperandPlan.from_dict(x) for x in _tuple(payload["operands"], "operands"))
    return payload


@dataclass(frozen=True)
class ArtifactRequest(Contract):
  schema: ClassVar[str] = "execution_bridge.artifact_request.v1"
  kind: str; required: bool = True; options: Mapping[str, Any] = field(default_factory=dict)
  def __post_init__(self): _text(self.kind, "kind"); _bool(self.required, "required"); _mapping(self.options, "options")


@dataclass(frozen=True)
class CounterGroupRequest(Contract):
  schema: ClassVar[str] = "execution_bridge.counter_group_request.v1"
  group_id: str; counters: tuple[str, ...]; optional_when_unsupported: bool = True; separate_pass: bool = True
  def __post_init__(self):
    _text(self.group_id, "group_id")
    if not self.counters or any(not isinstance(x, str) or not x for x in self.counters): raise ValueError("counters must be non-empty")
    _bool(self.optional_when_unsupported, "optional_when_unsupported"); _bool(self.separate_pass, "separate_pass")
  @classmethod
  def _coerce(cls, payload: dict[str, Any]) -> dict[str, Any]:
    if "counters" in payload: payload["counters"] = _tuple(payload["counters"], "counters")
    return payload


@dataclass(frozen=True)
class CorrectnessProtocol(Contract):
  schema: ClassVar[str] = "execution_bridge.correctness_protocol.v1"
  oracle: str; scope: str = "full_output"; atol: float = 0.0; rtol: float = 0.0
  require_finite: bool = True; immutable_inputs: bool = True
  def __post_init__(self):
    _text(self.oracle, "oracle"); _text(self.scope, "scope")
    if self.scope != "full_output": raise ValueError("correctness scope must be full_output")
    _non_negative_number(self.atol, "atol"); _non_negative_number(self.rtol, "rtol")
    _bool(self.require_finite, "require_finite"); _bool(self.immutable_inputs, "immutable_inputs")


@dataclass(frozen=True)
class GuardProtocol(Contract):
  schema: ClassVar[str] = "execution_bridge.guard_protocol.v1"
  hard_timeout_ms: int; guard_buffers: bool = True; health_preflight: bool = True; health_postflight: bool = True
  process_isolation: bool = True; nonconstant_inputs: bool = True
  def __post_init__(self):
    if not isinstance(self.hard_timeout_ms, int) or self.hard_timeout_ms <= 0: raise ValueError("hard_timeout_ms must be positive")
    for name in ("guard_buffers", "health_preflight", "health_postflight", "process_isolation", "nonconstant_inputs"): _bool(getattr(self, name), name)


@dataclass(frozen=True)
class TimingProtocol(Contract):
  schema: ClassVar[str] = "execution_bridge.timing_protocol.v1"
  warmups: int; rounds: int; randomization_seed: int; statistic: str = "median"; noise_threshold: float = 0.0
  synchronize: bool = True; same_session: bool = True
  def __post_init__(self):
    if not isinstance(self.warmups, int) or self.warmups < 0: raise ValueError("warmups must be non-negative")
    if not isinstance(self.rounds, int) or self.rounds <= 0: raise ValueError("rounds must be positive")
    if not isinstance(self.randomization_seed, int): raise ValueError("randomization_seed must be int")
    _text(self.statistic, "statistic")
    _non_negative_number(self.noise_threshold, "noise_threshold")
    _bool(self.synchronize, "synchronize"); _bool(self.same_session, "same_session")


@dataclass(frozen=True)
class ExecutionRequest(Contract):
  schema: ClassVar[str] = "execution_bridge.request.v1"
  experiment_id: str; candidate_id: str; comparator_id: str; workload_digest: str; schedule_digest: str
  transport_plan: TransportPlan; target_context: Mapping[str, Any]; compiler_context: Mapping[str, Any]
  candidate_knobs: Mapping[str, Any] = field(default_factory=dict); fixed_invariants: Mapping[str, Any] = field(default_factory=dict)
  artifacts: tuple[ArtifactRequest, ...] = (); counter_groups: tuple[CounterGroupRequest, ...] = ()
  correctness: CorrectnessProtocol | None = None; guard: GuardProtocol | None = None; timing: TimingProtocol | None = None
  def __post_init__(self):
    for name in ("experiment_id", "candidate_id", "comparator_id", "workload_digest", "schedule_digest"): _text(getattr(self, name), name)
    if not isinstance(self.transport_plan, TransportPlan): raise ValueError("transport_plan must be TransportPlan")
    for name in ("target_context", "compiler_context", "candidate_knobs", "fixed_invariants"): _mapping(getattr(self, name), name)
    if any(not isinstance(x, ArtifactRequest) for x in self.artifacts): raise ValueError("artifacts must contain ArtifactRequest values")
    if any(not isinstance(x, CounterGroupRequest) for x in self.counter_groups): raise ValueError("counter_groups must contain CounterGroupRequest values")
    if len({x.kind for x in self.artifacts}) != len(self.artifacts): raise ValueError("artifact kinds must be unique")
    if len({x.group_id for x in self.counter_groups}) != len(self.counter_groups): raise ValueError("counter group ids must be unique")
    if self.transport_plan.schedule_digest != self.schedule_digest: raise ValueError("transport and request schedule digests must match")
    if self.correctness is not None and not isinstance(self.correctness, CorrectnessProtocol): raise ValueError("invalid correctness protocol")
    if self.guard is not None and not isinstance(self.guard, GuardProtocol): raise ValueError("invalid guard protocol")
    if self.timing is not None and not isinstance(self.timing, TimingProtocol): raise ValueError("invalid timing protocol")
  @classmethod
  def _coerce(cls, payload: dict[str, Any]) -> dict[str, Any]:
    payload["transport_plan"] = TransportPlan.from_dict(payload["transport_plan"])
    payload["artifacts"] = tuple(ArtifactRequest.from_dict(x) for x in payload.get("artifacts", ()))
    payload["counter_groups"] = tuple(CounterGroupRequest.from_dict(x) for x in payload.get("counter_groups", ()))
    for key, typ in (("correctness", CorrectnessProtocol), ("guard", GuardProtocol), ("timing", TimingProtocol)):
      if payload.get(key) is not None: payload[key] = typ.from_dict(payload[key])
    return payload


@dataclass(frozen=True)
class UnsupportedOutcome(Contract):
  schema: ClassVar[str] = "execution_bridge.unsupported.v1"
  reason: str; phase: str; feature: str; provider_detail: Mapping[str, Any] = field(default_factory=dict)
  status: str = "unsupported"
  def __post_init__(self):
    for name in ("reason", "phase", "feature"): _text(getattr(self, name), name)
    if self.status != "unsupported": raise ValueError("unsupported outcome status must be unsupported")
    _mapping(self.provider_detail, "provider_detail")


@dataclass(frozen=True)
class PhaseResult(Contract):
  schema: ClassVar[str] = "execution_bridge.phase_result.v1"
  phase: str; status: str; identity: Mapping[str, Any] = field(default_factory=dict); evidence: Mapping[str, Any] = field(default_factory=dict)
  unsupported: tuple[UnsupportedOutcome, ...] = (); error: TypedError | None = None
  def __post_init__(self):
    _text(self.phase, "phase")
    if self.status not in RESULT_STATUSES: raise ValueError(f"status must be one of {RESULT_STATUSES}")
    _mapping(self.identity, "identity"); _mapping(self.evidence, "evidence")
    if any(not isinstance(x, UnsupportedOutcome) for x in self.unsupported): raise ValueError("unsupported must contain UnsupportedOutcome values")
    if self.status == "unsupported" and not self.unsupported: raise ValueError("unsupported status requires an unsupported outcome")
    if self.error is not None and not isinstance(self.error, TypedError): raise ValueError("error must be TypedError")
  @classmethod
  def _coerce(cls, payload: dict[str, Any]) -> dict[str, Any]:
    payload["unsupported"] = tuple(UnsupportedOutcome.from_dict(x) for x in payload.get("unsupported", ()))
    if payload.get("error") is not None: payload["error"] = TypedError.from_dict(payload["error"])
    return payload


@dataclass(frozen=True)
class ExecutionResult(Contract):
  schema: ClassVar[str] = "execution_bridge.result.v1"
  experiment_id: str; candidate_id: str; request_digest: str; phases: tuple[PhaseResult, ...]
  extensions: Mapping[str, Any] = field(default_factory=dict)
  def __post_init__(self):
    for name in ("experiment_id", "candidate_id", "request_digest"): _text(getattr(self, name), name)
    if not self.phases or any(not isinstance(x, PhaseResult) for x in self.phases): raise ValueError("phases must be non-empty PhaseResult values")
    if len({x.phase for x in self.phases}) != len(self.phases): raise ValueError("phase names must be unique")
    _mapping(self.extensions, "extensions")
  @classmethod
  def _coerce(cls, payload: dict[str, Any]) -> dict[str, Any]:
    payload["phases"] = tuple(PhaseResult.from_dict(x) for x in payload["phases"])
    return payload


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
  executable_digest: str; workload_digest: str; transport: str; state: str
  synthetic: bool = False; elapsed_ns: int | None = None; output_digest: str | None = None; errors: tuple[str, ...] = ()
  def __post_init__(self):
    for name in ("executable_digest", "workload_digest", "transport"): _text(getattr(self, name), name)
    dispatch_state(self.state, "state")
    if not isinstance(self.synthetic, bool): raise ValueError("synthetic must be bool")
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


@dataclass(frozen=True)
class TypedError(Contract):
  schema: ClassVar[str] = "execution_bridge.typed_error.v1"
  code: str; phase: str; recoverable: bool; candidate: str | None = None; run: str | None = None; context: Mapping[str, Any] = field(default_factory=dict)
  def __post_init__(self):
    _text(self.code, "code"); _text(self.phase, "phase")
    if not isinstance(self.recoverable, bool): raise ValueError("recoverable must be bool")
    for opt in ("candidate", "run"):
      if getattr(self, opt) is not None: _text(getattr(self, opt), opt)
    _mapping(self.context, "context")


__all__ = ["Contract", "WorkloadIdentity", "SemanticScheduleIdentity", "SemanticOperandPlan", "TransportPlan",
           "ArtifactRequest", "CounterGroupRequest", "CorrectnessProtocol", "GuardProtocol", "TimingProtocol",
           "ExecutionRequest", "UnsupportedOutcome", "PhaseResult", "ExecutionResult", "CompileArtifactMetadata",
           "ExecutableArtifactMetadata", "DispatchEvidence", "SafetyAdmission", "TypedError",
           "DISPATCH_STATES", "RESEARCH_VERDICTS", "SHIPPING_DECISIONS", "OPERAND_STRATEGIES", "RESULT_STATUSES", "dispatch_state",
           "canonical_json", "canonical_digest", "reject_synthetic"]
