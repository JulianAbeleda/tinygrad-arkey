"""Fact-only construction of complete memory-adaptive prefill policies.

This is the bridge between a selected model's invocation inventory and the
memory planner.  It deliberately does not know model names, profiles, or size
classes.  Route producers publish exact invocation coverage and target
requirements; this module only joins those facts and rejects partial policies.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib, json
from typing import Any, Iterable, Mapping, Sequence

from extra.qk.memory_adaptive_autoscan import AutoscanCandidate
from tinygrad.llm.prefill_memory_plan import ByteLifetime, ByteTerm, CandidateMemoryCoverage, Strategy
from extra.qk.prefill_workload_plan import CandidateKernelCapability, InvocationBytes, RemainderMapping

SCHEMA = "tinygrad.memory_adaptive_candidate_catalog.v1"
IDENTITY_DOMAIN = "tinygrad.memory_adaptive.whole_policy_identity.v1"
WORKLOAD_IDENTITY_DOMAIN = "tinygrad.memory_adaptive.whole_policy_identity.workload_expansion.v1"
_SELF_ROUTE_SENTINEL = ("tinygrad.memory_adaptive.route.self.v1",)
_FORBIDDEN = frozenset(("profile", "profile_id", "model_name", "model_path", "filename", "size_label",
                        "model_size_label"))
_POLICY_SERIALIZATION_FIELDS = frozenset(("candidate_id", "whole_policy_identity", "catalog_schema", "strategy", "routes"))


def _semantic(value: Any) -> Any:
  if isinstance(value, Mapping):
    return {str(k): _semantic(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))
            if str(k).lower().replace("-", "_") not in _FORBIDDEN}
  if isinstance(value, (list, tuple)): return [_semantic(x) for x in value]
  if value is None or isinstance(value, (str, int, float, bool)): return value
  raise TypeError(f"inventory facts must be JSON-shaped, got {type(value).__name__}")


def _invocation_id(row: Mapping[str, Any]) -> str:
  explicit = row.get("invocation_id")
  if explicit is not None:
    if not isinstance(explicit, str) or not explicit: raise ValueError("invocation_id must be a non-empty string")
    return explicit
  payload = json.dumps(_semantic(row), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
  return "invocation:sha256:" + hashlib.sha256(payload).hexdigest()


def inventory_invocation_ids(inventory: Mapping[str, Any]) -> tuple[str, ...]:
  """Return stable semantic IDs for every selected-model inventory row."""
  rows = inventory.get("rows")
  if not isinstance(rows, (list, tuple)): raise ValueError("selected model inventory requires a rows sequence")
  ids = tuple(_invocation_id(row) if isinstance(row, Mapping) else "" for row in rows)
  if any(not x for x in ids): raise ValueError("inventory rows must be mappings")
  if len(ids) != len(set(ids)): raise ValueError("inventory invocation identities must be unique")
  return ids


def _inventory_identity(inventory: Mapping[str, Any]) -> str:
  identity = inventory.get("inventory_identity")
  if not isinstance(identity, str) or not identity:
    raise ValueError("selected model inventory requires a non-empty inventory_identity")
  return identity


def _requirements_met(required: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
  """Recursive exact/subset matcher for published structural target facts."""
  for key, expected in required.items():
    if key not in actual: return False
    observed = actual[key]
    if isinstance(expected, Mapping):
      if not isinstance(observed, Mapping) or not _requirements_met(expected, observed): return False
    elif isinstance(expected, (list, tuple, set, frozenset)):
      if observed not in expected: return False
    elif observed != expected: return False
  return True


def _term(value: ByteTerm | Mapping[str, Any]) -> ByteTerm:
  if isinstance(value, ByteTerm): return value
  if not isinstance(value, Mapping): raise TypeError("candidate memory terms must be ByteTerm values or mappings")
  try: lifetime = value["lifetime"] if isinstance(value["lifetime"], ByteLifetime) else ByteLifetime(value["lifetime"])
  except (KeyError, ValueError) as exc: raise ValueError("candidate memory term has invalid lifetime") from exc
  return ByteTerm(str(value["name"]), value.get("bytes"), str(value["provenance"]), str(value["formula"]), lifetime)


def _sorted_json_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
  normalized = [dict(_semantic(row)) for row in rows]
  return sorted(normalized, key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False))


def derive_whole_policy_identity(*, inventory_identity: str, routes: Mapping[str, str], strategy: Strategy,
                                 memory_terms: Sequence[ByteTerm], target_requirements: Mapping[str, Any],
                                 semantic_policy: Mapping[str, Any], capability: CandidateKernelCapability,
                                 structurally_available: bool, evidence_available: bool,
                                 operational_self_alias: str | None = None) -> str:
  """Derive identity from validated complete-policy facts, never its operational alias."""
  policy = {key: value for key, value in semantic_policy.items() if key not in _POLICY_SERIALIZATION_FIELDS}
  semantic_routes = {key: _SELF_ROUTE_SENTINEL if operational_self_alias is not None and routes[key] == operational_self_alias
                     else routes[key] for key in sorted(routes)}
  payload = {
    "domain": IDENTITY_DOMAIN,
    "parent_inventory_identity": inventory_identity,
    "invocation_routes": semantic_routes,
    "strategy": strategy.value,
    "memory_terms": _sorted_json_rows(term.to_dict() for term in memory_terms),
    "target_requirements": _semantic(target_requirements),
    "semantic_policy": policy,
    "capability_proof_facts": {
      "full_m_values": sorted(set(capability.full_m_values)),
      "tail_m_values": sorted(set(capability.tail_m_values)),
      "correctness_m_values": sorted(set(capability.correctness_m_values)),
      "invocation_bytes": _sorted_json_rows({"m": row.m, "activation_bytes": row.activation_bytes,
                                               "scratch_bytes": row.scratch_bytes}
                                              for row in capability.invocation_bytes),
      "remainder_mappings": _sorted_json_rows({"logical_m": row.logical_m, "physical_m": row.physical_m,
                                                 "minimum_prompt_tokens": row.minimum_prompt_tokens}
                                                for row in capability.remainder_mappings),
      "structurally_available": structurally_available,
    },
    "evidence_available": evidence_available,
  }
  encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
  return "whole-policy:sha256:" + hashlib.sha256(encoded).hexdigest()


def derive_workload_policy_identity(*, base_whole_policy_identity: str, workload_choice: Mapping[str, Any],
                                    workload_memory_term: ByteTerm) -> str:
  """Bind a canonical base policy to one workload expansion, excluding only its operational alias."""
  if not isinstance(base_whole_policy_identity, str) or not base_whole_policy_identity:
    raise ValueError("base_whole_policy_identity must be a non-empty string")
  if not isinstance(workload_choice, Mapping): raise TypeError("workload_choice must be a mapping")
  if not isinstance(workload_memory_term, ByteTerm): raise TypeError("workload_memory_term must be a ByteTerm")
  semantic_choice = {key: value for key, value in workload_choice.items() if key not in ("candidate_id", "machine_candidate_id")}
  payload = {
    "domain": WORKLOAD_IDENTITY_DOMAIN,
    "base_whole_policy_identity": base_whole_policy_identity,
    "workload_choice": _semantic(semantic_choice),
    "workload_memory_term": _semantic(workload_memory_term.to_dict()),
  }
  encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
  return "whole-policy:sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CandidateSpec:
  """A producer-published complete-policy offer; coverage is exact, not inferred."""
  candidate_id: str
  strategy: Strategy
  covered_invocations: tuple[str, ...]
  memory_terms: tuple[ByteTerm | Mapping[str, Any], ...] = ()
  target_requirements: Mapping[str, Any] = None  # type: ignore[assignment]
  structurally_available: bool = True
  evidence_available: bool = True
  policy: Mapping[str, Any] = None  # type: ignore[assignment]
  full_m_values: tuple[int, ...] = ()
  tail_m_values: tuple[int, ...] = ()
  correctness_m_values: tuple[int, ...] = ()
  invocation_bytes: tuple[InvocationBytes | Mapping[str, Any], ...] = ()
  remainder_mappings: tuple[RemainderMapping | Mapping[str, Any], ...] = ()

  def __post_init__(self) -> None:
    if not self.candidate_id: raise ValueError("candidate_id must not be empty")
    if self.strategy is Strategy.REFUSE: raise ValueError("REFUSE is not a candidate strategy")
    if len(self.covered_invocations) != len(set(self.covered_invocations)): raise ValueError("covered invocations must be unique")
    object.__setattr__(self, "target_requirements", {} if self.target_requirements is None else self.target_requirements)
    object.__setattr__(self, "policy", {} if self.policy is None else self.policy)

  def kernel_capability(self) -> CandidateKernelCapability:
    """Return only explicitly published shape, proof, and byte facts."""
    rows = tuple(x if isinstance(x, InvocationBytes) else InvocationBytes(
      x["m"], x.get("activation_bytes"), x.get("scratch_bytes")) for x in self.invocation_bytes)
    mappings = tuple(x if isinstance(x, RemainderMapping) else RemainderMapping(
      x["logical_m"], x["physical_m"], x.get("minimum_prompt_tokens", 1)) for x in self.remainder_mappings)
    return CandidateKernelCapability(self.candidate_id, self.full_m_values, self.tail_m_values,
                                     rows, self.correctness_m_values, mappings)


def _spec(value: CandidateSpec | Mapping[str, Any]) -> CandidateSpec:
  if isinstance(value, CandidateSpec): return value
  if not isinstance(value, Mapping): raise TypeError("candidate specs must be CandidateSpec values or mappings")
  if "whole_policy_identity" in value: raise ValueError("candidate producers must not supply whole_policy_identity")
  return CandidateSpec(candidate_id=str(value["candidate_id"]), strategy=Strategy(value["strategy"]),
    covered_invocations=tuple(value.get("covered_invocations", ())), memory_terms=tuple(value.get("memory_terms", ())),
    target_requirements=value.get("target_requirements", {}),
    structurally_available=value.get("structurally_available", True),
    evidence_available=value.get("evidence_available", True), policy=value.get("policy", {}),
    full_m_values=tuple(value.get("full_m_values", ())), tail_m_values=tuple(value.get("tail_m_values", ())),
    correctness_m_values=tuple(value.get("correctness_m_values", ())),
    invocation_bytes=tuple(value.get("invocation_bytes", ())), remainder_mappings=tuple(value.get("remainder_mappings", ())))


def build_candidate_catalog(*, selected_model_inventory: Mapping[str, Any], target_capabilities: Mapping[str, Any],
                            candidate_specs: Iterable[CandidateSpec | Mapping[str, Any]]) -> tuple[AutoscanCandidate, ...]:
  """Build only complete, target-supported policies for autoscan.

  Unknown coverage is rejected.  In particular a bounded route is never
  catalogued unless its producer explicitly says both implementation and
  evidence are available for every required invocation.
  """
  inventory_identity = _inventory_identity(selected_model_inventory)
  required = inventory_invocation_ids(selected_model_inventory)
  required_set = set(required)
  out = []
  for raw in candidate_specs:
    spec = _spec(raw)
    covered = set(spec.covered_invocations)
    if covered - required_set: raise ValueError(f"{spec.candidate_id} covers unknown inventory invocations")
    if not spec.structurally_available or not _requirements_met(spec.target_requirements, target_capabilities): continue
    if spec.strategy is Strategy.BOUNDED_PACKED_TILES and not spec.evidence_available: continue
    if covered != required_set: continue
    memory = CandidateMemoryCoverage(spec.candidate_id, spec.strategy, tuple(_term(x) for x in spec.memory_terms),
      required, tuple(x for x in required if x in covered), supported=True)
    semantic_policy = _semantic(spec.policy)
    if not isinstance(semantic_policy, Mapping): raise TypeError("candidate policy must be a mapping")
    if "whole_policy_identity" in semantic_policy:
      raise ValueError("candidate producers must not supply whole_policy_identity")
    supplied_routes = semantic_policy.get("routes") if isinstance(semantic_policy, Mapping) else None
    if supplied_routes is None:
      routes = {invocation: spec.candidate_id for invocation in required}
    else:
      if not isinstance(supplied_routes, Mapping) or set(supplied_routes) != required_set:
        raise ValueError(f"{spec.candidate_id} policy routes must exactly cover the selected inventory")
      if any(not isinstance(route_id, str) or not route_id for route_id in supplied_routes.values()):
        raise ValueError(f"{spec.candidate_id} policy route IDs must be non-empty strings")
      routes = {invocation: supplied_routes[invocation] for invocation in required}
    identity = derive_whole_policy_identity(inventory_identity=inventory_identity, routes=routes, strategy=spec.strategy,
      memory_terms=memory.memory_terms, target_requirements=spec.target_requirements, semantic_policy=semantic_policy,
      capability=spec.kernel_capability(), structurally_available=spec.structurally_available,
      evidence_available=spec.evidence_available, operational_self_alias=spec.candidate_id)
    policy = {**semantic_policy, "candidate_id": spec.candidate_id, "routes": routes, "catalog_schema": SCHEMA,
              "whole_policy_identity": identity}
    out.append(AutoscanCandidate(memory, policy))
  if len({x.candidate_id for x in out}) != len(out): raise ValueError("candidate_id values must be unique")
  if len({x.whole_policy_identity for x in out}) != len(out):
    raise ValueError("whole_policy_identity values must be unique")
  return tuple(sorted(out, key=lambda x: (x.memory.strategy.value, x.candidate_id)))


__all__ = ["SCHEMA", "IDENTITY_DOMAIN", "WORKLOAD_IDENTITY_DOMAIN", "CandidateSpec", "build_candidate_catalog",
           "derive_whole_policy_identity", "derive_workload_policy_identity", "inventory_invocation_ids"]
