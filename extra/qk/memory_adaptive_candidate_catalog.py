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
from tinygrad.llm.prefill_workload_plan import CandidateKernelCapability, InvocationBytes, RemainderMapping

SCHEMA = "tinygrad.memory_adaptive_candidate_catalog.v1"
_FORBIDDEN = frozenset(("profile", "profile_id", "model_name", "model_path", "filename", "size_label",
                        "model_size_label"))


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
    supplied_routes = semantic_policy.get("routes") if isinstance(semantic_policy, Mapping) else None
    if supplied_routes is None:
      routes = {invocation: spec.candidate_id for invocation in required}
    else:
      if not isinstance(supplied_routes, Mapping) or set(supplied_routes) != required_set:
        raise ValueError(f"{spec.candidate_id} policy routes must exactly cover the selected inventory")
      if any(not isinstance(route_id, str) or not route_id for route_id in supplied_routes.values()):
        raise ValueError(f"{spec.candidate_id} policy route IDs must be non-empty strings")
      routes = {invocation: supplied_routes[invocation] for invocation in required}
    policy = {**semantic_policy, "candidate_id": spec.candidate_id, "routes": routes, "catalog_schema": SCHEMA}
    out.append(AutoscanCandidate(memory, policy))
  if len({x.candidate_id for x in out}) != len(out): raise ValueError("candidate_id values must be unique")
  return tuple(sorted(out, key=lambda x: (x.memory.strategy.value, x.candidate_id)))


__all__ = ["SCHEMA", "CandidateSpec", "build_candidate_catalog", "inventory_invocation_ids"]
