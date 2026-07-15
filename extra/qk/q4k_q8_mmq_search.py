"""Machine search for generated Q4_K x Q8_1 MMQ candidates.

Research-only: this module searches descriptors supplied by a generator and
never changes dispatch.  The session object is deliberately injected so a
backend can own compilation, launch, correctness, and timing details without
this runner inventing schedules or hardware counters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import itertools
import json
from typing import Any, Callable, Mapping, Protocol, Sequence

SCHEMA = "q4k-q8-1-mmq-search.v1"
DEFAULT_ROUTE = "direct_packed"
AGGREGATE_POLICY_SCHEMA = "q4k-q8-1-mmq-aggregate-policy.v1"


@dataclass(frozen=True)
class MMQDescriptor:
  """One generated candidate descriptor; axes are intentionally opaque."""
  candidate_id: str
  axes: Mapping[str, Any]

  def canonical(self) -> dict[str, Any]:
    return {"candidate_id": self.candidate_id, "axes": dict(self.axes)}


class SearchSession(Protocol):
  def prepare(self, descriptor: MMQDescriptor) -> Any: ...
  def check_correctness(self, prepared: Any) -> Mapping[str, Any]: ...
  def measure(self, prepared: Any, *, warmups: int, rounds: int) -> Mapping[str, Any]: ...
  def measure_direct_packed(self, *, warmups: int, rounds: int) -> Mapping[str, Any]: ...
  def evidence_gate(self, prepared: Any, correctness: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class SearchPolicy:
  warmups: int = 2
  rounds: int = 5
  default_route: str = DEFAULT_ROUTE
  resource_limits: Mapping[str, int | float] = field(default_factory=dict)


@dataclass(frozen=True)
class AggregatePolicy:
  """Policy for comparing one candidate across a same-session role set.

  Costs are milliseconds and are deliberately supplied as evidence by the
  caller (the search layer does not invent packing or reduction schedules).
  """
  required_roles: tuple[str, ...]
  preparation_ms: Mapping[str, float] = field(default_factory=dict)
  packing_ms: Mapping[str, float] = field(default_factory=dict)
  reduction_ms: Mapping[str, float] = field(default_factory=dict)
  direct_preparation_ms: Mapping[str, float] = field(default_factory=dict)
  direct_packing_ms: Mapping[str, float] = field(default_factory=dict)
  direct_reduction_ms: Mapping[str, float] = field(default_factory=dict)


def enumerate_descriptors(axes: Mapping[str, Sequence[Any]], *, id_prefix: str = "q4k_q8_1_mmq") -> tuple[MMQDescriptor, ...]:
  """Enumerate stable Cartesian products from generated candidate axes."""
  names = tuple(sorted(axes))
  if any(not axes[name] for name in names): return ()
  out = []
  for values in itertools.product(*(axes[name] for name in names)):
    payload = dict(zip(names, values))
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    out.append(MMQDescriptor(f"{id_prefix}.{hashlib.sha256(encoded.encode()).hexdigest()[:16]}", payload))
  if len({descriptor.candidate_id for descriptor in out}) != len(out):
    raise ValueError("generated axes contain duplicate descriptor identities")
  return tuple(out)


def replay_descriptors(report: Mapping[str, Any]) -> tuple[MMQDescriptor, ...]:
  """Recover the exact descriptor set from a search artifact, fail closed."""
  if report.get("schema") != SCHEMA:
    raise ValueError("unsupported search artifact schema")
  expected = report.get("artifact_sha256")
  if not isinstance(expected, str):
    raise ValueError("search artifact is missing artifact_sha256")
  unsigned = dict(report)
  unsigned.pop("artifact_sha256", None)
  encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
  if hashlib.sha256(encoded).hexdigest() != expected:
    raise ValueError("search artifact digest mismatch")
  if report.get("production_dispatch_changed") is not False:
    raise ValueError("search artifact is not research-only")
  if report.get("status") not in ("PASS", "NO_PASSING_CANDIDATE"):
    raise ValueError("search artifact has an incomplete status")
  descriptors = []
  for row in report.get("candidates", ()):
    descriptor = row.get("descriptor") if isinstance(row, Mapping) else None
    if not isinstance(descriptor, Mapping) or not isinstance(descriptor.get("candidate_id"), str) or not isinstance(descriptor.get("axes"), Mapping):
      raise ValueError("search artifact contains an invalid descriptor")
    descriptors.append(MMQDescriptor(descriptor["candidate_id"], dict(descriptor["axes"])))
  if len({descriptor.candidate_id for descriptor in descriptors}) != len(descriptors):
    raise ValueError("search artifact contains duplicate descriptor identities")
  return tuple(descriptors)


def _fits(descriptor: MMQDescriptor, limits: Mapping[str, int | float]) -> tuple[bool, str | None]:
  resources = descriptor.axes.get("resources", {})
  if not isinstance(resources, Mapping): return False, "resources must be a mapping"
  for key, limit in limits.items():
    value = resources.get(key)
    if value is not None and value > limit: return False, f"resource {key}={value} exceeds limit {limit}"
  return True, None


def _timing_ms(row: Mapping[str, Any]) -> float | None:
  value = row.get("min_ms", row.get("candidate_min_ms"))
  return float(value) if isinstance(value, (int, float)) and value > 0 else None


def _costs(policy: AggregatePolicy, role: str, *, direct: bool) -> dict[str, float] | None:
  names = ("preparation_ms", "packing_ms", "reduction_ms")
  maps = ((policy.direct_preparation_ms, policy.direct_packing_ms, policy.direct_reduction_ms)
          if direct else (policy.preparation_ms, policy.packing_ms, policy.reduction_ms))
  result = {}
  for name, values in zip(names, maps):
    value = values.get(role, 0.0)
    if not isinstance(value, (int, float)) or value < 0: return None
    result[name] = float(value)
  return result


def evaluate_aggregate_policy(*, candidate_rows: Mapping[str, Mapping[str, Any]],
                              direct_packed_rows: Mapping[str, Mapping[str, Any]],
                              policy: AggregatePolicy, session_id: str | None = None) -> dict[str, Any]:
  """Choose an aggregate candidate only when every role is fully evidenced.

  ``candidate_rows`` and ``direct_packed_rows`` are outputs from the same
  session.  This function is intentionally policy-only: it cannot promote a
  route or alter an emitter.
  """
  roles = tuple(policy.required_roles)
  if not roles or len(set(roles)) != len(roles):
    raise ValueError("required_roles must be non-empty and unique")
  rows = {}
  eligible = []
  for candidate_id, candidate in candidate_rows.items():
    role_rows = candidate.get("roles", candidate) if isinstance(candidate, Mapping) else {}
    if not isinstance(role_rows, Mapping): role_rows = {}
    blockers = []
    total = 0.0
    direct_total = 0.0
    for role in roles:
      row = role_rows.get(role)
      direct = direct_packed_rows.get(role)
      if not isinstance(row, Mapping) or not isinstance(direct, Mapping): blockers.append(f"{role}: missing evidence"); continue
      if session_id is not None and (row.get("session_id") != session_id or direct.get("session_id") != session_id):
        blockers.append(f"{role}: session identity mismatch"); continue
      timing, direct_timing = _timing_ms(row), _timing_ms(direct)
      costs, direct_costs = _costs(policy, role, direct=False), _costs(policy, role, direct=True)
      gate = row.get("evidence_gate", {})
      if row.get("status") not in ("measured", "PASS") or row.get("correctness", {}).get("passed") is not True or not isinstance(gate, Mapping) or gate.get("timing_allowed") is not True:
        blockers.append(f"{role}: incomplete candidate evidence"); continue
      if timing is None or direct_timing is None or costs is None or direct_costs is None or direct.get("passed", True) is False:
        blockers.append(f"{role}: incomplete timing/cost evidence"); continue
      total += timing + sum(costs.values()); direct_total += direct_timing + sum(direct_costs.values())
    result = {"candidate_id": candidate_id, "status": "BLOCKED" if blockers else "ELIGIBLE",
              "blockers": blockers, "aggregate_ms": None if blockers else total,
              "direct_packed_ms": None if blockers else direct_total,
              "speedup_vs_direct_packed": None if blockers else direct_total / total}
    rows[candidate_id] = result
    if not blockers: eligible.append(result)
  winner = min(eligible, key=lambda x: x["aggregate_ms"], default=None)
  return {"schema": AGGREGATE_POLICY_SCHEMA, "status": "PASS" if winner else "NO_AGGREGATE_WINNER",
          "winner": winner, "candidates": rows, "required_roles": roles,
          "production_dispatch_changed": False}


def run_search(*, axes: Mapping[str, Sequence[Any]], session_factory: Callable[[], SearchSession],
               policy: SearchPolicy = SearchPolicy()) -> dict[str, Any]:
  """Run correctness first, then timing, with direct_packed in the same session."""
  if policy.warmups < 0 or policy.rounds < 1: raise ValueError("warmups >= 0 and rounds >= 1 are required")
  descriptors = enumerate_descriptors(axes)
  rows: list[dict[str, Any]] = []
  passing: list[tuple[MMQDescriptor, dict[str, Any]]] = []
  for descriptor in descriptors:
    fits, reason = _fits(descriptor, policy.resource_limits)
    row: dict[str, Any] = {"descriptor": descriptor.canonical(), "status": "rejected" if not fits else "not_run"}
    if not fits:
      row["blocker"] = reason; rows.append(row); continue
    session = session_factory()
    try:
      prepared = session.prepare(descriptor)
      correctness = dict(session.check_correctness(prepared))
      row["correctness"] = correctness
      gate_fn = getattr(session, "evidence_gate", None)
      gate = dict(gate_fn(prepared, correctness)) if callable(gate_fn) else {
        "timing_allowed": False, "promotion_eligible": False,
        "blockers": ["complete candidate evidence gate unavailable"],
      }
      row["evidence_gate"] = gate
      # Persist the complete descriptor alongside the provenance returned by
      # the guarded compile/correctness session.  This is the join key for
      # every later evidence artifact and makes the report self-describing.
      row["candidate_identity"] = {
        "candidate_id": descriptor.candidate_id,
        "descriptor": descriptor.canonical(),
        "provenance": correctness.get("provenance", {}),
      }
      passed = correctness.get("passed") is True and gate.get("timing_allowed") is True
      if not passed:
        row.update(status="correctness_failed" if correctness.get("passed") is not True else "evidence_blocked",
                   blocker="candidate correctness did not pass" if correctness.get("passed") is not True
                   else "; ".join(gate.get("blockers", ())) or "candidate evidence gate did not pass")
      else:
        timing = dict(session.measure(prepared, warmups=policy.warmups, rounds=policy.rounds))
        direct = dict(session.measure_direct_packed(warmups=policy.warmups, rounds=policy.rounds))
        row.update(status="measured", timing=timing, direct_packed=direct)
        cand, base = _timing_ms(timing), _timing_ms(direct)
        row["speedup_vs_direct_packed"] = None if cand is None or base is None else base / cand
        # An unavailable/failed timing is evidence of no result, never a PASS.
        if cand is not None and base is not None and timing.get("passed", True) is not False and direct.get("passed", True) is not False:
          passing.append((descriptor, row))
        else:
          row["status"] = "timing_blocked"
          row["blocker"] = "candidate and direct timing evidence are both required"
    except Exception as exc:
      row.update(status="blocked", blocker=f"{type(exc).__name__}: {exc}")
    rows.append(row)
  measured = [row for _, row in passing]
  winner = min(measured, key=lambda row: _timing_ms(row["timing"]) or float("inf"), default=None)
  artifact = {"schema": SCHEMA, "default_route": policy.default_route, "production_dispatch_changed": False,
              "policy": {"warmups": policy.warmups, "rounds": policy.rounds, "resource_limits": dict(policy.resource_limits)},
              "candidates": rows, "winner": winner["descriptor"] if winner else None,
              "winner_evidence": winner, "status": "PASS" if winner else "NO_PASSING_CANDIDATE"}
  canonical = json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
  artifact["artifact_sha256"] = hashlib.sha256(canonical).hexdigest()
  return artifact


def write_artifact(report: Mapping[str, Any], path: str) -> None:
  # Never persist an artifact that cannot be replayed and whose digest is not
  # over the exact bytes being written.
  replay_descriptors(report)
  if report.get("artifact_sha256") is None:
    raise ValueError("search artifact is missing artifact_sha256")
  with open(path, "w", encoding="utf-8") as handle:
    json.dump(report, handle, sort_keys=True, indent=2)
    handle.write("\n")
