"""Pure, model-agnostic memory feasibility planning for LLM prefill.

This module deliberately knows nothing about model filenames, parameter-count
labels, GPUs, or kernel performance.  Callers scan the explicitly selected
model and pass the resulting byte and coverage facts here.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
from typing import Any, Iterable

PREFILL_MEMORY_PLAN_SCHEMA = "tinygrad.prefill_memory_plan.v1"


class Strategy(StrEnum):
  FULL_RESIDENT_OVERLAY = "FULL_RESIDENT_OVERLAY"
  BOUNDED_PACKED_TILES = "BOUNDED_PACKED_TILES"
  DIRECT_PACKED_FALLBACK = "DIRECT_PACKED_FALLBACK"
  REFUSE = "REFUSE"


class ByteLifetime(StrEnum):
  PERSISTENT = "persistent"
  PREFILL_PEAK = "prefill_peak"
  CANDIDATE_WORKSPACE = "candidate_workspace"
  SAFETY_RESERVE = "safety_reserve"


@dataclass(frozen=True)
class ByteTerm:
  name: str
  bytes: int | None
  provenance: str
  formula: str
  lifetime: ByteLifetime

  def __post_init__(self) -> None:
    if not self.name: raise ValueError("byte term name must not be empty")
    if not self.provenance: raise ValueError(f"byte term {self.name!r} requires provenance")
    if not self.formula: raise ValueError(f"byte term {self.name!r} requires a formula")
    if self.bytes is not None and (not isinstance(self.bytes, int) or isinstance(self.bytes, bool) or self.bytes < 0):
      raise ValueError(f"byte term {self.name!r} bytes must be a non-negative integer or None")

  @property
  def known(self) -> bool: return self.bytes is not None

  def to_dict(self) -> dict[str, Any]:
    return {"name": self.name, "bytes": self.bytes, "provenance": self.provenance,
            "formula": self.formula, "lifetime": self.lifetime.value}


@dataclass(frozen=True)
class DeviceMemoryFacts:
  total_bytes: int | None
  free_bytes: int | None
  safety_reserve: ByteTerm
  provenance: str

  def __post_init__(self) -> None:
    for name, value in (("total_bytes", self.total_bytes), ("free_bytes", self.free_bytes)):
      if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
        raise ValueError(f"{name} must be a non-negative integer or None")
    if self.safety_reserve.lifetime is not ByteLifetime.SAFETY_RESERVE:
      raise ValueError("device safety_reserve must have SAFETY_RESERVE lifetime")
    if not self.provenance: raise ValueError("device memory facts require provenance")

  def to_dict(self) -> dict[str, Any]:
    return {"total_bytes": self.total_bytes, "free_bytes": self.free_bytes,
            "safety_reserve": self.safety_reserve.to_dict(), "provenance": self.provenance}


@dataclass(frozen=True)
class CandidateMemoryCoverage:
  candidate_id: str
  strategy: Strategy
  memory_terms: tuple[ByteTerm, ...] = ()
  required_invocations: tuple[str, ...] = ()
  covered_invocations: tuple[str, ...] = ()
  supported: bool = True
  reasons: tuple[str, ...] = ()

  def __post_init__(self) -> None:
    if not self.candidate_id: raise ValueError("candidate_id must not be empty")
    if self.strategy is Strategy.REFUSE: raise ValueError("REFUSE is a decision, not a candidate strategy")
    if len(set(self.required_invocations)) != len(self.required_invocations): raise ValueError("required invocations must be unique")
    if len(set(self.covered_invocations)) != len(self.covered_invocations): raise ValueError("covered invocations must be unique")

  def to_dict(self) -> dict[str, Any]:
    return {"candidate_id": self.candidate_id, "strategy": self.strategy.value,
            "memory_terms": [x.to_dict() for x in self.memory_terms],
            "required_invocations": list(self.required_invocations),
            "covered_invocations": list(self.covered_invocations), "supported": self.supported,
            "reasons": list(self.reasons)}


@dataclass(frozen=True)
class CandidateDecision:
  candidate_id: str
  strategy: Strategy
  feasible: bool
  estimated_peak_bytes: int | None
  memory_terms: tuple[ByteTerm, ...]
  reasons: tuple[str, ...]

  def to_dict(self) -> dict[str, Any]:
    return {"candidate_id": self.candidate_id, "strategy": self.strategy.value, "feasible": self.feasible,
            "estimated_peak_bytes": self.estimated_peak_bytes, "memory_terms": [x.to_dict() for x in self.memory_terms],
            "reasons": list(self.reasons)}


@dataclass(frozen=True)
class PrefillMemoryPlan:
  decision: Strategy | None
  admitted_budget_bytes: int | None
  base_peak_bytes: int | None
  feasible_strategies: tuple[Strategy, ...]
  feasible_candidate_ids: tuple[str, ...]
  candidate_decisions: tuple[CandidateDecision, ...]
  reasons: tuple[str, ...]
  device: DeviceMemoryFacts
  base_terms: tuple[ByteTerm, ...]
  override: tuple[Strategy, ...] | None

  @property
  def strategy(self) -> Strategy | None:
    """The sole safe candidate's strategy, REFUSE, or None when machine search is required."""
    return self.decision

  @property
  def requires_machine_search(self) -> bool:
    return len(self.feasible_candidate_ids) > 1

  def to_dict(self) -> dict[str, Any]:
    return {"schema": PREFILL_MEMORY_PLAN_SCHEMA, "decision": None if self.decision is None else self.decision.value,
            "admitted_budget_bytes": self.admitted_budget_bytes, "base_peak_bytes": self.base_peak_bytes,
            "feasible_strategies": [x.value for x in self.feasible_strategies],
            "feasible_candidate_ids": list(self.feasible_candidate_ids),
            "candidate_decisions": [x.to_dict() for x in self.candidate_decisions], "reasons": list(self.reasons),
            "device": self.device.to_dict(), "base_terms": [x.to_dict() for x in self.base_terms],
            "override": None if self.override is None else [x.value for x in self.override]}

  def to_json(self) -> str:
    return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


_STRATEGY_ORDER = (Strategy.FULL_RESIDENT_OVERLAY, Strategy.BOUNDED_PACKED_TILES, Strategy.DIRECT_PACKED_FALLBACK)


def _known_sum(terms: Iterable[ByteTerm]) -> int | None:
  values = tuple(x.bytes for x in terms)
  return None if any(x is None for x in values) else sum(values)  # type: ignore[arg-type]


def plan_prefill_memory(*, device: DeviceMemoryFacts, base_terms: Iterable[ByteTerm],
                        candidates: Iterable[CandidateMemoryCoverage],
                        override: Strategy | Iterable[Strategy] | None = None) -> PrefillMemoryPlan:
  """Return the deterministic, complete safe set without making a performance choice.

  ``override`` can only restrict candidate strategies.  Unknown memory facts,
  incomplete coverage, and unsupported candidates always fail closed.
  """
  bases = tuple(base_terms)
  cands = tuple(candidates)
  if len({x.candidate_id for x in cands}) != len(cands): raise ValueError("candidate_id values must be unique")
  allowed = None if override is None else tuple(sorted({override} if isinstance(override, Strategy) else set(override), key=lambda x: x.value))
  if allowed is not None and Strategy.REFUSE in allowed: raise ValueError("REFUSE cannot be used as a strategy override")

  reserve = device.safety_reserve.bytes
  available = None if device.total_bytes is None or device.free_bytes is None else min(device.total_bytes, device.free_bytes)
  budget = None if available is None or reserve is None else max(0, available - reserve)
  base_peak = _known_sum(bases)
  decisions: list[CandidateDecision] = []
  for cand in sorted(cands, key=lambda x: (x.strategy.value, x.candidate_id)):
    reasons = list(cand.reasons if not cand.supported else ())
    if not cand.supported and not reasons: reasons.append("candidate capability is unsupported")
    if allowed is not None and cand.strategy not in allowed: reasons.append("excluded by explicit strategy override")
    missing = sorted(set(cand.required_invocations) - set(cand.covered_invocations))
    if missing: reasons.append("missing coverage: " + ", ".join(missing))
    unknown = sorted(x.name for x in (*bases, *cand.memory_terms) if not x.known)
    if unknown: reasons.append("unknown memory bytes: " + ", ".join(unknown))
    if budget is None: reasons.append("admitted VRAM budget is unknown")
    peak = _known_sum((*bases, *cand.memory_terms))
    if peak is not None and budget is not None and peak > budget:
      reasons.append(f"estimated peak {peak} exceeds admitted budget {budget} by {peak-budget} bytes")
    decisions.append(CandidateDecision(cand.candidate_id, cand.strategy, not reasons, peak, cand.memory_terms, tuple(reasons)))

  feasible = tuple(x for x in decisions if x.feasible)
  feasible_strategies = tuple(x for x in _STRATEGY_ORDER if any(y.strategy is x for y in feasible))
  refusal_reasons: list[str] = []
  if not feasible:
    if not cands: refusal_reasons.append("no strategy candidates were provided")
    if budget is None: refusal_reasons.append("memory safety cannot be proven because the admitted VRAM budget is unknown")
    if base_peak is None: refusal_reasons.append("memory safety cannot be proven because base residency contains unknown bytes")
    elif budget is not None and base_peak > budget:
      refusal_reasons.append(f"base residency {base_peak} exceeds admitted budget {budget} by {base_peak-budget} bytes")
    refusal_reasons.extend(f"{x.candidate_id}: {reason}" for x in decisions for reason in x.reasons)
  # Any candidate choice is a performance choice, even when all candidates share one strategy.
  decision = Strategy.REFUSE if not feasible else feasible[0].strategy if len(feasible) == 1 else None
  if len(feasible) > 1:
    refusal_reasons.append("multiple candidates are feasible; performance selection is deferred to machine search")
  return PrefillMemoryPlan(decision, budget, base_peak, feasible_strategies, tuple(x.candidate_id for x in feasible),
                           tuple(decisions), tuple(refusal_reasons), device, bases, allowed)


__all__ = ["PREFILL_MEMORY_PLAN_SCHEMA", "ByteLifetime", "ByteTerm", "CandidateDecision",
           "CandidateMemoryCoverage", "DeviceMemoryFacts", "PrefillMemoryPlan", "Strategy", "plan_prefill_memory"]
