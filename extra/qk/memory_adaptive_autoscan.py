"""Fact-only orchestration for memory-adaptive prefill machine search.

The caller has already selected a model.  This module neither opens a model nor
interprets its name: it joins supplied model/inventory facts, current device
facts, the pure memory planner, an injected guarded runner, and the pure policy
selector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from extra.qk.memory_adaptive_policy import cache_matches, make_cache_record, select_policy
from tinygrad.llm.device_facts import (
  DeviceFacts, DeviceFactsSchemaError, StaleDeviceFactsError, scan_device_facts, validate_device_facts_snapshot,
)
from extra.qk.memory_adaptive_device_facts import MemoryReservePolicy, calculate_admissible_budget
from tinygrad.llm.prefill_memory_plan import (ByteLifetime, ByteTerm, CandidateMemoryCoverage, DeviceMemoryFacts,
                                               PrefillMemoryPlan, Strategy, plan_prefill_memory)

SCHEMA = "tinygrad.memory_adaptive_autoscan.v1"
EvidenceRunner = Callable[["AutoscanCandidate"], Mapping[str, Any] | None]


@dataclass(frozen=True)
class AutoscanCandidate:
  """One supplied complete policy and its independently checkable memory contract."""
  memory: CandidateMemoryCoverage
  policy: Mapping[str, Any]

  def __post_init__(self) -> None:
    candidate_id = self.policy.get("candidate_id")
    if candidate_id != self.memory.candidate_id:
      raise ValueError("policy candidate_id must exactly match memory candidate_id")

  @property
  def candidate_id(self) -> str: return self.memory.candidate_id

  @property
  def whole_policy_identity(self) -> str:
    value = self.policy.get("whole_policy_identity")
    if not isinstance(value, str) or not value: raise ValueError("policy whole_policy_identity must be a non-empty string")
    return value

  def policy_record(self) -> dict[str, Any]:
    # Strategy is structural and therefore material even when callers omit it.
    return {**dict(self.policy), "candidate_id": self.candidate_id, "strategy": self.memory.strategy.value}


def _planning_device(facts: DeviceFacts, reserve_policy: MemoryReservePolicy) -> DeviceMemoryFacts:
  budget = calculate_admissible_budget(facts, reserve_policy)
  alignment = facts.capabilities.global_allocation_granularity
  scanned_reserve = None
  if None not in (facts.total_vram_bytes, facts.free_vram_bytes, alignment):
    occupied = max(0, facts.total_vram_bytes-facts.free_vram_bytes)
    scanned_reserve = ((occupied+alignment-1)//alignment)*alignment
  scan_derived = budget.reserve_bytes == scanned_reserve and scanned_reserve is not None
  reserve = ByteTerm("device safety reserve", budget.reserve_bytes,
                     f"{facts.memory_probe.source} live selected-device scan" if scan_derived else "device facts and runtime reserve policy",
                     "align_up(total_vram_bytes - free_vram_bytes, scanned_allocator_granularity)" if scan_derived else
                     "fixed_bytes + floor(total_vram_bytes * fraction_of_total)", ByteLifetime.SAFETY_RESERVE)
  return DeviceMemoryFacts(facts.total_vram_bytes, facts.free_vram_bytes, reserve,
                           f"{facts.memory_probe.source} selected-device memory probe")


def _model_record(selected_model_facts: Mapping[str, Any], selected_model_inventory: Mapping[str, Any]) -> dict[str, Any]:
  if not isinstance(selected_model_facts, Mapping) or not isinstance(selected_model_inventory, Mapping):
    raise TypeError("selected model facts and inventory must be mappings")
  return {"facts": dict(selected_model_facts), "inventory": dict(selected_model_inventory)}


def _key_args(*, facts: DeviceFacts, model_record: Mapping[str, Any], workload: Mapping[str, Any],
              candidates: Sequence[AutoscanCandidate], compiler_runtime_revision: Mapping[str, Any],
              search_revision: str) -> dict[str, Any]:
  return {"gpu_facts": facts.planning_snapshot(), "model_facts": model_record, "workload": workload,
          "candidates": [x.policy_record() for x in candidates],
          "compiler_runtime_revision": compiler_runtime_revision, "search_revision": search_revision}


def load_exact_cache(cache_record: Mapping[str, Any] | None, **search_key_args: Any) -> dict[str, Any] | None:
  """Return an exact-key cached policy, or ``None`` for stale/malformed input."""
  try: validate_device_facts_snapshot(search_key_args.get("gpu_facts"))
  except DeviceFactsSchemaError: return None
  if not isinstance(cache_record, Mapping): return None
  cached_result = cache_record.get("result")
  cached_inputs = cached_result.get("canonical_inputs") if isinstance(cached_result, Mapping) else None
  try:
    validate_device_facts_snapshot(cached_inputs.get("gpu_facts") if isinstance(cached_inputs, Mapping) else None)
  except DeviceFactsSchemaError:
    return None
  if not cache_matches(cache_record, **search_key_args): return None
  result = cache_record.get("result")
  return dict(result) if isinstance(result, Mapping) else None


def _refusal(plan: PrefillMemoryPlan, reason: str, *, interrupted: bool = False) -> dict[str, Any]:
  return {"schema": SCHEMA, "decision": Strategy.REFUSE.value, "selected_candidate_id": None,
          "reason": reason, "interrupted": interrupted, "from_cache": False,
          "memory_plan": plan.to_dict(), "policy": None, "cache_record": None}


def autoscan_selected_model(*, selected_model_facts: Mapping[str, Any],
                            selected_model_inventory: Mapping[str, Any], base_terms: Sequence[ByteTerm],
                            candidates: Sequence[AutoscanCandidate], workload: Mapping[str, Any],
                            compiler_runtime_revision: Mapping[str, Any], evidence_runner: EvidenceRunner,
                            baseline_candidate_id: str | None, device_facts: DeviceFacts | None = None,
                            selected_device: str | None = None,
                            device_scanner: Callable[..., DeviceFacts] = scan_device_facts,
                            reserve_policy: MemoryReservePolicy = MemoryReservePolicy(),
                            strategy_override: Strategy | Sequence[Strategy] | None = None,
                            cache_record: Mapping[str, Any] | None = None,
                            search_revision: str = SCHEMA, min_samples: int = 3,
                            max_relative_noise: float = 0.05, tie_relative_tolerance: float = 0.01) -> dict[str, Any]:
  """Plan, guarded-run, select, and cache policies for the user-selected model.

  ``evidence_runner`` is the only execution hook.  Returning ``None``, raising
  ``KeyboardInterrupt``, or failing before all candidates complete is treated
  as an interrupted/no-result search and never promotes partial non-baseline
  evidence.
  """
  if not callable(evidence_runner): raise TypeError("evidence_runner must be callable")
  supplied = tuple(candidates)
  if len({x.candidate_id for x in supplied}) != len(supplied): raise ValueError("candidate_id values must be unique")
  if baseline_candidate_id is not None and baseline_candidate_id not in {x.candidate_id for x in supplied}:
    raise ValueError("baseline_candidate_id must identify a supplied candidate")
  facts = device_facts if device_facts is not None else device_scanner(selected_device=selected_device)
  if not isinstance(facts, DeviceFacts): raise TypeError("device scanner must return DeviceFacts")
  try:
    validate_device_facts_snapshot(facts.planning_snapshot())
  except StaleDeviceFactsError:
    if device_facts is None: raise
    facts = device_scanner(selected_device=facts.selected_device)
    if not isinstance(facts, DeviceFacts): raise TypeError("device reprobe must return DeviceFacts")
    validate_device_facts_snapshot(facts.planning_snapshot())
  model_record = _model_record(selected_model_facts, selected_model_inventory)
  plan = plan_prefill_memory(device=_planning_device(facts, reserve_policy), base_terms=base_terms,
                             candidates=(x.memory for x in supplied), override=strategy_override)
  feasible_ids = set(plan.feasible_candidate_ids)
  feasible = tuple(x for x in supplied if x.candidate_id in feasible_ids)
  if not feasible: return _refusal(plan, "no supplied candidate has a complete safe feasible plan")

  key_args = _key_args(facts=facts, model_record=model_record, workload=workload, candidates=feasible,
                       compiler_runtime_revision=compiler_runtime_revision, search_revision=search_revision)
  cached = load_exact_cache(cache_record, **key_args)
  if cached is not None:
    return {"schema": SCHEMA, "decision": cached.get("decision"),
            "selected_candidate_id": cached.get("selected_candidate_id"), "reason": "exact-key cache hit",
            "interrupted": False, "from_cache": True, "memory_plan": plan.to_dict(),
            "policy": cached, "cache_record": dict(cache_record)}

  # Baseline first makes safe interruption fallback possible without executing
  # anything after an unhealthy/no-result run.
  ordered = sorted(feasible, key=lambda x: (x.candidate_id != baseline_candidate_id, x.candidate_id))
  evidence: dict[str, Mapping[str, Any]] = {}
  interrupted = False
  for candidate in ordered:
    try: proof = evidence_runner(candidate)
    except (KeyboardInterrupt, InterruptedError, Exception):
      interrupted = True
      break
    if proof is None:
      interrupted = True
      break
    if not isinstance(proof, Mapping): raise TypeError("evidence runner results must be mappings or None")
    evidence[candidate.candidate_id] = proof

  selection_candidates = feasible
  if interrupted:
    baseline = next((x for x in feasible if x.candidate_id == baseline_candidate_id), None)
    if baseline is None or baseline.candidate_id not in evidence:
      return _refusal(plan, "search interrupted/no-result and no feasible baseline has guarded evidence", interrupted=True)
    selection_candidates = (baseline,)
  policy_args = _key_args(facts=facts, model_record=model_record, workload=workload, candidates=selection_candidates,
                          compiler_runtime_revision=compiler_runtime_revision, search_revision=search_revision)
  policy = select_policy(**policy_args, evidence=evidence, baseline_candidate_id=baseline_candidate_id,
                         min_samples=min_samples, max_relative_noise=max_relative_noise,
                         tie_relative_tolerance=tie_relative_tolerance)
  if policy["decision"] != "SELECTED":
    why = "safe baseline failed guarded evidence" if interrupted else "no complete policy passed guarded evidence"
    return {**_refusal(plan, why, interrupted=interrupted), "policy": policy}
  # An interrupted search establishes only a safe temporary baseline, not the
  # optimum over the original feasible set.  Do not persist it as a completed
  # machine-search result.
  record = None if interrupted else make_cache_record(policy)
  return {"schema": SCHEMA, "decision": "SELECTED", "selected_candidate_id": policy["selected_candidate_id"],
          "reason": "safe baseline after interrupted/no-result search" if interrupted else policy["decision_reason"],
          "interrupted": interrupted, "from_cache": False, "memory_plan": plan.to_dict(),
          "policy": policy, "cache_record": record}


__all__ = ["SCHEMA", "AutoscanCandidate", "EvidenceRunner", "autoscan_selected_model", "load_exact_cache"]
