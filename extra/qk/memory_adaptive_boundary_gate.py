"""Independent, fail-closed validation of memory-adaptive prefill boundaries.

This is an evidence consumer, not another planner.  It accepts JSON-shaped case
records so synthetic tests and isolated GPU runners can use the same gate.
Names and benchmark profiles are deliberately excluded from semantic identity.
"""
from __future__ import annotations

import hashlib, json, math
from typing import Any, Mapping, Sequence

from tinygrad.llm.prefill_memory_plan import Strategy

SCHEMA = "tinygrad.memory_adaptive_boundary_gate.v1"
_EXECUTING = frozenset((Strategy.FULL_RESIDENT_OVERLAY.value, Strategy.BOUNDED_PACKED_TILES.value,
                        Strategy.DIRECT_PACKED_FALLBACK.value))
_OUTCOMES = _EXECUTING | {Strategy.REFUSE.value}
_NON_SEMANTIC = frozenset(("filename", "file_name", "model_filename", "model_path", "path", "model_name",
                           "display_name", "size_label", "model_size_label", "profile", "profile_id",
                           "benchmark_profile", "case_id", "label"))


def _canonical(value: Any, *, strip_names: bool = False) -> Any:
  if isinstance(value, Mapping):
    return {str(k): _canonical(v, strip_names=strip_names) for k, v in sorted(value.items(), key=lambda x: str(x[0]))
            if not strip_names or str(k).lower().replace("-", "_") not in _NON_SEMANTIC}
  if isinstance(value, (list, tuple)): return [_canonical(x, strip_names=strip_names) for x in value]
  if value is None or isinstance(value, (str, bool, int)): return value
  if isinstance(value, float) and math.isfinite(value): return value
  raise TypeError(f"evidence must be finite JSON data, got {type(value).__name__}")


def stable_json(value: Any) -> str:
  return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _fingerprint(value: Any) -> str:
  return "sha256:" + hashlib.sha256(stable_json(_canonical(value, strip_names=True)).encode()).hexdigest()


def _integer(value: Any) -> bool: return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _forbidden_keys(value: Any, path: str = "selected_policy") -> list[str]:
  found: list[str] = []
  if isinstance(value, Mapping):
    for key, child in value.items():
      normalized = str(key).lower().replace("-", "_")
      if normalized in _NON_SEMANTIC-{"case_id", "label"}: found.append(f"{path}.{key}")
      found.extend(_forbidden_keys(child, f"{path}.{key}"))
  elif isinstance(value, (list, tuple)):
    for i, child in enumerate(value): found.extend(_forbidden_keys(child, f"{path}[{i}]"))
  return found


def _terms_sum(terms: Any, errors: list[str], where: str) -> int | None:
  if not isinstance(terms, list): errors.append(f"{where} must be a list"); return None
  total = 0
  for idx, term in enumerate(terms):
    if not isinstance(term, Mapping) or not _integer(term.get("bytes")):
      errors.append(f"{where}[{idx}] has unknown or invalid bytes"); return None
    total += term["bytes"]
  return total


def _case(case: Mapping[str, Any], index: int) -> tuple[dict[str, Any], dict[str, Any]]:
  errors: list[str] = []
  case_id = case.get("case_id", f"case-{index}")
  if not isinstance(case_id, str) or not case_id: case_id = f"case-{index}"; errors.append("case_id must be a non-empty string")
  outcome = case.get("outcome")
  if outcome not in _OUTCOMES: errors.append("outcome is missing or invalid")
  for key in ("selected_model", "workload", "gpu_snapshot", "memory_plan", "selected_policy"):
    if not isinstance(case.get(key), Mapping): errors.append(f"{key} must be a mapping")

  plan = case.get("memory_plan") if isinstance(case.get("memory_plan"), Mapping) else {}
  base = _terms_sum(plan.get("base_terms"), errors, "memory_plan.base_terms")
  if base is not None and plan.get("base_peak_bytes") != base: errors.append("planned base peak does not equal base-term arithmetic")
  device = plan.get("device") if isinstance(plan.get("device"), Mapping) else {}
  reserve = device.get("safety_reserve") if isinstance(device.get("safety_reserve"), Mapping) else {}
  total, free, reserve_bytes = device.get("total_bytes"), device.get("free_bytes"), reserve.get("bytes")
  budget = max(0, min(total, free)-reserve_bytes) if all(_integer(x) for x in (total, free, reserve_bytes)) else None
  if plan.get("admitted_budget_bytes") != budget: errors.append("admitted budget does not match GPU snapshot arithmetic")
  snapshot = case.get("gpu_snapshot") if isinstance(case.get("gpu_snapshot"), Mapping) else {}
  for snapshot_key, plan_key in (("total_bytes", "total_bytes"), ("total_vram_bytes", "total_bytes"),
                                 ("free_bytes", "free_bytes"), ("free_vram_bytes", "free_bytes")):
    if snapshot_key in snapshot and snapshot[snapshot_key] != device.get(plan_key):
      errors.append(f"GPU snapshot {snapshot_key} disagrees with memory plan")

  policy = case.get("selected_policy") if isinstance(case.get("selected_policy"), Mapping) else {}
  forbidden = _forbidden_keys(policy)
  if forbidden: errors.append("policy uses forbidden model/profile semantic selectors: " + ", ".join(forbidden))
  selected_id = policy.get("selected_candidate_id")
  decisions = plan.get("candidate_decisions")
  decision = next((x for x in decisions if isinstance(x, Mapping) and x.get("candidate_id") == selected_id), None) if isinstance(decisions, list) else None
  if outcome in _EXECUTING:
    if not isinstance(selected_id, str) or not selected_id: errors.append("executing outcome requires a selected candidate")
    if policy.get("decision") != "SELECTED": errors.append("executing outcome requires SELECTED policy evidence")
    constraints = policy.get("target_constraints")
    if not isinstance(constraints, Mapping) or not constraints:
      errors.append("selected policy requires explicit target capability constraints")
    else:
      mismatches = sorted(str(k) for k, v in constraints.items() if k not in snapshot or snapshot.get(k) != v)
      if mismatches: errors.append("target capability constraints do not match scanned GPU facts: " + ", ".join(mismatches))
    if not isinstance(decision, Mapping): errors.append("selected candidate is absent from memory plan")
    else:
      if decision.get("strategy") != outcome: errors.append("selected candidate strategy disagrees with outcome")
      candidate_terms = decision.get("memory_terms")
      extra = _terms_sum(candidate_terms, errors, "selected candidate memory_terms")
      expected = base + extra if base is not None and extra is not None else None
      if not _integer(expected) or decision.get("estimated_peak_bytes") != expected: errors.append("planned peak arithmetic is not independently reproducible")
      if budget is None or expected > budget or decision.get("feasible") is not True: errors.append("selected candidate is not proven within its admitted bound")
  else:
    if selected_id is not None: errors.append("REFUSE must not select a candidate")
    if policy.get("decision") != "REFUSE": errors.append("REFUSE requires refusal policy evidence")
    if budget is not None and isinstance(decisions, list) and any(isinstance(x, Mapping) and x.get("feasible") is True for x in decisions):
      errors.append("REFUSE contradicts a feasible candidate")

  allocation = case.get("measured_allocation")
  census = case.get("route_census")
  output = case.get("output_evidence")
  if outcome in _EXECUTING:
    if not isinstance(allocation, Mapping) or not _integer(allocation.get("peak_bytes")): errors.append("measured peak allocation is required")
    elif isinstance(decision, Mapping) and _integer(decision.get("estimated_peak_bytes")) and allocation["peak_bytes"] > decision["estimated_peak_bytes"]:
      errors.append("measured peak exceeds planned peak")
    allocations = allocation.get("allocations") if isinstance(allocation, Mapping) else None
    if not isinstance(allocations, list): errors.append("measured allocation inventory is required")
    elif outcome != Strategy.FULL_RESIDENT_OVERLAY.value and any(isinstance(x, Mapping) and x.get("kind") in ("dense_overlay", "full_dequant_overlay") and x.get("bytes", 0) for x in allocations):
      errors.append("bounded/direct branch contains a hidden dense overlay allocation")
    required = case.get("selected_model", {}).get("inventory", {}).get("invocations") if isinstance(case.get("selected_model"), Mapping) else None
    observed = census.get("rows") if isinstance(census, Mapping) else None
    if not isinstance(required, list) or not isinstance(observed, list): errors.append("exact route census requires invocation and observed row lists")
    else:
      def rows(xs: list[Any]) -> list[tuple[Any, Any, Any]]:
        return sorted((x.get("invocation_id"), x.get("candidate_id"), x.get("call_count")) for x in xs if isinstance(x, Mapping))
      expected_rows = sorted((x.get("invocation_id"), selected_id, x.get("call_count")) for x in required if isinstance(x, Mapping))
      if len(expected_rows) != len(required) or rows(observed) != expected_rows: errors.append("route census is not an exact inventory/count/candidate match")
    if not isinstance(output, Mapping) or output.get("status") != "PASS" or not isinstance(output.get("content_digest"), str):
      errors.append("passing full-output evidence with a content digest is required")
  else:
    if allocation is not None or census is not None or output is not None: errors.append("REFUSE must occur before allocation, routing, or output")

  semantic = {"model": case.get("selected_model"), "workload": case.get("workload"), "gpu": case.get("gpu_snapshot")}
  result = {"case_id": case_id, "outcome": outcome, "passed": not errors, "errors": sorted(set(errors)),
            "semantic_key": _fingerprint(semantic)}
  meta = {"model_key": _fingerprint(case.get("selected_model")), "workload_key": _fingerprint(case.get("workload")),
          "gpu_key": _fingerprint(case.get("gpu_snapshot")), "semantic_key": result["semantic_key"], "outcome": outcome,
          "passed": not errors}
  return result, meta


def validate_boundary_gate(cases: Sequence[Mapping[str, Any]], *, require_all_outcomes: bool = True) -> dict[str, Any]:
  """Validate a complete synthetic/measured boundary matrix without executing it."""
  if isinstance(cases, (str, bytes)) or not isinstance(cases, Sequence): raise TypeError("cases must be a sequence of mappings")
  checked, metas = zip(*(_case(c, i) if isinstance(c, Mapping) else _case({}, i) for i, c in enumerate(cases))) if cases else ((), ())
  global_errors: list[str] = []
  outcomes = {x["outcome"] for x in checked if x["outcome"] in _OUTCOMES}
  missing = sorted(_OUTCOMES-outcomes) if require_all_outcomes else []
  if missing: global_errors.append("missing required outcomes: " + ", ".join(missing))
  groups: dict[str, list[int]] = {}
  for i, meta in enumerate(metas): groups.setdefault(meta["semantic_key"], []).append(i)
  for indexes in groups.values():
    if len(indexes) > 1 and len({checked[i]["outcome"] for i in indexes}) != 1: global_errors.append("same-content rename changed the outcome")
  model_groups: dict[str, list[int]] = {}
  for i, meta in enumerate(metas): model_groups.setdefault(meta["model_key"], []).append(i)
  for indexes in model_groups.values():
    # Repeated exact facts must be deterministic, including refusals above a bound.
    exact: dict[tuple[str, str], str] = {}
    for i in indexes:
      key = (metas[i]["gpu_key"], metas[i]["workload_key"])
      if key in exact and exact[key] != checked[i]["outcome"]: global_errors.append("identical boundary inputs produced non-deterministic outcomes")
      exact[key] = checked[i]["outcome"]
  if require_all_outcomes:
    varied_gpu = any(len({metas[i]["gpu_key"] for i in indexes}) > 1 for indexes in model_groups.values())
    varied_workload = any(len({metas[i]["workload_key"] for i in indexes}) > 1 for indexes in model_groups.values())
    if not varied_gpu: global_errors.append("matrix does not test the same model under different GPU/VRAM snapshots")
    if not varied_workload: global_errors.append("matrix does not test the same model under different context/KV choices")
  passed = not global_errors and all(x["passed"] for x in checked)
  return {"schema": SCHEMA, "passed": passed, "errors": sorted(set(global_errors)), "cases": list(checked),
          "outcomes_covered": sorted(outcomes)}


def validate_boundary_gate_json(cases: Sequence[Mapping[str, Any]], *, require_all_outcomes: bool = True) -> str:
  return stable_json(validate_boundary_gate(cases, require_all_outcomes=require_all_outcomes))


__all__ = ["SCHEMA", "stable_json", "validate_boundary_gate", "validate_boundary_gate_json"]
