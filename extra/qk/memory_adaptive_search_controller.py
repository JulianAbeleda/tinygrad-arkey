#!/usr/bin/env python3
"""Executable production seam for memory-adaptive whole-model autoscan.

The controller owns orchestration, not model loading or GPU dispatch.  A
production seam must inspect the user-selected model, enumerate every offered
policy, and perform an actual guarded whole-model run.  This module joins that
work to the existing device scanner, complete-policy catalog, evidence adapter,
memory planner, policy selector, and exact-fact cache.

The production entry constructs its concrete whole-model seam internally. The
caller selects only the model (plus workload/cache controls); device discovery,
candidate enumeration, and guarded execution are not injectable CLI choices.
"""
from __future__ import annotations

import argparse, json, os, pathlib, re, sys, tempfile
from dataclasses import replace
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

if __package__ in (None, ""):
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from extra.qk.memory_adaptive_autoscan import autoscan_selected_model
from extra.qk.memory_adaptive_candidate_catalog import CandidateSpec, build_candidate_catalog
from extra.qk.memory_adaptive_evidence_runner import CandidateArtifacts, EvidenceAdapter
from extra.qk.memory_adaptive_allocation_observer import validate_memory_facts
from extra.qk.memory_adaptive_policy import canonical_json
from tinygrad.llm.admission import scanned_device_memory_budget
from tinygrad.llm.device_facts import DeviceFacts, MemoryReservePolicy, scan_device_facts
from tinygrad.llm.prefill_memory_plan import ByteLifetime, ByteTerm, Strategy
from tinygrad.llm.prefill_workload_plan import LiveMemoryFacts, PrefillRequest, plan_prefill_workload

SCHEMA = "tinygrad.memory_adaptive_search_controller.v1"


@dataclass(frozen=True)
class SelectedModelScan:
  """Exact facts discovered from the model selected by the user."""
  facts: Mapping[str, Any]
  inventory: Mapping[str, Any]
  base_terms: Sequence[ByteTerm]
  workload: Mapping[str, Any]
  compiler_runtime_revision: Mapping[str, Any]


@runtime_checkable
class ProductionWholeModelSeam(Protocol):
  """Truthful injection point until the runtime has a unified whole-model API."""
  def scan_selected_model(self, model_path: str, device: DeviceFacts) -> SelectedModelScan: ...
  def enumerate_candidate_specs(self, model: SelectedModelScan, device: DeviceFacts) -> Sequence[CandidateSpec]: ...
  def collect_whole_model_artifacts(self, model_path: str, model: SelectedModelScan, candidate: Any,
                                    *, samples: int) -> CandidateArtifacts | Mapping[str, Any] | None: ...


def _refuse(reason: str, *, blocker: bool = False) -> dict[str, Any]:
  return {"schema": SCHEMA, "decision": Strategy.REFUSE.value, "selected_candidate_id": None,
          "reason": reason, "blocker": reason if blocker else None, "from_cache": False,
          "cache_record": None}


def _target_facts(device: DeviceFacts) -> dict[str, Any]:
  row = device.planning_snapshot()
  # Candidate requirements match only scanned structural facts.  Free/total
  # memory remain planning inputs and are never converted to a named tier.
  return {**row, "backend": device.backend, "architecture": device.architecture,
          "capabilities": device.capabilities.to_json()}


def _exception_diagnostic(candidate_id: str, exc: Exception) -> dict[str, Any]:
  """Return safe, deterministic failure facts without exposing a traceback."""
  message = str(exc).replace("\r", " ").replace("\n", " ")[:256]
  return {"candidate_id": candidate_id, "actual_whole_model_run": False,
          "exception_type": type(exc).__name__[:128], "exception_message": message,
          "blockers": [f"whole-model evidence exception: {type(exc).__name__}"],
          "memory_fact_evidence": False}


_SUMMARY_MAPS = ("count_by_phase", "bytes_by_phase", "count_by_category", "bytes_by_category",
                 "unowned_count_by_phase", "unowned_requested_bytes_by_phase",
                 "unowned_count_by_category", "unowned_requested_bytes_by_category")


def _bounded_summary(value: Any) -> dict[str, Any] | None:
  """Copy only numeric, aggregate ledger facts; never copy lifetimes or identities."""
  if not isinstance(value, Mapping): return None
  out: dict[str, Any] = {}
  for key in ("allocation_count", "requested_bytes", "physical_bytes", "binding_count", "bound_count",
              "unbound_count", "reuse_count", "reused_count"):
    item = value.get(key)
    if isinstance(item, int) and not isinstance(item, bool) and item >= 0: out[key] = item
  for key in _SUMMARY_MAPS:
    item = value.get(key)
    if not isinstance(item, Mapping): continue
    rows = {}
    for name in sorted(item, key=str)[:64]:
      number = item[name]
      if isinstance(name, str) and len(name) <= 96 and isinstance(number, int) and not isinstance(number, bool) and number >= 0:
        rows[name] = number
    if rows: out[key] = rows
  for key in ("binding_presence", "reuse"):
    item = value.get(key)
    if isinstance(item, Mapping):
      rows = {str(name)[:96]: item[name] for name in sorted(item, key=str)[:32]
              if isinstance(name, str) and isinstance(item[name], (bool, int))}
      if rows: out[key] = rows
  return out or None


def _failure_category(message: Any) -> str:
  text = str(message).lower()
  for pattern, category in (("manifest/evidence", "manifest_evidence"), ("ownership", "ownership"),
                            ("physical", "physical_ledger"), ("manifest", "manifest"),
                            ("allocation", "allocation"), ("cleanup", "cleanup"),
                            ("route", "route_census"), ("missing required artifact", "missing_artifact"),
                            ("correctness", "correctness"), ("device", "device")):
    if pattern in text: return category
  return "other"


def _diagnostic_projection(candidate_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
  blockers = raw.get("blockers", ())
  blockers = blockers if isinstance(blockers, (list, tuple)) else ()
  aggregate: dict[tuple[int | None, str], int] = {}
  for message in blockers:
    manifest = None
    match = re.search(r"schedule manifest (\d+)", str(message), re.IGNORECASE)
    if match: manifest = int(match.group(1))
    key = (manifest, _failure_category(message))
    aggregate[key] = aggregate.get(key, 0) + 1
  failures = [{"manifest": manifest, "category": category, "count": count}
              for (manifest, category), count in sorted(aggregate.items(), key=lambda x: (x[0][0] is None, x[0][0] or -1, x[0][1]))]
  measured = raw.get("measured_allocation")
  measured_ledger = measured.get("physical_ledger") if isinstance(measured, Mapping) else None
  summary = measured_ledger.get("structural_summary") if isinstance(measured_ledger, Mapping) else None
  if summary is None:
    physical_ledger = raw.get("physical_memory_ledger")
    summary = physical_ledger.get("structural_summary") if isinstance(physical_ledger, Mapping) else None
  return {"candidate_id": candidate_id, "actual_whole_model_run": raw.get("actual_whole_model_run"),
          "blocker_count": len(blockers), "schedule_failure_summary": failures[:128],
          "physical_structural_summary": _bounded_summary(summary),
          "memory_fact_evidence": raw.get("memory_fact_evidence") is not None}


def _run_controller_with_seam(*, model_path: str, seam: ProductionWholeModelSeam,
                              cache_record: Mapping[str, Any] | None = None, min_samples: int = 3) -> dict[str, Any]:
  """Scan, enumerate complete feasible policies, benchmark baseline first, and select.

  Candidate execution is accepted only when the injected seam explicitly marks
  it as an actual whole-model run.  All other evidence is refused; the
  controller never upgrades kernel-only, replayed, or synthetic data.
  """
  if min_samples < 3: raise ValueError("production autoscan requires at least 3 end-to-end tok/s samples")
  for method in ("scan_selected_model", "enumerate_candidate_specs", "collect_whole_model_artifacts"):
    if not callable(getattr(seam, method, None)): raise TypeError(f"production seam is missing {method}()")

  # Production hardware authority is one internal live scan. Tests replace this
  # module-level scanner; callers cannot inject facts, a device, or a reserve.
  device = scan_device_facts()
  if not isinstance(device, DeviceFacts): raise TypeError("device scanner must return DeviceFacts")
  scanned_budget = scanned_device_memory_budget(device)
  if scanned_budget.admitted_bytes is None:
    return _refuse("live device memory facts are incomplete: total/free VRAM and allocator granularity are required")
  reserve_policy = MemoryReservePolicy(fixed_bytes=scanned_budget.reserve_bytes)
  model = seam.scan_selected_model(model_path, device)
  # A CLI executed as ``__main__`` and a seam importing this module canonically can hold equivalent dataclass types
  # with distinct Python identities. Validate the protocol payload, then normalize it into this module's authority.
  if not all(hasattr(model, field) for field in ("facts", "inventory", "base_terms", "workload", "compiler_runtime_revision")):
    raise TypeError("scan_selected_model must return the SelectedModelScan structural contract")
  model = SelectedModelScan(model.facts, model.inventory, model.base_terms, model.workload, model.compiler_runtime_revision)
  specs = tuple(seam.enumerate_candidate_specs(model, device))
  base_catalog = build_candidate_catalog(selected_model_inventory=model.inventory,
    target_capabilities=_target_facts(device), candidate_specs=specs)
  if not base_catalog: return _refuse("candidate enumeration produced no complete target-supported policy")

  baselines = [x for x in base_catalog if x.memory.strategy is Strategy.DIRECT_PACKED_FALLBACK]
  if len(baselines) != 1:
    return _refuse("complete policy set must contain exactly one DIRECT_PACKED_FALLBACK baseline")
  prompt = model.workload.get("prompt_tokens")
  context = model.workload.get("context_tokens", prompt)
  try: request = PrefillRequest(prompt, context)
  except (TypeError, ValueError) as exc: return _refuse(f"selected workload facts are incomplete: {exc}")
  spec_by_id = {x.candidate_id: x for x in specs}
  catalog = []
  for policy in base_catalog:
    spec = spec_by_id[policy.candidate_id]
    live_values = tuple(x.bytes for x in (*model.base_terms, *policy.memory.memory_terms))
    live = None if any(x is None for x in live_values) else sum(live_values)  # type: ignore[arg-type]
    workload_plan = plan_prefill_workload(request=request,
      memory=LiveMemoryFacts(live, scanned_budget.admitted_bytes), candidates=(spec.kernel_capability(),))
    for choice in workload_plan.feasible_choices:
      machine_id = choice.machine_candidate_id
      physical_ms = sorted({choice.full_m} | ({choice.remainder_physical_m} if choice.remainder_call_count else set()))
      workload_term = ByteTerm(f"prefill physical invocation peak M={','.join(map(str, physical_ms))}", choice.peak_incremental_bytes,
        "candidate-published exact per-M activation and scratch bytes",
        "max(activation_bytes[physical M] + scratch_bytes[physical M]) over full and mapped remainder calls", ByteLifetime.CANDIDATE_WORKSPACE)
      memory = replace(policy.memory, candidate_id=machine_id,
                       memory_terms=(*policy.memory.memory_terms, workload_term))
      choice_record = choice.to_dict()
      catalog.append(type(policy)(memory, {**policy.policy, "candidate_id": machine_id,
        "policy_candidate_id": policy.candidate_id, "workload_choice": choice_record}))
  catalog = tuple(catalog)
  if not catalog:
    return _refuse("no complete policy+M workload choice has exact correctness, remainder coverage, and per-M bytes")
  baseline_ids = sorted(x.candidate_id for x in catalog if x.memory.strategy is Strategy.DIRECT_PACKED_FALLBACK)
  baseline_id = baseline_ids[0]
  adapter = EvidenceAdapter()
  diagnostics: dict[str, Any] = {}

  # A cache key includes measured facts. Hydrate only when the cached candidate
  # is otherwise byte-for-byte the same canonical policy; malformed bundles
  # are deliberately ignored and therefore cannot produce a cache hit.
  if isinstance(cache_record, Mapping) and isinstance(cache_record.get("result"), Mapping):
    cached_candidates = cache_record["result"].get("canonical_inputs", {}).get("candidates", ())
    if isinstance(cached_candidates, list):
      cached_by_id = {x.get("candidate_id"): x for x in cached_candidates if isinstance(x, Mapping)}
      for candidate in catalog:
        cached_candidate = cached_by_id.get(candidate.candidate_id)
        bundle = cached_candidate.get("memory_fact_evidence") if isinstance(cached_candidate, Mapping) else None
        if candidate.memory.strategy is Strategy.DIRECT_PACKED_FALLBACK or bundle is None: continue
        stripped = {k: v for k, v in cached_candidate.items() if k not in ("memory_facts", "memory_fact_evidence")}
        if canonical_json(stripped) == canonical_json(candidate.policy_record()):
          validated = validate_memory_facts(bundle, candidate_id=candidate.candidate_id)
          if validated is not None and isinstance(candidate.policy, dict):
            candidate.policy["memory_fact_evidence"] = validated
            candidate.policy["memory_facts"] = dict(validated["facts"])

  def evidence_runner(candidate: Any) -> Mapping[str, Any] | None:
    try:
      raw = seam.collect_whole_model_artifacts(model_path, model, candidate, samples=min_samples)
    except Exception as exc:
      diagnostics[candidate.candidate_id] = _exception_diagnostic(candidate.candidate_id, exc)
      return None
    if raw is None:
      diagnostics[candidate.candidate_id] = {"candidate_id": candidate.candidate_id,
        "actual_whole_model_run": False, "blockers": ["seam returned no result"],
        "memory_fact_evidence": False}
      return None
    if isinstance(raw, Mapping):
      diagnostics[candidate.candidate_id] = _diagnostic_projection(candidate.candidate_id, raw)
      measured = raw.get("measured_allocation")
      if diagnostics[candidate.candidate_id]["physical_structural_summary"] is None:
        diagnostics[candidate.candidate_id]["physical_structural_summary"] = _bounded_summary(raw.get("physical_memory_ledger"))
      if raw.get("actual_whole_model_run") is not True: return {"incomplete": True}
      payload = raw.get("artifacts")
      if payload is None: return {"incomplete": True}
      if candidate.memory.strategy is not Strategy.DIRECT_PACKED_FALLBACK:
        bundle = validate_memory_facts(raw.get("memory_fact_evidence"), candidate_id=candidate.candidate_id)
        if bundle is None: return {"incomplete": True, "memory_facts": "missing, partial, or unproven"}
        if not isinstance(candidate.policy, dict): return {"incomplete": True}
        candidate.policy["memory_fact_evidence"] = bundle
        candidate.policy["memory_facts"] = dict(bundle["facts"])
    else:
      # CandidateArtifacts has no whole-model attestation field, so accepting it
      # directly would make a microbenchmark indistinguishable from production.
      return {"incomplete": True}
    return adapter.translate(candidate, payload)

  result = autoscan_selected_model(selected_model_facts=model.facts, selected_model_inventory=model.inventory,
    base_terms=model.base_terms, candidates=catalog, workload=model.workload,
    compiler_runtime_revision=model.compiler_runtime_revision, evidence_runner=evidence_runner,
    baseline_candidate_id=baseline_id, device_facts=device, reserve_policy=reserve_policy,
    cache_record=cache_record, min_samples=min_samples)
  return {**result, "controller_schema": SCHEMA, "candidate_diagnostics": diagnostics}


def _production_seam() -> ProductionWholeModelSeam:
  # Lazy import avoids the protocol module's type-only back-reference while keeping
  # candidate enumeration and execution machinery internal to the production entry.
  from extra.qk.memory_adaptive_tinygrad_seam import TinygradWholeModelSeam
  return TinygradWholeModelSeam()


def run_controller(*, model_path: str, cache_record: Mapping[str, Any] | None = None,
                   min_samples: int = 3) -> dict[str, Any]:
  """Production machine search: the caller selects only the model."""
  return _run_controller_with_seam(model_path=model_path, seam=_production_seam(),
                                   cache_record=cache_record, min_samples=min_samples)


def _read_cache(path: str | None) -> Mapping[str, Any] | None:
  if not path or not pathlib.Path(path).is_file(): return None
  value = json.loads(pathlib.Path(path).read_text())
  return value if isinstance(value, Mapping) else None


def _write_cache(path: str, record: Mapping[str, Any]) -> None:
  target = pathlib.Path(path)
  target.parent.mkdir(parents=True, exist_ok=True)
  fd, temporary = tempfile.mkstemp(prefix=target.name+".", dir=target.parent)
  try:
    with os.fdopen(fd, "w") as handle: json.dump(record, handle, sort_keys=True, separators=(",", ":"))
    os.replace(temporary, target)
  finally:
    if os.path.exists(temporary): os.unlink(temporary)


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", required=True, help="user-selected model path; never used as a policy selector")
  parser.add_argument("--cache", help="exact-fact policy cache JSON path")
  parser.add_argument("--samples", type=int, default=3, help="end-to-end samples per policy (minimum 3)")
  args = parser.parse_args(argv)
  try:
    result = run_controller(model_path=args.model, cache_record=_read_cache(args.cache), min_samples=args.samples)
    if args.cache and isinstance(result.get("cache_record"), Mapping): _write_cache(args.cache, result["cache_record"])
  except Exception as exc:
    result = _refuse(f"controller error: {type(exc).__name__}: {exc}")
  print(json.dumps(result, sort_keys=True, separators=(",", ":")))
  return 0 if result.get("decision") == "SELECTED" else 2


if __name__ == "__main__": raise SystemExit(main())

__all__ = ["SCHEMA", "SelectedModelScan", "ProductionWholeModelSeam", "run_controller", "main"]
