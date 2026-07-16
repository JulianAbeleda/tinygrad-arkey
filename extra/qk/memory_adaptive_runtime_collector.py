"""Fail-closed bridge from completed machine-search caches to ``from_gguf``.

The model-facing request contains only facts observed while opening the selected
GGUF.  Configuration which cannot be observed there (the candidate set and
compiler/search revisions) may be pinned by the collector factory.  No model
name, profile, path, or size-class matching is performed.
"""
from __future__ import annotations

import hashlib, json, pathlib
from typing import Any, Mapping, Sequence

from extra.qk.memory_adaptive_policy import CACHE_SCHEMA, SCHEMA as POLICY_SCHEMA, canonical_json, canonical_search_key
from extra.qk.memory_adaptive_allocation_observer import validate_memory_facts

REQUEST_SCHEMA = "tinygrad.model_memory_adaptive_request.v1"


def _candidate_set_identity(candidates: Sequence[Mapping[str, Any]]) -> str:
  rows = sorted((json.loads(canonical_json(x)) for x in candidates), key=canonical_json)
  return "sha256:" + hashlib.sha256(canonical_json(rows).encode("utf-8")).hexdigest()


def _completed_cache(value: Mapping[str, Any]) -> tuple[Mapping[str, Any], str] | None:
  """Extract a cache and provenance without accepting partial controller state."""
  if value.get("schema") == CACHE_SCHEMA: return value, "exact_cache"
  cache = value.get("cache_record")
  if (value.get("decision") != "SELECTED" or value.get("interrupted") is not False or
      value.get("selected_candidate_id") is None or not isinstance(cache, Mapping)):
    return None
  result = cache.get("result")
  if (not isinstance(result, Mapping) or value.get("selected_candidate_id") != result.get("selected_candidate_id") or
      (isinstance(value.get("policy"), Mapping) and value.get("policy") != result)):
    return None
  return cache, "measured" if value.get("from_cache") is False else "exact_cache"


def collect_runtime_policy(request: Mapping[str, Any], source: Mapping[str, Any], *,
                           compiler_runtime_revision: Mapping[str, Any] | None = None,
                           search_revision: str | None = None,
                           candidate_set_identity: str | None = None) -> dict[str, Any] | None:
  """Validate one completed result and return the exact model collector shape."""
  try:
    if not isinstance(request, Mapping) or request.get("schema") != REQUEST_SCHEMA: return None
    inventory, device, runtime_workload = request.get("inventory"), request.get("device_facts"), request.get("workload")
    if not isinstance(inventory, Mapping) or not isinstance(device, Mapping) or not isinstance(runtime_workload, Mapping): return None
    ubatch = runtime_workload.get("prefill_ubatch")
    if not isinstance(ubatch, int) or isinstance(ubatch, bool) or ubatch <= 0: return None

    extracted = _completed_cache(source)
    if extracted is None: return None
    cache, validation = extracted
    result = cache.get("result")
    if cache.get("schema") != CACHE_SCHEMA or not isinstance(result, Mapping): return None
    inputs = result.get("canonical_inputs")
    if (result.get("schema") != POLICY_SCHEMA or result.get("decision") != "SELECTED" or
        not isinstance(inputs, Mapping) or inputs.get("schema") != POLICY_SCHEMA): return None
    candidates = inputs.get("candidates")
    revisions = inputs.get("compiler_runtime_revision")
    revision = inputs.get("search_revision")
    if not isinstance(candidates, list) or not all(isinstance(x, Mapping) for x in candidates): return None
    if not isinstance(revisions, Mapping) or not isinstance(revision, str) or not revision: return None

    key_args = {"gpu_facts": inputs.get("gpu_facts"), "model_facts": inputs.get("model_facts"),
                "workload": inputs.get("workload"), "candidates": candidates,
                "compiler_runtime_revision": revisions, "search_revision": revision}
    expected_key = canonical_search_key(**key_args)
    if cache.get("search_key") != expected_key or result.get("search_key") != expected_key: return None
    if compiler_runtime_revision is not None and canonical_json(revisions) != canonical_json(compiler_runtime_revision): return None
    if search_revision is not None and revision != search_revision: return None
    actual_set_identity = _candidate_set_identity(candidates)
    if candidate_set_identity is not None and actual_set_identity != candidate_set_identity: return None
    request_set_identity = request.get("candidate_set_identity")
    if request_set_identity is not None and request_set_identity != actual_set_identity: return None

    model_facts = inputs.get("model_facts")
    cached_inventory = model_facts.get("inventory") if isinstance(model_facts, Mapping) else None
    if canonical_json(cached_inventory) != canonical_json(inventory): return None
    if canonical_json(inputs.get("gpu_facts")) != canonical_json(device): return None

    selected_id = result.get("selected_candidate_id")
    selected = [x for x in candidates if x.get("candidate_id") == selected_id]
    accepted = result.get("accepted_candidates")
    if (not isinstance(selected_id, str) or len(selected) != 1 or not isinstance(accepted, list) or
        sum(isinstance(x, Mapping) and x.get("candidate_id") == selected_id for x in accepted) != 1): return None
    policy = dict(selected[0])
    if policy.get("strategy") != "DIRECT_PACKED_FALLBACK":
      bundle = validate_memory_facts(policy.get("memory_fact_evidence"), candidate_id=selected_id)
      if bundle is None or policy.get("memory_facts") != bundle["facts"]: return None
      policy["memory_fact_evidence"], policy["memory_facts"] = bundle, dict(bundle["facts"])
    elif ("memory_facts" in policy or "memory_fact_evidence" in policy):
      bundle = validate_memory_facts(policy.get("memory_fact_evidence"), candidate_id=selected_id)
      if bundle is None or policy.get("memory_facts") != bundle["facts"]: return None
    choice = policy.get("workload_choice")
    if (not isinstance(choice, Mapping) or choice.get("full_m") != ubatch or choice.get("feasible") is not True or
        choice.get("candidate_id") != policy.get("policy_candidate_id")): return None
    rows = inventory.get("rows")
    routes = policy.get("routes")
    if not isinstance(rows, list) or not isinstance(routes, Mapping): return None
    ids = [x.get("invocation_id") if isinstance(x, Mapping) else None for x in rows]
    if any(not isinstance(x, str) or not x for x in ids) or len(ids) != len(set(ids)) or set(routes) != set(ids): return None
    if any(not isinstance(x, str) or not x for x in routes.values()): return None
    return {"decision": "SELECTED", "validation": validation, "validated_request": dict(request), "policy": policy}
  except (KeyError, TypeError, ValueError, OverflowError):
    return None


def make_policy_collector(source: Mapping[str, Any], **pins: Any):
  """Return a callable suitable for ``Transformer.from_gguf(policy_collector=...)``."""
  snapshot = json.loads(json.dumps(source))
  return lambda request: collect_runtime_policy(request, snapshot, **pins)


def make_file_policy_collector(path: str | pathlib.Path, **pins: Any):
  """Load a JSON cache/controller file for each call (useful for later CLI wiring)."""
  target = pathlib.Path(path)
  def collector(request):
    try:
      value = json.loads(target.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
      return None
    return collect_runtime_policy(request, value, **pins) if isinstance(value, Mapping) else None
  return collector


__all__ = ["REQUEST_SCHEMA", "collect_runtime_policy", "make_policy_collector", "make_file_policy_collector"]

def _decode_candidate_set(value):
  from extra.qk.runtime_specs import FullKernelCandidateSet, admit_full_kernel_candidate_set
  return admit_full_kernel_candidate_set(FullKernelCandidateSet.from_json(value))

def install_model_adapters() -> None:
  from tinygrad.llm.memory_adaptive_authority import register_memory_adaptive_adapters
  register_memory_adaptive_adapters(policy_adapter=collect_runtime_policy, evidence_validator=validate_memory_facts,
                                    candidate_set_decoder=_decode_candidate_set)

install_model_adapters()
__all__.append("install_model_adapters")
