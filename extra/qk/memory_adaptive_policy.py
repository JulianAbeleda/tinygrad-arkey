"""Pure, deterministic machine-policy selection for memory-safe prefill candidates.

This module deliberately performs no device discovery or benchmarking.  Callers
provide canonical facts and guarded evidence collected by an isolated executor.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import statistics
from typing import Any, Mapping, Sequence


SCHEMA = "tinygrad.memory_adaptive_policy.v1"
CACHE_SCHEMA = "tinygrad.memory_adaptive_policy_cache.v1"
OBJECTIVE = "steady_state_end_to_end_tok_s"
_NON_SEMANTIC_KEYS = frozenset({
  "filename", "file_name", "model_filename", "model_path", "path",
  "model_name", "display_name", "size_label", "model_size_label",
  "profile", "profile_id", "benchmark_profile",
})
_REQUIRED_GATES = ("correctness", "resource", "gpu_health", "route_census")


def _canonical(value: Any) -> Any:
  """Return a JSON-shaped canonical value, dropping forbidden identity labels."""
  if isinstance(value, Mapping):
    return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            if str(k).lower().replace("-", "_") not in _NON_SEMANTIC_KEYS}
  if isinstance(value, (list, tuple)):
    return [_canonical(v) for v in value]
  if value is None or isinstance(value, (str, bool, int)):
    return value
  if isinstance(value, float):
    if not math.isfinite(value): raise ValueError("canonical facts must contain only finite numbers")
    return value
  raise TypeError(f"facts must be JSON-serializable, got {type(value).__name__}")


def canonical_json(value: Any) -> str:
  return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def canonical_search_record(*, gpu_facts: Mapping[str, Any], model_facts: Mapping[str, Any],
                            workload: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]],
                            compiler_runtime_revision: Mapping[str, Any], search_revision: str = SCHEMA) -> dict[str, Any]:
  """Material inputs to search identity. Candidate ordering is non-semantic."""
  canonical_candidates = [_canonical(candidate) for candidate in candidates]
  canonical_candidates.sort(key=canonical_json)
  return {
    "schema": SCHEMA,
    "search_revision": search_revision,
    "gpu_facts": _canonical(gpu_facts),
    "model_facts": _canonical(model_facts),
    "workload": _canonical(workload),
    "candidates": canonical_candidates,
    "compiler_runtime_revision": _canonical(compiler_runtime_revision),
  }


def canonical_search_key(**kwargs: Any) -> str:
  payload = canonical_json(canonical_search_record(**kwargs)).encode("utf-8")
  return "sha256:" + hashlib.sha256(payload).hexdigest()


def _passed(record: Any) -> bool:
  return isinstance(record, Mapping) and record.get("status") == "PASS"


@dataclass(frozen=True)
class TimingSummary:
  samples_tok_s: tuple[float, ...]
  median_tok_s: float
  mad_tok_s: float
  relative_noise: float
  confidence_low_tok_s: float
  confidence_high_tok_s: float

  def to_json(self) -> dict[str, Any]:
    return {
      "metric": "end_to_end_tok_s", "samples_tok_s": list(self.samples_tok_s),
      "median_tok_s": self.median_tok_s, "mad_tok_s": self.mad_tok_s,
      "relative_noise": self.relative_noise,
      "confidence_interval_tok_s": [self.confidence_low_tok_s, self.confidence_high_tok_s],
    }


def _timing_summary(record: Any, *, min_samples: int) -> tuple[TimingSummary | None, str | None]:
  if not isinstance(record, Mapping): return None, "missing end-to-end timing evidence"
  if record.get("scope") != "end_to_end" or record.get("metric") != "tok_s":
    return None, "timing evidence must be end-to-end tok/s"
  samples = record.get("samples")
  if not isinstance(samples, (list, tuple)) or len(samples) < min_samples:
    return None, f"end-to-end timing requires at least {min_samples} samples"
  if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in samples):
    return None, "end-to-end tok/s samples must be finite and positive"
  vals = tuple(float(v) for v in samples)
  median = statistics.median(vals)
  mad = statistics.median(abs(v-median) for v in vals)
  interval = record.get("confidence_interval_tok_s")
  if interval is None:
    radius = 1.4826 * mad
    low, high = max(0.0, median-radius), median+radius
  elif (not isinstance(interval, (list, tuple)) or len(interval) != 2 or
        any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in interval)):
    return None, "confidence interval must contain two finite tok/s values"
  else:
    low, high = float(interval[0]), float(interval[1])
    if low < 0 or low > median or high < median: return None, "confidence interval must contain the sample median"
  return TimingSummary(vals, median, mad, mad/median, low, high), None


def _candidate_id(candidate: Mapping[str, Any]) -> str:
  value = candidate.get("candidate_id")
  if not isinstance(value, str) or not value: raise ValueError("every candidate requires a non-empty candidate_id")
  return value


def select_policy(*, gpu_facts: Mapping[str, Any], model_facts: Mapping[str, Any], workload: Mapping[str, Any],
                  candidates: Sequence[Mapping[str, Any]], compiler_runtime_revision: Mapping[str, Any],
                  evidence: Mapping[str, Mapping[str, Any]], baseline_candidate_id: str | None,
                  search_revision: str = SCHEMA, min_samples: int = 3, max_relative_noise: float = 0.05,
                  tie_relative_tolerance: float = 0.01) -> dict[str, Any]:
  """Gate and rank an already memory-safe feasible set. Never executes candidates."""
  if min_samples < 1: raise ValueError("min_samples must be positive")
  if max_relative_noise < 0 or tie_relative_tolerance < 0: raise ValueError("noise tolerances must be non-negative")
  ids = [_candidate_id(c) for c in candidates]
  if len(ids) != len(set(ids)): raise ValueError("candidate_id values must be unique")
  key_args = dict(gpu_facts=gpu_facts, model_facts=model_facts, workload=workload, candidates=candidates,
                  compiler_runtime_revision=compiler_runtime_revision, search_revision=search_revision)
  key = canonical_search_key(**key_args)
  accepted: list[dict[str, Any]] = []
  rejected: list[dict[str, Any]] = []
  for candidate_id in sorted(ids):
    proof = evidence.get(candidate_id)
    reasons = []
    if not isinstance(proof, Mapping): reasons.append("missing guarded evidence")
    else:
      for gate in _REQUIRED_GATES:
        if not _passed(proof.get(gate)): reasons.append(f"{gate} evidence missing or not PASS")
      if _passed(proof.get("route_census")) and proof["route_census"].get("complete") is not True:
        reasons.append("route_census evidence does not attest complete coverage")
      timing, timing_error = _timing_summary(proof.get("end_to_end_timing"), min_samples=min_samples)
      if timing_error is not None: reasons.append(timing_error)
      elif timing is not None and timing.relative_noise > max_relative_noise:
        reasons.append(f"relative timing noise {timing.relative_noise:.6g} exceeds {max_relative_noise:.6g}")
    if reasons:
      rejected.append({"candidate_id": candidate_id, "reasons": reasons})
    else:
      assert timing is not None
      accepted.append({"candidate_id": candidate_id, "timing": timing.to_json(), "_timing": timing})

  decision, selected_id, tie_ids = "REFUSE", None, []
  reason = "no candidate passed every correctness, resource, GPU-health, route-census, and timing gate"
  if accepted:
    best_median = max(row["_timing"].median_tok_s for row in accepted)
    best = next(row for row in accepted if row["_timing"].median_tok_s == best_median)
    credible = [row for row in accepted
                if row["_timing"].confidence_high_tok_s >= best["_timing"].confidence_low_tok_s
                or (best_median-row["_timing"].median_tok_s)/best_median <= tie_relative_tolerance]
    tie_ids = sorted(row["candidate_id"] for row in credible)
    selected_id = baseline_candidate_id if baseline_candidate_id in tie_ids else tie_ids[0]
    decision = "SELECTED"
    reason = "statistical tie resolved deterministically" if len(tie_ids) > 1 else "fastest statistically credible end-to-end tok/s"

  public_accepted = [{k: v for k, v in row.items() if k != "_timing"} for row in accepted]
  return {
    "schema": SCHEMA, "search_key": key, "objective": OBJECTIVE,
    "decision": decision, "selected_candidate_id": selected_id, "baseline_candidate_id": baseline_candidate_id,
    "decision_reason": reason, "tie_candidate_ids": tie_ids,
    "accepted_candidates": public_accepted, "rejected_candidates": rejected,
    "selection_parameters": {"min_samples": min_samples, "max_relative_noise": max_relative_noise,
                             "tie_relative_tolerance": tie_relative_tolerance},
    "canonical_inputs": canonical_search_record(**key_args),
  }


def make_cache_record(result: Mapping[str, Any]) -> dict[str, Any]:
  if result.get("schema") != SCHEMA or not isinstance(result.get("search_key"), str):
    raise ValueError("not a memory-adaptive policy result")
  return {"schema": CACHE_SCHEMA, "search_key": result["search_key"], "result": _canonical(result)}


def cache_matches(cache_record: Mapping[str, Any], **search_key_args: Any) -> bool:
  """Exact-fact cache validation; false is the only response to malformed/stale data."""
  try:
    return (cache_record.get("schema") == CACHE_SCHEMA and
            cache_record.get("search_key") == canonical_search_key(**search_key_args) and
            isinstance(cache_record.get("result"), Mapping) and
            cache_record["result"].get("search_key") == cache_record["search_key"])
  except (TypeError, ValueError):
    return False
