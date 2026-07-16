"""Adapters from existing guarded artifacts to autoscan's strict evidence.

Collection is injected.  Production collectors must call the existing
isolated guarded infrastructure; this module performs no GPU execution and
does not manufacture absent evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from extra.qk.memory_adaptive_autoscan import AutoscanCandidate

SCHEMA = "tinygrad.memory_adaptive_guarded_evidence.v2"


def _mapping(value: Any) -> Mapping[str, Any]:
  if isinstance(value, Mapping): return value
  to_dict = getattr(value, "to_dict", None)
  if callable(to_dict):
    result = to_dict()
    if isinstance(result, Mapping): return result
  raise TypeError("evidence artifacts must be mappings or expose to_dict()")


def _phases(result: Any) -> dict[str, Mapping[str, Any]]:
  row = _mapping(result)
  phases = row.get("phases", ())
  if not isinstance(phases, (list, tuple)): raise ValueError("execution artifact phases must be a sequence")
  out = {}
  for phase in phases:
    item = _mapping(phase)
    name = item.get("phase")
    if not isinstance(name, str) or not name or name in out: raise ValueError("execution phases require unique names")
    out[name] = item
  return out


def _pass(ok: bool, artifact: Any) -> dict[str, Any]:
  return {"status": "PASS" if ok else "FAIL", "artifact": dict(_mapping(artifact))}


@dataclass(frozen=True)
class CandidateArtifacts:
  execution: Any
  resource: Any
  route_census: Any
  end_to_end_timing: Any
  whole_policy_identity: str
  compile: Any | None = None


class EvidenceAdapter:
  """Strict translation; missing or ambiguous facts become failed gates."""

  def translate(self, candidate: AutoscanCandidate, artifacts: CandidateArtifacts | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(artifacts, Mapping): artifacts = CandidateArtifacts(**artifacts)
    if not isinstance(artifacts, CandidateArtifacts): raise TypeError("collector must return CandidateArtifacts or a mapping")
    identity_ok = artifacts.whole_policy_identity == candidate.whole_policy_identity
    phases = _phases(artifacts.execution)
    compile_artifact = artifacts.compile if artifacts.compile is not None else phases.get("compile", {})
    compile_row = _mapping(compile_artifact)
    correctness = phases.get("correctness", {})
    correctness_evidence = _mapping(correctness.get("evidence", {})) if isinstance(correctness, Mapping) else {}
    correctness_ok = (correctness.get("status") == "passed" and
      correctness_evidence.get("full_output_compared") is True and correctness_evidence.get("numerical_passed") is True and
      correctness_evidence.get("finite_output") is True and correctness_evidence.get("inputs_unchanged") is True)

    execution = phases.get("execution", {})
    execution_evidence = _mapping(execution.get("evidence", {})) if isinstance(execution, Mapping) else {}
    health = _mapping(execution_evidence.get("health", {})) if isinstance(execution_evidence, Mapping) else {}
    health_ok = (execution.get("status") == "passed" and execution_evidence.get("dispatch_state") == "completed" and
                 health.get("preflight") is True and health.get("postflight") is True and health.get("device_fault") is False)

    resource = _mapping(artifacts.resource)
    resources = _mapping(resource.get("resources", {})) if isinstance(resource.get("resources", {}), Mapping) else {}
    # check_mmq_resource_evidence returns the validated trace unchanged.  Its
    # complete final-code-object shape is therefore also an accepted authority.
    checked_trace = resource.get("schema") == "tinygrad.kernel_resource_trace.v1" and all(key in resources for key in
      ("vgpr", "lds_bytes", "scratch_bytes", "vgpr_spills", "sgpr_spills", "workgroup_threads",
       "max_workgroup_threads", "wavefront_size", "occupancy")) and all(resources.get(key) == 0 for key in
      ("scratch_bytes", "vgpr_spills", "sgpr_spills"))
    resource_ok = resource.get("status") in ("PASS", "passed") or resource.get("passed") is True or checked_trace
    census = _mapping(artifacts.route_census)
    observed = census.get("covered_invocations", census.get("observed_invocations", ()))
    required = set(candidate.memory.required_invocations)
    explicit_complete = census.get("complete") is True and isinstance(observed, (list, tuple)) and set(observed) == required
    existing_complete = census.get("schema") == "prefill-candidate-set-route-census.v1" and \
      census.get("passed") is True and census.get("expected_entry_count") == census.get("selected_entry_count") == len(required) and \
      all(census.get(key) == [] for key in ("missing", "unexpected", "identity_mismatches"))
    census_ok = identity_ok and census.get("whole_policy_identity") == candidate.whole_policy_identity and \
      (census.get("status") in ("PASS", "passed") or census.get("passed") is True) and (explicit_complete or existing_complete)

    timing = _mapping(artifacts.end_to_end_timing)
    samples = timing.get("samples", timing.get("samples_tok_s"))
    timing_ok = timing.get("scope") == "end_to_end" and timing.get("metric") in ("tok_s", "end_to_end_tok_s") and isinstance(samples, (list, tuple))
    timing_record = {**dict(timing), "metric": "tok_s", "samples": list(samples)} if timing_ok else \
      {"scope": timing.get("scope"), "metric": timing.get("metric"), "samples": []}
    return {"schema": SCHEMA, "candidate_id": candidate.candidate_id,
      "whole_policy_identity": candidate.whole_policy_identity,
      "compile": _pass(identity_ok and (compile_row.get("status") in ("passed", "PASS") or compile_row.get("passed") is True), compile_row),
      "correctness": _pass(identity_ok and correctness_ok, correctness), "resource": _pass(identity_ok and resource_ok, resource),
      "gpu_health": _pass(identity_ok and health_ok, execution),
      "route_census": {**_pass(census_ok, census), "complete": bool(census_ok)},
      "end_to_end_timing": timing_record if identity_ok else {**timing_record, "samples": []}}


ArtifactCollector = Callable[[AutoscanCandidate], CandidateArtifacts | Mapping[str, Any] | None]


def make_evidence_runner(collector: ArtifactCollector, adapter: EvidenceAdapter | None = None) -> Callable[[AutoscanCandidate], Mapping[str, Any] | None]:
  """Inject collection while keeping all real dispatch in existing guards."""
  if not callable(collector): raise TypeError("collector must be callable")
  translator = adapter or EvidenceAdapter()
  def run(candidate: AutoscanCandidate) -> Mapping[str, Any] | None:
    artifacts = collector(candidate)
    return None if artifacts is None else translator.translate(candidate, artifacts)
  return run


__all__ = ["SCHEMA", "ArtifactCollector", "CandidateArtifacts", "EvidenceAdapter", "make_evidence_runner"]
