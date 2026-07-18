"""Default-off, host-only execution surface for the retained six-row research policy.

This module does not participate in production dispatch and imports no Tensor or
device runtime.  A caller must explicitly enable it and provide both execution
callbacks.  Unknown workloads and policy/identity drift raise a blocking error;
the candidate path never silently calls the direct-packed callback.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from extra.qk.prefill.six_row_policy_artifact import SCHEMA as POLICY_SCHEMA, _identity as _policy_identity
from extra.qk.prefill_route_census import collect_prefill_route_execution_census
from extra.qk.route_manifest import canonical_candidate_set_identity
from tinygrad.llm.prefill_route_observer import (
  PrefillRouteAttachment, PrefillRouteExecution, notify_prefill_route_execution,
)


SCHEMA = "tinygrad.qk_six_row_research_host_dispatch.v1"
DEFAULT_POLICY = Path(__file__).resolve().parents[3] / "docs/qwen3-14b-prefill-six-row-research-policy-20260718.json"
TARGET = {"backend":"AMD", "arch":"gfx1100", "wave_size":32}
RETAINED_POLICY_IDENTITY = "qk_exact_six_row_policy:sha256:800ad8a2047b0efd69d81afc45f387af7c71a038e141e73ec7c855bebe44814e"
RETAINED_CANDIDATE_SET_IDENTITY = "candidate_set:sha256:9938c196f86e9acf9b98edf38ed914807e078522f85835985e704cddaf9f087b"
CANDIDATE_ROUTE = "q4k_q8_five_buffer_research"
DIRECT_PACKED_ROUTE = "direct_packed"


class ResearchPolicyBlocked(ValueError): pass


@dataclass(frozen=True)
class ResearchWorkload:
  phase: str
  quant_format: str
  role: str
  m: int
  n: int
  k: int
  backend: str
  arch: str
  wave_size: int

  @property
  def target(self) -> dict[str, Any]:
    return {"backend":self.backend, "arch":self.arch, "wave_size":self.wave_size}

  @property
  def key(self) -> tuple[Any, ...]:
    return (self.phase, self.quant_format, self.role, self.m, self.n, self.k,
            self.backend, self.arch, self.wave_size)


@dataclass(frozen=True)
class ResearchGroup:
  invocation_id: str
  workload: ResearchWorkload
  expected_binding_identity: str
  expected_calls: int


@dataclass(frozen=True)
class ResearchSelection:
  invocation_id: str
  workload: ResearchWorkload
  binding_kind: str
  route_id: str
  binding_identity: str
  candidate_set_identity: str
  inventory_identity: str
  capability_identity: str


@dataclass(frozen=True)
class HostProgramExecution:
  executed_route_id: str
  binding_identity: str
  program_identity: str


def _workload(quant: str, role: str, m: int, n: int, k: int) -> ResearchWorkload:
  return ResearchWorkload("prefill", quant, role, m, n, k, "AMD", "gfx1100", 32)


GROUPS = (
  ResearchGroup("q4_ffn_gate_up", _workload("Q4_K", "ffn_gate_up", 512, 17_408, 5_120),
                "3e367ee04b0aa5db66ecaad085ec9f5ea414a56b8161b406cdefb125378356eb", 80),
  ResearchGroup("q4_attn_qo", _workload("Q4_K", "attn_qo", 512, 5_120, 5_120),
                "fallback:sha256:6b8b0c16a9bfa333bee14c702c09ccaa8ff2b76aea1017537e05cc28f4a4f139", 80),
  ResearchGroup("q4_ffn_down", _workload("Q4_K", "ffn_down", 512, 5_120, 17_408),
                "fallback:sha256:711e195c353ef77345063794e800bddf0f6409d41efe8e9933fd650abbe5c1be", 20),
  ResearchGroup("q4_attn_kv", _workload("Q4_K", "attn_kv", 512, 1_024, 5_120),
                "fallback:sha256:e77147b9b68ae129b055331ff1b8a1da52649d469febbc47402f919ded843e73", 60),
  ResearchGroup("q6_ffn_down", _workload("Q6_K", "ffn_down", 512, 5_120, 17_408),
                "fallback:sha256:16a9a0aee716ca28fea39d22f537e2956a8b75f163348ba76c8170554092d61c", 20),
  ResearchGroup("q6_attn_kv", _workload("Q6_K", "attn_kv", 512, 1_024, 5_120),
                "fallback:sha256:655ecb5003b2cfdd9019240cd49ceaa0a047a4e897555abdb6641c4bb84b5a05", 20),
)


def load_retained_policy(path: str | Path = DEFAULT_POLICY) -> dict[str, Any]:
  value = json.loads(Path(path).read_text())
  if not isinstance(value, dict): raise ResearchPolicyBlocked("retained six-row policy must be a JSON object")
  return value


def _row_workload(row: Mapping[str, Any]) -> ResearchWorkload:
  try:
    shape, target = row["shape"], row["target"]
    values = (shape["m"], shape["n"], shape["k"], target["wave_size"])
    if any(type(value) is not int or value <= 0 for value in values): raise TypeError
    workload = ResearchWorkload(
      row["phase"], row["quant"], row["role"], shape["m"], shape["n"], shape["k"],
      target["backend"], target["arch"], target["wave_size"])
  except (KeyError, TypeError): raise ResearchPolicyBlocked("policy row has malformed exact workload facts") from None
  if any(not isinstance(value, str) or not value for value in
         (workload.phase, workload.quant_format, workload.role, workload.backend, workload.arch)):
    raise ResearchPolicyBlocked("policy row has malformed exact workload labels")
  return workload


def _candidate_workload(entry: Mapping[str, Any]) -> ResearchWorkload:
  try:
    workload = entry["payload"]["workload"]
    shape, target = workload["shape"], workload["target"]
    return ResearchWorkload(
      "prefill", workload["dtypes"]["b"], workload["role"], shape["m"], shape["n"], shape["k"],
      target["backend"], target["arch"], target["wave_size"])
  except (KeyError, TypeError): raise ResearchPolicyBlocked("candidate entry has malformed exact workload facts") from None


def _fallback_workload(fallback: Mapping[str, Any]) -> ResearchWorkload:
  raw = fallback.get("workload")
  if not isinstance(raw, Mapping): raise ResearchPolicyBlocked("fallback has no exact workload")
  return _row_workload({**raw, "quant":raw.get("quant_format")})


class ExactSixRowResearchSelector:
  """Validated exact lookup over the immutable retained research artifact."""
  def __init__(self, policy: Mapping[str, Any], *, enabled: bool = False):
    self.enabled = bool(enabled)
    if not isinstance(policy, Mapping): raise ResearchPolicyBlocked("research policy must be a mapping")
    self.policy = dict(policy)
    unsigned = {key:value for key, value in policy.items() if key != "artifact_identity"}
    if policy.get("artifact_identity") != _policy_identity("qk_exact_six_row_policy", unsigned):
      raise ResearchPolicyBlocked("research policy artifact identity mismatch")
    if policy.get("artifact_identity") != RETAINED_POLICY_IDENTITY:
      raise ResearchPolicyBlocked("research policy is not the retained immutable v2 artifact")
    if policy.get("schema") != POLICY_SCHEMA or policy.get("status") != "research_only" or \
        policy.get("production_promotion") is not False:
      raise ResearchPolicyBlocked("policy is not the research-only v2 contract")
    candidate_set, rows = policy.get("candidate_set"), policy.get("policy_rows")
    if not isinstance(candidate_set, Mapping) or not isinstance(rows, list):
      raise ResearchPolicyBlocked("policy lacks candidate set or rows")
    try: set_identity = canonical_candidate_set_identity(candidate_set)
    except (KeyError, TypeError, ValueError) as exc:
      raise ResearchPolicyBlocked(f"candidate set validation failed: {exc}") from exc
    if set_identity != RETAINED_CANDIDATE_SET_IDENTITY:
      raise ResearchPolicyBlocked("candidate set identity differs from retained policy")
    entries, fallbacks = candidate_set.get("entries"), candidate_set.get("fallbacks")
    if not isinstance(entries, list) or len(entries) != 1 or not isinstance(fallbacks, list) or len(fallbacks) != 5:
      raise ResearchPolicyBlocked("policy must contain exactly one candidate and five fallbacks")

    candidate_entry = entries[0]
    if not isinstance(candidate_entry, Mapping) or not isinstance(candidate_entry.get("canonical_identity"), str):
      raise ResearchPolicyBlocked("candidate entry identity is missing")
    candidate_key = _candidate_workload(candidate_entry).key
    candidate_identity = candidate_entry["canonical_identity"]
    fallback_map = {}
    for fallback in fallbacks:
      if not isinstance(fallback, Mapping): raise ResearchPolicyBlocked("malformed fallback row")
      key, identity = _fallback_workload(fallback).key, fallback.get("fallback_identity")
      if key in fallback_map or not isinstance(identity, str) or not identity:
        raise ResearchPolicyBlocked("duplicate fallback workload or missing identity")
      fallback_map[key] = (identity, fallback.get("evidence_identity"))

    expected_keys = {group.workload.key for group in GROUPS}
    observed_keys = []
    selected = {}
    for row in rows:
      if not isinstance(row, Mapping): raise ResearchPolicyBlocked("malformed policy row")
      workload, kind = _row_workload(row), row.get("binding_kind")
      if workload.key in selected: raise ResearchPolicyBlocked("duplicate exact policy workload")
      observed_keys.append(workload.key)
      common = (row.get("candidate_set_identity") == set_identity and
                isinstance(row.get("inventory_identity"), str) and row["inventory_identity"] and
                isinstance(row.get("capability_identity"), str) and row["capability_identity"])
      if not common: raise ResearchPolicyBlocked("policy row identity authorities are missing or drifted")
      if kind == "candidate":
        valid = (workload.key == candidate_key and row.get("candidate_identity") == candidate_identity and
                 row.get("selected_route") == CANDIDATE_ROUTE and row.get("route_aliases") == [CANDIDATE_ROUTE])
        binding_identity = row.get("candidate_identity")
      elif kind == "fallback":
        declared = fallback_map.get(workload.key)
        valid = (declared is not None and row.get("fallback_identity") == declared[0] and
                 row.get("fallback_evidence_identity") == declared[1] and
                 row.get("selected_route") == DIRECT_PACKED_ROUTE and row.get("route_aliases") == [DIRECT_PACKED_ROUTE])
        binding_identity = row.get("fallback_identity")
      else:
        valid, binding_identity = False, None
      if not valid or not isinstance(binding_identity, str):
        raise ResearchPolicyBlocked("candidate/fallback policy binding identity mismatch")
      selected[workload.key] = (dict(row), binding_identity)
    if len(observed_keys) != 6 or set(observed_keys) != expected_keys:
      raise ResearchPolicyBlocked("policy is partial or contains an unknown workload")
    if candidate_key != GROUPS[0].workload.key:
      raise ResearchPolicyBlocked("the sole candidate is not exact Q4 ffn_gate_up")
    for group in GROUPS:
      if selected[group.workload.key][1] != group.expected_binding_identity:
        raise ResearchPolicyBlocked(f"retained binding identity drift for {group.invocation_id}")
    self._rows = selected

  def select(self, invocation_id: str, workload: ResearchWorkload, *,
             expected_binding_identity: str) -> ResearchSelection:
    if not self.enabled: raise ResearchPolicyBlocked("six-row research selector is disabled by default")
    if not isinstance(invocation_id, str) or not invocation_id:
      raise ResearchPolicyBlocked("invocation_id must be a non-empty string")
    found = self._rows.get(workload.key)
    if found is None: raise ResearchPolicyBlocked("unknown exact research workload; no fallback is implied")
    row, identity = found
    if expected_binding_identity != identity:
      raise ResearchPolicyBlocked("caller binding identity differs from retained exact policy")
    return ResearchSelection(
      invocation_id, workload, row["binding_kind"], row["selected_route"], identity,
      row["candidate_set_identity"], row["inventory_identity"], row["capability_identity"])


Callback = Callable[[ResearchSelection, int], HostProgramExecution]


def run_six_row_host_dispatch(*, enabled: bool = False, policy_path: str | Path = DEFAULT_POLICY,
                              candidate_callback: Callback | None = None,
                              direct_packed_callback: Callback | None = None) -> dict[str, Any]:
  """Execute the grouped host-only research matrix and return the live execution census."""
  base = {
    "schema":SCHEMA, "enabled":bool(enabled), "research_only":True, "production_promotion":False,
    "policy_identity":RETAINED_POLICY_IDENTITY, "candidate_set_identity":RETAINED_CANDIDATE_SET_IDENTITY,
    "expected_totals":{"candidate":80, "fallback":200, "all":280},
  }
  if not enabled:
    return {**base, "status":"DISABLED", "executed":False, "completed_calls":0,
            "actual_totals":{"candidate":0, "fallback":0, "all":0}, "execution_census":None}
  if not callable(candidate_callback) or not callable(direct_packed_callback):
    return {**base, "status":"BLOCKED", "executed":False, "completed_calls":0,
            "actual_totals":{"candidate":0, "fallback":0, "all":0},
            "exact_blocker":"enabled research dispatch requires explicit candidate and direct_packed callbacks",
            "execution_census":None}
  try: selector = ExactSixRowResearchSelector(load_retained_policy(policy_path), enabled=True)
  except Exception as exc:
    return {**base, "status":"BLOCKED", "executed":False, "completed_calls":0,
            "actual_totals":{"candidate":0, "fallback":0, "all":0},
            "exact_blocker":f"policy validation failed: {type(exc).__name__}: {exc}", "execution_census":None}

  required = tuple(group.invocation_id for group in GROUPS)
  expected_counts = {group.invocation_id:group.expected_calls for group in GROUPS}
  expected_bindings = {group.expected_binding_identity:group.expected_calls for group in GROUPS}
  completed, kinds, programs, blocker = 0, Counter(), Counter(), None
  with collect_prefill_route_execution_census(
      required, expected_candidate_counts=expected_bindings, expected_fallback_count=200,
      expected_counts=expected_counts) as census:
    for group in GROUPS:
      try:
        selection = selector.select(
          group.invocation_id, group.workload, expected_binding_identity=group.expected_binding_identity)
        callback = candidate_callback if selection.binding_kind == "candidate" else direct_packed_callback
        linear = SimpleNamespace(_prefill_route_attachment=PrefillRouteAttachment(
          group.invocation_id, selection.route_id, group.invocation_id,
          {"artifact_identity":RETAINED_POLICY_IDENTITY, "binding_identity":selection.binding_identity},
          {"target":dict(TARGET)}))
        for call_index in range(group.expected_calls):
          execution = callback(selection, call_index)
          if not isinstance(execution, HostProgramExecution):
            raise ResearchPolicyBlocked("execution callback returned no typed host program identity")
          if execution.executed_route_id != selection.route_id or execution.binding_identity != selection.binding_identity:
            raise ResearchPolicyBlocked("execution callback identity differs from selected exact binding")
          if not isinstance(execution.program_identity, str) or not execution.program_identity:
            raise ResearchPolicyBlocked("execution callback returned an empty program identity")
          fallback = selection.binding_kind == "fallback"
          notify_prefill_route_execution(linear, PrefillRouteExecution(
            group.invocation_id, execution.executed_route_id, execution.binding_identity,
            execution.program_identity, fallback,
            "exact six-row policy declares direct_packed fallback" if fallback else None))
          completed += 1; kinds[selection.binding_kind] += 1; programs[execution.program_identity] += 1
      except Exception as exc:
        blocker = f"{group.invocation_id} dispatch blocked: {type(exc).__name__}: {exc}"
        break
    census_artifact = census.artifact()
  passed = blocker is None and census_artifact.get("status") == "PASS" and \
    dict(kinds) == {"candidate":80, "fallback":200}
  return {
    **base, "status":"PASS" if passed else "BLOCKED", "executed":completed > 0,
    "completed_calls":completed,
    "actual_totals":{"candidate":kinds["candidate"], "fallback":kinds["fallback"], "all":completed},
    "program_identity_counts":dict(programs), "execution_census":census_artifact,
    **({"exact_blocker":blocker or census_artifact.get("blocker", "execution census failed")} if not passed else {}),
  }


__all__ = [
  "DEFAULT_POLICY", "GROUPS", "HostProgramExecution", "RETAINED_CANDIDATE_SET_IDENTITY",
  "RETAINED_POLICY_IDENTITY", "ResearchPolicyBlocked", "ResearchSelection", "ResearchWorkload",
  "ExactSixRowResearchSelector", "load_retained_policy", "run_six_row_host_dispatch",
]
