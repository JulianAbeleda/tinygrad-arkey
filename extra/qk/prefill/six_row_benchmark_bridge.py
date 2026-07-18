"""Explicit, default-off whole-model bridge for the six-row research policy.

This module only joins existing authorities: the retained policy, frozen
candidate bundle, direct-packed fallback declarations, runtime model
attachments, and the benchmark-only route implementation. It does not compile,
emit, or launch a kernel by itself.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from extra.qk.mmq_exact_role_spec import DEFAULT_INVENTORY, exact_role_spec
from extra.qk.prefill.frozen_exact_role_runtime import load_frozen_exact_role_binding
from extra.qk.prefill.six_row_research_selector import (
  GROUPS, RETAINED_POLICY_IDENTITY, TARGET, ExactSixRowResearchSelector, ResearchPolicyBlocked,
)
from extra.qk.prefill_research_routes import ExactResearchRouteAuthority, PrefillResearchRouteConfig
from tinygrad.llm.prefill_route_observer import PrefillRouteAttachment


def parse_identity_assignments(values: Sequence[str], *, label: str) -> dict[str, str]:
  """Parse repeatable ``IDENTITY=VALUE`` CLI declarations without guessing."""
  parsed: dict[str, str] = {}
  for value in values:
    identity, separator, target = value.partition("=")
    if not separator or not identity or not target:
      raise ResearchPolicyBlocked(f"{label} must use non-empty IDENTITY=VALUE declarations")
    if identity in parsed:
      raise ResearchPolicyBlocked(f"duplicate {label} identity {identity!r}")
    parsed[identity] = target
  return parsed


def _load_mapping(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
  if isinstance(value, Mapping): return dict(value)
  loaded = json.loads(Path(value).read_text())
  if not isinstance(loaded, dict): raise ResearchPolicyBlocked("research authority JSON must be an object")
  return loaded


def build_exact_research_authority(*, policy_path: str | Path,
                                   frozen_bundles: Mapping[str, str | Path],
                                   fallback_program_identities: Mapping[str, str],
                                   inventory: str | Path | Mapping[str, Any] = DEFAULT_INVENTORY
                                   ) -> ExactResearchRouteAuthority:
  """Validate every authority input before a model or device is opened."""
  policy = _load_mapping(policy_path)
  ExactSixRowResearchSelector(policy, enabled=True)
  inventory_value = _load_mapping(inventory)
  if inventory_value.get("inventory_identity") != policy.get("inventory_identity"):
    raise ResearchPolicyBlocked("research policy and candidate inventory identities differ")

  candidate_ids = {group.expected_binding_identity for group in GROUPS
                   if not group.expected_binding_identity.startswith("fallback:")}
  fallback_ids = {group.expected_binding_identity for group in GROUPS
                  if group.expected_binding_identity.startswith("fallback:")}
  if set(frozen_bundles) != candidate_ids:
    raise ResearchPolicyBlocked(
      f"frozen bundle declarations differ: supplied={sorted(frozen_bundles)!r}, expected={sorted(candidate_ids)!r}")
  if set(fallback_program_identities) != fallback_ids:
    raise ResearchPolicyBlocked(
      "declared fallback program identities differ: "
      f"supplied={sorted(fallback_program_identities)!r}, expected={sorted(fallback_ids)!r}")
  normalized_bundles: dict[str, Path] = {}
  for identity, raw_path in frozen_bundles.items():
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
      raise ResearchPolicyBlocked(f"frozen bundle for {identity!r} does not exist: {path}")
    normalized_bundles[identity] = path
  frozen_bindings = {}
  for group in GROUPS:
    if group.expected_binding_identity not in candidate_ids: continue
    role_spec = exact_role_spec(
      group.workload.role, shape=(group.workload.m, group.workload.n, group.workload.k), inventory=inventory_value)
    binding = load_frozen_exact_role_binding(
      role_spec, normalized_bundles[group.expected_binding_identity], inventory=inventory_value)
    if binding.candidate_identity != group.expected_binding_identity:
      raise ResearchPolicyBlocked("frozen bundle candidate identity differs from retained policy")
    frozen_bindings[group.expected_binding_identity] = binding
  if any(not isinstance(value, str) or not value for value in fallback_program_identities.values()):
    raise ResearchPolicyBlocked("declared fallback program identities must be non-empty strings")
  return ExactResearchRouteAuthority(policy, dict(TARGET), normalized_bundles,
                                     dict(fallback_program_identities), inventory_value, frozen_bindings)


def _inventory_rows(authority: ExactResearchRouteAuthority) -> tuple[dict[str, Any], ...]:
  inventory = _load_mapping(authority.inventory)
  raw_rows = inventory.get("inventory", {}).get("rows")
  if not isinstance(raw_rows, list): raise ResearchPolicyBlocked("candidate inventory lacks grouped rows")
  rows = tuple(dict(row) for row in raw_rows if isinstance(row, Mapping))
  if len(rows) != len(GROUPS): raise ResearchPolicyBlocked("candidate inventory does not contain exactly six grouped rows")
  return rows


def _group_for_row(row: Mapping[str, Any]):
  shape = row.get("shape")
  if not isinstance(shape, Mapping): raise ResearchPolicyBlocked("candidate inventory row lacks an exact shape")
  key = (row.get("quant_format"), row.get("role"), shape.get("m"), shape.get("n"), shape.get("k"))
  matches = [group for group in GROUPS if
             (group.workload.quant_format, group.workload.role, group.workload.m,
              group.workload.n, group.workload.k) == key]
  if len(matches) != 1: raise ResearchPolicyBlocked(f"candidate inventory contains an unknown or duplicate workload {key!r}")
  return matches[0]


def _resolve_model_linear(model: Any, tensor_identity: str) -> Any:
  if not isinstance(tensor_identity, str) or not tensor_identity.endswith(".weight"):
    raise ResearchPolicyBlocked(f"research tensor identity is not an exact weight path: {tensor_identity!r}")
  obj = model
  try:
    for component in tensor_identity[:-7].split("."):
      obj = obj[int(component)] if component.isdigit() and isinstance(obj, (list, tuple)) else getattr(obj, component)
  except (AttributeError, IndexError, TypeError) as exc:
    raise ResearchPolicyBlocked(f"research tensor has no whole-model runtime owner: {tensor_identity!r}") from exc
  return obj


def research_execution_census_expectations(model: Any, authority: ExactResearchRouteAuthority) -> dict[str, Any]:
  """Derive the one-call census contract from exact live attachments and retained groups."""
  selector = ExactSixRowResearchSelector(authority.policy, enabled=True)
  expected_counts, expected_candidates, fallback_count = {}, {}, 0
  for row in _inventory_rows(authority):
    group = _group_for_row(row)
    tensor_ids = row.get("tensor_identities")
    if not isinstance(tensor_ids, list) or len(tensor_ids) != group.expected_calls:
      raise ResearchPolicyBlocked(f"candidate inventory call coverage differs for {group.invocation_id}")
    for tensor_identity in tensor_ids:
      linear = _resolve_model_linear(model, tensor_identity)
      attachment = getattr(linear, "_prefill_route_attachment", None)
      if not isinstance(attachment, PrefillRouteAttachment) or attachment.tensor_identity != tensor_identity:
        raise ResearchPolicyBlocked(f"runtime attachment differs for research tensor {tensor_identity!r}")
      if attachment.invocation_id in expected_counts:
        raise ResearchPolicyBlocked("research census invocation coverage is not one-to-one")
      selection = selector.select(attachment.invocation_id, group.workload,
                                  expected_binding_identity=group.expected_binding_identity)
      expected_counts[attachment.invocation_id] = 1
      expected_candidates[selection.binding_identity] = expected_candidates.get(selection.binding_identity, 0) + 1
      fallback_count += int(selection.binding_kind == "fallback")
  return {
    "required_invocations": tuple(expected_counts),
    "expected_counts": expected_counts,
    "expected_candidate_counts": expected_candidates,
    "expected_fallback_count": fallback_count,
  }


@contextmanager
def exact_research_model_scope(model: Any, authority: ExactResearchRouteAuthority
                               ) -> Iterator[PrefillResearchRouteConfig]:
  """Temporarily replace production metadata attachments with exact research bindings."""
  if not isinstance(authority, ExactResearchRouteAuthority):
    raise TypeError("whole-model research bridge requires ExactResearchRouteAuthority")
  selector = ExactSixRowResearchSelector(authority.policy, enabled=True)
  saved: list[tuple[Any, PrefillRouteAttachment]] = []
  attached_ids, attached_objects = set(), set()
  try:
    for row in _inventory_rows(authority):
      group = _group_for_row(row)
      tensor_ids = row.get("tensor_identities")
      if not isinstance(tensor_ids, list) or len(tensor_ids) != group.expected_calls or \
         row.get("call_count") != group.expected_calls:
        raise ResearchPolicyBlocked(f"candidate inventory call coverage differs for {group.invocation_id}")
      for tensor_identity in tensor_ids:
        linear = _resolve_model_linear(model, tensor_identity)
        old = getattr(linear, "_prefill_route_attachment", None)
        if not isinstance(old, PrefillRouteAttachment) or old.tensor_identity != tensor_identity:
          raise ResearchPolicyBlocked(f"runtime attachment differs for research tensor {tensor_identity!r}")
        if id(linear) in attached_objects or old.invocation_id in attached_ids:
          raise ResearchPolicyBlocked("research whole-model attachment coverage is not one-to-one")
        selection = selector.select(old.invocation_id, group.workload,
                                    expected_binding_identity=group.expected_binding_identity)
        saved.append((linear, old))
        linear._prefill_route_attachment = PrefillRouteAttachment(
          old.invocation_id, selection.route_id, tensor_identity,
          {"artifact_identity": RETAINED_POLICY_IDENTITY, "binding_identity": selection.binding_identity,
           "research_only": True, "production_promotion": False},
          old.scanned_target_facts)
        attached_ids.add(old.invocation_id)
        attached_objects.add(id(linear))
    expected = sum(group.expected_calls for group in GROUPS)
    if len(saved) != expected:
      raise ResearchPolicyBlocked(f"research whole-model coverage differs: attached={len(saved)}, expected={expected}")
    yield PrefillResearchRouteConfig(exact_policy_enabled=True, exact_authority=authority)
  finally:
    for linear, attachment in reversed(saved): linear._prefill_route_attachment = attachment


def research_bridge_summary(authority: ExactResearchRouteAuthority) -> dict[str, Any]:
  return {
    "enabled": True, "default_off": True, "research_only": True, "production_promotion": False,
    "integration_only": True, "performance_qualified": False,
    "policy_identity": authority.policy.get("artifact_identity"),
    "inventory_identity": authority.policy.get("inventory_identity"),
    "frozen_bundles": {identity: str(path) for identity, path in authority.frozen_bundles.items()},
    "declared_fallback_program_identities": dict(authority.fallback_program_identities),
    "intended_calls": sum(group.expected_calls for group in GROUPS),
    "scheduler_owned_candidate_graph": True,
    "tinyjit_replay_authority": False,
    "performance_note": "scheduler-owned route integration only; TinyJit replay and a whole-model performance win remain unproven",
  }


__all__ = ["build_exact_research_authority", "exact_research_model_scope", "parse_identity_assignments",
           "research_bridge_summary", "research_execution_census_expectations"]
