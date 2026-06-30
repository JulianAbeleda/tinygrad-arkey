#!/usr/bin/env python3
"""Validate search-space manifests for the pure-machine-search primitive boundary.

This is intentionally lightweight: it checks the manifest contract, not GPU behavior.
Run:
  PYTHONPATH=. python3 extra/qk_search_space_manifest_check.py
  PYTHONPATH=. python3 extra/qk_search_space_manifest_check.py bench/qk-search-spaces/decode_attention_online_softmax_pv_tile_v1.json
"""
from __future__ import annotations

import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MANIFESTS = [ROOT / "bench/qk-search-spaces/decode_attention_online_softmax_pv_tile_v1.json"]

REQUIRED_TOP = [
  "search_space_id",
  "primitive_family",
  "supported_profiles",
  "required_primitive_boundary",
  "exposed_instruction_primitives",
  "exposed_memory_primitives",
  "exposed_scheduling_primitives",
  "exposed_dataflow_primitives",
  "exposed_runtime_primitives",
  "excluded_primitives",
  "proof_of_coverage",
  "classification",
]

REQUIRED_BOUNDARY = {
  "split_kv_decode_attention_tile",
  "whole_cache_kv_identity",
  "T_equals_1_parallelism_from_Hkv_times_S_workgroups",
  "query_head_and_GQA_parallelism",
  "v_dot2_or_equivalent_packed_fp16_dot",
  "cross_lane_reduction",
  "register_resident_online_softmax_state_m_l_accD",
  "PV_accumulation_inside_tile_lifecycle",
  "TILE_PLUS_COMBINE_lifecycle_accounting",
}

FORBIDDEN_PROMOTION_CLASSIFICATIONS = {
  "SEARCH_FOUND_PROMOTABLE",
  "DECODE_ATTENTION_ONLINE_PV_TILE_SEARCH_PROMOTABLE",
}


def _items_text(xs) -> str:
  return json.dumps(xs, sort_keys=True).lower()


def check_manifest(path: pathlib.Path) -> list[str]:
  rel = path.relative_to(ROOT) if path.is_absolute() and path.is_relative_to(ROOT) else path
  errors: list[str] = []
  try:
    data = json.loads(path.read_text())
  except Exception as e:  # noqa: BLE001
    return [f"{rel}: invalid JSON: {e}"]

  for key in REQUIRED_TOP:
    if key not in data:
      errors.append(f"{rel}: missing top-level key {key!r}")

  boundary = set(data.get("required_primitive_boundary", []))
  missing = sorted(REQUIRED_BOUNDARY - boundary)
  if missing:
    errors.append(f"{rel}: required_primitive_boundary missing {missing}")

  sid = data.get("search_space_id")
  if sid != "decode_attention_online_softmax_pv_tile_v1":
    errors.append(f"{rel}: unexpected search_space_id {sid!r}")

  classification = data.get("classification")
  if classification in FORBIDDEN_PROMOTION_CLASSIFICATIONS:
    errors.append(f"{rel}: classification {classification!r} is not allowed before structural+W==D gates pass")
  if classification not in {"SEARCH_SPACE_INCOMPLETE", "SEARCH_BLOCKED_BY_CODEGEN", "SEARCH_BLOCKED_BY_RUNTIME", "SEARCH_EXHAUSTED_SPACE"}:
    errors.append(f"{rel}: classification {classification!r} is not an allowed pre-implementation classification")

  negative = _items_text(data.get("known_negative_controls", []))
  if "a3_10" not in negative or "no_transfer" not in negative:
    errors.append(f"{rel}: known_negative_controls must include A3.10 no-transfer")

  excluded = _items_text(data.get("excluded_primitives", []))
  for forbidden in ("owned_flash_tile_gqa_whole", "owned_flash_combine"):
    if forbidden not in excluded:
      errors.append(f"{rel}: excluded_primitives must mark {forbidden} as oracle/fallback, not pure search")

  structural = _items_text(data.get("structural_gate", []))
  for required in ("e_49152", "whole-cache", "owned_flash_tile_gqa_whole", "owned_flash_combine", "tile+combine"):
    if required not in structural:
      errors.append(f"{rel}: structural_gate must mention {required}")

  proof = data.get("proof_of_coverage", {})
  if not isinstance(proof, dict):
    errors.append(f"{rel}: proof_of_coverage must be an object")
  else:
    incomplete = _items_text(proof.get("coverage_incomplete_for", []))
    for required in ("lane", "cross-lane", "online-softmax", "w==d"):
      if required not in incomplete:
        errors.append(f"{rel}: proof_of_coverage.coverage_incomplete_for must mention {required}")

  return errors


SEARCH_PROFILES = ROOT / "bench/qk-search-spaces/search_profiles.json"
_PROFILE_REQUIRED_TOP = ["target", "route_families", "profiles", "do_not_search", "status_vocab"]
_PROFILE_ROLE_REQUIRED = ["quant", "shape", "allowed_route_families", "status"]


def check_search_profiles(path: pathlib.Path) -> list[str]:
  """PMS-R3 search_profiles.json contract: route_families closed-set, every role declares allowed_route_families
  (subset of the closed set), every do_not_search row names a (role, route_family), and any role that cites a route_id
  resolves against the route manifest."""
  rel = path.relative_to(ROOT) if path.is_absolute() and path.is_relative_to(ROOT) else path
  errors: list[str] = []
  try:
    data = json.loads(path.read_text())
  except Exception as e:  # noqa: BLE001
    return [f"{rel}: invalid JSON: {e}"]
  for key in _PROFILE_REQUIRED_TOP:
    if key not in data: errors.append(f"{rel}: missing top-level key {key!r}")
  families = set(data.get("route_families", []))
  if not families: errors.append(f"{rel}: route_families must be a non-empty closed set")
  try:
    from extra.qk_route_manifest import ROUTES  # route_id cross-check (manifest is the source of truth)
    known_routes = set(ROUTES)
  except Exception:
    known_routes = None
  for pid, profile in data.get("profiles", {}).items():
    if "workload" not in profile: errors.append(f"{rel}: profile {pid!r} missing 'workload'")
    for role, rmeta in profile.get("roles", {}).items():
      for k in _PROFILE_ROLE_REQUIRED:
        if k not in rmeta: errors.append(f"{rel}: {pid}.{role} missing {k!r}")
      bad = set(rmeta.get("allowed_route_families", [])) - families
      if bad: errors.append(f"{rel}: {pid}.{role} allowed_route_families {sorted(bad)} not in route_families")
      if known_routes is not None and rmeta.get("route_id") and rmeta["route_id"] not in known_routes:
        errors.append(f"{rel}: {pid}.{role} route_id {rmeta['route_id']!r} not in route manifest")
  for i, d in enumerate(data.get("do_not_search", [])):
    if not d.get("role") or not d.get("route_family"):
      errors.append(f"{rel}: do_not_search[{i}] must name both 'role' and 'route_family'")
    if d.get("route_family") and d["route_family"] not in families:
      errors.append(f"{rel}: do_not_search[{i}] route_family {d['route_family']!r} not in route_families")
  return errors


def check_refuted_axis_drift() -> list[str]:
  """Assert the 3 refuted-axis sources agree: qk_route_manifest.REFUTED is the SINGLE SOURCE, and both
  search_profiles.json do_not_search and the quant known_refuted_route_families must be a subset of it that agrees on
  (key, disposition-class). key = route_id when present else axis; disposition compared by class token (refuted /
  deprioritized / exhausted / ...). Catches the drift where a route was refuted/classified differently in two places."""
  errors: list[str] = []
  try:
    from extra.qk_route_manifest import refuted_index, disposition_class
    from extra.qk_quant_semantics import QUANT_LIBRARY
  except Exception as e:  # pragma: no cover
    return [f"refuted-axis drift: cannot import sources: {e}"]
  canon = refuted_index()  # key -> disposition_class (the single source)
  if SEARCH_PROFILES.exists():
    for d in json.load(open(SEARCH_PROFILES)).get("do_not_search", []):
      key = d.get("route_id") or d.get("axis")
      dc = disposition_class(d.get("disposition", ""))
      if key not in canon:
        errors.append(f"do_not_search axis {key!r} not backed by manifest REFUTED (single source)")
      elif dc != canon[key]:
        errors.append(f"do_not_search {key!r} disposition class {dc!r} != REFUTED {canon[key]!r}")
  for fmt, q in QUANT_LIBRARY.items():
    for fam in getattr(q, "known_refuted_route_families", ()) or ():
      key = fam.get("route_id") or fam.get("route_family")
      dc = disposition_class(fam.get("disposition", ""))
      if key not in canon:
        errors.append(f"quant {fmt} known_refuted {key!r} not backed by manifest REFUTED (single source)")
      elif dc != canon[key]:
        errors.append(f"quant {fmt} {key!r} disposition class {dc!r} != REFUTED {canon[key]!r}")
  return errors


def main(argv: list[str]) -> int:
  paths = [ROOT / a if not pathlib.Path(a).is_absolute() else pathlib.Path(a) for a in argv[1:]] or DEFAULT_MANIFESTS
  errors: list[str] = []
  # always validate the PMS-R3 search_profiles.json contract + refuted-axis single-source agreement (default-run)
  if not argv[1:] and SEARCH_PROFILES.exists():
    errors.extend(check_search_profiles(SEARCH_PROFILES))
    errors.extend(check_refuted_axis_drift())
  for path in paths:
    if path.name == "search_profiles.json":
      errors.extend(check_search_profiles(path)); continue
    errors.extend(check_manifest(path))
  if errors:
    print(f"SEARCH SPACE MANIFEST CHECK: FAIL ({len(errors)} issue(s))")
    print("\n".join(errors))
    return 1
  print(f"SEARCH SPACE MANIFEST CHECK: PASS ({len(paths)} manifest(s))")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
