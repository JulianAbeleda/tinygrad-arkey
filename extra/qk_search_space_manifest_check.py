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


def main(argv: list[str]) -> int:
  paths = [ROOT / a if not pathlib.Path(a).is_absolute() else pathlib.Path(a) for a in argv[1:]] or DEFAULT_MANIFESTS
  errors: list[str] = []
  for path in paths:
    errors.extend(check_manifest(path))
  if errors:
    print(f"SEARCH SPACE MANIFEST CHECK: FAIL ({len(errors)} issue(s))")
    print("\n".join(errors))
    return 1
  print(f"SEARCH SPACE MANIFEST CHECK: PASS ({len(paths)} manifest(s))")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
