#!/usr/bin/env python3
"""Public M4 readiness records for the Q4_K/Q6_K direct-packed prefill routes.

This module intentionally stops before candidate generation.  The incumbent
routes expose a concrete UOp lowering, but not a generic topology grammar,
generic primitive lowerers, or an executable/ranking search loop.  In
particular, this file never imports their option builders: treating those
selected options as a search population would be circular.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from extra.llm_research.model_profiles import MODEL_PROFILES, prefill_role_shapes


SCHEMA = "tinygrad.prefill.direct-packed-m4-search.v1"
TARGET = {"backend": "AMD", "architecture": "gfx1100", "wave_size": 32}
SEARCH_SYSTEM = {"name": "BoltBeam", "revision": "f6ee2763f47316112fbba40b91b859e0e7068a6d"}
EXPORT_DIR = Path(__file__).with_name("m4_search_exports")
DIRECT_PACKED_M4_TOPOLOGIES = {
  "prefill_q4k_direct_tile4x4_default": {"quant": "Q4_K", "packed_storage": "uint32"},
  "prefill_q6k_direct_generated": {"quant": "Q6_K", "packed_storage": "uint16"},
}
TOPOLOGY_SCHEMA = {
  "dequant": ("quant_format", "packed_storage", "block_elements", "metadata_decode", "activation_load", "accumulator_dtype"),
  "tile": ("row_tile", "token_tile", "k_block_tile", "wave_layout", "threads"),
  "parts": ("count", "partition_axis", "tail_policy", "combine_protocol"),
  "output_layout": ("kind", "logical_axes", "physical_axes", "writeback_dtype"),
  "axes": ("global_axes", "local_axes", "reduce_axes", "axis_order"),
  "opts": ("transform_sequence", "legality_constraints", "application_order"),
  "memory": ("weight_addressing", "activation_addressing", "local_memory", "barriers", "resource_limits"),
}
_BLOCKERS = {
  "grammar": [
    "No complete cross-field topology grammar assigns legal value domains for tile, parts, output layout, axes, opts, and memory.",
    "The incumbent direct-route descriptors and their selected options are implementations, not an independently justified candidate domain.",
  ],
  "primitive_lowerers": [
    "generic Q4_K packed-block dequant-and-dot lowerer",
    "generic Q6_K packed-block dequant-and-dot lowerer",
    "generic direct/partials output-layout and cross-part reduction lowerer",
    "generic local-memory, barrier, and resource-legality lowerer",
  ],
  "ranker": ["No deterministic complete-plan ranker consumes this topology schema; structural ordering would not be a performance result."],
  "runner": ["No route-bound generic-plan runner compiles, dispatches, checks parity, and times a supplied exact plan."],
  "fixtures": [
    "Per shape-key fixture: packed GGUF bytes, fp16 activation tile, output reference, and tensor/layout metadata.",
    "A correctness oracle covering direct output, partials, and any required reduction semantics.",
  ],
  "execution_inputs": [
    "AMD gfx1100 execution through Linux KFD or macOS PCI+AMD.",
    "A target/runtime fingerprint, warmup/timing protocol, and isolated route-bound dispatch inputs for every requested shape key.",
  ],
}


def required_topology_schema() -> dict[str, Any]:
  """Return fields every future exact plan must carry; this is not a grammar."""
  return {"schema": SCHEMA, "kind": "required_topology_schema",
          "fields": {name: list(fields) for name, fields in TOPOLOGY_SCHEMA.items()},
          "status": "UNPROVEN", "not_a_grammar": True,
          "absence": ["no value domains", "no cross-field legality rules", "no enumerator", "no lowerers"]}


def _canonical(value: Any) -> str: return json.dumps(value, sort_keys=True, separators=(",", ":"))
def _sha256(value: Any) -> str: return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _weight_presence(profile_id: str, route_id: str, role: str) -> dict[str, str]:
  """Inventory applicability, deliberately separate from structural route support.

  Only the checked-in 14B GGUF inventory establishes mixed-Q4/Q6 tensor facts.
  The profile's Q4_K_M label alone cannot establish a corresponding 8B fact.
  """
  if profile_id == "qwen3_14b_q4k_m_gfx1100":
    present = {"Q4_K": {"attn_kv", "attn_qo", "ffn_down", "ffn_gate_up"},
               "Q6_K": {"attn_kv", "ffn_down"}}
    return {"status": "PRESENT" if role in present[DIRECT_PACKED_M4_TOPOLOGIES[route_id]["quant"]] else "ABSENT",
            "evidence": "docs/14b-role-facts-inventory-20260710.json"}
  return {"status": "UNVERIFIED", "evidence": "no checked-in 8B GGUF role/quant inventory"}


def workloads(route_id: str) -> list[dict[str, Any]]:
  if route_id not in DIRECT_PACKED_M4_TOPOLOGIES: raise ValueError(f"unknown direct-packed route {route_id!r}")
  quant = DIRECT_PACKED_M4_TOPOLOGIES[route_id]["quant"]
  return [{"shape_key": f"{profile.id}:{role.role}:m{role.M}-n{role.N}-k{role.K}",
           "profile": profile.id, "model_size": profile.size_label, "role": role.role,
           "quant": quant, "shape": {"M": role.M, "N": role.N, "K": role.K},
           "structural_route_support": "SUPPORTED",
           "weight_applicability": _weight_presence(profile.id, route_id, role.role)}
          for profile in MODEL_PROFILES for role in prefill_role_shapes(profile)]


def request(route_id: str) -> dict[str, Any]:
  if route_id not in DIRECT_PACKED_M4_TOPOLOGIES: raise ValueError(f"unknown direct-packed route {route_id!r}")
  record = {"schema": SCHEMA, "kind": "search_request", "public": True,
          "request_id": f"m4-{route_id}-qwen3-8b-14b-gfx1100-v1", "route_id": route_id,
          "target": dict(TARGET), "workloads": workloads(route_id),
          "required_topology_schema": required_topology_schema(),
          "candidate_space_status": "MISSING",
          "candidate_space_reason": "No independently justified finite domains exist; selected incumbent route constants are excluded.",
          "objective": {"name": "route_bound_latency_ms", "direction": "minimize", "ranking": "median_of_3",
                        "tie_break": "canonical_exact_plan_hash"},
          "required_export": ["complete_ranked_population", "exact_complete_plans", "per-shape correctness", "per-shape timing", "target/runtime fingerprint"],
          "search_system": dict(SEARCH_SYSTEM), "default": False, "catalog_entry": False}
  record["request_sha256"] = _sha256(record)
  return record


def blocked_record(route_id: str) -> dict[str, Any]:
  return {"schema": SCHEMA, "kind": "blocked_search_readiness", "status": "BLOCKED", "verdict": "UNPROVEN",
          "route_id": route_id, "request": request(route_id), "missing": {key: list(value) for key, value in _BLOCKERS.items()},
          "default": False, "catalog_entry": False,
          "next_action": "Define an independent grammar and lowerers, then supply fixtures and an AMD:gfx1100 runner before ranking or promotion."}


def _expected_workloads(route_id: str) -> list[dict[str, Any]]: return workloads(route_id)


def validate(record: Mapping[str, Any]) -> None:
  if record.get("schema") != SCHEMA or record.get("kind") not in ("search_request", "blocked_search_readiness"):
    raise ValueError("unknown M4 direct-packed search record")
  route_id = record.get("route_id")
  if route_id not in DIRECT_PACKED_M4_TOPOLOGIES: raise ValueError("record has unknown route")
  if record.get("default") is not False or record.get("catalog_entry") is not False:
    raise ValueError("M4 records must not be catalog/default entries")
  if record["kind"] == "blocked_search_readiness":
    if record.get("status") != "BLOCKED" or record.get("verdict") != "UNPROVEN":
      raise ValueError("blocked M4 record must be BLOCKED/UNPROVEN")
    missing = record.get("missing")
    if not isinstance(missing, Mapping) or set(missing) != set(_BLOCKERS) or any(not missing[k] for k in _BLOCKERS):
      raise ValueError("blocked M4 record must name every exact blocker class")
    validate(record.get("request", {})); return
  if record.get("target") != TARGET or record.get("workloads") != _expected_workloads(route_id):
    raise ValueError("request must bind all exact supported 8B/14B shape keys")
  if record.get("candidate_space_status") != "MISSING" or "finite_space" in record:
    raise ValueError("M4 request must not invent a finite candidate space")
  if record.get("search_system") != SEARCH_SYSTEM:
    raise ValueError("request must bind the reviewed BoltBeam revision")
  without_digest = dict(record); digest = without_digest.pop("request_sha256", None)
  if digest != _sha256(without_digest): raise ValueError("request SHA-256 drift")
  topology = record.get("required_topology_schema", {})
  if topology.get("fields") != {name: list(fields) for name, fields in TOPOLOGY_SCHEMA.items()} or topology.get("not_a_grammar") is not True:
    raise ValueError("request must retain the complete schema-only topology contract")


def export(out_dir: str | Path) -> tuple[Path, ...]:
  """Deterministically write one separate BLOCKED/UNPROVEN record per route."""
  out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
  paths = []
  for route_id in DIRECT_PACKED_M4_TOPOLOGIES:
    record = blocked_record(route_id); validate(record)
    path = out / f"{route_id}.blocked.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    paths.append(path)
  return tuple(paths)


def check_exports(out_dir: str | Path = EXPORT_DIR) -> None:
  """Fail when checked-in records no longer equal the deterministic export."""
  out = Path(out_dir)
  expected = {f"{route_id}.blocked.json": json.dumps(blocked_record(route_id), indent=2, sort_keys=True) + "\n" for route_id in DIRECT_PACKED_M4_TOPOLOGIES}
  actual = {path.name: path.read_text() for path in out.glob("*.blocked.json")} if out.is_dir() else {}
  if actual != expected: raise ValueError("M4 checked-in search exports drift; rerun direct_packed_m4_search.py --out-dir extra/llm_research/prefill/m4_search_exports")
  for text in actual.values(): validate(json.loads(text))
  catalog = Path(__file__).resolve().parents[3] / "tinygrad/llm/generated/catalog.json"
  artifacts = json.loads(catalog.read_text()).get("artifacts", [])
  m4_request_ids = {request(route_id)["request_id"] for route_id in DIRECT_PACKED_M4_TOPOLOGIES}
  if any(artifact.get("request_id") in m4_request_ids for artifact in artifacts if isinstance(artifact, Mapping)):
    raise ValueError("M4 search readiness must not have a generated catalog entry")


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="emit M4 direct-packed prefill BLOCKED search-readiness records")
  parser.add_argument("--out-dir", type=Path, default=EXPORT_DIR)
  parser.add_argument("--check", action="store_true", help="verify checked-in exports without writing")
  args = parser.parse_args(argv)
  if args.check: check_exports(args.out_dir)
  else: export(args.out_dir)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
