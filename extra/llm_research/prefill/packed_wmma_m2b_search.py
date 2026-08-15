#!/usr/bin/env python3
"""M2b readiness boundary for the packed-WMMA prefill route.

The live packed-WMMA route is a scheduler-generated matmul bound to a frozen,
shape-blind geometry table.  Its table and canary are evidence of one incumbent
implementation, not an independent candidate grammar or search result.
"""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
from typing import Any, Mapping
from tinygrad.llm.packed_wmma_prefill import PACKED_WMMA_ROUTES


SCHEMA = "tinygrad.prefill.packed-wmma-m2b-search.v1"
ROUTE_ID = "packed_wmma_prefill_generated"
TARGET = {"backend": "AMD", "architecture": "gfx1100", "wave_size": 32}
SEARCH_SYSTEM = {"name": "BoltBeam", "revision": "f6ee2763f47316112fbba40b91b859e0e7068a6d"}
EXPORT_DIR = Path(__file__).with_name("packed_wmma_m2b_search_exports")

# The route manifest's only exact packed-WMMA guards: six 14B mixed-quant rows.
# This is coverage inventory, not a candidate-space seed.
COVERED_SHAPES = tuple((row.quant, row.role, *row.shape) for row in PACKED_WMMA_ROUTES)
TOPOLOGY_SCHEMA = {
  "dequant_overlay": ("quant_format", "packed_storage", "block_decode", "overlay_layout", "metadata_semantics", "accumulator_dtype"),
  "wmma": ("instruction", "fragment_layout", "mma_k", "accumulator_layout", "epilogue"),
  "tile": ("m", "n", "k", "wave_m", "wave_n", "threads"),
  "global_local_axes": ("global_axes", "local_axes", "reduce_axes", "axis_order"),
  "lds": ("a_window", "b_window", "strides", "padding", "cooperative_loads", "barriers"),
  "pipeline": ("buffer_count", "stage_count", "dependency_waits", "tail_protocol"),
  "memory": ("weight_addressing", "activation_addressing", "output_layout", "resource_limits"),
}
BLOCKERS = {
  "grammar": ["No independent, shape-keyed packed-WMMA topology grammar defines domains and cross-field legality.",
              "The frozen incumbent geometry table is excluded: it has no reproducible candidate population provenance."],
  "primitive_lowerers": ["generic packed Q4_K/Q6_K overlay dequant lowerer", "generic AMD WMMA fragment/tile lowerer",
                         "generic LDS/cooperative-load/barrier/pipeline lowerer"],
  "ranker": ["No complete-plan ranker consumes an independent packed-WMMA topology grammar."],
  "runner": ["The current adapter and correctness canary bind the incumbent geometry; no runner accepts an arbitrary exact topology plan."],
  "fixtures": ["Per exact shape key: GGUF packed bytes, activation batch, reference output, and overlay-layout metadata."],
  "execution_inputs": ["AMD gfx1100 through Linux KFD or macOS PCI+AMD, plus target fingerprint and isolated timing protocol."],
}


def _canonical(value: Any) -> str: return json.dumps(value, sort_keys=True, separators=(",", ":"))
def _sha256(value: Any) -> str: return hashlib.sha256(_canonical(value).encode()).hexdigest()


def required_topology_schema() -> dict[str, Any]:
  return {"schema": SCHEMA, "kind": "required_topology_schema", "status": "UNPROVEN", "not_a_grammar": True,
          "fields": {name: list(values) for name, values in TOPOLOGY_SCHEMA.items()},
          "absence": ["no value domains", "no legality grammar", "no enumerator", "no generic lowerer"]}


def workloads() -> list[dict[str, Any]]:
  return [{"shape_key": f"qwen3_14b_q4k_m_gfx1100:{quant.name}:{role}:m{m}-n{n}-k{k}", "profile": "qwen3_14b_q4k_m_gfx1100",
           "quant": quant.name, "role": role, "shape": {"M": m, "N": n, "K": k}, "route_applicability": "COVERED"}
          for quant, role, m, n, k in COVERED_SHAPES]


def request() -> dict[str, Any]:
  record = {"schema": SCHEMA, "kind": "search_request", "public": True, "route_id": ROUTE_ID,
            "request_id": "m2b-packed-wmma-prefill-qwen3-14b-gfx1100-v1", "target": dict(TARGET), "workloads": workloads(),
            "required_topology_schema": required_topology_schema(), "candidate_space_status": "MISSING",
            "candidate_space_reason": "No independently justified finite domains; frozen incumbent geometry is excluded.",
            "objective": {"name": "route_bound_latency_ms", "direction": "minimize", "ranking": "median_of_3", "tie_break": "canonical_exact_plan_hash"},
            "search_system": dict(SEARCH_SYSTEM), "default": False, "catalog_entry": False}
  record["request_sha256"] = _sha256(record)
  return record


def blocked_record() -> dict[str, Any]:
  return {"schema": SCHEMA, "kind": "blocked_search_readiness", "status": "BLOCKED", "verdict": "UNPROVEN", "route_id": ROUTE_ID,
          "request": request(), "missing": {key: list(value) for key, value in BLOCKERS.items()}, "default": False, "catalog_entry": False}


def validate(record: Mapping[str, Any]) -> None:
  if record.get("schema") != SCHEMA or record.get("route_id") != ROUTE_ID: raise ValueError("unexpected M2b record")
  if record.get("default") is not False or record.get("catalog_entry") is not False: raise ValueError("M2b record cannot be catalog/default")
  if record.get("kind") == "blocked_search_readiness":
    if (record.get("status"), record.get("verdict")) != ("BLOCKED", "UNPROVEN"): raise ValueError("M2b record must be BLOCKED/UNPROVEN")
    if set(record.get("missing", ())) != set(BLOCKERS): raise ValueError("M2b blockers incomplete")
    validate(record.get("request", {})); return
  if record.get("kind") != "search_request" or record.get("target") != TARGET or record.get("workloads") != workloads(): raise ValueError("M2b exact coverage drift")
  if record.get("candidate_space_status") != "MISSING" or "finite_space" in record: raise ValueError("M2b must not claim a finite space")
  if record.get("search_system") != SEARCH_SYSTEM: raise ValueError("M2b search-system revision drift")
  copy = dict(record); digest = copy.pop("request_sha256", None)
  if digest != _sha256(copy): raise ValueError("M2b request SHA-256 drift")


def export(out_dir: str | Path = EXPORT_DIR) -> Path:
  record = blocked_record(); validate(record); path = Path(out_dir) / f"{ROUTE_ID}.blocked.json"; path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n"); return path


def check_exports(out_dir: str | Path = EXPORT_DIR) -> None:
  path = Path(out_dir) / f"{ROUTE_ID}.blocked.json"
  expected = json.dumps(blocked_record(), indent=2, sort_keys=True) + "\n"
  if not path.is_file() or path.read_text() != expected: raise ValueError("M2b checked-in export drift")
  validate(json.loads(path.read_text()))


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="emit/check M2b packed-WMMA BLOCKED readiness")
  parser.add_argument("--out-dir", type=Path, default=EXPORT_DIR); parser.add_argument("--check", action="store_true")
  args = parser.parse_args(argv)
  if args.check: check_exports(args.out_dir)
  else: export(args.out_dir)
  return 0


if __name__ == "__main__": raise SystemExit(main())
