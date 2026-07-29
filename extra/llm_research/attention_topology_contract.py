"""Fail-closed contracts for complete attention-topology search experiments.

This is deliberately declarative: it does not inspect or infer knobs from a
live builder.  A search result is accepted only when a separately supplied
executor reports exact plans and measurements.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA = "tinygrad.attention_topology_search.v1"
TOPOLOGY_FIELDS = ("tile", "split", "staging", "gqa_grouping", "online_softmax", "pv", "combine", "causal")

_REQUIRED_TOPOLOGY_SCHEMA: dict[str, tuple[str, ...]] = {
  "tile": ("q_tile", "kv_tile", "head_dim_tile", "wave_count"),
  "split": ("split_count", "split_axis", "split_schedule"),
  "staging": ("k", "v", "q", "none"),
  "gqa_grouping": ("query_heads_per_kv_head", "workgroup_grouping", "tail_policy"),
  "online_softmax": ("score_scale", "max_state", "denominator_state", "state_merge"),
  "pv": ("value_tile", "accumulator_layout", "normalization"),
  "combine": ("partial_state_layout", "merge_topology", "cross_workgroup_protocol"),
  "causal": ("mask_mode", "query_offset", "tail_validity"),
}

def required_topology_schema(route: str) -> dict[str, Any]:
  if route not in ("decode", "prefill"): raise ValueError("route must be decode or prefill")
  return {"schema": SCHEMA, "kind": "required_topology_schema", "route": route,
          "topology_fields": {field: list(values) for field, values in _REQUIRED_TOPOLOGY_SCHEMA.items()},
          "status": "schema_only", "not_a_grammar": True,
          "absence": ["no value domains", "no cross-field constraints", "no enumerator", "no lowerer"],
          "candidate_authorship": "executor-generated exact plans only; no builder-constant seeding"}

def request(route: str, target_id: str, shape: dict[str, int]) -> dict[str, Any]:
  if route not in ("decode", "prefill"): raise ValueError("route must be decode or prefill")
  expected = {"head_dim", "q_heads", "kv_heads"}
  if set(shape) != expected or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in shape.values()):
    raise ValueError("shape must contain positive integral head_dim, q_heads, and kv_heads")
  if shape["q_heads"] % shape["kv_heads"]: raise ValueError("q_heads must divide evenly by kv_heads for GQA")
  workload = ([{"context": 512}, {"context": 4096}] if route == "decode" else
              [{"q_len": n, "context": n} for n in (512, 1024, 2048, 4096)])
  return {"schema": SCHEMA, "kind": "search_request", "request_id": f"attention-topology-{route}-{target_id.lower()}-amd-gfx1100-wave32-v1",
          "route": route, "variant_id": target_id,
          "target": {"backend": "AMD", "architecture": "gfx1100", "wave_size": 32},
          "shape": dict(sorted(shape.items())), "required_topology_fields": list(TOPOLOGY_FIELDS),
          "required_topology_schema": required_topology_schema(route),
          "workloads": workload,
          "objective": {"primary": "minimize_route_bound_latency", "ranking": "median_of_3", "tie_break": "lower_max_latency"},
          "budget": {"max_generated_plans": 256, "max_measured_candidates": 64, "timing_repetitions": 3},
          "correctness_constraints": {"reference": "route_reference_attention", "causal": True,
                                      "require_output_parity": True, "require_online_softmax_semantics": True,
                                      "require_gqa_mapping": True},
          "require_exact_plan": True,
          "provenance_rule": "do not manufacture candidates from existing hand builder constants"}

def blocked_record(route: str, target_id: str, shape: dict[str, int]) -> dict[str, Any]:
  req = request(route, target_id, shape)
  grammar = ["only a required-topology schema exists; no value domains or cross-field constraints define an enumerable grammar"]
  if route == "decode": grammar.append("flash_kernels.py exposes a surviving hand-authored UOp factory, not an enumerable topology grammar")
  else: grammar.append("tinygrad/schedule/wmma/kernels.py exposes fixed/specialized builders, not an enumerable topology grammar")
  return {"schema": SCHEMA, "kind": "blocked", "status": "BLOCKED", "request": req,
          "missing": {"grammar": grammar,
                      "search": ["no enumerator/lowerer converts the required schema into complete exact plans", "no ranking implementation consumes the declared correctness and timing objective"],
                      "run": ["no route-bound executor accepts an exact complete plan and emits parity/timing evidence for the declared workloads"],
                      "gpu": ["no admitted AMD:gfx1100 target is available through a supported execution path (Linux KFD or macOS PCI+AMD)"]},
          "next_action": "implement grammar, plan lowerer, route-bound runner, and provide target GPU/workload inputs; then rerun this request"}

def validate(record: dict[str, Any]) -> None:
  if record.get("schema") != SCHEMA or record.get("kind") not in ("search_request", "required_topology_schema", "blocked"):
    raise ValueError("unknown attention topology contract")
  if record["kind"] == "blocked":
    missing = record.get("missing")
    if record.get("status") != "BLOCKED" or not isinstance(missing, dict) or set(missing) != {"grammar", "search", "run", "gpu"}:
      raise ValueError("BLOCKED records require separate grammar/search/run/gpu lists")
    if any(not isinstance(missing[k], list) or not missing[k] for k in missing): raise ValueError("each blocker class must be nonempty")
    forbidden = {"candidate", "candidates", "result", "results", "selected_plan", "selected_plans", "ranked_candidates", "exact_plan"}
    if forbidden & set(record): raise ValueError("BLOCKED records cannot claim candidates, results, or selected plans")
    validate(record.get("request", {}))
  if record["kind"] == "search_request":
    if tuple(record.get("required_topology_fields", ())) != TOPOLOGY_FIELDS: raise ValueError("request must require complete topology")
    if not isinstance(record.get("request_id"), str) or not record["request_id"]: raise ValueError("request requires request_id")
    if record.get("target") != {"backend": "AMD", "architecture": "gfx1100", "wave_size": 32}: raise ValueError("request target must be AMD:gfx1100 wave32")
    if not isinstance(record.get("workloads"), list) or not record["workloads"]: raise ValueError("request requires workloads")
    if not isinstance(record.get("objective"), dict) or not isinstance(record.get("budget"), dict): raise ValueError("request requires objective and budget")
    if not isinstance(record.get("correctness_constraints"), dict): raise ValueError("request requires correctness constraints")
    schema = record.get("required_topology_schema", {})
    if schema.get("not_a_grammar") is not True or schema.get("status") != "schema_only": raise ValueError("request must declare schema-only search readiness")

def export(record: dict[str, Any], out: str | Path) -> Path:
  validate(record)
  path = Path(out)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
  return path
