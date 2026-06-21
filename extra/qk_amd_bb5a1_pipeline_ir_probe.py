#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib, sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinygrad.renderer.amd.schedule import pipeline_stage_dump, pipeline_stage_metadata_from_records

OUT = ROOT / "bench/amd-broad-backend-roadmap"
from extra.qk_probe_harness import probe_io
read_json, write_json = probe_io(OUT)



def synthetic_wmma_prefill_pipeline_records() -> list[dict[str, Any]]:
  base = {
    "pipeline_id": "prefill_wmma_kloop",
    "stage_count": 2,
    "producer_distance": 1,
    "k_axis": "k_tile",
    "resource_budget": "prefetch_regs_plus_accumulator_budget_tracked_later",
    "source": "synthetic_wmma_prefill_stage_record",
  }
  return [
    {**base, "phase": "prologue", "stage_id": 0, "buffer_role": "global_load", "lds_slot": 0,
     "dependency_group": "prefill_wmma_kloop.prologue.load.k0", "semantic_order": 0, "op": "global_load_tile_k0"},
    {**base, "phase": "prologue", "stage_id": 0, "buffer_role": "lds_store", "lds_slot": 0,
     "dependency_group": "prefill_wmma_kloop.prologue.lds.k0", "semantic_order": 1, "op": "lds_store_tile_k0"},
    {**base, "phase": "prologue", "stage_id": 0, "buffer_role": "barrier", "lds_slot": 0,
     "dependency_group": "prefill_wmma_kloop.prologue.barrier.k0", "semantic_order": 2, "op": "barrier_after_k0"},
    {**base, "phase": "steady", "stage_id": 1, "buffer_role": "global_load", "lds_slot": 1,
     "dependency_group": "prefill_wmma_kloop.steady.prefetch.k_plus_1", "semantic_order": 3, "op": "global_load_tile_k_plus_1"},
    {**base, "phase": "steady", "stage_id": 1, "buffer_role": "lds_store", "lds_slot": 1,
     "dependency_group": "prefill_wmma_kloop.steady.lds.k_plus_1", "semantic_order": 4, "op": "lds_store_tile_k_plus_1"},
    {**base, "phase": "steady", "stage_id": 0, "buffer_role": "lds_load", "lds_slot": 0,
     "dependency_group": "prefill_wmma_kloop.steady.consume.k", "semantic_order": 5, "op": "lds_load_tile_k"},
    {**base, "phase": "steady", "stage_id": 0, "buffer_role": "wmma_consume", "lds_slot": 0,
     "dependency_group": "prefill_wmma_kloop.steady.consume.k", "semantic_order": 6, "op": "wmma_consume_tile_k"},
    {**base, "phase": "steady", "stage_id": 1, "buffer_role": "wait", "lds_slot": 1,
     "dependency_group": "prefill_wmma_kloop.steady.wait.k_plus_1", "semantic_order": 7, "op": "deferred_wait_for_k_plus_1"},
  ]


def main() -> int:
  rows = pipeline_stage_metadata_from_records(synthetic_wmma_prefill_pipeline_records())
  dump = pipeline_stage_dump(rows)
  summary = dump["summary"]
  required_roles = {"global_load", "lds_store", "lds_load", "wmma_consume"}
  required_phases = {"prologue", "steady"}
  roles = set(summary["counts"]["buffer_role"])
  phases = set(summary["counts"]["phase"])
  gate = {
    "stage_count_is_2": summary["stage_counts"] == [2],
    "required_phases_present": required_phases.issubset(phases),
    "required_roles_present": required_roles.issubset(roles),
    "lds_slots_0_and_1_present": {0, 1}.issubset(set(summary["lds_slots"])),
    "dependency_groups_present": summary["dependency_group_count"] >= 4,
    "steady_producer_distance_1_present": summary["has_steady_producer_distance_1"],
    "semantic_order_monotonic": summary["semantic_order_monotonic"],
    "default_behavior_changed": False,
    "performance_claim": False,
  }
  positive_gate_keys = [
    "stage_count_is_2", "required_phases_present", "required_roles_present", "lds_slots_0_and_1_present",
    "dependency_groups_present", "steady_producer_distance_1_present", "semantic_order_monotonic",
  ]
  gate_pass = all(gate[key] for key in positive_gate_keys) and not gate["default_behavior_changed"] and not gate["performance_claim"]
  result = {
    "date": "2026-06-19",
    "phase": "BB-5a.1_pipeline_ir",
    "schema": "amd_bb5a1_pipeline_ir_result_v1",
    "verdict": "PASS_PIPELINE_IR_SURFACE" if gate_pass else "FAIL_PIPELINE_IR_SURFACE",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "scope_doc": "docs/amd-broad-backend-bb5a1-pipeline-ir-scope-20260619.md",
    "implementation": {
      "schema": "AMDPipelineStageMeta",
      "extractor": "pipeline_stage_metadata_from_records",
      "dump": "pipeline_stage_dump",
      "summary": "pipeline_stage_summary",
      "codegen_mutation": False,
    },
    "pipeline_metadata": dump,
    "gate": gate,
    "decision": (
      "BB-5a.1 passes as a read-only IR surface. It does not lower double-buffered LDS, move waits, "
      "change allocation, or claim TFLOPS. BB-5a.2 may scope double-buffered LDS lowering next."
    ),
    "next_action": "Scope BB-5a.2 double-buffered LDS lowering over this two-stage metadata contract.",
  }
  write_json("bb5a1_pipeline_ir_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a1_pipeline_ir_result.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "row_count": summary["row_count"],
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
