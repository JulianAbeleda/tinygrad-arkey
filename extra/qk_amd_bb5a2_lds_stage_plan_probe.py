#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib, sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinygrad.renderer.amd.schedule import (
  lds_stage_plan_dump, lds_stage_plan_from_pipeline, pipeline_stage_metadata_from_records,
)

OUT = ROOT / "bench/amd-broad-backend-roadmap"
from extra.qk_probe_harness import probe_io
read_json, write_json = probe_io(OUT)




def main() -> int:
  pipe = read_json("bench/amd-broad-backend-roadmap/bb5a1_pipeline_ir_result.json", {})
  rows = pipeline_stage_metadata_from_records((pipe.get("pipeline_metadata") or {}).get("rows", []))
  # 128x16 half tile is the current prefill-shaped planning unit from the hand-UOp double-buffer attempt.
  slot_bytes = 128 * 16 * 2
  plan = lds_stage_plan_from_pipeline(rows, slot_bytes=slot_bytes)
  dump = lds_stage_plan_dump(plan)
  summary = dump["summary"]
  gate = {
    "input_pipeline_ir_pass": pipe.get("verdict") == "PASS_PIPELINE_IR_SURFACE" and bool(pipe.get("gate_pass")),
    "slot_count_is_2": summary["slot_count"] == 2,
    "slots_0_and_1_present": summary["has_two_slots"],
    "alias_safe": summary["alias_safe"],
    "required_local_bytes_recorded": summary["required_local_bytes"] == slot_bytes * 2,
    "dependency_groups_present": summary["dependency_group_count"] >= 4,
    "lowering_status_planned": summary["lowering_status"] == "planned",
    "default_behavior_changed": False,
    "performance_claim": False,
  }
  positive = [
    "input_pipeline_ir_pass", "slot_count_is_2", "slots_0_and_1_present", "alias_safe",
    "required_local_bytes_recorded", "dependency_groups_present", "lowering_status_planned",
  ]
  gate_pass = all(gate[x] for x in positive) and not gate["default_behavior_changed"] and not gate["performance_claim"]
  result = {
    "date": "2026-06-19",
    "phase": "BB-5a.2_layer_1_lds_stage_plan",
    "schema": "amd_bb5a2_lds_stage_plan_result_v1",
    "verdict": "PASS_LDS_STAGE_PLAN" if gate_pass else "FAIL_LDS_STAGE_PLAN",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "slot_bytes": slot_bytes,
    "lds_stage_plan": dump,
    "gate": gate,
    "decision": (
      "Layer 1 passes: pipeline lds_slot=0/1 maps to deterministic alias-safe LDS slots. "
      "This is still metadata-only; Layer 2 must implement the gated postrange/rangeify lowering hook."
    ),
    "next_action": "Implement BB-5a.2 Layer 2: preserve lds_slot=0/1 through STAGE/DEFINE_LOCAL lowering.",
  }
  write_json("bb5a2_lds_stage_plan_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a2_lds_stage_plan_result.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "required_local_bytes": summary["required_local_bytes"],
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
