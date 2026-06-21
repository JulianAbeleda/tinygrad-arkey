#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"
from extra.qk_probe_harness import probe_io
read_json, write_json = probe_io(OUT)




def phase(phase_id: str, title: str, status: str, inputs: list[str], artifacts: list[str],
          likely_files: list[str], minimum_pass: str, kill: list[str], blocks: list[str] | None = None) -> dict[str, Any]:
  return {
    "id": phase_id,
    "title": title,
    "status": status,
    "inputs": inputs,
    "artifacts": artifacts,
    "likely_files": likely_files,
    "minimum_pass": minimum_pass,
    "kill_conditions": kill,
    "blocks": blocks or [],
  }


def build_plan() -> dict[str, Any]:
  aggregate = read_json("bench/amd-broad-backend-roadmap/result.json", {})
  bb5a1 = read_json("bench/amd-broad-backend-roadmap/bb5a1_pipeline_ir_result.json", {})
  bb5 = read_json("bench/amd-broad-backend-roadmap/software_pipeline_result.json", {})
  bb5a1_pass = bb5a1.get("verdict") == "PASS_PIPELINE_IR_SURFACE" and bool(bb5a1.get("gate_pass"))
  phases = [
    phase(
      "BB-5a.2", "Double-buffered LDS lowering", "READY" if bb5a1_pass else "BLOCKED_ON_BB5A1",
      ["bb5a1_pipeline_ir_result.json", "AMDPipelineStageMeta", "prior hand-UOp byte-identical evidence"],
      ["bb5a2_double_buffer_lds_result.json"],
      ["tinygrad/renderer/amd/schedule.py", "tinygrad/codegen/opt/postrange.py", "tinygrad/codegen/late/linearizer.py",
       "extra/qk_amd_bb5a2_double_buffer_lds_probe.py"],
      "two LDS stages lower from BB-5a.1 metadata into non-byte-identical AMD ISA without changing defaults",
      ["two slots cannot survive lowering", "rendered structure remains byte-identical to serialized path", "requires hand assembly"],
      ["BB-5a.3", "BB-5a.4", "BB-5a.5", "BB-5a.6", "BB-5a.7", "BB-6"],
    ),
    phase(
      "BB-5a.3", "Semantic wait scheduler integration", "BLOCKED_ON_BB5A2",
      ["bb5a2_double_buffer_lds_result.json", "wait_scheduler_result.json", "pipeline dependency groups"],
      ["bb5a3_wait_scheduler_integration_result.json"],
      ["tinygrad/renderer/amd/schedule.py", "extra/qk_amd_bb5a3_wait_scheduler_integration_probe.py"],
      "dependency-aware waits attach to lowered WMMA prefill-shaped instruction stream with reasons and unchanged defaults",
      ["waits remain probe-only", "placement is static text insertion", "correctness requires wait-after-every-load serialization"],
      ["BB-5a.5", "BB-5a.6", "BB-5a.7", "BB-6"],
    ),
    phase(
      "BB-5a.4", "Allocator and live-range control", "BLOCKED_ON_BB5A2",
      ["bb5a2_double_buffer_lds_result.json", "register_resource_result.json", "pipeline resource budgets"],
      ["bb5a4_allocator_resource_result.json"],
      ["tinygrad/renderer/amd/schedule.py", "tinygrad/renderer/isa/__init__.py",
       "extra/qk_amd_bb5a4_allocator_resource_probe.py"],
      "candidate reports VGPR/SGPR/LDS/spill-risk/occupancy and is spill-free or deterministically rejected",
      ["allocation remains accounting-only", "candidate silently spills", "selection/rejection has no resource reason"],
      ["BB-5a.5", "BB-5a.6", "BB-5a.7", "BB-6"],
    ),
    phase(
      "BB-5a.5", "Resource policy", "BLOCKED_ON_BB5A3_BB5A4",
      ["bb5a3_wait_scheduler_integration_result.json", "bb5a4_allocator_resource_result.json", "authority prefill shapes"],
      ["bb5a5_resource_policy_result.json"],
      ["tinygrad/renderer/amd/schedule.py", "extra/qk_amd_bb5a5_resource_policy_probe.py"],
      "policy selects or rejects a pipelined candidate with shape/resource reasons and unchanged defaults",
      ["hardcoded shape-only switch", "fallback unsupported", "default changes before correctness/performance gates"],
      ["BB-5a.6", "BB-5a.7", "BB-6"],
    ),
    phase(
      "BB-5a.6", "Correctness harness", "BLOCKED_ON_BB5A5",
      ["bb5a5_resource_policy_result.json", "lowered pipelined candidate"],
      ["bb5a6_correctness_result.json"],
      ["extra/qk_amd_bb5a6_correctness_probe.py"],
      "small WMMA and one authority prefill matmul correctness pass; graph smoke passes if attempted",
      ["correctness only passes by serialized fallback", "graph replay invalidates staged buffers", "tolerance exceeds policy"],
      ["BB-5a.7", "BB-6"],
    ),
    phase(
      "BB-5a.7", "Performance gate / BB-5 reopen", "BLOCKED_ON_BB5A6",
      ["bb5a6_correctness_result.json", "authority prefill oracle suite", "controlled clock methodology"],
      ["bb5a7_performance_gate_result.json", "software_pipeline_result.json"],
      ["extra/qk_amd_software_pipeline_probe.py", "extra/qk_amd_bb5a7_performance_gate.py"],
      "pure tinygrad authority prefill reaches >=60 TFLOPS with real pipelined ISA, correctness, and unchanged defaults",
      ["uses external code objects", "ISA remains serialized", "win disappears under controlled clock methodology"],
      ["BB-6"],
    ),
    phase(
      "BB-6", "Q8 transfer handoff", "BLOCKED_ON_BB5A7",
      ["bb5a7_performance_gate_result.json", "reopened software_pipeline_result.json"],
      ["q8_transfer_result.json", "model_gate_result.json"],
      ["future q8 transfer scripts"],
      "native q8 consumer <=75us to continue, <=60us strong pass, W==D >=3% before policy promotion",
      ["requires q8-only scheduler patch", "q8 remains above 75us", "model-level W==D does not move"],
    ),
  ]
  return {
    "date": "2026-06-19",
    "phase": "BB-5a_full_implementation_plan",
    "schema": "amd_bb5a_full_implementation_plan_v1",
    "scope_doc": "docs/amd-broad-backend-bb5a-full-implementation-plan-20260619.md",
    "verdict": "BB5A_FULL_PLAN_READY_BB5A2_NEXT" if bb5a1_pass and bb5.get("verdict") == "BLOCKED_REAL_RENDERER_ALLOCATOR_INTEGRATION" else
               "BB5A_FULL_PLAN_BLOCKED_ON_BB5A1",
    "gate_pass": bool(bb5a1_pass),
    "default_behavior_changed": False,
    "current_aggregate_verdict": aggregate.get("verdict"),
    "completed_prerequisites": {
      "BB-5a.1": bb5a1.get("verdict"),
      "BB-5": bb5.get("verdict"),
    },
    "phases": phases,
    "artifact_order": [
      "bb5a2_double_buffer_lds_result.json",
      "bb5a3_wait_scheduler_integration_result.json",
      "bb5a4_allocator_resource_result.json",
      "bb5a5_resource_policy_result.json",
      "bb5a6_correctness_result.json",
      "bb5a7_performance_gate_result.json",
      "software_pipeline_result.json",
      "q8_transfer_result.json",
    ],
    "next": {
      "phase": "BB-5a.2" if bb5a1_pass else "BB-5a.1",
      "implementation_target": "double_buffered_lds_lowering" if bb5a1_pass else "pipeline_ir_surface",
      "minimum_pass": "two LDS stages lower from BB-5a.1 metadata into non-byte-identical AMD ISA without changing defaults" if bb5a1_pass else
                      "PASS_PIPELINE_IR_SURFACE",
    },
    "q8_transfer_rule": "BB-6 remains blocked until BB-5a.7 passes or formally blocks with implemented reusable backend capability.",
  }


def main() -> int:
  result = build_plan()
  write_json("bb5a_full_plan.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a_full_plan.json",
    "verdict": result["verdict"],
    "next": result["next"],
    "phase_count": len(result["phases"]),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
