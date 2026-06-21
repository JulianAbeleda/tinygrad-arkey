#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib, re, sys
from enum import Enum
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinygrad.codegen.opt import OptOps

OUT = ROOT / "bench/amd-broad-backend-roadmap"
from extra.qk_probe_harness import probe_io
read_json, write_json = probe_io(OUT)



def read_text(rel: str) -> str:
  path = ROOT / rel
  return path.read_text() if path.exists() else ""



def optops_inventory() -> dict[str, Any]:
  names = [x.name for x in OptOps] if issubclass(OptOps, Enum) else []
  required = ["PREFETCH", "PIPELINE", "DOUBLE_BUFFER"]
  return {
    "names": names,
    "required_for_bb5a": required,
    "missing": [name for name in required if name not in names],
    "has_required": all(name in names for name in required),
  }


def count_pattern(rel: str, pattern: str) -> int:
  return len(re.findall(pattern, read_text(rel)))


def repo_static_evidence() -> dict[str, Any]:
  schedule_py = read_text("tinygrad/renderer/amd/schedule.py")
  amd_renderer_refs = {
    rel: count_pattern(rel, r"amd\.schedule|apply_instruction_schedule|plan_schedule_actions|AMDScheduleMeta")
    for rel in [
      "tinygrad/renderer/amd/__init__.py",
      "tinygrad/renderer/amd/dsl.py",
      "tinygrad/renderer/isa/__init__.py",
      "tinygrad/codegen/__init__.py",
    ]
  }
  return {
    "schedule_module_present": bool(schedule_py),
    "schedule_probe_helpers": {
      "metadata_from_uops": "metadata_from_uops" in schedule_py,
      "apply_instruction_schedule": "apply_instruction_schedule" in schedule_py,
      "resource_summary_from_instructions": "resource_summary_from_instructions" in schedule_py,
    },
    "renderer_integration_refs": amd_renderer_refs,
    "renderer_integrated": any(v > 0 for v in amd_renderer_refs.values()),
    "linearizer_has_stage_policy": count_pattern("tinygrad/codegen/late/linearizer.py", r"prefetch|pipeline|double_buffer|lds_stage") > 0,
    "postrange_has_pipeline_opt": count_pattern("tinygrad/codegen/opt/postrange.py", r"PREFETCH|PIPELINE|DOUBLE_BUFFER") > 0,
    "isa_renderer_has_regalloc_hooks": all(x in read_text("tinygrad/renderer/isa/__init__.py") for x in ["pre_regalloc_matcher", "post_regalloc_matcher"]),
  }


def capability_rows(optops: dict[str, Any], static: dict[str, Any], bb5: dict[str, Any]) -> list[dict[str, Any]]:
  prior = bb5.get("prior_attempt", {})
  wait_scope = ((bb5.get("bb3_wait_scheduler") or {}).get("usable_for_bb5") or "")
  reg_scope = ((bb5.get("bb4_register_resource") or {}).get("usable_for_bb5") or "")
  return [
    {
      "id": "pipeline_ir_surface",
      "required": "stage-aware IR or opt vocabulary for prologue/steady/epilogue K-loop pipeline",
      "current_state": "missing" if optops["missing"] else "present_requires_lowering_validation",
      "evidence": {"missing_optops": optops["missing"], "postrange_has_pipeline_opt": static["postrange_has_pipeline_opt"]},
      "minimum_pass": "stage IDs survive lowering and metadata dumping for a WMMA prefill-shaped kernel",
    },
    {
      "id": "double_buffered_lds_lowering",
      "required": "two LDS stages with alternating producer/consumer semantics",
      "current_state": "missing_renderer_integration" if prior.get("isa_result") == "byte_identical_to_single_buffer_base" else "unknown",
      "evidence": prior,
      "minimum_pass": "generated ISA is non-byte-identical and metadata shows at least two LDS stages",
    },
    {
      "id": "semantic_wait_scheduler_integration",
      "required": "dependency-aware s_waitcnt/s_clause/s_delay_alu placement in AMD rendering",
      "current_state": "probe_level_only" if "planning_only" in wait_scope or not static["renderer_integrated"] else "integrated_requires_validation",
      "evidence": {
        "bb3_usable_for_bb5": wait_scope,
        "renderer_integration_refs": static["renderer_integration_refs"],
      },
      "minimum_pass": "lowered WMMA prefill kernel shows semantic wait movement with correctness unchanged",
    },
    {
      "id": "allocator_live_range_control",
      "required": "spill-free register allocation/resource policy for accumulator plus prefetch live ranges",
      "current_state": "accounting_only" if "accounting_only" in reg_scope else "unknown",
      "evidence": {
        "bb4_usable_for_bb5": reg_scope,
        "isa_renderer_has_regalloc_hooks": static["isa_renderer_has_regalloc_hooks"],
      },
      "minimum_pass": "candidate reports VGPR/SGPR/LDS/spill/occupancy metadata and either stays spill-free or is rejected",
    },
    {
      "id": "resource_policy",
      "required": "deterministic select/reject policy tied to shape, stage count, VGPR/SGPR/LDS, and occupancy",
      "current_state": "missing_control_policy",
      "evidence": "BB-4 reports resource spans but no schedule selection or rejection control",
      "minimum_pass": "policy artifact explains selection/rejection before rendering a pipelined candidate",
    },
    {
      "id": "correctness_harness",
      "required": "small WMMA and authority prefill correctness checks for pipelined lowering",
      "current_state": "missing_for_real_renderer_path",
      "evidence": "prior hand-UOp path was correct, but it did not render a real pipelined schedule",
      "minimum_pass": "correctness passes on small deterministic WMMA and one authority prefill matmul",
    },
    {
      "id": "performance_gate",
      "required": "pure tinygrad authority prefill matmul >=60 TFLOPS",
      "current_state": "blocked_below_gate",
      "evidence": bb5.get("target", {}),
      "minimum_pass": "rerun BB-5 with pure tinygrad >=60 TFLOPS and real pipelined ISA evidence",
    },
  ]


def main() -> int:
  bb5 = read_json("bench/amd-broad-backend-roadmap/software_pipeline_result.json", {})
  optops = optops_inventory()
  static = repo_static_evidence()
  rows = capability_rows(optops, static, bb5)
  missing = [row for row in rows if row["current_state"] not in {"present_requires_lowering_validation", "integrated_requires_validation"}]
  gate_pass = not missing and (bb5.get("target") or {}).get("reaches_required_tflops") is True
  result = {
    "date": "2026-06-19",
    "phase": "BB-5a_renderer_allocator_scope",
    "schema": "amd_bb5a_renderer_allocator_scope_v1",
    "scope_doc": "docs/amd-broad-backend-bb5a-renderer-allocator-scope-20260619.md",
    "verdict": "BB5A_READY_TO_REOPEN_BB5" if gate_pass else "BB5A_SCOPE_COMPLETE_IMPLEMENTATION_NOT_READY",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "parent_bb5_verdict": bb5.get("verdict"),
    "optops_inventory": optops,
    "repo_static_evidence": static,
    "capability_rows": rows,
    "missing_capabilities": missing,
    "completion_gate": {
      "pipeline_ir_surface": "stage IDs survive lowering and metadata dumping",
      "double_buffered_lds_lowering": "two LDS stages visible in metadata and non-byte-identical ISA",
      "semantic_wait_scheduler_integration": "wait placement integrated into AMD renderer path",
      "allocator_live_range_control": "spill-free or deterministic rejection with resource metadata",
      "resource_policy": "shape/resource select-reject reasons emitted before render",
      "correctness_harness": "small WMMA and authority prefill correctness pass",
      "performance_gate": "pure tinygrad authority prefill >=60 TFLOPS",
    },
    "next_action": (
      "Implement BB-5a.1 pipeline IR surface, then BB-5a.2 double-buffered LDS lowering, then BB-5a.3/5a.4 "
      "renderer wait scheduling and allocator/resource control. Do not start BB-6 q8 transfer."
    ),
  }
  write_json("bb5a_renderer_allocator_scope.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a_renderer_allocator_scope.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "missing_capability_count": len(missing),
  }, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
