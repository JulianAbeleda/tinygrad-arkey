#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib, re, sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "bench/amd-broad-backend-roadmap"
from extra.qk_probe_harness import probe_io
read_json, write_json = probe_io(OUT)


def read_text(rel: str) -> str:
  path = ROOT / rel
  return path.read_text() if path.exists() else ""




def has_pattern(rel: str, pattern: str) -> bool:
  return re.search(pattern, read_text(rel)) is not None


def current_surface() -> dict[str, Any]:
  schedule = read_text("tinygrad/renderer/amd/schedule.py")
  return {
    "ops_stage_exists": has_pattern("tinygrad/uop/ops.py", r"Ops\.STAGE"),
    "bufferize_opts_exists": has_pattern("tinygrad/schedule/indexing.py", r"class BufferizeOpts"),
    "rangeify_stage_bufferization_exists": has_pattern("tinygrad/schedule/rangeify.py", r"UPat\(Ops\.STAGE"),
    "amd_schedule_meta_exists": "AMDScheduleMeta" in schedule,
    "amd_pipeline_stage_meta_exists": "AMDPipelineStageMeta" in schedule,
    "pipeline_stage_dump_exists": "pipeline_stage_dump" in schedule,
    "pipeline_stage_summary_exists": "pipeline_stage_summary" in schedule,
    "metadata_from_uops_has_pipeline_fields": all(x in schedule for x in ["prefetch_stage", "lds_stage"]),
    "pipeline_optops_exist": has_pattern("tinygrad/codegen/opt/__init__.py", r"PREFETCH|PIPELINE|DOUBLE_BUFFER"),
    "postrange_pipeline_lowering_exists": has_pattern("tinygrad/codegen/opt/postrange.py", r"PREFETCH|PIPELINE|DOUBLE_BUFFER|pipeline"),
    "default_renderer_integration_exists": has_pattern("tinygrad/codegen/__init__.py", r"pipeline_stage|AMDPipelineStageMeta") or
                                           has_pattern("tinygrad/renderer/amd/__init__.py", r"pipeline_stage|AMDPipelineStageMeta"),
  }


def required_schema_fields() -> list[dict[str, str]]:
  return [
    {"field": "pipeline_id", "purpose": "groups one software-pipelined K loop"},
    {"field": "phase", "purpose": "prologue, steady, or epilogue"},
    {"field": "stage_id", "purpose": "logical stage number"},
    {"field": "stage_count", "purpose": "number of active pipeline stages"},
    {"field": "producer_distance", "purpose": "distance from consumed tile to produced tile"},
    {"field": "k_axis", "purpose": "pipelined K loop identity"},
    {"field": "buffer_role", "purpose": "global_load, lds_store, lds_load, wmma_consume, barrier, or wait"},
    {"field": "lds_slot", "purpose": "logical LDS buffer slot"},
    {"field": "dependency_group", "purpose": "future wait/barrier scheduling group"},
    {"field": "semantic_order", "purpose": "coarse order within a stage"},
    {"field": "resource_budget", "purpose": "optional budget for prefetch, LDS, and accumulators"},
  ]


def work_packages(surface: dict[str, Any]) -> list[dict[str, Any]]:
  probe_exists = (ROOT / "extra/qk_amd_bb5a1_pipeline_ir_probe.py").exists()
  result = read_json("bench/amd-broad-backend-roadmap/bb5a1_pipeline_ir_result.json", {})
  return [
    {
      "id": "bb5a1a_stage_schema",
      "target_files": ["tinygrad/renderer/amd/schedule.py"],
      "required": ["AMDPipelineStageMeta", "pipeline_stage_summary", "pipeline_stage_dump"],
      "current_state": "present" if surface["amd_pipeline_stage_meta_exists"] and surface["pipeline_stage_dump_exists"] and surface["pipeline_stage_summary_exists"] else "missing",
      "minimum_pass": "schema serializes every required field without touching renderer output",
    },
    {
      "id": "bb5a1b_stage_extraction",
      "target_files": ["tinygrad/renderer/amd/schedule.py"],
      "required": ["read-only extractor from structured records or UOps", "two-stage WMMA prefill-shaped metadata"],
      "current_state": "present" if surface["amd_pipeline_stage_meta_exists"] and surface["pipeline_stage_dump_exists"] else
                       "partial_existing_stage_primitives" if surface["ops_stage_exists"] and surface["bufferize_opts_exists"] else "missing",
      "minimum_pass": "extractor produces prologue and steady rows with producer_distance=1",
    },
    {
      "id": "bb5a1c_probe",
      "target_files": ["extra/qk_amd_bb5a1_pipeline_ir_probe.py"],
      "required": ["bb5a1_pipeline_ir_result.json"],
      "current_state": "pass" if result.get("verdict") == "PASS_PIPELINE_IR_SURFACE" and result.get("gate_pass") else
                       "present_pending_run" if probe_exists else "missing",
      "minimum_pass": "result proves roles, phases, LDS slots, dependency groups, and default_behavior_changed=false",
    },
    {
      "id": "bb5a1d_roadmap_integration",
      "target_files": ["extra/qk_amd_broad_backend_roadmap.py", "docs/amd-broad-backend-roadmap-result-20260619.md"],
      "required": ["phase_status includes BB-5a.1", "next advances only after passing IR result"],
      "current_state": "missing",
      "minimum_pass": "aggregate artifact names BB-5a.1 implementation/result state explicitly",
    },
  ]


def main() -> int:
  surface = current_surface()
  packages = work_packages(surface)
  bb5a = read_json("bench/amd-broad-backend-roadmap/bb5a_renderer_allocator_scope.json", {})
  result = {
    "date": "2026-06-19",
    "phase": "BB-5a.1_pipeline_ir_scope",
    "schema": "amd_bb5a1_pipeline_ir_scope_v1",
    "scope_doc": "docs/amd-broad-backend-bb5a1-pipeline-ir-scope-20260619.md",
    "verdict": "BB5A1_SCOPE_COMPLETE_IMPLEMENTATION_NOT_READY",
    "gate_pass": True,
    "default_behavior_changed": False,
    "parent_bb5a_verdict": bb5a.get("verdict"),
    "current_surface": surface,
    "required_schema_fields": required_schema_fields(),
    "work_packages": packages,
    "minimum_ir_pass": {
      "stage_count": 2,
      "required_phases": ["prologue", "steady"],
      "required_roles": ["global_load", "lds_store", "lds_load", "wmma_consume"],
      "required_lds_slots": [0, 1],
      "producer_distance": 1,
      "semantic_change_allowed": False,
      "performance_claim_allowed": False,
    },
    "next_action": (
      "Implement BB-5a.1a stage schema and BB-5a.1b read-only extraction in tinygrad/renderer/amd/schedule.py; "
      "then add a probe that emits bb5a1_pipeline_ir_result.json."
    ),
  }
  write_json("bb5a1_pipeline_ir_scope.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a1_pipeline_ir_scope.json",
    "verdict": result["verdict"],
    "gate_pass": result["gate_pass"],
    "work_package_count": len(packages),
  }, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
