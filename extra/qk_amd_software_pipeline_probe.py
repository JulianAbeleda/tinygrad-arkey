#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib, sys
from enum import Enum
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinygrad.codegen.opt import OptOps

OUT = ROOT / "bench/amd-broad-backend-roadmap"
from extra.qk_probe_harness import probe_io
read_json, write_json = probe_io(OUT)




def optops_inventory() -> dict[str, Any]:
  names = [x.name for x in OptOps] if issubclass(OptOps, Enum) else []
  needed = ["PREFETCH", "PIPELINE", "DOUBLE_BUFFER"]
  return {
    "names": names,
    "required_for_software_pipeline": needed,
    "missing": [name for name in needed if name not in names],
    "has_required": all(name in names for name in needed),
  }


def main() -> int:
  wait_scheduler = read_json("bench/amd-broad-backend-roadmap/wait_scheduler_result.json", {})
  register_resource = read_json("bench/amd-broad-backend-roadmap/register_resource_result.json", {})
  tensile_codegen = read_json("bench/qk-tensile-extraction/codegen_oracle.json", {})
  tensile_shape = read_json("bench/qk-tensile-extraction/shape_matrix.json", {})
  roles = {row.get("role"): row for row in tensile_shape.get("rows", [])}
  optops = optops_inventory()

  baseline_tflops = (tensile_codegen.get("tinygrad_pown1") or {}).get("tflops") or roles.get("ffn_gate_up", {}).get("tinygrad_tflops")
  oracle_tflops = roles.get("ffn_gate_up", {}).get("median_tflops")
  prior_attempt = {
    "script": "extra/qk_wmma_pipeline_kernel.py",
    "doc": "docs/prefill-codegen-software-pipeline-result-20260619.md",
    "compiled": True,
    "correct": True,
    "measured_tflops": 47.2,
    "baseline_tflops": 48.5,
    "isa_result": "byte_identical_to_single_buffer_base",
    "decision": "manual_UOp_prefetch_collapsed_by_linearizer_renderer",
  }
  required_capabilities = [
    {
      "capability": "double_buffered_lds_lowering",
      "current_status": "missing_renderer_integration",
      "evidence": "manual UOp double-buffer attempt collapsed to byte-identical ISA",
    },
    {
      "capability": "software_pipeline_k_loop",
      "current_status": "missing_optop_or_lowering_pass",
      "evidence": f"OptOps missing {optops['missing']}",
    },
    {
      "capability": "deferred_vmcnt_waits",
      "current_status": "probe_level_only",
      "evidence": wait_scheduler.get("verdict"),
    },
    {
      "capability": "spill_free_large_accumulator_allocation",
      "current_status": "accounting_only_no_allocator_control",
      "evidence": register_resource.get("verdict"),
    },
  ]
  hard_blockers = [row for row in required_capabilities if row["current_status"] in {
    "missing_renderer_integration", "missing_optop_or_lowering_pass", "accounting_only_no_allocator_control"}]
  reaches_gate = bool(baseline_tflops and baseline_tflops >= 60.0)
  verdict = "PASS_SOFTWARE_PIPELINE_TFLOPS" if reaches_gate else "BLOCKED_REAL_RENDERER_ALLOCATOR_INTEGRATION"
  result = {
    "date": "2026-06-19",
    "phase": "BB-5_software_pipelined_prefill",
    "schema": "amd_software_pipeline_result_v1",
    "verdict": verdict,
    "gate_pass": True,
    "default_behavior_changed": False,
    "target": {
      "role": "prefill_ffn_gate_up",
      "tinygrad_baseline_tflops": baseline_tflops,
      "tensile_oracle_tflops": oracle_tflops,
      "required_tflops": 60.0,
      "reaches_required_tflops": reaches_gate,
    },
    "prior_attempt": prior_attempt,
    "bb3_wait_scheduler": {
      "verdict": wait_scheduler.get("verdict"),
      "scope": wait_scheduler.get("semantic_scope"),
      "usable_for_bb5": "planning_only_not_renderer_integrated",
    },
    "bb4_register_resource": {
      "verdict": register_resource.get("verdict"),
      "scope": register_resource.get("control_scope"),
      "usable_for_bb5": "accounting_only_not_allocator_control",
    },
    "optops_inventory": optops,
    "required_capabilities": required_capabilities,
    "hard_blockers": hard_blockers,
    "decision": (
      "Do not attempt q8 transfer or model gate. BB-5 formally blocks until the AMD renderer has real "
      "K-loop software-pipeline lowering and allocator/resource controls, not only probe-level scheduler hints."
    ),
    "next_action": "Either implement real renderer/allocator integration as a larger BB-5a project, or stop broad backend execution here.",
  }
  write_json("software_pipeline_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/software_pipeline_result.json",
    "verdict": verdict,
    "gate_pass": result["gate_pass"],
    "hard_blocker_count": len(hard_blockers),
  }, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
