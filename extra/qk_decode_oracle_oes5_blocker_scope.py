#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import shutil
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SEMANTIC = ROOT / "bench/qk-decode-primitive-transfer/decode_oracle_semantic_map_result.json"
LOADER = ROOT / "bench/q8-ffn-amd-scheduler-project/artifact_loader.json"
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_oracle_oes5_blocker_scope_result.json"


def load(path: pathlib.Path, default: Any = None) -> Any:
  return json.loads(path.read_text()) if path.exists() else default


def has_text(path: str, needle: str) -> bool:
  p = ROOT / path
  return p.exists() and needle in p.read_text(errors="ignore")


def main() -> int:
  semantic = load(SEMANTIC, {})
  loader = load(LOADER, {})
  tools = {
    "rocprofv3": shutil.which("rocprofv3") or "/opt/rocm/bin/rocprofv3",
    "rocprof_compute_viewer": shutil.which("rocprof-compute-viewer"),
    "llvm_objdump": shutil.which("llvm-objdump") or "/opt/rocm/llvm/bin/llvm-objdump",
  }
  tool_exists = {k: pathlib.Path(v).exists() if v else False for k, v in tools.items()}

  evidence = {
    "semantic_map_passed": semantic.get("gate_pass") is True,
    "oracle_loader_is_hcq_amdprogram": "AMDProgram" in str(loader.get("route", "")),
    "hcq_invisible_to_rocprof_documented": has_text("docs/inference-perf-measured-map-20260619.md", "rocprofv3 CANNOT trace tinygrad's HCQ/KFD dispatches"),
    "primitive_scope_says_hcq_invisible": has_text("docs/primitive-hcq-attribution-scope-20260619.md", "rocprofv3 is proven useful for HIP/rocBLAS controls but invisible to the tinygrad HCQ"),
    "att_decoder_prior_blocker_documented": has_text("docs/amd-att-decoder-blocker-scope-20260619.md", "rocprofv3 --att"),
    "tools_installed": all(tool_exists.values()),
  }

  missing_to_run_oes5 = [
    {
      "id": "rocprof_visible_oracle_runner",
      "why": "The current q8_mmvq_gateup oracle artifact is loaded through AMDProgram/HCQ; rocprofv3 cannot attribute HCQ dispatch PCs.",
      "construction_path": "Build a tiny HIP host executable that launches the same q8_mmvq_gateup source/object with the same launch geometry and deterministic buffers, then run rocprofv3 --kernel-trace/--att on that HIP process.",
    },
    {
      "id": "native_pc_join_surface",
      "why": "Oracle OES-4 PC ranges are known, but native/C7C PCs must be joined to the same semantic stage names before attribution can compare stages.",
      "construction_path": "Export native/C7C disasm PCs from the existing q8 native probes and map them to S1/S2/S3/S4/S5-equivalent stages.",
    },
    {
      "id": "att_or_counter_fallback_policy",
      "why": "Prior ATT repair docs show ATT capture can be environment-sensitive; OES-5 needs a fallback if thread trace is unavailable.",
      "construction_path": "Prefer ATT PC timeline; if blocked, collect kernel-trace resource fields plus available PMC counters and stop at coarse stall-family attribution.",
    },
  ]

  gates = {
    "oes4_complete": evidence["semantic_map_passed"],
    "profiling_tools_present": evidence["tools_installed"],
    "current_runner_not_rocprof_visible": evidence["oracle_loader_is_hcq_amdprogram"] and evidence["hcq_invisible_to_rocprof_documented"],
    "missing_items_named": len(missing_to_run_oes5) == 3,
  }

  result = {
    "date": "2026-06-20",
    "phase": "DECODE_ORACLE_OES5_BLOCKER_SCOPE",
    "schema": "decode_oracle_oes5_blocker_scope_v1",
    "verdict": "BLOCKED_OES5_NEEDS_ROCPROF_VISIBLE_ORACLE_RUNNER",
    "gate_pass": False,
    "default_behavior_changed": False,
    "performance_claim": False,
    "tools": tools,
    "tool_exists": tool_exists,
    "evidence": evidence,
    "gates": gates,
    "missing_to_run_oes5": missing_to_run_oes5,
    "next_executable_plan": [
      "Create a minimal HIP host runner for q8_mmvq_gateup using the same source, launch geometry (12288,2,1)/(32,4,1), and synthetic deterministic buffers.",
      "Run rocprofv3 --kernel-trace first to prove the dispatch is visible and resources match the extracted artifact envelope.",
      "Run rocprofv3 --att if the decoder path works; otherwise record the ATT blocker and fall back to coarse resource/counter attribution.",
      "Join oracle PCs to the OES-4 semantic stages and compare against native/C7C stage PCs before reopening native scheduling.",
    ],
  }

  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "tools": tool_exists,
    "gates": gates,
    "missing_to_run_oes5": [row["id"] for row in missing_to_run_oes5],
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 1


if __name__ == "__main__":
  raise SystemExit(main())

