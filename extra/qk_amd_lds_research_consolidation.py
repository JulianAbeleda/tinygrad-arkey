#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"


def read_json(rel: str, default: Any = None) -> Any:
  path = ROOT / rel
  if not path.exists(): return default
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def doc(path: str, verdict: str, state: str, lesson: str, artifacts: list[str]) -> dict[str, Any]:
  return {"doc": path, "verdict": verdict, "state": state, "lesson": lesson, "artifacts": artifacts}


def main() -> int:
  capture = read_json("bench/amd-broad-backend-roadmap/bb5a8_authority_kernel_capture_result.json", {})
  causal = read_json("bench/amd-broad-backend-roadmap/bb5a9_causal_delta_package_result.json", {})
  tensile = read_json("bench/qk-tensile-extraction/codegen_oracle.json", {})
  pown = read_json("bench/qk-tensile-extraction/shape_matrix.json", {})
  bb5a2 = read_json("bench/amd-broad-backend-roadmap/bb5a2_pipelined_dataflow_result.json", {})
  bb5a3 = read_json("bench/amd-broad-backend-roadmap/bb5a3_wait_scheduler_integration_result.json", {})
  bb5a4 = read_json("bench/amd-broad-backend-roadmap/bb5a4_allocator_resource_result.json", {})

  rows = [
    doc(
      "docs/amd-lds-tiling-existing-primitives-20260617.md",
      "EXPRESSIBLE",
      "banked",
      "LDS, barriers, local address space, and custom UOp templates exist. Expressibility is not the blocker.",
      [],
    ),
    doc(
      "docs/prefill-wmma-lds-tiling-result-20260619.md",
      "REFUTED_AS_BOUNDED_WIN",
      "closed",
      "Naive hand-LDS WMMA tiling was correct but only 1.02x over default. Do not reopen plain LDS tiling as the lever.",
      ["extra/qk_prefill_wmma_lds_probe.py"],
    ),
    doc(
      "docs/route-a-a3-p2-p3-lds-refuted-20260619.md",
      "REFUTED_AS_HAND_KERNEL_FAMILY",
      "closed",
      "Multi-wave LDS-staged GEMM with DBUF/PAD/BK/occupancy levers was correct but far slower than global-direct WMMA.",
      ["bench/qk-codegen-wmma/route_a_a3_lds_multiwave.json"],
    ),
    doc(
      "docs/prefill-own-wmma-kernel-result-20260619.md",
      "REFUTED_CONFIG_SWEEP",
      "closed",
      "More waves, bigger tiles, BLOCK_K, and no-LDS did not break the 42 TFLOPS plateau.",
      ["bench/qk-prefill-own-wmma/sweep.txt"],
    ),
    doc(
      "docs/prefill-codegen-software-pipeline-result-20260619.md",
      "RENDERER_CAPABILITY_REQUIRED",
      "open_project_level",
      "Manual UOp double-buffer prefetch compiled to byte-identical ISA; needs renderer scheduling/lowering capability.",
      ["extra/qk_wmma_pipeline_kernel.py"],
    ),
    doc(
      "docs/prefill-fp16-load-vectorize-renderer-scope-20260619.md",
      "OPEN_BUT_SUBSUMED_BY_BB5A9_ACCEPTANCE",
      "open_project_level",
      "Wide global/LDS read concerns remain relevant, but must now be judged against BB-5a.9 same-kernel causal deltas.",
      ["bench/qk-codegen-wmma/wide_copy.json"],
    ),
    doc(
      "docs/amd-broad-backend-bb5a9-causal-delta-package-20260619.md",
      "CURRENT_AUTHORITY",
      "canonical",
      "Captured tinygrad authority uses WMMA with zero LDS; Tensile uses WMMA plus LDS-staged wide reads/stores and software-pipelined prefetch.",
      ["bench/amd-broad-backend-roadmap/bb5a9_causal_delta_package_result.json"],
    ),
  ]

  canonical = {
    "what_is_closed": [
      "Do not reopen plain LDS tiling as a bounded prefill win.",
      "Do not reopen multi-wave hand-LDS GEMM tuning without a new correctness/performance premise.",
      "Do not retry waves/tile/BLOCK_K/no-LDS config sweeps.",
      "Do not retry manual UOp prefetch expecting schedule movement; prior attempt was byte-identical.",
      "Do not start q8 transfer before pure tinygrad prefill reaches >=60 TFLOPS.",
    ],
    "what_is_open": [
      "Real renderer-level LDS layout/lowering for the authority shape.",
      "Wide LDS read path, specifically ds_load_b128 feeding WMMA.",
      "Software-pipelined K-loop with prologue/steady-state/epilogue and deferred waits.",
      "Resource policy that rejects scratch/private spill regressions before timing.",
    ],
    "reconciliation": (
      "Older LDS-refutation docs killed naive LDS tiling and hand multi-wave LDS kernels. BB-5a.9 does not revive those. "
      "It narrows the open path to Tensile-class staged LDS plus software-pipelined scheduling, proven against a same-kernel tinygrad capture."
    ),
  }

  result = {
    "date": "2026-06-19",
    "schema": "amd_lds_research_consolidation_v1",
    "verdict": "LDS_RESEARCH_CONSOLIDATED_DO_NOT_LOOP",
    "gate_pass": True,
    "current_authority": {
      "tinygrad_capture_verdict": capture.get("verdict"),
      "bb5a9_verdict": causal.get("verdict"),
      "tinygrad_best_tflops": capture.get("timing", {}).get("best_tflops"),
      "tinygrad_lds_bytes": capture.get("resource", {}).get("lds_bytes"),
      "tinygrad_v_wmma": capture.get("mix", {}).get("disasm", {}).get("v_wmma"),
      "tinygrad_ds_load_b128": capture.get("mix", {}).get("disasm", {}).get("ds_load_b128"),
      "tensile_schedule": tensile.get("tensile_schedule"),
      "tensile_instruction_mix": tensile.get("tensile_instruction_mix"),
      "shape_matrix_verdict": pown.get("verdict"),
    },
    "bb5a_skeleton_state": {
      "two_slot_source_skeleton": bb5a2.get("verdict"),
      "wait_scheduler_skeleton": bb5a3.get("verdict"),
      "resource_policy_skeleton": bb5a4.get("verdict"),
    },
    "documents": rows,
    "canonical_rules": canonical,
    "next_tracks": causal.get("parallel_tracks"),
    "next_action": "Use this consolidation as the pre-BB-5a.10 checkpoint. Start only the open tracks; do not reopen closed LDS variants.",
  }
  write_json("lds_research_consolidation_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/lds_research_consolidation_result.json",
    "verdict": result["verdict"],
    "closed_count": len(canonical["what_is_closed"]),
    "open_count": len(canonical["what_is_open"]),
  }, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
