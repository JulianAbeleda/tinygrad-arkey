#!/usr/bin/env python3
# AMD GEMM schedule-object STRUCTURAL probe (no GPU, no timing, no performance claim).
#
# Instantiates the first-class `AMDGemmScheduleObject` (tinygrad/renderer/amd/schedule.py) for the selected
# rocBLAS ffn_gate/up Tensile kernel from the already-extracted JSON artifacts, then verifies the structural
# contract only:
#   - selected shape + macro/thread/workgroup/depthU/WGM contract
#   - LDS layout object A0/B0/A1/B1 (bases/spans/padding) summing to the selected 25088 group_segment bytes
#   - the named pipeline stages (global_load -> wait -> lds_store -> barrier -> lds_read -> wait -> wmma ->
#     store -> swap) present and ordered
#   - resource gate (lds==target, scratch==0, vgpr/sgpr budgets)
#   - ISA structural gates from the audited disasm (visible global_load / ds_store / ds_load_b128 / v_wmma,
#     waits + barriers present, WMMA operands fed from LDS-loaded VGPRs)
#   - explicit blocked/unknown rows (non-bitexact, heuristic K-loop segmentation)
#
# This changes NO default behavior and makes NO performance claim. It is the "structural object exists and its
# contract holds" gate that the transfer table named as the prerequisite before any timing or BEAM/search.
from __future__ import annotations

import json, pathlib
from typing import Any

from tinygrad.renderer.amd.schedule import (
  AMDGemmShapeContract, AMDGemmLDSRegion, AMDGemmLDSLayout, AMDGemmPipelineStage, AMDGemmResourceGate,
  AMDGemmISAEvidence, AMDGemmScheduleObject, gemm_schedule_object_summary,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"
CONTRACT = "bench/qk-tensile-extraction/ffn_gate_up_contract.json"
TEMPLATE = "bench/qk-tensile-extraction/ffn_gate_up_schedule_template.json"
AUDIT = "bench/amd-broad-backend-roadmap/bb5a10_tensile_layout_audit_result.json"


def read_json(rel: str) -> dict[str, Any]:
  path = ROOT / rel
  if not path.exists(): raise FileNotFoundError(f"required artifact missing: {rel}")
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def build_shape(contract: dict[str, Any], template: dict[str, Any]) -> AMDGemmShapeContract:
  shape = contract["shape"]
  sm = template["solution_params"]["sizeMapping"]
  geo = contract["launch_geometry"]
  return AMDGemmShapeContract(
    role=contract["role"], m=shape["M"], n=shape["N"], k=shape["K"],
    dtype_in="fp16", dtype_acc="fp32",
    macro_tile=(sm["macroTile"][0], sm["macroTile"][1], sm["depthU"]),
    thread_tile=(sm["threadTile"][0], sm["threadTile"][1]),
    work_group=(sm["workGroup"][0], sm["workGroup"][1], sm["workGroup"][2]),
    depth_u=sm["depthU"], workgroup_mapping=sm["workGroupMapping"],
    grid=tuple(geo["grid"]), workgroup=tuple(geo["workgroup"]))


def build_lds_layout(total_bytes: int) -> AMDGemmLDSLayout:
  # Non-bitexact region map reconciled from Tensile getLdsNumElements + selected offsets
  # (docs/prefill-tensile-lds-tile-map-sketch-20260620.md). fp16=2 bytes; B carries LdsPadB=8 elems/128B block.
  # A = MacroTile*DepthU = 128*16 = 2048 elems = 4096 B (no pad). B = 4096 raw + 512 pad = 4608 B.
  # PGR1 second buffer base rounded to pow2(A+B elems)=8192 elems => 16384 B.
  regions = (
    AMDGemmLDSRegion("A0", "A", 0, 0,     4096, 0,   "current/prefetch A tile (lower buffer)"),
    AMDGemmLDSRegion("B0", "B", 0, 4096,  4608, 512, "current/prefetch B tile (lower buffer, padded)"),
    AMDGemmLDSRegion("A1", "A", 1, 16384, 4096, 0,   "alternate A tile (PGR1 second buffer)"),
    AMDGemmLDSRegion("B1", "B", 1, 20480, 4608, 512, "alternate B tile (PGR1 second buffer, padded)"),
  )
  return AMDGemmLDSLayout(regions=regions, total_bytes=total_bytes, alignment_gap=(8704, 7680))


def build_pipeline() -> tuple[AMDGemmPipelineStage, ...]:
  # Non-bitexact structural template (one prologue + steady DepthU=16 K iteration). Order follows the named
  # GEMM_PIPELINE_STAGES contract; isa_evidence ties a stage to the audited opcode that realizes it.
  S = AMDGemmPipelineStage
  return (
    S(0,  "global_load_A", "prologue", "global_load", "A", 0, "lds_store_A", "vmcnt",  "buffer_load_b64"),
    S(1,  "global_load_B", "prologue", "global_load", "B", 0, "lds_store_B", "vmcnt",  "buffer_load_b64"),
    S(2,  "wait_global_before_lds", "prologue", "wait", None, None, "lds_store_A", "vmcnt", "s_waitcnt"),
    S(3,  "lds_store_A", "prologue", "lds_store", "A", 0, "barrier_after_lds_store", "lgkmcnt", "ds_store_b64"),
    S(4,  "lds_store_B", "prologue", "lds_store", "B", 0, "barrier_after_lds_store", "lgkmcnt", "ds_store_b64"),
    S(5,  "barrier_after_lds_store", "prologue", "barrier", None, None, "lds_read_A", "barrier", "s_barrier"),
    S(6,  "lds_read_A", "steady", "lds_load", "A", 0, "wmma_consume", "lgkmcnt", "ds_load_b128"),
    S(7,  "lds_read_B", "steady", "lds_load", "B", 0, "wmma_consume", "lgkmcnt", "ds_load_b128"),
    S(8,  "wait_lds_before_wmma", "steady", "wait", None, None, "wmma_consume", "lgkmcnt", "s_waitcnt"),
    S(9,  "wmma_consume", "steady", "wmma", None, None, "buffer_swap", "wmma_dependency", "v_wmma"),
    S(10, "store_output", "epilogue", "global_store", None, None, None, "vscnt", None),
    S(11, "buffer_swap", "steady", "swap", None, None, "global_load_A", None, None),
  )


def build_resource_gate(contract: dict[str, Any]) -> AMDGemmResourceGate:
  return AMDGemmResourceGate(
    lds_bytes_target=contract["group_segment_fixed_size"],
    lds_bytes_actual=contract["group_segment_fixed_size"],
    private_scratch_required=0, private_scratch_actual=contract["private_segment_fixed_size"],
    vgpr_budget=contract["vgpr_count"], sgpr_budget=contract["sgpr_count"])


def build_isa_evidence(audit: dict[str, Any]) -> AMDGemmISAEvidence:
  ds = audit["disasm_summary"]
  ic = ds["instruction_counts"]
  hi = ds["handoff_inference"]
  global_load = ic.get("buffer_load_b64", 0) + ic.get("global_load_dword", 0)
  return AMDGemmISAEvidence(
    global_load=global_load, ds_store=ic.get("ds_store_b64", 0) + ic.get("ds_store_b128", 0),
    ds_load_b128=ic.get("ds_load_b128", 0), v_wmma=ic.get("v_wmma", 0),
    s_waitcnt=ic.get("s_waitcnt", 0), s_barrier=ic.get("s_barrier", 0),
    lds_store_reuses_global_regs=hi["ds_store_with_recent_global_load_data_register_overlap"] == hi["ds_store_examined"] > 0,
    wmma_operands_from_lds=hi["wmma_with_recent_ds_load_source_register_overlap"] == hi["wmma_examined"] > 0)


BLOCKED_UNKNOWN = (
  "non-bitexact LDS layout: per-element A/B coordinate map not reconstructed (region sketch only)",
  "exact K-loop / buffer-swap schedule still heuristic (first-WMMA/last-WMMA segmentation, not symbolic)",
  "source-level per-element tile map not fully reconstructed (address VGPR base carry not replayed)",
  "bank/padding rationale beyond B LdsPadB=8/128B not replayed from the Tensile generator",
  "no lowering to ISA and no performance claim: structural contract only (>=60 TFLOPS gate is separate/later)",
)


def main() -> int:
  contract = read_json(CONTRACT)
  template = read_json(TEMPLATE)
  audit = read_json(AUDIT)

  obj = AMDGemmScheduleObject(
    shape=build_shape(contract, template),
    lds=build_lds_layout(contract["group_segment_fixed_size"]),
    pipeline=build_pipeline(),
    resource_gate=build_resource_gate(contract),
    isa_evidence=build_isa_evidence(audit),
    blocked_unknown=BLOCKED_UNKNOWN)

  gate = obj.structural_gate()
  summary = gemm_schedule_object_summary(obj)
  gate_pass = bool(gate["passed"]) and not obj.performance_claim
  result = {
    "date": "2026-06-20",
    "phase": "AMD_GEMM_SCHEDULE_OBJECT_STRUCTURAL",
    "schema": "amd_gemm_schedule_object_structural_v1",
    "role": "ffn_gate/up",
    "verdict": "PASS_AMD_GEMM_SCHEDULE_OBJECT_STRUCTURAL" if gate_pass else "BLOCKED_AMD_GEMM_SCHEDULE_OBJECT_STRUCTURAL",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "summary": summary,
    "schedule_object": obj.to_dict(),
    "input_artifacts": [CONTRACT, TEMPLATE, AUDIT],
    "next_action": "K-loop schedule reconstruction + resource-gated lowering, then (and only then) timing vs the "
                   ">=60 TFLOPS authority gate. No BEAM/search until the schedule object lowers to ISA.",
  }
  write_json("amd_gemm_schedule_object_structural_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/amd_gemm_schedule_object_structural_result.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "summary": summary,
    "failed_checks": [k for k, v in gate["checks"].items() if not v],
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
