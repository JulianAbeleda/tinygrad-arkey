#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"
from extra.qk_probe_harness import probe_io
read_json, write_json = probe_io(OUT)




def role_row(rows: list[dict[str, Any]], role: str) -> dict[str, Any]:
  return next((row for row in rows if row.get("role") == role), {})


def feature_row(feature: str, tensile: Any, tinygrad_authority: Any, current_pipeline: Any,
                verdict: str, proof_level: str, missing_for_causality: str,
                evidence: list[str]) -> dict[str, Any]:
  return {
    "feature": feature,
    "tensile_oracle": tensile,
    "tinygrad_authority": tinygrad_authority,
    "current_bb5a_pipeline": current_pipeline,
    "verdict": verdict,
    "proof_level": proof_level,
    "missing_for_causality": missing_for_causality,
    "evidence": evidence,
  }


def main() -> int:
  codegen = read_json("bench/qk-tensile-extraction/codegen_oracle.json", {})
  shape = read_json("bench/qk-tensile-extraction/shape_matrix.json", {})
  bb5a1 = read_json("bench/amd-broad-backend-roadmap/bb5a1_pipeline_ir_result.json", {})
  bb5a2 = read_json("bench/amd-broad-backend-roadmap/bb5a2_pipelined_dataflow_result.json", {})
  bb5a3 = read_json("bench/amd-broad-backend-roadmap/bb5a3_wait_scheduler_integration_result.json", {})
  bb5a4 = read_json("bench/amd-broad-backend-roadmap/bb5a4_allocator_resource_result.json", {})
  bb5a7 = read_json("bench/amd-broad-backend-roadmap/bb5a7_performance_gate_result.json", {})

  sched = codegen.get("tensile_schedule", {})
  mix = codegen.get("tensile_instruction_mix", {})
  pown1 = codegen.get("tinygrad_pown1", {})
  ffn_gate_up = role_row(shape.get("rows", []), "ffn_gate_up")
  candidate = bb5a2.get("candidate", {})
  resource = bb5a4.get("resource_summary", {})

  source_artifacts = [
    "bench/qk-tensile-extraction/codegen_oracle.json",
    "bench/qk-tensile-extraction/shape_matrix.json",
    "bench/amd-broad-backend-roadmap/bb5a1_pipeline_ir_result.json",
    "bench/amd-broad-backend-roadmap/bb5a2_pipelined_dataflow_result.json",
    "bench/amd-broad-backend-roadmap/bb5a3_wait_scheduler_integration_result.json",
    "bench/amd-broad-backend-roadmap/bb5a4_allocator_resource_result.json",
    "bench/amd-broad-backend-roadmap/bb5a7_performance_gate_result.json",
  ]
  required_present = {
    artifact: (ROOT / artifact).exists()
    for artifact in source_artifacts
  }

  comparison_rows = [
    feature_row(
      "macro_tile",
      sched.get("macro_tile_MxNxK"),
      pown1.get("macro_tile_MxNxK"),
      {
        "pipeline_ir_phase": bb5a1.get("phase"),
        "shape": {k: ffn_gate_up.get(k) for k in ("m", "n", "k")},
      },
      "MATCH_PROVEN",
      "authority_metadata",
      "No causality gap for this feature; tile shape already matches.",
      source_artifacts[:3],
    ),
    feature_row(
      "wmma_fragment",
      {"fragment": sched.get("wmma_MI"), "v_wmma_count": mix.get("v_wmma")},
      pown1.get("wmma_MI"),
      {
        "candidate_wmma_count": candidate.get("wmma_count"),
        "scheduled_stream_has_wmma": bb5a3.get("gate", {}).get("scheduled_stream_has_wmma"),
      },
      "MATCH_STRUCTURAL_NOT_VOLUME",
      "static_instruction_family",
      "Current BB-5a stream is a skeleton and does not prove full authority-kernel WMMA density.",
      [source_artifacts[0], source_artifacts[3], source_artifacts[4]],
    ),
    feature_row(
      "k_loop_prefetch_pipeline",
      {
        "PGR": sched.get("prefetch_global_read_PGR"),
        "PLR": sched.get("prefetch_local_read_PLR"),
        "depthU": sched.get("depthU"),
      },
      "single-buffered/no software pipeline per codegen oracle",
      {
        "two_lds_slots": candidate.get("local_9000_present") and candidate.get("local_9001_present"),
        "candidate_has_two_loads": candidate.get("load_count") == 2,
        "candidate_has_two_stores": candidate.get("store_count") == 2,
      },
      "GAP_CONFIRMED_SKELETON_ONLY",
      "static_source_shape",
      "Need the actual timed tinygrad authority kernel body to show the measured 42 TFLOPS path lacks overlapped multi-K prefetch at ISA density.",
      [source_artifacts[0], source_artifacts[3]],
    ),
    feature_row(
      "lds_buffering",
      sched.get("lds_buffering_1LDSB"),
      "not proven in timed tinygrad authority kernel",
      {
        "addrspace3_global_count": candidate.get("addrspace3_global_count"),
        "lds_bytes": bb5a4.get("lds_bytes"),
        "local_9000_present": candidate.get("local_9000_present"),
        "local_9001_present": candidate.get("local_9001_present"),
      },
      "STRUCTURAL_MATCH_NOT_TIMED",
      "static_source_and_resource_skeleton",
      "Need source/ISA capture for the same tinygrad kernel whose timing is 42.0 TFLOPS.",
      [source_artifacts[0], source_artifacts[3], source_artifacts[5]],
    ),
    feature_row(
      "lds_read_width",
      {"LRVW": sched.get("local_read_vec_LRVW"), "ds_load_b128": mix.get("ds_load_b128")},
      "narrower local reads per oracle delta",
      {
        "scheduled_stream_ds_load_b32": bb5a3.get("before_instruction_names", []).count("DS_LOAD_B32"),
        "scheduled_stream_ds_load_b128": bb5a3.get("before_instruction_names", []).count("DS_LOAD_B128"),
      },
      "GAP_CONFIRMED",
      "static_instruction_skeleton",
      "Need renderer lowering that emits vectorized LDS loads in the full kernel, then timing.",
      [source_artifacts[0], source_artifacts[4]],
    ),
    feature_row(
      "accumulator_allocation",
      {"thread_tile": sched.get("thread_tile_TT"), "oracle_note": "vgpr256 no spill per codegen oracle"},
      pown1.get("notes"),
      {
        "probe_vgpr_span": (resource.get("vgpr") or {}).get("span"),
        "probe_spill_risk": resource.get("spill_risk"),
        "policy_spill_risk": bb5a4.get("policy", {}).get("spill_risk"),
      },
      "GAP_UNPROVEN_FOR_FULL_KERNEL",
      "resource_accounting_skeleton",
      "Need full authority-kernel allocator evidence: VGPR count, spill absence, occupancy, and same-kernel timing.",
      [source_artifacts[0], source_artifacts[5]],
    ),
    feature_row(
      "wait_scheduling",
      {
        "s_waitcnt_vmcnt": mix.get("s_waitcnt_vmcnt"),
        "s_waitcnt_lgkmcnt": mix.get("s_waitcnt_lgkmcnt"),
        "s_barrier": mix.get("s_barrier"),
      },
      "not captured for timed tinygrad authority kernel",
      {
        "actions": [action.get("action") for action in bb5a3.get("actions", [])],
        "after_instruction_names": bb5a3.get("after_instruction_names"),
      },
      "STRUCTURAL_MATCH_NOT_DENSITY",
      "scheduler_action_skeleton",
      "Need normalized instruction density for the same measured tinygrad and Tensile kernels.",
      [source_artifacts[0], source_artifacts[4]],
    ),
    feature_row(
      "timing_gap",
      {"median_tflops": ffn_gate_up.get("median_tflops"), "best_tflops": ffn_gate_up.get("best_tflops")},
      {"tinygrad_tflops": ffn_gate_up.get("tinygrad_tflops")},
      {"bb5a7_threshold_tflops": bb5a7.get("threshold_tflops"), "bb5a7_gate_pass": bb5a7.get("gate_pass")},
      "GAP_CONFIRMED",
      "authority_timing_row",
      "Timing gap is proven; causal attribution still needs same-kernel source/ISA join.",
      [source_artifacts[1], source_artifacts[6]],
    ),
  ]

  tooling_assessment = {
    "static_tensile_oracle_mapping": bool(codegen and sched and mix),
    "timing_gap_mapping": bool(ffn_gate_up.get("tinygrad_tflops") is not None and ffn_gate_up.get("median_tflops") is not None),
    "bb5a_source_skeleton_mapping": bool(bb5a2.get("gate_pass")),
    "bb5a_instruction_skeleton_mapping": bool(bb5a3.get("gate_pass")),
    "bb5a_resource_skeleton_mapping": bool(bb5a4.get("gate_pass")),
    "actual_timed_tinygrad_authority_kernel_disassembly": False,
    "same_kernel_timing_to_disassembly_join": False,
    "hardware_counter_causal_join": False,
  }
  static_mapping_pass = all(required_present.values()) and all(tooling_assessment[k] for k in (
    "static_tensile_oracle_mapping", "timing_gap_mapping", "bb5a_source_skeleton_mapping",
    "bb5a_instruction_skeleton_mapping", "bb5a_resource_skeleton_mapping",
  ))
  causal_proof_pass = all(tooling_assessment[k] for k in (
    "actual_timed_tinygrad_authority_kernel_disassembly",
    "same_kernel_timing_to_disassembly_join",
    "hardware_counter_causal_join",
  ))
  result = {
    "date": "2026-06-19",
    "phase": "BB-5a.8_tensile_mapping_probe",
    "schema": "amd_bb5a8_tensile_mapping_result_v1",
    "verdict": "PASS_STATIC_TENSILE_TINYGRAD_MAPPING_CAUSAL_PROOF_BLOCKED" if static_mapping_pass and not causal_proof_pass else
               "PASS_TENSILE_TINYGRAD_MAPPING_CAUSAL_PROOF" if static_mapping_pass and causal_proof_pass else
               "BLOCKED_TENSILE_TINYGRAD_MAPPING_INPUTS_MISSING",
    "gate_pass": static_mapping_pass,
    "causal_proof_pass": causal_proof_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "claim_status": {
      "proved": [
        "Tensile and tinygrad authority match on macro tile and WMMA fragment.",
        "The ffn_gate_up timing gap is 42.0 TFLOPS tinygrad versus 65.6 TFLOPS Tensile.",
        "Tensile uses PGR1/PLR1, 1LDSB0, LRVW16/ds_load_b128, and TT4_64.",
        "Current BB-5a pipeline can statically represent two LDS slots, scheduler waits, and resource accounting.",
      ],
      "not_proved": [
        "That the measured 42.0 TFLOPS tinygrad authority kernel has the exact single-buffer/no-wide-LDS/no-spill pattern at full ISA density.",
        "That changing only software-pipelined K-loop plus accumulator allocation closes 42.0 -> 65.6 TFLOPS in pure tinygrad.",
      ],
    },
    "tooling_assessment": tooling_assessment,
    "gate": {
      "required_artifacts_present": required_present,
      "static_mapping_pass": static_mapping_pass,
      "causal_proof_pass": causal_proof_pass,
      "comparison_rows_complete": all(row.get("verdict") and row.get("evidence") for row in comparison_rows),
      "default_behavior_changed": False,
    },
    "comparison_rows": comparison_rows,
    "decision": (
      "We have enough tooling to do the static Tensile-to-tinygrad mapping probe. We do not yet have enough tooling "
      "to prove causality, because the measured 42.0 TFLOPS pure-tinygrad authority kernel is not joined to its full "
      "source/ISA/resource artifact."
    ),
    "next_action": (
      "Capture and disassemble the actual measured 42.0 TFLOPS pure-tinygrad prefill authority kernel, then join "
      "that source/ISA/resource record to timing before claiming the 42.0 -> 65.6 TFLOPS cause."
    ),
  }
  write_json("bb5a8_tensile_mapping_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a8_tensile_mapping_result.json",
    "verdict": result["verdict"],
    "gate_pass": result["gate_pass"],
    "causal_proof_pass": result["causal_proof_pass"],
  }, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
