#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-tensile-primitive-transfer"


def read_json(rel: str, default: Any = None) -> Any:
  path = ROOT / rel
  if not path.exists(): return default
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main() -> int:
  shape = read_json("bench/qk-tensile-extraction/shape_matrix.json", {})
  codegen = read_json("bench/qk-tensile-extraction/codegen_oracle.json", {})
  timing_auth = read_json("bench/amd-broad-backend-roadmap/bb5a10_p8_timing_authority_reconciliation_result.json", {})
  readiness = read_json("bench/qk-decode-native-tooling/readiness.json", {})
  schedule = codegen.get("tensile_schedule", {})
  mix = codegen.get("tensile_instruction_mix", {})
  ffn_gate_up = next((r for r in shape.get("rows", []) if r.get("role") == "ffn_gate_up"), {})

  rows = [
    {
      "primitive": "problem_form_dense_gemm",
      "tensile_basis": "Tensile is a benchmark-driven backend for GEMM, GEMM-like, and tensor-contraction problems; selected prefill rows are dense fp16 GEMMs.",
      "prefill_status": "PROVEN_DIRECT",
      "decode_transfer_status": "MOSTLY_BLOCKED",
      "why": "Prefill ffn_gate/up is M=512,N=12288,K=4096 dense fp16 GEMM. Decode q8/MMVQ is batch-1, quantized, and lifecycle/consumer-bound, so the primitive form is not the same.",
      "minimum_pass": "A decode candidate must preserve q8 weight format and lifecycle placement while producing timing-grade movement; dense GEMM substitution alone is not enough.",
      "current_evidence": ["bench/qk-tensile-extraction/shape_matrix.json", "bench/qk-decode-native-tooling/readiness.json"],
    },
    {
      "primitive": "macro_tile_workgroup_threadtile",
      "tensile_basis": {"macro_tile": schedule.get("macro_tile_MxNxK"), "thread_tile": schedule.get("thread_tile_TT"), "workgroup_map": schedule.get("workgroup_map_WGM")},
      "prefill_status": "MATCHED_AS_STATIC_FEATURE",
      "decode_transfer_status": "CONDITIONAL",
      "why": "tinygrad already matched the 128x128x16/WMMA family for prefill, but q8 decode has a different shape and may need smaller or role-joined tiles.",
      "minimum_pass": "Prove a tile family for q8 gate/up that improves the native-to-oracle timing gap by >=30us, or mark it non-transferable.",
      "current_evidence": ["bench/qk-tensile-extraction/codegen_oracle.json", "bench/amd-broad-backend-roadmap/bb5a9_causal_delta_package_result.json"],
    },
    {
      "primitive": "matrix_instruction_wmma",
      "tensile_basis": {"matrix_instruction": schedule.get("wmma_MI"), "tensile_v_wmma": mix.get("v_wmma")},
      "prefill_status": "PROVEN_SHARED",
      "decode_transfer_status": "LOW_DIRECT_TRANSFER",
      "why": "Both paths can use RDNA3 WMMA for dense fp16 work. q8 decode's core work is fused dequant plus dot/MMVQ, not plain fp16 GEMM.",
      "minimum_pass": "Only keep as a decode row if fused dequant->WMMA or equivalent accumulation is shown correct and timing-grade.",
      "current_evidence": ["bench/qk-tensile-extraction/codegen_oracle.json"],
    },
    {
      "primitive": "global_read_vectorization_and_coalescing",
      "tensile_basis": {"global_load_vec": schedule.get("global_load_vec_GLVWA"), "global_read_vec": schedule.get("global_read_vec_GRVW")},
      "prefill_status": "PROVEN_IN_TENSILE",
      "decode_transfer_status": "CONDITIONAL",
      "why": "Dense prefill reads contiguous fp16 tiles. q8 decode reads packed quantized blocks plus scales; vectorization must preserve format and consumer placement.",
      "minimum_pass": "Same-binary q8 ablation with vectorized packed-block loads must show >=30us timing movement or close the row.",
      "current_evidence": ["bench/qk-tensile-extraction/shape_matrix.json", "bench/q8-ffn-amd-scheduler-project/oracle_contract.json"],
    },
    {
      "primitive": "lds_staging_layout",
      "tensile_basis": {"lds_buffering": schedule.get("lds_buffering_1LDSB"), "ds_store_b128": mix.get("ds_store_b128"), "ds_load_b128": mix.get("ds_load_b128")},
      "prefill_status": "DESCRIPTIVE_NOT_SUFFICIENT",
      "decode_transfer_status": "BLOCKED_UNLESS_PAIRED",
      "why": "Tensile uses LDS, but our selected-compatible LDS P8 candidate is correct and slow. LDS by itself is not the primitive; the transferable unit is LDS plus overlap, waits, and resource policy.",
      "minimum_pass": "Do not pursue standalone LDS. Require a paired software-pipelined K-loop/resource candidate that beats the synchronized authority baseline.",
      "current_evidence": [
        "bench/amd-broad-backend-roadmap/bb5a10_p8_performance_result.json",
        "bench/amd-broad-backend-roadmap/bb5a10_p8_bottleneck_classification_result.json",
      ],
    },
    {
      "primitive": "software_pipelined_k_loop",
      "tensile_basis": {"prefetch_global_read": schedule.get("prefetch_global_read_PGR"), "prefetch_local_read": schedule.get("prefetch_local_read_PLR"), "depth_u": schedule.get("depthU")},
      "prefill_status": "PRIMARY_OPEN_TRANSFER_CANDIDATE",
      "decode_transfer_status": "CONDITIONAL_SHARED_SCHEDULER_CAPABILITY",
      "why": "Official Tensile parameters split global/local prefetch; our local oracle names this as the smallest prefill codegen gap. For decode it is relevant only if q8 has a timing-grade scheduler-resource bucket.",
      "minimum_pass": "First complete same-harness authority timing bridge. Then test one K-loop overlap candidate against the valid baseline before any q8 transfer.",
      "current_evidence": ["bench/qk-tensile-extraction/codegen_oracle.json", "bench/amd-broad-backend-roadmap/bb5a10_p8_timing_authority_reconciliation_result.json"],
    },
    {
      "primitive": "wait_barrier_schedule",
      "tensile_basis": {"s_waitcnt_vmcnt": mix.get("s_waitcnt_vmcnt"), "s_waitcnt_lgkmcnt": mix.get("s_waitcnt_lgkmcnt"), "s_barrier": mix.get("s_barrier")},
      "prefill_status": "PART_OF_PIPELINE_CANDIDATE",
      "decode_transfer_status": "CONDITIONAL_SHARED_SCHEDULER_CAPABILITY",
      "why": "Wait/barrier density matters only with staged in-flight movement. It should not be optimized independently from the dataflow it orders.",
      "minimum_pass": "A candidate must show dependency-correct waits with correctness and positive timing movement; otherwise keep this folded into the K-loop row.",
      "current_evidence": ["bench/qk-tensile-extraction/codegen_oracle.json", "bench/amd-broad-backend-roadmap/bb5a10_p4_wait_barrier_result.json"],
    },
    {
      "primitive": "spill_free_accumulator_resource_policy",
      "tensile_basis": {"thread_tile": schedule.get("thread_tile_TT"), "workspace": ffn_gate_up.get("workspace")},
      "prefill_status": "PRIMARY_OPEN_TRANSFER_CANDIDATE",
      "decode_transfer_status": "CONDITIONAL",
      "why": "Prior local evidence says more accumulators can spill and collapse performance. This is a backend resource-control primitive, not a Tensile dependency primitive.",
      "minimum_pass": "Candidate resource metadata must show scratch/private 0 and acceptable VGPR/occupancy before timing promotion.",
      "current_evidence": ["bench/qk-tensile-extraction/codegen_oracle.json", "bench/amd-broad-backend-roadmap/bb5a10_p5_resource_policy_result.json"],
    },
    {
      "primitive": "library_logic_solution_selection",
      "tensile_basis": "Tensile benchmark phases retain winners for problem sizes and emit library logic; rocBLAS typically consumes the winning YAML/kernel configurations.",
      "prefill_status": "PROVEN_EXTERNAL_ARTIFACT_ROUTE",
      "decode_transfer_status": "POLICY_BLOCKED_OR_ARTIFACT_PROJECT",
      "why": "Using the extracted code object is a dependency/policy route. Reproducing solution selection natively is broader than one kernel.",
      "minimum_pass": "Either accept the external artifact policy for prefill only, or define a tinygrad-native selection/tuning project with explicit decode quality and fallback gates.",
      "current_evidence": ["bench/qk-tensile-extraction/shape_matrix.json", "docs/prefill-tensile-DEFINITIVE-source-of-truth-20260619.md"],
    },
    {
      "primitive": "timing_and_launch_authority",
      "tensile_basis": {"tensile_global_size": ffn_gate_up.get("global_size"), "tensile_local_size": ffn_gate_up.get("local_size"), "tensile_median_tflops": ffn_gate_up.get("median_tflops")},
      "prefill_status": "OPEN_SAME_HARNESS_BRIDGE",
      "decode_transfer_status": "REQUIRED_FOR_ANY_TRANSFER",
      "why": "Current P8 reconciliation proved mixed-kernel/mixed-harness TFLOPS rows cannot decide transfer direction.",
      "minimum_pass": "Time captured authority and current candidates under one common synchronized or device-timestamp harness.",
      "current_evidence": ["bench/amd-broad-backend-roadmap/bb5a10_p8_timing_authority_reconciliation_result.json"],
    },
  ]

  phases = [
    {
      "phase": "PTM-0",
      "name": "freeze_matrix_and_sources",
      "goal": "Create a row-level primitive transfer matrix from official Tensile docs plus local artifacts.",
      "minimum_pass": "Every row has a primitive, prefill status, decode transfer status, evidence, and minimum pass/fail criterion.",
      "status": "complete_by_this_scope",
    },
    {
      "phase": "PTM-1",
      "name": "same_harness_authority_bridge",
      "goal": "Stop mixed-harness reasoning before any new kernel work.",
      "minimum_pass": "Captured 43.026 TFLOPS authority kernel and current P8 candidates are timed under one common harness.",
      "status": "next",
    },
    {
      "phase": "PTM-2",
      "name": "prefill_transfer_decision",
      "goal": "Choose exactly one prefill-native transfer candidate row.",
      "minimum_pass": "Candidate must be one of software_pipelined_k_loop, spill_free_accumulator_resource_policy, or timing/launch correction; standalone LDS is disallowed.",
      "status": "blocked_on_PTM_1",
    },
    {
      "phase": "PTM-3",
      "name": "decode_applicability_gate",
      "goal": "Map only scheduler/resource primitives with q8 timing-grade evidence to decode.",
      "minimum_pass": "q8 row needs >=30us same-binary timing movement, W==D quality, and role-joined gate/up evidence.",
      "status": "blocked_on_PTM_2_and_decode_tooling",
    },
    {
      "phase": "PTM-4",
      "name": "native_or_artifact_policy",
      "goal": "Decide whether this stays a native backend project or an external artifact dependency.",
      "minimum_pass": "Policy decision names fallback, default behavior, source/ELF provenance, and model quality gates.",
      "status": "blocked_on_PTM_2",
    },
  ]

  gate = {
    "official_tensile_docs_used": True,
    "local_tensile_shape_matrix_present": bool(shape),
    "local_codegen_oracle_present": bool(codegen),
    "timing_authority_reconciliation_present": timing_auth.get("verdict") == "PASS_BB5A10_P8_TIMING_AUTHORITY_RECONCILED_SAME_HARNESS_REQUIRED",
    "decode_readiness_loaded": bool(readiness),
    "rows_have_transfer_status": all(r.get("decode_transfer_status") and r.get("minimum_pass") for r in rows),
    "standalone_lds_disallowed": any(r["primitive"] == "lds_staging_layout" and r["decode_transfer_status"] == "BLOCKED_UNLESS_PAIRED" for r in rows),
  }
  gate_pass = all(gate.values())
  result = {
    "date": "2026-06-20",
    "schema": "qk_tensile_primitive_transfer_matrix_scope_v1",
    "phase": "PTM-0",
    "verdict": "PASS_TENSILE_PRIMITIVE_TRANSFER_MATRIX_SCOPED" if gate_pass else "BLOCKED_TENSILE_PRIMITIVE_TRANSFER_MATRIX_SCOPE",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "online_sources": [
      {
        "name": "Tensile kernel parameters",
        "url": "https://rocm.docs.amd.com/projects/Tensile/en/docs-7.1.1/src/conceptual/kernel-parameters.html",
        "used_for": "parameter-to-primitive rows: WorkGroup, ThreadTile, MatrixInstruction, PGR, PLR, WGM, LoopUnroll/DepthU, VectorWidth",
      },
      {
        "name": "Tensile benchmark protocol",
        "url": "https://rocm.docs.amd.com/projects/Tensile/en/docs-7.1.1/src/conceptual/benchmarking.html",
        "used_for": "solution-selection and benchmark-phasing rows",
      },
      {
        "name": "Tensile nomenclature",
        "url": "https://rocm.docs.amd.com/projects/Tensile/en/docs-7.0.2/src/reference/nomenclature.html",
        "used_for": "GEMM/tensor-contraction problem-form row",
      },
      {
        "name": "ROCm/Tensile repository note",
        "url": "https://github.com/ROCm/Tensile",
        "used_for": "repository status and relationship to rocBLAS/ROCm libraries",
      },
    ],
    "local_sources": [
      "bench/qk-tensile-extraction/shape_matrix.json",
      "bench/qk-tensile-extraction/codegen_oracle.json",
      "bench/amd-broad-backend-roadmap/bb5a10_p8_timing_authority_reconciliation_result.json",
      "bench/qk-decode-native-tooling/readiness.json",
      "docs/prefill-tensile-DEFINITIVE-source-of-truth-20260619.md",
    ],
    "matrix": rows,
    "phases": phases,
    "gate": gate,
    "next_action": "Run PTM-1 same-harness authority bridge before choosing any new Tensile-transfer implementation row.",
  }
  write_json("scope.json", result)
  print(json.dumps({
    "out": "bench/qk-tensile-primitive-transfer/scope.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "rows": len(rows),
    "next": result["next_action"],
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
