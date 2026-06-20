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
  # Inputs (provenance + gate sources). All read defensively; gate flags presence.
  matrix = read_json("bench/qk-tensile-primitive-transfer/scope.json", {})
  shape = read_json("bench/qk-tensile-extraction/shape_matrix.json", {})
  codegen = read_json("bench/qk-tensile-extraction/codegen_oracle.json", {})
  timing_auth = read_json("bench/amd-broad-backend-roadmap/bb5a10_p8_timing_authority_reconciliation_result.json", {})
  readiness = read_json("bench/qk-decode-native-tooling/readiness.json", {})

  schedule = codegen.get("tensile_schedule", {})
  mix = codegen.get("tensile_instruction_mix", {})
  ffn_gate_up = next((r for r in shape.get("rows", []) if r.get("role") == "ffn_gate_up"), {})
  ffn_down = next((r for r in shape.get("rows", []) if r.get("role") == "ffn_down"), {})
  attn_q_o = next((r for r in shape.get("rows", []) if r.get("role") == "attn_q_o"), {})
  perf = timing_auth.get("performance_summary", {})
  ident = timing_auth.get("identity_comparison", {})
  prior_auth = ident.get("prior_authority_kernel", {})
  prior_auth_mix = prior_auth.get("instruction_counts", {})
  start_gate = readiness.get("start_gate", {})
  oracle_gap = next((f.get("timing", {}).get("gap_us") for f in readiness.get("feature_join", [])
                     if f.get("timing", {}).get("gap_us") is not None), None)

  # --- Two corrections the Explanation Track must state explicitly (cross-file inconsistencies) ---
  corrections = [
    {
      "id": "authority_43_is_global_direct_not_tensile_lds",
      "claim": "The captured 43.026 TFLOPS authority kernel is NOT the Tensile LDS-staged kernel.",
      "evidence": {
        "prior_authority_kernel_name": prior_auth.get("name"),
        "v_wmma": prior_auth_mix.get("v_wmma"),
        "ds_load_b128": prior_auth_mix.get("ds_load_b128"),
        "interpretation": "It is tinygrad's own global-direct WMMA authority (ds_load_b128=0, 64 v_wmma). "
                          "Do not conflate '43 TFLOPS captured authority' with the Tensile schedule.",
      },
      "source": "bench/amd-broad-backend-roadmap/bb5a10_p8_timing_authority_reconciliation_result.json",
    },
    {
      "id": "v_wmma_count_is_scope_dependent",
      "claim": "v_wmma count differs by scope: whole-.so disasm vs isolated selected-function body.",
      "evidence": {
        "whole_object_v_wmma": mix.get("v_wmma"),
        "isolated_body_v_wmma": 80,
        "isolated_body_extra": "256 v_fma_mix; selected function uses ds_store_b64 (not ds_store_b128) for global->LDS stores.",
        "cite": "Use the per-body 80 v_wmma + 256 v_fma_mix numbers; the whole-object 13810 is the full code object.",
      },
      "source": "docs/amd-broad-backend-bb5a10-tensile-layout-audit-20260619.md",
    },
  ]

  # --- Track 1: Tensile Explanation ---
  track_explanation = {
    "track": "tensile_explanation",
    "goal": "Explain why the selected Tensile kernel works, with claim discipline and no mixed-harness reasoning.",
    "official_parameter_map": [
      {"param": "WorkGroup / WGM", "value": schedule.get("workgroup_map_WGM"), "primitive_row": "macro_tile_workgroup_threadtile"},
      {"param": "ThreadTile (TT)", "value": schedule.get("thread_tile_TT"), "primitive_row": "macro_tile_workgroup_threadtile"},
      {"param": "MacroTile (MT)", "value": schedule.get("macro_tile_MxNxK"), "primitive_row": "macro_tile_workgroup_threadtile"},
      {"param": "MatrixInstruction (MI)", "value": schedule.get("wmma_MI"), "primitive_row": "matrix_instruction_wmma"},
      {"param": "PrefetchGlobalRead (PGR)", "value": schedule.get("prefetch_global_read_PGR"), "primitive_row": "software_pipelined_k_loop"},
      {"param": "PrefetchLocalRead (PLR)", "value": schedule.get("prefetch_local_read_PLR"), "primitive_row": "software_pipelined_k_loop"},
      {"param": "DepthU", "value": schedule.get("depthU"), "primitive_row": "software_pipelined_k_loop"},
      {"param": "GlobalLoadVectorWidth (GLVWA/GRVW)", "value": [schedule.get("global_load_vec_GLVWA"), schedule.get("global_read_vec_GRVW")], "primitive_row": "global_read_vectorization_and_coalescing"},
      {"param": "LDS buffering (1LDSB)", "value": schedule.get("lds_buffering_1LDSB"), "primitive_row": "lds_staging_layout"},
    ],
    "selected_kernel_metadata": {
      "symbol_prefix": "Cijk_Ailk_Bljk_HHS_BH_MT128x128x16_MI16x16x16x1_...PGR1_PLR1_...WGM8",
      "code_object": "/opt/rocm-7.2.4/lib/rocblas/library/Kernels.so-000-gfx1100.hsaco",
      "per_role_tflops": {
        "ffn_gate_up": {"median": ffn_gate_up.get("median_tflops"), "best": ffn_gate_up.get("best_tflops"),
                        "streamk": ffn_gate_up.get("streamk"), "kernarg_size": ffn_gate_up.get("kernarg_size")},
        "ffn_down": {"median": ffn_down.get("median_tflops"), "best": ffn_down.get("best_tflops"),
                     "streamk": ffn_down.get("streamk"), "kernarg_size": ffn_down.get("kernarg_size")},
        "attn_q_o": {"median": attn_q_o.get("median_tflops"), "best": attn_q_o.get("best_tflops"),
                     "streamk": attn_q_o.get("streamk"), "kernarg_size": attn_q_o.get("kernarg_size")},
      },
      "no_workspace": True,
      "no_layout_copies": True,
      "weighted_model_full_pp_speedup": shape.get("full_pp_speedup"),
    },
    "instruction_mix_whole_object": {"v_wmma": mix.get("v_wmma"), "ds_load_b128": mix.get("ds_load_b128"),
                                     "ds_store_b128": mix.get("ds_store_b128"), "s_barrier": mix.get("s_barrier")},
    "timing_authority": {
      "rule": "Per-primitive throughput attribution and absolute TFLOPS are NOT validated across methods; "
              "the robust prefill number is the e2e 1.84x byte-identical (47%->87% llama).",
      "same_harness_required": perf.get("prior_43_validates_current_p8_candidates", None) is False,
    },
    "corrections": corrections,
    "claim_discipline": "Every future experiment names its matrix row + local artifact + pass/fail criterion (carried from PTM-0).",
  }

  # --- Track 2: Prefill Transfer ---
  track_prefill = {
    "track": "prefill_transfer",
    "goal": "Decide native tinygrad reproduction vs external Tensile artifact for prefill dense fp16 GEMM.",
    "ptm1_same_harness_authority_bridge": {
      "why": "Resolve whether the 43.026 vs current P8 (LDS 18.4 / no-LDS 17.9) gap is real kernel quality "
             "or harness/identity mismatch before any new kernel work.",
      "numbers": {"prior_captured_authority_best_tflops": perf.get("prior_captured_authority_best_tflops"),
                  "p8_lds_best_tflops": perf.get("p8_lds_best_tflops"),
                  "p8_global_direct_best_tflops": perf.get("p8_global_direct_best_tflops"),
                  "tensile_best_tflops": perf.get("tensile_best_tflops")},
      "minimum_pass": "Captured authority kernel AND current P8 candidates timed under ONE synchronized/"
                      "device-timestamp harness; no mixed-kernel/mixed-harness comparison.",
    },
    "forced_native_candidate_choice": {
      "decided_in": "PTM-2",
      "options": ["software_pipelined_k_loop", "spill_free_accumulator_resource_policy", "timing_launch_correction"],
      "rule": "Choose exactly one. codegen_oracle names the missing capability as the software-pipelined K-loop "
              "with double-buffered global->LDS->reg prefetch.",
    },
    "standalone_lds": {
      "status": "CLOSED",
      "rule": "No standalone LDS work. LDS is meaningful only paired with K-loop overlap + waits + resource scheduling.",
      "evidence": "A3 P2/P3 refuted naive LDS (~6 vs ~32 TFLOPS, net-negative on IC-served global reads).",
    },
    "gates": {
      "correctness": "RMSE vs authority within tolerance (prior selected-compatible path RMSE ~0.000209).",
      "resource": "scratch/private 0 and acceptable VGPR/occupancy before any timing promotion.",
      "performance": "Same-harness TFLOPS vs the PTM-1-bridged authority baseline only.",
    },
    "artifact_dependency_fallback_policy": {
      "rule": "Native codegen transfer and external rocBLAS .co artifact route are SEPARATE projects.",
      "external_co_route": "policy-gated (vendoring vs no-deps); default PREFILL_TENSILE_GEMM=0; ~87% llama, byte-identical.",
      "fallback": "Dependency-free rests at WMMA ~47% llama + shipped concrete-KV 1.24x.",
    },
  }

  # --- Track 3: Decode Applicability ---
  track_decode = {
    "track": "decode_applicability",
    "goal": "Determine whether any Tensile primitive affects q8 decode; default assumption is no direct transfer.",
    "domain_mismatch": "Decode is q8/MMVQ, batch-1, quantized, lifecycle/consumer-bound; NOT dense fp16 GEMM.",
    "current_readiness": {
      "verdict": readiness.get("verdict"),
      "n2_candidate_count": start_gate.get("n2_candidate_count"),
      "max_timing_grade_movement_us": start_gate.get("max_timing_grade_movement_us"),
      "oracle_gap_us": oracle_gap,
      "required_for_n2": start_gate.get("required_for_n2"),
    },
    "required_gates_before_any_transfer_claim": [
      "q8 role-joined gate/up evidence",
      "same-binary primitive ablation",
      ">=30us timing-grade movement (current max attributed 14.087us < gate)",
      "W==D quality unchanged",
      "packed q8 format preserved",
      "no dense GEMM substitution from prefill-only evidence",
    ],
  }

  tracks = [track_explanation, track_prefill, track_decode]

  # --- Phases PTM-1..PTM-5 ---
  phases = [
    {"phase": "PTM-1", "name": "same_harness_authority_bridge",
     "goal": "Stop mixed-harness reasoning before any new kernel work.",
     "minimum_pass": "Captured 43.026 TFLOPS authority kernel and current P8 candidates timed under one common harness.",
     "status": "next"},
    {"phase": "PTM-2", "name": "prefill_primitive_decision",
     "goal": "Choose exactly one prefill-native transfer candidate row.",
     "minimum_pass": "One of software_pipelined_k_loop / spill_free_accumulator_resource_policy / timing_launch_correction; standalone LDS disallowed.",
     "status": "blocked_on_PTM_1"},
    {"phase": "PTM-3", "name": "native_candidate_scope",
     "goal": "Detailed scope for the single chosen native row only.",
     "minimum_pass": "Names dataflow, resource policy, correctness/resource/performance gates for the chosen row; no other variant.",
     "status": "blocked_on_PTM_2"},
    {"phase": "PTM-4", "name": "external_artifact_policy_scope",
     "goal": "Decide vendored .co dependency vs no-deps, kept separate from native.",
     "minimum_pass": "Policy decision names fallback, default behavior (PREFILL_TENSILE_GEMM=0), source/ELF provenance, model quality gates.",
     "status": "parallel_policy"},
    {"phase": "PTM-5", "name": "decode_transfer_gate",
     "goal": "Gate whether any Tensile primitive transfers to q8 decode.",
     "minimum_pass": "A q8 row clears >=30us same-binary movement, W==D quality, role-joined gate/up evidence, packed q8 preserved.",
     "status": "blocked_until_q8_row_clears_30us"},
  ]

  stop_rules = [
    "Do not run another P8 kernel variant unless it names a matrix row.",
    "No standalone LDS work; LDS only with overlapped movement + resource control.",
    "No mixed-kernel / mixed-harness TFLOPS comparison.",
    "Do not treat the external .co artifact route and native tinygrad codegen transfer as one project.",
    "Do not reopen q8 transfer yet.",
    "Every future experiment must name the primitive row it is proving.",
  ]

  gate = {
    "matrix_scope_present": matrix.get("verdict") == "PASS_TENSILE_PRIMITIVE_TRANSFER_MATRIX_SCOPED",
    "shape_matrix_present": bool(shape),
    "codegen_oracle_present": bool(codegen),
    "timing_authority_reconciliation_present": timing_auth.get("verdict") == "PASS_BB5A10_P8_TIMING_AUTHORITY_RECONCILED_SAME_HARNESS_REQUIRED",
    "decode_readiness_loaded": bool(readiness),
    "three_tracks_populated": len(tracks) == 3 and all(t.get("goal") for t in tracks),
    "five_phases_defined": [p["phase"] for p in phases] == ["PTM-1", "PTM-2", "PTM-3", "PTM-4", "PTM-5"],
    "stop_rules_present": len(stop_rules) >= 5,
    "standalone_lds_closed": track_prefill["standalone_lds"]["status"] == "CLOSED",
    "corrections_recorded": len(corrections) == 2,
  }
  gate_pass = all(gate.values())
  result = {
    "date": "2026-06-20",
    "schema": "qk_tensile_roadmap_scope_v1",
    "phase": "PTM-ROADMAP",
    "verdict": "PASS_TENSILE_ROADMAP_SCOPED" if gate_pass else "BLOCKED_TENSILE_ROADMAP_SCOPE",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "online_sources": [
      {"name": "Tensile kernel parameters",
       "url": "https://rocm.docs.amd.com/projects/Tensile/en/docs-7.1.1/src/conceptual/kernel-parameters.html",
       "used_for": "parameter-to-primitive rows: WorkGroup, ThreadTile, MatrixInstruction, PGR, PLR, WGM, DepthU, VectorWidth"},
      {"name": "Tensile benchmark protocol",
       "url": "https://rocm.docs.amd.com/projects/Tensile/en/docs-7.1.1/src/conceptual/benchmarking.html",
       "used_for": "solution-selection and benchmark-phasing rows"},
      {"name": "Tensile nomenclature",
       "url": "https://rocm.docs.amd.com/projects/Tensile/en/docs-7.0.2/src/reference/nomenclature.html",
       "used_for": "GEMM/tensor-contraction problem-form row"},
      {"name": "ROCm/Tensile repository note",
       "url": "https://github.com/ROCm/Tensile",
       "used_for": "repository status and relationship to rocBLAS/ROCm libraries"},
    ],
    "local_sources": [
      "bench/qk-tensile-primitive-transfer/scope.json",
      "bench/qk-tensile-extraction/shape_matrix.json",
      "bench/qk-tensile-extraction/codegen_oracle.json",
      "bench/amd-broad-backend-roadmap/bb5a10_p8_timing_authority_reconciliation_result.json",
      "bench/qk-decode-native-tooling/readiness.json",
      "docs/prefill-tensile-DEFINITIVE-source-of-truth-20260619.md",
      "docs/amd-broad-backend-bb5a10-tensile-layout-audit-20260619.md",
    ],
    "tracks": tracks,
    "phases": phases,
    "stop_rules": stop_rules,
    "gate": gate,
    "next_action": "Run PTM-1 same-harness authority bridge: time the captured 43.026 TFLOPS authority kernel and "
                   "current P8 candidates under one synchronized/device-timestamp harness before choosing any "
                   "prefill-native transfer row.",
  }
  write_json("roadmap.json", result)
  print(json.dumps({
    "out": "bench/qk-tensile-primitive-transfer/roadmap.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "tracks": len(tracks),
    "phases": len(phases),
    "next": result["next_action"],
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
