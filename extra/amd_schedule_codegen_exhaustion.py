#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]

FEATURES = [
  "special_instruction_selection",
  "vector_global_load_shape",
  "waitcnt_placement",
  "s_clause_delay_alu",
  "register_allocation_live_ranges",
  "occupancy_vgpr_sgpr_policy",
  "software_pipelining",
  "lds_staging_layout",
  "reduction_topology",
  "launch_kernarg_contract",
  "graph_rebind_boundary",
  "attribution_tooling",
]

def load(rel:str) -> Any:
  path = ROOT / rel
  if not path.exists(): raise FileNotFoundError(rel)
  with path.open() as f: return json.load(f)

def feature(name:str, q8:str, prefill:str, cls:str, evidence:list[str], next_action:str) -> dict[str, Any]:
  return {
    "feature": name,
    "q8_decode": q8,
    "prefill_tensile": prefill,
    "classification": cls,
    "evidence": evidence,
    "next_action": next_action,
  }

def main() -> None:
  ap = argparse.ArgumentParser(description="Read-only AMD schedule/codegen exhaustion matrix generator.")
  ap.add_argument("--out", type=pathlib.Path, default=ROOT / "bench/amd-schedule-codegen-exhaustion/oracle_matrix.json")
  args = ap.parse_args()

  q8_route = load("bench/q8-ffn-amd-scheduler-project/route_a_result.json")
  q8_cap = load("bench/q8-ffn-amd-scheduler-project/dsl_capability_map.json")
  q8_contract = load("bench/q8-ffn-amd-scheduler-project/oracle_contract.json")
  q8_pmu = load("bench/q8-ffn-amd-scheduler-project/pmu_sqtt_evidence.json")
  q8_artifact = load("bench/q8-ffn-amd-scheduler-project/result.json")

  tensile_shape = load("bench/qk-tensile-extraction/shape_matrix.json")
  tensile_codegen = load("bench/qk-tensile-extraction/codegen_oracle.json")
  tensile_inmodel = load("bench/qk-tensile-extraction/inmodel_measurement.json")
  tensile_block = load("bench/qk-tensile-extraction/block_transfer.json")

  q8_features = {f["feature"]: f for f in q8_cap["features"]}
  tensile_roles = {r["role"]: r for r in tensile_shape["rows"]}

  rows = [
    {
      "primitive": "q8_decode_mmvq_lifecycle",
      "phase": "decode",
      "oracle": "hipcc/LLD q8 gate/up artifact through HCQ",
      "oracle_docs": [
        "q8-ffn-artifact-import-route-result-20260619.md",
        "q8-ffn-handwritten-a4-decode-result-20260619.md",
        "q8-ffn-route-a-scheduler-codegen-result-20260619.md",
        "q8-ffn-route-a-pmu-sqtt-evidence-result-20260619.md",
      ],
      "current_tinygrad": "AMD DSL/ASM q8_b2b_fullrow_reduce",
      "oracle_result": {
        "artifact_verdict": q8_artifact["verdict"],
        "artifact_lifecycle_us": q8_artifact["summary"]["lifecycle_us"],
        "tinygrad_asm_gateup_us": q8_contract["known_timings_us"]["tinygrad_asm_gateup_full"],
        "comgr_fused_gateup_us": q8_contract["known_timings_us"]["comgr_fused_gateup"],
        "route_a_verdict": q8_route["a1_verdict"],
        "pmu_sqtt_verdict": q8_pmu["classification"]["verdict"],
        "pmu_profile_runnable": q8_pmu["classification"]["pmc_profile_runnable"],
        "sqtt_profile_runnable": q8_pmu["classification"]["sqtt_profile_runnable"],
        "sqtt_decode_usable": q8_pmu["classification"]["sqtt_decode_usable"],
      },
      "quality_gate": "dNLL <= 0.01, W==D decode >=3%",
      "state": "artifact_only_research_pass__native_project_level",
    },
    {
      "primitive": "prefill_tensile_fp16_gemm",
      "phase": "prefill",
      "oracle": "rocBLAS/Tensile Cijk kernels extracted and launched through HCQ",
      "oracle_docs": [
        "prefill-tensile-tpe5-shape-matrix-result-20260619.md",
        "prefill-tensile-tpe6-block-transfer-result-20260619.md",
        "prefill-tensile-research-measurement-result-20260619.md",
        "prefill-own-wmma-kernel-result-20260619.md",
      ],
      "current_tinygrad": "PREFILL_V2 fp16 WMMA matmul",
      "oracle_result": {
        "shape_matrix_verdict": tensile_shape["verdict"],
        "ffn_gate_up_tflops": tensile_roles["ffn_gate_up"]["median_tflops"],
        "ffn_down_tflops": tensile_roles["ffn_down"]["median_tflops"],
        "attn_q_o_tflops": tensile_roles["attn_q_o"]["median_tflops"],
        "tinygrad_tflops": tensile_roles["ffn_gate_up"]["tinygrad_tflops"],
        "weighted_shape_model_speedup": tensile_shape["full_pp_speedup"],
        "inmodel_ffn_speedup": tensile_inmodel["warm_pp512_speedup"],
        "inmodel_attn_qo_speedup": tensile_inmodel["A5_attn_qo_routed"]["warm_pp512_speedup"],
        "block_transfer_verdict": tensile_block["verdict"],
        "codegen_oracle_verdict": tensile_codegen["verdict"],
      },
      "quality_gate": "warm pp512/pp1024 and dNLL <= 0.01",
      "state": "artifact_only_policy_gated__native_project_level",
    },
  ]

  features = [
    feature("special_instruction_selection",
      "dot4 already emitted by tinygrad/COMGR/oracle",
      "WMMA fragment matches Tensile (16x16x16x1)",
      "expressible_now",
      ["q8 native_dot4", "Tensile codegen_oracle macro-tile/WMMA identical"],
      "do not build as standalone"),
    feature("vector_global_load_shape",
      "oracle 11 global loads with b128; tinygrad 22 loads with b32/u8/u16",
      "Tensile GLVWA/GRVW vectorized global reads",
      "project_level",
      [q8_features["vector_or_coalesced_global_loads"]["evidence"][1], "Tensile GLVWA4/GRVW4 for ffn_gate/up"],
      "own only as part of a scheduler/register layout, not one-off q8 A2"),
    feature("waitcnt_placement",
      "grouped waits moved only ~0.84us in q8 DSO",
      "Tensile has dense wait scheduling around WMMA/LDS loop",
      "project_level",
      [q8_features["waitcnt_grouping"]["evidence"][1], "codegen_oracle reports software-pipelined K-loop as missing"],
      "requires latency-aware scheduler before build"),
    feature("s_clause_delay_alu",
      "oracle has s_clause/s_delay_alu; tinygrad ASM has none",
      "likely part of mature schedule but not separately isolated",
      "project_level",
      q8_features["schedule_annotations_s_clause_delay_alu"]["evidence"],
      "do not emit manually without semantic insertion rules"),
    feature("register_allocation_live_ranges",
      "suspected q8 gap, no bounded attribution",
      "Tensile keeps TT4_64/vgpr256 no-spill; tinygrad larger-acc configs spill/regress",
      "project_level",
      ["q8 PMU/SQTT decode unusable for attribution", "codegen_oracle accumulator register allocation delta"],
      "renderer/register allocator project"),
    feature("occupancy_vgpr_sgpr_policy",
      "q8 evidence insufficient to isolate",
      "POWN-1 more waves/bigger tiles/noLDS all regress",
      "project_level",
      ["q8 body-insensitive ladder", "POWN-1 best remains 42 TFLOPS"],
      "close bounded knob search; keep as compiler policy"),
    feature("software_pipelining",
      "not named as bounded q8 feature",
      "Tensile PGR1/PLR1 K-loop overlap is the named 42->67 TFLOPS delta",
      "project_level",
      ["codegen_oracle smallest_codegen_change"],
      "requires AMD renderer scheduler work; no bounded proof yet"),
    feature("lds_staging_layout",
      "producer needs staged reductions; current UOp expression killed",
      "Tensile uses LDS + wide local reads; noLDS tinygrad regressed",
      "project_level",
      ["Q8L-2 store-group expression killed", "Tensile LRVW16/ds_load_b128"],
      "own only with staged-kernel/scheduler capability"),
    feature("reduction_topology",
      "q8 reduction rewrite moved ~13us, below A2 gate",
      "not the prefill GEMM bottleneck",
      "not_worth_owning",
      q8_features["reduction_rewrite"]["evidence"],
      "do not reopen standalone"),
    feature("launch_kernarg_contract",
      "q8 artifact loader/graph route works",
      "Tensile named descriptor/raw kernarg works and generalizes",
      "artifact_only",
      ["q8 artifact route PASS_RESEARCH", "TPE-5 one code object + pointer convention"],
      "policy decision for artifacts; runtime helper only if route is accepted"),
    feature("graph_rebind_boundary",
      "q8 graph-safe",
      "prefill in-model route passes pp512, default off/policy-gated",
      "bounded_extension",
      ["q8 artifact graph route", "Tensile A5 attn_q/o routed pp512 1.761x"],
      "finish policy/fallback/shape coverage if accepting artifacts"),
    feature("attribution_tooling",
      "PMU/SQTT capture works; SQTT decode unusable",
      "HCQ attribution Level 3 works; Level 4 absent",
      "tooling_blocked",
      [q8_pmu["classification"]["blockers"][0] if q8_pmu["classification"]["blockers"] else "SQTT decode usable=false",
       "primitive-hcq-attribution-result: Level 3, not Level 4"],
      "build SQTT decode/counter attribution before making stall-level claims"),
  ]

  missing = [f for f in FEATURES if f not in {x["feature"] for x in features}]
  if missing: raise RuntimeError(f"missing feature rows: {missing}")

  classifications = {}
  for f in features: classifications[f["classification"]] = classifications.get(f["classification"], 0) + 1

  matrix = {
    "date": "2026-06-19",
    "schema": "amd_schedule_codegen_exhaustion_v1",
    "scope_doc": "docs/amd-schedule-codegen-exhaustion-scope-20260619.md",
    "rows": rows,
    "features": features,
    "classification_counts": classifications,
    "sce0_gate": {
      "rows_populated": len(rows) == 2,
      "all_features_classified": len(missing) == 0,
      "required_evidence_present": True,
    },
    "sce1_verdict": "PASS_MATRIX_BUILT",
    "native_codegen_decision": (
      "No bounded native schedule/codegen feature is identified. q8 decode native generation remains project-level; "
      "prefill native generation remains project-level around software-pipelined K-loop + spill-free accumulator allocation. "
      "Near-term measured path is artifact/policy/graph routing, not renderer rewrite."
    ),
    "next": "Only start native compiler work if a feature is promoted from project_level/tooling_blocked to bounded_extension with a movement gate.",
  }

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(matrix, indent=2) + "\n")
  print(json.dumps({
    "out": str(args.out),
    "sce1_verdict": matrix["sce1_verdict"],
    "classification_counts": classifications,
    "native_codegen_decision": matrix["native_codegen_decision"],
  }, indent=2))

if __name__ == "__main__":
  main()
