#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"
from extra.qk_probe_harness import probe_io
read_json, write_json = probe_io(OUT)




def by_role(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
  return {str(row.get("role")): row for row in rows}


def verdict_row(phase: str, verdict: str, reason: str, next_action: str, gate_pass: bool = False) -> dict[str, Any]:
  return {
    "date": "2026-06-19",
    "phase": phase,
    "verdict": verdict,
    "gate_pass": gate_pass,
    "reason": reason,
    "next_action": next_action,
  }


def build_authority() -> dict[str, Any]:
  q8_contract = read_json("bench/q8-ffn-amd-scheduler-project/oracle_contract.json", {})
  q8_artifact = read_json("bench/q8-ffn-amd-scheduler-project/result.json", {})
  q8_route = read_json("bench/q8-ffn-amd-scheduler-project/route_a_result.json", {})
  readiness = read_json("bench/qk-decode-native-tooling/readiness.json", {})
  tensile_shape = read_json("bench/qk-tensile-extraction/shape_matrix.json", {})
  tensile_codegen = read_json("bench/qk-tensile-extraction/codegen_oracle.json", {})
  clock = read_json("bench/qk-prefill-clock-dpm-authority/prefill_clock_matrix.json", {})
  exhaustion = read_json("bench/amd-schedule-codegen-exhaustion/oracle_matrix.json", {})

  roles = by_role(tensile_shape.get("rows", []))
  timings = q8_contract.get("known_timings_us", {})
  start_gate = readiness.get("start_gate", {})
  classes = exhaustion.get("classification_counts", {})
  max_movement = start_gate.get("max_timing_grade_movement_us")
  n2_count = start_gate.get("n2_candidate_count")
  return {
    "date": "2026-06-19",
    "schema": "amd_broad_backend_authority_v1",
    "scope_doc": "docs/amd-broad-backend-roadmap-scope-20260619.md",
    "acceptance": {
      "verdict": "BROAD_BACKEND_ACCEPTED",
      "source": "user accepted roadmap execution after ROADMAP_ONLY decode tooling pass",
      "boundary": "backend/compiler project only; not a q8-specific N2 patch",
    },
    "decode_q8": {
      "native_tinygrad_us": timings.get("tinygrad_asm_gateup_full"),
      "hipcc_lld_oracle_us": timings.get("hipcc_lld_gateup_current_loader"),
      "gap_us": round(timings.get("tinygrad_asm_gateup_full", 0) - timings.get("hipcc_lld_gateup_current_loader", 0), 3)
                if timings else None,
      "artifact_lifecycle_us": (q8_artifact.get("summary") or {}).get("lifecycle_us"),
      "artifact_research_verdict": q8_artifact.get("verdict"),
      "native_route_verdict": q8_route.get("a1_verdict"),
      "n2_candidate_count": n2_count,
      "max_timing_grade_movement_us": max_movement,
      "n2_gate_us": 30.0,
      "decision": "do_not_start_q8_specific_native_scheduler_patch",
    },
    "prefill_wmma": {
      "ffn_gate_up_tinygrad_tflops": roles.get("ffn_gate_up", {}).get("tinygrad_tflops"),
      "ffn_gate_up_tensile_tflops": roles.get("ffn_gate_up", {}).get("median_tflops"),
      "ffn_down_tensile_tflops": roles.get("ffn_down", {}).get("median_tflops"),
      "codegen_oracle_verdict": tensile_codegen.get("verdict"),
      "clock_authority_verdict": clock.get("verdict"),
      "native_gap_class": "software_pipelined_k_loop_plus_spill_free_register_allocation",
    },
    "cross_primitive": {
      "classification_counts": classes,
      "native_codegen_decision": exhaustion.get("native_codegen_decision"),
      "broad_backend_requirement": "must serve q8 decode and prefill WMMA/Tensile authority cases, or document why one transfer fails",
    },
    "gate_pass": bool(q8_contract and tensile_shape and tensile_codegen and readiness),
  }


def build_oracle_suite(authority: dict[str, Any]) -> dict[str, Any]:
  q8_contract = read_json("bench/q8-ffn-amd-scheduler-project/oracle_contract.json", {})
  q8_artifact = read_json("bench/q8-ffn-amd-scheduler-project/result.json", {})
  q8_policy = read_json("bench/q8-ffn-amd-scheduler-project/artifact_policy_boundary.json", {})
  q8_graph = read_json("bench/q8-ffn-amd-scheduler-project/artifact_graph_route.json", {})
  n1 = read_json("bench/q8-ffn-amd-scheduler-project/n1_attribution.json", {})
  tensile_shape = read_json("bench/qk-tensile-extraction/shape_matrix.json", {})
  tensile_codegen = read_json("bench/qk-tensile-extraction/codegen_oracle.json", {})
  tensile_runtime = read_json("bench/qk-tensile-extraction/runtime.json", {})
  hcq = read_json("bench/qk-hcq-attribution/result.json", {})
  pmu = read_json("bench/qk-pmu-observability/result.json", {})

  roles = by_role(tensile_shape.get("rows", []))
  timings = q8_contract.get("known_timings_us", {})
  rows = [
    {
      "name": "q8_decode_gate_up_consumer",
      "phase": "decode",
      "program_name": "q8_mmvq_gateup / q8_b2b_fullrow_reduce",
      "code_hash": (q8_artifact.get("summary") or {}).get("gateup_hash"),
      "shape": {"m": 1, "n": 12288, "k": 4096, "roles": ["ffn_gate", "ffn_up"]},
      "launch_global": (q8_contract.get("launch_contract") or {}).get("global_size"),
      "launch_local": (q8_contract.get("launch_contract") or {}).get("local_size"),
      "correctness": {"proxy": "PASS in q8 artifact route", "graph_route": q8_graph.get("verdict")},
      "timing": {
        "native_tinygrad_us": timings.get("tinygrad_asm_gateup_full"),
        "hipcc_lld_oracle_us": timings.get("hipcc_lld_gateup_current_loader"),
        "artifact_lifecycle_us": (q8_artifact.get("summary") or {}).get("lifecycle_us"),
        "n2_gate_us": 30.0,
      },
      "disassembly_path": None,
      "resource_metadata": q8_contract.get("resource_contract"),
      "authority_level": "timing_proxy_plus_inmodel_research_quality",
      "quality_gate": "W==D >=3% and dNLL <=0.01 if promoted; current artifact remains research/default-off",
      "fallback_policy": q8_policy.get("fallback_policy", "default path unchanged"),
      "source_artifacts": [
        "bench/q8-ffn-amd-scheduler-project/oracle_contract.json",
        "bench/q8-ffn-amd-scheduler-project/result.json",
        "bench/q8-ffn-amd-scheduler-project/artifact_graph_route.json",
        "bench/q8-ffn-amd-scheduler-project/n1_attribution.json",
      ],
      "n1_verdict": n1.get("verdict"),
    },
    {
      "name": "prefill_tensile_ffn_gate_up",
      "phase": "prefill",
      "program_name": roles.get("ffn_gate_up", {}).get("kernel_symbol"),
      "code_hash": None,
      "shape": {k: roles.get("ffn_gate_up", {}).get(k) for k in ("m", "n", "k")},
      "launch_global": roles.get("ffn_gate_up", {}).get("global_size"),
      "launch_local": roles.get("ffn_gate_up", {}).get("local_size"),
      "correctness": {"correct": roles.get("ffn_gate_up", {}).get("correct"), "rel_err": roles.get("ffn_gate_up", {}).get("rel_err")},
      "timing": {
        "tinygrad_tflops": roles.get("ffn_gate_up", {}).get("tinygrad_tflops"),
        "oracle_tflops": roles.get("ffn_gate_up", {}).get("median_tflops"),
        "speedup_vs_tinygrad": roles.get("ffn_gate_up", {}).get("speedup_vs_tinygrad"),
        "backend_gate_tflops": 60.0,
      },
      "disassembly_path": None,
      "resource_metadata": {"kernarg_size": roles.get("ffn_gate_up", {}).get("kernarg_size"), "workspace": roles.get("ffn_gate_up", {}).get("workspace")},
      "authority_level": "isolated_timing_oracle_clock_controlled_context",
      "quality_gate": "fp16 rel_err <=1e-3 and pp model gate if routed",
      "fallback_policy": "external Tensile remains policy/research evidence; pure tinygrad default unchanged",
      "source_artifacts": [
        "bench/qk-tensile-extraction/shape_matrix.json",
        "bench/qk-tensile-extraction/codegen_oracle.json",
        "bench/qk-prefill-clock-dpm-authority/prefill_clock_matrix.json",
      ],
    },
    {
      "name": "prefill_tensile_ffn_down",
      "phase": "prefill",
      "program_name": roles.get("ffn_down", {}).get("kernel_symbol"),
      "code_hash": None,
      "shape": {k: roles.get("ffn_down", {}).get(k) for k in ("m", "n", "k")},
      "launch_global": roles.get("ffn_down", {}).get("global_size"),
      "launch_local": roles.get("ffn_down", {}).get("local_size"),
      "correctness": {"correct": roles.get("ffn_down", {}).get("correct"), "rel_err": roles.get("ffn_down", {}).get("rel_err")},
      "timing": {
        "tinygrad_tflops": roles.get("ffn_down", {}).get("tinygrad_tflops"),
        "oracle_tflops": roles.get("ffn_down", {}).get("median_tflops"),
        "speedup_vs_tinygrad": roles.get("ffn_down", {}).get("speedup_vs_tinygrad"),
        "backend_gate_tflops": 60.0,
      },
      "disassembly_path": None,
      "resource_metadata": {
        "kernarg_size": roles.get("ffn_down", {}).get("kernarg_size"),
        "workspace": roles.get("ffn_down", {}).get("workspace"),
        "streamk": roles.get("ffn_down", {}).get("streamk"),
      },
      "authority_level": "isolated_timing_oracle_clock_controlled_context",
      "quality_gate": "fp16 rel_err <=1e-3 and pp model gate if routed",
      "fallback_policy": "external Tensile remains policy/research evidence; pure tinygrad default unchanged",
      "source_artifacts": [
        "bench/qk-tensile-extraction/shape_matrix.json",
        "bench/qk-tensile-extraction/codegen_oracle.json",
        "bench/qk-prefill-clock-dpm-authority/prefill_clock_matrix.json",
      ],
    },
    {
      "name": "small_hcq_profiler_smoke",
      "phase": "tooling",
      "program_name": "HCQ attribution smoke / PMU observability smoke",
      "code_hash": None,
      "shape": None,
      "launch_global": None,
      "launch_local": None,
      "correctness": {"hcq_attribution_present": bool(hcq), "pmu_observability_present": bool(pmu)},
      "timing": None,
      "disassembly_path": None,
      "resource_metadata": None,
      "authority_level": "visibility_smoke",
      "quality_gate": "program metadata visible; no timing promotion from packet counts",
      "fallback_policy": "observability only",
      "source_artifacts": ["bench/qk-hcq-attribution/result.json", "bench/qk-pmu-observability/result.json"],
    },
  ]
  required = [
    row["correctness"] is not None and row["authority_level"] and row["source_artifacts"]
    for row in rows
  ]
  return {
    "date": "2026-06-19",
    "schema": "amd_broad_backend_oracle_suite_v1",
    "phase": "BB-1",
    "authority_pointer": "bench/amd-broad-backend-roadmap/authority.json",
    "rows": rows,
    "codegen_oracle_summary": tensile_codegen.get("smallest_codegen_change"),
    "gate": {
      "q8_decode_present": bool(timings),
      "prefill_gate_up_present": bool(roles.get("ffn_gate_up")),
      "prefill_down_present": bool(roles.get("ffn_down")),
      "smoke_present": bool(hcq or pmu or tensile_runtime),
      "all_rows_have_common_fields": all(required),
      "accepted_as_broad_backend": authority.get("acceptance", {}).get("verdict") == "BROAD_BACKEND_ACCEPTED",
    },
    "gate_pass": bool(timings and roles.get("ffn_gate_up") and roles.get("ffn_down") and all(required)),
  }


def main() -> int:
  authority = build_authority()
  oracle_suite = build_oracle_suite(authority)
  schedule_metadata = read_json("bench/amd-broad-backend-roadmap/schedule_metadata_ir_result.json")
  if not schedule_metadata:
    schedule_metadata = verdict_row(
      "BB-2",
      "PENDING_BUILD",
      "No schedule metadata IR exists yet; this is the first implementation phase after oracle acceptance.",
      "Design and implement metadata that survives lowering for one q8-shaped probe and one WMMA-shaped probe.",
    )
  bb2_pass = schedule_metadata.get("verdict") == "PASS_SCHEDULE_METADATA_IR" and bool(schedule_metadata.get("gate_pass"))
  wait_scheduler = read_json("bench/amd-broad-backend-roadmap/wait_scheduler_result.json")
  if not wait_scheduler:
    wait_scheduler = verdict_row(
      "BB-3",
      "READY" if bb2_pass else "BLOCKED_ON_BB_2",
      "Schedule metadata exists; semantic wait/s_clause/s_delay_alu placement can start." if bb2_pass else
      "Semantic wait/s_clause/s_delay_alu placement needs schedule metadata and dependency groups first.",
      "Build semantic wait/scheduler emitter over schedule metadata." if bb2_pass else
      "Start only after BB-2 proves metadata is carried into AMD lowering.",
    )
  bb3_pass = wait_scheduler.get("verdict") == "PASS_SEMANTIC_WAIT_SCHEDULER_PROBE" and bool(wait_scheduler.get("gate_pass"))
  register_resource = read_json("bench/amd-broad-backend-roadmap/register_resource_result.json")
  if not register_resource:
    register_resource = verdict_row(
      "BB-4",
      "READY" if bb2_pass else "BLOCKED_ON_BB_2",
      "Schedule metadata exists; register/resource accounting can start." if bb2_pass else
      "Register/live-range controls need metadata and resource accounting before meaningful probes.",
      "Build register/live-range/resource controls and prove controlled VGPR/occupancy movement." if bb2_pass else
      "Start after BB-2, then prove controlled VGPR/occupancy movement without regressions.",
    )
  bb4_pass = register_resource.get("verdict") == "PASS_RESOURCE_ACCOUNTING_PROBE" and bool(register_resource.get("gate_pass"))
  software_pipeline = read_json("bench/amd-broad-backend-roadmap/software_pipeline_result.json")
  if not software_pipeline:
    software_pipeline = verdict_row(
      "BB-5",
      "READY" if bb3_pass and bb4_pass else "BLOCKED_ON_BB_3_BB_4",
      "Scheduler hints and resource accounting probes passed; software-pipeline probe can start." if bb3_pass and bb4_pass else
      "Tensile-class prefill requires scheduler plus register/resource control.",
      "Build BB-5 software-pipelined prefill probe; gate is >=60 TFLOPS pure tinygrad." if bb3_pass and bb4_pass else
      "Build after wait scheduler and register/resource controls; gate is >=60 TFLOPS pure tinygrad.",
    )
  bb5_done = software_pipeline.get("verdict") in {"PASS_SOFTWARE_PIPELINE_TFLOPS", "BLOCKED_REAL_RENDERER_ALLOCATOR_INTEGRATION"} and bool(software_pipeline.get("gate_pass"))
  bb5a_renderer_allocator = read_json("bench/amd-broad-backend-roadmap/bb5a_renderer_allocator_scope.json")
  if not bb5a_renderer_allocator:
    bb5a_renderer_allocator = verdict_row(
      "BB-5a",
      "READY_TO_SCOPE" if software_pipeline.get("verdict") == "BLOCKED_REAL_RENDERER_ALLOCATOR_INTEGRATION" else "BLOCKED_ON_BB_5",
      "BB-5 formally blocked on renderer/allocator integration; BB-5a must scope the missing implementation layer." if software_pipeline.get("verdict") == "BLOCKED_REAL_RENDERER_ALLOCATOR_INTEGRATION" else
      "BB-5a starts only after BB-5 identifies renderer/allocator integration as the blocker.",
      "Build BB-5a renderer/allocator scope and static readiness artifact." if software_pipeline.get("verdict") == "BLOCKED_REAL_RENDERER_ALLOCATOR_INTEGRATION" else
      "Run BB-5 first.",
    )
  bb5a_ready = bb5a_renderer_allocator.get("verdict") == "BB5A_READY_TO_REOPEN_BB5" and bool(bb5a_renderer_allocator.get("gate_pass"))
  bb5a1_pipeline_ir_scope = read_json("bench/amd-broad-backend-roadmap/bb5a1_pipeline_ir_scope.json")
  if not bb5a1_pipeline_ir_scope:
    bb5a1_pipeline_ir_scope = verdict_row(
      "BB-5a.1",
      "READY_TO_SCOPE" if bb5a_renderer_allocator.get("verdict") == "BB5A_SCOPE_COMPLETE_IMPLEMENTATION_NOT_READY" else "BLOCKED_ON_BB_5A",
      "BB-5a points next at the pipeline IR surface; scope the durable stage metadata contract first." if bb5a_renderer_allocator.get("verdict") == "BB5A_SCOPE_COMPLETE_IMPLEMENTATION_NOT_READY" else
      "Pipeline IR scope starts only after BB-5a identifies renderer/allocator implementation as the blocker.",
      "Build BB-5a.1 pipeline IR scope and static readiness artifact." if bb5a_renderer_allocator.get("verdict") == "BB5A_SCOPE_COMPLETE_IMPLEMENTATION_NOT_READY" else
      "Run BB-5a scope first.",
    )
  bb5a1_scoped = bb5a1_pipeline_ir_scope.get("verdict") == "BB5A1_SCOPE_COMPLETE_IMPLEMENTATION_NOT_READY" and bool(bb5a1_pipeline_ir_scope.get("gate_pass"))
  bb5a1_pipeline_ir = read_json("bench/amd-broad-backend-roadmap/bb5a1_pipeline_ir_result.json")
  if not bb5a1_pipeline_ir:
    bb5a1_pipeline_ir = verdict_row(
      "BB-5a.1",
      "READY" if bb5a1_scoped else "BLOCKED_ON_BB_5A_1_SCOPE",
      "Pipeline IR scope exists; implement the stage schema and read-only extraction probe." if bb5a1_scoped else
      "Pipeline IR result needs BB-5a.1 scope first.",
      "Add AMDPipelineStageMeta, read-only extraction, and a two-stage WMMA prefill-shaped probe." if bb5a1_scoped else
      "Run BB-5a.1 scope first.",
    )
  bb5a1_pass = bb5a1_pipeline_ir.get("verdict") == "PASS_PIPELINE_IR_SURFACE" and bool(bb5a1_pipeline_ir.get("gate_pass"))
  bb5a_full_plan = read_json("bench/amd-broad-backend-roadmap/bb5a_full_plan.json")
  if not bb5a_full_plan:
    bb5a_full_plan = verdict_row(
      "BB-5a-full-plan",
      "READY" if bb5a1_pass else "BLOCKED_ON_BB_5A_1",
      "BB-5a.1 passes; consolidate BB-5a.2 through BB-5a.7 into one execution plan." if bb5a1_pass else
      "Full BB-5a plan needs the pipeline IR surface first.",
      "Generate bb5a_full_plan.json and keep BB-6 blocked until BB-5a.7." if bb5a1_pass else
      "Run BB-5a.1 first.",
    )
  bb5a_full_plan_ready = bb5a_full_plan.get("verdict") == "BB5A_FULL_PLAN_READY_BB5A2_NEXT" and bool(bb5a_full_plan.get("gate_pass"))
  bb5a_execution = read_json("bench/amd-broad-backend-roadmap/bb5a_execution_result.json")
  if not bb5a_execution:
    bb5a_execution = verdict_row(
      "BB-5a-execution",
      "READY" if bb5a_full_plan_ready else "BLOCKED_ON_BB_5A_PLAN",
      "Full BB-5a plan is ready; execute phase gates starting at BB-5a.2." if bb5a_full_plan_ready else
      "BB-5a execution needs the full plan first.",
      "Run extra/qk_amd_bb5a_execute_plan.py." if bb5a_full_plan_ready else "Generate BB-5a full plan first.",
    )
  bb5a_execution_blocked = bb5a_execution.get("verdict") in {
    "BB5A_EXECUTION_BLOCKED_BB5A2_REAL_LDS_LOWERING", "BB5A_EXECUTION_BLOCKED_BB5A2_RENDER_ISA_EVIDENCE",
    "BB5A_EXECUTION_BLOCKED_BB5A2_REAL_LOWERING_INTEGRATION", "BB5A_EXECUTION_BLOCKED_BB5A3_WAIT_SCHEDULER",
    "BB5A_EXECUTION_BLOCKED_BB5A4_ALLOCATOR_RESOURCE", "BB5A_EXECUTION_BLOCKED_BB5A5_RESOURCE_POLICY",
    "BB5A_EXECUTION_BLOCKED_BB5A6_CORRECTNESS", "BB5A_EXECUTION_BLOCKED_BB5A7_PERFORMANCE_GATE",
  }
  bb5a2_solution = read_json("bench/amd-broad-backend-roadmap/bb5a2_solution_scope.json")
  if not bb5a2_solution:
    bb5a2_solution = verdict_row(
      "BB-5a.2-solution",
      "READY_TO_SCOPE" if bb5a_execution_blocked else "BLOCKED_ON_BB_5A_EXECUTION",
      "BB-5a execution blocked at real LDS lowering; scope the actual postrange/renderer solution." if bb5a_execution_blocked else
      "BB-5a.2 solution scope needs the execution blocker first.",
      "Build BB-5a.2 real LDS lowering solution scope." if bb5a_execution_blocked else "Run BB-5a execution first.",
    )
  bb5a2_solution_scoped = bb5a2_solution.get("verdict") == "BB5A2_SOLUTION_SCOPED_REAL_LOWERING_REQUIRED" and bool(bb5a2_solution.get("gate_pass"))
  bb5a2_lds_stage_plan = read_json("bench/amd-broad-backend-roadmap/bb5a2_lds_stage_plan_result.json")
  if not bb5a2_lds_stage_plan:
    bb5a2_lds_stage_plan = verdict_row(
      "BB-5a.2-layer-1",
      "READY" if bb5a2_solution_scoped else "BLOCKED_ON_BB_5A_2_SOLUTION",
      "BB-5a.2 solution is scoped; implement AMDLDSStagePlan and alias-safe slot planning." if bb5a2_solution_scoped else
      "Layer 1 needs the BB-5a.2 solution scope first.",
      "Add AMDLDSStagePlan plus LDS stage plan probe." if bb5a2_solution_scoped else "Run BB-5a.2 solution scope first.",
    )
  bb5a2_layer1_pass = bb5a2_lds_stage_plan.get("verdict") == "PASS_LDS_STAGE_PLAN" and bool(bb5a2_lds_stage_plan.get("gate_pass"))
  bb5a2_lowering_hook = read_json("bench/amd-broad-backend-roadmap/bb5a2_lowering_hook_result.json")
  if not bb5a2_lowering_hook:
    bb5a2_lowering_hook = verdict_row(
      "BB-5a.2-layer-2",
      "READY" if bb5a2_layer1_pass else "BLOCKED_ON_BB_5A_2_LAYER_1",
      "Layer 1 passes; lower planned LDS slots into durable DEFINE_LOCAL UOps." if bb5a2_layer1_pass else
      "Layer 2 needs the LDS stage plan first.",
      "Add gated DEFINE_LOCAL lowering helper and probe." if bb5a2_layer1_pass else "Run BB-5a.2 Layer 1 first.",
    )
  bb5a2_layer2_pass = bb5a2_lowering_hook.get("verdict") == "PASS_DEFINE_LOCAL_LOWERING_HOOK" and bool(bb5a2_lowering_hook.get("gate_pass"))
  bb5a2_render_isa = read_json("bench/amd-broad-backend-roadmap/bb5a2_render_isa_evidence_result.json")
  if not bb5a2_render_isa:
    bb5a2_render_isa = verdict_row(
      "BB-5a.2-layer-3",
      "READY" if bb5a2_layer2_pass else "BLOCKED_ON_BB_5A_2_LAYER_2",
      "Layer 2 passes; prove AMD render/ELF evidence for the lowered two-slot LDS structure." if bb5a2_layer2_pass else
      "Layer 3 needs the DEFINE_LOCAL lowering hook first.",
      "Add AMD render/ELF evidence probe." if bb5a2_layer2_pass else "Run BB-5a.2 Layer 2 first.",
    )
  bb5a2_layer3_pass = bb5a2_render_isa.get("verdict") == "PASS_RENDER_ELF_LDS_EVIDENCE" and bool(bb5a2_render_isa.get("gate_pass"))
  bb5a2_integration = read_json("bench/amd-broad-backend-roadmap/bb5a2_real_lowering_integration_result.json", {})
  bb5a2_dataflow = read_json("bench/amd-broad-backend-roadmap/bb5a2_pipelined_dataflow_result.json", {})
  bb5a3_integration = read_json("bench/amd-broad-backend-roadmap/bb5a3_wait_scheduler_integration_result.json", {})
  bb5a4_allocator = read_json("bench/amd-broad-backend-roadmap/bb5a4_allocator_resource_result.json", {})
  bb5a5_policy = read_json("bench/amd-broad-backend-roadmap/bb5a5_resource_policy_result.json", {})
  bb5a6_correctness = read_json("bench/amd-broad-backend-roadmap/bb5a6_correctness_result.json", {})
  bb5a7_performance = read_json("bench/amd-broad-backend-roadmap/bb5a7_performance_gate_result.json", {})
  bb5a8_mapping = read_json("bench/amd-broad-backend-roadmap/bb5a8_tensile_mapping_result.json", {})
  bb5a8_capture = read_json("bench/amd-broad-backend-roadmap/bb5a8_authority_kernel_capture_result.json", {})
  bb5a9_causal_delta = read_json("bench/amd-broad-backend-roadmap/bb5a9_causal_delta_package_result.json", {})
  bb5a10_layout_audit = read_json("bench/amd-broad-backend-roadmap/bb5a10_tensile_layout_audit_result.json", {})
  bb5a10_implementation_plan = read_json("bench/amd-broad-backend-roadmap/bb5a10_implementation_plan_result.json", {})
  bb5a10_p1_layout_spec = read_json("bench/amd-broad-backend-roadmap/bb5a10_p1_layout_spec_result.json", {})
  bb5a10_p2_rendered_lds = read_json("bench/amd-broad-backend-roadmap/bb5a10_p2_rendered_lds_result.json", {})
  bb5a10_p3_kloop_stage = read_json("bench/amd-broad-backend-roadmap/bb5a10_p3_kloop_stage_result.json", {})
  bb5a10_p4_wait_barrier = read_json("bench/amd-broad-backend-roadmap/bb5a10_p4_wait_barrier_result.json", {})
  bb5a10_p5_resource_policy = read_json("bench/amd-broad-backend-roadmap/bb5a10_p5_resource_policy_result.json", {})
  bb5a10_p6_structural = read_json("bench/amd-broad-backend-roadmap/bb5a10_p6_structural_candidate_result.json", {})
  bb5a10_p7_scope = read_json("bench/amd-broad-backend-roadmap/bb5a10_p7_correctness_scope_result.json", {})
  bb5a10_p7ab = read_json("bench/amd-broad-backend-roadmap/bb5a10_p7a_p7b_correctness_result.json", {})
  bb5a10_p7c = read_json("bench/amd-broad-backend-roadmap/bb5a10_p7c_numeric_correctness_result.json", {})
  bb5a10_p7d = read_json("bench/amd-broad-backend-roadmap/bb5a10_p7d_authority_correctness_result.json", {})
  bb5a10_p7e = read_json("bench/amd-broad-backend-roadmap/bb5a10_p7e_p8_handoff_result.json", {})
  bb5a10_p8 = read_json("bench/amd-broad-backend-roadmap/bb5a10_p8_performance_result.json", {})
  bb5a10_p8_tta_scope = read_json("bench/amd-broad-backend-roadmap/bb5a10_p8_tta_scope_result.json", {})
  bb5a10_p8_tta_completion = read_json("bench/amd-broad-backend-roadmap/bb5a10_p8_tta_completion_scope_result.json", {})
  bb5a10_p8_tta1 = read_json("bench/amd-broad-backend-roadmap/bb5a10_p8_tta1_full_grid_correctness_result.json", {})
  bb5a10_p8_tta2 = read_json("bench/amd-broad-backend-roadmap/bb5a10_p8_tta2_authority_sample_correctness_result.json", {})
  bb5a10_p8_tta3 = read_json("bench/amd-broad-backend-roadmap/bb5a10_p8_tta3_macro_candidate_result.json", {})
  bb5a10_p8_tta3a = read_json("bench/amd-broad-backend-roadmap/bb5a10_p8_tta3a_ds64_macro_conversion_result.json", {})
  bb5a10_p8_bottleneck = read_json("bench/amd-broad-backend-roadmap/bb5a10_p8_bottleneck_classification_result.json", {})
  bb5a10_p8_global_direct = read_json("bench/amd-broad-backend-roadmap/bb5a10_p8_global_direct_candidate_decision_result.json", {})
  bb5a10_p8_timing_authority = read_json("bench/amd-broad-backend-roadmap/bb5a10_p8_timing_authority_reconciliation_result.json", {})
  bb5a2_full_pass = read_json("bench/amd-broad-backend-roadmap/bb5a2_double_buffer_lds_result.json", {}).get("verdict") == "PASS_DOUBLE_BUFFERED_LDS_LOWERING"
  bb5a3_pass = bb5a3_integration.get("verdict") == "PASS_BB5A3_WAIT_SCHEDULER_INTEGRATION" and bool(bb5a3_integration.get("gate_pass"))
  bb5a4_pass = bb5a4_allocator.get("verdict") == "PASS_BB5A4_ALLOCATOR_RESOURCE_CONTROL" and bool(bb5a4_allocator.get("gate_pass"))
  bb5a5_pass = bb5a5_policy.get("verdict") == "PASS_BB5A5_RESOURCE_POLICY" and bool(bb5a5_policy.get("gate_pass"))
  bb5a6_pass = bb5a6_correctness.get("verdict") == "PASS_BB5A6_CORRECTNESS" and bool(bb5a6_correctness.get("gate_pass"))
  bb5a7_blocked = bb5a7_performance.get("verdict") == "BLOCKED_BB5A7_PERFORMANCE_GATE_NOT_MET"
  bb5a8_static_mapping = bb5a8_mapping.get("verdict") == "PASS_STATIC_TENSILE_TINYGRAD_MAPPING_CAUSAL_PROOF_BLOCKED" and bool(bb5a8_mapping.get("gate_pass"))
  bb5a8_capture_pass = bb5a8_capture.get("verdict") == "PASS_AUTHORITY_KERNEL_CAPTURE_CAUSAL_INPUTS_READY" and bool(bb5a8_capture.get("gate_pass"))
  bb5a9_pass = bb5a9_causal_delta.get("verdict") == "PASS_BB5A9_CAUSAL_DELTA_PACKAGE_IMPLEMENTATION_TRACKS_READY" and bool(bb5a9_causal_delta.get("gate_pass"))
  bb5a10_pass = bb5a10_layout_audit.get("verdict") == "PASS_TENSILE_LAYOUT_AUDIT_CANDIDATE_SPEC_READY_NOT_BITEXACT" and bool(bb5a10_layout_audit.get("gate_pass"))
  bb5a10_plan_pass = bb5a10_implementation_plan.get("verdict") == "PASS_BB5A10_IMPLEMENTATION_PLAN_READY" and bool(bb5a10_implementation_plan.get("gate_pass"))
  bb5a10_p1_pass = bb5a10_p1_layout_spec.get("verdict") == "PASS_BB5A10_P1_LAYOUT_SPEC_READY" and bool(bb5a10_p1_layout_spec.get("gate_pass"))
  bb5a10_p2_p5_pass = (
    bb5a10_p2_rendered_lds.get("verdict") == "PASS_BB5A10_P2_RENDERED_LDS_STORE_READ" and bool(bb5a10_p2_rendered_lds.get("gate_pass")) and
    bb5a10_p3_kloop_stage.get("verdict") == "PASS_BB5A10_P3_KLOOP_STAGE_SCHEDULER" and bool(bb5a10_p3_kloop_stage.get("gate_pass")) and
    bb5a10_p4_wait_barrier.get("verdict") == "PASS_BB5A10_P4_WAIT_BARRIER_SCHEDULE" and bool(bb5a10_p4_wait_barrier.get("gate_pass")) and
    bb5a10_p5_resource_policy.get("verdict") == "PASS_BB5A10_P5_RESOURCE_POLICY" and bool(bb5a10_p5_resource_policy.get("gate_pass")))
  bb5a10_p6_pass = bb5a10_p6_structural.get("verdict") == "PASS_BB5A10_P6_STRUCTURAL_CANDIDATE" and bool(bb5a10_p6_structural.get("gate_pass"))
  bb5a10_p7_scope_pass = bb5a10_p7_scope.get("verdict") == "PASS_BB5A10_P7_CORRECTNESS_SCOPE_READY" and bool(bb5a10_p7_scope.get("gate_pass"))
  bb5a10_p7ab_pass = bb5a10_p7ab.get("verdict") == "PASS_BB5A10_P7A_P7B_EXECUTABLE_WRAPPER" and bool(bb5a10_p7ab.get("gate_pass"))
  bb5a10_p7c_pass = bb5a10_p7c.get("verdict") == "PASS_BB5A10_P7C_SMALL_NUMERIC_CORRECTNESS" and bool(bb5a10_p7c.get("gate_pass"))
  bb5a10_p7d_pass = bb5a10_p7d.get("verdict") == "PASS_BB5A10_P7D_AUTHORITY_SUBSET_CORRECTNESS" and bool(bb5a10_p7d.get("gate_pass"))
  bb5a10_p7e_pass = bb5a10_p7e.get("verdict") == "PASS_BB5A10_P7E_P8_HANDOFF_PACKAGE" and bool(bb5a10_p7e.get("gate_pass"))
  bb5a10_p8_blocked = bb5a10_p8.get("verdict") == "BLOCKED_BB5A10_P8_FULL_AUTHORITY_LAUNCH_MAPPING_REQUIRED"
  bb5a10_p8_perf_blocked = bb5a10_p8.get("verdict") == "BLOCKED_BB5A10_P8_PERFORMANCE_GATE_NOT_MET"
  bb5a10_p8_pass = bb5a10_p8.get("verdict") == "PASS_BB5A10_P8_PERFORMANCE_GATE" and bool(bb5a10_p8.get("gate_pass"))
  bb5a10_p8_tta_scoped = bb5a10_p8_tta_scope.get("verdict") == "PASS_BB5A10_P8_TTA_SCOPE_READY" and bool(bb5a10_p8_tta_scope.get("gate_pass"))
  bb5a10_p8_tta_completion_scoped = bb5a10_p8_tta_completion.get("verdict") == "PASS_BB5A10_P8_TTA_COMPLETION_SCOPE_READY" and bool(bb5a10_p8_tta_completion.get("gate_pass"))
  bb5a10_p8_tta1_pass = bb5a10_p8_tta1.get("verdict") == "PASS_BB5A10_P8_TTA1_FULL_GRID_CORRECTNESS" and bool(bb5a10_p8_tta1.get("gate_pass"))
  bb5a10_p8_tta2_pass = bb5a10_p8_tta2.get("verdict") == "PASS_BB5A10_P8_TTA2_AUTHORITY_SAMPLE_CORRECTNESS" and bool(bb5a10_p8_tta2.get("gate_pass"))
  bb5a10_p8_tta3_blocked = bb5a10_p8_tta3.get("verdict") == "BLOCKED_BB5A10_P8_TTA3_SELECTED_COMPATIBLE_MACRO_CANDIDATE"
  bb5a10_p8_tta3_pass = bb5a10_p8_tta3.get("verdict") == "PASS_BB5A10_P8_TTA3_MACRO_CANDIDATE" and bool(bb5a10_p8_tta3.get("gate_pass"))
  bb5a10_p8_bottleneck_classified = bb5a10_p8_bottleneck.get("verdict") == "PASS_BB5A10_P8_BOTTLENECK_CLASSIFIED_LDS_STAGING_FAMILY" and bool(bb5a10_p8_bottleneck.get("gate_pass"))
  bb5a10_p8_global_direct_decision = bb5a10_p8_global_direct.get("verdict") == "PASS_BB5A10_P8_GLOBAL_DIRECT_CANDIDATE_DECISION" and bool(bb5a10_p8_global_direct.get("gate_pass"))
  bb5a10_p8_timing_authority_reconciled = bb5a10_p8_timing_authority.get("verdict") == "PASS_BB5A10_P8_TIMING_AUTHORITY_RECONCILED_SAME_HARNESS_REQUIRED" and bool(bb5a10_p8_timing_authority.get("gate_pass"))
  q8_transfer = verdict_row(
    "BB-6",
    "BLOCKED_ON_SHARED_BACKEND_CAPABILITY" if not bb5a_ready else "READY",
    "q8 native transfer is a downstream consumer, not the first implementation target." if not bb5a_ready else
    "BB-5a reports shared renderer/allocator capability ready; q8 transfer may be scoped after BB-5 rerun.",
    "Attempt only after shared scheduler/resource capability exists; continue at <=75us, strong pass <=60us." if not bb5a_ready else
    "Rerun BB-5 performance gate before starting q8 transfer.",
  )
  model_gate = verdict_row(
    "BB-7",
    "BLOCKED_ON_PRIMITIVE_MOVEMENT",
    "No model gate can run before primitive backend movement exists.",
    "Run W==D decode and pp prefill only after BB-5/BB-6 produce candidate kernels.",
  )
  result = {
    "date": "2026-06-19",
    "schema": "amd_broad_backend_roadmap_execution_v1",
    "verdict": "BROAD_BACKEND_ACCEPTED_BB5A10_P8_TIMING_AUTHORITY_RECONCILED_SAME_HARNESS_NEXT_Q8_BLOCKED" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a10_p8_timing_authority_reconciled else
               "BROAD_BACKEND_ACCEPTED_BB5A10_P8_GLOBAL_DIRECT_DECISION_DONE_TIMING_AUTHORITY_NEXT_Q8_BLOCKED" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a10_p8_global_direct_decision else
               "BROAD_BACKEND_ACCEPTED_BB5A10_P8_BOTTLENECK_LDS_FAMILY_CLASSIFIED_GLOBAL_DIRECT_NEXT_Q8_BLOCKED" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a10_p8_bottleneck_classified else
               "BROAD_BACKEND_ACCEPTED_BB5A10_P8_PERFORMANCE_PASS_P9_NEXT" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a10_p8_pass else
               "BROAD_BACKEND_ACCEPTED_BB5A10_P8_PERFORMANCE_BLOCKED_18TFLOPS_Q8_BLOCKED" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a10_p8_perf_blocked else
               "BROAD_BACKEND_ACCEPTED_BB5A10_P8_TTA3_MACRO_DONE_P8_TIMING_NEXT_Q8_BLOCKED" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a10_p8_tta3_pass else
               "BROAD_BACKEND_ACCEPTED_BB5A10_P8_TTA3_BLOCKED_DS64_MACRO_STORES_TTA3A_NEXT_Q8_BLOCKED" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a10_p8_tta3_blocked else
               "BROAD_BACKEND_ACCEPTED_BB5A10_P8_TTA2_AUTHORITY_SAMPLE_DONE_TTA3_NEXT_Q8_BLOCKED" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a10_p8_tta2_pass else
               "BROAD_BACKEND_ACCEPTED_BB5A10_P8_TTA1_FULL_GRID_DONE_TTA2_NEXT_Q8_BLOCKED" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a10_p8_tta1_pass else
               "BROAD_BACKEND_ACCEPTED_BB5A10_P8_TTA_COMPLETION_SCOPED_TTA1_NEXT_Q8_BLOCKED" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a10_p8_tta_completion_scoped else
               "BROAD_BACKEND_ACCEPTED_BB5A10_P8_TTA_SCOPED_TTA1_NEXT_Q8_BLOCKED" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a10_p8_tta_scoped else
               "BROAD_BACKEND_ACCEPTED_BB5A10_P8_BLOCKED_FULL_AUTHORITY_LAUNCH_MAPPING_Q8_BLOCKED" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a10_p8_blocked else
               "BROAD_BACKEND_ACCEPTED_BB5A10_P7E_HANDOFF_DONE_P8_NEXT_Q8_BLOCKED" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a10_p7e_pass else
               "BROAD_BACKEND_ACCEPTED_BB5A10_P7D_AUTHORITY_SUBSET_DONE_P7E_NEXT_Q8_BLOCKED" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a10_p7d_pass else
               "BROAD_BACKEND_ACCEPTED_BB5A10_P7C_SMALL_NUMERIC_DONE_P7D_NEXT_Q8_BLOCKED" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a10_p7c_pass else
               "BROAD_BACKEND_ACCEPTED_BB5A10_P7A_P7B_DONE_P7C_NEXT_Q8_BLOCKED" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a10_p7ab_pass else
               "BROAD_BACKEND_ACCEPTED_BB5A10_P7_CORRECTNESS_SCOPED_P7A_P7B_NEXT_Q8_BLOCKED" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a10_p7_scope_pass else
               "BROAD_BACKEND_ACCEPTED_BB5A10_P6_STRUCTURAL_DONE_P7_CORRECTNESS_NEXT_Q8_BLOCKED" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a10_p6_pass else
               "BROAD_BACKEND_ACCEPTED_BB5A10_P2_P5_DONE_P6_NEXT_Q8_BLOCKED" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a10_p2_p5_pass else
               "BROAD_BACKEND_ACCEPTED_BB5A10_P1_LAYOUT_SPEC_DONE_P2_P5_NEXT_Q8_BLOCKED" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a10_p1_pass else
               "BROAD_BACKEND_ACCEPTED_BB5A10_IMPLEMENTATION_PLAN_READY_Q8_BLOCKED" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a10_plan_pass else
               "BROAD_BACKEND_ACCEPTED_BB5A10_LAYOUT_AUDIT_DONE_CANDIDATE_SPEC_READY_Q8_BLOCKED" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a10_pass else
               "BROAD_BACKEND_ACCEPTED_BB5A9_CAUSAL_DELTA_DONE_PARALLEL_IMPLEMENTATION_READY_Q8_BLOCKED" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a9_pass else
               "BROAD_BACKEND_ACCEPTED_BB5A8_AUTHORITY_CAPTURE_DONE_CAUSAL_DELTA_NEXT_Q8_BLOCKED" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a8_capture_pass else
               "BROAD_BACKEND_ACCEPTED_BB5A8_STATIC_MAPPING_DONE_CAUSAL_PROOF_BLOCKED_Q8_BLOCKED" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a8_static_mapping else
               "BROAD_BACKEND_ACCEPTED_BB5A7_PERFORMANCE_BLOCKED_Q8_BLOCKED" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a7_blocked else
               "BROAD_BACKEND_ACCEPTED_BB5A6_CORRECTNESS_PASS_BB5A7_NEXT" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a6_pass else
               "BROAD_BACKEND_ACCEPTED_BB5A5_POLICY_PASS_BB5A6_NEXT" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a5_pass else
               "BROAD_BACKEND_ACCEPTED_BB5A4_RESOURCE_PASS_BB5A5_NEXT" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a4_pass else
               "BROAD_BACKEND_ACCEPTED_BB5A3_WAIT_PASS_BB5A4_NEXT" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a3_pass else
               "BROAD_BACKEND_ACCEPTED_BB5A2_PASS_BB5A3_NEXT" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and bb5a2_full_pass else
               "BROAD_BACKEND_ACCEPTED_BB5A2_LAYER3_PASS_INTEGRATION_NEXT" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and software_pipeline.get("verdict") == "BLOCKED_REAL_RENDERER_ALLOCATOR_INTEGRATION" and bb5a2_layer3_pass else
               "BROAD_BACKEND_ACCEPTED_BB5A2_LAYER2_PASS_LAYER3_NEXT" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and software_pipeline.get("verdict") == "BLOCKED_REAL_RENDERER_ALLOCATOR_INTEGRATION" and bb5a2_layer2_pass else
               "BROAD_BACKEND_ACCEPTED_BB5A2_LAYER1_PASS_LAYER2_NEXT" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and software_pipeline.get("verdict") == "BLOCKED_REAL_RENDERER_ALLOCATOR_INTEGRATION" and bb5a2_layer1_pass else
               "BROAD_BACKEND_ACCEPTED_BB5A2_SOLUTION_SCOPED_REAL_LDS_LOWERING_REQUIRED" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and software_pipeline.get("verdict") == "BLOCKED_REAL_RENDERER_ALLOCATOR_INTEGRATION" and bb5a2_solution_scoped else
               "BROAD_BACKEND_ACCEPTED_BB5A_EXECUTION_BLOCKED_BB5A2_REAL_LDS_LOWERING" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and software_pipeline.get("verdict") == "BLOCKED_REAL_RENDERER_ALLOCATOR_INTEGRATION" and bb5a_execution_blocked else
               "BROAD_BACKEND_ACCEPTED_BB5A_FULL_PLAN_READY_BB5A2_NEXT" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and software_pipeline.get("verdict") == "BLOCKED_REAL_RENDERER_ALLOCATOR_INTEGRATION" and bb5a_renderer_allocator.get("verdict") == "BB5A_SCOPE_COMPLETE_IMPLEMENTATION_NOT_READY" and bb5a_full_plan_ready else
               "BROAD_BACKEND_ACCEPTED_BB5A1_PASS_BB5A2_NEXT" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and software_pipeline.get("verdict") == "BLOCKED_REAL_RENDERER_ALLOCATOR_INTEGRATION" and bb5a_renderer_allocator.get("verdict") == "BB5A_SCOPE_COMPLETE_IMPLEMENTATION_NOT_READY" and bb5a1_pass else
               "BROAD_BACKEND_ACCEPTED_BB5A1_SCOPED_IMPLEMENTATION_NOT_READY" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and software_pipeline.get("verdict") == "BLOCKED_REAL_RENDERER_ALLOCATOR_INTEGRATION" and bb5a_renderer_allocator.get("verdict") == "BB5A_SCOPE_COMPLETE_IMPLEMENTATION_NOT_READY" and bb5a1_scoped else
               "BROAD_BACKEND_ACCEPTED_BB5A_SCOPED_IMPLEMENTATION_NOT_READY" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and software_pipeline.get("verdict") == "BLOCKED_REAL_RENDERER_ALLOCATOR_INTEGRATION" and bb5a_renderer_allocator.get("verdict") == "BB5A_SCOPE_COMPLETE_IMPLEMENTATION_NOT_READY" else
               "BROAD_BACKEND_ACCEPTED_BB5_BLOCKED_RENDERER_ALLOCATOR" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and software_pipeline.get("verdict") == "BLOCKED_REAL_RENDERER_ALLOCATOR_INTEGRATION" else
               "BROAD_BACKEND_ACCEPTED_BB5_TFLOPS_PASS" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass and software_pipeline.get("verdict") == "PASS_SOFTWARE_PIPELINE_TFLOPS" else
               "BROAD_BACKEND_ACCEPTED_BB4_PASS_BB5_READY" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass and bb3_pass and bb4_pass else
               "BROAD_BACKEND_ACCEPTED_BB2_PASS_BB3_READY" if authority.get("gate_pass") and oracle_suite.get("gate_pass") and bb2_pass else
               "BROAD_BACKEND_ACCEPTED_BB1_PASS_BB2_NEXT" if authority.get("gate_pass") and oracle_suite.get("gate_pass") else
               "BROAD_BACKEND_ACCEPTED_BB1_INCOMPLETE",
    "decision": ("Stop before BB-6; BB-5a.10 P8 timing authority is reconciled. The prior 43.026 TFLOPS captured authority row does not validate current P8 hand-ASM candidates because kernel identity and timing harness differ. Build a same-harness authority timing bridge next; q8 transfer remains blocked." if bb5a10_p8_timing_authority_reconciled else
                 "Stop before BB-6; BB-5a.10 P8 global-direct decision is complete. Existing no-LDS candidates are correct but still miss the P8 gate, so reconcile timing authority against the prior ~43 TFLOPS artifact before new scheduling work. q8 transfer remains blocked." if bb5a10_p8_global_direct_decision else
                 "Stop before BB-6; BB-5a.10 P8 bottleneck is classified as LDS staging family overhead. Stop tuning this LDS macro; next is a selected-compatible global-direct/IC-served WMMA candidate decision. q8 transfer remains blocked." if bb5a10_p8_bottleneck_classified else
                 "BB-5a.10 P8 performance gate passes. Scope P9 q8 transfer reopen next." if bb5a10_p8_pass else
                 "Stop before BB-6; BB-5a.10 P8 performance is blocked at 18.38 TFLOPS best versus the 60 TFLOPS gate. Classify the P8 bottleneck; q8 transfer remains blocked." if bb5a10_p8_perf_blocked else
                 "Stop before BB-6; BB-5a.10 P8 TTA3 macro candidate passes after TTA3a ds_store_b64 conversion. Run P8 timing gate next; q8 transfer remains blocked until P8 passes." if bb5a10_p8_tta3_pass else
                 "Stop before BB-6; BB-5a.10 P8 TTA3 is blocked because the 128x128 macro helper uses ds_store_b128 instead of selected-compatible ds_store_b64. Implement TTA3a store conversion next; q8 transfer remains blocked until P8 passes." if bb5a10_p8_tta3_blocked else
                 "Stop before BB-6; BB-5a.10 P8 TTA2 authority sampled correctness passes. Implement TTA3 selected macro-tile candidate next; q8 transfer remains blocked until P8 passes." if bb5a10_p8_tta2_pass else
                 "Stop before BB-6; BB-5a.10 P8 TTA1 full-grid correctness passes. Implement TTA2 authority sampled correctness next; q8 transfer remains blocked until P8 passes." if bb5a10_p8_tta1_pass else
                 "Stop before BB-6; BB-5a.10 P8 TTA is scoped through completion. Implement TTA1 full-grid correctness bridge next; q8 transfer remains blocked until P8 passes." if bb5a10_p8_tta_completion_scoped else
                 "Stop before BB-6; BB-5a.10 P8 TTA is scoped. Implement TTA1 full-grid correctness bridge next; q8 transfer remains blocked until P8 passes." if bb5a10_p8_tta_scoped else
                 "Stop before BB-6; BB-5a.10 P8 is blocked because full-authority M=512,N=12288 launch mapping is not implemented. q8 transfer remains blocked until P8 passes." if bb5a10_p8_blocked else
                 "Stop before BB-6; BB-5a.10 P7e P8 handoff package passes. Run P8 performance gate next; q8 transfer remains blocked until P8 passes." if bb5a10_p7e_pass else
                 "Stop before BB-6; BB-5a.10 P7d authority-subset correctness passes. Build P7e P8 handoff package next; q8 transfer remains blocked until P8 passes." if bb5a10_p7d_pass else
                 "Stop before BB-6; BB-5a.10 P7c small numeric correctness passes. Build P7d authority-shape correctness smoke next; q8 transfer remains blocked until P8 passes." if bb5a10_p7c_pass else
                 "Stop before BB-6; BB-5a.10 P7a/P7b pass. Build P7c small deterministic numeric correctness next; q8 transfer remains blocked until P8 passes." if bb5a10_p7ab_pass else
                 "Stop before BB-6; BB-5a.10 P7 correctness is scoped. Implement P7a known-good LDS WMMA smoke and P7b executable structural wrapper next; q8 transfer remains blocked until P8 passes." if bb5a10_p7_scope_pass else
                 "Stop before BB-6; BB-5a.10 P6 structural candidate passes. Build P7 executable correctness harness next; q8 transfer remains blocked until P8 passes." if bb5a10_p6_pass else
                 "Stop before BB-6; BB-5a.10 P2/P3/P4/P5 pass. Run P6 structural candidate gate next; q8 transfer remains blocked until P8 passes." if bb5a10_p2_p5_pass else
                 "Stop before BB-6; BB-5a.10 P1 layout spec is complete. Run P2/P3/P4/P5 as one coordinated implementation batch, then P6/P7/P8 gates; q8 transfer remains blocked until P8 passes." if bb5a10_p1_pass else
                 "Stop before BB-6; BB-5a.10 implementation plan is ready. Run P1-P5 as one coordinated implementation batch, then P6/P7/P8 gates; q8 transfer remains blocked until P8 passes." if bb5a10_plan_pass else
                 "Stop before BB-6; BB-5a.10 layout audit is complete. Implement the non-bitexact staged-LDS authority candidate against selected-kernel-compatible LDS stores, ds_load_b128, semantic waits/barriers, and scratch-free resource policy; q8 transfer remains blocked." if bb5a10_pass else
                 "Stop before BB-6; BB-5a.9 causal delta package is complete. Start parallel implementation tracks B_LDS_layout, C_K_loop_scheduler, and D_resource_policy; keep measured candidate and q8 transfer blocked." if bb5a9_pass else
                 "Stop before BB-6; BB-5a.8 authority capture is complete. The captured same-kernel row is timing-equivalent to the 42.0 TFLOPS authority row and has source/ELF/disassembly/resource evidence. Run the causal-delta probe next; Q8 transfer remains blocked." if bb5a8_capture_pass else
                 "Stop before BB-6; BB-5a.8 proves static Tensile-to-tinygrad mapping but blocks causal proof until the measured 42.0 TFLOPS tinygrad authority kernel is captured as source/ISA/resource evidence. Q8 transfer remains blocked." if bb5a8_static_mapping else
                 "Stop before BB-6; BB-5a.7 performance gate is blocked at 42.0 TFLOPS versus the 60.0 TFLOPS pure tinygrad gate. Q8 transfer remains blocked." if bb5a7_blocked else
                 "Stop before BB-6; BB-5a.6 correctness passes. Run BB-5a.7 performance gate next." if bb5a6_pass else
                 "Stop before BB-6; BB-5a.5 resource policy passes. Run BB-5a.6 correctness next." if bb5a5_pass else
                 "Stop before BB-6; BB-5a.4 resource control passes. Run BB-5a.5 resource policy next." if bb5a4_pass else
                 "Stop before BB-6; BB-5a.3 wait scheduler integration passes. Run BB-5a.4 resource control next." if bb5a3_pass else
                 "Stop before BB-6; BB-5a.2 double-buffered LDS lowering passes. Run BB-5a.3 wait scheduler integration next." if bb5a2_full_pass else
                 "Stop before BB-6; BB-5a.2 Layer 3 passes as ELF LDS evidence. Integrate the gated path into real postrange/AMD renderer lowering next." if bb5a2_layer3_pass else
                 "Stop before BB-6; BB-5a.2 Layer 2 passes. Implement renderer/ISA evidence for the lowered two-slot LDS structure next." if bb5a2_layer2_pass else
                 "Stop before BB-6; BB-5a.2 Layer 1 passes. Implement the gated postrange/rangeify lowering hook next." if bb5a2_layer1_pass else
                 "Stop before BB-6; BB-5a.2 solution is scoped. Implement AMDLDSStagePlan, then a real LDS lowering probe and gated lowering hook." if bb5a2_solution_scoped else
                 "Stop before BB-6; BB-5a execution is blocked at BB-5a.2 because no real AMD lowering path consumes pipeline stage metadata into distinct LDS slots." if bb5a_execution_blocked else
                 "Stop before BB-6; full BB-5a plan is ready and BB-5a.2 double-buffered LDS lowering is next." if bb5a_full_plan_ready else
                 "Stop before BB-6; BB-5a.1 pipeline IR passes as read-only metadata, so BB-5a.2 double-buffered LDS lowering is next." if bb5a1_pass else
                 "Stop before BB-6; BB-5a.1 is scoped and pipeline IR implementation remains not ready." if bb5a1_scoped else
                 "Stop before BB-6; BB-5a is scoped and implementation remains not ready." if bb5a_renderer_allocator.get("verdict") == "BB5A_SCOPE_COMPLETE_IMPLEMENTATION_NOT_READY" else
                 "Stop before BB-6; real renderer/allocator integration is required before q8 transfer." if software_pipeline.get("verdict") == "BLOCKED_REAL_RENDERER_ALLOCATOR_INTEGRATION" else
                 "Start BB-6 q8 transfer only if BB-5 produced a TFLOPS pass." if software_pipeline.get("verdict") == "PASS_SOFTWARE_PIPELINE_TFLOPS" else
                 "Start BB-5 software-pipelined prefill probe next. Do not start q8-specific scheduler work." if bb3_pass and bb4_pass else
                 "Start BB-3 semantic wait/scheduler emitter next. Do not start q8-specific scheduler work." if bb2_pass else
                 "Start BB-2 schedule metadata IR next. Do not start q8-specific scheduler work."),
    "artifacts": {
      "authority": "bench/amd-broad-backend-roadmap/authority.json",
      "oracle_suite": "bench/amd-broad-backend-roadmap/oracle_suite.json",
      "schedule_metadata_ir_result": "bench/amd-broad-backend-roadmap/schedule_metadata_ir_result.json",
      "wait_scheduler_result": "bench/amd-broad-backend-roadmap/wait_scheduler_result.json",
      "register_resource_result": "bench/amd-broad-backend-roadmap/register_resource_result.json",
      "software_pipeline_result": "bench/amd-broad-backend-roadmap/software_pipeline_result.json",
      "bb5a_renderer_allocator_scope": "bench/amd-broad-backend-roadmap/bb5a_renderer_allocator_scope.json",
      "bb5a1_pipeline_ir_scope": "bench/amd-broad-backend-roadmap/bb5a1_pipeline_ir_scope.json",
      "bb5a1_pipeline_ir_result": "bench/amd-broad-backend-roadmap/bb5a1_pipeline_ir_result.json",
      "bb5a_full_plan": "bench/amd-broad-backend-roadmap/bb5a_full_plan.json",
      "bb5a_execution_result": "bench/amd-broad-backend-roadmap/bb5a_execution_result.json",
      "bb5a2_solution_scope": "bench/amd-broad-backend-roadmap/bb5a2_solution_scope.json",
      "bb5a2_lds_stage_plan_result": "bench/amd-broad-backend-roadmap/bb5a2_lds_stage_plan_result.json",
      "bb5a2_lowering_hook_result": "bench/amd-broad-backend-roadmap/bb5a2_lowering_hook_result.json",
      "bb5a2_render_isa_evidence_result": "bench/amd-broad-backend-roadmap/bb5a2_render_isa_evidence_result.json",
      "bb5a2_real_lowering_integration_result": "bench/amd-broad-backend-roadmap/bb5a2_real_lowering_integration_result.json",
      "bb5a2_pipelined_dataflow_result": "bench/amd-broad-backend-roadmap/bb5a2_pipelined_dataflow_result.json",
      "bb5a3_wait_scheduler_integration_result": "bench/amd-broad-backend-roadmap/bb5a3_wait_scheduler_integration_result.json",
      "bb5a4_allocator_resource_result": "bench/amd-broad-backend-roadmap/bb5a4_allocator_resource_result.json",
      "bb5a5_resource_policy_result": "bench/amd-broad-backend-roadmap/bb5a5_resource_policy_result.json",
      "bb5a6_correctness_result": "bench/amd-broad-backend-roadmap/bb5a6_correctness_result.json",
      "bb5a7_performance_gate_result": "bench/amd-broad-backend-roadmap/bb5a7_performance_gate_result.json",
      "bb5a8_tensile_mapping_result": "bench/amd-broad-backend-roadmap/bb5a8_tensile_mapping_result.json",
      "bb5a8_authority_kernel_capture_result": "bench/amd-broad-backend-roadmap/bb5a8_authority_kernel_capture_result.json",
      "bb5a9_causal_delta_package_result": "bench/amd-broad-backend-roadmap/bb5a9_causal_delta_package_result.json",
      "bb5a10_tensile_layout_audit_result": "bench/amd-broad-backend-roadmap/bb5a10_tensile_layout_audit_result.json",
      "bb5a10_implementation_plan_result": "bench/amd-broad-backend-roadmap/bb5a10_implementation_plan_result.json",
      "bb5a10_p1_layout_spec_result": "bench/amd-broad-backend-roadmap/bb5a10_p1_layout_spec_result.json",
      "bb5a10_p2_rendered_lds_result": "bench/amd-broad-backend-roadmap/bb5a10_p2_rendered_lds_result.json",
      "bb5a10_p3_kloop_stage_result": "bench/amd-broad-backend-roadmap/bb5a10_p3_kloop_stage_result.json",
      "bb5a10_p4_wait_barrier_result": "bench/amd-broad-backend-roadmap/bb5a10_p4_wait_barrier_result.json",
      "bb5a10_p5_resource_policy_result": "bench/amd-broad-backend-roadmap/bb5a10_p5_resource_policy_result.json",
      "bb5a10_p6_structural_candidate_result": "bench/amd-broad-backend-roadmap/bb5a10_p6_structural_candidate_result.json",
      "bb5a10_p7_correctness_scope_result": "bench/amd-broad-backend-roadmap/bb5a10_p7_correctness_scope_result.json",
      "bb5a10_p7a_p7b_correctness_result": "bench/amd-broad-backend-roadmap/bb5a10_p7a_p7b_correctness_result.json",
      "bb5a10_p7c_numeric_correctness_result": "bench/amd-broad-backend-roadmap/bb5a10_p7c_numeric_correctness_result.json",
      "bb5a10_p7d_authority_correctness_result": "bench/amd-broad-backend-roadmap/bb5a10_p7d_authority_correctness_result.json",
      "bb5a10_p7e_p8_handoff_result": "bench/amd-broad-backend-roadmap/bb5a10_p7e_p8_handoff_result.json",
      "bb5a10_p8_performance_result": "bench/amd-broad-backend-roadmap/bb5a10_p8_performance_result.json",
      "bb5a10_p8_tta_scope_result": "bench/amd-broad-backend-roadmap/bb5a10_p8_tta_scope_result.json",
      "bb5a10_p8_tta_completion_scope_result": "bench/amd-broad-backend-roadmap/bb5a10_p8_tta_completion_scope_result.json",
      "bb5a10_p8_tta1_full_grid_correctness_result": "bench/amd-broad-backend-roadmap/bb5a10_p8_tta1_full_grid_correctness_result.json",
      "bb5a10_p8_tta2_authority_sample_correctness_result": "bench/amd-broad-backend-roadmap/bb5a10_p8_tta2_authority_sample_correctness_result.json",
      "bb5a10_p8_tta3_macro_candidate_result": "bench/amd-broad-backend-roadmap/bb5a10_p8_tta3_macro_candidate_result.json",
      "bb5a10_p8_tta3a_ds64_macro_conversion_result": "bench/amd-broad-backend-roadmap/bb5a10_p8_tta3a_ds64_macro_conversion_result.json",
      "bb5a10_p8_bottleneck_classification_result": "bench/amd-broad-backend-roadmap/bb5a10_p8_bottleneck_classification_result.json",
      "bb5a10_p8_global_direct_candidate_decision_result": "bench/amd-broad-backend-roadmap/bb5a10_p8_global_direct_candidate_decision_result.json",
      "bb5a10_p8_timing_authority_reconciliation_result": "bench/amd-broad-backend-roadmap/bb5a10_p8_timing_authority_reconciliation_result.json",
      "q8_transfer_result": "bench/amd-broad-backend-roadmap/q8_transfer_result.json",
      "model_gate_result": "bench/amd-broad-backend-roadmap/model_gate_result.json",
    },
    "phase_status": {
      "BB-0": authority.get("acceptance", {}).get("verdict"),
      "BB-1": "PASS" if oracle_suite.get("gate_pass") else "INCOMPLETE",
      "BB-2": schedule_metadata["verdict"],
      "BB-3": wait_scheduler["verdict"],
      "BB-4": register_resource["verdict"],
      "BB-5": software_pipeline["verdict"],
      "BB-5a": bb5a_renderer_allocator["verdict"],
      "BB-5a.1": bb5a1_pipeline_ir["verdict"],
      "BB-5a-plan": bb5a_full_plan["verdict"],
      "BB-5a-execution": bb5a_execution["verdict"],
      "BB-5a.2-solution": bb5a2_solution["verdict"],
      "BB-5a.2-layer-1": bb5a2_lds_stage_plan["verdict"],
      "BB-5a.2-layer-2": bb5a2_lowering_hook["verdict"],
      "BB-5a.2-layer-3": bb5a2_render_isa["verdict"],
      "BB-5a.2-integration": bb5a2_integration.get("verdict"),
      "BB-5a.2-dataflow": bb5a2_dataflow.get("verdict"),
      "BB-5a.3": bb5a3_integration.get("verdict"),
      "BB-5a.4": bb5a4_allocator.get("verdict"),
      "BB-5a.5": bb5a5_policy.get("verdict"),
      "BB-5a.6": bb5a6_correctness.get("verdict"),
      "BB-5a.7": bb5a7_performance.get("verdict"),
      "BB-5a.8-mapping": bb5a8_mapping.get("verdict"),
      "BB-5a.8-capture": bb5a8_capture.get("verdict"),
      "BB-5a.9-causal-delta": bb5a9_causal_delta.get("verdict"),
      "BB-5a.10-layout-audit": bb5a10_layout_audit.get("verdict"),
      "BB-5a.10-plan": bb5a10_implementation_plan.get("verdict"),
      "BB-5a.10-P1-layout-spec": bb5a10_p1_layout_spec.get("verdict"),
      "BB-5a.10-P2-rendered-LDS": bb5a10_p2_rendered_lds.get("verdict"),
      "BB-5a.10-P3-kloop-stage": bb5a10_p3_kloop_stage.get("verdict"),
      "BB-5a.10-P4-wait-barrier": bb5a10_p4_wait_barrier.get("verdict"),
      "BB-5a.10-P5-resource-policy": bb5a10_p5_resource_policy.get("verdict"),
      "BB-5a.10-P6-structural-candidate": bb5a10_p6_structural.get("verdict"),
      "BB-5a.10-P7-correctness-scope": bb5a10_p7_scope.get("verdict"),
      "BB-5a.10-P7a-P7b-correctness": bb5a10_p7ab.get("verdict"),
      "BB-5a.10-P7c-numeric-correctness": bb5a10_p7c.get("verdict"),
      "BB-5a.10-P7d-authority-correctness": bb5a10_p7d.get("verdict"),
      "BB-5a.10-P7e-P8-handoff": bb5a10_p7e.get("verdict"),
      "BB-5a.10-P8-performance": bb5a10_p8.get("verdict"),
      "BB-5a.10-P8-TTA-scope": bb5a10_p8_tta_scope.get("verdict"),
      "BB-5a.10-P8-TTA-completion-scope": bb5a10_p8_tta_completion.get("verdict"),
      "BB-5a.10-P8-TTA1-full-grid-correctness": bb5a10_p8_tta1.get("verdict"),
      "BB-5a.10-P8-TTA2-authority-sample-correctness": bb5a10_p8_tta2.get("verdict"),
      "BB-5a.10-P8-TTA3-macro-candidate": bb5a10_p8_tta3.get("verdict"),
      "BB-5a.10-P8-TTA3a-ds64-macro-conversion": bb5a10_p8_tta3a.get("verdict"),
      "BB-5a.10-P8-bottleneck-classification": bb5a10_p8_bottleneck.get("verdict"),
      "BB-5a.10-P8-global-direct-decision": bb5a10_p8_global_direct.get("verdict"),
      "BB-5a.10-P8-timing-authority-reconciliation": bb5a10_p8_timing_authority.get("verdict"),
      "BB-6": "BLOCKED_ON_BB5A_IMPLEMENTATION" if bb5a_renderer_allocator.get("verdict") == "BB5A_SCOPE_COMPLETE_IMPLEMENTATION_NOT_READY" else
              "BLOCKED_ON_BB5_RENDERER_ALLOCATOR" if software_pipeline.get("verdict") == "BLOCKED_REAL_RENDERER_ALLOCATOR_INTEGRATION" else q8_transfer["verdict"],
      "BB-7": model_gate["verdict"],
    },
    "next": {
      "phase": "BB-5a.10-P8-same-harness-authority-timing-bridge" if bb5a10_p8_timing_authority_reconciled else
               "BB-5a.10-P8-timing-authority-reconciliation" if bb5a10_p8_global_direct_decision else
               "BB-5a.10-P8-global-direct-candidate-decision" if bb5a10_p8_bottleneck_classified else
               "BB-5a.10-P9-q8-transfer-reopen" if bb5a10_p8_pass else
               "BB-5a.10-P8-bottleneck-classification" if bb5a10_p8_perf_blocked else
               "BB-5a.10-P8-performance-gate" if bb5a10_p8_tta3_pass else
               "BB-5a.10-P8-TTA3a-selected-ds64-macro-store-conversion" if bb5a10_p8_tta3_blocked else
               "BB-5a.10-P8-TTA3-selected-macro-tile-candidate" if bb5a10_p8_tta2_pass else
               "BB-5a.10-P8-TTA2-authority-sample-correctness" if bb5a10_p8_tta1_pass else
               "BB-5a.10-P8-TTA1-full-grid-correctness-bridge" if bb5a10_p8_tta_completion_scoped or bb5a10_p8_tta_scoped else
               "BB-5a.10-P8-full-authority-launch-mapping" if bb5a10_p8_blocked else
               "BB-5a.10-P8-performance-gate" if bb5a10_p7e_pass else
               "BB-5a.10-P7e-P8-handoff-package" if bb5a10_p7d_pass else
               "BB-5a.10-P7d-authority-shape-correctness" if bb5a10_p7c_pass else
               "BB-5a.10-P7c-small-numeric-correctness" if bb5a10_p7ab_pass else
               "BB-5a.10-P7a-P7b-correctness-harness" if bb5a10_p7_scope_pass else
               "BB-5a.10-P7-correctness" if bb5a10_p6_pass else
               "BB-5a.10-P6-structural-candidate" if bb5a10_p2_p5_pass else
               "BB-5a.10-P2-P5-implementation-batch" if bb5a10_p1_pass else
               "BB-5a.10-P1-P5-implementation-batch" if bb5a10_plan_pass else
               "BB-5a.10-implementation-plan" if bb5a10_pass else
               "BB-5a.10-layout-audit" if bb5a9_pass else
               "BB-5a.9-causal-delta" if bb5a8_capture_pass else
               "BB-5a.8-authority-kernel-capture" if bb5a8_static_mapping else
               "BB-5a.8" if bb5a7_blocked else
               "BB-5a.7" if bb5a6_pass else
               "BB-5a.6" if bb5a5_pass else
               "BB-5a.5" if bb5a4_pass else
               "BB-5a.4" if bb5a3_pass else
               "BB-5a.3" if bb5a2_full_pass else
               "BB-5a.2-integration" if bb5a2_layer3_pass else
               "BB-5a.2-layer-3" if bb5a2_layer2_pass else
               "BB-5a.2-layer-2" if bb5a2_layer1_pass else
               "BB-5a.2-layer-1" if bb5a2_solution_scoped else
               "BB-5a.2" if bb5a_execution_blocked or bb5a_full_plan_ready or bb5a1_pass else
               "BB-5a.1a" if bb5a1_scoped else
               "BB-5a.1" if bb5a_renderer_allocator.get("verdict") == "BB5A_SCOPE_COMPLETE_IMPLEMENTATION_NOT_READY" else
               "BB-5a" if software_pipeline.get("verdict") == "BLOCKED_REAL_RENDERER_ALLOCATOR_INTEGRATION" else
               "BB-6" if software_pipeline.get("verdict") == "PASS_SOFTWARE_PIPELINE_TFLOPS" else
               "BB-5" if bb3_pass and bb4_pass else "BB-3" if bb2_pass else "BB-2",
      "implementation_target": "same_harness_authority_timing_bridge" if bb5a10_p8_timing_authority_reconciled else
                               "p8_timing_authority_reconciliation" if bb5a10_p8_global_direct_decision else
                               "selected_compatible_global_direct_wmma_candidate_decision" if bb5a10_p8_bottleneck_classified else
                               "p9_q8_transfer_reopen" if bb5a10_p8_pass else
                               "p8_bottleneck_classification" if bb5a10_p8_perf_blocked else
                               "p8_performance_gate" if bb5a10_p8_tta3_pass else
                               "tta3a_ds_store_b64_macro_conversion" if bb5a10_p8_tta3_blocked else
                               "tta3_selected_macro_tile_candidate" if bb5a10_p8_tta2_pass else
                               "tta2_authority_sample_correctness" if bb5a10_p8_tta1_pass else
                               "tta1_gidx_full_grid_correctness_bridge" if bb5a10_p8_tta_completion_scoped or bb5a10_p8_tta_scoped else
                               "full_authority_launch_mapping_for_p8" if bb5a10_p8_blocked else
                               "p8_performance_gate" if bb5a10_p7e_pass else
                               "p8_handoff_package" if bb5a10_p7d_pass else
                               "authority_shape_correctness_smoke" if bb5a10_p7c_pass else
                               "small_deterministic_numeric_correctness" if bb5a10_p7ab_pass else
                               "known_good_lds_wmma_smoke_and_executable_wrapper" if bb5a10_p7_scope_pass else
                               "executable_correctness_harness" if bb5a10_p6_pass else
                               "structural_candidate_gate" if bb5a10_p2_p5_pass else
                               "renderer_kloop_wait_resource_batch" if bb5a10_p1_pass else
                               "layout_renderer_kloop_wait_resource_batch" if bb5a10_plan_pass else
                               "bb5a10_phase_plan" if bb5a10_pass else
                               "selected_tensile_layout_audit" if bb5a9_pass else
                               "tensile_tinygrad_same_kernel_causal_delta" if bb5a8_capture_pass else
                               "actual_tinygrad_prefill_kernel_disassembly" if bb5a8_static_mapping else
                               "tensile_tinygrad_mapping_probe" if bb5a7_blocked else
                               "performance_gate_blocked" if bb5a7_blocked else
                               "performance_gate" if bb5a6_pass else
                               "correctness_harness" if bb5a5_pass else
                               "resource_policy" if bb5a4_pass else
                               "allocator_resource_control" if bb5a3_pass else
                               "semantic_wait_scheduler_integration" if bb5a2_full_pass else
                               "real_postrange_renderer_lowering_integration" if bb5a2_layer3_pass else
                               "render_isa_evidence" if bb5a2_layer2_pass else
                               "postrange_rangeify_lds_lowering_hook" if bb5a2_layer1_pass else
                               "lds_stage_plan" if bb5a2_solution_scoped else
                               "real_double_buffered_lds_lowering" if bb5a_execution_blocked else
                               "double_buffered_lds_lowering" if bb5a_full_plan_ready or bb5a1_pass else
                               "pipeline_stage_schema" if bb5a1_scoped else
                               "pipeline_ir_surface" if bb5a_renderer_allocator.get("verdict") == "BB5A_SCOPE_COMPLETE_IMPLEMENTATION_NOT_READY" else
                               "real_renderer_allocator_integration" if software_pipeline.get("verdict") == "BLOCKED_REAL_RENDERER_ALLOCATOR_INTEGRATION" else
                               "q8_transfer" if software_pipeline.get("verdict") == "PASS_SOFTWARE_PIPELINE_TFLOPS" else
                               "software_pipelined_prefill_probe" if bb3_pass and bb4_pass else
                               "semantic_wait_scheduler_emitter" if bb2_pass else "schedule_metadata_ir",
      "minimum_pass": "time the captured 43.026 TFLOPS authority kernel and current P8 candidates under one common synchronized or device-timestamp harness before further scheduling/ILP work" if bb5a10_p8_timing_authority_reconciled else
                      "reconcile synchronized P8 timing against prior ~43 TFLOPS global-direct artifact and choose the valid timing authority before further scheduling work" if bb5a10_p8_global_direct_decision else
                      "decide whether to reopen a selected-compatible global-direct/IC-served WMMA candidate or prove why selected Tensile layout cannot transfer without LDS round-trip" if bb5a10_p8_bottleneck_classified else
                      "scope q8 downstream transfer with <=75us continuation gate and <=60us strong pass" if bb5a10_p8_pass else
                      "classify why converted ds_store_b64 macro candidate reaches only ~18.38 TFLOPS despite correctness and scratch/private 0" if bb5a10_p8_perf_blocked else
                      "same converted 128x128 ds_store_b64 macro candidate reaches >=60 TFLOPS on authority shape without scratch/private spill" if bb5a10_p8_tta3_pass else
                      "convert 128x128 macro helper cooperative LDS stores from ds_store_b128 to selected-compatible ds_store_b64, preserving grid=(96,4,1), ds_load_b128, WMMA, and scratch/private 0" if bb5a10_p8_tta3_blocked else
                      "promote from 16x16 correctness bridge to selected-compatible 128x128 macro tile with scratch/private 0 and resource metadata" if bb5a10_p8_tta2_pass else
                      "run full M=512,N=12288,K=4096 launch and verify deterministic sampled output tiles against numpy/reference slices" if bb5a10_p8_tta1_pass else
                      "extend P7d to gidx0/gidx1 grid=(768,32,1), one 16x16 tile per workgroup, full K=4096, RMSE <=1e-3 on deterministic subset" if bb5a10_p8_tta_completion_scoped or bb5a10_p8_tta_scoped else
                      "map the proven P7d K-loop into a full M=512,N=12288,K=4096 launch candidate before timing" if bb5a10_p8_blocked else
                      "pure tinygrad authority prefill reaches >=60 TFLOPS without scratch/private spill" if bb5a10_p7e_pass else
                      "correct executable candidate artifact records source/ISA/resource metadata and exact P8 timing command" if bb5a10_p7d_pass else
                      "authority-shape or tiled authority-subset correctness smoke passes without performance timing" if bb5a10_p7c_pass else
                      "selected-compatible ds_store_b64 -> ds_load_b128 -> WMMA tile returns rel RMSE <=0.05" if bb5a10_p7ab_pass else
                      "P7a known-good LDS WMMA smoke passes and P7b wraps the structural candidate with real kernargs, LDS allocation, lidx/gidx, and output store" if bb5a10_p7_scope_pass else
                      "small WMMA correctness and authority-shape correctness harness pass before any P8 timing" if bb5a10_p6_pass else
                      "P2/P3/P4/P5 pass together as one staged-LDS WMMA structural candidate" if bb5a10_p2_p5_pass else
                      "P2-P5 produce one structural candidate render/schedule/resource package ready for P6 structural gate" if bb5a10_p1_pass else
                      "P1-P5 produce one structural candidate plan/render/schedule/resource package ready for P6 structural gate" if bb5a10_plan_pass else
                      "list P0-P9 phases with blocked continuations and keep q8 transfer blocked until P8 performance passes" if bb5a10_pass else
                      "isolate selected Tensile function and prove whether candidate LDS layout implementation is ready or bitexact layout reconstruction is still blocked" if bb5a9_pass else
                      "compare captured tinygrad same-kernel instruction/resource mix against Tensile and classify proven causal gaps without starting q8 transfer" if bb5a8_capture_pass else
                      "capture source/ISA/resource evidence for the measured 42.0 TFLOPS pure-tinygrad prefill authority kernel and join it to timing" if bb5a8_static_mapping else
                      "static Tensile-to-tinygrad feature map complete; causal proof may remain blocked" if bb5a7_blocked else
                      "blocked: pure tinygrad authority prefill is 42.0 TFLOPS below the 60.0 TFLOPS gate" if bb5a7_blocked else
                      "pure tinygrad authority prefill reaches >=60 TFLOPS with real pipelined ISA" if bb5a6_pass else
                      "small WMMA and one authority prefill matmul correctness pass" if bb5a5_pass else
                      "policy selects or rejects a pipelined candidate with shape/resource reasons" if bb5a4_pass else
                      "candidate reports VGPR/SGPR/LDS/spill-risk/occupancy and is spill-free or deterministically rejected" if bb5a3_pass else
                      "dependency-aware waits attach to lowered WMMA prefill-shaped instruction stream" if bb5a2_full_pass else
                      "real lowering path consumes pipeline LDS plan and emits non-byte-identical pipelined AMD source/ISA" if bb5a2_layer3_pass else
                      "AMD render/assembly sees two-slot LDS structure and source/hash/ISA differs from serialized baseline" if bb5a2_layer2_pass else
                      "gated lowering preserves lds_slot=0/1 as durable DEFINE_LOCAL slots or non-foldable offsets through local-buffer cleanup" if bb5a2_layer1_pass else
                      "AMDLDSStagePlan maps pipeline lds_slot=0/1 to deterministic alias-safe LDS slots with required_local_bytes" if bb5a2_solution_scoped else
                      "postrange/renderer lowering maps lds_slot=0/1 to distinct LDS regions and emits non-byte-identical ISA" if bb5a_execution_blocked else
                      "two LDS stages lower from BB-5a.1 metadata into non-byte-identical AMD ISA without changing defaults" if bb5a_full_plan_ready or bb5a1_pass else
                      "AMDPipelineStageMeta schema plus read-only stage extraction serializes a two-stage WMMA prefill-shaped pipeline" if bb5a1_scoped else
                      "stage-aware pipeline IR survives lowering and metadata dumping for a WMMA prefill-shaped kernel" if bb5a_renderer_allocator.get("verdict") == "BB5A_SCOPE_COMPLETE_IMPLEMENTATION_NOT_READY" else
                      "implement true K-loop software-pipeline lowering plus allocator controls; rerun BB-5 for >=60 TFLOPS" if software_pipeline.get("verdict") == "BLOCKED_REAL_RENDERER_ALLOCATOR_INTEGRATION" else
                      "native q8 consumer reaches <=75us to continue" if software_pipeline.get("verdict") == "PASS_SOFTWARE_PIPELINE_TFLOPS" else
                      "pure tinygrad prefill probe reaches >=60 TFLOPS or formally blocks on missing real scheduler/allocator integration" if bb3_pass and bb4_pass else
                      "intended wait/s_clause/s_delay_alu ISA changes on q8-shaped and WMMA-shaped probes with correctness preserved" if bb2_pass else
                      "metadata describes one q8-shaped probe and one WMMA-shaped probe without semantic changes",
    },
  }

  write_json("authority.json", authority)
  write_json("oracle_suite.json", oracle_suite)
  write_json("schedule_metadata_ir_result.json", schedule_metadata)
  write_json("wait_scheduler_result.json", wait_scheduler)
  write_json("register_resource_result.json", register_resource)
  write_json("software_pipeline_result.json", software_pipeline)
  write_json("bb5a_renderer_allocator_scope.json", bb5a_renderer_allocator)
  write_json("bb5a1_pipeline_ir_scope.json", bb5a1_pipeline_ir_scope)
  write_json("bb5a1_pipeline_ir_result.json", bb5a1_pipeline_ir)
  write_json("bb5a_full_plan.json", bb5a_full_plan)
  write_json("bb5a_execution_result.json", bb5a_execution)
  write_json("bb5a2_solution_scope.json", bb5a2_solution)
  write_json("bb5a2_lds_stage_plan_result.json", bb5a2_lds_stage_plan)
  write_json("bb5a2_lowering_hook_result.json", bb5a2_lowering_hook)
  write_json("bb5a2_render_isa_evidence_result.json", bb5a2_render_isa)
  if bb5a2_integration: write_json("bb5a2_real_lowering_integration_result.json", bb5a2_integration)
  if bb5a2_dataflow: write_json("bb5a2_pipelined_dataflow_result.json", bb5a2_dataflow)
  if bb5a3_integration: write_json("bb5a3_wait_scheduler_integration_result.json", bb5a3_integration)
  if bb5a4_allocator: write_json("bb5a4_allocator_resource_result.json", bb5a4_allocator)
  if bb5a5_policy: write_json("bb5a5_resource_policy_result.json", bb5a5_policy)
  if bb5a6_correctness: write_json("bb5a6_correctness_result.json", bb5a6_correctness)
  if bb5a7_performance: write_json("bb5a7_performance_gate_result.json", bb5a7_performance)
  if bb5a8_mapping: write_json("bb5a8_tensile_mapping_result.json", bb5a8_mapping)
  if bb5a8_capture: write_json("bb5a8_authority_kernel_capture_result.json", bb5a8_capture)
  if bb5a9_causal_delta: write_json("bb5a9_causal_delta_package_result.json", bb5a9_causal_delta)
  if bb5a10_layout_audit: write_json("bb5a10_tensile_layout_audit_result.json", bb5a10_layout_audit)
  if bb5a10_implementation_plan: write_json("bb5a10_implementation_plan_result.json", bb5a10_implementation_plan)
  if bb5a10_p1_layout_spec: write_json("bb5a10_p1_layout_spec_result.json", bb5a10_p1_layout_spec)
  if bb5a10_p2_rendered_lds: write_json("bb5a10_p2_rendered_lds_result.json", bb5a10_p2_rendered_lds)
  if bb5a10_p3_kloop_stage: write_json("bb5a10_p3_kloop_stage_result.json", bb5a10_p3_kloop_stage)
  if bb5a10_p4_wait_barrier: write_json("bb5a10_p4_wait_barrier_result.json", bb5a10_p4_wait_barrier)
  if bb5a10_p5_resource_policy: write_json("bb5a10_p5_resource_policy_result.json", bb5a10_p5_resource_policy)
  if bb5a10_p6_structural: write_json("bb5a10_p6_structural_candidate_result.json", bb5a10_p6_structural)
  if bb5a10_p7_scope: write_json("bb5a10_p7_correctness_scope_result.json", bb5a10_p7_scope)
  if bb5a10_p7ab: write_json("bb5a10_p7a_p7b_correctness_result.json", bb5a10_p7ab)
  if bb5a10_p7c: write_json("bb5a10_p7c_numeric_correctness_result.json", bb5a10_p7c)
  if bb5a10_p7d: write_json("bb5a10_p7d_authority_correctness_result.json", bb5a10_p7d)
  if bb5a10_p7e: write_json("bb5a10_p7e_p8_handoff_result.json", bb5a10_p7e)
  if bb5a10_p8: write_json("bb5a10_p8_performance_result.json", bb5a10_p8)
  if bb5a10_p8_tta_scope: write_json("bb5a10_p8_tta_scope_result.json", bb5a10_p8_tta_scope)
  if bb5a10_p8_tta_completion: write_json("bb5a10_p8_tta_completion_scope_result.json", bb5a10_p8_tta_completion)
  if bb5a10_p8_tta1: write_json("bb5a10_p8_tta1_full_grid_correctness_result.json", bb5a10_p8_tta1)
  if bb5a10_p8_tta2: write_json("bb5a10_p8_tta2_authority_sample_correctness_result.json", bb5a10_p8_tta2)
  if bb5a10_p8_tta3: write_json("bb5a10_p8_tta3_macro_candidate_result.json", bb5a10_p8_tta3)
  if bb5a10_p8_tta3a: write_json("bb5a10_p8_tta3a_ds64_macro_conversion_result.json", bb5a10_p8_tta3a)
  if bb5a10_p8_bottleneck: write_json("bb5a10_p8_bottleneck_classification_result.json", bb5a10_p8_bottleneck)
  if bb5a10_p8_global_direct: write_json("bb5a10_p8_global_direct_candidate_decision_result.json", bb5a10_p8_global_direct)
  if bb5a10_p8_timing_authority: write_json("bb5a10_p8_timing_authority_reconciliation_result.json", bb5a10_p8_timing_authority)
  write_json("q8_transfer_result.json", q8_transfer)
  write_json("model_gate_result.json", model_gate)
  write_json("result.json", result)
  print(json.dumps({"out": str(OUT.relative_to(ROOT)), "verdict": result["verdict"], "next": result["next"]}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
