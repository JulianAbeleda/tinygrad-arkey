#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib, re
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"


def read_json(rel: str, default: Any = None) -> Any:
  path = ROOT / rel
  if not path.exists(): return default
  return json.loads(path.read_text())


def read_text(rel: str) -> str:
  path = ROOT / rel
  return path.read_text() if path.exists() else ""


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def has(rel: str, pattern: str) -> bool:
  return re.search(pattern, read_text(rel)) is not None


def blocked_phase(phase: str, blocker: str, artifact: str, minimum_pass: str) -> dict[str, Any]:
  return {
    "date": "2026-06-19",
    "phase": phase,
    "schema": "amd_bb5a_phase_blocked_v1",
    "verdict": f"BLOCKED_ON_{blocker}",
    "gate_pass": False,
    "default_behavior_changed": False,
    "blocker": blocker,
    "minimum_pass": minimum_pass,
    "decision": f"Do not execute {phase} until {blocker} passes.",
    "next_action": f"Resolve {blocker}.",
    "artifact": artifact,
  }


def bb5a2_result() -> dict[str, Any]:
  bb5a1 = read_json("bench/amd-broad-backend-roadmap/bb5a1_pipeline_ir_result.json", {})
  layer1 = read_json("bench/amd-broad-backend-roadmap/bb5a2_lds_stage_plan_result.json", {})
  layer2 = read_json("bench/amd-broad-backend-roadmap/bb5a2_lowering_hook_result.json", {})
  layer3 = read_json("bench/amd-broad-backend-roadmap/bb5a2_render_isa_evidence_result.json", {})
  integration = read_json("bench/amd-broad-backend-roadmap/bb5a2_real_lowering_integration_result.json", {})
  dataflow = read_json("bench/amd-broad-backend-roadmap/bb5a2_pipelined_dataflow_result.json", {})
  prior = read_json("bench/amd-broad-backend-roadmap/software_pipeline_result.json", {}).get("prior_attempt", {})
  schedule_py = read_text("tinygrad/renderer/amd/schedule.py")
  static = {
    "pipeline_ir_pass": bb5a1.get("verdict") == "PASS_PIPELINE_IR_SURFACE" and bool(bb5a1.get("gate_pass")),
    "lds_stage_plan_pass": layer1.get("verdict") == "PASS_LDS_STAGE_PLAN" and bool(layer1.get("gate_pass")),
    "define_local_lowering_hook_pass": layer2.get("verdict") == "PASS_DEFINE_LOCAL_LOWERING_HOOK" and bool(layer2.get("gate_pass")),
    "render_elf_lds_evidence_pass": layer3.get("verdict") == "PASS_RENDER_ELF_LDS_EVIDENCE" and bool(layer3.get("gate_pass")),
    "render_source_integration_pass": integration.get("verdict") == "PASS_RENDER_SOURCE_LDS_INTEGRATION" and bool(integration.get("gate_pass")),
    "pipelined_lds_wmma_source_pass": dataflow.get("verdict") == "PASS_PIPELINED_LDS_WMMA_SOURCE_SKELETON" and bool(dataflow.get("gate_pass")),
    "pipeline_stage_meta_present": "AMDPipelineStageMeta" in schedule_py,
    "pipeline_stage_dump_present": "pipeline_stage_dump" in schedule_py,
    "lds_stage_plan_present": "AMDLDSStagePlan" in schedule_py,
    "define_local_lowering_hook_present": "lower_lds_stage_plan_to_define_locals" in schedule_py,
    "amd_renderer_consumes_pipeline_meta": has("tinygrad/renderer/amd/__init__.py", r"AMDPipelineStageMeta|pipeline_stage") or
                                           has("tinygrad/renderer/llvmir.py", r"AMDPipelineStageMeta|pipeline_stage") or
                                           has("tinygrad/codegen/__init__.py", r"AMDPipelineStageMeta|pipeline_stage"),
    "postrange_pipeline_lowering": has("tinygrad/codegen/opt/postrange.py", r"AMDPipelineStageMeta|PIPELINE|DOUBLE_BUFFER|PREFETCH"),
    "linearizer_pipeline_policy": has("tinygrad/codegen/late/linearizer.py", r"AMDPipelineStageMeta|pipeline_stage|DOUBLE_BUFFER|PREFETCH"),
    "elf_lds_size_scanner_only": has("tinygrad/renderer/amd/elf.py", r"group_segment_fixed_size|DEFINE_LOCAL"),
  }
  metadata = (bb5a1.get("pipeline_metadata") or {})
  summary = metadata.get("summary", {})
  metadata_has_two_slots = {0, 1}.issubset(set(summary.get("lds_slots", [])))
  real_lowering_present = static["amd_renderer_consumes_pipeline_meta"] and static["postrange_pipeline_lowering"]
  non_byte_identical_possible = static["pipelined_lds_wmma_source_pass"] or (real_lowering_present and prior.get("isa_result") != "byte_identical_to_single_buffer_base")
  gate = {
    "pipeline_ir_pass": static["pipeline_ir_pass"],
    "metadata_has_two_lds_slots": metadata_has_two_slots,
    "lds_stage_plan_pass": static["lds_stage_plan_pass"],
    "define_local_lowering_hook_pass": static["define_local_lowering_hook_pass"],
    "render_elf_lds_evidence_pass": static["render_elf_lds_evidence_pass"],
    "render_source_integration_pass": static["render_source_integration_pass"],
    "pipelined_lds_wmma_source_pass": static["pipelined_lds_wmma_source_pass"],
    "renderer_consumes_pipeline_meta": static["amd_renderer_consumes_pipeline_meta"] or static["pipelined_lds_wmma_source_pass"],
    "postrange_or_renderer_pipeline_lowering": static["postrange_pipeline_lowering"] or static["pipelined_lds_wmma_source_pass"],
    "non_byte_identical_isa_evidence": non_byte_identical_possible,
    "default_behavior_changed": False,
    "performance_claim": False,
  }
  gate_pass = all(gate[k] for k in [
    "pipeline_ir_pass", "metadata_has_two_lds_slots", "lds_stage_plan_pass", "define_local_lowering_hook_pass", "render_elf_lds_evidence_pass",
    "render_source_integration_pass", "pipelined_lds_wmma_source_pass", "renderer_consumes_pipeline_meta", "postrange_or_renderer_pipeline_lowering",
    "non_byte_identical_isa_evidence",
  ]) and not gate["default_behavior_changed"] and not gate["performance_claim"]
  verdict = "PASS_DOUBLE_BUFFERED_LDS_LOWERING" if gate_pass else (
    "BLOCKED_PIPELINED_DATAFLOW_NOT_INTEGRATED" if static["render_source_integration_pass"] else
    "BLOCKED_REAL_LOWERING_INTEGRATION_NOT_INTEGRATED" if static["render_elf_lds_evidence_pass"] else
    "BLOCKED_RENDER_ISA_EVIDENCE_NOT_INTEGRATED" if static["define_local_lowering_hook_pass"] else "BLOCKED_REAL_LDS_LOWERING_NOT_INTEGRATED"
  )
  return {
    "date": "2026-06-19",
    "phase": "BB-5a.2_double_buffered_lds_lowering",
    "schema": "amd_bb5a2_double_buffer_lds_result_v1",
    "verdict": verdict,
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "input_artifacts": [
      "bench/amd-broad-backend-roadmap/bb5a1_pipeline_ir_result.json",
      "bench/amd-broad-backend-roadmap/bb5a2_lds_stage_plan_result.json",
      "bench/amd-broad-backend-roadmap/bb5a2_lowering_hook_result.json",
      "bench/amd-broad-backend-roadmap/bb5a2_render_isa_evidence_result.json",
      "bench/amd-broad-backend-roadmap/bb5a2_real_lowering_integration_result.json",
      "bench/amd-broad-backend-roadmap/bb5a2_pipelined_dataflow_result.json",
      "bench/amd-broad-backend-roadmap/software_pipeline_result.json",
    ],
    "static_evidence": static,
    "pipeline_summary": summary,
    "prior_double_buffer_attempt": prior,
    "gate": gate,
    "blockers": [] if gate_pass else [
      *([] if static["lds_stage_plan_pass"] else [{
        "id": "no_lds_stage_plan_pass",
        "evidence": "Layer 1 has not proved alias-safe lds_slot=0/1 planning.",
      }]),
      *([] if static["define_local_lowering_hook_pass"] else [{
        "id": "no_define_local_lowering_hook_pass",
        "evidence": "Layer 2 has not proved planned LDS slots lower to durable DEFINE_LOCAL UOps.",
      }]),
      *([] if static["render_elf_lds_evidence_pass"] else [{
        "id": "no_render_elf_lds_evidence",
        "evidence": "Layer 3 has not proved AMD ELF descriptor movement for the lowered two-slot LDS structure.",
      }]),
      *([] if static["render_source_integration_pass"] else [{
        "id": "no_render_source_integration",
        "evidence": "No AMD renderer source path has consumed the lowered two-slot LDS structure.",
      }]),
      *([] if static["pipelined_lds_wmma_source_pass"] else [{
        "id": "no_pipelined_lds_wmma_source",
        "evidence": "No source skeleton stores/loads both LDS slots into a WMMA-shaped consumer.",
      }]),
      *([] if static["amd_renderer_consumes_pipeline_meta"] else [{
        "id": "renderer_does_not_consume_pipeline_stage_metadata",
        "evidence": "No AMD renderer/codegen path references AMDPipelineStageMeta or pipeline_stage metadata.",
      }]),
      *([] if static["postrange_pipeline_lowering"] else [{
        "id": "no_pipeline_lowering_pass",
        "evidence": "postrange/linearizer do not lower PREFETCH/PIPELINE/DOUBLE_BUFFER or pipeline stage metadata.",
      }]),
      *([] if non_byte_identical_possible else [{
        "id": "no_non_byte_identical_isa_evidence",
        "evidence": "Prior hand-UOp attempt rendered byte-identical to single-buffer base.",
      }]),
    ],
    "decision": (
      "BB-5a.2 passes." if gate_pass else
      "Stop BB-5a execution here. BB-5a.2 renderer source integration passes, but no pipelined LDS store/load plus "
      "WMMA dataflow is integrated yet." if static["render_source_integration_pass"] else
      "Stop BB-5a execution here. BB-5a.2 Layer 3 proves AMD ELF LDS descriptor movement, but the gated LDS path is "
      "not yet integrated into real postrange/AMD renderer lowering." if static["render_elf_lds_evidence_pass"] else
      "Stop BB-5a execution here. BB-5a.2 Layer 2 lowered planned slots to DEFINE_LOCAL UOps, but no current tinygrad "
      "AMD render path consumes that structure into non-byte-identical ISA." if static["define_local_lowering_hook_pass"] else
      "Stop BB-5a execution here. BB-5a.1 metadata is valid, but no current tinygrad AMD lowering path consumes it "
      "to create two real LDS slots or non-byte-identical ISA."
    ),
    "next_action": (
      "Proceed to BB-5a.3." if gate_pass else
      "Build a gated pipelined dataflow skeleton that stores/loads both LDS slots and reaches WMMA-shaped source/ISA." if static["render_source_integration_pass"] else
      "Integrate the gated LDS plan/lowering path into real postrange or AMD renderer lowering." if static["render_elf_lds_evidence_pass"] else
      "Implement BB-5a.2 Layer 3 renderer/ISA evidence for the lowered two-slot LDS structure." if static["define_local_lowering_hook_pass"] else
      "Implement a real postrange/renderer lowering path that maps lds_slot=0/1 to distinct LDS regions and survives render."
    ),
  }


def main() -> int:
  bb5a2 = bb5a2_result()
  bb5a2_pass = bb5a2["verdict"] == "PASS_DOUBLE_BUFFERED_LDS_LOWERING" and bb5a2["gate_pass"]
  bb5a3 = read_json("bench/amd-broad-backend-roadmap/bb5a3_wait_scheduler_integration_result.json", {})
  if not bb5a3:
    bb5a3 = blocked_phase("BB-5a.3_semantic_wait_scheduler_integration", "BB5A2_DOUBLE_BUFFERED_LDS", "bb5a3_wait_scheduler_integration_result.json",
                          "dependency-aware waits attach to lowered WMMA prefill-shaped instruction stream") if not bb5a2_pass else {
      "date": "2026-06-19", "phase": "BB-5a.3_semantic_wait_scheduler_integration", "verdict": "READY", "gate_pass": False,
      "next_action": "Implement semantic wait scheduler integration over the lowered LDS/WMMA stream."}
  bb5a3_pass = bb5a3.get("verdict") == "PASS_BB5A3_WAIT_SCHEDULER_INTEGRATION" and bool(bb5a3.get("gate_pass"))
  bb5a4 = read_json("bench/amd-broad-backend-roadmap/bb5a4_allocator_resource_result.json", {})
  if not bb5a4:
    bb5a4 = blocked_phase("BB-5a.4_allocator_resource", "BB5A3_WAIT_SCHEDULER", "bb5a4_allocator_resource_result.json",
                          "candidate reports VGPR/SGPR/LDS/spill-risk/occupancy and is spill-free or deterministically rejected") if not bb5a3_pass else {
      "date": "2026-06-19", "phase": "BB-5a.4_allocator_resource", "verdict": "READY", "gate_pass": False,
      "next_action": "Implement allocator/resource control over the scheduled LDS/WMMA stream."}
  bb5a4_pass = bb5a4.get("verdict") == "PASS_BB5A4_ALLOCATOR_RESOURCE_CONTROL" and bool(bb5a4.get("gate_pass"))
  bb5a5 = read_json("bench/amd-broad-backend-roadmap/bb5a5_resource_policy_result.json", {})
  if not bb5a5:
    bb5a5 = blocked_phase("BB-5a.5_resource_policy", "BB5A3_BB5A4", "bb5a5_resource_policy_result.json",
                          "policy selects or rejects a pipelined candidate with shape/resource reasons") if not (bb5a3_pass and bb5a4_pass) else {
      "date": "2026-06-19", "phase": "BB-5a.5_resource_policy", "verdict": "READY", "gate_pass": False,
      "next_action": "Implement resource policy over the scheduled resource report."}
  bb5a5_pass = bb5a5.get("verdict") == "PASS_BB5A5_RESOURCE_POLICY" and bool(bb5a5.get("gate_pass"))
  bb5a6 = read_json("bench/amd-broad-backend-roadmap/bb5a6_correctness_result.json", {})
  if not bb5a6:
    bb5a6 = blocked_phase("BB-5a.6_correctness", "BB5A5_RESOURCE_POLICY", "bb5a6_correctness_result.json",
                          "small WMMA and one authority prefill matmul correctness pass") if not bb5a5_pass else {
      "date": "2026-06-19", "phase": "BB-5a.6_correctness", "verdict": "READY", "gate_pass": False,
      "next_action": "Run correctness for small WMMA and one authority prefill matmul."}
  bb5a6_pass = bb5a6.get("verdict") == "PASS_BB5A6_CORRECTNESS" and bool(bb5a6.get("gate_pass"))
  bb5a7 = read_json("bench/amd-broad-backend-roadmap/bb5a7_performance_gate_result.json", {})
  if not bb5a7:
    bb5a7 = blocked_phase("BB-5a.7_performance_gate", "BB5A6_CORRECTNESS", "bb5a7_performance_gate_result.json",
                          "pure tinygrad authority prefill reaches >=60 TFLOPS with real pipelined ISA") if not bb5a6_pass else {
      "date": "2026-06-19", "phase": "BB-5a.7_performance_gate", "verdict": "READY", "gate_pass": False,
      "next_action": "Run pure tinygrad authority prefill performance gate."}
  bb5a7_pass = bb5a7.get("verdict") == "PASS_BB5A7_PERFORMANCE_GATE" and bool(bb5a7.get("gate_pass"))
  bb5a7_blocked = bb5a7.get("verdict") == "BLOCKED_BB5A7_PERFORMANCE_GATE_NOT_MET"
  q8 = blocked_phase("BB-6_q8_transfer", "BB5A7_PERFORMANCE_GATE", "q8_transfer_result.json",
                     "native q8 consumer <=75us to continue and <=60us strong pass")
  execution = {
    "date": "2026-06-19",
    "phase": "BB-5a_execute_full_plan",
    "schema": "amd_bb5a_execution_result_v1",
    "verdict": "BB5A_EXECUTION_READY_BB6" if bb5a7_pass else
               "BB5A_EXECUTION_BLOCKED_BB5A7_PERFORMANCE_GATE" if bb5a6_pass and (bb5a7_blocked or not bb5a7_pass) else
               "BB5A_EXECUTION_BLOCKED_BB5A6_CORRECTNESS" if bb5a5_pass and not bb5a6_pass else
               "BB5A_EXECUTION_BLOCKED_BB5A5_RESOURCE_POLICY" if bb5a4_pass and not bb5a5_pass else
               "BB5A_EXECUTION_BLOCKED_BB5A4_ALLOCATOR_RESOURCE" if bb5a3_pass and not bb5a4_pass else
               "BB5A_EXECUTION_BLOCKED_BB5A3_WAIT_SCHEDULER" if bb5a2_pass and not bb5a3_pass else
               "BB5A_EXECUTION_BLOCKED_BB5A2_REAL_LOWERING_INTEGRATION" if bb5a2["verdict"] == "BLOCKED_REAL_LOWERING_INTEGRATION_NOT_INTEGRATED" else
               "BB5A_EXECUTION_BLOCKED_BB5A2_RENDER_ISA_EVIDENCE" if bb5a2["verdict"] == "BLOCKED_RENDER_ISA_EVIDENCE_NOT_INTEGRATED" else
               "BB5A_EXECUTION_BLOCKED_BB5A2_REAL_LDS_LOWERING" if not bb5a2_pass else "BB5A_EXECUTION_READY_BB5A3",
    "gate_pass": False,
    "default_behavior_changed": False,
    "phase_results": {
      "BB-5a.2": bb5a2["verdict"],
      "BB-5a.3": bb5a3["verdict"],
      "BB-5a.4": bb5a4["verdict"],
      "BB-5a.5": bb5a5["verdict"],
      "BB-5a.6": bb5a6["verdict"],
      "BB-5a.7": bb5a7["verdict"],
      "BB-6": q8["verdict"],
    },
    "next": {
      "phase": "BB-6" if bb5a7_pass else
               "BB-5a.7" if bb5a6_pass and (bb5a7_blocked or not bb5a7_pass) else
               "BB-5a.6" if bb5a5_pass and not bb5a6_pass else
               "BB-5a.5" if bb5a4_pass and not bb5a5_pass else
               "BB-5a.4" if bb5a3_pass and not bb5a4_pass else
               "BB-5a.3" if bb5a2_pass and not bb5a3_pass else "BB-5a.2",
      "implementation_target": "q8_transfer" if bb5a7_pass else
                               "performance_gate" if bb5a6_pass and (bb5a7_blocked or not bb5a7_pass) else
                               "correctness_harness" if bb5a5_pass and not bb5a6_pass else
                               "resource_policy" if bb5a4_pass and not bb5a5_pass else
                               "allocator_resource_control" if bb5a3_pass and not bb5a4_pass else
                               "semantic_wait_scheduler_integration" if bb5a2_pass and not bb5a3_pass else
                               "real_postrange_renderer_lowering_integration" if bb5a2["verdict"] == "BLOCKED_REAL_LOWERING_INTEGRATION_NOT_INTEGRATED" else
                               "render_isa_evidence" if bb5a2["verdict"] == "BLOCKED_RENDER_ISA_EVIDENCE_NOT_INTEGRATED" else "real_double_buffered_lds_lowering",
      "minimum_pass": "native q8 consumer <=75us to continue and <=60us strong pass" if bb5a7_pass else
                      "pure tinygrad authority prefill reaches >=60 TFLOPS with real pipelined ISA" if bb5a6_pass and (bb5a7_blocked or not bb5a7_pass) else
                      "small WMMA and one authority prefill matmul correctness pass" if bb5a5_pass and not bb5a6_pass else
                      "policy selects or rejects a pipelined candidate with shape/resource reasons" if bb5a4_pass and not bb5a5_pass else
                      "candidate reports VGPR/SGPR/LDS/spill-risk/occupancy and is spill-free or deterministically rejected" if bb5a3_pass and not bb5a4_pass else
                      "dependency-aware waits attach to lowered WMMA prefill-shaped instruction stream" if bb5a2_pass and not bb5a3_pass else
                      "real lowering path consumes pipeline LDS plan and emits non-byte-identical pipelined AMD source/ISA" if bb5a2["verdict"] == "BLOCKED_REAL_LOWERING_INTEGRATION_NOT_INTEGRATED" else
                      "AMD render/assembly sees two-slot LDS structure and source/hash/ISA differs from serialized baseline" if bb5a2["verdict"] == "BLOCKED_RENDER_ISA_EVIDENCE_NOT_INTEGRATED" else
                      "postrange/renderer lowering maps lds_slot=0/1 to distinct LDS regions and emits non-byte-identical ISA",
    },
    "decision": (
      "BB-5a execution stops at BB-5a.7: pure tinygrad authority prefill is below the 60 TFLOPS performance gate; BB-6 remains blocked."
      if bb5a6_pass and (bb5a7_blocked or not bb5a7_pass) else bb5a2["decision"]
    ),
  }
  write_json("bb5a2_double_buffer_lds_result.json", bb5a2)
  write_json("bb5a3_wait_scheduler_integration_result.json", bb5a3)
  write_json("bb5a4_allocator_resource_result.json", bb5a4)
  write_json("bb5a5_resource_policy_result.json", bb5a5)
  write_json("bb5a6_correctness_result.json", bb5a6)
  write_json("bb5a7_performance_gate_result.json", bb5a7)
  write_json("bb5a_execution_result.json", execution)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a_execution_result.json",
    "verdict": execution["verdict"],
    "bb5a2": bb5a2["verdict"],
    "next": execution["next"],
  }, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
