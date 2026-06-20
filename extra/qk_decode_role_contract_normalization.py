#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer"

TRANSFER = "bench/qk-decode-primitive-transfer/decode_primitive_transfer_result.json"
SCHEDULE_OBJECT = "bench/qk-decode-primitive-transfer/decode_mmvq_schedule_object_result.json"
READINESS = "bench/qk-decode-native-tooling/readiness.json"
P5 = "bench/qk-decode-mmvq-large-project/p5_lifecycle_probe.json"
P6 = "bench/qk-decode-mmvq-large-project/p6_q4_shape_matrix.json"
Q8_PROMOTION = "bench/q8-ffn-artifact-promotion/promotion_result.json"
LLAMA_RECONCILIATION = "bench/qk-llama-promotion/reconciliation.json"


def read_json(rel: str, default: Any = None) -> Any:
  path = ROOT / rel
  if not path.exists(): return default
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def q4_matrix_row(p6: dict[str, Any], tensor: str) -> dict[str, Any] | None:
  for row in p6.get("rows", []):
    if row.get("tensor") == tensor: return row
  return None


def decode_promotion_row(recon: dict[str, Any], candidate: str) -> dict[str, Any] | None:
  for row in recon.get("promotion_rows", []):
    if row.get("phase") == "decode" and row.get("candidate") == candidate: return row
  return None


def timing_ref(row: dict[str, Any] | None, field: str, default: Any = None) -> Any:
  if not row: return default
  return row.get(field, default)


def main() -> int:
  transfer = read_json(TRANSFER, {})
  sched = read_json(SCHEDULE_OBJECT, {})
  readiness = read_json(READINESS, {})
  p5 = read_json(P5, {})
  p6 = read_json(P6, {})
  q8 = read_json(Q8_PROMOTION, {})
  recon = read_json(LLAMA_RECONCILIATION, {})

  p6_attn_o = q4_matrix_row(p6, "blk.0.attn_output.weight")
  p6_gate = q4_matrix_row(p6, "blk.0.ffn_gate.weight")
  p6_up = q4_matrix_row(p6, "blk.0.ffn_up.weight")
  banked_default = decode_promotion_row(recon, "banked_default_decode_stack")
  q8_row = decode_promotion_row(recon, "q8_ffn_handwritten_artifact_route")
  native_q8_row = decode_promotion_row(recon, "native_q8_scheduler_renderer")
  mmvq_row = decode_promotion_row(recon, "mmvq_contract_preservation_or_source_import")

  role_contracts = [
    {
      "id": "decode_default_stack",
      "role_group": "full_decode",
      "quant_format": "mixed Q4_K/Q6_K",
      "shape": None,
      "llama_contract": {
        "reference": "llama-relative W==D decode authority",
        "relative": timing_ref(banked_default, "llama_relative"),
      },
      "tinygrad_contract": {
        "route": "promoted default decode stack",
        "status": "PROMOTED_DEFAULT",
        "default_on": True,
        "decode_tok_s": "68.2/66.4/60.7 at ctx512/1024/4096",
      },
      "lifecycle": "current Q4/Q6 primitives plus flash-decode stack",
      "launch_resource": {"authority": "W==D model-level, not per-kernel"},
      "timing_quality": {"authority": "W==D", "performance_claim": True},
      "quality": {"byte_identical_or_default": True},
      "ownership": "tinygrad default",
      "decision": "keep_as_authority_baseline",
      "native_renderer_start": False,
    },
    {
      "id": "q4_attn_output_imported_llama",
      "role_group": "attn_output",
      "quant_format": "Q4_K x q8_1",
      "shape": [4096, 4096],
      "llama_contract": {
        "source": "ggml-cuda/mmvq.cu: mul_mat_vec_q + vecdotq.cuh: vec_dot_q4_K_q8_1",
        "batch_max": 8,
        "q8_producer": "block_q8_1",
      },
      "tinygrad_contract": {
        "route": "source-import lifecycle probe",
        "status": "PASS_DEVICE_LIFECYCLE",
        "default_on": False,
      },
      "lifecycle": {
        "producer": "q8_quantize_4096",
        "producer_byte_exact": (p5.get("producer") or {}).get("byte_match_cpu_q8_blocks"),
        "consumer": "imported llama Q4_K MMVQ",
      },
      "launch_resource": (p5.get("consumer") or {}).get("launch"),
      "timing_quality": {
        "authority": "device_lifecycle_probe_not_WD",
        "producer_device_ms": (p5.get("timing") or {}).get("producer_device_ms_median"),
        "consumer_device_ms": (p5.get("timing") or {}).get("consumer_device_ms_median"),
        "candidate_device_ms_sum": (p5.get("timing") or {}).get("candidate_device_ms_sum"),
        "pct_hbm": (p5.get("timing") or {}).get("candidate_lifecycle_pct_hbm"),
      },
      "quality": (p5.get("correctness") or {}),
      "ownership": "external llama source/object import",
      "decision": "continue_graph_safe_source_import_track",
      "native_renderer_start": False,
    },
    {
      "id": "q4_ffn_gate_imported_llama",
      "role_group": "ffn_gate",
      "quant_format": "Q4_K x q8_1",
      "shape": [12288, 4096],
      "llama_contract": {
        "source": "vec_dot_q4_K_q8_1_impl_vmmq",
        "vdr": 2,
        "inner_loop": "packed nibble extract + sdot4 + q8 sum/min correction + per-group scale",
      },
      "tinygrad_contract": {
        "route": "imported Q4 shape matrix, not graph-safe default",
        "status": "PASS_Q4_MATRIX" if (p6.get("verdict") == "PASS_Q4_MATRIX" and p6_gate and p6_gate.get("correct")) else "UNKNOWN",
        "default_on": False,
      },
      "lifecycle": "explicit q8 producer still graph-route gated; consumer shape proven",
      "launch_resource": (p6_gate or {}).get("launch"),
      "timing_quality": {
        "authority": "standalone_imported_consumer_device_time",
        "device_ms": (p6_gate or {}).get("device_ms_per_launch"),
        "q4_gbs": (p6_gate or {}).get("q4_gbs"),
        "pct_hbm_effective": (p6_gate or {}).get("pct_hbm"),
      },
      "quality": {"consumer_correct": bool(p6_gate and p6_gate.get("correct")), "max_abs": (p6_gate or {}).get("max_abs")},
      "ownership": "external llama source/object import",
      "decision": "candidate_for_graph_safe_Q4_route",
      "native_renderer_start": False,
    },
    {
      "id": "q4_ffn_up_imported_llama",
      "role_group": "ffn_up",
      "quant_format": "Q4_K x q8_1",
      "shape": [12288, 4096],
      "llama_contract": {
        "source": "vec_dot_q4_K_q8_1_impl_vmmq",
        "vdr": 2,
        "inner_loop": "packed nibble extract + sdot4 + q8 sum/min correction + per-group scale",
      },
      "tinygrad_contract": {
        "route": "imported Q4 shape matrix, not graph-safe default",
        "status": "PASS_Q4_MATRIX" if (p6.get("verdict") == "PASS_Q4_MATRIX" and p6_up and p6_up.get("correct")) else "UNKNOWN",
        "default_on": False,
      },
      "lifecycle": "explicit q8 producer still graph-route gated; consumer shape proven",
      "launch_resource": (p6_up or {}).get("launch"),
      "timing_quality": {
        "authority": "standalone_imported_consumer_device_time",
        "device_ms": (p6_up or {}).get("device_ms_per_launch"),
        "q4_gbs": (p6_up or {}).get("q4_gbs"),
        "pct_hbm_effective": (p6_up or {}).get("pct_hbm"),
      },
      "quality": {"consumer_correct": bool(p6_up and p6_up.get("correct")), "max_abs": (p6_up or {}).get("max_abs")},
      "ownership": "external llama source/object import",
      "decision": "candidate_for_graph_safe_Q4_route",
      "native_renderer_start": False,
    },
    {
      "id": "q6_selected_roles_default",
      "role_group": "ffn_down,lm_head",
      "quant_format": "Q6_K",
      "shape": "ffn_down 4096x12288; lm_head model-vocab dependent",
      "llama_contract": {
        "source": "vecdotq.cuh: vec_dot_q6_K_q8_1_impl_mmvq",
        "vdr": 1,
      },
      "tinygrad_contract": {
        "route": "Q6KPrimitiveLinear cooperative/default role policy",
        "status": "promoted for selected roles",
        "default_on": True,
      },
      "lifecycle": "fp16 activation path; Q6 cooperative pos-lane partials + sum",
      "launch_resource": {"known_gap": "full Q6 imported/source parity still open"},
      "timing_quality": {"authority": "banked decode stack, not isolated source-import parity"},
      "quality": {"exact_dequant_route": True},
      "ownership": "tinygrad custom kernels",
      "decision": "keep_default_continue_Q6_source_import_coverage",
      "native_renderer_start": False,
    },
    {
      "id": "q8_ffn_gateup_artifact",
      "role_group": "ffn_gate/up",
      "quant_format": "Q8_FFN_ARTIFACT",
      "shape": [12288, 4096],
      "llama_contract": {
        "analogue": "q8 activation lifecycle + packed Q4_K dot consumer",
        "native_oracle": "hipcc/lld artifact, not llama default route",
      },
      "tinygrad_contract": {
        "route": "Q8_FFN_HANDWRITTEN=1",
        "status": q8.get("verdict"),
        "default_on": False,
      },
      "lifecycle": "fused RMSNorm/q8 producer plus fused Q4_K x q8 gate/up consumer",
      "launch_resource": {"coverage": "Qwen3-8B Q4_K_M dim=4096 hidden=12288 gfx1100"},
      "timing_quality": {
        "authority": "W==D opt-in promotion matrix",
        "min_speedup": ((q8.get("summary") or {}).get("performance") or {}).get("min_speedup"),
        "llama_relative": timing_ref(q8_row, "llama_relative"),
      },
      "quality": ((q8.get("summary") or {}).get("quality") or {}),
      "ownership": "external artifact, hardened opt-in",
      "decision": "keep_hardened_opt_in_default_off",
      "native_renderer_start": False,
    },
    {
      "id": "native_q8_scheduler_renderer",
      "role_group": "ffn_gate/up",
      "quant_format": "native Q4_K/Q8_1 scheduler target",
      "shape": [12288, 4096],
      "llama_contract": {
        "target": "preserve MMVQ/q8 lifecycle and low-resource packed-dot behavior",
      },
      "tinygrad_contract": {
        "route": "not implemented",
        "status": "ROADMAP_ONLY",
        "default_on": False,
      },
      "lifecycle": "would replace/import q8 artifact/source contract with native-owned schedule",
      "launch_resource": {"start_gate": readiness.get("start_gate")},
      "timing_quality": {
        "authority": "native tooling readiness",
        "max_timing_grade_movement_us": (readiness.get("start_gate") or {}).get("max_timing_grade_movement_us"),
        "required_movement_us": 30,
        "llama_relative": timing_ref(native_q8_row, "llama_relative"),
      },
      "quality": {"not_applicable_until_route_exists": True},
      "ownership": "would be tinygrad native renderer",
      "decision": "blocked_no_N2_feature",
      "native_renderer_start": False,
    },
    {
      "id": "mmvq_contract_project_option",
      "role_group": "all_high_share_weight_gemv",
      "quant_format": "Q4_K/Q6_K MMVQ",
      "shape": "role matrix",
      "llama_contract": {
        "target": "source-contract import or native contract preservation",
        "potential": timing_ref(mmvq_row, "llama_relative"),
      },
      "tinygrad_contract": {
        "route": "project-level option",
        "status": "not started as default/native",
        "default_on": False,
      },
      "lifecycle": "unify activation format, packed weight load, dot, reduction, route policy",
      "launch_resource": {"source_import": "Q4 proven; Q6 open"},
      "timing_quality": {"authority": "project estimate + existing probes"},
      "quality": {"requires_WD_and_dNLL_if_q8_used": True},
      "ownership": "choice: source import vs tinygrad native",
      "decision": "funded_project_only",
      "native_renderer_start": False,
    },
  ]

  gates = {
    "transfer_passed": transfer.get("gate_pass") is True,
    "schedule_object_passed": sched.get("gate_pass") is True,
    "readiness_roadmap_only": readiness.get("verdict") in {"ROADMAP_ONLY", "TOOLING_NOT_READY"},
    "q4_import_rows_present": all(x is not None for x in (p6_attn_o, p6_gate, p6_up)),
    "q8_promotion_present": q8.get("verdict") == "PASS_Q8_FFN_ARTIFACT_PROMOTION_TO_HARDENED_OPT_IN",
    "has_default_decode_authority": banked_default is not None,
    "all_rows_have_decisions": all(bool(r.get("decision")) for r in role_contracts),
    "native_renderer_not_started": not any(r.get("native_renderer_start") for r in role_contracts),
  }
  verdict = "PASS_DECODE_ROLE_CONTRACT_NORMALIZATION_NATIVE_BLOCKED" if all(gates.values()) else "BLOCKED_DECODE_ROLE_CONTRACT_NORMALIZATION"
  native_start_blockers = [
    "max timing-grade native feature movement is below 30us",
    "Q4 source-import path still needs graph-safe route before default consideration",
    "Q6 source/import parity coverage remains open",
    "q8 artifact is hardened opt-in and externally owned; it is not native renderer ownership",
  ]
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_ROLE_CONTRACT_NORMALIZATION",
    "schema": "decode_role_contract_normalization_v1",
    "verdict": verdict,
    "gate_pass": all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "role_contracts": role_contracts,
    "gates": gates,
    "native_renderer_start_allowed": False,
    "native_start_blockers": native_start_blockers,
    "decisions": {
      "default_decode": "keep promoted default",
      "q4_source_import": "next concrete implementation track is graph-safe route, not more Q4 kernel work",
      "q6_source_import": "parallel coverage track",
      "q8_artifact": "keep hardened opt-in default-off",
      "native_renderer": "blocked until attribution gate clears or broad backend work is explicitly accepted",
    },
    "input_artifacts": [TRANSFER, SCHEDULE_OBJECT, READINESS, P5, P6, Q8_PROMOTION, LLAMA_RECONCILIATION],
    "next_action": "DPT-4 graph-safe Q4 source-import route scope/probe, with Q6 coverage as a parallel track; do not start native renderer or BEAM/search.",
  }
  write_json("decode_role_contract_normalization_result.json", result)
  print(json.dumps({
    "out": str(OUT / "decode_role_contract_normalization_result.json"),
    "verdict": verdict,
    "gate_pass": result["gate_pass"],
    "row_count": len(role_contracts),
    "native_renderer_start_allowed": False,
    "next_action": result["next_action"],
    "failed_gates": [k for k, v in gates.items() if not v],
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
