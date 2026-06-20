#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3_scheduler_resource_scope_result.json"


def read_json(rel: str) -> dict[str, Any]:
  with (ROOT / rel).open() as f:
    return json.load(f)


def top_count(rows: list[list[Any]], name: str) -> int:
  for k, v in rows:
    if k == name: return int(v)
  return 0


def main() -> int:
  dnr2 = read_json("bench/qk-decode-primitive-transfer/decode_native_renderer_dnr2_lowering_result.json")
  oracle = read_json("bench/q8-ffn-amd-scheduler-project/oracle_contract.json")
  dsl = read_json("bench/q8-ffn-amd-scheduler-project/dsl_capability_map.json")
  n1 = read_json("bench/q8-ffn-amd-scheduler-project/n1_attribution.json")
  audit = read_json("bench/q8-ffn-codegen-transfer/asm_schedule_audit.json")

  instr = oracle["instruction_contract"]
  oracle_grouped = instr["oracle_grouped"]
  native_grouped = instr["tinygrad_asm_grouped"]
  oracle_top = instr["oracle_top_mnemonics"]
  native_top = instr["tinygrad_asm_top_mnemonics"]
  dynamic = oracle.get("dynamic_contract", {})
  timings = dnr2["timing"]

  feature_rows = {row["feature"]: row for row in dsl.get("features", [])}
  bounded_closed = [
    {
      "feature": "native_dot4_instruction_selection",
      "status": "closed",
      "why": "native and oracle both emit 16 v_dot4_i32_iu8",
      "movement_us": 0.0,
    },
    {
      "feature": "global_load_shape_coalescing",
      "status": "closed_as_standalone",
      "why": "expressible, but load-only movement is below DNR-3 start gate",
      "movement_us": feature_rows.get("vector_or_coalesced_global_loads", {}).get("estimated_or_measured_us"),
    },
    {
      "feature": "waitcnt_grouping",
      "status": "closed_as_standalone",
      "why": "expressible, but grouped wait movement is below DNR-3 start gate",
      "movement_us": feature_rows.get("waitcnt_grouping", {}).get("estimated_or_measured_us"),
    },
    {
      "feature": "reduction_topology",
      "status": "closed_as_standalone",
      "why": "native and oracle both use 5 ds_bpermute; reduction-only movement is below gate",
      "movement_us": feature_rows.get("reduction_rewrite", {}).get("estimated_or_measured_us"),
    },
  ]

  required_capabilities = [
    {
      "capability": "semantic_schedule_ir",
      "needed_for": "attach dependency groups and issue clusters to q8 loads, dot4, conversions, reduction, and store",
      "current_state": "DNR-2 has a correct handwritten instruction stream, but no reusable semantic scheduler IR for decode MMVQ",
      "entry_gate": "native stream can be represented as stages with def/use, memory space, lane role, and legal reordering constraints",
      "status": "missing",
    },
    {
      "capability": "s_clause_s_delay_alu_policy",
      "needed_for": "recover oracle markers: s_clause=3 and s_delay_alu=30",
      "current_state": f"native s_clause={top_count(native_top, 's_clause')}, native s_delay_alu={top_count(native_top, 's_delay_alu')}",
      "entry_gate": "policy inserts markers based on dependency/latency model, not static copying",
      "status": "missing_semantics",
    },
    {
      "capability": "coalesced_load_lowering_policy",
      "needed_for": "reduce native grouped global loads from 22 toward oracle 11 while preserving Q4_K/q8 correctness",
      "current_state": f"native global_load={native_grouped.get('global_load')}; oracle global_load={oracle_grouped.get('global_load')}; oracle b128 top count={top_count(oracle_top, 'global_load_b128')}",
      "entry_gate": "global_load_b128/u8 schedule generated from the Q4_K/q8 address model with correctness passing",
      "status": "missing_policy_not_opcode",
    },
    {
      "capability": "register_live_range_resource_policy",
      "needed_for": "choose instruction order/register reuse so vector loads, dot4 accumulators, and reduction temporaries do not serialize",
      "current_state": "classified as project-level; no timing-grade attribution without better trace/counter decode",
      "entry_gate": "resource ledger predicts live VGPR/SGPR pressure and validates no scratch/private spill",
      "status": "missing",
    },
    {
      "capability": "branch_and_exec_policy",
      "needed_for": "match oracle branch/exec control where beneficial without changing lane semantics",
      "current_state": f"native branch={native_grouped.get('branch')}; oracle branch={oracle_grouped.get('branch')}",
      "entry_gate": "branch predicates are derived from semantic lane roles and validated by correctness",
      "status": "missing",
    },
    {
      "capability": "hardware_attribution_oracle",
      "needed_for": "turn static scheduler/resource hypotheses into timing-grade decisions",
      "current_state": f"SQTT capture runnable={n1['gate'].get('sqtt_capture_runnable')}, SQTT decode usable={n1['gate'].get('sqtt_decode_usable')}, PMC runnable={n1['gate'].get('pmc_profile_runnable')}",
      "entry_gate": "counter/timeline attribution identifies one rule with >=30us credible movement or validates a compound candidate",
      "status": "blocked_tooling",
    },
  ]

  gates = {
    "dnr2_correct": dnr2.get("gate_pass") is True,
    "native_slower_than_oracle": timings.get("historical_native_minus_oracle_us", 0) > 0,
    "gap_large_enough_for_project": timings.get("historical_native_minus_oracle_us", 0) >= 30.0,
    "no_bounded_n2_candidate": n1["gate"].get("n2_candidate_count") == 0,
    "standalone_features_below_gate": all((row.get("movement_us") or 0.0) < 30.0 for row in bounded_closed),
    "scheduler_markers_missing": top_count(native_top, "s_clause") == 0 and top_count(native_top, "s_delay_alu") == 0,
    "body_insensitive_ladder": bool(dynamic.get("body_insensitive_variant_ladder")),
    "hardware_attribution_blocked": n1["gate"].get("sqtt_decode_usable") is False,
  }

  result = {
    "date": "2026-06-20",
    "phase": "DNR-3_DECODE_SCHEDULER_RESOURCE_SCOPE",
    "schema": "decode_native_renderer_dnr3_scheduler_resource_scope_v1",
    "verdict": "BLOCKED_DNR3_NEEDS_BROAD_SCHEDULER_RESOURCE_MODEL_AND_ATTRIBUTION",
    "gate_pass": False,
    "default_behavior_changed": False,
    "performance_claim": False,
    "timing_context": {
      "native_historical_us": timings.get("historical_median_us"),
      "oracle_consumer_us": timings.get("oracle_consumer_us"),
      "native_minus_oracle_us": timings.get("historical_native_minus_oracle_us"),
      "target_gate_us": oracle.get("target_gate_us"),
    },
    "static_delta": {
      "oracle_grouped": oracle_grouped,
      "native_grouped": native_grouped,
      "native_minus_oracle": audit.get("deltas", {}).get("tinygrad_asm_minus_hipcc_lld_grouped"),
      "scheduler_markers": instr.get("scheduler_markers"),
      "key_load_shape": instr.get("key_load_shape"),
      "reduction_shape": instr.get("reduction_shape"),
    },
    "bounded_features_closed": bounded_closed,
    "required_capabilities": required_capabilities,
    "gates": gates,
    "blocked_at": {
      "next_phase": "DNR-3A semantic scheduler/resource implementation",
      "reason": "DNR-2 correctness exists, but the remaining movement is compound scheduler/resource behavior with no timing-grade bounded feature.",
      "minimum_unblock": [
        "semantic schedule IR for decode MMVQ def/use and legal reordering",
        "resource/live-range ledger for VGPR/SGPR/private/LDS",
        "correctness-preserving coalesced load policy",
        "semantic s_clause/s_delay_alu insertion policy",
        "hardware attribution or a compound candidate that passes correctness and closes a meaningful fraction of the 73.109us gap",
      ],
      "do_not_do": ["BEAM/search", "standalone waitcnt patch", "standalone global-load patch", "standalone reduction patch", "static-copy s_delay_alu/s_clause"],
    },
    "input_artifacts": [
      "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr2_lowering_result.json",
      "bench/q8-ffn-amd-scheduler-project/oracle_contract.json",
      "bench/q8-ffn-amd-scheduler-project/dsl_capability_map.json",
      "bench/q8-ffn-amd-scheduler-project/n1_attribution.json",
      "bench/q8-ffn-codegen-transfer/asm_schedule_audit.json",
    ],
    "next_action": "Do not continue with one-off decode tweaks. Start DNR-3A only as broad backend scheduler/resource implementation, or pause native decode and keep the q8 artifact as the default-off oracle.",
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "timing_context": result["timing_context"],
    "gates": gates,
    "required_missing": [x["capability"] for x in required_capabilities if x["status"].startswith("missing") or x["status"].startswith("blocked")],
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
