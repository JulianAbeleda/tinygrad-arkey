#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tinygrad.renderer.amd.schedule import (
  DecodeMMVQResourceLedger, DecodeMMVQSchedulerFeature, DecodeMMVQSchedulerResourcePlan,
  decode_mmvq_scheduler_resource_plan_summary,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3a_scheduler_resource_plan_result.json"


def read_json(rel: str) -> dict[str, Any]:
  with (ROOT / rel).open() as f:
    return json.load(f)


def feature(features: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
  return features.get(name, {})


def main() -> int:
  dnr2 = read_json("bench/qk-decode-primitive-transfer/decode_native_renderer_dnr2_lowering_result.json")
  dnr3 = read_json("bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3_scheduler_resource_scope_result.json")
  oracle = read_json("bench/q8-ffn-amd-scheduler-project/oracle_contract.json")
  dsl = read_json("bench/q8-ffn-amd-scheduler-project/dsl_capability_map.json")
  n1 = read_json("bench/q8-ffn-amd-scheduler-project/n1_attribution.json")
  loader = read_json("bench/q8-ffn-amd-scheduler-project/artifact_loader.json")

  instr = oracle["instruction_contract"]
  og, ng = instr["oracle_grouped"], instr["tinygrad_asm_grouped"]
  markers = instr["scheduler_markers"]
  features = {row["feature"]: row for row in dsl.get("features", [])}
  timing = dnr2["timing"]

  plan = DecodeMMVQSchedulerResourcePlan(
    role="ffn_gate/up",
    quant_format="Q4_K x q8_1",
    resource_ledger=DecodeMMVQResourceLedger(
      native_time_us=float(timing["historical_median_us"]),
      oracle_time_us=float(timing["oracle_consumer_us"]),
      native_group_segment_size=16,
      oracle_group_segment_size=int(loader["loader"]["gateup"]["group_segment_size"]),
      native_private_segment_size=0,
      oracle_private_segment_size=int(loader["loader"]["gateup"]["private_segment_size"]),
      native_global_loads=int(ng["global_load"]),
      oracle_global_loads=int(og["global_load"]),
      native_ds_ops=int(ng["ds"]),
      oracle_ds_ops=int(og["ds"]),
      native_waitcnt=int(ng["waitcnt"]),
      oracle_waitcnt=int(og["waitcnt"]),
      native_branch=int(ng["branch"]),
      oracle_branch=int(og["branch"]),
      native_s_clause=int(markers["tinygrad_s_clause"]),
      oracle_s_clause=int(markers["oracle_s_clause"]),
      native_s_delay_alu=int(markers["tinygrad_s_delay_alu"]),
      oracle_s_delay_alu=int(markers["oracle_s_delay_alu"]),
    ),
    features=(
      DecodeMMVQSchedulerFeature(
        "global_load_shape", "memory_schedule",
        f"native grouped global_load={ng['global_load']}",
        f"oracle grouped global_load={og['global_load']}, b128 top count={instr['key_load_shape']['oracle_global_load_b128_top_count']}",
        feature(features, "vector_or_coalesced_global_loads").get("estimated_or_measured_us"),
        "derive coalesced b128/u8 loads from Q4_K/q8 address semantics and revalidate correctness",
        "policy_missing_not_opcode_missing",
      ),
      DecodeMMVQSchedulerFeature(
        "waitcnt_policy", "dependency_schedule",
        f"native waitcnt={ng['waitcnt']}",
        f"oracle waitcnt={og['waitcnt']}",
        feature(features, "waitcnt_grouping").get("estimated_or_measured_us"),
        "emit vmcnt/lgkmcnt waits from producer-consumer edges instead of blanket wait placement",
        "standalone_closed_policy_needed_for_compound",
      ),
      DecodeMMVQSchedulerFeature(
        "reduction_resource_shape", "lds_reduction",
        f"native ds={ng['ds']}",
        f"oracle ds={og['ds']}, ds_bpermute={instr['reduction_shape']['oracle_ds_bpermute']}",
        feature(features, "reduction_rewrite").get("estimated_or_measured_us"),
        "keep reduction correctness while reducing LDS traffic as part of compound schedule",
        "standalone_closed_policy_needed_for_compound",
      ),
      DecodeMMVQSchedulerFeature(
        "s_clause_s_delay_alu", "issue_schedule",
        f"native s_clause={markers['tinygrad_s_clause']}, s_delay_alu={markers['tinygrad_s_delay_alu']}",
        f"oracle s_clause={markers['oracle_s_clause']}, s_delay_alu={markers['oracle_s_delay_alu']}",
        None,
        "insert markers from semantic latency/resource policy, not static copying",
        "semantic_policy_missing",
      ),
      DecodeMMVQSchedulerFeature(
        "register_live_range_resource", "allocator_resource",
        "native stream correct but body-insensitive variants remain slow",
        "oracle appears to schedule/reuse resources differently",
        None,
        "live VGPR/SGPR ledger with no scratch/private spill and legal instruction reordering",
        "allocator_policy_missing",
      ),
      DecodeMMVQSchedulerFeature(
        "branch_exec_policy", "control_schedule",
        f"native branch={ng['branch']}",
        f"oracle branch={og['branch']}",
        None,
        "derive branch/exec predicates from lane role semantics and validate output equality",
        "control_policy_missing",
      ),
    ),
    required_capabilities=tuple(dnr3["blocked_at"]["minimum_unblock"]),
    closed_standalone_features=("dot4", "global_load_shape", "waitcnt", "reduction"),
    hardware_attribution_status="blocked: SQTT capture runnable but RDNA3 HCQ decode unusable; PMC runnable but not counter-grade in saved artifacts",
  )

  summary = decode_mmvq_scheduler_resource_plan_summary(plan)
  structural_gate = plan.structural_gate()
  implementation_gates = {
    "structural_plan_passed": structural_gate["passed"],
    "dnr2_correct": dnr2.get("gate_pass") is True,
    "dnr3_scope_blocked": dnr3.get("verdict") == "BLOCKED_DNR3_NEEDS_BROAD_SCHEDULER_RESOURCE_MODEL_AND_ATTRIBUTION",
    "no_bounded_n2_candidate": n1["gate"].get("n2_candidate_count") == 0,
    "has_semantic_schedule_ir_object": True,
    "can_emit_compound_candidate_now": False,
    "can_attribute_hardware_now": False,
  }
  result = {
    "date": "2026-06-20",
    "phase": "DNR-3A_DECODE_SCHEDULER_RESOURCE_PLAN",
    "schema": "decode_native_renderer_dnr3a_scheduler_resource_plan_v1",
    "verdict": "PASS_DNR3A_PLAN_STRUCTURAL_BLOCKED_ON_COMPOUND_EMITTER_AND_ATTRIBUTION",
    "gate_pass": structural_gate["passed"],
    "default_behavior_changed": False,
    "performance_claim": False,
    "plan": plan.to_dict(),
    "summary": summary,
    "implementation_gates": implementation_gates,
    "blocked_at": {
      "next_phase": "DNR-3B compound scheduler/resource emitter",
      "reason": "The scheduler/resource contract is now first-class, but no emitter exists that can transform the correct DNR-2 stream into a correctness-preserving oracle-shaped compound candidate.",
      "minimum_unblock": [
        "lower DNR-2 instruction stream through this plan",
        "apply coalesced load, wait, marker, branch, and register policies together",
        "launch candidate and pass gate/up correctness",
        "time candidate against the q8 oracle",
        "attribute movement enough to decide whether to continue or kill native decode scheduler work",
      ],
    },
    "input_artifacts": [
      "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr2_lowering_result.json",
      "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3_scheduler_resource_scope_result.json",
      "bench/q8-ffn-amd-scheduler-project/oracle_contract.json",
      "bench/q8-ffn-amd-scheduler-project/dsl_capability_map.json",
      "bench/q8-ffn-amd-scheduler-project/n1_attribution.json",
      "bench/q8-ffn-amd-scheduler-project/artifact_loader.json",
    ],
    "next_action": "DNR-3B must build a compound emitter. Do not run BEAM/search until a DNR-3B candidate is correct and timed.",
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "gate_pass": result["gate_pass"],
    "summary": summary,
    "implementation_gates": implementation_gates,
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if structural_gate["passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
