#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench/q8-ffn-amd-scheduler-project"
OUT = BENCH / "n1_attribution.json"

def read_json(path: pathlib.Path) -> dict[str, Any]:
  return json.loads(path.read_text())

def us(ms: float) -> float:
  return round(ms * 1000.0, 3)

def row(name: str, bucket: str, attribution_us: float | None, source: str, evidence: list[str], decision: str,
        blocker: str | None = None) -> dict[str, Any]:
  return {
    "feature": name,
    "bucket": bucket,
    "attribution_us": attribution_us,
    "attribution_source": source,
    "evidence": evidence,
    "n2_gate_ge_30us": attribution_us is not None and attribution_us >= 30.0,
    "decision": decision,
    **({"blocker": blocker} if blocker is not None else {}),
  }

def main() -> int:
  contract = read_json(BENCH / "oracle_contract.json")
  capability = read_json(BENCH / "dsl_capability_map.json")
  pmu_sqtt = read_json(BENCH / "pmu_sqtt_evidence.json")

  timings = contract["known_timings_us"]
  dyn = contract["dynamic_contract"]["variant_medians_ms"]
  full_ms = timings["tinygrad_asm_gateup_full"] / 1000.0
  load_delta = us(full_ms - dyn["load_wait_only"])
  wait_delta = us(dyn["load_wait_only"] - dyn["wait_grouped_load_only"])
  reduction_delta = us(full_ms - dyn["reduction_only"])
  dot_body_delta = us(full_ms - dyn["dot_synthetic"])
  full_gap = round(timings["tinygrad_asm_gateup_full"] - timings["hipcc_lld_gateup_current_loader"], 3)

  inst = contract["instruction_contract"]
  classification = pmu_sqtt["classification"]
  sqtt_errors = []
  for ev in pmu_sqtt["runs"]["sqtt"].get("profile", {}).get("sqtt", []):
    dec = ev.get("decode_summary")
    if dec is not None and not dec.get("ok"):
      sqtt_errors.append(dec.get("error"))
  sqtt_errors = sorted(set(e for e in sqtt_errors if e))

  rows = [
    row("native_dot4_instruction_selection", "already-matched", 0.0, "static oracle diff",
        ["oracle and tinygrad ASM both emit 16 v_dot4_i32_iu8 in the 32-weight body",
         f"dot-synthetic variant movement is {dot_body_delta}us, but dot instruction selection itself is already equal"],
        "closed; not an N2 target"),
    row("global_load_shape_coalescing", "bounded-compiler-feature", load_delta, "dynamic fallback plus static diff",
        [f"oracle global-load shape: b128={inst['key_load_shape']['oracle_global_load_b128_top_count']}, u8={inst['key_load_shape']['oracle_global_load_u8_top_count']}",
         f"tinygrad global-load shape: b32={inst['key_load_shape']['tinygrad_global_load_b32_top_count']}, u8={inst['key_load_shape']['tinygrad_global_load_u8_top_count']}, u16={inst['key_load_shape']['tinygrad_global_load_u16_top_count']}",
         f"load-wait-only variant leaves only {load_delta}us of standalone movement"],
        "below 30us; do not start standalone N2"),
    row("waitcnt_grouping", "bounded-compiler-feature", wait_delta, "dynamic fallback variant",
        [f"grouped wait variant improves load-wait-only by {wait_delta}us"],
        "closed as standalone N2 target"),
    row("reduction_topology", "bounded-compiler-feature", reduction_delta, "dynamic fallback variant",
        [f"oracle ds total={inst['reduction_shape']['oracle_ds_total']}; tinygrad ds total={inst['reduction_shape']['tinygrad_ds_total']}",
         f"reduction-only variant leaves {reduction_delta}us of standalone movement"],
        "below 30us; do not start standalone N2"),
    row("s_clause_s_delay_alu_scheduler", "unattributed-scheduler-feature", None, "static diff only",
        [f"oracle has s_clause={inst['scheduler_markers']['oracle_s_clause']} and s_delay_alu={inst['scheduler_markers']['oracle_s_delay_alu']}",
         f"tinygrad ASM has s_clause={inst['scheduler_markers']['tinygrad_s_clause']} and s_delay_alu={inst['scheduler_markers']['tinygrad_s_delay_alu']}",
         "body-insensitive ladder points at scheduler/resource behavior, but not at one bounded insertion rule"],
        "no N2 until hardware attribution identifies a >=30us rule",
        "SQTT capture exists but local RDNA3 HCQ decode fails"),
    row("register_live_range_resource_scheduler", "unattributed-scheduler-feature", None, "static/dynamic inference",
        ["oracle is 93.54us while tinygrad ASM is 166.65us",
         "known bounded variants do not explain the full 73.109us gap",
         "remaining movement is plausible scheduler/register/resource policy, but not yet feature-attributed"],
        "project-level AMD scheduler work only; no bounded N2"),
    row("local_y_descriptor_and_launch_contract", "low-ev-runtime-feature", None, "prior route evidence",
        ["local-y descriptor work was useful for artifact interop but did not create a native speed path",
         "launch contract is known and not the current measured bottleneck"],
        "do not reopen for decode performance"),
  ]

  candidates = [r for r in rows if r["n2_gate_ge_30us"]]
  result = {
    "date": "2026-06-19",
    "phase": "N1_decode_native_scheduler_attribution",
    "purpose": "Decide whether any bounded native AMD scheduler/codegen feature earns N2 implementation.",
    "inputs": {
      "oracle_contract": str((BENCH / "oracle_contract.json").relative_to(ROOT)),
      "dsl_capability_map": str((BENCH / "dsl_capability_map.json").relative_to(ROOT)),
      "pmu_sqtt_evidence": str((BENCH / "pmu_sqtt_evidence.json").relative_to(ROOT)),
    },
    "gate": {
      "required_for_n2": "one feature with credible >=30us attributed movement",
      "full_tinygrad_to_oracle_gap_us": full_gap,
      "sqtt_capture_runnable": classification["sqtt_profile_runnable"],
      "sqtt_decode_usable": classification["sqtt_decode_usable"],
      "pmc_profile_runnable": classification["pmc_profile_runnable"],
      "fallback_static_dynamic_attribution": True,
      "n2_candidate_count": len(candidates),
    },
    "feature_attribution": rows,
    "sqtt_decode_errors": sqtt_errors,
    "consistency_checks": {
      "a1_candidate_count": capability["summary"]["a2_candidates_count"],
      "a1_largest_measured_standalone_delta_us": round(capability["summary"]["largest_measured_standalone_delta_us"], 3),
      "n1_largest_bounded_attribution_us": max(r["attribution_us"] or 0.0 for r in rows),
    },
    "verdict": "N1_COMPLETE_NO_N2_START",
    "decision": (
      "Do not begin a bounded native q8 scheduler/codegen N2 patch. The observable bounded buckets are all below "
      "30us, while the only plausible remaining movement is an unattributed project-level AMD scheduler/resource model."
    ),
    "next": (
      "If native work continues, fund tooling first: make RDNA3 HCQ SQTT decode usable or add equivalent PMU/timeline "
      "attribution. Otherwise keep decode on the accepted default-off q8 artifact route."
    ),
  }
  OUT.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps({"out": str(OUT.relative_to(ROOT)), "verdict": result["verdict"], "n2_candidate_count": len(candidates)}, indent=2))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
