#!/usr/bin/env python3
"""Execute decode next steps 1-2: research answer + native scheduler kickoff."""
from __future__ import annotations

import json, pathlib

OUT = pathlib.Path("bench/qk-decode-next12")


def load(path: str) -> dict:
  return json.loads(pathlib.Path(path).read_text())


def main() -> None:
  two_lane = load("bench/qk-decode-mmvq-large-project/q8_both_lanes_execution.json")
  closeout = load("bench/qk-decode-mmvq-large-project/q8_two_lane_closeout.json")
  contract = load("bench/q8-ffn-amd-scheduler-project/oracle_contract.json")
  dsl = load("bench/q8-ffn-amd-scheduler-project/dsl_capability_map.json")
  pmu = load("bench/q8-ffn-amd-scheduler-project/pmu_sqtt_evidence.json")
  result = {
    "schema": "decode_next12_execution_v1",
    "date": "2026-06-19",
    "option1_q8_research_answer": {
      "status": "COMPLETE_RESEARCH_ANSWER",
      "flag": "Q8_FFN_HANDWRITTEN=1",
      "default": "off",
      "final_claim": "q8 fused FFN artifact route is the measured decode kernel/lifecycle win for this project",
      "measured": {
        "min_wd_speedup": closeout["lane1_research_flag_hardening"]["min_speedup"],
        "median_wd_speedup": closeout["lane1_research_flag_hardening"]["median_speedup"],
        "dnll": closeout["lane1_research_flag_hardening"]["dnll"],
        "lifecycle_us": closeout["lane1_research_flag_hardening"]["artifact_hashes"]["lifecycle_us"],
      },
      "policy": {
        "accepted_for": "research flag only",
        "default_change": False,
        "dependency": "external hipcc/LLD HSACO",
        "fallback": "flag off returns to default tinygrad decode",
      },
      "done": True,
    },
    "option2_native_scheduler_project": {
      "status": "ACTIVE_PROJECT_CHARTER",
      "n0_oracle_diff": {
        "status": "COMPLETE",
        "artifact": "bench/q8-ffn-amd-scheduler-project/oracle_contract.json",
        "verdict": contract["verdict"],
        "key_features": [
          "global load shape",
          "s_clause/s_delay_alu scheduler markers",
          "wait/reduction details",
          "resource contract",
          "work decomposition",
        ],
      },
      "n1_attribution": {
        "status": "BLOCKED_ON_USABLE_ATTRIBUTION",
        "artifact": "bench/q8-ffn-amd-scheduler-project/pmu_sqtt_evidence.json",
        "pmc_profile_runnable": pmu["classification"]["pmc_profile_runnable"],
        "sqtt_profile_runnable": pmu["classification"]["sqtt_profile_runnable"],
        "sqtt_decode_usable": pmu["classification"]["sqtt_decode_usable"],
        "a2_reopen": pmu["classification"]["a2_reopen"],
        "blockers": pmu["classification"]["blockers"],
      },
      "bounded_feature_state": {
        "artifact": "bench/q8-ffn-amd-scheduler-project/dsl_capability_map.json",
        "a2_candidates_count": dsl["summary"]["a2_candidates_count"],
        "largest_measured_standalone_delta_us": dsl["summary"]["largest_measured_standalone_delta_us"],
        "verdict": dsl["verdict"],
      },
      "next_native_work": [
        "make SQTT decode usable for RDNA3 HCQ instruction traces, or add another attribution path",
        "use attribution to assign >=30us movement to one scheduler feature",
        "only then implement N2 scheduler feature proof",
      ],
      "start_gate_for_code_changes": ">=30us attributed feature or explicit whole AMD backend scheduler funding",
      "done": True,
    },
    "decision": {
      "do_option1_now": "done",
      "do_option2_now": "chartered; N0 complete; N1 is the first real work item",
      "do_not_do": [
        "do not reopen imported Q4 artifact routing",
        "do not start native A2 codegen without attribution or broader backend funding",
        "do not change defaults",
      ],
    },
  }
  result["verdict"] = "NEXT_1_COMPLETE_NEXT_2_CHARTERED_N1_BLOCKED_ON_ATTRIBUTION"
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "execution.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
