#!/usr/bin/env python3
"""Record the accepted execution state for both q8 decode lanes."""
from __future__ import annotations

import json, pathlib

OUT = pathlib.Path("bench/qk-decode-mmvq-large-project")


def load(path: str) -> dict:
  return json.loads(pathlib.Path(path).read_text())


def main() -> None:
  closeout = load("bench/qk-decode-mmvq-large-project/q8_two_lane_closeout.json")
  policy = load("bench/q8-ffn-amd-scheduler-project/artifact_policy_boundary.json")
  result = {
    "schema": "decode_q8_both_lanes_execution_v1",
    "date": "2026-06-19",
    "decision": "DO_BOTH",
    "lane1_artifact_research_flag": {
      "decision": "ACCEPT_FOR_RESEARCH_FLAG_ONLY",
      "flag": closeout["lane1_research_flag_hardening"]["flag"],
      "default": "off",
      "accepted_dependency": "external hipcc/LLD HSACO accepted only for research-flag use",
      "supported": policy["supported"],
      "requirements": policy["requirements"],
      "non_goals": policy["non_goals"],
      "evidence": {
        "min_wd_speedup": closeout["lane1_research_flag_hardening"]["min_speedup"],
        "median_wd_speedup": closeout["lane1_research_flag_hardening"]["median_speedup"],
        "dnll": closeout["lane1_research_flag_hardening"]["dnll"],
        "lifecycle_us": closeout["lane1_research_flag_hardening"]["artifact_hashes"]["lifecycle_us"],
        "no_in_process_hip_runtime": closeout["lane1_research_flag_hardening"]["artifact_hashes"]["no_hip_runtime_in_process"],
      },
      "maintenance_actions": [
        "keep Q8_FFN_HANDWRITTEN default off",
        "keep artifact source strings and hashes documented",
        "rerun W==D and dNLL if source strings, toolchain, model shape, or graph route changes",
        "do not generalize beyond gfx1100/Qwen3-8B dense Q4_K gate/up without a new scope",
      ],
    },
    "lane2_native_project": {
      "decision": "FUND_AS_PROJECT_LEVEL_BACKEND_WORK",
      "not_a_bounded_patch": True,
      "oracle": closeout["lane2_native_transfer_roadmap"]["oracle"],
      "current_native_failures": closeout["lane2_native_transfer_roadmap"]["current_native_failures"],
      "capabilities_required": closeout["lane2_native_transfer_roadmap"]["project_capabilities_required"],
      "phase_plan": [
        {
          "phase": "N0_oracle_diff_tool",
          "deliverable": "machine-readable diff between artifact oracle, COMGR, and tinygrad AMD DSL schedules",
          "gate": "diff labels instruction groups, load widths, wait placement, resources, and timing without manual prose",
        },
        {
          "phase": "N1_attribution",
          "deliverable": "PMU/SQTT or deterministic proxy that attributes >=30us to one scheduler feature",
          "gate": "one feature has measured or attributed >=30us movement",
        },
        {
          "phase": "N2_scheduler_feature",
          "deliverable": "one backend capability implemented behind an AMD-only experimental flag",
          "gate": "q8 consumer improves by >=25us and remains correct",
        },
        {
          "phase": "N3_native_rebuild",
          "deliverable": "native fused q8 gate/up consumer",
          "gate": "consumer <=75us, credible path to <=60us, max_abs <=2e-3",
        },
        {
          "phase": "N4_model_gate",
          "deliverable": "native route behind default-off flag",
          "gate": "W==D >=3%, dNLL <=0.01, no external artifact",
        },
      ],
      "start_gate": closeout["lane2_native_transfer_roadmap"]["start_gate"],
    },
    "verdict": "BOTH_ACCEPTED_ARTIFACT_RESEARCH_AND_NATIVE_PROJECT_CHARTERED",
  }
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "q8_both_lanes_execution.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
