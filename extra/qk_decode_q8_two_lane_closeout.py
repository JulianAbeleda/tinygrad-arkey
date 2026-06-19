#!/usr/bin/env python3
"""Consolidate the q8 decode artifact lane and native-transfer lane decisions."""
from __future__ import annotations

import json, pathlib, statistics

OUT = pathlib.Path("bench/qk-decode-mmvq-large-project")


def load(path: str) -> dict:
  return json.loads(pathlib.Path(path).read_text())


def main() -> None:
  p8 = load("bench/qk-decode-mmvq-large-project/p8_fused_lifecycle_decision.json")
  wd_base = load("bench/q8-ffn-handwritten-oracle/decode_wd_baseline.json")
  wd_q8 = load("bench/q8-ffn-handwritten-oracle/decode_wd_q8_route.json")
  nll_base = load("bench/q8-ffn-handwritten-oracle/nll_baseline.json")
  nll_q8 = load("bench/q8-ffn-handwritten-oracle/nll_q8_route.json")
  artifact_result = load("bench/q8-ffn-amd-scheduler-project/result.json")
  artifact_loader = load("bench/q8-ffn-amd-scheduler-project/artifact_loader.json")
  artifact_policy = load("bench/q8-ffn-amd-scheduler-project/artifact_policy_boundary.json")
  dsl_map = load("bench/q8-ffn-amd-scheduler-project/dsl_capability_map.json")

  wd_rows = []
  for base, cand in zip(wd_base["rows"], wd_q8["rows"]):
    wd_rows.append({
      "ctx": base["ctx"],
      "baseline_tok_s": base["tok_s_W"],
      "q8_tok_s": cand["tok_s_W"],
      "speedup": cand["tok_s_W"] / base["tok_s_W"],
    })
  speedups = [r["speedup"] for r in wd_rows]
  dnll = nll_q8["nll"] - nll_base["nll"]
  lane1_gates = {
    "default_off": artifact_policy["default_changed"] is False,
    "no_in_process_hip_runtime": artifact_policy["requirements"]["no_in_process_hip_runtime"] is True,
    "artifact_route_pass": artifact_result["verdict"] == "PASS",
    "graph_route_pass": artifact_loader["verdict"] == "PASS",
    "min_decode_speedup_ge_1p03": min(speedups) >= 1.03,
    "dnll_lte_0p01": dnll <= 0.01,
    "policy_boundary_documented": artifact_policy["status"] == "research_only",
  }

  lane2_gates = {
    "native_current_routes_not_enough": p8["p8b_current_uop_expressibility"]["dsl_capability_verdict"] == "FAIL_A1_NO_BOUNDED_FEATURE",
    "no_bounded_a2_feature": dsl_map["summary"]["a2_candidates_count"] == 0,
    "artifact_oracle_available": p8["p8c_handwritten_prototype"]["verdict"] == "PASS_ARTIFACT_PROTOTYPE",
    "native_marked_project_level": p8["p8d_decision"]["native_tinygrad_route"] == "PROJECT_LEVEL_RENDERER_SCHEDULER",
  }

  result = {
    "schema": "decode_q8_two_lane_closeout_v1",
    "date": "2026-06-19",
    "purpose": "Close the immediate decode scope into research-artifact hardening plus native-transfer roadmap.",
    "lane1_research_flag_hardening": {
      "status": "PASS_RESEARCH_FLAG_READY",
      "flag": "Q8_FFN_HANDWRITTEN=1",
      "default": "off",
      "scope": artifact_policy["supported"],
      "artifact_hashes": artifact_result["summary"],
      "wd_rows": wd_rows,
      "min_speedup": min(speedups),
      "median_speedup": statistics.median(speedups),
      "nll_baseline": nll_base["nll"],
      "nll_q8": nll_q8["nll"],
      "dnll": dnll,
      "policy_gate": artifact_policy["policy_gate"],
      "gates": lane1_gates,
    },
    "lane2_native_transfer_roadmap": {
      "status": "PROJECT_LEVEL_NOT_BOUNDED_PATCH",
      "oracle": {
        "local_lifecycle_us": p8["p8c_handwritten_prototype"]["hipcc_lld_artifact_lifecycle_us"],
        "local_speedup_vs_current_gateup": p8["p8c_handwritten_prototype"]["artifact_speedup_vs_p7e_baseline"],
        "graph_route_pass": p8["p8c_handwritten_prototype"]["artifact_graph_route_pass"],
      },
      "current_native_failures": {
        "comgr_lifecycle_us": p8["p8b_current_uop_expressibility"]["native_comgr_lifecycle_us"],
        "amd_dsl_consumer_us": p8["p8b_current_uop_expressibility"]["native_amd_dsl_consumer_us"],
        "dsl_capability_verdict": p8["p8b_current_uop_expressibility"]["dsl_capability_verdict"],
      },
      "project_capabilities_required": [
        "latency-aware AMD instruction scheduling",
        "register allocation and live-range control for low-VGPR high-occupancy kernels",
        "semantic waitcnt/s_clause/s_delay_alu placement",
        "global load grouping/coalescing as part of scheduling, not a standalone knob",
        "staged reductions and post-barrier multi-output stores",
        "SQTT/PMU attribution good enough to assign >=30us movement to a bounded feature",
      ],
      "start_gate": "Do not start native transfer until one bounded feature has >=30us measured or attributed movement, or the project funds the whole AMD backend scheduler effort.",
      "gates": lane2_gates,
    },
  }
  result["verdict"] = "BOTH_LANES_SCOPED_ARTIFACT_READY_NATIVE_PROJECT_LEVEL" if all(lane1_gates.values()) and all(lane2_gates.values()) else "INCOMPLETE"
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "q8_two_lane_closeout.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
