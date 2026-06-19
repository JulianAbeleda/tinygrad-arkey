#!/usr/bin/env python3
"""P8 fused q8+MMVQ lifecycle decision from measured artifacts."""
from __future__ import annotations

import json, pathlib

ROOT = pathlib.Path("bench")
OUT = ROOT / "qk-decode-mmvq-large-project"


def load(path: str) -> dict:
  return json.loads(pathlib.Path(path).read_text())


def main() -> None:
  p7e = load("bench/qk-decode-mmvq-large-project/p7e_gateup_amortization.json")
  p5 = load("bench/qk-decode-mmvq-large-project/p5_lifecycle_probe.json")
  p6 = load("bench/qk-decode-mmvq-large-project/p6_q4_shape_matrix.json")
  q8_lifecycle = load("bench/q8-ffn-handwritten-oracle/gate_up_lifecycle.json")
  fast_artifact = load("bench/q8-ffn-amd-scheduler-project/artifact_loader.json")
  graph_artifact = load("bench/q8-ffn-amd-scheduler-project/artifact_graph_route.json")
  comgr = load("bench/q8-ffn-codegen-transfer/comgr_fused_gateup.json")
  asm_full = load("bench/q8-ffn-codegen-transfer/asm_gateup_full.json")
  dsl_map = load("bench/q8-ffn-amd-scheduler-project/dsl_capability_map.json")
  wd_base = load("bench/q8-ffn-handwritten-oracle/decode_wd_baseline.json")
  wd_q8 = load("bench/q8-ffn-handwritten-oracle/decode_wd_q8_route.json")

  baseline_gateup_us = p7e["timing"]["baseline_ms_median"] * 1000.0
  p8a_gate_us = baseline_gateup_us / 1.10
  p6_gate = next(r for r in p6["rows"] if r["tensor"] == "blk.0.ffn_gate.weight")
  p6_up = next(r for r in p6["rows"] if r["tensor"] == "blk.0.ffn_up.weight")
  consumer_math_us = (p6_gate["device_ms_per_launch"] + p6_up["device_ms_per_launch"]) * 1000.0
  producer_us = p5["timing"]["producer_device_ms_median"] * 1000.0
  impossible_lower_us = max(consumer_math_us, producer_us)
  additive_lower_us = consumer_math_us + producer_us
  fast_lifecycle_us = fast_artifact["perf_gateup"]["gate_up_lifecycle_us"]
  q8_modeled_us = q8_lifecycle["components"]["gate_up_q8_lifecycle_us"]
  comgr_us = comgr["gate_up_lifecycle_us"]
  asm_consumer_us = asm_full["timing"]["median_ms"] * 1000.0

  rows = []
  for base, cand in zip(wd_base["rows"], wd_q8["rows"]):
    rows.append({
      "ctx": base["ctx"],
      "baseline_tok_s": base["tok_s_W"],
      "q8_route_tok_s": cand["tok_s_W"],
      "speedup": cand["tok_s_W"] / base["tok_s_W"],
    })

  result = {
    "schema": "decode_mmvq_large_project_p8_fused_lifecycle_decision_v1",
    "date": "2026-06-19",
    "phase": "P8_fused_lifecycle_1_to_4",
    "p8a_lower_bound_model": {
      "baseline_gateup_us": baseline_gateup_us,
      "required_for_1p10_us": p8a_gate_us,
      "consumer_math_us_from_p6": consumer_math_us,
      "producer_us_from_p5": producer_us,
      "impossible_overlap_lower_bound_us": impossible_lower_us,
      "additive_lower_bound_us": additive_lower_us,
      "build_worth_if_fused_can_remove_launch_lifecycle": additive_lower_us <= p8a_gate_us,
    },
    "p8b_current_uop_expressibility": {
      "stub_graphs_multi_output": True,
      "graph_artifact_route_pass": graph_artifact["verdict"] == "PASS",
      "native_comgr_lifecycle_us": comgr_us,
      "native_comgr_passes_1p10_gate": comgr_us <= p8a_gate_us,
      "native_amd_dsl_consumer_us": asm_consumer_us,
      "native_amd_dsl_consumer_only_passes_1p10_gate": asm_consumer_us <= p8a_gate_us,
      "dsl_capability_verdict": dsl_map["verdict"],
      "verdict": "PARTIAL_STUB_AND_ARTIFACT_GRAPH_ONLY",
    },
    "p8c_handwritten_prototype": {
      "modeled_oracle_lifecycle_us": q8_modeled_us,
      "hipcc_lld_artifact_lifecycle_us": fast_lifecycle_us,
      "artifact_speedup_vs_p7e_baseline": baseline_gateup_us / fast_lifecycle_us,
      "artifact_graph_route_pass": graph_artifact["verdict"] == "PASS",
      "oneblock_oracle_available": pathlib.Path("bench/q8-ffn-handwritten-oracle/oneblock_fast_artifact_route.json").exists(),
      "verdict": "PASS_ARTIFACT_PROTOTYPE",
    },
    "p8d_decision": {
      "imported_llama_q4_route": "CLOSED_LOCAL_TIMING_WIN",
      "fused_q8_mmvq_artifact_route": "FEASIBLE_RESEARCH_FLAG",
      "native_tinygrad_route": "PROJECT_LEVEL_RENDERER_SCHEDULER",
      "do_next": "Do not continue imported Q4 artifact routing. If funding decode, use the existing q8 fused artifact as the measured target or fund native renderer transfer.",
      "whole_decode_wd_rows": rows,
      "median_wd_speedup": sorted(r["speedup"] for r in rows)[len(rows)//2],
    },
    "gates": {
      "p8a_build_worth": additive_lower_us <= p8a_gate_us,
      "p8b_native_current_uops_not_enough": comgr_us > p8a_gate_us and asm_consumer_us > p8a_gate_us,
      "p8c_artifact_clears_local_gate": fast_lifecycle_us <= p8a_gate_us,
      "p8c_graph_route_pass": graph_artifact["verdict"] == "PASS",
    },
  }
  result["verdict"] = "P8_COMPLETE_ARTIFACT_YES_NATIVE_PROJECT_LEVEL" if all(result["gates"].values()) else "P8_INCONCLUSIVE"
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "p8_fused_lifecycle_decision.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
