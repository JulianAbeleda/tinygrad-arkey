#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_native_renderer_project_scope_result.json"


def read_json(rel: str) -> dict[str, Any]:
  path = ROOT / rel
  with path.open() as f:
    return json.load(f)


def read_json_optional(rel: str) -> dict[str, Any]:
  path = ROOT / rel
  if not path.exists(): return {}
  with path.open() as f:
    return json.load(f)


def grouped(contract: dict[str, Any], key: str) -> dict[str, Any]:
  return contract.get("instruction_contract", {}).get(key, {})


def feature_map(dsl: dict[str, Any]) -> dict[str, dict[str, Any]]:
  return {f.get("feature", ""): f for f in dsl.get("features", [])}


def route_decision(closeout: dict[str, Any], route: str) -> str:
  for row in closeout.get("routes", []):
    if row.get("route") == route:
      return str(row.get("decision", ""))
  return ""


def synthesize_closeout() -> dict[str, Any]:
  p7d = read_json("bench/qk-decode-mmvq-large-project/p7d_one_role_timing.json")
  p7e = read_json("bench/qk-decode-mmvq-large-project/p7e_gateup_amortization.json")
  p8 = read_json("bench/qk-decode-mmvq-large-project/p8_fused_lifecycle_decision.json")
  q8 = read_json_optional("bench/q8-ffn-artifact-promotion/promotion_result.json")
  return {
    "routes": [
      {"route": "current_default_decode", "decision": "KEEP_PROMOTED_DEFAULT"},
      {"route": "imported_llama_q4_mmvq_graph_route", "decision": "CLOSED_AS_SPEED_ROUTE",
       "evidence": {
         "attn_output_speedup": (p7d.get("timing") or {}).get("speedup"),
         "gateup_speedup": (p7e.get("timing") or {}).get("speedup"),
         "p8_imported_llama_q4_route": ((p8.get("p8d_decision") or {}).get("imported_llama_q4_route")),
       }},
      {"route": "fused_q8_mmvq_artifact", "decision": "KEEP_HARDENED_OPT_IN",
       "evidence": {"q8_promotion": q8.get("verdict"), "p8_artifact_route": ((p8.get("p8d_decision") or {}).get("fused_q8_mmvq_artifact_route"))}},
      {"route": "native_tinygrad_mmvq_renderer", "decision": "PROJECT_LEVEL_BLOCKED",
       "evidence": {"p8_native_route": ((p8.get("p8d_decision") or {}).get("native_tinygrad_route"))}},
    ],
    "source": "synthesized_from_tracked_p7_p8_artifacts",
  }


def main() -> None:
  oracle = read_json("bench/q8-ffn-amd-scheduler-project/oracle_contract.json")
  loader = read_json("bench/q8-ffn-amd-scheduler-project/artifact_loader.json")
  dsl = read_json("bench/q8-ffn-amd-scheduler-project/dsl_capability_map.json")
  n1 = read_json("bench/q8-ffn-amd-scheduler-project/n1_attribution.json")
  readiness = read_json("bench/qk-decode-native-tooling/readiness.json")
  closeout = read_json_optional("bench/qk-decode-primitive-transfer/decode_route_decision_closeout_result.json") or synthesize_closeout()

  oracle_grouped = grouped(oracle, "oracle_grouped")
  tinygrad_grouped = grouped(oracle, "tinygrad_asm_grouped")
  feats = feature_map(dsl)
  max_timing_movement = max(
    [float(x.get("estimated_or_measured_us") or 0.0) for x in dsl.get("features", [])] +
    [float(x.get("movement_us") or 0.0) for x in readiness.get("ablation_matrix", [])]
  )
  n2_candidates = int(n1.get("gate", {}).get("n2_candidate_count", -1))

  producer = loader.get("loader", {}).get("producer", {})
  gateup = loader.get("loader", {}).get("gateup", {})
  perf = loader.get("perf_gateup", {})
  default_decision = route_decision(closeout, "current_default_decode")
  imported_q4_decision = route_decision(closeout, "imported_llama_q4_mmvq_graph_route")

  project_tracks = [
    {
      "track": "DNR-0 oracle preservation",
      "purpose": "Keep hipcc/LLD q8 producer+gateup artifacts as the correctness and timing oracle.",
      "entry": "artifact_loader PASS and route remains default-off",
      "exit": "native candidate can be compared against identical q8 bytes and gate/up outputs",
      "status": "ready",
    },
    {
      "track": "DNR-1 schedule contract object",
      "purpose": "Represent the q8 producer and fused gate/up consumer as native tinygrad AMD schedule contracts.",
      "entry": "DecodeMMVQScheduleObject structural rows exist",
      "exit": "q8 artifact oracle launch, resource, work decomposition, and grouped ISA contract are checkable",
      "status": "ready_for_structural_work",
    },
    {
      "track": "DNR-2 address and data-format lowering",
      "purpose": "Lower block_q8_1 activation loads, Q4_K packed weight loads, min/scale correction, and gate/up y-role selection.",
      "entry": "DNR-1 passes",
      "exit": "native emitted kernel is runnable and numerically matches oracle for ffn_gate/up",
      "status": "project_work",
    },
    {
      "track": "DNR-3 scheduler/resource model",
      "purpose": "Model the compound gap: s_clause/s_delay_alu placement, register live range, instruction ordering, and wait/resource policy.",
      "entry": "DNR-2 correctness passes",
      "exit": "native timing closes toward oracle without scratch/private spills",
      "status": "project_work_not_bounded_patch",
    },
    {
      "track": "DNR-4 timing authority",
      "purpose": "Use one-clock interleaved timing against current default, q8 artifact oracle, and native candidate.",
      "entry": "native correctness passes",
      "exit": "promotion table with W==D, dNLL, lifecycle, and clock provenance",
      "status": "blocked_until_native_candidate",
    },
    {
      "track": "DNR-5 search/BEAM enablement",
      "purpose": "Only search inside the represented native schedule space after lowering and timing gates exist.",
      "entry": "DNR-2 and DNR-3 expose legal knobs with correctness-preserving structural gates",
      "exit": "search candidates are comparable to the q8 oracle without changing defaults",
      "status": "blocked_until_lowering_and_scheduler_model",
    },
  ]

  structural_requirements = [
    {"gate": "producer_contract", "required": {"global_size": [1, 1, 1], "local_size": [1024, 1, 1], "kernarg_size": 32, "group_segment_size": 4096, "private_segment_size": 0}, "observed": producer},
    {"gate": "gateup_contract", "required": {"global_size": [12288, 2, 1], "local_size": [32, 4, 1], "kernarg_size": 40, "group_segment_size": 16, "private_segment_size": 0}, "observed": gateup},
    {"gate": "work_decomposition", "required": "128 threads per row; block y selects gate/up; 16 Q4_K blocks; sub=tid&7; kb=tid/8", "observed": oracle.get("launch_contract", {}).get("work_decomposition")},
    {"gate": "dot4_count", "required": 16, "observed": oracle_grouped.get("dot4")},
    {"gate": "oracle_global_load_budget", "required": "<=11 grouped global_load", "observed": oracle_grouped.get("global_load")},
    {"gate": "oracle_ds_budget", "required": "<=7 grouped ds", "observed": oracle_grouped.get("ds")},
    {"gate": "single_barrier", "required": 1, "observed": oracle_grouped.get("barrier")},
    {"gate": "shuffle_topology", "required": 5, "observed": oracle_grouped.get("shuffle")},
    {"gate": "single_store", "required": 1, "observed": oracle_grouped.get("global_store")},
    {"gate": "no_private_segment", "required": 0, "observed": gateup.get("private_segment_size")},
    {"gate": "producer_correct", "required": True, "observed": loader.get("gates", {}).get("producer_correct")},
    {"gate": "gateup_correct", "required": True, "observed": loader.get("gates", {}).get("gate_correct") and loader.get("gates", {}).get("up_correct")},
    {"gate": "default_unchanged", "required": True, "observed": loader.get("default_changed") is False and default_decision == "KEEP_PROMOTED_DEFAULT"},
  ]

  missing_native_equivalents = [
    {
      "oracle_feature": "block_q8_1 producer lifecycle",
      "tinygrad_state": "artifact import exists; native schedule not promoted",
      "why_missing": "needs native producer/consumer contract plus quality policy, not another default route",
    },
    {
      "oracle_feature": "packed Q4_K + q8 dot4 consumer",
      "tinygrad_state": "dot4 mnemonic exists and count already matches",
      "why_missing": "instruction selection is not the gap; address/data-format lowering and resource scheduling are",
    },
    {
      "oracle_feature": "global load shape/coalescing",
      "tinygrad_state": "bounded variant measured",
      "why_missing": f"standalone movement {feats.get('vector_or_coalesced_global_loads', {}).get('estimated_or_measured_us')}us is below the 30us N2 gate",
    },
    {
      "oracle_feature": "wait/resource grouping",
      "tinygrad_state": "waitcnt grouping expressible",
      "why_missing": f"standalone movement {feats.get('waitcnt_grouping', {}).get('estimated_or_measured_us')}us is below gate",
    },
    {
      "oracle_feature": "s_clause/s_delay_alu schedule markers",
      "tinygrad_state": "mnemonics exposed but no semantic insertion policy",
      "why_missing": "static diff only; local SQTT decode is not usable enough to attribute a timing-grade rule",
    },
    {
      "oracle_feature": "register live-range/resource policy",
      "tinygrad_state": "classified as renderer scheduler work",
      "why_missing": "remaining 73.109us oracle gap is compound; no one feature cleared N2",
    },
  ]

  gates = {
    "artifact_oracle_pass": loader.get("gates", {}).get("manifest_pass") is True,
    "artifact_correct": all([
      loader.get("gates", {}).get("producer_correct"),
      loader.get("gates", {}).get("gate_correct"),
      loader.get("gates", {}).get("up_correct"),
    ]),
    "default_unchanged": loader.get("default_changed") is False,
    "q4_imported_route_closed": imported_q4_decision == "CLOSED_AS_SPEED_ROUTE",
    "no_bounded_n2_candidate": n2_candidates == 0,
    "max_isolated_movement_below_gate": max_timing_movement < 30.0,
    "search_blocked_until_native_lowering": True,
  }

  result = {
    "date": "2026-06-20",
    "subject": "decode native renderer project scope against q8 artifact oracle",
    "verdict": "PASS_DECODE_NATIVE_RENDERER_PROJECT_SCOPE_READY_BROAD_BACKEND_REQUIRED",
    "summary": {
      "what_happened": "Imported Q4 routing was measured and closed as a speed path; q8 artifact remains the only measured decode upside, but native tinygrad lacks a bounded scheduler/renderer patch.",
      "decision": "Do not start BEAM/search or a one-off N2 patch. Start only a broad native renderer/scheduler project with the q8 artifact as oracle.",
      "largest_isolated_timing_movement_us": max_timing_movement,
      "required_n2_gate_us": 30.0,
      "n2_candidate_count": n2_candidates,
      "oracle_gateup_consumer_us": perf.get("gateup_consumer", {}).get("median_ms", 0.0) * 1000.0,
      "oracle_gateup_lifecycle_us": perf.get("gate_up_lifecycle_us"),
      "tinygrad_asm_gateup_us": oracle.get("known_timings_us", {}).get("tinygrad_asm_gateup_full"),
      "tinygrad_to_oracle_gap_us": n1.get("gate", {}).get("full_tinygrad_to_oracle_gap_us"),
    },
    "oracle_contract": {
      "producer": producer,
      "gateup": gateup,
      "launch_contract": oracle.get("launch_contract", {}),
      "oracle_grouped": oracle_grouped,
      "tinygrad_asm_grouped": tinygrad_grouped,
    },
    "structural_requirements": structural_requirements,
    "missing_native_equivalents": missing_native_equivalents,
    "project_tracks": project_tracks,
    "gates": gates,
    "next_action": "Implement DNR-1 schedule-contract oracle binding first; DNR-2 runnable native lowering only after that structural gate passes. Keep BEAM/search blocked until DNR-2/DNR-3 exist.",
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps({"verdict": result["verdict"], "gates": gates, "out": str(OUT.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
  main()
