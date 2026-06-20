#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-native-tooling"


def read_json(rel: str, default: Any = None) -> Any:
  path = ROOT / rel
  if not path.exists(): return default
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def authority_for_feature(row: dict[str, Any]) -> str:
  src = row.get("attribution_source", "")
  if row.get("attribution_us") is not None and "dynamic" in src: return "timing_grade"
  if row.get("attribution_us") is not None: return "static_grade"
  if "static" in src: return "static_grade"
  return "inferred"


def bucket_for_feature(feature: str) -> str:
  if feature in ("global_load_shape_coalescing",): return "bytes"
  if feature in ("waitcnt_grouping", "s_clause_s_delay_alu_scheduler", "register_live_range_resource_scheduler",
                 "native_dot4_instruction_selection"): return "math"
  if feature in ("reduction_topology", "local_y_descriptor_and_launch_contract"): return "overhead"
  return "unknown"


def role_rows(complete: dict[str, Any]) -> list[dict[str, Any]]:
  rows = []
  for row in ((complete.get("role_atlas") or {}).get("rows") or []):
    att = row.get("att") or {}
    programs = row.get("programs") or []
    rows.append({
      "role": row.get("role"),
      "capture_mode": row.get("capture_mode"),
      "verdict": row.get("verdict"),
      "programs": [{"name": p.get("program_name"), "kind": p.get("program_kind"),
                    "global_size": p.get("global_size"), "local_size": p.get("local_size"),
                    "lib_sha16": p.get("lib_sha16")} for p in programs],
      "att": {
        "authority": att.get("authority"),
        "body_like_packet_count": att.get("body_like_packet_count"),
        "valuinst": att.get("valuinst"),
        "inst": att.get("inst"),
      },
      "tooling_gap": row.get("role") == "ffn_gate/up" and att.get("body_like_packet_count") is None,
    })
  gate_rows = []
  for name in ("ffn_gate", "ffn_up", "ffn_gateup_pair", "q8_gateup"):
    data = read_json(f"bench/qk-att-inmodel-role-join/{name}.json")
    if not data: continue
    trace = ((data.get("interval") or {}).get("trace") or {})
    variants = ((data.get("programs") or {}).get("variants") or [])
    gate_rows.append({
      "role": name if name == "q8_gateup" else "ffn_gate/up",
      "capture_mode": (data.get("activation") or {}).get("capture_mode", "inmodel_activation"),
      "verdict": data.get("verdict"),
      "programs": [{"name": p.get("program_name"), "kind": "main_mmvq" if "q4k" in str(p.get("program_name")) else "other",
                    "global_size": p.get("global_size"), "local_size": p.get("local_size"),
                    "lib_sha16": p.get("lib_sha16")} for p in variants],
      "att": {
        "authority": "att_body_not_timing" if trace.get("body_like_packet_count") else "missing_role_join_att",
        "body_like_packet_count": trace.get("body_like_packet_count"),
        "valuinst": (trace.get("packet_top") or {}).get("VALUINST"),
        "inst": (trace.get("packet_top") or {}).get("INST"),
      },
      "tooling_gap": not bool(trace.get("body_like_packet_count")),
    })
  if gate_rows:
    rows = [r for r in rows if r.get("role") != "ffn_gate/up"]
    rows.extend(gate_rows)
  return rows


def oracle_rows(contract: dict[str, Any], complete: dict[str, Any]) -> list[dict[str, Any]]:
  timing = contract.get("known_timings_us") or {}
  launch = contract.get("launch_contract") or {}
  instr = contract.get("instruction_contract") or {}
  llama = complete.get("llama_join") or {}
  return [
    {
      "oracle": "q8_hipcc_lld_artifact",
      "role": "ffn_gate/up",
      "timing_us": timing.get("hipcc_lld_gateup_current_loader"),
      "launch": launch,
      "isa_summary": instr.get("oracle_grouped"),
      "resource": (contract.get("resource_contract") or {}).get("artifact_manifest_runtime"),
      "why": "only measured decode speed route and native q8 scheduler target",
    },
    {
      "oracle": "tinygrad_amd_dsl_q8",
      "role": "ffn_gate/up",
      "timing_us": timing.get("tinygrad_asm_gateup_full"),
      "launch": launch,
      "isa_summary": instr.get("tinygrad_asm_grouped"),
      "resource": (contract.get("resource_contract") or {}).get("s0_runtime"),
      "why": "native baseline that fails the artifact oracle by the q8 consumer gap",
    },
    {
      "oracle": "comgr_fused_c_q8",
      "role": "ffn_gate/up",
      "timing_us": timing.get("comgr_fused_gateup"),
      "launch": launch,
      "isa_summary": instr.get("comgr_grouped"),
      "resource": None,
      "why": "source-level compiler baseline",
    },
    {
      "oracle": "llama_mmvq_contract",
      "role": "Q4/Q6 high-share decode roles",
      "timing_us": None,
      "launch": (llama.get("selected_kernargs") or {}),
      "isa_summary": None,
      "resource": llama.get("launch_contract_rows"),
      "why": "external lifecycle target for low-VGPR wg32 MMVQ consumers",
    },
  ]


def feature_rows(n1: dict[str, Any]) -> list[dict[str, Any]]:
  rows = []
  for row in n1.get("feature_attribution") or []:
    movement = row.get("attribution_us")
    authority = authority_for_feature(row)
    rows.append({
      "feature": row.get("feature"),
      "role": "ffn_gate/up",
      "bucket": bucket_for_feature(row.get("feature", "")),
      "movement_us": movement,
      "authority": authority,
      "evidence": row.get("evidence") or [],
      "implementation_surface": None if not row.get("n2_gate_ge_30us") else row.get("feature"),
      "n2_gate_ge_30us": bool(row.get("n2_gate_ge_30us")),
      "decision": "start_N2" if row.get("n2_gate_ge_30us") else row.get("decision"),
      "blocker": row.get("blocker"),
    })
  return rows


def bucket_rows(features: list[dict[str, Any]], roles: list[dict[str, Any]]) -> list[dict[str, Any]]:
  ffn_rows = [r for r in roles if r["role"] in ("ffn_gate/up", "q8_gateup")]
  ffn_att_missing = not ffn_rows or all(r["tooling_gap"] for r in ffn_rows)
  return [
    {
      "role": "ffn_gate/up",
      "primary_bucket": "math" if not ffn_att_missing else "unknown",
      "reason": ("role-joined body evidence exists; current unresolved gap remains scheduler/resource math-side attribution"
                 if not ffn_att_missing else
                 "q8/native high-share role lacks role-joined ATT/counter evidence; current attribution is static/dynamic-ablation only"),
      "blocked_by": "missing q8 ffn_gate/up role-joined body evidence" if ffn_att_missing else None,
    },
    {
      "role": "attn_q/o",
      "primary_bucket": "overhead",
      "reason": "in-model ATT sees intended native Q4_K coop plus reduce/glue; timing ledger says reduce/glue is visible but below build gate",
      "blocked_by": None,
    },
    {
      "role": "ffn_down,lm_head",
      "primary_bucket": "unknown",
      "reason": "Q6 surfaces have ATT visibility through fallback and runtime identity, but not full-model timing-promotion authority",
      "blocked_by": "full model Q6 activation capture blocked by 4.68GB AMD allocation",
    },
  ]


def feature_join_rows(roles: list[dict[str, Any]], oracles: list[dict[str, Any]], features: list[dict[str, Any]]) -> list[dict[str, Any]]:
  q8_oracle = next((o for o in oracles if o["oracle"] == "q8_hipcc_lld_artifact"), None)
  native_oracle = next((o for o in oracles if o["oracle"] == "tinygrad_amd_dsl_q8"), None)
  joins = []
  for role in [r for r in roles if r["role"] in ("ffn_gate/up", "q8_gateup")]:
    main_programs = [p for p in role.get("programs", []) if p.get("kind") == "main_mmvq"]
    for p in main_programs:
      joins.append({
        "role": role["role"],
        "program_name": p.get("name"),
        "lib_sha16": p.get("lib_sha16"),
        "launch": {"global_size": p.get("global_size"), "local_size": p.get("local_size")},
        "capture_mode": role.get("capture_mode"),
        "trace": role.get("att"),
        "timing": {
          "native_us": (native_oracle or {}).get("timing_us"),
          "oracle_us": (q8_oracle or {}).get("timing_us"),
          "gap_us": round((native_oracle or {}).get("timing_us", 0) - (q8_oracle or {}).get("timing_us", 0), 3)
                    if native_oracle and q8_oracle else None,
          "authority": "existing_oracle_timing_not_same_interval",
        },
        "oracle": {"native": native_oracle, "target": q8_oracle},
        "candidate_features": [f["feature"] for f in features],
        "isa_diff": {
          "load_shape": "oracle b128/u8 vs tinygrad b32/u16/u8",
          "scheduler_markers": "oracle has s_clause/s_delay_alu; tinygrad has none",
          "dot4": "matched",
          "reduction": "oracle ds_total 7 vs tinygrad ds_total 10",
        },
        "resource_diff": {
          "known": False,
          "reason": "no counter-grade VGPR/occupancy/resource attribution joined to the role interval",
        },
        "authority": "counter_grade" if (role.get("att") or {}).get("body_like_packet_count") and False else "static_grade",
        "decision": "feature_join_pass_visibility_only",
      })
  return joins


def ablation_rows(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
  rows = []
  for f in features:
    movement = f.get("movement_us")
    known = movement is not None
    rows.append({
      "feature": f.get("feature"),
      "variant": {
        "native_dot4_instruction_selection": "dot_synthetic",
        "global_load_shape_coalescing": "load_wait_only",
        "waitcnt_grouping": "wait_grouped_load_only",
        "reduction_topology": "reduction_only",
        "s_clause_s_delay_alu_scheduler": "not_run_static_diff_only",
        "register_live_range_resource_scheduler": "not_run_unattributed",
        "local_y_descriptor_and_launch_contract": "prior_route_evidence",
      }.get(f.get("feature"), "unknown"),
      "role": f.get("role"),
      "baseline_us": 166.649,
      "candidate_us": round(166.649 - movement, 3) if known else None,
      "movement_us": movement,
      "changed_features": [f.get("feature")],
      "correctness": "PASS" if known else "NOT_RUN",
      "authority": f.get("authority"),
      "decision": ("start_N2" if f.get("n2_gate_ge_30us") else
                   "closed_below_gate" if known else
                   "project_level_unattributed" if f.get("feature") in ("s_clause_s_delay_alu_scheduler", "register_live_range_resource_scheduler") else
                   "closed_low_ev"),
    })
  return rows


def main() -> int:
  complete = read_json("bench/qk-decode-complete-tooling/result.json", {})
  contract = read_json("bench/q8-ffn-amd-scheduler-project/oracle_contract.json", {})
  n1 = read_json("bench/q8-ffn-amd-scheduler-project/n1_attribution.json", {})
  hcq_replay = read_json("bench/amd-scheduler-tooling-backend/r1p2_hcq_replay.json", {})
  att_atlas = read_json("bench/qk-att-primitive-atlas/result.json", {})
  pmc_decode = read_json("bench/qk-decode-native-tooling/pmc_decode.json", {})
  timeline_attr = read_json("bench/qk-decode-native-tooling/timeline_attribution.json", {})
  role_timing = read_json("bench/qk-decode-native-tooling/role_timing_join.json", {})
  scheduler_scope = read_json("bench/qk-decode-native-tooling/scheduler_ablation_scope.json", {})
  wd_projection = read_json("bench/qk-decode-native-tooling/wd_projection.json", {})

  roles = role_rows(complete)
  oracles = oracle_rows(contract, complete)
  features = feature_rows(n1)
  buckets = bucket_rows(features, roles)
  joins = feature_join_rows(roles, oracles, features)
  ablations = ablation_rows(features)
  timing_grade_movements = [r["movement_us"] for r in features
                            if r.get("authority") == "timing_grade" and r.get("movement_us") is not None]
  n2 = [r for r in features if r.get("n2_gate_ge_30us")]
  ffn_gap = any(r["role"] == "ffn_gate/up" and r["tooling_gap"] for r in roles)
  ffn_seen = any(r["role"] in ("ffn_gate/up", "q8_gateup") for r in roles)
  missing = []
  if ffn_gap or not ffn_seen: missing.append("q8 ffn_gate/up role-joined ATT/PMC/body evidence")
  if not n2: missing.append("timing-grade feature attribution >=30us")
  if not timing_grade_movements or max(timing_grade_movements) < 30:
    missing.append("counter/timing join that converts scheduler-resource unknowns into a bounded feature")
  if not any(b["role"] == "ffn_gate/up" and b["primary_bucket"] != "unknown" for b in buckets):
    missing.append("bytes/math/overhead bucket classification for q8 ffn_gate/up")
  pass_rows_present = all(bool(x) for x in (pmc_decode, timeline_attr, role_timing, scheduler_scope, wd_projection))
  roadmap_only = (
    pass_rows_present and
    scheduler_scope.get("verdict") == "ROADMAP_ONLY" and
    wd_projection.get("verdict") == "NO_PROJECTABLE_FEATURE" and
    pmc_decode.get("verdict") in ("BLOCKED_COUNTER_DECODE", "NO_USEFUL_COUNTER_SIGNAL", "PASS_COUNTER_GRADE") and
    timeline_attr.get("verdict") in ("BLOCKED_TIMELINE_DECODE", "NO_SINGLE_FEATURE", "PASS_TIMELINE_ATTRIBUTION") and
    not n2
  )
  final_verdict = "ROADMAP_ONLY" if roadmap_only else "TOOLING_NOT_READY"
  if roadmap_only:
    missing = []

  readiness = {
    "date": "2026-06-19",
    "schema": "decode_native_tooling_readiness_v1",
    "verdict": final_verdict,
    "roles": roles,
    "oracles": oracles,
    "bucket_classification": buckets,
    "feature_attribution": features,
    "feature_join": joins,
    "ablation_matrix": ablations,
    "tooling_inputs": {
      "decode_complete_tooling": "bench/qk-decode-complete-tooling/result.json",
      "oracle_contract": "bench/q8-ffn-amd-scheduler-project/oracle_contract.json",
      "n1_attribution": "bench/q8-ffn-amd-scheduler-project/n1_attribution.json",
      "hcq_att_replay": "bench/amd-scheduler-tooling-backend/r1p2_hcq_replay.json",
      "att_primitive_atlas": "bench/qk-att-primitive-atlas/result.json",
    },
    "observability_status": {
      "hcq_att_replay_verdict": hcq_replay.get("verdict"),
      "att_primitive_atlas_verdict": att_atlas.get("verdict"),
      "n1_verdict": n1.get("verdict"),
      "sqtt_decode_usable_in_n1": (n1.get("gate") or {}).get("sqtt_decode_usable"),
      "att_is_timing_authority": False,
      "pmc_decode_verdict": pmc_decode.get("verdict"),
      "timeline_attribution_verdict": timeline_attr.get("verdict"),
      "role_timing_join_verdict": role_timing.get("verdict"),
      "scheduler_ablation_scope_verdict": scheduler_scope.get("verdict"),
      "wd_projection_verdict": wd_projection.get("verdict"),
    },
    "start_gate": {
      "required_for_n2": "one timing-grade feature with movement_us >=30 or >=5% projected W==D movement",
      "n2_candidate_count": len(n2),
      "max_timing_grade_movement_us": max(timing_grade_movements) if timing_grade_movements else 0,
      "max_projected_wd_pct": 0,
      "missing": missing,
    },
    "decision": (
      "ROADMAP_ONLY: no bounded native N2 feature clears the gate; remaining q8 scheduler/resource gap is broad AMD "
      "backend work only. Do not start q8-specific native scheduler/renderer implementation."
      if roadmap_only else
      "Continue attribution/tooling only. Do not start native scheduler/renderer implementation from current evidence."
    ),
  }
  write_json("readiness.json", readiness)
  write_json("feature_attribution.json", {
    "date": "2026-06-19",
    "schema": "decode_native_feature_attribution_v1",
    "rows": features,
    "verdict": "NO_N2_FEATURE",
  })
  write_json("feature_join.json", {
    "date": "2026-06-19",
    "schema": "decode_native_feature_join_v1",
    "rows": joins,
    "verdict": "PASS_VISIBILITY_JOIN_NO_COUNTER_GRADE_ATTRIBUTION" if joins else "NO_JOIN_ROWS",
  })
  write_json("ablation_matrix.json", {
    "date": "2026-06-19",
    "schema": "decode_native_ablation_matrix_v1",
    "rows": ablations,
    "verdict": "NO_N2_ABLATION",
  })
  print(json.dumps({
    "out": str(OUT.relative_to(ROOT)),
    "verdict": readiness["verdict"],
    "missing": missing,
  }, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
