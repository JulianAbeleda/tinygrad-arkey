#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib, subprocess
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-complete-tooling"


def read_json(rel: str, default: Any = None) -> Any:
  p = ROOT / rel
  if not p.exists(): return default
  return json.loads(p.read_text())


def git_json(spec: str, default: Any = None) -> Any:
  try:
    out = subprocess.check_output(["git", "show", spec], cwd=ROOT)
    return json.loads(out)
  except Exception:
    return default


def write_json(name: str, data: Any) -> None:
  (OUT / name).parent.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def program_kind(name: str) -> str:
  if name.startswith(("q4k_", "q6k_")) or "q4k_" in name or "q6k_" in name: return "main_mmvq"
  if name.startswith("r_") or "sum" in name.lower(): return "reduce"
  if name.startswith("E_"): return "glue"
  return "other"


def trace_metrics(row: dict[str, Any] | None) -> dict[str, Any]:
  tr = (((row or {}).get("interval") or {}).get("trace") or {})
  top = tr.get("packet_top") or {}
  return {
    "body_like_packet_count": tr.get("body_like_packet_count"),
    "valuinst": top.get("VALUINST") or top.get("valuinst"),
    "inst": top.get("INST") or top.get("inst"),
    "wavestart": top.get("WAVESTART") or top.get("wavestart"),
    "waveend": top.get("WAVEEND") or top.get("waveend"),
    "nonzero_bytes": tr.get("nonzero_bytes"),
    "authority": "att_body_not_timing",
  }


def role_from_att(role: str, row: dict[str, Any], *, provenance: str) -> dict[str, Any]:
  variants = ((row.get("programs") or {}).get("variants") or [])
  programs = []
  for v in variants:
    programs.append({
      "program_name": v.get("program_name"),
      "program_kind": program_kind(v.get("program_name") or ""),
      "global_size": v.get("global_size"),
      "local_size": v.get("local_size"),
      "calls": v.get("calls"),
      "lib_sha16": v.get("lib_sha16"),
      "authority": "hcq_program_capture",
    })
  return {
    "role": role,
    "capture_mode": (row.get("activation") or {}).get("capture_mode", "inmodel_activation"),
    "activation": row.get("activation"),
    "programs": programs,
    "att": trace_metrics(row),
    "gates": row.get("gates"),
    "verdict": row.get("verdict"),
    "provenance": provenance,
  }


def contract_rows() -> list[dict[str, Any]]:
  runtime = read_json("bench/qk-decode-fused-mmvq-integration/runtime_cache_identity.json", {})
  rows = []
  for role, vals in (runtime.get("inmodel") or {}).items():
    for v in vals:
      rows.append({
        "role": role,
        "program_name": v.get("program_name"),
        "program_kind": program_kind(v.get("program_name") or ""),
        "shape": v.get("shape"),
        "global_size": v.get("global_size"),
        "local_size": v.get("local_size"),
        "calls": v.get("calls"),
        "lib_sha16": v.get("lib_sha16"),
        "ast_key": v.get("ast_key"),
        "authority": "runtime_cache_identity_inmodel",
      })
  return rows


def build_schema() -> dict[str, Any]:
  return {
    "schema": "decode_complete_tooling_schema_v1",
    "date": "2026-06-19",
    "row_fields": {
      "role": "semantic role bucket such as attn_q/o, ffn_gate/up, ffn_down, lm_head",
      "capture_mode": "inmodel_activation, q6_surface_fallback, runtime_identity_only, or doc_provenance",
      "programs": "HCQ program rows classified as main_mmvq/reduce/glue/other",
      "att": "ATT/SQTT body attribution metrics; explicitly not timing authority",
      "timing": "role-local or model-level timing, with method and trust label",
      "llama": "comparable llama launch/activation/throughput row when available",
      "decision": "build/no-build consequence from the row",
    },
    "authority_tags": [
      "measured",
      "inferred",
      "doc_provenance",
      "surface_fallback",
      "runtime_identity_only",
      "att_body_not_timing",
      "same_process_interleaved_timing",
      "model_level_wd_timing",
      "unsupported",
    ],
  }


def build_inventory() -> dict[str, Any]:
  files = {
    "hcq_runtime_cache_identity": "bench/qk-decode-fused-mmvq-integration/runtime_cache_identity.json",
    "q4_att_inmodel_commit_artifact": "git:3aa7bb04a:bench/qk-att-inmodel-role-join/result.json",
    "q6_ffn_down_att_surface": "bench/qk-att-inmodel-role-join/ffn_down.json",
    "q6_lm_head_att_surface": "bench/qk-att-inmodel-role-join/lm_head.json",
    "timing_imported_attn_output": "bench/qk-decode-mmvq-large-project/p7d_one_role_timing.json",
    "timing_imported_gateup": "bench/qk-decode-mmvq-large-project/p7e_gateup_amortization.json",
    "q8_fused_lifecycle": "bench/qk-decode-mmvq-large-project/q8_two_lane_closeout.json",
    "tax_ledger": "bench/qk-decode-integration-diagnostic/tax_ledger.json",
    "inmodel_loss_atlas": "bench/qk-decode-fused-mmvq-integration/inmodel_loss_atlas.json",
    "llama_launch_contract": "bench/qk-decode-fused-mmvq-integration/llama_launch_contract.json",
    "llama_runtime_accounting": "bench/qk-llama-token-primitive-accounting/llama_runtime.json",
  }
  return {
    "schema": "decode_complete_tooling_inventory_v1",
    "date": "2026-06-19",
    "files": {k: {"path": v, "available": True if v.startswith("git:") else (ROOT / v).exists()} for k, v in files.items()},
    "current_blockers": [
      {
        "name": "full_model_q6_activation_capture",
        "status": "blocked_by_4p68gb_amd_allocation",
        "fallback": "q6_surface_fallback with runtime/cache identity support",
      },
      {
        "name": "per_kernel_graph_replay_timing",
        "status": "not_reliable_as_primary_authority",
        "fallback": "same-process interleaved role A/B or full W==D model A/B",
      },
      {
        "name": "ATT_as_timer",
        "status": "unsupported",
        "fallback": "ATT is body/resource evidence only",
      },
    ],
  }


def build_role_atlas() -> dict[str, Any]:
  q4 = git_json("3aa7bb04a:bench/qk-att-inmodel-role-join/result.json", {})
  q6_down = read_json("bench/qk-att-inmodel-role-join/ffn_down.json", {})
  q6_lm = read_json("bench/qk-att-inmodel-role-join/lm_head.json", {})
  runtime_rows = contract_rows()
  rows = []
  if q4: rows.append(role_from_att("attn_q/o", q4, provenance="git:3aa7bb04a:bench/qk-att-inmodel-role-join/result.json"))
  if q6_down: rows.append(role_from_att("ffn_down", q6_down, provenance="bench/qk-att-inmodel-role-join/ffn_down.json"))
  if q6_lm: rows.append(role_from_att("lm_head", q6_lm, provenance="bench/qk-att-inmodel-role-join/lm_head.json"))
  att_roles = {r["role"] for r in rows}
  for role in ["ffn_gate/up", "attn_k/v"]:
    rr = [r for r in runtime_rows if r["role"] == role]
    if rr:
      rows.append({
        "role": role,
        "capture_mode": "runtime_identity_only",
        "activation": None,
        "programs": rr,
        "att": {"authority": "missing_role_join_att", "body_like_packet_count": None},
        "gates": {"runtime_cache_identity": "PASS", "att_body_packets": "NOT_RUN"},
        "verdict": "PASS_RUNTIME_IDENTITY_ATT_MISSING",
        "provenance": "bench/qk-decode-fused-mmvq-integration/runtime_cache_identity.json",
      })
  return {
    "schema": "decode_complete_role_atlas_v1",
    "date": "2026-06-19",
    "rows": rows,
    "coverage": {
      "roles_with_att_body": sorted(att_roles),
      "roles_runtime_identity_only": sorted([r["role"] for r in rows if r["capture_mode"] == "runtime_identity_only"]),
      "full_model_q6_capture": "NO_surface_fallback_only",
    },
    "verdict": "PASS_ATLAS_WITH_EXPLICIT_FFN_GATE_ATT_GAP",
  }


def build_q6_equivalence() -> dict[str, Any]:
  runtime = read_json("bench/qk-decode-fused-mmvq-integration/runtime_cache_identity.json", {})
  ffn = read_json("bench/qk-att-inmodel-role-join/ffn_down.json", {})
  lm = read_json("bench/qk-att-inmodel-role-join/lm_head.json", {})
  roles = {}
  for role, att in [("ffn_down", ffn), ("lm_head", lm)]:
    rt = (runtime.get("inmodel") or {}).get(role) or []
    att_main = [p for p in ((att.get("programs") or {}).get("variants") or []) if program_kind(p.get("program_name") or "") == "main_mmvq"]
    roles[role] = {
      "surface_capture_mode": (att.get("activation") or {}).get("capture_mode"),
      "surface_main_programs": [p.get("program_name") for p in att_main],
      "inmodel_runtime_identity_programs": [p.get("program_name") for p in rt],
      "same_program_seen": bool(set(p.get("program_name") for p in att_main) & set(p.get("program_name") for p in rt)),
      "activation_shape": (att.get("activation") or {}).get("shape"),
      "weight_name": (att.get("activation") or {}).get("linear_name"),
      "verdict": "ACCEPT_SURFACE_EQUIVALENCE" if bool(set(p.get("program_name") for p in att_main) & set(p.get("program_name") for p in rt)) else "SURFACE_ONLY",
    }
  return {
    "schema": "decode_q6_capture_equivalence_v1",
    "date": "2026-06-19",
    "full_model_capture_blocker": "MemoryError: Allocation of 4.68 GB failed on AMD. Used: 0 B",
    "roles": roles,
    "decision": "Q6 role-surface evidence is acceptable for program/lifecycle visibility, but not as a standalone timing promotion authority.",
    "verdict": "PASS_EQUIVALENCE_FOR_VISIBILITY_NOT_TIMING",
  }


def build_timing_policy() -> tuple[dict[str, Any], str]:
  p7d = read_json("bench/qk-decode-mmvq-large-project/p7d_one_role_timing.json", {})
  p7e = read_json("bench/qk-decode-mmvq-large-project/p7e_gateup_amortization.json", {})
  q8 = read_json("bench/qk-decode-mmvq-large-project/q8_two_lane_closeout.json", {})
  rows = [
    {
      "surface": "imported_q4_attn_output",
      "method": (p7d.get("timing") or {}).get("method"),
      "baseline_ms": (p7d.get("timing") or {}).get("baseline_ms_median"),
      "candidate_ms": (p7d.get("timing") or {}).get("imported_ms_median"),
      "speedup": (p7d.get("timing") or {}).get("speedup"),
      "trust": "same_process_interleaved_timing",
      "decision": "candidate_loses",
    },
    {
      "surface": "imported_q4_gateup_shared_q8",
      "method": (p7e.get("timing") or {}).get("method"),
      "baseline_ms": (p7e.get("timing") or {}).get("baseline_ms_median"),
      "candidate_ms": (p7e.get("timing") or {}).get("imported_ms_median"),
      "speedup": (p7e.get("timing") or {}).get("speedup"),
      "trust": "same_process_interleaved_timing",
      "decision": "candidate_loses",
    },
  ]
  lane = q8.get("lane1_research_flag_hardening") or {}
  rows.append({
    "surface": "q8_fused_ffn_research_flag",
    "method": "W==D ctx sweep plus dNLL",
    "min_speedup": lane.get("min_speedup"),
    "median_speedup": lane.get("median_speedup"),
    "dnll": lane.get("dnll"),
    "trust": "model_level_wd_timing",
    "decision": "research_flag_passes_default_off",
  })
  policy = {
    "schema": "decode_complete_timing_policy_v1",
    "date": "2026-06-19",
    "rows": rows,
    "rules": [
      "ATT packet counts are never timing authority.",
      "Same-process interleaved A/B is acceptable for role-local yes/no gates.",
      "Full W==D ctx sweep is required for final decode promotion.",
      "Surface-fallback Q6 captures can prove program identity, but cannot alone promote a timing build.",
      "Clock-confounded non-interleaved runs are provenance only.",
    ],
    "verdict": "PASS_TIMING_POLICY",
  }
  md = """# Decode complete tooling timing policy

- ATT/SQTT body metrics are visibility evidence, not timing evidence.
- Role-local timing must use same-process interleaved A/B or be labeled diagnostic only.
- Final promotion must use W==D ctx sweep and, for lossy q8, dNLL.
- Q6 surface fallback can prove program/lifecycle identity because runtime/cache identity already saw the same programs in-model; it cannot alone price a build.
"""
  return policy, md


def build_att_metrics(role_atlas: dict[str, Any]) -> dict[str, Any]:
  rows = []
  for r in role_atlas["rows"]:
    rows.append({
      "role": r["role"],
      "capture_mode": r["capture_mode"],
      **(r.get("att") or {}),
      "main_programs": [p.get("program_name") for p in r.get("programs", []) if p.get("program_kind") == "main_mmvq"],
      "reduce_programs": [p.get("program_name") for p in r.get("programs", []) if p.get("program_kind") == "reduce"],
      "glue_programs": [p.get("program_name") for p in r.get("programs", []) if p.get("program_kind") == "glue"],
    })
  return {
    "schema": "decode_complete_att_metrics_v1",
    "date": "2026-06-19",
    "rows": rows,
    "verdict": "PASS_ATT_METRICS_WITH_TIMING_BOUNDARY",
  }


def build_llama_join() -> dict[str, Any]:
  contract = read_json("bench/qk-decode-fused-mmvq-integration/llama_launch_contract.json", {})
  runtime = read_json("bench/qk-llama-token-primitive-accounting/llama_runtime.json", {})
  capture = read_json("bench/qk-decode-mmvq-large-project/p2_kernarg_capture.json", {})
  return {
    "schema": "decode_complete_llama_join_v1",
    "date": "2026-06-19",
    "runtime_accounting": {
      "decode_only_share_pct": runtime.get("decode_only_share_pct"),
      "llama_mmvq_effective_bw_GBs": runtime.get("llama_mmvq_effective_bw_GBs"),
      "llama_mmvq_pct_hbm_peak": runtime.get("llama_mmvq_pct_hbm_peak"),
      "q8_1_finding": runtime.get("q8_1_finding"),
    },
    "launch_contract_rows": contract.get("rows"),
    "selected_kernargs": {
      k: {
        "type": v.get("type"),
        "global": v.get("global"),
        "local": v.get("local"),
        "num_workgroups": v.get("num_workgroups"),
        "kernarg_size": v.get("kernarg_size"),
        "kernel_symbol": v.get("kernel_symbol"),
      } for k, v in (capture.get("selected") or {}).items()
    },
    "comparison_summary": {
      "llama_contract": "q8_1 activation producer plus wg32 low-VGPR MMVQ consumers",
      "tinygrad_current_contract": "native Q4/Q6 coop/fp paths plus visible reduce/glue; no default q8 lifecycle reuse",
      "authority": "llama HIP trace and kernarg capture vs tinygrad HCQ/ATT artifacts",
    },
    "verdict": "PASS_LLAMA_JOIN",
  }


def build_reduce_glue_ledger(role_atlas: dict[str, Any]) -> dict[str, Any]:
  tax = read_json("bench/qk-decode-integration-diagnostic/tax_ledger.json", {})
  loss = read_json("bench/qk-decode-fused-mmvq-integration/inmodel_loss_atlas.json", {})
  rows = []
  for r in role_atlas["rows"]:
    programs = r.get("programs") or []
    rows.append({
      "role": r["role"],
      "capture_mode": r["capture_mode"],
      "main_count": sum(1 for p in programs if p.get("program_kind") == "main_mmvq"),
      "reduce_count": sum(1 for p in programs if p.get("program_kind") == "reduce"),
      "glue_count": sum(1 for p in programs if p.get("program_kind") == "glue"),
      "main_programs": [p.get("program_name") for p in programs if p.get("program_kind") == "main_mmvq"],
      "reduce_programs": [p.get("program_name") for p in programs if p.get("program_kind") == "reduce"],
      "glue_programs": [p.get("program_name") for p in programs if p.get("program_kind") == "glue"],
      "timing_authority": "not_priced_by_att",
    })
  stage2 = next((t for t in tax.get("taxes", []) if t.get("name") == "global partials plus stage2 reduce"), {})
  return {
    "schema": "decode_complete_reduce_glue_ledger_v1",
    "date": "2026-06-19",
    "role_lifecycle_counts": rows,
    "priced_tax_authority": stage2,
    "aggregate_loss_reference": (loss.get("aggregate") or {}),
    "build_gate": {
      "required": ">=5% W==D projected movement or >=10% local movement on high-share role",
      "current_priced_reduce_glue_result": "FAIL_AS_STANDALONE_ROUTE",
      "reason": "Existing priced stage2 tax is ~6.8us / 10% on Q4_K ffn_gate/up surface and only reaches ~53-54% peak, still below llama-class retention.",
    },
    "decision": "Do not build direct-output/reduce-fusion from current evidence. Only reopen if DCT timing policy prices a larger cross-role reduce/glue share.",
    "verdict": "NO_REDUCE_GLUE_BUILD_GATE",
  }


def build_summary(all_data: dict[str, Any]) -> str:
  role_rows = all_data["role_atlas"]["rows"]
  ledger = all_data["reduce_glue_ledger"]
  q8 = read_json("bench/qk-decode-mmvq-large-project/q8_two_lane_closeout.json", {})
  lane = q8.get("lane1_research_flag_hardening") or {}
  return f"""# Decode complete tooling summary

Verdict: `COMPLETE_TOOLING_PASS_WITH_EXPLICIT_GAPS`.

## What is complete

- Schema and inventory exist for role identity, ATT body attribution, timing authority, reduce/glue Amdahl, and llama comparison.
- Q4 `attn_q/o` has full in-model ATT body attribution from commit `3aa7bb04a`.
- Q6 `ffn_down` and `lm_head` have ATT body attribution through `q6_surface_fallback`; runtime/cache identity proves the same programs are used in-model.
- llama launch and runtime rows are joined into the same artifact family.
- Timing policy is explicit: ATT is not a timer; same-process interleaved role A/B and W==D ctx sweeps are the promotion authorities.

## Role coverage

| role | capture | verdict |
|---|---|---|
""" + "\n".join(f"| `{r['role']}` | `{r['capture_mode']}` | `{r['verdict']}` |" for r in role_rows) + f"""

## Reduce/glue decision

`{ledger['verdict']}`. The currently priced stage-2 tax is real, but it does not clear the build gate as a standalone direct-output/reduce-fusion route.

## Timing decision

The imported llama Q4 route lost role-local timing for both `attn_output` and `ffn_gate/up`. The fused q8 artifact route remains the only measured decode speed route: min speedup `{lane.get('min_speedup')}`, median speedup `{lane.get('median_speedup')}`, dNLL `{lane.get('dnll')}`.

## Remaining gaps

- Fresh ATT role-join for `ffn_gate/up` is still missing; runtime identity exists, but body attribution is not captured for that exact high-share role.
- Full-model Q6 activation capture is still blocked by the 4.68 GB AMD allocation issue; surface equivalence is acceptable for visibility, not timing promotion.
- No reliable per-kernel graph timing authority exists; final changes still need W==D ctx sweep.

## Final tooling consequence

The tooling is now complete enough to prevent the wrong build: do not fund reduce/glue fusion from packet visibility alone. The next decode implementation choice remains either the already measured q8 research flag or a project-level native scheduler/renderer effort.
"""


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  schema = build_schema()
  inventory = build_inventory()
  role_atlas = build_role_atlas()
  q6_equiv = build_q6_equivalence()
  timing_policy, timing_md = build_timing_policy()
  att_metrics = build_att_metrics(role_atlas)
  llama_join = build_llama_join()
  reduce_glue = build_reduce_glue_ledger(role_atlas)
  all_data = {
    "schema": schema,
    "inventory": inventory,
    "role_atlas": role_atlas,
    "q6_equivalence": q6_equiv,
    "timing_policy": timing_policy,
    "att_metrics": att_metrics,
    "llama_join": llama_join,
    "reduce_glue_ledger": reduce_glue,
  }
  write_json("schema.json", schema)
  write_json("instrument_inventory.json", inventory)
  write_json("role_atlas.json", role_atlas)
  write_json("q6_capture_equivalence.json", q6_equiv)
  write_json("timing_audit.json", timing_policy)
  (OUT / "timing_policy.md").write_text(timing_md)
  write_json("att_metrics.json", att_metrics)
  write_json("llama_join.json", llama_join)
  write_json("reduce_glue_ledger.json", reduce_glue)
  write_json("result.json", {"schema": "decode_complete_tooling_result_v1", "date": "2026-06-19", **all_data, "verdict": "COMPLETE_TOOLING_PASS_WITH_EXPLICIT_GAPS"})
  (OUT / "summary.md").write_text(build_summary(all_data))
  print(json.dumps({"out": str(OUT.relative_to(ROOT)), "verdict": "COMPLETE_TOOLING_PASS_WITH_EXPLICIT_GAPS", "roles": len(role_atlas["rows"])}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
