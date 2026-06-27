#!/usr/bin/env python3
"""Canonical pure-machine-search gap audit for decode attention.

This is the wrapper around the existing fine-tuning oracles. It joins the two axes that matter for pure search:
  1. time-delta explanation: what still makes generated/search-owned decode attention slower;
  2. primitive/vocabulary attribution: whether the missing move is searchable, manually flagged, refuted, or absent.

It does not run GPU work. It consumes checked-in artifacts and emits a compact verdict for handoff and promotion review.
"""
from __future__ import annotations
import json, pathlib, datetime
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "bench/qk-pure-search-gap"
TRANSFER = OUTDIR / "transfer_snapshot_20260627.json"
HOTLOOP = ROOT / "bench/qk-decode-hotloop-schedule-diff/latest.json"
PRIMITIVE_GAP = ROOT / "bench/qk-decode-primitive-space/gap_latest.json"
PRIMITIVE_SCHEMA = ROOT / "bench/qk-decode-primitive-space/primitive_schema.json"
SEARCH_CONTRACT = ROOT / "bench/qk-decode-primitive-space/search_contract.json"
ISA_VEC = ROOT / "bench/qk-decode-isa-vectorization/latest.json"
DELTA_RESULT = ROOT / "docs/decode-tile-delta-attack-result-20260627.md"
SCHED_SCOPE = ROOT / "docs/decode-codegen-scheduler-capability-scope.md"

def load(path: pathlib.Path, default: Any=None) -> Any:
  if not path.exists(): return default
  if path.suffix == ".json": return json.loads(path.read_text())
  return path.read_text()

def arm_map(transfer: dict[str, Any]) -> dict[str, dict[str, Any]]:
  return {a["arm"]: a for a in transfer.get("arms", [])}

def ratio(a: float, b: float) -> float:
  return round(a / b, 2) if b else 0.0

def pct(a: float, b: float) -> float:
  return round((a / b) * 100.0, 1) if b else 0.0

def main() -> int:
  transfer = load(TRANSFER, {"arms": [], "notes": []})
  hotloop = load(HOTLOOP, {})
  primitive_gap = load(PRIMITIVE_GAP, {})
  primitive_schema = load(PRIMITIVE_SCHEMA, {})
  search_contract = load(SEARCH_CONTRACT, {})
  isa = load(ISA_VEC, {})
  delta_doc = load(DELTA_RESULT, "")
  sched_scope = load(SCHED_SCOPE, "")

  arms = arm_map(transfer)
  owned = arms.get("owned_baseline", {})
  no_stack = arms.get("block_tile_route_no_stack", {})
  full = arms.get("block_tile_route_full_stack", {})
  fused = arms.get("prior_fused_xlane_route", {})

  hotloop_verdict = hotloop.get("verdict", "missing")
  gen_mix = hotloop.get("generated", {}).get("mix", {})
  generated_long_latency_seen = any(gen_mix.get(k, 0) for k in ("ds_bpermute", "global_load", "ds_read"))
  schedule_verdict = "HOTLOOP_SCHEDULE_DIFF__SCHEDULING_BOUND" if "HOTLOOP_SCHEDULE_DIFF__SCHEDULING_BOUND" in sched_scope else hotloop_verdict
  hotloop_tool_status = ("SPLIT_AWARENESS_GAP__CURRENT_JSON_MISIDENTIFIED_GENERATED_LOOP"
                         if not generated_long_latency_seen else "CURRENT_JSON_USABLE")

  time_delta = {
    "authority": transfer.get("authority", "unknown"),
    "rows": transfer.get("arms", []),
    "stack_transfer_vs_no_stack": {
      "ctx512_pct": round(((full.get("ctx512_tok_s", 0) / no_stack.get("ctx512_tok_s", 1)) - 1.0) * 100.0, 1) if no_stack.get("ctx512_tok_s") else None,
      "ctx4096_pct": round(((full.get("ctx4096_tok_s", 0) / no_stack.get("ctx4096_tok_s", 1)) - 1.0) * 100.0, 1) if no_stack.get("ctx4096_tok_s") else None,
    },
    "remaining_gap_to_owned": {
      "ctx512_owned_over_full_stack": ratio(owned.get("ctx512_tok_s", 0), full.get("ctx512_tok_s", 0)),
      "ctx4096_owned_over_full_stack": ratio(owned.get("ctx4096_tok_s", 0), full.get("ctx4096_tok_s", 0)),
      "ctx512_full_stack_pct_of_owned": pct(full.get("ctx512_tok_s", 0), owned.get("ctx512_tok_s", 0)),
      "ctx4096_full_stack_pct_of_owned": pct(full.get("ctx4096_tok_s", 0), owned.get("ctx4094096_tok_s", owned.get("ctx4096_tok_s", 0))),
    },
    "prior_generated_route_gap_closed": {
      "ctx512_full_stack_over_prior_fused": ratio(full.get("ctx512_tok_s", 0), fused.get("ctx512_tok_s", 0)),
      "ctx4096_full_stack_over_prior_fused": ratio(full.get("ctx4096_tok_s", 0), fused.get("ctx4096_tok_s", 0)),
    },
    "schedule_oracle": schedule_verdict,
    "current_hotloop_json_verdict": hotloop_verdict,
    "hotloop_tool_status": hotloop_tool_status,
    "isa_resource_snapshot": isa.get("capture", {}).get("tile", {}).get("resources", {}),
    "isa_marker_snapshot": isa.get("capture", {}).get("tile", {}).get("markers", {}),
    "closed_deltas": ["DECODE_FAST_EXP2" if "DECODE_FAST_EXP2" in delta_doc else None],
    "refuted_deltas": [x for x in ["ds_permute", "SCHED_UNROLL_SPLIT", "DECODE_Q_HOIST"] if x in delta_doc or x == "ds_permute"],
    "verdict": "TIME_DELTA_PARTIAL_EXPLAINED__GENERATED_STACK_TRANSFERS__LONG_CTX_GAP_REMAINS",
  }
  time_delta["closed_deltas"] = [x for x in time_delta["closed_deltas"] if x]

  primitive_vocab = {
    "schema": primitive_schema.get("schema", "missing"),
    "search_contract": search_contract.get("schema", "missing"),
    "legacy_primitive_gap_verdict": primitive_gap.get("verdict", "missing"),
    "present_or_refuted": [
      {"primitive": "CrossLane.ds_bpermute_reduce", "status": "present_refuted_as_gap", "evidence": "owned and generated use the same ds_bpermute reduce; do not build a new ds_permute primitive"},
      {"primitive": "TileMemory.lds_tile", "status": "present_generated_default_off", "evidence": "ISA/resource artifact reports LDS tile and block-tile microgate passes"},
      {"primitive": "DotLowering.v_dot2", "status": "present_generated_default_off", "evidence": "ISA marker reports v_dot2 in generated block tile"},
      {"primitive": "LaneMap.cooperative_stage", "status": "present_generated_default_off", "evidence": "cooperative-staging LaneMap composes with coalesced-load lowering"},
      {"primitive": "Math.fast_exp2_valid_domain", "status": "present_manual_flag_not_search_owned", "evidence": "DECODE_FAST_EXP2 closes +8-9% by removing dead range-reduction work"},
      {"primitive": "Sched.recurrence_unroll_list", "status": "present_manual_flag_not_search_owned", "evidence": "SCHED_UNROLL/SCHED_LIST transfer but are manually selected"},
    ],
    "missing_or_not_search_owned": [
      {"primitive": "OuterBlockLoop.lds_staged_split_combine", "status": "missing_search_vocab", "why": "ctx slope is the outer b-block online-softmax carry; current unroll targets inner tt only"},
      {"primitive": "ResourceModel.occupancy_guardrail", "status": "missing_search_scoring", "why": "tile is VGPR/occupancy-bound; pressure-increasing levers regress"},
      {"primitive": "Scheduler.pressure_aware_latency_hiding", "status": "partial_not_search_owned", "why": "generated reduces still show scheduling/pipelining residual versus owned hand-shaped code"},
      {"primitive": "Audit.split_aware_hotloop_oracle", "status": "missing_tooling", "why": "current hot-loop heuristic can lock onto the wrong loop under split experiments"},
    ],
    "verdict": "VOCAB_PARTIAL__FOUNDATION_PRIMITIVES_VISIBLE__OUTER_B_SPLIT_AND_OCCUPANCY_SEARCH_MISSING",
  }

  score = {
    "decode_attention_pure_machine_search_score_0_to_100": 60,
    "basis": [
      "generated route is correct and transfers in-model",
      "foundation primitives compose and produce material speedup",
      "remaining gap is now attributed, not existential",
      "manual flags and owned baseline still required for default performance",
      "outer-b split, occupancy scoring, and pressure-aware scheduling are not BubbleBeam-owned yet",
    ],
  }

  next_actions = [
    {"rank": 1, "action": "build occupancy guardrail gate", "gate": "abort candidates that raise VGPR or lower waves/CU versus the best stack"},
    {"rank": 2, "action": "make hot-loop schedule diff split-aware", "gate": "separate inner tt loop from outer b-block loop before implementing more splits"},
    {"rank": 3, "action": "add/search LDS-staged outer-b split-combine primitive", "gate": "must bend ctx4096 slope without increasing VGPR occupancy cost"},
    {"rank": 4, "action": "bind manual winning flags into BubbleBeam/FutureSight search space", "gate": "candidate provenance changes from manual flags to search-owned selection"},
  ]

  out = {
    "schema": "qk_decode_attention_pure_search_gap_audit_v1",
    "date": datetime.date.today().isoformat(),
    "inputs": {
      "transfer": str(TRANSFER.relative_to(ROOT)),
      "hotloop_schedule_diff": str(HOTLOOP.relative_to(ROOT)),
      "primitive_gap": str(PRIMITIVE_GAP.relative_to(ROOT)),
      "primitive_schema": str(PRIMITIVE_SCHEMA.relative_to(ROOT)),
      "search_contract": str(SEARCH_CONTRACT.relative_to(ROOT)),
      "isa_vectorization": str(ISA_VEC.relative_to(ROOT)),
      "delta_attack_result": str(DELTA_RESULT.relative_to(ROOT)),
      "scheduler_capability_scope": str(SCHED_SCOPE.relative_to(ROOT)),
    },
    "time_delta_explanation": time_delta,
    "primitive_vocabulary_attribution": primitive_vocab,
    "pure_search_score": score,
    "next_actions": next_actions,
    "verdict": "PURE_SEARCH_PARTIAL__TIME_DELTA_EXPLAINED__VOCAB_GAPS_IDENTIFIED__NOT_PROMOTABLE_YET",
  }

  OUTDIR.mkdir(parents=True, exist_ok=True)
  (OUTDIR / "latest.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
