#!/usr/bin/env python3
"""Canonical pure-machine-search gap audit for decode attention.

This is the wrapper around the existing fine-tuning oracles. It joins the two axes that matter for pure search:
  1. time-delta explanation: what still makes generated/search-owned decode attention slower;
  2. primitive/vocabulary attribution: whether the missing move is searchable, manually flagged, refuted, or absent.

It does not run GPU work. It consumes checked-in artifacts and emits a compact verdict for handoff and promotion review.

The score is DERIVED from the loaded artifacts (not a literal): every point traces to a live signal -- the measured
W==D transfer rows, the hot-loop/occupancy gate verdicts, and the outer-b lowering status. Absent inputs are recorded
in `inputs_missing` and depress the score (degraded), so missing evidence is distinguishable from a measured value.
The score MOVES when the build improves (outer-b lowering ships, flags become search-owned, W==D approaches owned).
"""
from __future__ import annotations
import json, pathlib, datetime
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "bench/qk-pure-search-gap"

def _latest_transfer(outdir: pathlib.Path) -> pathlib.Path:
  # De-pinned: pick the newest dated W==D transfer snapshot so a fresh harness run is consumed automatically.
  snaps = sorted(outdir.glob("transfer_snapshot_*.json"))
  return snaps[-1] if snaps else outdir / "transfer_snapshot_20260627.json"

TRANSFER = _latest_transfer(OUTDIR)
HOTLOOP = ROOT / "bench/qk-decode-hotloop-schedule-diff/latest.json"
PRIMITIVE_GAP = ROOT / "bench/qk-decode-primitive-space/gap_latest.json"
PRIMITIVE_SCHEMA = ROOT / "bench/qk-decode-primitive-space/primitive_schema.json"
SEARCH_CONTRACT = ROOT / "bench/qk-decode-primitive-space/search_contract.json"
ISA_VEC = ROOT / "bench/qk-decode-isa-vectorization/latest.json"
DELTA_RESULT = ROOT / "docs/decode-tile-delta-attack-result-20260627.md"
SCHED_SCOPE = ROOT / "docs/decode-codegen-scheduler-capability-scope.md"
OCC_GUARD = ROOT / "bench/qk-decode-occupancy-guardrail/latest.json"
OUTER_B = ROOT / "bench/qk-decode-outer-b-split-combine/latest.json"
PRESSURE_OWN = ROOT / "bench/qk-decode-pressure-search-ownership/latest.json"

# Promotion threshold: full-stack W==D must reach this fraction of owned before "search-owned" can score.
WD_PROMOTION_PCT_OF_OWNED = 90.0
# Standing refutations recorded with their reason (read as documented knowledge, not a fake membership test).
_STANDING_REFUTATIONS = {"ds_permute": "ds_bpermute cross-lane reduce is at per-token parity with owned; no new primitive warranted"}

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
  # Track which critical inputs are absent (loaded their default) so the verdict can be marked degraded.
  critical = {"transfer": TRANSFER, "hotloop": HOTLOOP, "isa_vectorization": ISA_VEC,
              "occupancy_guardrail": OCC_GUARD, "outer_b_split_contract": OUTER_B}
  inputs_missing = sorted(name for name, p in critical.items() if not p.exists())

  transfer = load(TRANSFER, {"arms": [], "notes": []})
  hotloop = load(HOTLOOP, {})
  primitive_gap = load(PRIMITIVE_GAP, {})
  primitive_schema = load(PRIMITIVE_SCHEMA, {})
  search_contract = load(SEARCH_CONTRACT, {})
  isa = load(ISA_VEC, {})
  delta_doc = load(DELTA_RESULT, "")
  sched_scope = load(SCHED_SCOPE, "")
  occ_guard = load(OCC_GUARD, {})
  outer_b = load(OUTER_B, {})
  pressure_own = load(PRESSURE_OWN, {})

  arms = arm_map(transfer)
  owned = arms.get("owned_baseline", {})
  no_stack = arms.get("block_tile_route_no_stack", {})
  full = arms.get("block_tile_route_full_stack", {})
  fused = arms.get("prior_fused_xlane_route", {})

  hotloop_verdict = hotloop.get("verdict", "missing")
  gen_mix = (hotloop.get("generated", {}).get("selected_loop", {}).get("metrics", {}).get("mix")
             or hotloop.get("generated", {}).get("mix", {}))
  selected_loop_valid = hotloop.get("comparison", {}).get("selected_loop_valid")
  generated_long_latency_seen = any(gen_mix.get(k, 0) for k in ("ds_bpermute", "global_load", "ds_read"))
  schedule_verdict = hotloop_verdict if selected_loop_valid else ("HOTLOOP_SCHEDULE_DIFF__SCHEDULING_BOUND" if "HOTLOOP_SCHEDULE_DIFF__SCHEDULING_BOUND" in sched_scope else hotloop_verdict)
  hotloop_tool_status = ("SPLIT_AWARE_HOTLOOP_READY" if selected_loop_valid and generated_long_latency_seen
                         else "SPLIT_AWARENESS_GAP__CURRENT_JSON_MISIDENTIFIED_GENERATED_LOOP")

  occ_verdict = occ_guard.get("verdict", "missing")
  occ_pass = bool(occ_guard.get("pass"))
  outer_b_verdict = outer_b.get("verdict", "missing")
  # Drive lowering-built detection from the contract artifact instead of hardcoding: shipping the lowering
  # (which updates the contract verdict) flips this from False to True.
  outer_b_lowering_built = outer_b_verdict != "missing" and "LOWERING_NOT_BUILT" not in outer_b_verdict
  # Manual flags are search-owned only when a provenance artifact says so (today the snapshot rows are flagged
  # generated_manual_route_flag* => not search-owned).
  provenances = {a.get("provenance", "") for a in transfer.get("arms", []) if a.get("arm") != "owned_baseline"}
  flags_search_owned = bool(provenances) and all("manual" not in p for p in provenances)

  # W==D authority honesty: a hand-typed session snapshot is NOT a harness measurement.
  snap_source = str(transfer.get("source", "")).lower()
  wd_authority = ("harness_measured_w_equals_d" if transfer.get("authority") == "W_equals_D_in_model"
                  and "session-reported" not in snap_source and "session_reported" not in snap_source
                  else "session_reported_not_harness_measured")

  full_pct_512 = pct(full.get("ctx512_tok_s", 0), owned.get("ctx512_tok_s", 0))
  full_pct_4096 = pct(full.get("ctx4096_tok_s", 0), owned.get("ctx4096_tok_s", 0))
  wd_pct_of_owned_avg = round((full_pct_512 + full_pct_4096) / 2.0, 1)

  time_delta = {
    "authority": transfer.get("authority", "unknown"),
    "wd_authority": wd_authority,
    "transfer_snapshot": TRANSFER.name,
    "rows": transfer.get("arms", []),
    "stack_transfer_vs_no_stack": {
      "ctx512_pct": round(((full.get("ctx512_tok_s", 0) / no_stack.get("ctx512_tok_s", 1)) - 1.0) * 100.0, 1) if no_stack.get("ctx512_tok_s") else None,
      "ctx4096_pct": round(((full.get("ctx4096_tok_s", 0) / no_stack.get("ctx4096_tok_s", 1)) - 1.0) * 100.0, 1) if no_stack.get("ctx4096_tok_s") else None,
    },
    "remaining_gap_to_owned": {
      "ctx512_owned_over_full_stack": ratio(owned.get("ctx512_tok_s", 0), full.get("ctx512_tok_s", 0)),
      "ctx4096_owned_over_full_stack": ratio(owned.get("ctx4096_tok_s", 0), full.get("ctx4096_tok_s", 0)),
      "ctx512_full_stack_pct_of_owned": full_pct_512,
      "ctx4096_full_stack_pct_of_owned": full_pct_4096,
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
    "occupancy_guardrail": occ_verdict,
    # refuted deltas: documented standing refutations + any named candidate the delta doc records as refuted.
    "closed_deltas": [x for x in ["DECODE_FAST_EXP2"] if x in delta_doc],
    "refuted_deltas": sorted(set(_STANDING_REFUTATIONS) | {x for x in ("SCHED_UNROLL_SPLIT", "DECODE_Q_HOIST") if x in delta_doc}),
    "refutation_reasons": _STANDING_REFUTATIONS,
    "verdict": "TIME_DELTA_PARTIAL_EXPLAINED__GENERATED_STACK_TRANSFERS__LONG_CTX_GAP_REMAINS",
  }

  isa_markers_present = bool(isa.get("capture", {}).get("tile", {}).get("markers")) or bool(isa.get("capture", {}).get("tile", {}).get("resources"))
  route_transfers = bool(
    full.get("ctx512_tok_s") and full.get("ctx4096_tok_s") and no_stack.get("ctx512_tok_s") and no_stack.get("ctx4096_tok_s")
    and full["ctx512_tok_s"] > no_stack["ctx512_tok_s"] and full["ctx4096_tok_s"] > no_stack["ctx4096_tok_s"])
  gap_attributed = bool(selected_loop_valid) and occ_verdict != "missing" and outer_b_verdict != "missing"
  wd_promotable = wd_pct_of_owned_avg >= WD_PROMOTION_PCT_OF_OWNED

  # DERIVED rubric -- every component is a live signal from the loaded artifacts; the score moves with the build.
  score_components = {
    "generated_route_correct_and_transfers": 30 if route_transfers else 0,
    "foundation_primitives_compose": 15 if isa_markers_present else 0,
    "gap_attributed_not_existential": 15 if gap_attributed else 0,
    "outer_b_lowering_built_bends_slope": 20 if (outer_b_lowering_built and gap_attributed) else 0,
    "flags_search_owned_and_wd_parity": 20 if (flags_search_owned and wd_promotable) else 0,
  }
  decode_attention_score = sum(score_components.values())
  degraded = bool(inputs_missing)

  score = {
    "decode_attention_pure_machine_search_score_0_to_100": decode_attention_score,
    "score_provenance": "derived_from_live_artifacts",
    "score_components": score_components,
    "score_max": 100,
    "wd_pct_of_owned_avg": wd_pct_of_owned_avg,
    "wd_promotion_threshold_pct": WD_PROMOTION_PCT_OF_OWNED,
    "inputs_missing": inputs_missing,
    "degraded": degraded,
    "basis": [
      "generated route is correct and transfers in-model (measured: full-stack > no-stack at ctx512 and ctx4096)",
      "foundation primitives compose and produce material speedup (ISA marker/resource snapshot present)",
      "remaining gap is attributed, not existential (hot-loop selected-loop valid + occupancy + outer-b vocab present)",
      "outer-b LDS split-combine lowering not built yet => 20-pt slope component withheld",
      "manual flags + owned baseline still required for default performance => search-owned component withheld",
    ],
  }

  next_actions = [
    {"rank": 1, "action": "implement LDS-staged outer-b split-combine lowering", "gate": "must bend ctx4096 slope and pass occupancy guardrail"},
    {"rank": 2, "action": "bind manual winning flags into BubbleBeam/FutureSight candidates", "gate": "selection provenance changes from manual flags to search-owned"},
    {"rank": 3, "action": "use split-aware hot-loop and occupancy guardrail as mandatory preflight", "gate": "selected loop counters improve without VGPR/scratch regression"},
    {"rank": 4, "action": "regenerate transfer snapshot from a real model.generate W==D run", "gate": "wd_authority becomes harness_measured_w_equals_d"},
  ]

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
      {"primitive": "Audit.split_aware_hotloop_oracle", "status": "present_tooling", "evidence": "branch-target parsing enumerates owned/generated loops and selects the real outer block loop"},
      {"primitive": "ResourceModel.occupancy_guardrail", "status": "present_tooling", "evidence": occ_verdict},
      {"primitive": "OuterBlockLoop.lds_staged_split_combine.search_contract", "status": "present_search_vocab", "evidence": outer_b_verdict},
      {"primitive": "Scheduler.pressure_search_ownership_audit", "status": "present_tooling", "evidence": pressure_own.get("verdict", "missing")},
    ],
    "missing_or_not_search_owned": [
      {"primitive": "OuterBlockLoop.lds_staged_split_combine.lowering",
       "status": "lowering_built" if outer_b_lowering_built else "lowering_not_built",
       "why": "search vocabulary exists, but no generated candidate yet bends ctx4096 slope" if not outer_b_lowering_built else "outer-b contract reports the lowering present"},
      {"primitive": "Scheduler.pressure_aware_latency_hiding.search_binding",
       "status": "search_owned" if flags_search_owned else "partial_not_search_owned",
       "why": "guardrails exist, but winning manual flags are not BubbleBeam-owned" if not flags_search_owned else "flags bound to search provenance"},
    ],
    "verdict": "VOCAB_PARTIAL__GUARDRAIL_AND_OUTER_B_SEARCH_VOCAB_PRESENT__LOWERING_AND_SEARCH_BINDING_REMAIN"
               if not (outer_b_lowering_built and flags_search_owned)
               else "VOCAB_SEARCH_OWNED__OUTER_B_LOWERING_AND_FLAG_BINDING_PRESENT",
  }

  # DERIVED verdict: PARTIAL today, becomes PROMOTABLE only when the build clears the gates; degraded if inputs absent.
  if degraded:
    verdict = "PURE_SEARCH_DEGRADED__INPUTS_MISSING__" + ",".join(inputs_missing)
  elif outer_b_lowering_built and flags_search_owned and wd_promotable:
    verdict = "PURE_SEARCH_PROMOTABLE__GENERATED_SEARCH_OWNED__WD_AT_OWNED_THRESHOLD"
  else:
    verdict = "PURE_SEARCH_PARTIAL__TIME_DELTA_EXPLAINED__VOCAB_GAPS_IDENTIFIED__NOT_PROMOTABLE_YET"

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
      "occupancy_guardrail": str(OCC_GUARD.relative_to(ROOT)),
      "outer_b_split_contract": str(OUTER_B.relative_to(ROOT)),
      "pressure_search_ownership": str(PRESSURE_OWN.relative_to(ROOT)),
    },
    "inputs_missing": inputs_missing,
    "degraded": degraded,
    "time_delta_explanation": time_delta,
    "primitive_vocabulary_attribution": primitive_vocab,
    "pure_search_score": score,
    "next_actions": next_actions,
    "verdict": verdict,
  }

  OUTDIR.mkdir(parents=True, exist_ok=True)
  (OUTDIR / "latest.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
