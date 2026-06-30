#!/usr/bin/env python3
"""TG6: Template Search Evaluator -- gate TG2-authored topology candidates with the REAL evaluator.

This closes the loop: a candidate AUTHORED by the TG2 grammar (a TopologySpec, not a hand route) is run through
the same strict gate ladder as a handoff route, ending in the PMS-R2 evaluator (extra/qk_candidate_evaluator.py).

Gate ladder (scope TG6):
  1. template schema validation        (TG1 LaneMapTemplate.validate())
  2. generated kernel builds            (TG1 emit -> UOp program; for the Q4_K candidate, byte-identical to the
                                         promoted G3 route, proving the authored topology lowers to a real kernel)
  3. route attribution proves the intended candidate fired   (candidate->route_id by SPEC IDENTITY, not a hardcode;
                                         then the evaluator's route_bound/no-hidden-fallback evidence)
  4. token/logit correctness            (PMS-R2 correctness gate from the authority artifact)
  5. W==D / whole-prefill authority      (PMS-R2 replay of the cited authority artifact)
  6. attribution/ceiling explains movement
  7. ledger update                       (PMS-R2 appendable ledger row)

Controls (TG6 acceptance):
  * G3-rediscovery candidate (TG2 author, TopologySpec == G3's) -> PASS + maps to the promoted
    decode_q4k_g3_generated (replays bench/amd-isa-backend-g3-weight-promotion/latest.json; no re-measure).
  * Known-bad candidate (Q6_K direct half-warp, as built) -> stays REFUTED (replays decode_q6k_direct_refuted).
  * Missing-target candidate (same topology, target=nvidia/metal) -> SEARCH_BLOCKED_BY_RUNTIME (no backend here),
    via the TG5 target-feature gate -- NOT a fake pass.

AUDIT/REPLAY only: no GPU re-measure (the authority artifacts are replayed), no default change, no live-route
repoint, no new GPU kernel, no reopened refuted route.

Run: PYTHONPATH=. python3 extra/qk_template_candidate_gate.py
"""
from __future__ import annotations
import json, pathlib

from extra.qk_lanemap_template import (LaneMapTemplate, TopologySpec, QuantSpec, TargetSpec, ShapeSpec,
                                       g3_template, _reference_sink, LaneMapTemplateError, CROSS_LANE_WAVE_REDUCE)
from extra.qk_candidate_evaluator import evaluate
from extra.qk_target_features import target, gate_candidate_on_target, TARGET_OK, TARGET_BACKEND_INCOMPLETE, TARGET_PRUNED

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-template-candidate-gate"

SEARCH_BLOCKED_BY_RUNTIME = "SEARCH_BLOCKED_BY_RUNTIME"
SEARCH_BLOCKED_BY_CODEGEN = "SEARCH_BLOCKED_BY_CODEGEN"


# ---- candidate -> measured route mapping (by SPEC IDENTITY / refuted-axis token, never a route_id hardcode) ----
def map_candidate_to_route(candidate: dict) -> tuple[str | None, str]:
  """Map an authored candidate to the route_id whose authority artifact attests it.

  G3-rediscovery: TopologySpec equality with g3_template(...).topology -> decode_q4k_g3_generated.
  Q6_K half-warp direct: the refuted grammar token (lane_grouping=half_warp, quant Q6_K) -> decode_q6k_direct_refuted
    (this is the refuted_axis_exclusions route_id, i.e. preserving a measured refutation, not inventing one).
  Anything else: no measured authority -> UNMAPPED."""
  spec = candidate.get("_spec_obj")
  quant = candidate.get("quant", "Q4_K")
  if quant == "Q4_K" and spec is not None and spec == g3_template("ffn_gate_up", 12288, 4096).topology:
    return "decode_q4k_g3_generated", "TopologySpec == G3 promoted topology (spec identity)"
  if quant == "Q6_K" and candidate.get("lane_grouping") == "half_warp":
    return "decode_q6k_direct_refuted", "refuted grammar token: Q6_K half_warp direct (refuted_axis_exclusions route_id)"
  return None, "no measured authority artifact maps to this topology"


# ---- the gate ladder --------------------------------------------------------------------------------------
def gate_candidate(candidate: dict, shape: dict, target_id: str = "amd_gfx1100") -> dict:
  ladder = {}

  # gate 0 (TG5): target lowering/profiling must exist to RUN/promote. Missing -> SEARCH_BLOCKED_BY_RUNTIME.
  tgt = target(target_id)
  tcand = dict(candidate); tcand.setdefault("target_feature_required", "wave32")
  tverdict, treason = gate_candidate_on_target(tcand, tgt)
  ladder["gate0_target"] = {"verdict": tverdict, "reason": treason}
  if tverdict == TARGET_BACKEND_INCOMPLETE:
    return {"candidate_id": candidate.get("candidate_id", "anon"), "target": target_id,
            "verdict": SEARCH_BLOCKED_BY_RUNTIME, "ladder": ladder,
            "explanation": f"authored topology is algorithmically plausible on {target_id} but {treason}"}
  if tverdict == TARGET_PRUNED:
    return {"candidate_id": candidate.get("candidate_id", "anon"), "target": target_id,
            "verdict": SEARCH_BLOCKED_BY_CODEGEN, "ladder": ladder,
            "explanation": f"target feature mismatch (no lowering): {treason}"}

  route_id, map_reason = map_candidate_to_route(candidate)
  ladder["map"] = {"route_id": route_id, "reason": map_reason}

  # gates 1-2: schema validation + the generated kernel builds. For the Q4_K G3 candidate we EMIT via the TG1 IR
  # and prove the UOp program is byte-identical to the promoted route (a real, route-clean kernel). For the Q6_K
  # half-warp candidate, the Q4_K-shaped LaneMapTemplate IR does not express it; its authority is the replay, and
  # schema validation is recorded as N/A_IR_SHAPE (it is still a legitimate refuted control via its artifact).
  spec = candidate.get("_spec_obj")
  if candidate.get("quant", "Q4_K") == "Q4_K" and spec is not None:
    try:
      t = LaneMapTemplate(topology=spec, quant=QuantSpec.from_library("Q4_K"), target=TargetSpec(),
                          shape=ShapeSpec(rows=shape["N"], k=shape["K"], role=shape.get("role", "")))
      t.validate()
      built = t.emit()
      ref = _reference_sink(shape["N"], shape["K"])
      builds_identical = (built.key == ref.key)
      ladder["gate1_schema"] = {"valid": True}
      ladder["gate2_builds"] = {"emitted": True, "uop_key_identical_to_promoted_route": bool(builds_identical),
                                "kernel_name": built.arg.name}
    except LaneMapTemplateError as e:
      ladder["gate1_schema"] = {"valid": False, "error": str(e)}
      return {"candidate_id": candidate.get("candidate_id", "anon"), "target": target_id,
              "verdict": SEARCH_BLOCKED_BY_CODEGEN, "ladder": ladder,
              "explanation": f"template schema invalid: {e}"}
  else:
    ladder["gate1_schema"] = {"valid": "N/A_IR_SHAPE",
                              "note": "Q4_K-shaped LaneMapTemplate IR does not express this topology; "
                                      "authority is the measured-artifact replay (refuted control)."}
    ladder["gate2_builds"] = {"emitted": False, "note": "no Q4_K IR emission; replay-only refuted control."}

  # gates 3-7: route attribution + correctness + W==D authority + ledger -- via the PMS-R2 evaluator (REPLAY).
  if route_id is None:
    ladder["gate3_7_evaluator"] = {"status": "UNMAPPED", "note": "no measured authority artifact"}
    return {"candidate_id": candidate.get("candidate_id", "anon"), "target": target_id,
            "verdict": "CORRECT_NOT_FAST_OR_UNMEASURED", "ladder": ladder,
            "explanation": "authored topology has no measured authority artifact; needs a fresh W==D before promotion"}
  ev = evaluate(route_id)
  ladder["gate3_route_attribution"] = {"route_bound_all_ctx": ev["route_attribution"]["route_bound_all_ctx"],
                                       "no_hidden_fallback": ev["route_attribution"]["no_hidden_fallback"]}
  ladder["gate4_correctness"] = ev["correctness"]
  ladder["gate5_authority"] = {"authority_type": ev["authority_type"], "speed_stats_pct": ev["speed_stats_pct"],
                               "tier": ev["tier_classification"], "disposition": ev["disposition"]}
  ladder["gate6_attribution_ceiling"] = {"decision_reproduced": ev["decision_reproduced"],
                                         "artifact_verdict": ev["artifact_verdict"]}
  ladder["gate7_ledger"] = ev["ledger_row"]
  return {"candidate_id": candidate.get("candidate_id", "anon"), "target": target_id,
          "mapped_route_id": route_id, "tier_classification": ev["tier_classification"],
          "disposition": ev["disposition"], "decision_reproduced": ev["decision_reproduced"],
          "verdict": ev["tier_classification"], "ladder": ladder,
          "explanation": f"authored topology maps to {route_id}; evaluator replays {ev['artifact_verdict']}"}


# ---- controls ---------------------------------------------------------------------------------------------
def _g3_rediscovery_candidate() -> dict:
  """The TG2-authored candidate whose TopologySpec == G3's (taken from the grammar enumeration, not hand-built)."""
  from extra.qk_topology_candidate_author import load_profile_facts, enumerate_candidates
  facts = load_profile_facts()
  cands, _ = enumerate_candidates(facts)
  g3_spec = g3_template("ffn_gate_up", 12288, 4096).topology
  match = next(c for c in cands if c["_spec_obj"] == g3_spec)
  match = dict(match); match["candidate_id"] = "tg2_authored_g3_rediscovery"; match["quant"] = "Q4_K"
  return match


def _q6k_halfwarp_refuted_candidate() -> dict:
  """The known-bad Q6_K direct half-warp candidate (refuted as built). The Q4_K-shaped IR does not express it; its
  authority is the measured refutation artifact it maps to."""
  return {"candidate_id": "q6k_halfwarp_direct_refuted_control", "quant": "Q6_K",
          "lane_ownership_axis": "packed_byte", "lane_grouping": "half_warp",
          "reduction_pattern": CROSS_LANE_WAVE_REDUCE, "_spec_obj": None,
          "target_feature_required": "wave32",
          "note": "2 rows x 16-lane half-warp partition; W==D -5.44% median (refuted_axis_exclusions)."}


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)

  g3 = _g3_rediscovery_candidate()
  g3_res = gate_candidate(g3, {"N": 12288, "K": 4096, "role": "ffn_gate_up"}, "amd_gfx1100")

  q6k = _q6k_halfwarp_refuted_candidate()
  q6k_res = gate_candidate(q6k, {"N": 4096, "K": 12288, "role": "ffn_down"}, "amd_gfx1100")

  # missing-target control: the SAME authored G3 topology on NVIDIA -> SEARCH_BLOCKED_BY_RUNTIME (not a fake pass).
  missing_res = gate_candidate(g3, {"N": 12288, "K": 4096, "role": "ffn_gate_up"}, "nvidia_sm89")

  g3_pass = (g3_res["verdict"] == "SPEED_EQUIVALENT_PASS" and g3_res.get("mapped_route_id") == "decode_q4k_g3_generated"
             and g3_res["decision_reproduced"]
             and g3_res["ladder"]["gate2_builds"].get("uop_key_identical_to_promoted_route") is True)
  q6k_refuted = (q6k_res["verdict"] == "REFUTED_REGRESSION"
                 and q6k_res.get("mapped_route_id") == "decode_q6k_direct_refuted" and q6k_res["decision_reproduced"])
  missing_blocked = (missing_res["verdict"] == SEARCH_BLOCKED_BY_RUNTIME)

  ready = g3_pass and q6k_refuted and missing_blocked
  verdict = "TG6_PASS_TEMPLATE_EVALUATOR_REPLAYS_CONTROLS" if ready else "TG6_BLOCKED_EVALUATOR_ROUTE_ATTRIBUTION"

  result = {
    "scope": "TG6 template search evaluator: run TG2-authored topology candidates through the gate ladder ending in "
             "the PMS-R2 evaluator. Controls: G3 rediscovery passes + maps to promoted route; Q6_K half-warp stays "
             "refuted; missing-target -> SEARCH_BLOCKED_BY_RUNTIME. AUDIT/REPLAY: no GPU re-measure.",
    "verdict": verdict,
    "gate": "extra/qk_template_candidate_gate.py", "evaluator": "extra/qk_candidate_evaluator.py",
    "control_g3_rediscovery": g3_res,
    "control_q6k_halfwarp_refuted": q6k_res,
    "control_missing_target_nvidia": missing_res,
    "summary": {"g3_rediscovery_passes_and_maps_to_promoted_route": g3_pass,
                "q6k_halfwarp_stays_refuted": q6k_refuted,
                "missing_target_blocked_by_runtime": missing_blocked},
    "do_not": ["no GPU re-measure (replay of cited authority artifacts)", "no default change",
               "no live-route repoint", "no new GPU kernel", "no reopened refuted route"],
  }
  json.dump(result, open(OUT / "latest.json", "w"), indent=2)

  md = [f"# TG6 Template Candidate Gate -- verdict: **{verdict}**", "",
        "TG2-authored candidates run through the gate ladder (schema -> builds -> route attribution -> correctness "
        "-> W==D authority -> ceiling -> ledger), ending in the PMS-R2 evaluator (replay).", "",
        "## Controls", "",
        "| control | candidate | maps to | gate verdict | reproduces |", "|---|---|---|---|:--:|",
        f"| G3 rediscovery (TG2-authored) | `{g3['candidate_id']}` | {g3_res.get('mapped_route_id')} | "
        f"{g3_res['verdict']} | {g3_res.get('decision_reproduced')} |",
        f"| Q6_K half-warp (known bad) | `{q6k['candidate_id']}` | {q6k_res.get('mapped_route_id')} | "
        f"{q6k_res['verdict']} | {q6k_res.get('decision_reproduced')} |",
        f"| missing-target (G3 topo on NVIDIA) | `{g3['candidate_id']}` | n/a | {missing_res['verdict']} | n/a |", "",
        "## Gate-ladder detail (G3 rediscovery)", "",
        f"- gate1 schema valid: {g3_res['ladder']['gate1_schema'].get('valid')}",
        f"- gate2 builds (UOp key == promoted route): "
        f"{g3_res['ladder']['gate2_builds'].get('uop_key_identical_to_promoted_route')} "
        f"(`{g3_res['ladder']['gate2_builds'].get('kernel_name')}`)",
        f"- gate3 route-bound all ctx: {g3_res['ladder']['gate3_route_attribution']['route_bound_all_ctx']}",
        f"- gate4 correctness: {g3_res['ladder']['gate4_correctness']}",
        f"- gate5 authority: {g3_res['ladder']['gate5_authority']['tier']} "
        f"({g3_res['ladder']['gate5_authority']['speed_stats_pct']})", ""]
  (OUT / "summary.md").write_text("\n".join(md))

  print(verdict)
  print(f"  G3 rediscovery: {g3_res['verdict']} -> {g3_res.get('mapped_route_id')} "
        f"(builds UOp-identical={g3_res['ladder']['gate2_builds'].get('uop_key_identical_to_promoted_route')}, "
        f"reproduced={g3_res.get('decision_reproduced')})")
  print(f"  Q6_K half-warp: {q6k_res['verdict']} -> {q6k_res.get('mapped_route_id')} "
        f"(reproduced={q6k_res.get('decision_reproduced')})")
  print(f"  missing-target NVIDIA: {missing_res['verdict']}")
  return 0 if ready else 1


if __name__ == "__main__":
  raise SystemExit(main())
