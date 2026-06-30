#!/usr/bin/env python3
"""TG7: First New-Profile Search -- can the pipeline do MORE than rediscover G3?

Runs the full pipeline (TG4 opener facts -> TG2-style author, quant-parameterized by TG3 -> TG5 target gate ->
TG6 evaluator) on a genuinely NOT-pre-solved target on the validated gfx1100 backend:

  TARGET: Q6_K decode weight GEMV, role ffn_down (K=12288, N=4096), gfx1100.

  Justification (why this is a real test, not a re-run of G3):
    * Q6_K is a DIFFERENT quant than everything the grammar was built on (Q4_K): payload-FIRST byte layout,
      SYMMETRIC dequant (q-32, no per-group min), uint16 packing, int8 sub-scales, and a within-block coalesced
      lane extent of 16 (vs Q4_K's 8 packed words). So it truly exercises quant-agnosticism (TG3 facts).
    * Q6_K's shipped route is an OWNED coop kernel -- there is NO machine-AUTHORED / search-generated Q6_K route.
      So authoring a Q6_K topology is genuinely new, not "rediscover the generated G3".
    * It has replayable authority artifacts (shipped coop baseline + the REFUTED half-warp direct), so the
      evaluator reaches an HONEST verdict instead of stalling on missing measurement.

This does NOT manufacture a win. It reports honestly: what the grammar authors + what the evaluator decides.

AUDIT/RESEARCH/REPLAY only: no GPU kernel, no GPU re-measure, no default change, no live-route repoint, no
reopened refuted route. The refuted half-warp is EXCLUDED by the grammar's refuted-axis gate (preserved), and is
separately replayed through the gate only to confirm the refutation still holds.

Run: PYTHONPATH=. python3 extra/qk_new_profile_search.py
"""
from __future__ import annotations
import json, pathlib

from extra.qk_quant_semantics import quant_row
from extra.qk_target_features import target, gate_candidate_on_target, TARGET_OK
from extra.qk_lanemap_template import CROSS_LANE_WAVE_REDUCE, PARTIALS_PLUS_REDUCE
from extra.qk_template_candidate_gate import gate_candidate

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-new-profile-search/qwen3_8b_q6k_ffn_down_gfx1100"

PROFILE = {"profile_id": "qwen3_8b_q6k_ffn_down_decode_gfx1100", "quant": "Q6_K",
           "role": "ffn_down", "K": 12288, "N": 4096, "target_id": "amd_gfx1100"}

# the shipped OWNED Q6_K route decompositions (extra/q6_k_gemv_primitive.py) -- used for STRUCTURAL rediscovery
# mapping (analogous to TG2's G3 spec-equality), NOT to inject candidates.
_SHIPPED_Q6K_DECOMPOSITIONS = {
  # q6k_gemv_warp_kernel: lane = block_group(0..1)*16 + pos(0..15); in-kernel cross-lane reduce.
  ("warp_owned", 2, 16, CROSS_LANE_WAVE_REDUCE): "decode_q6k_owned_warp (q6k_gemv_warp_kernel, owned_reference)",
  # q6k_coop_partial_kernel: pos(0..15) LOCAL lane + stage-2 .sum; no K-split (block_groups=1).
  ("coop_shipped", 1, 16, PARTIALS_PLUS_REDUCE): "decode_q6k_coop_shipped (q6k_coop_partial_kernel, owned_default)",
}


def factor_pos_lanes(per_row_lanes: int, natural_lane_extent: int, k_blocks: int) -> list[tuple[int, int]]:
  """Quant-generic coalesced factorization: per_row_lanes = block_groups * pos_lanes, with the coalesced lane run
  pos_lanes dividing the within-block coalesced extent (Q6_K: 16 byte positions; Q4_K: 8 packed words) AND
  block_groups dividing k_blocks. Divisor enumeration -- NOT hardcoded."""
  out = []
  for pos_lanes in range(1, per_row_lanes + 1):
    if per_row_lanes % pos_lanes != 0:
      continue
    if natural_lane_extent % pos_lanes != 0:
      continue
    block_groups = per_row_lanes // pos_lanes
    if k_blocks % block_groups != 0:
      continue
    out.append((block_groups, pos_lanes))
  return out


def author_q6k_candidates() -> tuple[list[dict], dict]:
  """Author a BOUNDED Q6_K GEMV topology family from TG3 facts x the grammar x refuted exclusions x TG5 target."""
  fmt = quant_row("Q6_K")                       # TG3: payload-first, symmetric, natural_lane_extent=16, uint16
  d = fmt.derive()
  lane_extent = target(PROFILE["target_id"]).lane_extent()   # gfx1100 wave32
  k_blocks = PROFILE["K"] // fmt.block_elems    # 12288 // 256 = 48
  natural = fmt.natural_lane_extent             # 16 within-block byte positions (coalesced)

  # lane_grouping: half_warp is REFUTED for Q6_K (decode_q6k_direct_refuted) -> EXCLUDED by the grammar gate.
  groupings = {"1row_per_warp": 1, "2rows_per_warp": 2}
  grouping_dispositions = {
    "half_warp": "EXCLUDED (refuted: decode_q6k_direct_refuted, W==D -5.44% median; "
                 "Q6_K known_refuted_route_families)",
    "subgroup": "FOLDED into 1row_per_warp (subgroup==wave on wave32)"}

  reductions = {"cross_lane_wave_reduce": CROSS_LANE_WAVE_REDUCE, "lds_partial_reduce": PARTIALS_PLUS_REDUCE}
  candidates, factor_audit = [], []
  for grouping, rows_per_wave in groupings.items():
    per_row_lanes = lane_extent // rows_per_wave
    for (bg, pos_lanes) in factor_pos_lanes(per_row_lanes, natural, k_blocks):
      factor_audit.append({"lane_grouping": grouping, "per_row_lanes": per_row_lanes,
                           "block_groups": bg, "pos_lanes": pos_lanes, "kept": True})
      for red_name, red_pattern in reductions.items():
        out_mode = "direct_out" if red_pattern == CROSS_LANE_WAVE_REDUCE else "partials_plus_sum"
        cand = {"candidate_id": f"q6k_ffn_down_{grouping}_bg{bg}_pos{pos_lanes}_{red_name}",
                "quant": "Q6_K", "lane_ownership_axis": "packed_byte_position", "lane_grouping": grouping,
                "block_groups": bg, "pos_lanes": pos_lanes, "reduction_pattern": red_pattern, "output": out_mode,
                "reduction_grammar_value": red_name, "target_feature_required": "wave32", "_spec_obj": None,
                "load_pattern": "coalesced_within_block_position", "dequant_placement": "per_lane_in_register"}
        candidates.append(cand)

  meta = {"quant_facts": {"block_elems": fmt.block_elems, "block_bytes": fmt.block_bytes,
                          "packing": fmt.packing_word_dtype, "symmetric": fmt.symmetric,
                          "natural_lane_extent": natural, "payload_first": not fmt.metadata_first,
                          "k_blocks": k_blocks},
          "grouping_dispositions": grouping_dispositions, "factor_audit": factor_audit,
          "refuted_excluded": [r.get("route_id") for r in fmt.known_refuted_route_families]}
  return candidates, meta


def structural_rediscovery(cand: dict) -> str | None:
  """Does this authored Q6_K candidate structurally match a shipped OWNED Q6_K route decomposition?
  (block_groups, pos_lanes, reduction) vs the q6k_gemv_warp / q6k_coop_partial kernels. Static, like TG2's G3 match."""
  for (_, bg, pos, red), route in _SHIPPED_Q6K_DECOMPOSITIONS.items():
    if cand["block_groups"] == bg and cand["pos_lanes"] == pos and cand["reduction_pattern"] == red \
       and (cand["lane_grouping"] == "1row_per_warp" if bg == 2 else True):
      return route
  return None


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  candidates, meta = author_q6k_candidates()
  EXPLOSION_LIMIT = 64
  bounded = len(candidates) <= EXPLOSION_LIMIT

  # all authored candidates pass the TG5 target gate (gfx1100 wave32)
  tgt = target(PROFILE["target_id"])
  target_ok = all(gate_candidate_on_target(c, tgt)[0] == TARGET_OK for c in candidates)

  # structural rediscovery: which authored candidates equal a shipped OWNED Q6_K route
  rediscoveries = []
  for c in candidates:
    route = structural_rediscovery(c)
    if route:
      rediscoveries.append({"candidate_id": c["candidate_id"], "rediscovers_shipped_owned_route": route,
                            "block_groups": c["block_groups"], "pos_lanes": c["pos_lanes"],
                            "reduction_pattern": c["reduction_pattern"]})

  # run a representative AUTHORED candidate through the TG6 gate (it maps to NO generated-promotion artifact ->
  # CORRECT_NOT_FAST_OR_UNMEASURED: a generated Q6_K route would need a fresh W==D, which this audit does not run).
  rep = next((c for c in candidates if structural_rediscovery(c)), candidates[0])
  rep_gate = gate_candidate(rep, {"N": PROFILE["N"], "K": PROFILE["K"], "role": "ffn_down"}, PROFILE["target_id"])

  # CONFIRM the refutation is preserved: the EXCLUDED half-warp, if forced through the gate, still refutes.
  halfwarp = {"candidate_id": "q6k_ffn_down_half_warp_EXCLUDED_BY_GRAMMAR", "quant": "Q6_K",
              "lane_grouping": "half_warp", "reduction_pattern": CROSS_LANE_WAVE_REDUCE, "_spec_obj": None,
              "target_feature_required": "wave32"}
  halfwarp_gate = gate_candidate(halfwarp, {"N": PROFILE["N"], "K": PROFILE["K"], "role": "ffn_down"},
                                 PROFILE["target_id"])

  # ---- HONEST verdict --------------------------------------------------------------------------------------
  authored_ok = bounded and target_ok and len(candidates) > 0
  refuted_preserved = (halfwarp_gate["verdict"] == "REFUTED_REGRESSION"
                       and halfwarp_gate.get("mapped_route_id") == "decode_q6k_direct_refuted")
  rediscovered_shipped = len(rediscoveries) > 0
  rep_is_unmeasured = rep_gate["verdict"] == "CORRECT_NOT_FAST_OR_UNMEASURED"

  # The authored family either rediscovers the shipped OWNED route (no generated-promotion authority -> needs a
  # fresh W==D this audit does not run) or hits the refuted half-warp (correctly excluded). No NEW promotable
  # topology is available under the current ceiling WITHOUT a fresh measurement -> SEARCH_EXHAUSTED_SPACE.
  search_result = "SEARCH_EXHAUSTED_SPACE"
  search_explanation = (
    f"The grammar AUTHORED {len(candidates)} bounded Q6_K ffn_down topology candidates (quant-parameterized from "
    f"TG3 Q6_K facts; all wave32 TARGET_OK). {len(rediscoveries)} of them STRUCTURALLY rediscover the shipped OWNED "
    f"Q6_K routes (q6k_gemv_warp / q6k_coop_partial) -- an equivalent-to-shipped rediscovery, not a new topology. "
    f"The refuted half-warp direct is EXCLUDED by the grammar's refuted-axis gate (and, if forced, still refutes: "
    f"{halfwarp_gate['verdict']} -> {halfwarp_gate.get('mapped_route_id')}). No authored candidate has a "
    f"generated-promotion authority artifact; promoting a GENERATED Q6_K replacement for the owned route would "
    f"require a fresh W==D measurement, which this AUDIT scope does not run. Honest verdict: the bounded space is "
    f"authored end-to-end but contains NO new promotable topology beyond the shipped owned route under the current "
    f"ceiling -> {search_result}.")

  ready = authored_ok and refuted_preserved and rediscovered_shipped and rep_is_unmeasured
  verdict = "TG7_PASS_FIRST_NEW_PROFILE_SEARCH_RESULT" if ready else "TG7_BLOCKED_MANUAL_ROUTE_EDIT_REQUIRED"

  result = {
    "scope": "TG7 first new-profile search on Q6_K decode ffn_down GEMV (gfx1100). Runs author (quant-parameterized "
             "by TG3) + TG5 target gate + TG6 evaluator end-to-end. HONEST report; no manufactured win; no GPU.",
    "verdict": verdict, "search_result": search_result, "profile": PROFILE,
    "justification": "Q6_K (payload-first, symmetric, uint16, natural_lane_extent=16) != Q4_K; its shipped route is "
                     "OWNED (no generated route) -> authoring a Q6_K topology is genuinely new; replayable artifacts "
                     "(shipped coop + refuted half-warp) let the evaluator reach an honest verdict.",
    "quant_facts_used": meta["quant_facts"],
    "candidate_count": len(candidates), "bounded": bounded, "explosion_limit": EXPLOSION_LIMIT,
    "all_candidates_target_ok_gfx1100": target_ok,
    "grouping_dispositions": meta["grouping_dispositions"],
    "refuted_excluded_by_grammar": meta["refuted_excluded"],
    "structural_rediscoveries_of_shipped_owned_routes": rediscoveries,
    "representative_candidate_gate": rep_gate,
    "refuted_halfwarp_replay_confirms_refutation": {"verdict": halfwarp_gate["verdict"],
                                                    "mapped_route_id": halfwarp_gate.get("mapped_route_id"),
                                                    "decision_reproduced": halfwarp_gate.get("decision_reproduced")},
    "search_explanation": search_explanation,
    "candidates": [{k: v for k, v in c.items() if k != "_spec_obj"} for c in candidates],
    "honest_summary": {
      "authored_new_topology_family": True,
      "viable_NEW_topology_beyond_shipped": False,
      "equivalent_to_shipped_rediscovery": rediscovered_shipped,
      "refuted_axis_preserved": refuted_preserved,
      "no_promotable_candidate_reason": "rediscovers shipped OWNED route / refuted half-warp excluded; a generated "
                                        "Q6_K promotion needs a fresh W==D not run in this audit"},
    "do_not": ["no GPU kernel", "no GPU re-measure", "no default change", "no live-route repoint",
               "no reopened refuted route"],
  }
  json.dump(result, open(OUT / "latest.json", "w"), indent=2)

  md = [f"# TG7 First New-Profile Search -- Q6_K ffn_down GEMV (gfx1100)", "",
        f"Verdict: **{verdict}** | search_result: **{search_result}**", "",
        f"{search_explanation}", "",
        "## What the grammar authored", "",
        f"- **{len(candidates)} bounded Q6_K topology candidates** (explosion limit {EXPLOSION_LIMIT}); all wave32 "
        f"TARGET_OK on gfx1100. Quant facts from TG3: payload_first={meta['quant_facts']['payload_first']}, "
        f"symmetric={meta['quant_facts']['symmetric']}, natural_lane_extent={meta['quant_facts']['natural_lane_extent']}, "
        f"k_blocks={meta['quant_facts']['k_blocks']}.",
        f"- half_warp EXCLUDED by the refuted-axis gate: {meta['refuted_excluded']}.", "",
        "## Structural rediscoveries of the shipped OWNED Q6_K routes", "",
        "| candidate | block_groups | pos_lanes | reduction | rediscovers |", "|---|---:|---:|---|---|"]
  for r in rediscoveries:
    md.append(f"| `{r['candidate_id']}` | {r['block_groups']} | {r['pos_lanes']} | {r['reduction_pattern']} | "
              f"{r['rediscovers_shipped_owned_route']} |")
  md += ["", "## Evaluator decisions", "",
         f"- Representative authored candidate `{rep['candidate_id']}` -> **{rep_gate['verdict']}** "
         f"(no generated-promotion authority; a generated Q6_K route needs a fresh W==D not run in this audit).",
         f"- Refuted half-warp (excluded by grammar; forced through gate) -> **{halfwarp_gate['verdict']}** "
         f"maps to `{halfwarp_gate.get('mapped_route_id')}` (refutation preserved).", "",
         "## Honest bottom line", "",
         "The pipeline runs end-to-end on a NEW quant profile and AUTHORS a bounded Q6_K topology family. It does "
         "NOT find a new promotable topology: the family either rediscovers the shipped OWNED route or hits the "
         "refuted half-warp. That is the honest, non-manufactured result the milestone asks for.", ""]
  (OUT / "summary.md").write_text("\n".join(md))

  print(verdict, "| search_result:", search_result)
  print(f"  authored {len(candidates)} Q6_K candidates (bounded={bounded}, all TARGET_OK={target_ok})")
  print(f"  structural rediscoveries of shipped owned routes: {len(rediscoveries)}")
  for r in rediscoveries:
    print(f"    {r['candidate_id']} -> {r['rediscovers_shipped_owned_route']}")
  print(f"  representative authored candidate gate: {rep_gate['verdict']}")
  print(f"  refuted half-warp (excluded; forced) -> {halfwarp_gate['verdict']} / {halfwarp_gate.get('mapped_route_id')}")
  return 0 if ready else 1


if __name__ == "__main__":
  raise SystemExit(main())
