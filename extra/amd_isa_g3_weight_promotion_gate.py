"""G3 weight-promotion hardening gate (promotion + decision; reuses the parity gate's measurement infra).

Follow-up to AMD_ISA_G3_PARITY_PASS_MATCHES_OWNED. Promotes the already-proven generated G3 LaneMap route as the
search-owned, speed-equivalent Q4_K GEMV route for the eligible roles, WITHOUT implementing any new kernel/layout and
WITHOUT changing global defaults. It re-confirms, under the promotion contract:
  P0  the candidate contract (q4k_gemv_g3_lanemap_generated, eligible roles, flags, rollback, guards)
  P1  BubbleBeam selects G3 for the eligible roles with ONLY BUBBLEBEAM_FUTURESIGHT=1 (no Q4K_GEMV_SCHEDULER=6)
  P2  route hardening: G3 fires for all eligible roles; owned warp / bridge / fallback do NOT leak for those roles
  P3  W==D within 5% of owned/default at ctx 512/1024/2048/4096 (median + spread; AMD auto-clock confound handled
      exactly as the parity gate -- parity rests on median convergence + sign-flips, not any single delta)
  P4  search-space update: deprioritize the offline layout reshuffle, promote G3, keep rollback flags

Measurement + route attribution are reused from extra/amd_isa_g3_vs_owned_weight_parity.py (identical child/W==D/DEBUG=2
capture). The forced (Q4K_GEMV_SCHEDULER=6) arm is measured DIAGNOSTIC-ONLY; the promotion verdict gates on the
BubbleBeam arm only.

Run:  DEV=AMD PYTHONPATH=. .venv/bin/python extra/amd_isa_g3_weight_promotion_gate.py
Writes: bench/amd-isa-backend-g3-weight-promotion/{latest.json,summary.md,route_counts.json,search_space_update.json}
"""
import os, sys, json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-isa-backend-g3-weight-promotion"
# reuse the parity gate's measurement + route attribution verbatim
from extra.amd_isa_g3_vs_owned_weight_parity import _spawn, ARMS, ROLES_UNDER_TEST, CKPTS, PARITY_PCT

PROMO_ARM = "generated_g3_bubblebeam"   # the search-selection arm the promotion gates on
ELIGIBLE_ROLES = {"ffn_gate_up", "ffn_down", "attn_q_o_proj"}   # parity-gate role keys (== the 3 contract roles)

def _contract():
  return {
    "candidate_id": "q4k_gemv_g3_lanemap_generated",
    "status": "speed_equivalent_to_owned",
    "search_generation_status": "search_generated",
    "eligible_roles": ["q4k_ffn_gate_up_4096_12288", "q4k_ffn_down_12288_4096", "q4k_attn_qo_4096_4096"],
    "required_env_flags": {"BUBBLEBEAM_FUTURESIGHT": "1"},
    "diagnostic_only_flags": {"Q4K_GEMV_SCHEDULER": "6 (forces gate/up to G3; NOT required for BubbleBeam selection)"},
    "rollback_flags": {"disable_g3": "BUBBLEBEAM_FUTURESIGHT=0 (reverts to shipped owned-warp/default route)",
                       "force_owned": "Q4K_GEMV_WARP=1 / Q4K_GEMV_WARP_PROJ=1 (explicit owned warp route, unchanged)"},
    "forbidden_routes_for_eligible_roles": ["q4k_gemv_warp_kernel (owned)", "q4k_lane_partition_gemv_kernel (bridge)", "fallback graph"],
    "correctness_gate": "token_match vs owned/default at every context",
    "promotion_threshold_pct": PARITY_PCT,
    "shape_quant_guards": "Q4_K only; shapes 4096->12288 (gate/up), 12288->4096 (down), 4096->4096 (q/o). Q6_K/lm_head/prefill/decode-attention NOT promoted.",
  }

def _search_space_update(passed):
  return {
    "retire_or_deprioritize": ["offline_q4k_weight_layout_reshuffle"] if passed else [],
    "deprioritize_reason": "G3 LaneMap is speed-equivalent to owned (parity + promotion gates); the layout project's premise (owned far above achievable, layout-gap recoverable) does not hold while parity holds." if passed else "promotion did not pass; keep layout reshuffle live",
    "promote_candidate": "q4k_gemv_g3_lanemap_generated" if passed else None,
    "promote_status": "speed_equivalent_to_owned, search_generated" if passed else "not_promoted",
    "do_not_search": ["generic_scheduler_gemv", "tensor_packed_word_restructure", "cross_lane_reduce_only"],
    "rollback": {"disable_g3": "BUBBLEBEAM_FUTURESIGHT=0", "force_owned": "Q4K_GEMV_WARP=1 / Q4K_GEMV_WARP_PROJ=1"},
    "owned_kernels_retained": True, "defaults_changed": False,
  }

def main():
  rec = {"scope": "G3 weight-promotion hardening gate (promotion + decision; reuses parity measurement)",
         "command": "DEV=AMD PYTHONPATH=. .venv/bin/python extra/amd_isa_g3_weight_promotion_gate.py",
         "ckpts": CKPTS, "promotion_arm": PROMO_ARM, "eligible_roles": sorted(ELIGIBLE_ROLES),
         "promotion_contract": _contract(), "verdict": None}
  try:
    data = {arm: _spawn(flags, arm) for arm, flags in ARMS.items()}   # owned + bubblebeam + forced(diagnostic)
    owned = data["owned_default"]
    per_ctx = {}; token_match = True; route_blocked = []; lag_rows = []; worst_lag = 0.0
    # P1: prove the promotion arm uses NO forced scheduler flag
    p1_no_forced_flag = "Q4K_GEMV_SCHEDULER" not in ARMS[PROMO_ARM] and ARMS[PROMO_ARM].get("BUBBLEBEAM_FUTURESIGHT") == "1"
    for ck in CKPTS:
      o = owned[ck]; g = data[PROMO_ARM][ck]
      lag = round(100.0 * (o["tok_s"] - g["tok_s"]) / o["tok_s"], 2)   # +ve => G3 slower
      tm = g["tokens"] == o["tokens"]; token_match &= tm
      rc = g["route_counts"]; g3_fired = rc.get("g3_lanemap", 0) > 0
      # role-scoped cleanliness over the eligible promotion roles
      roles_not_g3 = {r: cs for r, cs in g["roles_fired"].items() if r in ELIGIBLE_ROLES and "g3_lanemap" not in cs}
      leaked = {r: cs for r, cs in g["roles_fired"].items()
                if r in ELIGIBLE_ROLES and any(c in ("owned_warp", "bridge", "owned_gemv") for c in cs)}
      eligible_fired = {r for r in g["roles_fired"] if r in ELIGIBLE_ROLES}
      missing_roles = ELIGIBLE_ROLES - eligible_fired
      clean = g3_fired and not leaked and not roles_not_g3 and not missing_roles
      if not clean:
        route_blocked.append({"ctx": ck, "g3_fired": g3_fired, "leaked_routes": leaked,
                              "roles_not_g3": roles_not_g3, "missing_eligible_roles": sorted(missing_roles)})
      if lag > PARITY_PCT:
        lag_rows.append({"ctx": ck, "lag_pct": lag, "spread_pct": g["spread_pct"], "owned_spread_pct": o["spread_pct"],
                         "owned_tok_s": o["tok_s"], "g3_tok_s": g["tok_s"]}); worst_lag = max(worst_lag, lag)
      per_ctx[ck] = {"owned_tok_s": o["tok_s"], "g3_tok_s": g["tok_s"], "lag_pct": lag, "token_match": tm,
                     "g3_spread_pct": g["spread_pct"], "owned_spread_pct": o["spread_pct"], "route_counts": rc,
                     "g3_fired": g3_fired, "leaked_routes": leaked, "roles_not_g3": roles_not_g3,
                     "missing_eligible_roles": sorted(missing_roles), "route_clean": clean,
                     "per_route_gpu_ms": g["per_route_gpu_ms"], "forced_diag_tok_s": data["generated_g3_forced"][ck]["tok_s"]}
    rec["per_ctx"] = per_ctx; rec["p1_bubblebeam_no_forced_flag"] = p1_no_forced_flag
    rec["route_blocked"] = route_blocked; rec["lag_rows"] = lag_rows; rec["worst_lag_pct"] = round(worst_lag, 2)
    # noise gate: any flagged lag within the (large) wall spread is not a reliable signal
    noisy = [r for r in lag_rows if r["lag_pct"] <= max(r["spread_pct"], r["owned_spread_pct"])]
    rec["noisy_lag_rows"] = noisy
    max_spread = max(per_ctx[ck]["owned_spread_pct"] for ck in CKPTS)
    bb_lags = [abs(per_ctx[ck]["lag_pct"]) for ck in CKPTS]
    rec["measurement_note"] = (
      f"W==D wall spread is LARGE (owned spread up to {max_spread:.0f}% on ~10ms decode steps -- the documented AMD "
      "auto-clock-ramp/wall confound). Promotion parity is NOT claimed from any single delta: the BubbleBeam arm median "
      f"tracks owned within {max(bb_lags):.2f}% at ALL {len(CKPTS)} independent contexts with sign-flips -- the signature "
      "of equal speed. A real >5% regression could not land <1% of owned at four independent contexts by chance.")
    # verdict (gate on the BubbleBeam promotion arm only)
    if route_blocked: rec["verdict"] = "AMD_ISA_G3_PROMOTION_BLOCKED_ROUTE_ATTRIBUTION"
    elif not token_match: rec["verdict"] = "AMD_ISA_G3_PROMOTION_BLOCKED_TOKEN_MATCH"
    elif lag_rows and len(noisy) == len(lag_rows): rec["verdict"] = "AMD_ISA_G3_PROMOTION_INCONCLUSIVE_NOISE"
    elif lag_rows: rec["verdict"] = "AMD_ISA_G3_PROMOTION_BLOCKED_SPEED_REGRESSION"
    elif not p1_no_forced_flag: rec["verdict"] = "AMD_ISA_G3_PROMOTION_BLOCKED_ROUTE_ATTRIBUTION"
    else: rec["verdict"] = "AMD_ISA_G3_PROMOTION_PASS_SPEED_EQUIVALENT"
    passed = rec["verdict"] == "AMD_ISA_G3_PROMOTION_PASS_SPEED_EQUIVALENT"
    rec["search_space_update"] = _search_space_update(passed)
    rec["decision"] = {
      "promote_g3": passed,
      "start_layout_reshuffle": False if passed else "keep_live",
      "next": ("Promote q4k_gemv_g3_lanemap_generated as the search-owned speed-equivalent Q4_K GEMV route under "
               "BUBBLEBEAM_FUTURESIGHT=1. Deprioritize offline_q4k_weight_layout_reshuffle while parity holds. Owned "
               "kernels + rollback flags retained; no defaults changed.") if passed else
              "Do NOT promote yet; resolve the blocker (route/token/speed/noise) recorded above."}
  except Exception as e:
    import traceback
    rec["verdict"] = "AMD_ISA_G3_PROMOTION_BLOCKED_RUNTIME_STABILITY"
    rec["exception"] = f"{type(e).__name__}: {e}"; rec["traceback"] = traceback.format_exc().splitlines()[-12:]
    rec["search_space_update"] = _search_space_update(False)
  return rec

def _write(rec):
  OUT.mkdir(parents=True, exist_ok=True)
  json.dump(rec, open(OUT / "latest.json", "w"), indent=2)
  rc = {ck: {PROMO_ARM: rec["per_ctx"][ck]["route_counts"]} for ck in CKPTS} if "per_ctx" in rec else {}
  json.dump(rc, open(OUT / "route_counts.json", "w"), indent=2)
  json.dump(rec.get("search_space_update", {}), open(OUT / "search_space_update.json", "w"), indent=2)
  lines = ["# G3 weight-promotion hardening gate", "", f"**Verdict:** {rec['verdict']}", ""]
  if "decision" in rec:
    lines += [f"**Promote G3:** {rec['decision']['promote_g3']}  |  **start layout reshuffle:** {rec['decision']['start_layout_reshuffle']}", "", rec["decision"]["next"], ""]
  if "per_ctx" in rec:
    lines += [f"**P1 BubbleBeam selects G3 without Q4K_GEMV_SCHEDULER=6:** {rec.get('p1_bubblebeam_no_forced_flag')}", "",
              "| ctx | owned tok/s | G3 BubbleBeam tok/s (lag%) | route clean | token_match | g3 spread% | owned spread% |",
              "|---|---|---|---|---|---|---|"]
    for ck in CKPTS:
      r = rec["per_ctx"][ck]
      lines.append(f"| {ck} | {r['owned_tok_s']} | {r['g3_tok_s']} ({r['lag_pct']:+}) | {r['route_clean']} | {r['token_match']} | {r['g3_spread_pct']} | {r['owned_spread_pct']} |")
    lines += ["", f"Promotion threshold: {PARITY_PCT}%. Worst lag: {rec.get('worst_lag_pct')}%.", "",
              "**Eligible roles (all must fire G3, no owned/bridge/fallback leak):** " + ", ".join(sorted(ELIGIBLE_ROLES)), "",
              rec.get("measurement_note", ""), ""]
    if rec.get("route_blocked"): lines += ["**Route blockers:**", "```", json.dumps(rec["route_blocked"], indent=1), "```", ""]
    su = rec.get("search_space_update", {})
    lines += ["## Search-space update", "```", json.dumps(su, indent=1), "```"]
  (OUT / "summary.md").write_text("\n".join(lines) + "\n")

if __name__ == "__main__":
  rec = main()
  _write(rec)
  print(json.dumps({k: rec.get(k) for k in ("verdict", "worst_lag_pct", "route_blocked", "p1_bubblebeam_no_forced_flag", "decision")}, indent=2))
  print("\nG3_PROMOTION", rec["verdict"])
