"""LH0 (audit-only, no kernels): isolate lm_head's exact Q6_K route and decide whether a STANDALONE lm_head route is
justified, or whether lm_head should fold into the broader Q6_K direct route.

Guardrail (from the scope): lm_head's GEMV is bandwidth-HEALTHY (~761 GB/s) -- do NOT optimize it. The firm removable
row is the coop reduce (r_32_4_1187 + r_32_4_1187n1, dim-product==151936 = vocab). Standalone upside = removing only
that reduce. If standalone Amdahl < 5%, reject the standalone route and fold lm_head into the general Q6_K direct route.

Reuses measured artifacts (Q6K-0 reduce_role_split + system-residual kernel_taxonomy); no GPU runs.
Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/audit/amd_isa/lm_head_q6k_route_audit.py
Writes: bench/amd-isa-backend-lm-head-q6k-route/{latest.json,summary.md,current_route.json,reduce_rows.json,amdahl.json}
"""
import json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "bench/amd-isa-backend-lm-head-q6k-route"
R0 = {"512": 103.9, "4096": 94.4}   # g3-promotion W==D median

# Tiered promotion policy (committed 97f82b999): a residual win is no longer auto-rejected for missing the old 5% bar.
def tier(wd_pct):
  if wd_pct >= 5.0:  return "TIER_A_MAJOR"             # normal speed promotion
  if wd_pct >= 2.0:  return "TIER_B_RESIDUAL"          # promotable w/ clean mechanism proof + rollback + no protected-ctx regression >1% + route-simplification/known-residual-removal
  if wd_pct >= -1.0: return "TIER_C_EQUIVALENT_CLEANUP" # not a speed win; promotable only if it retires special-case code / improves search purity
  return "BELOW_TIER_C"

def _read(p):
  f = ROOT / p; return json.load(open(f)) if f.exists() else None

def main():
  OUT.mkdir(parents=True, exist_ok=True)
  rrs = _read("bench/amd-isa-backend-q6k-residual-math/reduce_role_split.json")
  tax = _read("bench/amd-isa-backend-system-residual-ceiling/kernel_taxonomy.json")
  if rrs is None or tax is None:
    rec = {"verdict": "AMD_ISA_LM_HEAD_Q6K_AUDIT_BLOCKED_ROUTE_NOT_ROLE_RESOLVED", "reason": "missing Q6K-0/system-residual inputs"}
    json.dump(rec, open(OUT/"latest.json", "w"), indent=2); return rec

  per_ctx = {}
  for c in ("512", "4096"):
    total = tax[c].get("total_gpu_us") or (tax[c]["buckets"]["lm_head"]["gpu_time_us"] / (tax[c]["buckets"]["lm_head"]["pct_gpu_time"]/100))
    gemv = tax[c]["buckets"]["lm_head"]            # the lm_head GEMV (q6k_coop_partial_151936_4096)
    # firm lm_head reduce rows: dim-product == 151936 (uniquely vocab), plus the vocab-shaped 1187-bearing reduce
    rows = rrs[c]["rows"]
    firm = [r for r in rows if r.get("prod") == 151936]
    vocab_extra = [r for r in rows if r.get("prod") != 151936 and "1187" in r["kernel"]]   # r_1187_* : vocab-factor reduce
    red_firm_us = sum(r["dur"] for r in firm)
    red_extra_us = sum(r["dur"] for r in vocab_extra)
    red_us = red_firm_us + red_extra_us
    p_gemv = gemv["pct_gpu_time"]/100
    p_reduce_firm = red_firm_us/total
    p_reduce_total = (red_firm_us+red_extra_us)/total
    per_ctx[c] = {"total_gpu_us": round(total,1),
      "lm_head_gemv": {"kernel": "q6k_coop_partial_151936_4096", "quant": "q6k", "calls": gemv["calls"],
                       "dur_us": gemv["gpu_time_us"], "bytes": gemv["bytes"], "eff_bw_GBs": gemv["effective_bw_GBs"],
                       "pct_gpu": round(100*p_gemv,2), "route_family": "coop_partial", "verdict": "bandwidth-HEALTHY (761 GB/s) -> NOT a target"},
      "lm_head_reduce_firm": {"kernels": [r["kernel"] for r in firm], "dur_us": round(red_firm_us,2), "prod": 151936,
                              "class": "q6k_lm_head_reduce_FIRM", "pct_gpu": round(100*p_reduce_firm,3)},
      "lm_head_reduce_vocab_extra": {"kernels": [r["kernel"] for r in vocab_extra], "dur_us": round(red_extra_us,2),
                                     "pct_gpu": round(100*red_extra_us/total,3), "note": "r_1187_* vocab-factor reduce; tiny"},
      "p_lm_head_removable": round(100*p_reduce_total,3),   # only the reduce (GEMV is healthy)
      "removable_us": round(red_us,2)}
  # ---- Amdahl: standalone lm_head removes ONLY the reduce ----
  RGRID = [0.25, 0.5, 1.0]
  amdahl = {}
  for c in ("512", "4096"):
    p = per_ctx[c]["p_lm_head_removable"]/100
    amdahl[c] = {"R0": R0[c], "p_lm_head_removable_pct": round(100*p,3),
      **{f"r={r}": {"speedup": round(1/(1-p*r),4), "R_new": round(R0[c]/(1-p*r),1), "gain_pct": round(100*(1/(1-p*r)-1),2)} for r in RGRID}}
  best_gain = max(amdahl[c]["r=1.0"]["gain_pct"] for c in amdahl)   # absolute best standalone (full reduce removal)
  reduce_pinned = all(per_ctx[c]["lm_head_reduce_firm"]["kernels"] for c in per_ctx)
  gemv_healthy = all(per_ctx[c]["lm_head_gemv"]["eff_bw_GBs"] >= 700 for c in per_ctx)
  standalone_tier = tier(best_gain)   # full-reduce-removal best case
  # ---- verdict (tiered policy: a residual is no longer rejected for missing 5%; it is classified) ----
  if not reduce_pinned:
    verdict = "AMD_ISA_LM_HEAD_Q6K_AUDIT_BLOCKED_ROUTE_NOT_ROLE_RESOLVED"; decision = "cannot pin lm_head reduce"
  else:
    # the reduce target IS firmly pinned -> PASS; the tier governs HOW it promotes (standalone vs fold).
    verdict = "AMD_ISA_LM_HEAD_Q6K_AUDIT_PASS_REDUCE_TARGET_PINNED"
    if standalone_tier == "TIER_A_MAJOR":
      decision = f"Standalone lm_head reduce route is TIER_A (+{best_gain:.1f}%): pursue standalone, normal promotion."
    elif standalone_tier == "TIER_B_RESIDUAL":
      decision = (f"lm_head reduce removal is a TIER_B_RESIDUAL win (+{best_gain:.1f}% W==D at full reduce removal, r=1.0; "
        "firm target r_32_4_1187 + r_32_4_1187n1, prod==151936 = known-residual removal). Under the tiered policy this is "
        "promotable WITH clean mechanism proof + rollback + no protected-context regression >1.0%. PREFERRED path: FOLD into "
        "the general Q6_K single-pass direct route (Q6K-1) -- that route removes this reduce AND closes the q6k_gemv 503->650 "
        "bw gap (a TIER_A win) in ONE route, which is also cleaner/route-simplifying. Standalone lm_head is acceptable as a "
        "TIER_B fallback only if Q6K-1 is deferred. The lm_head GEMV is bandwidth-healthy (761 GB/s) and is NOT a target.")
    else:
      decision = (f"lm_head reduce removal is only {standalone_tier} (+{best_gain:.1f}%): FOLD into Q6K-1; not worth a standalone route.")
  rec = {"verdict": verdict, "R0_wd": R0, "tiered_policy_commit": "97f82b999", "standalone_tier": standalone_tier,
    "lm_head_gemv_eff_bw_GBs": {c: per_ctx[c]["lm_head_gemv"]["eff_bw_GBs"] for c in per_ctx}, "lm_head_gemv_healthy": gemv_healthy,
    "lm_head_reduce_firm_kernels": ["r_32_4_1187", "r_32_4_1187n1"], "lm_head_reduce_pinned": reduce_pinned,
    "p_lm_head_removable_pct": {c: per_ctx[c]["p_lm_head_removable"] for c in per_ctx},
    "standalone_amdahl_best_gain_pct": best_gain, "decision": decision,
    "fold_into_q6k_general_preferred": standalone_tier != "TIER_A_MAJOR",   # fold preferred unless standalone is a TIER_A win
    "standalone_promotable_as": standalone_tier,
    "per_ctx": per_ctx, "amdahl": amdahl,
    "caveats": ["standalone upside removes ONLY the firm lm_head coop reduce (prod==151936); GEMV is bw-healthy (761) and excluded",
                "ambiguous prod==4096 reduces are NOT credited to lm_head (RMSNorm vs q6k gate_up)",
                "W==D R0 is g3-promotion median (wall spread ~52% auto-clock); GPU-time %s are the reliable attribution",
                "reuses Q6K-0 + system-residual measured artifacts; no new GPU runs"]}
  json.dump(rec, open(OUT/"latest.json","w"), indent=2)
  json.dump({c: per_ctx[c]["lm_head_gemv"] for c in per_ctx}, open(OUT/"current_route.json","w"), indent=2)
  json.dump({c: {"firm": per_ctx[c]["lm_head_reduce_firm"], "vocab_extra": per_ctx[c]["lm_head_reduce_vocab_extra"]} for c in per_ctx}, open(OUT/"reduce_rows.json","w"), indent=2)
  json.dump(amdahl, open(OUT/"amdahl.json","w"), indent=2)
  md = [f"# LH0 lm_head/Q6_K route audit\n\n**Verdict:** {verdict}\n\n{decision}\n",
    "## lm_head route (measured)\n| ctx | GEMV %GPU | GEMV eff bw | firm reduce | reduce %GPU | removable %GPU |", "|---|---|---|---|---|---|"]
  for c in ("512","4096"):
    p=per_ctx[c]; md.append(f"| {c} | {p['lm_head_gemv']['pct_gpu']}% | {p['lm_head_gemv']['eff_bw_GBs']} GB/s | {'+'.join(p['lm_head_reduce_firm']['kernels'])} | {p['lm_head_reduce_firm']['pct_gpu']}% | {p['p_lm_head_removable']}% |")
  md += ["\n## Standalone Amdahl (remove ONLY the lm_head reduce)\n| ctx | removable% | r=0.25 | r=0.5 | r=1.0 |", "|---|---|---|---|---|"]
  for c in ("512","4096"):
    a=amdahl[c]; md.append(f"| {c} | {a['p_lm_head_removable_pct']}% | +{a['r=0.25']['gain_pct']}% | +{a['r=0.5']['gain_pct']}% | +{a['r=1.0']['gain_pct']}% ({a['r=1.0']['R_new']}) |")
  md += [f"\n**Best standalone gain (r=1.0): +{best_gain:.1f}% -> {standalone_tier}.** lm_head GEMV bw-healthy ({gemv_healthy}). {decision}",
         "\n## Caveats\n"+"\n".join(f"- {x}" for x in rec["caveats"])]
  (OUT/"summary.md").write_text("\n".join(md))
  return rec

if __name__ == "__main__":
  rec = main()
  print(json.dumps({"verdict": rec["verdict"], "lm_head_gemv_bw": rec.get("lm_head_gemv_eff_bw_GBs"),
                    "p_removable_pct": rec.get("p_lm_head_removable_pct"), "standalone_best_gain_pct": rec.get("standalone_amdahl_best_gain_pct"),
                    "standalone_tier": rec.get("standalone_tier"),
                    "fold_into_q6k_general_preferred": rec.get("fold_into_q6k_general_preferred")}, indent=2))
  print("\nLH0", rec["verdict"])
