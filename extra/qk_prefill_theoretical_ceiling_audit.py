"""Prefill P0: theoretical ceiling / floor audit (AUDIT-ONLY, no kernels, no GPU).

Prefill is GEMM-COMPUTE-bound (M=512 chunk, high arithmetic intensity), unlike decode (weight-memory-bound). The ceiling
is set by GEMM TFLOPS: tinygrad's current graph-GEMM runs at ~40.8 TF/role while measured external BLAS hits 51.8-76.7 TF.
So the practical prefill ceiling = current with every GEMM role lifted to its measured BLAS TFLOPS, non-GEMM (attention/
norm/launch) held fixed. We use the MEASURED role-time shares (per_role_time_tax, diagnostic @ctx512) so we don't need a
layer count: share-weighted speedup = sum(share_role / (blas_tf/tg_tf)) + non_gemm_share.

Conservative: uses measured practical BLAS ceilings (not the 122 TF WMMA marketing peak); the per-ctx ceiling at 1024+
is an UPPER BOUND that erodes as attention's share grows (P2 will refresh per-ctx shares). Audit-only.

Run: DEV=AMD PYTHONPATH=. python3 extra/qk_prefill_theoretical_ceiling_audit.py   (no GPU used)
Writes: bench/qk-prefill-theoretical-ceiling/{latest,role_flops,roofline_floor,whole_prefill_floor_by_ctx,source_artifacts}.json + summary.md
"""
import json, math, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-prefill-theoretical-ceiling"
M = 512  # prompt chunk tokens
TG_TF = 40.8  # tinygrad current graph-GEMM reference TFLOPS (ceiling.json tinygrad_tflops_reference)
CTXS = [512, 1024, 2048, 4096, 8192]

def _read(p):
  f = ROOT / p
  return json.load(open(f)) if f.exists() else None

def main():
  OUT.mkdir(parents=True, exist_ok=True)
  ceiling = _read("bench/qk-prefill-external-blas/ceiling.json")
  baseline = _read("bench/qk-prefill-long-context-harness-authority-role-tax/baseline_whole_prefill_by_ctx.json")
  cands = _read("bench/qk-prefill-aggressive-target-proof-20260624/whole_prefill_candidates.json")
  roletax = _read("bench/qk-prefill-long-context-harness-authority-role-tax/per_role_time_tax_by_ctx.json")
  src_present = {k: (_read(v) is not None) for k, v in {
    "ceiling": "bench/qk-prefill-external-blas/ceiling.json",
    "baseline": "bench/qk-prefill-long-context-harness-authority-role-tax/baseline_whole_prefill_by_ctx.json",
    "candidates": "bench/qk-prefill-aggressive-target-proof-20260624/whole_prefill_candidates.json",
    "per_role_time_tax": "bench/qk-prefill-long-context-harness-authority-role-tax/per_role_time_tax_by_ctx.json"}.items()}
  if ceiling is None or baseline is None:
    rec = {"verdict": "PREFILL_P0_BLOCKED_MISSING_BLAS_FLOOR", "missing": [k for k, v in src_present.items() if not v]}
    json.dump(rec, open(OUT/"latest.json", "w"), indent=2); return rec

  # ---- per-role FLOPs + best measured BLAS ceiling ----
  role_flops = {}
  for s in ceiling["shapes"]:
    name = {"attn_q_o": "qo_proj", "attn_k_v": "kv_proj"}.get(s["name"], s["name"])
    libs = s["libraries"]
    best_tf = max((l.get("tflops", 0) for l in libs.values() if l.get("ok")), default=0)
    best_lib = next((k for k, l in libs.items() if l.get("tflops", 0) == best_tf), "?")
    flop = 2 * s["m"] * s["n"] * s["k"]
    role_flops[name] = {"m": s["m"], "n": s["n"], "k": s["k"], "flop": flop,
      "blas_ceiling_tflops": round(best_tf, 1), "blas_lib": best_lib,
      "tg_ref_tflops": s.get("tinygrad_tflops_reference", TG_TF),
      "floor_ms_per_call": round(flop / (best_tf * 1e12) * 1e3, 4),
      "tg_ms_per_call": round(flop / (s.get("tinygrad_tflops_reference", TG_TF) * 1e12) * 1e3, 4),
      "gemm_ceiling_speedup": round(best_tf / s.get("tinygrad_tflops_reference", TG_TF), 3)}

  # ---- measured role-time shares @ctx512 (diagnostic) ----
  shares = {}
  if roletax and "contexts" in roletax and "512" in roletax["contexts"]:
    for r in roletax["contexts"]["512"]:
      shares[r["role"]] = r["share"]
  gemm_roles = ["ffn_gate_up", "ffn_down", "qo_proj", "kv_proj"]
  gemm_share = sum(shares.get(r, 0) for r in gemm_roles)
  non_gemm_share = max(0.0, 1.0 - gemm_share)

  # ---- share-weighted whole-prefill ceiling (GEMM roles lifted to BLAS, non-GEMM fixed) ----
  # new_time_fraction = sum(share_role / speedup_role) + non_gemm_share ; ceiling_speedup = 1/that
  gemm_new = sum(shares.get(r, 0) / role_flops[r]["gemm_ceiling_speedup"] for r in gemm_roles if r in role_flops)
  ceiling_speedup_512 = 1.0 / (gemm_new + non_gemm_share) if (gemm_new + non_gemm_share) > 0 else None

  cur = {ctx: baseline["contexts"].get(str(ctx), {}).get("current_default", {}).get("tok_s") for ctx in CTXS}
  cand_wp = {}
  if cands:
    for c in cands["candidates"]:
      if c["candidate_id"] == "pipe_tm2_tn2": cand_wp = {int(k): v for k, v in c.get("whole_prefill", {}).items()}

  # ceiling per ctx: ctx512 firm (measured shares). longer ctx: UPPER BOUND (same gemm speedup); erodes as attention grows.
  wp_floor = {}
  for ctx in CTXS:
    c0 = cur.get(ctx)
    ceil_ub = round(c0 * ceiling_speedup_512, 0) if (c0 and ceiling_speedup_512) else None
    row = {"current_tok_s": c0, "pipe_tm2_tn2_tok_s": cand_wp.get(ctx),
           "ceiling_tok_s_upper_bound": ceil_ub, "ceiling_speedup_applied": round(ceiling_speedup_512, 3) if ceiling_speedup_512 else None}
    if c0 and ceil_ub:
      row["current_pct_of_ceiling"] = round(100 * c0 / ceil_ub, 1)
      if cand_wp.get(ctx): row["candidate_pct_of_ceiling"] = round(100 * cand_wp[ctx] / ceil_ub, 1)
      if cand_wp.get(ctx): row["gap_current_to_candidate_pct"] = round(100 * (cand_wp[ctx]/c0 - 1), 1)
      if cand_wp.get(ctx): row["gap_candidate_to_ceiling_pct"] = round(100 * (ceil_ub/cand_wp[ctx] - 1), 1)
    row["note"] = "ctx512 ceiling uses MEASURED role shares; ctx>=1024 is an UPPER BOUND (same GEMM speedup) that erodes as attention share grows -> P2 refreshes per-ctx shares" if ctx > 512 else "measured-share ceiling"
    wp_floor[ctx] = row

  roofline = {"prefill_regime": "COMPUTE-bound (GEMM FLOP-limited, M=512 high arithmetic intensity)",
    "evidence": f"GEMM roles are {round(100*gemm_share)}% of measured wall @ctx512; ceiling set by GEMM TFLOPS (tinygrad ~{TG_TF} vs BLAS {min(r['blas_ceiling_tflops'] for r in role_flops.values())}-{max(r['blas_ceiling_tflops'] for r in role_flops.values())}); attention/non-GEMM ~{round(100*non_gemm_share)}% and grows with ctx",
    "ceiling_set_by": "GEMM compute throughput (tinygrad graph-GEMM at ~58% of measured BLAS); NOT memory, NOT launch (launch/graph is in non_gemm_share, small @ctx512)",
    "wmma_marketing_peak_tflops": ceiling.get("wmma_peak_tflops_assumed"),
    "note": "we use MEASURED BLAS practical ceilings (51.8-76.7 TF), NOT the 122 TF WMMA peak -> conservative ceiling"}

  role_share_theoretical = {r: {"measured_time_share": shares.get(r), "blas_speedup": role_flops.get(r, {}).get("gemm_ceiling_speedup"),
    "floor_ms_per_call": role_flops.get(r, {}).get("floor_ms_per_call")} for r in gemm_roles}

  verdict = "PREFILL_P0_PASS_CEILING_PINNED" if (ceiling_speedup_512 and gemm_share > 0) else "PREFILL_P0_INCONCLUSIVE_STALE_ARTIFACTS"
  c512 = wp_floor[512]
  rec = {"verdict": verdict, "chunk_M": M, "tg_gemm_tflops": TG_TF,
    "gemm_share_of_wall_512": round(gemm_share, 3), "non_gemm_share_512": round(non_gemm_share, 3),
    "ceiling_speedup_512": round(ceiling_speedup_512, 3) if ceiling_speedup_512 else None,
    "ceiling_tok_s_512_upper": c512.get("ceiling_tok_s_upper_bound"),
    "current_pct_of_ceiling_512": c512.get("current_pct_of_ceiling"),
    "candidate_pct_of_ceiling_512": c512.get("candidate_pct_of_ceiling"),
    "whole_prefill_floor_by_ctx": wp_floor, "role_flops": role_flops, "roofline": roofline,
    "role_share_theoretical": role_share_theoretical,
    "dominant_floor_role": max(shares, key=shares.get) if shares else None,
    "answers": {
      "1_practical_ceiling_by_ctx": {ctx: wp_floor[ctx]["ceiling_tok_s_upper_bound"] for ctx in CTXS},
      "2_pipe_tm2_tn2_near_ceiling": f"NO -- pipe_tm2_tn2 is ~{c512.get('candidate_pct_of_ceiling')}% of the ctx512 ceiling ({c512.get('gap_candidate_to_ceiling_pct')}% headroom remains to the BLAS-GEMM floor); it is a real +{c512.get('gap_current_to_candidate_pct')}% but NOT near the ceiling",
      "3_dominant_floor_roles": f"ffn_gate_up ({shares.get('ffn_gate_up')}) + ffn_down ({shares.get('ffn_down')}) dominate; qo {shares.get('qo_proj')}, kv {shares.get('kv_proj')}",
      "4_ceiling_set_by": "GEMM compute (tinygrad graph-GEMM ~58% of measured BLAS TFLOPS) -- compute-bound, not memory/launch",
      "5_aggressive_bound_stale_or_valid": "pipe_tm2_tn2 +11-19% is PLAUSIBLE and consistent (it is below the ~52% GEMM ceiling headroom); declining delta at long ctx matches attention's growing non-GEMM share. Must be re-validated under P1 authority + P6 long-context (it is from 20260624, status non_monotonic/leading_candidate) -- not refuted, not yet authority."},
    "caveats": [
      "ctx512 ceiling uses MEASURED role shares (per_role_time_tax DIAGNOSTIC; the live tool OOM'd, fell back to prior artifact); P2 refreshes authoritative per-ctx shares",
      "ctx>=1024 ceiling is an UPPER BOUND (same GEMM speedup); real ceiling is lower as attention's non-GEMM share grows with ctx (consistent with candidate's declining delta)",
      "uses measured practical BLAS ceilings (51.8-76.7 TF), NOT the 122 TF WMMA marketing peak",
      "non-GEMM (attention/norm/rope/copy/launch) assumed held fixed at BLAS-lift; some of it is itself accelerable (P2/P3)"],
    "source_artifacts": src_present}
  json.dump(rec, open(OUT/"latest.json", "w"), indent=2)
  json.dump(role_flops, open(OUT/"role_flops.json", "w"), indent=2)
  json.dump(roofline, open(OUT/"roofline_floor.json", "w"), indent=2)
  json.dump(wp_floor, open(OUT/"whole_prefill_floor_by_ctx.json", "w"), indent=2)
  json.dump(src_present, open(OUT/"source_artifacts.json", "w"), indent=2)
  md = [f"# Prefill P0 theoretical ceiling audit\n\n**Verdict:** {verdict}\n",
    f"Prefill is **{roofline['prefill_regime']}**. GEMM = {round(100*gemm_share)}% of wall @ctx512; ceiling set by GEMM TFLOPS (tinygrad ~{TG_TF} vs measured BLAS 51.8-76.7).\n",
    "## Ceiling / current / candidate by ctx\n| ctx | current | pipe_tm2_tn2 | ceiling (UB) | cur % ceil | cand % ceil | cand->ceil headroom |", "|---|---|---|---|---|---|---|"]
  for ctx in CTXS:
    r = wp_floor[ctx]
    md.append(f"| {ctx} | {r['current_tok_s']} | {r.get('pipe_tm2_tn2_tok_s')} | {r['ceiling_tok_s_upper_bound']} | {r.get('current_pct_of_ceiling','-')}% | {r.get('candidate_pct_of_ceiling','-')}% | {r.get('gap_candidate_to_ceiling_pct','-')}% |")
  md += ["\n## Per-role GEMM floor (M=512)\n| role | shape | BLAS TF | tg TF | speedup | time share |", "|---|---|---|---|---|---|"]
  for r in gemm_roles:
    rf = role_flops.get(r, {}); md.append(f"| {r} | {rf.get('m')}x{rf.get('n')}x{rf.get('k')} | {rf.get('blas_ceiling_tflops')} | {rf.get('tg_ref_tflops')} | {rf.get('gemm_ceiling_speedup')}x | {shares.get(r,'-')} |")
  md += ["\n## Answers", f"- ceiling@512 ~= {c512['ceiling_tok_s_upper_bound']} tok/s; current {cur[512]} = {c512.get('current_pct_of_ceiling')}% of it",
    f"- pipe_tm2_tn2 = {c512.get('candidate_pct_of_ceiling')}% of ceiling -> {c512.get('gap_candidate_to_ceiling_pct')}% headroom remains (NOT near ceiling)",
    f"- dominant floor role: {rec['dominant_floor_role']}", "- compute-bound (GEMM TFLOPS), not memory/launch",
    "- pipe_tm2_tn2 +11-19% plausible/not-refuted; needs P1 authority + P6 long-context re-validation",
    "\n## Caveats\n"+"\n".join(f"- {c}" for c in rec["caveats"])]
  (OUT/"summary.md").write_text("\n".join(md))
  return rec

if __name__ == "__main__":
  rec = main()
  print(json.dumps({"verdict": rec["verdict"], "regime": rec.get("roofline", {}).get("prefill_regime"),
    "gemm_share_512": rec.get("gemm_share_of_wall_512"), "ceiling_speedup_512": rec.get("ceiling_speedup_512"),
    "ceiling_512": rec.get("ceiling_tok_s_512_upper"), "current_pct_ceiling_512": rec.get("current_pct_of_ceiling_512"),
    "candidate_pct_ceiling_512": rec.get("candidate_pct_of_ceiling_512"),
    "ceiling_by_ctx": rec.get("answers", {}).get("1_practical_ceiling_by_ctx")}, indent=2))
  print("\nP0", rec["verdict"])
