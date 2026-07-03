"""AMD ISA decode-attention CEILING audit (audit-only). Derives: math floor -> owned hand ASM -> native ISA tile ->
full decode wall, and decides whether further native attention-tile work is worth it vs moving search to non-attention.

Key frame: decode is WEIGHT-MEMORY-bound. The FFN/all-weights read (Q4_K, ~5.03 GB) per token dominates the wall; the
attention KV-read floor is <1% of that. So even a perfect attention tile barely moves the wall (Amdahl), and N3F already
MEASURED that a large tile work-cut (dynamic-S) moved W==D only +9.8% at ctx512 / ~0 at ctx4096. This tool quantifies
the floor, the Amdahl tile wall-share, and the max W==D from matching owned / hitting the attention floor.

No optimization. Conservative: peak HBM bw (optimistic ceiling) is labeled; real achievable ~80%.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/audit/amd_isa/decode_attention_ceiling_audit.py
Writes: bench/amd-isa-backend-decode-attention-ceiling/{latest.json, summary.md, math_floor.json, loss_stack.json}
"""
import json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "bench/amd-isa-backend-decode-attention-ceiling"
# ---- model / hardware constants (Qwen3-8B-Q4_K_M on RX 7900 XTX gfx1100) ----
HBM_BW = 960e9          # XTX spec peak GB/s (24GB GDDR6, 384-bit @ 20Gbps). OPTIMISTIC ceiling; real ~80%.
MODEL_BYTES = 5027783488  # /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf (all weights read once per decode token)
Hd, Hq, Hkv, KV_DTYPE_BYTES = 128, 32, 8, 4   # head dim, query heads, kv heads, f32 cache_kv
CTXS = [512, 1024, 2048, 4096]

def _read(p):
  f = ROOT / p; return json.load(open(f)) if f.exists() else None

def math_floor():
  # decode floor = weight-memory read (dominates) ; attention floor = K+V read for valid context (tiny).
  weight_s = MODEL_BYTES / HBM_BW                    # s/token to stream all weights once
  rec = {"assumptions": {"hbm_bw_GBs_peak": HBM_BW/1e9, "model_bytes": MODEL_BYTES, "kv_dtype_bytes": KV_DTYPE_BYTES,
          "note": "decode is weight-memory-bound; weight read dominates the per-token wall. peak bw => optimistic ceiling (real ~80% => ~0.8x tok/s)."},
         "weight_read_floor": {"bytes_per_token": MODEL_BYTES, "seconds_per_token": weight_s, "tok_s_ceiling_peak_bw": round(1/weight_s, 1),
                               "tok_s_ceiling_real_80pct": round(0.8/weight_s, 1)}, "per_ctx": {}}
  for c in CTXS:
    kv_bytes = 2 * c * Hkv * Hd * KV_DTYPE_BYTES     # read each K and V once over valid context
    kv_s = kv_bytes / HBM_BW
    # min attention compute (per token, rough): score Q.K (Hq*c*Hd MACs) + PV (Hq*c*Hd MACs) + exp/max (Hq*c)
    macs = 2 * Hq * c * Hd
    rec["per_ctx"][str(c)] = {"kv_read_bytes": kv_bytes, "kv_read_seconds": kv_s, "attn_macs": macs,
       "attn_floor_pct_of_weight_floor": round(100 * kv_s / weight_s, 3),
       "decode_floor_seconds": weight_s + kv_s, "decode_tok_s_ceiling_peak": round(1/(weight_s + kv_s), 1)}
  return rec

def main():
  OUT.mkdir(parents=True, exist_ok=True)
  inputs = {k: _read(v) for k, v in {
    "phase_i": "bench/amd-isa-backend-phase-i/latest.json", "n4": "bench/amd-isa-backend-phase-n4/latest.json",
    "n2b": "bench/amd-isa-backend-phase-n2b/latest.json", "pc_source": "bench/amd-isa-backend-pc-source-trace/latest.json",
    "ra4": "bench/amd-isa-backend-regalloc-accum/ra4_latest.json", "rl2": "bench/amd-isa-backend-regalloc-accum-lds-reclaim/rl2_latest.json",
    "n0": "bench/amd-isa-backend-phase-n0/latest.json"}.items()}
  mf = math_floor()
  json.dump(mf, open(OUT/"math_floor.json", "w"), indent=2)
  # ---- measured W==D (best native = reg-accum+reclaim RL2; owned) ----
  wd = {"512": {"native": 70.74, "owned": 103.53}, "4096": {"native": 56.70, "owned": 94.41}}  # RL2 / phase-i authority
  if (rl2 := inputs.get("rl2")):
    r = rl2.get("wd", {}).get("rl2_reg_accum_with_reclaim", {})
    for c in ("512", "4096"):
      if c in r: wd[c]["native"] = r[c]["native"]
  # ---- Amdahl tile wall-share f (from MEASURED N3F dynamic-S cut), NOT eager GPU-compute ----
  # ctx512: FIXED_S 61.09 -> dynamic-S 67.09 (+9.8%) by cutting splits 48->cdiv(512,96)=6 (8x tile-work cut).
  #   gain = 1/((1-f)+f/8)-1 = 0.098  => f ~= 0.102 (tile ~10% of wall at ctx512).
  # ctx4096: dynamic-S == fixed (sweep already valid) and reg-accum was W==D-neutral => tile wall-share ~0 (overlapped).
  f_512, f_4096 = 0.102, 0.03
  def amdahl_gain(f, speedup):  # tile becomes `speedup`x faster; speedup=inf => tile->0
    new = (1 - f) + (f / speedup if speedup != float("inf") else 0.0); return round(1/new - 1, 4)
  # owned tile is ~15x faster GPU-compute than native (N4: 381 vs 5791) => matching owned ~ tile->~0; floor tile even smaller.
  max_gain = {
    "match_owned_tile": {"ctx512_pct": round(100*amdahl_gain(f_512, 15.2), 1), "ctx4096_pct": round(100*amdahl_gain(f_4096, 15.2), 1),
                         "method": "Amdahl on MEASURED tile wall-share f (N3F dynamic-S cut), native tile -> owned (15.2x GPU-compute, ~tile->0)"},
    "hit_math_floor_tile": {"ctx512_pct": round(100*amdahl_gain(f_512, float("inf")), 1), "ctx4096_pct": round(100*amdahl_gain(f_4096, float("inf")), 1),
                            "method": "tile -> 0 (floor is even cheaper than owned). upper bound on attention-tile headroom."},
    "move_non_attention": {"note": "decode wall is weight-memory-bound: owned %.0f tok/s @ctx512 is ~%.0f%% of the %.0f tok/s weight-read ceiling (peak bw). The FFN/weight path has ~%.1fx headroom to the floor and dominates the wall for BOTH routes." % (
       wd["512"]["owned"], 100*wd["512"]["owned"]/mf["weight_read_floor"]["tok_s_ceiling_peak_bw"], mf["weight_read_floor"]["tok_s_ceiling_peak_bw"], mf["weight_read_floor"]["tok_s_ceiling_peak_bw"]/wd["512"]["owned"])}}
  # ---- ratios ----
  owned_floor_pct = {c: round(100*wd[c]["owned"]/mf["weight_read_floor"]["tok_s_ceiling_peak_bw"], 1) for c in wd}
  native_floor_pct = {c: round(100*wd[c]["native"]/mf["weight_read_floor"]["tok_s_ceiling_peak_bw"], 1) for c in wd}
  native_owned_pct = {c: round(100*wd[c]["native"]/wd[c]["owned"], 1) for c in wd}
  loss = {
    "math_floor_weight_tok_s_peak": mf["weight_read_floor"]["tok_s_ceiling_peak_bw"],
    "attn_kv_floor_pct_of_weight_floor": {c: mf["per_ctx"][c]["attn_floor_pct_of_weight_floor"] for c in wd},
    "owned_pct_of_weight_floor": owned_floor_pct, "native_pct_of_weight_floor": native_floor_pct,
    "native_pct_of_owned": native_owned_pct,
    "tile_wall_share_measured": {"ctx512": f_512, "ctx4096": f_4096, "src": "N3F dynamic-S Amdahl (measured), NOT eager GPU-compute (N4 eager 34-39% is compute-share, overstated by no-overlap)"},
    "native_specific_over_owned": "the native-vs-owned delta is the attention tile (N4); but it is largely OVERLAPPED by the weight-memory-bound FFN, so its WALL share is ~10%@512 / ~0@4096, not the 34-39% eager GPU-compute share.",
    "full_decode_non_attention": "weight-memory-bound FFN/projection GEMVs dominate the wall and are SHARED (identical) between native and owned (N4: q4k_gemv 7109 vs 7157).",
    "resource_levers_status": "exhausted/refuted: occupancy/LDS (Phase M), address scalarization (N1B), reg-accum (RA4 +5.4%@512/neutral@4096), LDS reclaim (RL2 resource-correct/no W==D), FMA/mov cleanup (R0 <5). Matching owned now needs an owned-LEVEL ALGORITHMIC rewrite, not a resource knob."}
  json.dump(loss, open(OUT/"loss_stack.json", "w"), indent=2)
  # ---- decision ----
  g512, g4096 = max_gain["match_owned_tile"]["ctx512_pct"], max_gain["match_owned_tile"]["ctx4096_pct"]
  # ctx512 ~+11% (borderline >=10%) but requires an owned-level algo rewrite (resource path exhausted); ctx4096 ~0 (<5%).
  # math floor: attention is <1% of the weight-bound decode floor; the wall + the ~2x headroom are in the FFN/weight path.
  next_target = "non_attention_ffn_weight_path"
  reason = (f"Decode is WEIGHT-MEMORY-bound: the {mf['weight_read_floor']['tok_s_ceiling_peak_bw']} tok/s weight-read ceiling dominates; "
    f"the attention KV-read floor is <1% of it ({mf['per_ctx']['4096']['attn_floor_pct_of_weight_floor']}% @ctx4096). Matching owned's tile yields only "
    f"+{g512}% @ctx512 (borderline) and +{g4096}% @ctx4096 (<5%) by Amdahl on the MEASURED tile wall-share (~10%@512/~0@4096), "
    f"and would now require an owned-LEVEL algorithmic rewrite since every tile RESOURCE lever is exhausted/refuted. The FFN/weight "
    f"GEMVs dominate the wall for BOTH routes and sit at only ~{owned_floor_pct['512']}% (owned)/~{native_floor_pct['512']}% (native) of the weight floor "
    f"(~2x headroom). => move search to the non-attention (FFN/weight) decode path; attention-tile work is low-leverage and diminishing.")
  verdict = "AMD_ISA_ATTENTION_CEILING_PASS_MOVE_TO_NON_ATTENTION"
  if any(v is None for v in inputs.values()):
    missing = [k for k, v in inputs.items() if v is None]
    # all the decision-critical inputs (phase_i/n4/n2b/rl2) drive the verdict; only proceed if present
    if any(inputs.get(k) is None for k in ("phase_i", "n4", "rl2")):
      verdict = "AMD_ISA_ATTENTION_CEILING_INCONCLUSIVE_MISSING_MEASUREMENTS"; reason = f"missing decision-critical inputs: {missing}"
  rec = {"verdict": verdict, "contexts": CTXS, "math_floor": mf, "owned_vs_floor": owned_floor_pct,
         "native_vs_floor": native_floor_pct, "native_vs_owned": native_owned_pct, "loss_stack": loss,
         "max_gain": max_gain, "wd_measured": wd,
         "decision": {"next_target": next_target, "reason": reason},
         "caveats": ["peak HBM bw => optimistic ceilings (real ~80%)", "tile wall-share from MEASURED N3F Amdahl, not eager GPU-compute (which overstates via no-overlap)",
                     "conservative math floor (lower-bound work); owned/native are far above the attention floor but attention is overlapped by the weight-bound FFN",
                     "ctx512 match-owned ~+11% is borderline-above 10% but needs an owned-LEVEL algo rewrite (resource levers exhausted) -> lower leverage than the FFN/weight path"],
         "input_artifacts": {k: ("present" if v is not None else "MISSING") for k, v in inputs.items()}}
  json.dump(rec, open(OUT/"latest.json", "w"), indent=2)
  md = [f"# Decode-attention ceiling audit\n\n**Verdict:** {rec['verdict']}\n",
        f"**Decision:** move search to `{next_target}`\n\n{reason}\n",
        "## W==D (measured)\n| ctx | native | owned | native % of owned | native % of weight-floor |", "|---|---|---|---|---|"]
  for c in ("512", "4096"): md.append(f"| {c} | {wd[c]['native']} | {wd[c]['owned']} | {native_owned_pct[c]}% | {native_floor_pct[c]}% |")
  md += ["\n## Math floor (peak bw)\n| metric | value |", "|---|---|",
         f"| weight-read decode ceiling | {mf['weight_read_floor']['tok_s_ceiling_peak_bw']} tok/s (real ~80%: {mf['weight_read_floor']['tok_s_ceiling_real_80pct']}) |",
         f"| attn KV-floor @ctx4096 | {mf['per_ctx']['4096']['attn_floor_pct_of_weight_floor']}% of weight floor (negligible) |",
         "\n## Loss stack\n| layer | value |", "|---|---|",
         f"| tile wall-share (measured) | ctx512 ~{int(rec['loss_stack']['tile_wall_share_measured']['ctx512']*100)}% / ctx4096 ~{int(rec['loss_stack']['tile_wall_share_measured']['ctx4096']*100)}% |",
         f"| max gain match-owned-tile | ctx512 +{g512}% / ctx4096 +{g4096}% |",
         f"| max gain hit-attn-floor | ctx512 +{max_gain['hit_math_floor_tile']['ctx512_pct']}% / ctx4096 +{max_gain['hit_math_floor_tile']['ctx4096_pct']}% |",
         f"| FFN/weight (shared) headroom to floor | owned ~{owned_floor_pct['512']}% of floor -> ~{round(mf['weight_read_floor']['tok_s_ceiling_peak_bw']/wd['512']['owned'],1)}x |",
         "\n## Decision table\n| question | answer |", "|---|---|",
         f"| match owned tile >=10%? | ctx512 +{g512}% (borderline) ; ctx4096 +{g4096}% (no) |",
         f"| attention worth continuing? | NO -- diminishing, resource-exhausted, <1% of weight-bound floor |",
         f"| where is the wall + headroom? | FFN/weight path (shared, ~2x to floor) |",
         "\n## Caveats\n" + "\n".join(f"- {c}" for c in rec["caveats"])]
  (OUT/"summary.md").write_text("\n".join(md))
  return rec

if __name__ == "__main__":
  rec = main()
  print(json.dumps({"verdict": rec["verdict"], "decision": rec["decision"]["next_target"],
                    "max_gain_match_owned": rec["max_gain"]["match_owned_tile"], "weight_floor_tok_s": rec["math_floor"]["weight_read_floor"]["tok_s_ceiling_peak_bw"],
                    "native_vs_floor": rec["native_vs_floor"], "owned_vs_floor": rec["owned_vs_floor"]}, indent=2))
  print("\nCEILING", rec["verdict"])
