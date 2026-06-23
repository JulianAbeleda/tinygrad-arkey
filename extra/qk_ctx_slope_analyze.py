#!/usr/bin/env python3
"""Ctx-slope audit analysis (no measurement): consumes wd_by_ctx.json (W==D authority) + kernel_attribution_{A,B}.json
(PROFILE attribution) and emits slope_fit.json + llama_comparison.json. Linear fit ms = fixed + slope*ctx (least
squares), decomposes the A-vs-B saving into fixed materialization (E_49152) + attention-tile ctx-delta. Attribution
timings are NOT promotion authority (per HARNESS_GUIDE); wall fits use the W==D medians."""
from __future__ import annotations
import json, pathlib

OUT = pathlib.Path("bench/qk-decode-ctx-slope-audit")
LLAMA = {512: 97.71, 1024: 97.39, 2048: 95.0, 4096: 92.37}  # post-parity authority.json refs (tok/s)

def lsq(xs, ys):
  n = len(xs); sx = sum(xs); sy = sum(ys); sxx = sum(x*x for x in xs); sxy = sum(x*y for x, y in zip(xs, ys))
  slope = (n*sxy - sx*sy) / (n*sxx - sx*sx); fixed = (sy - slope*sx) / n
  resid = [y - (fixed + slope*x) for x, y in zip(xs, ys)]
  return fixed, slope, max(abs(r) for r in resid)

def fit_row(label, ctxs, ms):
  fixed, slope_per_tok, err = lsq(ctxs, ms)
  return {"config": label, "fixed_ms": round(fixed, 4), "slope_ms_per_1k_ctx": round(slope_per_tok*1000, 4),
          "max_resid_ms": round(err, 4), "ms_by_ctx": {str(c): round(m, 4) for c, m in zip(ctxs, ms)}}

def main():
  wd = json.loads((OUT/"wd_by_ctx.json").read_text())
  ctxs = [int(c) for c in wd["ckpts"]]
  A_ms = [wd["configs"]["A_whole_default"][str(c)]["ms_token"]["median"] for c in ctxs]
  B_ms = [wd["configs"]["B_slice_identity0"][str(c)]["ms_token"]["median"] for c in ctxs]
  L_ms = [1000.0/LLAMA[c] for c in ctxs]
  saved = [b-a for a, b in zip(A_ms, B_ms)]

  # attribution: extract exact tile + materialization per ctx
  def load(p): return {r["ctx"]: r for r in json.loads((OUT/p).read_text())["rows"]}
  Aatt, Batt = load("kernel_attribution_A.json"), load("kernel_attribution_B.json")
  def kern(row, name): return row["top_kernels"].get(name, 0.0)/1e3  # ms
  A_tile = [kern(Aatt[c], "owned_flash_tile_gqa_whole") for c in ctxs]
  B_tile = [kern(Batt[c], "owned_flash_tile_gqa") for c in ctxs]
  E49152 = [(kern(Batt[c], "E_49152_32_3") + kern(Batt[c], "E_49152_32_3n1")) for c in ctxs]

  slope_fit = {
    "authority": "wall fits use W==D ms medians (promotion-grade); tile/materialization use PROFILE GPU-busy (attribution only)",
    "flag_stack": "Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1 Q4K_GEMV_WARP_PROJ=1 (canonical; reproduces doc 102.9/101.3/98.7/94.2)",
    "fits": [fit_row("A_whole_default", ctxs, A_ms), fit_row("B_slice_identity0", ctxs, B_ms),
             fit_row("llama_cpp_ref", ctxs, L_ms), fit_row("saved_B_minus_A", ctxs, saved)],
    "attribution_ms_by_ctx": {
      "A_whole_tile": dict(zip(map(str, ctxs), [round(x, 3) for x in A_tile])),
      "B_slice_tile":  dict(zip(map(str, ctxs), [round(x, 3) for x in B_tile])),
      "E_49152_materialization_B_only": dict(zip(map(str, ctxs), [round(x, 3) for x in E49152])),
    },
    "attribution_slopes_ms_per_1k_ctx": {
      "A_whole_tile": round(lsq(ctxs, A_tile)[1]*1000, 4),
      "B_slice_tile": round(lsq(ctxs, B_tile)[1]*1000, 4),
      "E_49152_materialization": round(lsq(ctxs, E49152)[1]*1000, 4),
    },
    "saving_decomposition_ms": [
      {"ctx": c, "saved_wall_ms": round(s, 3), "fixed_materialization_E49152_ms": round(e, 3),
       "tile_penalty_A_minus_B_ms": round(at-bt, 3),
       "note": "saved ~= E49152(fixed) - tile_penalty(grows with ctx) + other(combine/overlap)"}
      for c, s, e, at, bt in zip(ctxs, saved, E49152, A_tile, B_tile)],
  }
  (OUT/"slope_fit.json").write_text(json.dumps(slope_fit, indent=2))

  # llama comparison
  a_fixed, a_slope, _ = lsq(ctxs, A_ms); l_fixed, l_slope, _ = lsq(ctxs, L_ms)
  # crossover ctx where A_ms == llama_ms (A currently faster i.e. lower ms; steeper slope -> crosses above)
  cross = (l_fixed - a_fixed)/(a_slope - l_slope) if a_slope != l_slope else None
  tg_vs_llama = {str(c): round(100*(1000/a)/LLAMA[c], 1) for c, a in zip(ctxs, A_ms)}
  llama_cmp = {
    "tinygrad_A_vs_llama_pct_by_ctx": tg_vs_llama,
    "A_slope_ms_per_1k": round(a_slope*1000, 4), "llama_slope_ms_per_1k": round(l_slope*1000, 4),
    "A_steeper_than_llama": a_slope > l_slope,
    "slope_ratio_A_over_llama": round(a_slope/l_slope, 2),
    "projected_crossover_ctx_A_falls_below_llama": round(cross) if cross else None,
    "max_supported_ctx_MAXC": 4608,
    "within_supported_ctx_A_stays_above_llama": (cross is None or cross > 4608),
    "questions": {
      "tinygrad_above_llama_at_all_measured_ctx": all(v >= 100 for v in tg_vs_llama.values()),
      "tinygrad_worse_ctx_linear_slope_than_llama": a_slope > l_slope,
      "ctx4096_margin_lower_due_to": "higher A slope (A 0.25 vs llama 0.165 ms/1k) eroding a large fixed advantage",
      "remaining_long_ctx_attention_inefficiency": "yes, bounded: whole-cache strided K/V read slope > slice/contiguous",
      "worth_bounded_long_ctx_tile_search": "marginal: ~+1.9% at ctx4096, ~0% at ctx512; A already >= llama within MAXC",
    },
  }
  (OUT/"llama_comparison.json").write_text(json.dumps(llama_cmp, indent=2))

  print("=== SLOPE FITS (ms = fixed + slope*ctx) ===")
  for f in slope_fit["fits"]:
    print(f"  {f['config']:<20} fixed {f['fixed_ms']:7.3f}ms  slope {f['slope_ms_per_1k_ctx']:+.3f} ms/1k  (resid {f['max_resid_ms']:.3f})")
  print("=== ATTRIBUTION SLOPES ===")
  for k, v in slope_fit["attribution_slopes_ms_per_1k_ctx"].items(): print(f"  {k:<28} {v:+.3f} ms/1k")
  print("=== SAVING DECOMPOSITION ===")
  for r in slope_fit["saving_decomposition_ms"]:
    print(f"  ctx{r['ctx']:>5}: saved {r['saved_wall_ms']:.3f}  = E49152(fixed) {r['fixed_materialization_E49152_ms']:.3f}  - tile_penalty {r['tile_penalty_A_minus_B_ms']:+.3f}")
  print("=== LLAMA ===")
  print(f"  A slope {llama_cmp['A_slope_ms_per_1k']:.3f} vs llama {llama_cmp['llama_slope_ms_per_1k']:.3f} ms/1k (ratio {llama_cmp['slope_ratio_A_over_llama']})")
  print(f"  tg/llama %: {tg_vs_llama}")
  print(f"  projected crossover ctx: {llama_cmp['projected_crossover_ctx_A_falls_below_llama']} (MAXC {llama_cmp['max_supported_ctx_MAXC']}) -> within-ctx A>=llama: {llama_cmp['within_supported_ctx_A_stays_above_llama']}")

if __name__ == "__main__":
  main()
