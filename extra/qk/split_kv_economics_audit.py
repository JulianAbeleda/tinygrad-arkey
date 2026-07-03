#!/usr/bin/env python3
"""Split-KV economics audit for decode-attention candidates (durable, reusable).

WHY THIS EXISTS
---------------
Earlier decode-attention audits caught kernel quality (tinygrad can't emit a llama-quality tile),
graph integration (B2 batched the launches, B4 made the owned `.co` a JIT graph node), and raw
dispatch overhead. They did NOT explicitly require split-KV *reduction economics* before a W==D
promotion attempt. B4 exposed the next lifecycle layer: a flash-decode tile that splits the KV cache
into `S` chunks must MERGE the per-split partials in a separate `combine` kernel, and that combine is
a flat latency/occupancy floor that gives back part of the tile win -- so a local attention A/B win
can be non-promotable for reasons the tile A/B never measures.

This audit makes the split-KV economics a FIRST-CLASS, PERMANENT check. For each context it reports,
from a tile/combine attribution artifact (no remeasure by default):
  - tile_us / combine_us / total_us / combine_fraction
  - combine_bytes_est and combine_effective_gbps (vs HBM peak -> latency- vs bandwidth-bound)
  - tile_workgroups / combine_workgroups and an occupancy proxy (workgroups vs CU count)
  - the per-context optimal split S (min total attention us)
  - an Amdahl projection of whole-decode W==D for measured / half / free combine
  - a classification: COMBINE_TAX_DOMINATES | COMBINE_SMALL_AMDAHL_LIMIT | POLICY_ONLY | MEASUREMENT_UNSTABLE

It only reads measured artifacts + computes derived economics; it builds NO kernel, changes NO default.

DEFAULT (read-only) -- reuse the B4 combine-tax data, do not remeasure:
  DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk/split_kv_economics_audit.py

LIVE (regenerate the attribution first via the B4 harness, then audit) -- only if fields are missing:
  DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk/split_kv_economics_audit.py --live

GENERAL (audit any candidate that emits the same attribution schema):
  ... extra/qk/split_kv_economics_audit.py --attribution <tile_combine.json> --wd <routed_wd.json> \
      --candidate <id> --out bench/<dir>

ATTRIBUTION SCHEMA the audit consumes (the contract for future split-KV candidates):
  {"rows":[{"ctx","S","tile_us","combine_us","total_us","combine_frac","tile_workgroups","combine_bytes"}...]}
WD SCHEMA (optional, for the measured Amdahl anchor + operative S):
  {"routed_per_ctx":{"<ctx>":{"best_S","delta_pct","spread_pct","route_fired"}...}}  (qk_b4_policy_sweep output)
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
ATTRIBUTION_DEFAULT = ROOT / "bench/qk-decode-attention-route-b-b4-combine-tax/latest.json"
WD_DEFAULT = ROOT / "bench/qk-decode-attention-route-b-b4-combine-tax/policy_sweep.json"
OUT_DEFAULT = ROOT / "bench/qk-split-kv-economics-audit"

# --- machine / model constants (gfx1100 RX 7900 XTX, Qwen3-8B-Q4_K_M) ----------------------------------------------
CU_COUNT = 96            # gfx1100 compute units
HBM_PEAK_GBPS = 960.0    # RX 7900 XTX HBM bandwidth peak
N_LAYERS = 36            # Qwen3-8B transformer blocks (attention runs once per layer per token)
COMBINE_WORKGROUPS = 32  # owned_flash_combine launch grid = (Hq=32,1,1): one workgroup per query head
COMBINE_THREADS_PER_WG = 32

# --- promotion gate (the W==D bar a split-KV candidate must clear to promote) --------------------------------------
GATE_CTX1024_PCT = 5.0
GATE_CTX4096_PCT = 7.0
# baseline whole-decode token time (ms) per ctx -- the canonical default decode curve
# (bench/qk-decode-runtime-overhead/result.json, gqa_coop_vec default). ctx2048 interpolated (flagged).
BASE_TOK_MS = {512: 14.619, 1024: 15.015, 2048: 15.479, 4096: 16.408}
BASE_TOK_MS_INTERPOLATED = {2048}


def _load(p: pathlib.Path) -> dict:
  return json.loads(pathlib.Path(p).read_text())


def _rows_by_ctx(attr: dict) -> dict[int, list[dict]]:
  by: dict[int, list[dict]] = {}
  for r in attr["rows"]:
    by.setdefault(int(r["ctx"]), []).append(r)
  return by


def _pick_row(rows: list[dict], S: int | None) -> dict:
  """Return the row at split S, else the min-total-us row."""
  if S is not None:
    for r in rows:
      if int(r["S"]) == int(S):
        return r
  return min(rows, key=lambda r: r["total_us"])


def _amdahl(base_ms: float, d_meas_pct: float, combine_us: float) -> dict:
  """Project whole-decode W==D under measured / halved / free combine.

  Model (additive, anchored on the MEASURED routed delta):
    The owned tile+combine replaces coop attention in all N_LAYERS layers. The measured routed delta
    d_meas implies a per-token saving:
        saved_meas = T_base * (d_meas/100) / (1 + d_meas/100)
    Making the combine cheaper by fraction f frees an extra N_LAYERS*combine_us*f per token:
        saved(f)   = saved_meas + N_LAYERS*combine_us*f                      (f=0 measured, .5 half, 1 free)
        delta(f)   = saved(f) / (T_base - saved(f)) * 100
  This reproduces the B4 result-doc projection (ctx4096: ~+5.4% measured -> ~+7% half -> ~+8.6% free).
  Honest scope: meaningful only where the tile already wins (route fires, d_meas > 0); a degenerate
  d_meas<=0 (tile loses, route off) is reported as null projection (cheaper combine cannot rescue a
  losing tile -- that is a tile problem, not a combine problem)."""
  base_us = base_ms * 1e3
  if d_meas_pct is None or d_meas_pct <= 0.0:
    return {"measured_pct": (None if d_meas_pct is None else round(d_meas_pct, 2)),
            "half_combine_pct": None, "free_combine_pct": None,
            "note": "route off / tile does not win here -> a cheaper combine cannot make it promotable (tile-bound, not combine-bound)"}
  saved_meas = base_us * (d_meas_pct / 100.0) / (1.0 + d_meas_pct / 100.0)
  combine_total = N_LAYERS * combine_us

  def delta(f: float) -> float:
    saved = saved_meas + combine_total * f
    return saved / (base_us - saved) * 100.0

  return {"measured_pct": round(delta(0.0), 2), "half_combine_pct": round(delta(0.5), 2),
          "free_combine_pct": round(delta(1.0), 2),
          "combine_ms_per_token": round(combine_total / 1e3, 3),
          "base_ms": round(base_ms, 3), "n_layers": N_LAYERS}


def audit(attr: dict, wd: dict | None, candidate: str) -> dict:
  rows_by_ctx = _rows_by_ctx(attr)
  routed = (wd or {}).get("routed_per_ctx", {})
  ctxs = sorted(rows_by_ctx)

  per_ctx = []
  for ctx in ctxs:
    rows = rows_by_ctx[ctx]
    opt_min_total = int(min(rows, key=lambda r: r["total_us"])["S"])
    wd_row = routed.get(str(ctx)) or routed.get(ctx)
    # operative S = the split the W==D was actually measured at (so the Amdahl anchor + economics agree);
    # fall back to the min-total optimal S when no W==D anchor exists.
    operative_S = int(wd_row["best_S"]) if wd_row and wd_row.get("best_S") is not None else opt_min_total
    r = _pick_row(rows, operative_S)
    combine_us = float(r["combine_us"]); tile_us = float(r["tile_us"])
    total_us = float(r.get("total_us", tile_us + combine_us))
    combine_bytes = int(r["combine_bytes"])
    eff_gbps = combine_bytes / (combine_us * 1e-6) / 1e9 if combine_us else None
    tile_wg = int(r["tile_workgroups"])

    d_meas = wd_row.get("delta_pct") if wd_row else None
    route_fired = bool(wd_row.get("route_fired")) if wd_row else False
    spread = wd_row.get("spread_pct") if wd_row else None
    base_ms = BASE_TOK_MS.get(ctx)
    proj = _amdahl(base_ms, d_meas, combine_us) if base_ms is not None else {
      "measured_pct": d_meas, "half_combine_pct": None, "free_combine_pct": None, "note": "no baseline token time for this ctx"}

    per_ctx.append({
      "ctx": ctx,
      "operative_S": operative_S,
      "optimal_S_min_total": opt_min_total,
      "tile_us": round(tile_us, 2),
      "combine_us": round(combine_us, 2),
      "total_us": round(total_us, 2),
      "combine_fraction": round(combine_us / total_us, 3) if total_us else None,
      "combine_bytes_est": combine_bytes,
      "combine_effective_gbps": round(eff_gbps, 1) if eff_gbps is not None else None,
      "combine_pct_of_hbm_peak": round(100.0 * eff_gbps / HBM_PEAK_GBPS, 1) if eff_gbps is not None else None,
      "tile_workgroups": tile_wg,
      "combine_workgroups": COMBINE_WORKGROUPS,
      "combine_threads_total": COMBINE_WORKGROUPS * COMBINE_THREADS_PER_WG,
      "tile_occupancy_proxy_wg_per_cu": round(tile_wg / CU_COUNT, 2),
      "combine_occupancy_proxy_wg_per_cu": round(COMBINE_WORKGROUPS / CU_COUNT, 2),
      "combine_underoccupied": COMBINE_WORKGROUPS < CU_COUNT,
      "measured_wd_delta_pct": d_meas,
      "wd_spread_pct": spread,
      "route_fired": route_fired,
      "base_tok_ms": base_ms,
      "base_tok_ms_interpolated": ctx in BASE_TOK_MS_INTERPOLATED,
      "amdahl_projection": proj,
    })

  classification, rationale = _classify(per_ctx, wd)
  spreads = [c["wd_spread_pct"] for c in per_ctx if c["wd_spread_pct"] is not None]

  return {
    "schema": "split_kv_economics_audit_v1",
    "date": "2026-06-21",
    "candidate_id": candidate,
    "family": "north_star_flash_attn_tile",
    "primitive_class": "llama-style",  # local single-stream T=1 decode-attention primitive
    "role": "decode_attention",
    "comparator": "gqa_coop_vec",
    "contexts": ctxs,
    "machine": {"hardware": "RX 7900 XTX / gfx1100", "cu_count": CU_COUNT, "hbm_peak_gbps": HBM_PEAK_GBPS,
                "model": "Qwen3-8B-Q4_K_M", "n_layers": N_LAYERS},
    "gate": f"(ctx1024 >= +{GATE_CTX1024_PCT}% OR ctx4096 >= +{GATE_CTX4096_PCT}%) AND no ctx512/ctx1024 regression",
    "gate_rule": f"(ctx1024 >= +{GATE_CTX1024_PCT}% OR ctx4096 >= +{GATE_CTX4096_PCT}%) AND no ctx512/ctx1024 regression",
    "correctness": "inherited from the attribution artifact (per-row numpy GQA rmse; all rows correct); this audit derives economics, it does not re-measure correctness",
    "compile_handling": "n/a -- derived from measured GPU-busy attribution + measured routed W==D; this audit runs no live benchmark",
    "spread_pct": round(max(spreads), 3) if spreads else None,
    "attribution_source": "measured tile/combine GPU-busy attribution (standalone, wait=True median); NOT remeasured by this audit",
    "wd_source": "measured routed W==D delta per ctx (interleaved in-process) for the Amdahl anchor + operative S",
    "method": ("read measured tile/combine attribution + routed W==D; derive combine economics (bandwidth, occupancy), "
               "per-ctx optimal S, and an Amdahl projection of W==D for measured/half/free combine; classify"),
    "per_ctx": per_ctx,
    "classification": classification,
    "classification_rationale": rationale,
    "verdict": classification,
    "stop_reason": rationale,
  }


def _classify(per_ctx: list[dict], wd: dict | None) -> tuple[str, str]:
  """COMBINE_TAX_DOMINATES | COMBINE_SMALL_AMDAHL_LIMIT | POLICY_ONLY | MEASUREMENT_UNSTABLE.

  Order:
    1. MEASUREMENT_UNSTABLE  -- the W==D signal is inside its own noise band (can't trust the delta).
    2. POLICY_ONLY           -- some context already clears the gate as measured (a routing policy promotes it).
    3. COMBINE_TAX_DOMINATES -- a cheaper combine is the lever: HALVING the combine is projected to clear the
                                gate at a winning ctx (and the combine is latency-bound: under-occupied + far
                                below HBM peak).
    4. COMBINE_SMALL_AMDAHL_LIMIT -- even a FREE combine does not clear the gate -> the ceiling is the Amdahl
                                attention share, not the combine."""
  # 0) no measured W==D anchor at all -> the Amdahl projection cannot be computed; do not pretend a ceiling was tested.
  if not any(c["measured_wd_delta_pct"] is not None for c in per_ctx):
    return ("MEASUREMENT_UNSTABLE",
            "no measured whole-decode W==D anchor available (no routed-W==D artifact) -> the Amdahl projection of a "
            "cheaper combine cannot be computed; reopen a routed W==D authority run through BoltBeam before "
            "classifying the combine economics")

  fired = [c for c in per_ctx if c["route_fired"]]
  # 1) measurement stability: any fired ctx whose |delta| is not clearly above its spread band
  unstable = [c for c in fired if c["measured_wd_delta_pct"] is not None and c["wd_spread_pct"] is not None
              and abs(c["measured_wd_delta_pct"]) <= c["wd_spread_pct"]]
  if fired and len(unstable) == len(fired):
    return ("MEASUREMENT_UNSTABLE",
            "every routed W==D delta is within its own reproducibility band -> the signal cannot be trusted; "
            "tighten the harness (more repeats / clock control) before classifying the combine economics")

  # the Amdahl projection is an ESTIMATE (combine-reduction headroom), so the gate check uses a small tolerance.
  GATE_TOL = 0.25

  def bar_for(ctx: int) -> float:
    return GATE_CTX4096_PCT if ctx >= 2048 else GATE_CTX1024_PCT

  def clears(c: dict, key: str, tol: float = 0.0) -> bool:
    p = c["amdahl_projection"]
    v = p.get(key)
    if v is None: return False
    # only ctx that can legally promote count: 1024 (>=+5%) or >=2048 (>=+7%); ctx512 is gate-excluded
    if c["ctx"] < 1024: return False
    return v >= bar_for(c["ctx"]) - tol

  # 2) policy-only: measured (f=0) already clears at some legal ctx
  if any(clears(c, "measured_pct") for c in per_ctx):
    return ("POLICY_ONLY",
            "a context already clears the promotion gate as measured -> the lever is a routing policy (ctx-gated "
            "opt-in), not a cheaper combine")

  latency_bound = any((c["combine_pct_of_hbm_peak"] or 100) < 25.0 and c["combine_underoccupied"] for c in per_ctx)
  free_clears = [c for c in per_ctx if clears(c, "free_combine_pct")]
  half_clears = [c for c in per_ctx if clears(c, "half_combine_pct", tol=GATE_TOL)]

  # 3) combine tax dominates: a cheaper combine is the lever -- a FREE combine clears the gate at a winning ctx,
  #    the combine is latency-bound, and HALVING it already reaches (within estimate tolerance) the gate.
  if free_clears and latency_bound:
    min_gbps = min(c["combine_effective_gbps"] for c in per_ctx if c["combine_effective_gbps"])
    reach = (f"halving the combine already reaches the gate ({', '.join(f'ctx{c['ctx']}' for c in half_clears)})"
             if half_clears else "only a near-free (fully-fused) combine clears the gate; halving helps but is short")
    return ("COMBINE_TAX_DOMINATES",
            f"a cheaper/fused combine is the actionable lever: a free combine is projected to clear the gate "
            f"({', '.join(f'ctx{c['ctx']}' for c in free_clears)}) and {reach}. The combine is latency-bound "
            f"(~{min_gbps:.0f} GB/s << {HBM_PEAK_GBPS:.0f} GB/s peak; {COMBINE_WORKGROUPS} workgroups << {CU_COUNT} "
            f"CUs) so it is fixable with more parallelism / fusion. Amdahl co-limits the long-ctx ceiling.")

  # 4) free combine still cannot clear -> the Amdahl attention share is the ceiling
  return ("COMBINE_SMALL_AMDAHL_LIMIT",
          "even a FREE (fully-fused, zero-cost) combine is not projected to clear the gate -> the ceiling is the "
          "Amdahl attention share of the decode step, not the combine; a cheaper combine cannot promote this "
          "candidate (attack the FFN/GEMV share instead)")


def main():
  ap = argparse.ArgumentParser(description="Split-KV economics audit for decode-attention candidates")
  ap.add_argument("--attribution", default=str(ATTRIBUTION_DEFAULT), help="tile/combine attribution artifact")
  ap.add_argument("--wd", default=str(WD_DEFAULT), help="routed W==D artifact (for Amdahl anchor + operative S)")
  ap.add_argument("--candidate", default="decode_attention_llama_flash_tile_owned_amdgcn_b4")
  ap.add_argument("--out", default=str(OUT_DEFAULT))
  ap.add_argument("--live", action="store_true",
                  help="removed stale replay path; use committed attribution artifacts or reopen through BoltBeam")
  args = ap.parse_args()

  if args.live:
    raise RuntimeError("--live B4 replay was removed from the compact repo surface; use committed artifacts or reopen through BoltBeam")

  attr = _load(args.attribution)
  wd = None
  wd_path = pathlib.Path(args.wd)
  if wd_path.exists():
    wd = _load(wd_path)
  else:
    print(f"[warn] no W==D artifact at {wd_path}; Amdahl anchor + operative S fall back to min-total optimal S",
          file=sys.stderr)

  out = audit(attr, wd, args.candidate)

  # stamp the harness contract envelope (import-safe; no GPU)
  try:
    from extra.qk.harness_contract import stamp
    out = stamp(out, comparator_id="gqa_coop_vec",
                comparator_why=("shipped default decode-attention primitive; a split-KV candidate must beat it in "
                                "whole-decode W==D AFTER paying its combine tax, not only in a tile A/B"),
                timing_authority=("derived from measured GPU-busy tile/combine attribution + measured routed W==D; "
                                  "the Amdahl projection of half/free combine is an ESTIMATE, not a measured W==D"),
                ledger_links=["docs/split-kv-economics-audit-result-20260621.md",
                              "docs/b4-split-kv-combine-tax-result-20260621.md",
                              "bench/qk-decode-attention-route-b-b4-combine-tax/latest.json"])
  except Exception as e:  # pragma: no cover - stamping is best-effort
    print(f"[warn] could not stamp harness contract: {e}", file=sys.stderr)

  outdir = pathlib.Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
  (outdir / "latest.json").write_text(json.dumps(out, indent=2) + "\n")

  # human-readable summary
  print(f"candidate: {out['candidate_id']}  comparator: {out['comparator']}")
  print(f"{'ctx':>5} {'S':>4} {'tile_us':>8} {'comb_us':>8} {'comb%':>6} {'eff_GB/s':>9} {'comb_wg':>8} "
        f"{'wd%':>7} {'half%':>7} {'free%':>7}")
  for c in out["per_ctx"]:
    p = c["amdahl_projection"]
    print(f"{c['ctx']:>5} {c['operative_S']:>4} {c['tile_us']:>8.1f} {c['combine_us']:>8.1f} "
          f"{100*c['combine_fraction']:>5.0f}% {c['combine_effective_gbps'] or 0:>9.1f} {c['combine_workgroups']:>8} "
          f"{(c['measured_wd_delta_pct'] if c['measured_wd_delta_pct'] is not None else 0):>+6.2f}% "
          f"{(p.get('half_combine_pct') or 0):>+6.2f}% {(p.get('free_combine_pct') or 0):>+6.2f}%")
  print(f"\nclassification: {out['classification']}")
  print(f"rationale: {out['classification_rationale']}")
  print(f"artifact: {outdir / 'latest.json'}")


if __name__ == "__main__":
  main()
