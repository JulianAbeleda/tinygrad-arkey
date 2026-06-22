#!/usr/bin/env python3
"""B4 Phase-2 policy consolidation: from the measured per-ctx/per-S W==D adaptive sweep (extra/qk_b4_decode_eval.py
--policy adaptive --splits 48 64 --ckpts 512 1024 2048 4096), derive the four ctx-aware routing policies and evaluate
each against the promotion gate. Emits bench/qk-decode-attention-route-b-b4-combine-tax/policy_sweep.json.

Policies (route = the owned AMDGCN graph-node tile; off = the shipped gqa_coop_vec default, i.e. 0% delta):
  - off_below_4096 : route only ctx>=4096
  - off_below_2048 : route only ctx>=2048
  - adaptive_bestS : route ctx>=1024 with the per-ctx best S
  - no_regression  : route only where the measured routed delta clears the noise band (spread)

Gate: (d1024 >= +5% OR d4096 >= +7%) AND no ctx512 regression AND d1024 >= -1% AND tokens match.

  run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_b4_policy_sweep.py
"""
from __future__ import annotations
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SWEEP = ROOT / "bench/qk-decode-attention-route-b-b4/latest.json"   # the adaptive 4-ctx/S{48,64} W==D artifact
OUT = ROOT / "bench/qk-decode-attention-route-b-b4-combine-tax/policy_sweep.json"

def main():
  d = json.loads(SWEEP.read_text())
  # per-ctx best-S routed result (max amdgcn tok/s among split candidates)
  routed = {}
  for row in d["rows"]:
    ck = row["ctx"]
    best = max(row["split_candidates"], key=lambda s: (s.get("amd_tok_s_median") or -1))
    routed[ck] = {"best_S": best["split_S"], "delta_pct": best.get("delta_pct"),
                  "tokens_match": best.get("tokens_match"),
                  "spread_pct": max(best.get("base_spread_pct") or 0.0, best.get("amd_spread_pct") or 0.0),
                  "route_fired": best.get("route_fired_amdgcn")}
  ctxs = sorted(routed)

  def routed_delta(ck): return routed[ck]["delta_pct"] if ck in routed and routed[ck]["route_fired"] else 0.0
  def tokens_ok(cks): return all(routed[c]["tokens_match"] for c in cks if c in routed and routed[c]["route_fired"])

  def policy_deltas(active_ctxs):
    return {c: (routed_delta(c) if c in active_ctxs else 0.0) for c in ctxs}

  policies = {}
  # 1) off_below_4096
  policies["off_below_4096"] = policy_deltas({c for c in ctxs if c >= 4096})
  # 2) off_below_2048
  policies["off_below_2048"] = policy_deltas({c for c in ctxs if c >= 2048})
  # 3) adaptive_bestS (route ctx>=1024)
  policies["adaptive_bestS"] = policy_deltas({c for c in ctxs if c >= 1024})
  # 4) no_regression: route a ctx only if its routed delta clears its noise band
  nr = set()
  for c in ctxs:
    if routed[c]["route_fired"] and routed_delta(c) > max(0.5, routed[c]["spread_pct"]): nr.add(c)
  policies["no_regression"] = policy_deltas(nr)

  results = []
  for name, deltas in policies.items():
    d512, d1024, d2048, d4096 = deltas.get(512,0.0), deltas.get(1024,0.0), deltas.get(2048,0.0), deltas.get(4096,0.0)
    active = [c for c in ctxs if deltas.get(c, 0.0) != 0.0]
    gate = (tokens_ok(active) and d512 >= -1.0 and d1024 >= -1.0 and (d1024 >= 5.0 or d4096 >= 7.0))
    results.append({"policy": name, "active_ctxs": active, "deltas_pct": {str(c): round(deltas[c],2) for c in ctxs},
                    "best_S_by_active_ctx": {str(c): routed[c]["best_S"] for c in active},
                    "gate_pass": gate})
    print(f"{name:16}: " + " ".join(f"{c}:{deltas[c]:+.2f}%" for c in ctxs) + f"  -> gate {'PASS' if gate else 'FAIL'}")

  any_pass = any(r["gate_pass"] for r in results)
  verdict = "POLICY_PASS_OPT_IN" if any_pass else "NO_POLICY_CLEARS_GATE"
  out = {"date": "2026-06-21", "phase": "B4_SPLIT_KV_POLICY_SWEEP",
         "comparator": "gqa_coop_vec", "source_wd_artifact": str(SWEEP.relative_to(ROOT)),
         "gate_rule": "(d1024>=+5% OR d4096>=+7%) AND no ctx512 regression AND d1024>=-1% AND tokens match",
         "routed_per_ctx": {str(c): routed[c] for c in ctxs}, "policies": results,
         "any_policy_passes": any_pass, "verdict": verdict,
         "note": ("'off' ctxs use the shipped gqa_coop_vec default (0% delta by construction). Routed deltas are the "
                  "measured best-S W==D gain when the owned graph-node tile is active at that ctx.")}
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(out, indent=2))
  print(f"\nverdict: {verdict}\nartifact: {OUT}")

if __name__ == "__main__":
  main()
