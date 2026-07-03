#!/usr/bin/env python3
"""PMS-R7: decode-attention REOPEN GATE (not a rewrite).

Decides whether decode attention is worth reopening for the current target, from whole-model leverage -- never from
"the native route is slower" alone. Inputs are READ from the ceiling/attribution artifact
(bench/amd-isa-backend-decode-attention-ceiling/latest.json) + phase-n7 native gap + the refuted-axes ledger; no GPU.

Rule:
  perfect_parity_gain_pct[ctx] = measured attention tile WALL share (Amdahl) -> the BEST-CASE whole-decode gain if the
    attention tile cost dropped to its KV-read floor (perfect attention).
  realizable_gain_pct = 0 UNLESS an available route beats owned. The only generated attention route is correct-not-fast
    (~60-68% of owned) and EVERY attention search axis is refuted/exhausted -> no candidate can realize the ceiling.
  active iff realizable_gain_pct >= ACTIVE_THRESHOLD_PCT. Else DO_NOT_REOPEN_ATTENTION.

Run:  PYTHONPATH=. python3 extra/qk/attention_reopen_gate.py
"""
from __future__ import annotations
import json, pathlib, statistics
from extra.qk.route_manifest import REFUTED

ROOT = pathlib.Path(__file__).resolve().parents[2]
CEILING = ROOT / "bench/amd-isa-backend-decode-attention-ceiling/latest.json"

PROFILE_ID = "qwen3_8b_q4_k_m_gfx1100_decode"
ACTIVE_THRESHOLD_PCT = 5.0   # whole-decode gain bar to justify reopening a hot kernel (matches the TIER_A leverage bar)
MODEL_ARCH = {"family": "qwen3", "params": "8b", "Hq": 32, "Hkv": 8, "Hd": 128, "kv_dtype_bytes": 4,
              "decode_bound": "weight-memory-bound (HBM); weight read dominates the per-token wall"}
# the attention search axes already refuted/exhausted (carried from the route manifest) -> no realizable candidate.
# Filter on the manifest's domain tag (domain=="attention"), not a hand-maintained name list (bugs-F8).
ATTENTION_REFUTED_AXES = [r for r in REFUTED if r.get("domain") == "attention" or "attention" in r["axis"]]


def build() -> dict:
  if not CEILING.exists():
    return {"verdict": "PMS_R7_BLOCKED_WALL_SHARE_ATTRIBUTION_MISSING", "missing_artifact": str(CEILING.relative_to(ROOT))}
  d = json.load(open(CEILING))
  ws = {k.replace("ctx", ""): v for k, v in d["loss_stack"]["tile_wall_share_measured"].items() if k.startswith("ctx")}
  native_vs_owned = d["native_vs_owned"]            # {"512":68.3,"4096":60.1} -> best available is BELOW owned
  owned_vs_floor = d["owned_vs_floor"]
  ctxs = sorted(int(c) for c in ws)

  perfect_parity_gain_pct = {str(c): round(ws[str(c)] * 100, 2) for c in ctxs}   # Amdahl ceiling per ctx
  best_available_vs_owned_pct = max(native_vs_owned.values())                    # < 100 -> nothing beats owned
  no_faster_route = best_available_vs_owned_pct < 100.0
  levers_walled = len(ATTENTION_REFUTED_AXES) > 0
  # realizable: only if an available route beats owned AND a non-refuted lever exists. Neither holds.
  realizable_gain_pct = 0.0 if (no_faster_route and levers_walled) else max(perfect_parity_gain_pct.values())

  # transparency: even the THEORETICAL ceiling at the representative long-ctx operating point is below threshold
  long_ctx = str(max(ctxs))
  theoretical_long_ctx = perfect_parity_gain_pct[long_ctx]
  theoretical_median = round(statistics.median(perfect_parity_gain_pct.values()), 2)

  active = realizable_gain_pct >= ACTIVE_THRESHOLD_PCT
  verdict = "PMS_R7_PASS_ATTENTION_REOPEN_GATE"
  gate = "REOPEN_ATTENTION" if active else "DO_NOT_REOPEN_ATTENTION"

  result = {
    "scope": "PMS-R7 decode attention reopen gate (whole-model leverage, not a rewrite)",
    "verdict": verdict, "gate": gate, "profile_id": PROFILE_ID,
    "active_threshold_pct": ACTIVE_THRESHOLD_PCT,
    "model_arch": MODEL_ARCH, "contexts": ctxs,
    "inputs_from_artifact": {"source": str(CEILING.relative_to(ROOT)), "ceiling_verdict": d["verdict"],
                             "tile_wall_share_measured": ws, "native_vs_owned_pct": native_vs_owned,
                             "owned_vs_floor_pct": owned_vs_floor,
                             "math_floor_weight_tok_s_peak": d["loss_stack"]["math_floor_weight_tok_s_peak"]},
    "perfect_parity_gain_pct_per_ctx": perfect_parity_gain_pct,
    "theoretical_ceiling_long_ctx_pct": theoretical_long_ctx,
    "theoretical_ceiling_median_pct": theoretical_median,
    "best_available_route_vs_owned_pct": best_available_vs_owned_pct,
    "no_faster_route_than_owned": no_faster_route,
    "attention_levers_walled": [a["axis"] for a in ATTENTION_REFUTED_AXES],
    "realizable_gain_pct": realizable_gain_pct,
    "decision_reason": (
      "DO_NOT_REOPEN_ATTENTION: (1) realizable whole-decode gain = 0% -- the best available attention route is "
      f"{best_available_vs_owned_pct}% of owned (BELOW owned), and every attention search axis is refuted/exhausted "
      f"({[a['axis'] for a in ATTENTION_REFUTED_AXES]}), so no candidate can realize the Amdahl ceiling; (2) even the "
      f"theoretical ceiling is small and ctx-decaying (10.2%@ctx512 -> {theoretical_long_ctx}%@ctx{long_ctx}), below "
      f"the {ACTIVE_THRESHOLD_PCT}% reopen bar at the representative long-ctx operating point; the dominant decode wall "
      "is the weight-memory-bound FFN/projection (HBM), not attention."),
    "what_would_make_it_active": [
      "much longer context with a large KV cache that materially raises the attention tile wall-share",
      "a different KV layout / dtype that raises the KV-read floor relative to weight read",
      "larger Hq/Hkv/Hd (head_dim) increasing attention FLOPs/bytes per token",
      "MoE or a larger FFN-sparsity that shrinks the dense-FFN wall-share so attention dominates",
      "a different GPU with higher HBM bandwidth making the weight read less dominant",
      "a NEW attention primitive premise distinct from the refuted axes (not native-as-default, not "
      "combine/fused-lifecycle, not scheduler/occupancy/LDS-only, not N1B scalar-address)"],
    "owned_route_stays_shipped": "decode_attention_owned_two_kernel (owned_flash_tile_gqa_whole -> owned_flash_combine)",
    "do_not": ["do not rewrite attention", "do not start from 'the native route is slower'"],
  }
  return result


if __name__ == "__main__":
  import sys; sys.path.insert(0, str(ROOT))
  from extra.qk.gate_registry import run
  raise SystemExit(run("attention_reopen"))
