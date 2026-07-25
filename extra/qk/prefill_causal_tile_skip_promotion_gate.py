#!/usr/bin/env python3
"""Authority gate for flipping PREFILL_CAUSAL_TILE_SKIP's DEFAULT from off to on.

This is NOT a new-route promotion. `PREFILL_CAUSAL_TILE_SKIP` (commit c44905a18) only changes the KV-loop's runtime
extent inside the ALREADY-promoted `prefill_flash_attention_generated` row (tinygrad/schedule/wmma/kernels.py
amd_gfx1100_q16_grid_hd128_loop_attention): full_kv_tiles vs a dynamic min(full_kv_tiles, ceildiv(query_start +
q_tile*16 + 16, 16)) bound. Route id, emitter, route_attribution chain, and provenance (machine_authored_generated)
are unchanged either way -- so this gate does not touch extra/qk/route_manifest.py's ROUTES table structurally, it
only gatekeeps a default-value change inside one existing row's flag.

The bar (docs/prefill-current-state.md "Change rule"): exact typed candidate, isolated correctness, whole-model
parity, route-census identity, pinned timing -- and per BoltBeam's authority_completeness_gate pattern
(extra/qk/prefill_whole_synced.py), a promotion claim is refused if any required field is missing rather than
silently treated as satisfied.

Critically: prefill_flash_attention_generated's own shape_guards admit TWO grids (8B Hq=32/Hkv=8 and 14B Hq=40/
Hkv=8, both Hd=128). This gate requires evidence for EVERY admitted shape_guard, not just the one that happens to
have been measured. It reads ONLY the pre-collected evidence in
docs/prefill-causal-tile-skip-evidence-20260724.json -- it never invents a number, and a shape with no evidence
recorded (or an evidence file that is missing/malformed) makes the gate FAIL closed, never PASS-by-omission.

This script does not flip any default. It reports PASS/FAIL for informing a human's decision only.

Run: PYTHONPATH=. python3 extra/qk/prefill_causal_tile_skip_promotion_gate.py
"""
from __future__ import annotations

import json
import pathlib
import sys

from extra.qk.route_manifest import ROUTES

ROUTE_ID = "prefill_flash_attention_generated"
FLAG = "PREFILL_CAUSAL_TILE_SKIP"
EVIDENCE_PATH = pathlib.Path(__file__).resolve().parents[2] / "docs" / "prefill-causal-tile-skip-evidence-20260724.json"

# Thresholds derived directly from the recorded evidence's own methodology section (docs/prefill-needle-theories-
# 20260724.md "Measurement methodology"): a claim only counts as signal if it clears the measured same-config
# back-to-back noise floor by a real margin, and needs >=3 independent same-session pairs to be credible at all.
MIN_PAIRS = 3
MIN_MEAN_DELTA_PCT = 1.0          # conservative: well under the observed 1.66-1.77% means
MIN_SIGNAL_TO_NOISE = 2.0         # observed ~3x; require at least 2x
MAX_NOISE_FLOOR_PCT = 1.0         # observed 0.59%; sanity ceiling so a noisy rerun can't be waved through


def _required_shape_keys() -> list[str]:
  """Every grid this route's manifest row claims eligibility for -- derived from the manifest, not hardcoded."""
  guards = ROUTES[ROUTE_ID]["shape_guards"]
  keys = []
  for g in guards:
    hq = g.get("Hq")
    if hq == 32: keys.append("8B")
    elif hq == 40: keys.append("14B")
    else: keys.append(f"Hq={hq}")
  return keys


def _load_evidence() -> dict | None:
  if not EVIDENCE_PATH.is_file(): return None
  try:
    data = json.loads(EVIDENCE_PATH.read_text())
  except (json.JSONDecodeError, OSError):
    return None
  if data.get("_schema") != "prefill-causal-tile-skip-promotion-evidence.v1": return None
  if data.get("route_id") != ROUTE_ID or data.get("flag") != FLAG: return None
  return data


def _check_shape(name: str, entry: dict | None) -> list[str]:
  """Return a list of failure strings for one required shape; empty means that shape is fully covered."""
  if entry is None:
    return [f"shape {name}: no evidence entry at all"]
  fails = []
  numerics = entry.get("numerics")
  if not numerics or numerics.get("status") != "PASS":
    fails.append(f"shape {name}: numerics missing or not PASS ({numerics!r})")
  parity = entry.get("token_parity")
  if not parity or parity.get("status") != "PASS":
    fails.append(f"shape {name}: token_parity missing or not PASS ({parity!r})")
  ab = entry.get("whole_model_paired_ab")
  if not ab or ab.get("status") != "PASS":
    fails.append(f"shape {name}: whole_model_paired_ab missing or not PASS ({ab!r})")
  else:
    pairs = ab.get("pairs") or []
    if len(pairs) < MIN_PAIRS:
      fails.append(f"shape {name}: only {len(pairs)} paired A/B runs recorded, need >= {MIN_PAIRS}")
    noise = ab.get("back_to_back_same_config_noise_pct")
    if noise is None or noise > MAX_NOISE_FLOOR_PCT:
      fails.append(f"shape {name}: same-config noise floor missing or too high ({noise!r} > {MAX_NOISE_FLOOR_PCT})")
    for metric in ("pp512_mean_delta_pct", "pp4096_mean_delta_pct"):
      val = ab.get(metric)
      if val is None or val < MIN_MEAN_DELTA_PCT:
        fails.append(f"shape {name}: {metric}={val!r} below required minimum {MIN_MEAN_DELTA_PCT}")
      if val is not None and noise not in (None, 0) and (val / noise) < MIN_SIGNAL_TO_NOISE:
        fails.append(f"shape {name}: {metric} signal/noise {val / noise:.2f}x below required {MIN_SIGNAL_TO_NOISE}x")
  pmc = entry.get("attention_local_pmc")
  if not pmc or pmc.get("status") != "PASS":
    fails.append(f"shape {name}: attention_local_pmc missing or not PASS ({pmc!r})")
  return fails


def evaluate() -> dict:
  route_row = ROUTES.get(ROUTE_ID)
  if route_row is None:
    return {"gate": "prefill_causal_tile_skip_promotion", "verdict": "FAIL",
            "reason": f"manifest has no route {ROUTE_ID!r}"}
  if route_row.get("provenance") != "machine_authored_generated":
    return {"gate": "prefill_causal_tile_skip_promotion", "verdict": "FAIL",
            "reason": f"{ROUTE_ID} provenance={route_row.get('provenance')!r}, expected machine_authored_generated "
                      f"(a dynamic loop bound derived from descriptor fields must stay in the allowed-default set)"}

  required = _required_shape_keys()
  evidence = _load_evidence()
  if evidence is None:
    return {"gate": "prefill_causal_tile_skip_promotion", "verdict": "FAIL",
            "reason": f"evidence artifact missing or invalid: {EVIDENCE_PATH}", "required_shapes": required}

  shapes = evidence.get("shapes", {})
  failures: dict[str, list[str]] = {}
  for name in required:
    fails = _check_shape(name, shapes.get(name))
    if fails: failures[name] = fails

  verdict = "PASS" if not failures else "FAIL"
  return {
    "gate": "prefill_causal_tile_skip_promotion",
    "route_id": ROUTE_ID,
    "flag": FLAG,
    "provenance": route_row.get("provenance"),
    "required_shapes": required,
    "evidence_path": str(EVIDENCE_PATH),
    "failures": failures,
    "verdict": verdict,
    "note": ("all required shapes have complete, passing, sufficiently-powered evidence" if verdict == "PASS" else
             "at least one manifest-admitted shape lacks complete promotion evidence; do NOT flip the default"),
  }


def main() -> int:
  report = evaluate()
  print(json.dumps(report, indent=2))
  print(f"AUTHORITY_GATE: {report['verdict']}")
  return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
  sys.exit(main())
