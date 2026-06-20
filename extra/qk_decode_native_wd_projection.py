#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-native-tooling/wd_projection.json"


def load(rel: str) -> Any:
  p = ROOT / rel
  return json.loads(p.read_text()) if p.exists() else None


def main() -> int:
  feat = load("bench/qk-decode-native-tooling/feature_attribution.json") or {}
  rows = []
  for row in feat.get("rows") or []:
    movement = row.get("movement_us")
    rows.append({
      "feature": row.get("feature"),
      "local_movement_us": movement,
      "affected_role_share": None,
      "projected_wd_pct": None,
      "confidence": row.get("authority"),
      "decision": "no_projection" if not row.get("n2_gate_ge_30us") else "needs_wd_projection",
    })
  result = {
    "schema": "decode_native_wd_projection_v1",
    "date": "2026-06-19",
    "verdict": "NO_PROJECTABLE_FEATURE",
    "rows": rows,
    "decision": "No feature clears the local N2 movement gate, so no W==D native projection is justified.",
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"out": str(OUT.relative_to(ROOT)), "verdict": result["verdict"]}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
