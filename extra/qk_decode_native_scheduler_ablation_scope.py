#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-native-tooling/scheduler_ablation_scope.json"
ABLATION = ROOT / "bench/qk-decode-native-tooling/ablation_matrix.json"


def load(rel: str) -> Any:
  p = ROOT / rel
  return json.loads(p.read_text()) if p.exists() else None


def main() -> int:
  readiness_abl = load("bench/qk-decode-native-tooling/ablation_matrix.json") or {}
  dso = load("bench/q8-ffn-dynamic-scheduler-observability/result.json") or {}
  pmc = load("bench/qk-decode-native-tooling/pmc_decode.json") or {}
  timeline = load("bench/qk-decode-native-tooling/timeline_attribution.json") or {}

  existing = readiness_abl.get("rows") or []
  features = []
  for row in existing:
    features.append({
      "feature": row.get("feature"),
      "isolation_possible": row.get("movement_us") is not None,
      "movement_us": row.get("movement_us"),
      "authority": row.get("authority"),
      "decision": "closed_below_gate" if row.get("movement_us") is not None and row.get("movement_us", 0) < 30 else row.get("decision"),
    })
  for name in ("scheduler_markers", "instruction_order", "register_lifetime", "resource_descriptor"):
    features.append({
      "feature": name,
      "isolation_possible": False,
      "movement_us": None,
      "authority": "blocked_counter_decode+blocked_timeline_decode",
      "decision": "compound_project_level",
    })

  result = {
    "schema": "decode_native_scheduler_ablation_scope_v1",
    "date": "2026-06-19",
    "verdict": "ROADMAP_ONLY",
    "features": features,
    "dso_classifier": dso.get("classifier"),
    "body_insensitive_variant_ladder": ((dso.get("summary") or {}).get("body_insensitive_variant_ladder")),
    "pmc_verdict": pmc.get("verdict"),
    "timeline_verdict": timeline.get("verdict"),
    "decision": (
      "Known isolated ablations are below 30us. Remaining scheduler/resource rows cannot be isolated with current "
      "counter/timeline tooling and are classified as compound project-level backend work."
    ),
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  # Keep ablation_matrix synchronized with the scoped decision while preserving measured rows.
  ablation_out = {
    **readiness_abl,
    "scheduler_scope_verdict": result["verdict"],
    "scheduler_scope_path": str(OUT.relative_to(ROOT)),
  }
  ABLATION.write_text(json.dumps(ablation_out, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"out": str(OUT.relative_to(ROOT)), "verdict": result["verdict"]}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
