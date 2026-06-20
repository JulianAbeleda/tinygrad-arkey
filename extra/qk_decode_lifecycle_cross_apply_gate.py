#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIFECYCLE = ROOT / "bench/qk-decode-primitive-transfer/decode_owned_q8_lifecycle_attribution_result.json"
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_lifecycle_cross_apply_gate_result.json"


def main() -> int:
  lifecycle = json.loads(LIFECYCLE.read_text())
  interp = lifecycle["interpretation"]
  rows = lifecycle["rows"]
  mixed = float(interp["mixed_perf_gateup_lifecycle_us"])
  target = float(interp["target_lifecycle_us"])
  producer = float(rows["perf_gateup_full_mixed"]["producer"]["median_ms"]) * 1000.0
  consumer = float(rows["perf_gateup_full_mixed"]["gateup_consumer"]["median_ms"]) * 1000.0
  gap = mixed - target
  gates = {
    "lifecycle_correct": all(lifecycle.get("gates", {}).values()),
    "lifecycle_beats_target": mixed <= target,
    "consumer_near_expected": consumer <= 95.0,
    "producer_is_current_blocker": producer >= 25.0,
  }
  if not gates["lifecycle_correct"]:
    verdict = "BLOCKED_DECODE_CROSS_APPLY_INCORRECT"
  elif gates["lifecycle_beats_target"]:
    verdict = "PASS_DECODE_Q8_LIFECYCLE_PROMOTABLE"
  elif gates["producer_is_current_blocker"]:
    verdict = "BLOCKED_DECODE_Q8_PROMOTION_ON_PRODUCER_CONTEXT"
  else:
    verdict = "BLOCKED_DECODE_Q8_PROMOTION_LIFECYCLE_NOT_MATERIAL"

  result = {
    "date": "2026-06-20",
    "phase": "DECODE_LIFECYCLE_CROSS_APPLY_GATE",
    "schema": "decode_lifecycle_cross_apply_gate_v1",
    "verdict": verdict,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": False,
    "inputs": {
      "mixed_lifecycle_us": mixed,
      "target_lifecycle_us": target,
      "gap_us": round(gap, 3),
      "producer_us": round(producer, 3),
      "consumer_us": round(consumer, 3),
    },
    "gates": gates,
    "decision": {
      "next": "producer-only batch/context isolate, then lifecycle rerun; do not promote from isolated producer speed",
      "why": "the prefill audit rule applies directly: component speed must compose into the lifecycle",
    },
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"verdict": verdict, "inputs": result["inputs"], "gates": gates, "out": str(OUT.relative_to(ROOT))}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
