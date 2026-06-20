#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_owned_q8_lifecycle_attribution_scope_result.json"


def main() -> int:
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_OWNED_Q8_LIFECYCLE_ATTRIBUTION_SCOPE",
    "schema": "decode_owned_q8_lifecycle_attribution_scope_v1",
    "verdict": "PASS_DECODE_OWNED_Q8_LIFECYCLE_ATTRIBUTION_SCOPE_READY",
    "gate_pass": True,
    "default_behavior_changed": False,
    "performance_claim": False,
    "problem": {
      "standalone_owned_producer_us": 15.70,
      "projected_artifact_consumer_us": 93.54,
      "projected_mixed_lifecycle_us": 109.24,
      "measured_mixed_producer_us": 30.44,
      "measured_mixed_consumer_us": 101.66,
      "measured_mixed_lifecycle_us": 132.10,
      "target_lifecycle_us": 115.24,
    },
    "questions": [
      "Does loading the hipcc/LLD gateup artifact perturb the owned COMGR producer timing in the same process?",
      "Does the gateup consumer timing depend on whether the producer was just run in the same buffer lifecycle?",
      "Is the mixed slowdown explained by producer inflation, consumer inflation, or both?",
      "Can the first causal audit proceed without ATT/PC timeline data?",
    ],
    "next_executable_probe": "extra/qk_decode_owned_q8_lifecycle_attribution_probe.py",
    "gates": {
      "same_process_timing_ladder_defined": True,
      "standalone_vs_loaded_producer_defined": True,
      "consumer_after_owned_producer_defined": True,
      "correctness_required_for_all_timed_rows": True,
      "no_route_or_default_change": True,
    },
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"verdict": result["verdict"], "out": str(OUT.relative_to(ROOT))}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
