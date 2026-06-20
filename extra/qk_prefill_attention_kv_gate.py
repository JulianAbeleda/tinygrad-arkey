#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "bench/qk-inmodel-integration-penalty/inmodel_integration_penalty_audit_result.json"
OUT = ROOT / "bench/qk-inmodel-integration-penalty/prefill_attention_kv_gate_result.json"


def required_component_speedup(share: float, target_full_speedup: float) -> float:
  denom = (1.0 / target_full_speedup) - (1.0 - share)
  return float("inf") if denom <= 0 else share / denom


def full_speedup(share: float, component_speedup: float) -> float:
  return 1.0 / ((1.0 - share) + share / component_speedup)


def main() -> int:
  audit = json.loads(AUDIT.read_text())
  attention_share = float(audit["amdahl"]["attention_share_of_span"])
  required_for_1p10 = required_component_speedup(attention_share, 1.10)
  required_for_1p15 = required_component_speedup(attention_share, 1.15)
  at_1p25 = full_speedup(attention_share, 1.25)
  at_2p00 = full_speedup(attention_share, 2.00)
  gates = {
    "audit_passed": audit.get("gate_pass") is True,
    "attention_bucket_measured": attention_share > 0.0,
    "attention_1p25_material": at_1p25 >= 1.10,
    "attention_2p00_reaches_1p15": at_2p00 >= 1.15,
  }
  if not gates["audit_passed"]:
    verdict = "BLOCKED_PREFILL_ATTENTION_KV_AUDIT_MISSING"
  elif not gates["attention_1p25_material"]:
    verdict = "PARK_PREFILL_ATTENTION_KV_LOW_AMDAHL_FOR_PP512"
  elif gates["attention_2p00_reaches_1p15"]:
    verdict = "PASS_PREFILL_ATTENTION_KV_MATERIAL_ONLY_IF_BIG_REWRITE"
  else:
    verdict = "BLOCKED_PREFILL_ATTENTION_KV_NO_MATERIAL_ROUTE"

  result = {
    "date": "2026-06-20",
    "phase": "PREFILL_ATTENTION_KV_GATE",
    "schema": "prefill_attention_kv_gate_v1",
    "verdict": verdict,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": False,
    "inputs": {"attention_share": attention_share},
    "thresholds": {
      "full_speedup_if_attention_1p25x": round(at_1p25, 4),
      "full_speedup_if_attention_2p00x": round(at_2p00, 4),
      "required_attention_speedup_for_1p10_full": round(required_for_1p10, 4),
      "required_attention_speedup_for_1p15_full": round(required_for_1p15, 4),
    },
    "gates": gates,
    "decision": {
      "next": "park pp512 attention/KV unless long-context changes the share or a >=2x bounded route appears",
      "why": "a 1.25x attention/KV win projects only about 5% full-prefill speedup",
    },
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"verdict": verdict, "thresholds": result["thresholds"], "gates": gates, "out": str(OUT.relative_to(ROOT))}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
