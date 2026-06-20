#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_owned_q8_mixed_lifecycle_scope_result.json"


def load(rel: str, default: Any = None) -> Any:
  p = ROOT / rel
  return json.loads(p.read_text()) if p.exists() else default


def main() -> int:
  producer = load("bench/qk-decode-primitive-transfer/decode_owned_q8_producer_hcq_parity_closeout_result.json", {})
  parity = load("bench/qk-decode-primitive-transfer/decode_owned_q8_artifact_parity_harness_result.json", {})
  artifact = load("bench/q8-ffn-amd-scheduler-project/artifact_loader.json", {})
  successor = load("bench/qk-decode-primitive-transfer/decode_owned_q8_lifecycle_successor_object_result.json", {})

  producer_us = (producer.get("owned_producer_row") or {}).get("producer_us")
  artifact_perf = artifact.get("perf_gateup") or {}
  artifact_consumer_us = ((artifact_perf.get("gateup_consumer") or {}).get("median_ms") or 0.0) * 1000.0
  artifact_lifecycle_us = artifact_perf.get("gate_up_lifecycle_us")
  mixed_target_us = ((successor.get("object") or {}).get("parity") or {}).get("artifact_lifecycle_us")
  projected_mixed_us = producer_us + artifact_consumer_us if producer_us and artifact_consumer_us else None

  candidate = {
    "name": "owned_producer_plus_hcq_artifact_gateup_consumer",
    "producer": "owned_hcq_comgr_q8_rmsnorm_side",
    "consumer": "hipcc_lld_hcq_q8_mmvq_gateup",
    "ownership": "mixed: owned producer, external artifact consumer",
    "purpose": "measure whether owned producer row improves the artifact lifecycle before consumer ownership resumes",
    "expected_lifecycle_us": projected_mixed_us,
    "artifact_lifecycle_us": artifact_lifecycle_us,
    "target_lifecycle_us": mixed_target_us,
  }

  gates = {
    "producer_hcq_parity_passed": producer.get("gate_pass") is True,
    "parity_harness_ready": parity.get("gate_pass") is True,
    "artifact_loader_ready": artifact.get("verdict") == "PASS",
    "successor_object_ready": successor.get("gate_pass") is True,
    "candidate_projected_lte_artifact": projected_mixed_us is not None and mixed_target_us is not None and projected_mixed_us <= mixed_target_us,
    "mixed_ownership_explicit": candidate["ownership"].startswith("mixed:"),
  }
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_OWNED_Q8_MIXED_LIFECYCLE_SCOPE",
    "schema": "decode_owned_q8_mixed_lifecycle_scope_v1",
    "verdict": "PASS_DECODE_OWNED_Q8_MIXED_LIFECYCLE_SCOPE_READY" if all(gates.values()) else "BLOCKED_DECODE_OWNED_Q8_MIXED_LIFECYCLE_SCOPE",
    "gate_pass": all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "candidate": candidate,
    "next_executable_probe": "extra/qk_decode_owned_q8_mixed_lifecycle_probe.py",
    "boundaries": [
      "this is not fully owned because the gate/up consumer remains the hipcc/LLD artifact",
      "passing this row does not default-on q8",
      "consumer ownership remains parked until ATT or a lowerable consumer objective exists",
    ],
    "gates": gates,
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "candidate": candidate,
    "gates": gates,
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
