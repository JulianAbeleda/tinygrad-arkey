#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_owned_q8_producer_hcq_parity_closeout_result.json"


def load(rel: str, default: Any = None) -> Any:
  p = ROOT / rel
  return json.loads(p.read_text()) if p.exists() else default


def main() -> int:
  reconcile = load("bench/qk-decode-primitive-transfer/decode_owned_q8_producer_target_reconcile_result.json", {})
  candidate = load("bench/qk-decode-primitive-transfer/decode_owned_q8_producer_cache_lowering_candidate_result.json", {})
  ref = load("bench/qk-decode-primitive-transfer/decode_owned_q8_producer_cache_reference_result.json", {})
  parity = load("bench/qk-decode-primitive-transfer/decode_owned_q8_artifact_parity_harness_result.json", {})

  rows = {r["row"]: r for r in reconcile.get("rows", [])}
  owned_us = rows.get("owned_comgr_hcq_candidate", {}).get("producer_us")
  artifact_us = rows.get("hipcc_lld_hcq_artifact_loader", {}).get("producer_us")
  hip_us = rows.get("hip_runtime_modeled_oracle", {}).get("producer_us")
  speedup_vs_hcq_artifact = artifact_us / owned_us if owned_us and artifact_us else None
  slowdown_vs_hip_oracle = owned_us / hip_us if owned_us and hip_us else None

  gates = {
    "target_reconciled": reconcile.get("gate_pass") is True,
    "candidate_correct": candidate.get("gates", {}).get("fp_correct") is True and candidate.get("gates", {}).get("q8_dequant_bounded") is True,
    "reference_semantics_ready": ref.get("gate_pass") is True,
    "parity_harness_ready": parity.get("gate_pass") is True,
    "owned_beats_hcq_artifact": speedup_vs_hcq_artifact is not None and speedup_vs_hcq_artifact > 1.0,
    "owned_not_hip_oracle": slowdown_vs_hip_oracle is not None and slowdown_vs_hip_oracle > 1.0,
  }
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_OWNED_Q8_PRODUCER_HCQ_PARITY_CLOSEOUT",
    "schema": "decode_owned_q8_producer_hcq_parity_closeout_v1",
    "verdict": "PASS_DECODE_OWNED_Q8_PRODUCER_HCQ_PARITY_ROW" if all(gates.values()) else "BLOCKED_DECODE_OWNED_Q8_PRODUCER_HCQ_PARITY_CLOSEOUT",
    "gate_pass": all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": True,
    "owned_producer_row": {
      "name": "owned_hcq_comgr_q8_rmsnorm_side",
      "runtime": "tinygrad AMD HCQ / COMGR",
      "producer_us": owned_us,
      "correctness": candidate.get("correctness", {}),
      "reference_semantics": "block_q8_1-compatible q8 producer/cache",
      "reuse_count": 2,
      "status": "HCQ_PARITY_CANDIDATE",
    },
    "comparison": {
      "hcq_artifact_producer_us": artifact_us,
      "hip_oracle_producer_us": hip_us,
      "speedup_vs_hcq_artifact": speedup_vs_hcq_artifact,
      "slowdown_vs_hip_oracle": slowdown_vs_hip_oracle,
    },
    "decision": {
      "use_for_owned_successor": True,
      "remaining_gap": "HIP-runtime producer oracle delta remains, but it is not the HCQ artifact parity gate",
      "next": "scope HIP-oracle producer delta separately; do not block route-level HCQ parity on it",
    },
    "gates": gates,
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "owned_producer_row": result["owned_producer_row"],
    "comparison": result["comparison"],
    "gates": gates,
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
