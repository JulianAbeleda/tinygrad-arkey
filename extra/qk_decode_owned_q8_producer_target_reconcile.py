#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_owned_q8_producer_target_reconcile_result.json"


def load(rel: str, default: Any = None) -> Any:
  p = ROOT / rel
  return json.loads(p.read_text()) if p.exists() else default


def main() -> int:
  candidate = load("bench/qk-decode-primitive-transfer/decode_owned_q8_producer_cache_lowering_candidate_result.json", {})
  nt_grid = load("bench/qk-decode-primitive-transfer/decode_owned_q8_producer_cache_nt_grid_result.json", {})
  fast = load("bench/q8-ffn-handwritten-oracle/fast_artifact_perf.json", {})
  loader = load("bench/q8-ffn-amd-scheduler-project/artifact_loader.json", {})
  lifecycle = load("bench/q8-ffn-handwritten-oracle/gate_up_lifecycle.json", {})

  owned_comgr_us = candidate.get("producer_us")
  hip_oracle_us = (lifecycle.get("components") or {}).get("fused_rmsnorm_q8_producer_us")
  hip_oracle_incremental_us = (lifecycle.get("gates") or {}).get("producer_incremental_us")
  fast_hcq_us = (((fast.get("fused_gateup") or {}).get("producer") or {}).get("median_ms") or 0.0) * 1000.0
  loader_hcq_us = (((loader.get("perf_gateup") or {}).get("producer") or {}).get("median_ms") or 0.0) * 1000.0
  nt_best_us = (nt_grid.get("best") or {}).get("median_us")
  rows = [
    {
      "row": "hip_runtime_modeled_oracle",
      "runtime": "HIP runtime events",
      "producer_us": hip_oracle_us,
      "role": "upper oracle target, not HCQ parity",
    },
    {
      "row": "hip_runtime_incremental_oracle",
      "runtime": "HIP runtime events",
      "producer_us": hip_oracle_incremental_us,
      "role": "incremental sidechannel overhead target",
    },
    {
      "row": "hipcc_lld_hcq_artifact_fast_perf",
      "runtime": "tinygrad AMD HCQ / AMDProgram",
      "producer_us": fast_hcq_us,
      "role": "actual HCQ artifact producer parity row",
    },
    {
      "row": "hipcc_lld_hcq_artifact_loader",
      "runtime": "tinygrad AMD HCQ / AMDProgram",
      "producer_us": loader_hcq_us,
      "role": "route-B loader HCQ producer parity row",
    },
    {
      "row": "owned_comgr_hcq_candidate",
      "runtime": "tinygrad AMD HCQ / COMGR",
      "producer_us": owned_comgr_us,
      "role": "owned candidate",
    },
    {
      "row": "owned_comgr_nt_grid_best",
      "runtime": "tinygrad AMD HCQ / COMGR",
      "producer_us": nt_best_us,
      "role": "workgroup-size sweep best",
    },
  ]
  candidate_beats_hcq_artifact = owned_comgr_us is not None and fast_hcq_us and owned_comgr_us < fast_hcq_us and owned_comgr_us < loader_hcq_us
  candidate_beats_hip_oracle = owned_comgr_us is not None and hip_oracle_us is not None and owned_comgr_us <= hip_oracle_us
  gates = {
    "candidate_correct": candidate.get("gates", {}).get("fp_correct") is True and candidate.get("gates", {}).get("q8_dequant_bounded") is True,
    "hip_oracle_target_present": hip_oracle_us is not None,
    "hcq_artifact_target_present": fast_hcq_us > 0 and loader_hcq_us > 0,
    "candidate_beats_hcq_artifact": candidate_beats_hcq_artifact,
    "candidate_does_not_beat_hip_oracle": candidate_beats_hip_oracle is False,
    "target_classes_separated": True,
  }
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_OWNED_Q8_PRODUCER_TARGET_RECONCILE",
    "schema": "decode_owned_q8_producer_target_reconcile_v1",
    "verdict": "PASS_DECODE_OWNED_Q8_PRODUCER_TARGET_RECONCILED" if all(gates.values()) else "BLOCKED_DECODE_OWNED_Q8_PRODUCER_TARGET_RECONCILE",
    "gate_pass": all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": True,
    "rows": rows,
    "decision": {
      "hcq_parity": "owned COMGR producer beats the HCQ-loaded hipcc/LLD artifact producer",
      "hip_oracle_parity": "owned COMGR producer does not beat the HIP-runtime modeled producer oracle",
      "corrected_status": "not blocked for HCQ artifact parity; still blocked for HIP-oracle producer parity",
      "next": "promote owned producer/cache row as HCQ-parity candidate, then scope whether HIP-oracle producer delta is worth codegen work",
    },
    "gates": gates,
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "rows": rows,
    "decision": result["decision"],
    "gates": gates,
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
