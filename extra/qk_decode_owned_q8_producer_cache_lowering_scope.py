#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_owned_q8_producer_cache_lowering_scope_result.json"


def load(rel: str, default: Any = None) -> Any:
  p = ROOT / rel
  return json.loads(p.read_text()) if p.exists() else default


def main() -> int:
  scope = load("bench/qk-decode-primitive-transfer/decode_owned_q8_producer_cache_scope_result.json", {})
  obj = load("bench/qk-decode-primitive-transfer/decode_owned_q8_producer_cache_object_result.json", {})
  ref = load("bench/qk-decode-primitive-transfer/decode_owned_q8_producer_cache_reference_result.json", {})
  hcq = load("bench/q8-ffn-handwritten-oracle/hcq_artifact.json", {})
  lifecycle = load("bench/q8-ffn-handwritten-oracle/gate_up_lifecycle.json", {})

  candidate = {
    "name": "owned_hcq_comgr_q8_rmsnorm_side",
    "source": "extra.q8_ffn_hcq_artifact.NORM_SOURCE",
    "runtime": "tinygrad AMD HCQ via Device.compiler COMGR path",
    "hip_runtime_in_process_allowed": False,
    "operation": "rmsnorm output plus block_q8_1 sidechannel",
    "launch": {"global_size": [1, 1, 1], "local_size": [256, 1, 1]},
    "layout": {"elements": 4096, "blocks": 128, "bytes": 4608, "block_bytes": 36},
    "target": {
      "producer_lifecycle_us_lte": (scope.get("contract", {}).get("targets", {}) or {}).get("producer_lifecycle_us_lte"),
      "incremental_us_lte": (scope.get("contract", {}).get("targets", {}) or {}).get("incremental_us_lte"),
      "reference_q8_max_abs_lte": 0.02,
      "reference_fp_max_abs_lte": 1e-5,
    },
  }
  gates = {
    "producer_scope_ready": scope.get("gate_pass") is True,
    "producer_object_ready": obj.get("gate_pass") is True,
    "producer_reference_ready": ref.get("gate_pass") is True,
    "existing_hcq_semantics_pass": hcq.get("verdict") == "PASS" and (hcq.get("producer") or {}).get("q8_bytes") == 4608,
    "lifecycle_target_available": lifecycle.get("verdict") == "PASS",
    "candidate_named": candidate["name"] == "owned_hcq_comgr_q8_rmsnorm_side",
  }
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_OWNED_Q8_PRODUCER_CACHE_LOWERING_SCOPE",
    "schema": "decode_owned_q8_producer_cache_lowering_scope_v1",
    "verdict": "PASS_DECODE_OWNED_Q8_PRODUCER_CACHE_LOWERING_SCOPE_READY" if all(gates.values()) else "BLOCKED_DECODE_OWNED_Q8_PRODUCER_CACHE_LOWERING_SCOPE",
    "gate_pass": all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "candidate": candidate,
    "next_executable_probe": "extra/qk_decode_owned_q8_producer_cache_lowering_candidate.py",
    "gates": gates,
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "gates": gates,
    "candidate": candidate["name"],
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
