#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_owned_q8_producer_cache_scope_result.json"


def load(rel: str, default: Any = None) -> Any:
  p = ROOT / rel
  return json.loads(p.read_text()) if p.exists() else default


def main() -> int:
  first = load("bench/qk-decode-primitive-transfer/decode_owned_q8_first_build_scope_result.json", {})
  parity = load("bench/qk-decode-primitive-transfer/decode_owned_q8_artifact_parity_harness_result.json", {})
  successor = load("bench/qk-decode-primitive-transfer/decode_owned_q8_lifecycle_successor_object_result.json", {})
  lifecycle = load("bench/q8-ffn-handwritten-oracle/gate_up_lifecycle.json", {})
  promotion = load("bench/q8-ffn-artifact-promotion/promotion_result.json", {})

  components = lifecycle.get("components", {})
  life_gates = lifecycle.get("gates", {})
  obj = successor.get("object", {})
  producer = obj.get("producer", {})
  policy = obj.get("policy", {})
  quality = (promotion.get("summary") or {}).get("quality", {})

  contract = {
    "name": "OwnedQ8ProducerCache",
    "status": "scope_only_unwired",
    "source_activation": producer.get("activation_source", "post_norm_decode_activation"),
    "q8_format": producer.get("q8_format", "block_q8_1_or_artifact_compatible_q8"),
    "shape": {"elements": 4096, "blocks": 128, "bytes": 128 * 36, "block_bytes": 36},
    "producer_kernel_contract": {
      "operation": "rmsnorm output plus q8 sidechannel",
      "workgroup": [256, 1, 1],
      "global": [1, 1, 1],
      "eps": 1.0e-6,
      "writes_fp_out": True,
      "writes_q8_cache": True,
      "q8_scale": "max(abs(vals[32])) / 127, fp16 d, fp16 s=0",
      "q8_values": "round-nearest int8 clipped to [-128, 127]",
    },
    "cache_contract": {
      "reuse_count": producer.get("reuse_count"),
      "consumers": ["ffn_gate", "ffn_up"],
      "lifetime": producer.get("lifetime"),
      "fallback": policy.get("fallback"),
      "default_on": policy.get("default_on"),
    },
    "targets": {
      "producer_lifecycle_us_lte": components.get("fused_rmsnorm_q8_producer_us"),
      "incremental_us_lte": life_gates.get("producer_incremental_us_required_lte"),
      "measured_incremental_us": life_gates.get("producer_incremental_us"),
      "quality_max_dnll_lte": quality.get("threshold"),
      "artifact_quality_max_dnll": quality.get("max_dnll"),
    },
  }

  phases = [
    {
      "id": "OPC-1-structural-object",
      "purpose": "instantiate producer/cache object from existing q8 artifact evidence",
      "exit_gate": "shape, format, reuse, fallback, and targets pass structural gate",
      "do_now": True,
      "probe": "extra/qk_decode_owned_q8_producer_cache_object_probe.py",
    },
    {
      "id": "OPC-2-byte-semantics-reference",
      "purpose": "freeze CPU/reference q8 bytes and dequant semantics for the producer cache",
      "exit_gate": "q8 bytes are block_q8_1-compatible and reference dequant error is bounded",
      "do_now": True,
      "probe": "extra/qk_decode_owned_q8_producer_cache_reference_probe.py",
    },
    {
      "id": "OPC-3-owned-lowering-candidate",
      "purpose": "build a tinygrad-owned producer/cache implementation candidate",
      "exit_gate": "matches reference bytes/semantics and lifecycle target before W==D",
      "do_now": False,
      "blocked_on": "implementation work, not scope",
    },
  ]

  gates = {
    "first_build_scope_ready": first.get("gate_pass") is True,
    "parity_harness_ready": parity.get("gate_pass") is True,
    "successor_object_ready": successor.get("gate_pass") is True,
    "lifecycle_evidence_passed": lifecycle.get("verdict") == "PASS",
    "producer_reuse_count_two": contract["cache_contract"]["reuse_count"] == 2,
    "shape_4096_block_q8": contract["shape"]["elements"] == 4096 and contract["shape"]["bytes"] == 4608,
    "producer_incremental_target_available": contract["targets"]["incremental_us_lte"] == 4.8,
    "measured_incremental_passed_target": contract["targets"]["measured_incremental_us"] <= contract["targets"]["incremental_us_lte"],
    "quality_target_available": contract["targets"]["quality_max_dnll_lte"] == 0.01,
    "phases_named": len(phases) == 3,
  }
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_OWNED_Q8_PRODUCER_CACHE_SCOPE",
    "schema": "decode_owned_q8_producer_cache_scope_v1",
    "verdict": "PASS_DECODE_OWNED_Q8_PRODUCER_CACHE_SCOPE_READY" if all(gates.values()) else "BLOCKED_DECODE_OWNED_Q8_PRODUCER_CACHE_SCOPE_INCOMPLETE",
    "gate_pass": all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "contract": contract,
    "phases": phases,
    "next_executable_probe": "extra/qk_decode_owned_q8_producer_cache_object_probe.py",
    "do_not_do": [
      "do not implement gate/up consumer schedule work in the producer scope",
      "do not default-on q8 artifact or successor",
      "do not claim owned implementation until OPC-3 exists and measures",
      "do not start BEAM/search from producer metadata alone",
    ],
    "gates": gates,
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "gates": gates,
    "next_executable_probe": result["next_executable_probe"],
    "targets": contract["targets"],
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
