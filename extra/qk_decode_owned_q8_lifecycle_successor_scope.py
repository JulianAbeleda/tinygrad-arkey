#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_owned_q8_lifecycle_successor_scope_result.json"


def load(rel: str, default: Any = None) -> Any:
  p = ROOT / rel
  return json.loads(p.read_text()) if p.exists() else default


def main() -> int:
  q8 = load("bench/q8-ffn-artifact-promotion/promotion_result.json", {})
  p8 = load("bench/qk-decode-mmvq-large-project/p8_fused_lifecycle_decision.json", {})
  ledger = load("bench/qk-decode-primitive-transfer/decode_route_level_primitive_ledger_result.json", {})
  schedule = load("bench/qk-decode-primitive-transfer/decode_mmvq_schedule_object_result.json", {})
  att = load("bench/qk-decode-primitive-transfer/decode_att_unblock_audit_result.json", {})

  q8_summary = q8.get("summary", {})
  q8_policy = q8_summary.get("policy_decision", {})
  q8_quality = q8_summary.get("quality", {})
  p8_decision = p8.get("p8d_decision", {})
  p8_artifact = p8.get("p8c_handwritten_prototype", {})
  wd_rows = p8_decision.get("whole_decode_wd_rows", [])
  min_wd_speedup = min((row.get("speedup", 0.0) for row in wd_rows), default=0.0)

  successor_object = {
    "name": "OwnedQ8LifecycleSuccessor",
    "status": "scope_only_unwired",
    "target_route": "replace external q8 FFN artifact with tinygrad-owned route-level primitive",
    "roles": [
      {"role": "ffn_gate", "quant": "Q4_K", "shape": {"in_features": 4096, "out_features": 12288}},
      {"role": "ffn_up", "quant": "Q4_K", "shape": {"in_features": 4096, "out_features": 12288}},
    ],
    "producer": {
      "name": "activation_q8_producer_cache",
      "format": "block_q8_1_or_artifact_compatible_q8",
      "source_activation": "post_norm_decode_activation",
      "reuse_count": 2,
      "lifetime": "one token, shared across gate/up consumers",
      "ownership_required": True,
      "quality_policy": q8_policy.get("quality_threshold"),
    },
    "consumers": [
      {
        "name": "gate_q4k_q8_consumer",
        "weight_format": "Q4_K",
        "dot": "packed q4/q8 dot4 with scale/min correction",
        "output": "ffn_gate row output",
      },
      {
        "name": "up_q4k_q8_consumer",
        "weight_format": "Q4_K",
        "dot": "packed q4/q8 dot4 with scale/min correction",
        "output": "ffn_up row output",
      },
    ],
    "policy": {
      "default_on_initially": False,
      "fallback": "existing default tinygrad decode",
      "release_flag_start": "successor flag required; do not reuse Q8_FFN_HANDWRITTEN until parity is proven",
      "supported_model_set_target": q8_policy.get("supported_model_set"),
      "default_on_requires": [
        "owned implementation, not external artifact dependency",
        "multi-window dNLL <= 0.01",
        "W==D min speedup >= artifact min speedup",
        "fallback and model coverage accepted",
      ],
    },
    "parity_targets": {
      "artifact_lifecycle_us": p8_artifact.get("hipcc_lld_artifact_lifecycle_us"),
      "modeled_oracle_lifecycle_us": p8_artifact.get("modeled_oracle_lifecycle_us"),
      "artifact_speedup_vs_p7e_baseline": p8_artifact.get("artifact_speedup_vs_p7e_baseline"),
      "wd_min_speedup": min_wd_speedup,
      "wd_median_speedup": p8_decision.get("median_wd_speedup"),
      "quality_max_dnll": q8_quality.get("max_dnll"),
      "quality_threshold": q8_quality.get("threshold"),
    },
  }

  phases = [
    {
      "id": "OQ8-1-object-contract",
      "purpose": "make producer/cache, two consumers, policy, quality, and fallback first-class",
      "exit_gate": "structural probe instantiates OwnedQ8LifecycleSuccessor from existing artifacts",
      "can_do_now": True,
    },
    {
      "id": "OQ8-2-artifact-parity-harness",
      "purpose": "bind the successor to the same W==D/dNLL/lifecycle gates as the q8 artifact",
      "exit_gate": "same-harness matrix names baseline, artifact, and successor target rows",
      "can_do_now": True,
    },
    {
      "id": "OQ8-3-owned-producer-candidate",
      "purpose": "replace external q8 producer side with tinygrad-owned producer/cache path",
      "exit_gate": "q8 bytes/scale semantics and dNLL gate match artifact policy",
      "can_do_now": False,
      "blocked_on": "needs implementation work beyond metadata scope",
    },
    {
      "id": "OQ8-4-owned-consumer-candidate",
      "purpose": "replace external gate/up consumers with tinygrad-owned packed q4/q8 dot consumers",
      "exit_gate": "correctness and lifecycle <= artifact target before W==D",
      "can_do_now": False,
      "blocked_on": "native consumer schedule still lacks material attributed lever",
    },
    {
      "id": "OQ8-5-promotion-decision",
      "purpose": "decide hardened opt-in, default-on, or reject",
      "exit_gate": "quality, fallback, coverage, W==D timing, ownership, and ATT/native attribution all reconciled",
      "can_do_now": False,
      "blocked_on": "requires OQ8-3/OQ8-4 evidence, or explicit policy acceptance for artifact route",
    },
  ]

  gates = {
    "route_ledger_ready": ledger.get("gate_pass") is True,
    "decode_schedule_object_ready": schedule.get("gate_pass") is True,
    "q8_artifact_hardened_optin": q8.get("verdict") == "PASS_Q8_FFN_ARTIFACT_PROMOTION_TO_HARDENED_OPT_IN",
    "q8_artifact_default_off": q8_policy.get("default_on") is False,
    "q8_quality_target_available": q8_quality.get("threshold") == 0.01 and q8_quality.get("max_dnll", 1.0) <= 0.01,
    "artifact_parity_targets_available": p8_artifact.get("hipcc_lld_artifact_lifecycle_us") is not None and min_wd_speedup >= 1.05,
    "producer_reuse_count_two": successor_object["producer"]["reuse_count"] == 2,
    "two_consumers_named": len(successor_object["consumers"]) == 2,
    "fallback_policy_named": successor_object["policy"]["fallback"] == "existing default tinygrad decode",
    "att_independent_local_scope": att.get("verdict") == "BLOCKED_DECODE_ATT_DECODER_SO_MISSING",
    "scope_has_next_phases": len(phases) == 5,
  }

  result = {
    "date": "2026-06-20",
    "phase": "DECODE_OWNED_Q8_LIFECYCLE_SUCCESSOR_SCOPE",
    "schema": "decode_owned_q8_lifecycle_successor_scope_v1",
    "verdict": "PASS_DECODE_OWNED_Q8_LIFECYCLE_SUCCESSOR_SCOPE_READY" if all(gates.values()) else "BLOCKED_DECODE_OWNED_Q8_LIFECYCLE_SUCCESSOR_SCOPE_INCOMPLETE",
    "gate_pass": all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "successor_object": successor_object,
    "phases": phases,
    "next_executable_probe": "extra/qk_decode_owned_q8_lifecycle_successor_object_probe.py",
    "do_not_do": [
      "do not default-on q8 artifact through this successor scope",
      "do not resume local native MMVQ schedule edits without ATT or a route-level parity gate",
      "do not start BEAM/search until the owned successor has a lowerable candidate and measured objective",
      "do not treat the external artifact as owned implementation",
    ],
    "gates": gates,
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "gates": gates,
    "next_executable_probe": result["next_executable_probe"],
    "parity_targets": successor_object["parity_targets"],
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
