#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_owned_q8_lifecycle_successor_object_result.json"


@dataclass(frozen=True)
class Q8ProducerCacheContract:
  activation_source: str
  q8_format: str
  reuse_count: int
  lifetime: str
  owned_by_tinygrad: bool
  quality_threshold_dnll: float


@dataclass(frozen=True)
class Q8ConsumerContract:
  role: str
  quant_format: str
  in_features: int
  out_features: int
  dot_contract: str
  output_contract: str


@dataclass(frozen=True)
class Q8RoutePolicy:
  default_on: bool
  fallback: str
  release_flag: str
  supported_model_set: str
  ownership: str


@dataclass(frozen=True)
class Q8ParityTargets:
  artifact_lifecycle_us: float
  modeled_oracle_lifecycle_us: float
  wd_min_speedup: float
  wd_median_speedup: float
  quality_max_dnll: float
  quality_threshold_dnll: float


@dataclass(frozen=True)
class OwnedQ8LifecycleSuccessorObject:
  producer: Q8ProducerCacheContract
  consumers: tuple[Q8ConsumerContract, ...]
  policy: Q8RoutePolicy
  parity: Q8ParityTargets
  lowering_status: str = "metadata_only_unwired"
  performance_claim: bool = False

  def to_dict(self) -> dict[str, Any]:
    ret = asdict(self)
    ret["consumers"] = [asdict(c) for c in self.consumers]
    return ret

  def structural_gate(self) -> dict[str, Any]:
    consumer_roles = {c.role for c in self.consumers}
    checks = {
      "producer.owned": self.producer.owned_by_tinygrad is True,
      "producer.reuse_count_two": self.producer.reuse_count == 2,
      "producer.q8_format_named": bool(self.producer.q8_format),
      "producer.quality_threshold_valid": self.producer.quality_threshold_dnll > 0,
      "consumers.two_roles": consumer_roles == {"ffn_gate", "ffn_up"},
      "consumers.q4k": all(c.quant_format == "Q4_K" for c in self.consumers),
      "consumers.shape_target": all(c.in_features == 4096 and c.out_features == 12288 for c in self.consumers),
      "consumers.dot_contract_named": all("q4/q8" in c.dot_contract for c in self.consumers),
      "policy.default_off": self.policy.default_on is False,
      "policy.fallback_named": self.policy.fallback == "existing default tinygrad decode",
      "policy.ownership_owned": self.policy.ownership == "tinygrad_owned_successor",
      "parity.lifecycle_target_present": self.parity.artifact_lifecycle_us > 0 and self.parity.modeled_oracle_lifecycle_us > 0,
      "parity.wd_min_speedup_ge_1_05": self.parity.wd_min_speedup >= 1.05,
      "parity.quality_pass_target": self.parity.quality_max_dnll <= self.parity.quality_threshold_dnll,
      "lowering.metadata_only": self.lowering_status == "metadata_only_unwired",
      "no_performance_claim": self.performance_claim is False,
    }
    return {"passed": all(checks.values()), "checks": checks, "failed": [k for k, v in checks.items() if not v]}


def load(rel: str, default: Any = None) -> Any:
  p = ROOT / rel
  return json.loads(p.read_text()) if p.exists() else default


def main() -> int:
  scope = load("bench/qk-decode-primitive-transfer/decode_owned_q8_lifecycle_successor_scope_result.json", {})
  successor = scope.get("successor_object", {})
  policy = successor.get("policy", {})
  producer = successor.get("producer", {})
  parity = successor.get("parity_targets", {})

  obj = OwnedQ8LifecycleSuccessorObject(
    producer=Q8ProducerCacheContract(
      activation_source=producer.get("source_activation", "post_norm_decode_activation"),
      q8_format=producer.get("format", "block_q8_1_or_artifact_compatible_q8"),
      reuse_count=int(producer.get("reuse_count", 0)),
      lifetime=producer.get("lifetime", "one token"),
      owned_by_tinygrad=True,
      quality_threshold_dnll=float(parity.get("quality_threshold", 0.0)),
    ),
    consumers=(
      Q8ConsumerContract("ffn_gate", "Q4_K", 4096, 12288, "packed q4/q8 dot4 with scale/min correction", "ffn_gate row output"),
      Q8ConsumerContract("ffn_up", "Q4_K", 4096, 12288, "packed q4/q8 dot4 with scale/min correction", "ffn_up row output"),
    ),
    policy=Q8RoutePolicy(
      default_on=bool(policy.get("default_on_initially", True)),
      fallback=policy.get("fallback", ""),
      release_flag=policy.get("release_flag_start", ""),
      supported_model_set=policy.get("supported_model_set_target", ""),
      ownership="tinygrad_owned_successor",
    ),
    parity=Q8ParityTargets(
      artifact_lifecycle_us=float(parity.get("artifact_lifecycle_us", 0.0)),
      modeled_oracle_lifecycle_us=float(parity.get("modeled_oracle_lifecycle_us", 0.0)),
      wd_min_speedup=float(parity.get("wd_min_speedup", 0.0)),
      wd_median_speedup=float(parity.get("wd_median_speedup", 0.0)),
      quality_max_dnll=float(parity.get("quality_max_dnll", 1.0)),
      quality_threshold_dnll=float(parity.get("quality_threshold", 0.0)),
    ),
  )
  gate = obj.structural_gate()
  gates = {
    "scope_ready": scope.get("gate_pass") is True,
    "object_structural_gate": gate["passed"],
    "object_default_off": obj.policy.default_on is False,
    "object_metadata_only": obj.lowering_status == "metadata_only_unwired",
  }
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_OWNED_Q8_LIFECYCLE_SUCCESSOR_OBJECT",
    "schema": "decode_owned_q8_lifecycle_successor_object_v1",
    "verdict": "PASS_DECODE_OWNED_Q8_LIFECYCLE_SUCCESSOR_OBJECT_STRUCTURAL" if all(gates.values()) else "BLOCKED_DECODE_OWNED_Q8_LIFECYCLE_SUCCESSOR_OBJECT_INCOMPLETE",
    "gate_pass": all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "object": obj.to_dict(),
    "structural_gate": gate,
    "gates": gates,
    "next": {
      "next_local_probe": "artifact parity harness: baseline vs q8 artifact vs owned-successor target rows",
      "blocked_implementation": [
        "owned q8 producer/cache implementation",
        "owned packed q4/q8 gate/up consumers at artifact lifecycle target",
      ],
      "search_status": "blocked until lowerable owned candidate exists",
    },
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "gates": gates,
    "failed": gate["failed"],
    "next": result["next"],
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
