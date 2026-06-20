#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_owned_q8_producer_cache_object_result.json"


@dataclass(frozen=True)
class OwnedQ8ByteLayout:
  elements: int
  block_elems: int
  block_bytes: int
  blocks: int
  total_bytes: int
  fields: tuple[str, ...]


@dataclass(frozen=True)
class OwnedQ8ProducerCacheObject:
  source_activation: str
  operation: str
  layout: OwnedQ8ByteLayout
  reuse_count: int
  consumers: tuple[str, ...]
  lifetime: str
  fallback: str
  default_on: bool
  producer_lifecycle_us_lte: float
  incremental_us_lte: float
  measured_incremental_us: float
  quality_max_dnll_lte: float
  lowering_status: str = "metadata_only_unwired"
  performance_claim: bool = False

  def to_dict(self) -> dict[str, Any]:
    ret = asdict(self)
    ret["consumers"] = list(self.consumers)
    ret["layout"]["fields"] = list(self.layout.fields)
    return ret

  def structural_gate(self) -> dict[str, Any]:
    checks = {
      "layout.elements_4096": self.layout.elements == 4096,
      "layout.block_elems_32": self.layout.block_elems == 32,
      "layout.block_bytes_36": self.layout.block_bytes == 36,
      "layout.blocks_128": self.layout.blocks == 128,
      "layout.total_bytes_4608": self.layout.total_bytes == 4608,
      "layout.fields_block_q8_1": self.layout.fields == ("d_fp16", "s_fp16", "qs_i8x32"),
      "reuse.two_consumers": self.reuse_count == 2 and set(self.consumers) == {"ffn_gate", "ffn_up"},
      "policy.default_off": self.default_on is False,
      "policy.fallback_named": self.fallback == "existing default tinygrad decode",
      "target.lifecycle_present": self.producer_lifecycle_us_lte > 0,
      "target.incremental_present": self.incremental_us_lte > 0,
      "target.measured_incremental_pass": self.measured_incremental_us <= self.incremental_us_lte,
      "target.quality_threshold_present": self.quality_max_dnll_lte == 0.01,
      "lowering.metadata_only": self.lowering_status == "metadata_only_unwired",
      "no_performance_claim": self.performance_claim is False,
    }
    return {"passed": all(checks.values()), "checks": checks, "failed": [k for k, v in checks.items() if not v]}


def load(rel: str, default: Any = None) -> Any:
  p = ROOT / rel
  return json.loads(p.read_text()) if p.exists() else default


def main() -> int:
  scope = load("bench/qk-decode-primitive-transfer/decode_owned_q8_producer_cache_scope_result.json", {})
  contract = scope.get("contract", {})
  shape = contract.get("shape", {})
  cache = contract.get("cache_contract", {})
  targets = contract.get("targets", {})
  kernel = contract.get("producer_kernel_contract", {})

  obj = OwnedQ8ProducerCacheObject(
    source_activation=contract.get("source_activation", "post_norm_decode_activation"),
    operation=kernel.get("operation", "rmsnorm output plus q8 sidechannel"),
    layout=OwnedQ8ByteLayout(
      elements=int(shape.get("elements", 0)),
      block_elems=32,
      block_bytes=int(shape.get("block_bytes", 0)),
      blocks=int(shape.get("blocks", 0)),
      total_bytes=int(shape.get("bytes", 0)),
      fields=("d_fp16", "s_fp16", "qs_i8x32"),
    ),
    reuse_count=int(cache.get("reuse_count", 0)),
    consumers=tuple(cache.get("consumers", [])),
    lifetime=cache.get("lifetime", ""),
    fallback=cache.get("fallback", ""),
    default_on=bool(cache.get("default_on", True)),
    producer_lifecycle_us_lte=float(targets.get("producer_lifecycle_us_lte", 0.0)),
    incremental_us_lte=float(targets.get("incremental_us_lte", 0.0)),
    measured_incremental_us=float(targets.get("measured_incremental_us", 999.0)),
    quality_max_dnll_lte=float(targets.get("quality_max_dnll_lte", 0.0)),
  )
  structural = obj.structural_gate()
  gates = {
    "scope_ready": scope.get("gate_pass") is True,
    "object_structural": structural["passed"],
    "object_metadata_only": obj.lowering_status == "metadata_only_unwired",
    "owned_implementation_not_claimed": True,
  }
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_OWNED_Q8_PRODUCER_CACHE_OBJECT",
    "schema": "decode_owned_q8_producer_cache_object_v1",
    "verdict": "PASS_DECODE_OWNED_Q8_PRODUCER_CACHE_OBJECT_STRUCTURAL" if all(gates.values()) else "BLOCKED_DECODE_OWNED_Q8_PRODUCER_CACHE_OBJECT_INCOMPLETE",
    "gate_pass": all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "object": obj.to_dict(),
    "structural_gate": structural,
    "gates": gates,
    "next": {
      "next_probe": "extra/qk_decode_owned_q8_producer_cache_reference_probe.py",
      "purpose": "freeze byte/reference semantics before any owned lowering candidate",
      "implementation_status": "not_built",
    },
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "gates": gates,
    "failed": structural["failed"],
    "next": result["next"],
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
