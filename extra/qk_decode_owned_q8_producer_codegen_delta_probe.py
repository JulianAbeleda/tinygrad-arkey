#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tinygrad.device import Device
from extra.q8_ffn_codegen_transfer_audit import inspect_blob
from extra.q8_ffn_fast_artifact_probe import compile_hipcc_linked, hip_norm_source
from extra.q8_ffn_hcq_artifact import NORM_SOURCE


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_owned_q8_producer_codegen_delta_result.json"


def load(rel: str, default: Any = None) -> Any:
  p = ROOT / rel
  return json.loads(p.read_text()) if p.exists() else default


def delta_counts(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
  keys = sorted(set(a) | set(b))
  return {k: int(a.get(k, 0)) - int(b.get(k, 0)) for k in keys}


def main() -> int:
  scope = load("bench/qk-decode-primitive-transfer/decode_owned_q8_producer_hip_delta_scope_result.json", {})
  dev = Device["AMD"]
  hip_blob = compile_hipcc_linked(hip_norm_source(1024), "gfx1100")
  comgr_blob = dev.compiler.compile(NORM_SOURCE)
  hip = inspect_blob("decode_owned_q8_producer_hipcc_lld", hip_blob, "decode_owned_q8_producer_hipcc_lld")
  comgr = inspect_blob("decode_owned_q8_producer_comgr", comgr_blob, "decode_owned_q8_producer_comgr")
  hip_counts = hip.get("disasm", {}).get("grouped_counts", {})
  comgr_counts = comgr.get("disasm", {}).get("grouped_counts", {})
  hip_runtime = hip.get("runtime", {})
  comgr_runtime = comgr.get("runtime", {})
  deltas = {
    "comgr_minus_hip_grouped_counts": delta_counts(comgr_counts, hip_counts),
    "comgr_minus_hip_instruction_count": int(comgr.get("disasm", {}).get("instruction_count", 0)) - int(hip.get("disasm", {}).get("instruction_count", 0)),
    "comgr_minus_hip_group_segment_size": int(comgr_runtime.get("group_segment_size", 0)) - int(hip_runtime.get("group_segment_size", 0)),
    "comgr_minus_hip_private_segment_size": int(comgr_runtime.get("private_segment_size", 0)) - int(hip_runtime.get("private_segment_size", 0)),
  }
  gates = {
    "scope_ready": scope.get("gate_pass") is True,
    "hip_loads": hip_runtime.get("loads_in_amdprogram") is True,
    "comgr_loads": comgr_runtime.get("loads_in_amdprogram") is True,
    "both_no_relocs": not hip.get("readelf", {}).get("readelf_relocations") and not comgr.get("readelf", {}).get("readelf_relocations"),
    "delta_identified": deltas["comgr_minus_hip_instruction_count"] != 0 or any(deltas["comgr_minus_hip_grouped_counts"].values()),
  }
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_OWNED_Q8_PRODUCER_CODEGEN_DELTA",
    "schema": "decode_owned_q8_producer_codegen_delta_v1",
    "verdict": "PASS_DECODE_OWNED_Q8_PRODUCER_CODEGEN_DELTA_CAPTURED" if all(gates.values()) else "BLOCKED_DECODE_OWNED_Q8_PRODUCER_CODEGEN_DELTA",
    "gate_pass": all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "hipcc_lld_producer": {
      "runtime": hip_runtime,
      "instruction_count": hip.get("disasm", {}).get("instruction_count"),
      "grouped_counts": hip_counts,
      "top_mnemonics": hip.get("disasm", {}).get("top_mnemonics", [])[:20],
      "elf_bytes": hip.get("elf", {}).get("bytes"),
    },
    "comgr_producer": {
      "runtime": comgr_runtime,
      "instruction_count": comgr.get("disasm", {}).get("instruction_count"),
      "grouped_counts": comgr_counts,
      "top_mnemonics": comgr.get("disasm", {}).get("top_mnemonics", [])[:20],
      "elf_bytes": comgr.get("elf", {}).get("bytes"),
    },
    "deltas": deltas,
    "decision": {
      "if_static_delta_clear": "Use the largest grouped-count/resource deltas to scope a producer codegen optimization.",
      "if_static_delta_not_clear": "Treat remaining HIP delta as compiler/runtime boundary until PC timing exists.",
    },
    "gates": gates,
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "hipcc_lld": result["hipcc_lld_producer"],
    "comgr": result["comgr_producer"],
    "deltas": deltas,
    "gates": gates,
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
