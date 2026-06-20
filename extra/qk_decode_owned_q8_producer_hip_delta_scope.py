#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_owned_q8_producer_hip_delta_scope_result.json"


def load(rel: str, default: Any = None) -> Any:
  p = ROOT / rel
  return json.loads(p.read_text()) if p.exists() else default


def main() -> int:
  closeout = load("bench/qk-decode-primitive-transfer/decode_owned_q8_producer_hcq_parity_closeout_result.json", {})
  manifest = load("bench/q8-ffn-amd-scheduler-project/artifact_build_manifest.json", {})
  candidate = load("bench/qk-decode-primitive-transfer/decode_owned_q8_producer_cache_lowering_candidate_result.json", {})

  prod_artifact = (((manifest.get("artifacts") or {}).get("producer") or {}).get("inspect") or {})
  artifact_disasm = prod_artifact.get("disasm", {})
  artifact_runtime = prod_artifact.get("runtime", {})
  owned_us = candidate.get("producer_us")
  hip_us = (closeout.get("comparison") or {}).get("hip_oracle_producer_us")
  scope_rows = [
    {
      "id": "HD-1-runtime-boundary",
      "question": "Is the 7.5us vs 15.7us gap mostly HIP-runtime/event path versus HCQ dispatch/measurement boundary?",
      "evidence_needed": "same binary producer timed in one-clock HCQ and HIP-style harness, if possible",
      "do_now": False,
      "blocked_on": "HIP/HCQ process boundary and artifact policy",
    },
    {
      "id": "HD-2-ISA-delta",
      "question": "Which producer ISA/resource differences separate hipcc/LLD from COMGR raw-C producer?",
      "evidence_needed": "producer-only disasm/resource diff: instruction groups, LDS, waitcnt, registers, descriptor",
      "do_now": True,
      "probe": "extra/qk_decode_owned_q8_producer_codegen_delta_probe.py",
    },
    {
      "id": "HD-3-owned-optimized-producer",
      "question": "Can an owned producer be lowered closer to HIP oracle without external artifact dependency?",
      "evidence_needed": "new lowerable candidate at <=7.5us or explicit rejection",
      "do_now": False,
      "blocked_on": "HD-2 result or hand-ASM/codegen work",
    },
  ]
  gates = {
    "hcq_parity_closeout_passed": closeout.get("gate_pass") is True,
    "hip_delta_exists": owned_us is not None and hip_us is not None and owned_us > hip_us,
    "artifact_producer_inspect_available": bool(artifact_disasm) and artifact_runtime.get("loads_in_amdprogram") is True,
    "scope_rows_named": len(scope_rows) == 3,
  }
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_OWNED_Q8_PRODUCER_HIP_DELTA_SCOPE",
    "schema": "decode_owned_q8_producer_hip_delta_scope_v1",
    "verdict": "PASS_DECODE_OWNED_Q8_PRODUCER_HIP_DELTA_SCOPE_READY" if all(gates.values()) else "BLOCKED_DECODE_OWNED_Q8_PRODUCER_HIP_DELTA_SCOPE",
    "gate_pass": all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "delta": {
      "owned_comgr_us": owned_us,
      "hip_oracle_us": hip_us,
      "owned_slowdown_vs_hip_oracle": owned_us / hip_us if owned_us and hip_us else None,
    },
    "artifact_producer_static": {
      "instruction_count": artifact_disasm.get("instruction_count"),
      "grouped_counts": artifact_disasm.get("grouped_counts"),
      "runtime": artifact_runtime,
    },
    "scope_rows": scope_rows,
    "next_executable_probe": "extra/qk_decode_owned_q8_producer_codegen_delta_probe.py",
    "gates": gates,
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "delta": result["delta"],
    "next_executable_probe": result["next_executable_probe"],
    "gates": gates,
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
