#!/usr/bin/env python3
# SW-pipeline charter P0 — baseline + premise check (no GPU; parses the saved authority disasm).
# Confirms that the authority WMMA kernel is compute-light / address+load-heavy (the premise behind
# Lever A addressing-mode lowering) and records the PTM-1 in-harness authority TFLOPS as the number any
# Phase-1 renderer change must beat. Does NOT change any code path.
from __future__ import annotations

import json, pathlib, re
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"
DISASM = ROOT / "bench/amd-broad-backend-roadmap/bb5a8_authority_kernel_capture/tinygrad_ffn_gate_up_authority.disasm"

PATTERNS = {
  "v_wmma": r"\bv_wmma[a-z0-9_]*",
  "v_int_alu_addr": r"\bv_(add|sub|addc|lshl|lshlrev|mul_lo|mul_hi|ashr|and|or)[a-z0-9_]*",
  "s_int_alu": r"\bs_(add|addc|lshl|mul)[a-z0-9_]*",
  "global_load": r"\bglobal_load[a-z0-9_]*",
  "ds_load_store": r"\bds_(load|store)[a-z0-9_]*",
  "s_waitcnt": r"\bs_waitcnt[a-z0-9_]*",
  "v_fp_alu": r"\bv_(fma|mac|mad|mul_f|add_f)[a-z0-9_]*",
}


def read_json(rel: str, default: Any = None) -> Any:
  path = ROOT / rel
  if not path.exists(): return default
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main() -> int:
  text = DISASM.read_text(errors="ignore") if DISASM.exists() else ""
  mix = {name: len(re.findall(pat, text)) for name, pat in PATTERNS.items()}
  wmma = mix["v_wmma"] or 1
  addr_to_wmma = mix["v_int_alu_addr"] / wmma
  load_to_wmma = mix["global_load"] / wmma

  ptm1 = read_json("bench/amd-broad-backend-roadmap/bb5a10_ptm1_same_harness_authority_bridge_result.json", {})
  authority_tflops = ptm1.get("authority_best_tflops")

  # Premise: address-arith + loads dominate; compute (wmma) is a small minority; kernel is global-direct (no LDS).
  premise = {
    "compute_is_minority": mix["v_wmma"] < mix["v_int_alu_addr"] and mix["v_wmma"] < mix["global_load"],
    "address_arith_dominates_compute": addr_to_wmma >= 2.0,
    "global_direct_no_lds": mix["ds_load_store"] == 0,
  }
  gate = {
    "disasm_present": bool(text),
    "premise_holds": all(premise.values()),
    "ptm1_authority_baseline_present": authority_tflops is not None,
  }
  gate_pass = all(gate.values())
  result = {
    "date": "2026-06-20",
    "phase": "SW_PIPELINE_P0_baseline",
    "schema": "amd_sw_pipeline_p0_baseline_v1",
    "verdict": "PASS_SW_PIPELINE_P0_BASELINE" if gate_pass else "BLOCKED_SW_PIPELINE_P0_BASELINE",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "authority_disasm_instruction_mix": mix,
    "ratios": {"addr_int_alu_per_wmma": addr_to_wmma, "global_load_per_wmma": load_to_wmma},
    "premise": premise,
    "ptm1_authority_baseline_tflops": authority_tflops,
    "phase1_kill_gate": {
      "lever": "addressing_mode_lowering (Lever A): base+immediate-offset loads + strength-reduce base ptr across K",
      "scope": "local AMD renderer change; NO new IR op, NO new pass",
      "pass_criterion": "isolated/in-harness authority WMMA matmul improves >=1.2x (byte-identical correctness) under the PTM-1 interleaved harness",
      "if_fail": "kill Lever A; do NOT proceed to Lever B pipelining capability on addressing grounds",
    },
    "gate": gate,
    "next_action": "Phase 1 (Lever A): implement AMD renderer base+immediate-offset addressing + base-pointer strength reduction; "
                   "measure under qk_amd_bb5a10_ptm1_same_harness_bridge; gate >=1.2x before any Lever B work.",
    "input_artifacts": [
      "bench/amd-broad-backend-roadmap/bb5a8_authority_kernel_capture/tinygrad_ffn_gate_up_authority.disasm",
      "bench/amd-broad-backend-roadmap/bb5a10_ptm1_same_harness_authority_bridge_result.json",
    ],
  }
  write_json("sw_pipeline_p0_baseline_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/sw_pipeline_p0_baseline_result.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "instruction_mix": mix,
    "addr_per_wmma": round(addr_to_wmma, 2),
    "authority_baseline_tflops": authority_tflops,
    "next": result["next_action"],
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
