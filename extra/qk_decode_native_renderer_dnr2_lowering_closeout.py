#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr2_lowering_result.json"
FRESH = ROOT / "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr2_gateup_full_fresh.json"
HISTORICAL = ROOT / "bench/q8-ffn-codegen-transfer/asm_gateup_full.json"


def read_json(path: Path) -> dict[str, Any]:
  with path.open() as f:
    return json.load(f)


def artifact(path: Path) -> dict[str, Any]:
  if path.exists(): return read_json(path)
  return {"missing": str(path.relative_to(ROOT))}


def main() -> int:
  dnr1 = read_json(ROOT / "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr1_oracle_binding_result.json")
  oracle = read_json(ROOT / "bench/q8-ffn-amd-scheduler-project/oracle_contract.json")
  fresh = artifact(FRESH)
  historical = artifact(HISTORICAL)

  correctness = fresh.get("correctness", {})
  gates = fresh.get("gates", {})
  timing = fresh.get("timing", {})
  fresh_median_us = float(timing.get("median_ms", 0.0)) * 1000.0 if timing else None
  historical_median_us = float((historical.get("timing") or {}).get("median_ms", 0.0)) * 1000.0 if "timing" in historical else None
  oracle_consumer_us = float(oracle.get("known_timings_us", {}).get("hipcc_lld_gateup_current_loader", 0.0))
  tinygrad_asm_us = float(oracle.get("known_timings_us", {}).get("tinygrad_asm_gateup_full", 0.0))

  address_data_format_contract = [
    {
      "piece": "work decomposition",
      "native_lowering": "gidx0=row, gidx1=gate/up select, lidx0 tid in 128-thread row group",
      "oracle_match": "128 threads per row; block y selects gate/up; 16 Q4_K blocks; sub=tid&7; kb=tid/8",
    },
    {
      "piece": "Q4_K block address",
      "native_lowering": "row * 2304 + kb * 144",
      "oracle_match": "16 Q4_K blocks per 4096-wide row, 144 bytes per block",
    },
    {
      "piece": "Q4_K scale/min",
      "native_lowering": "load d, dmin, q[sub], q[sub+4], q[sub&3], then select lt4/ge4 scale/min form",
      "oracle_match": "llama Q4_K get_scale_min contract",
    },
    {
      "piece": "Q4 nibble selection",
      "native_lowering": "qs + 16 + (sub/2)*32, choose low/high nibbles with sub&1",
      "oracle_match": "8 sub-blocks of 32 values per Q4_K block",
    },
    {
      "piece": "block_q8_1 address",
      "native_lowering": "(kb * 8 + sub) * 36; load half d and 32 int8 qs",
      "oracle_match": "4096 activations as 128 block_q8_1 records",
    },
    {
      "piece": "dot/min correction",
      "native_lowering": "16 v_dot4_i32_iu8: q4*q8 plus ones*q8 for min correction",
      "oracle_match": "oracle and tinygrad both emit 16 dot4 operations",
    },
    {
      "piece": "row reduction/output",
      "native_lowering": "wave ds_bpermute reduce, LDS 4-wave reduce, one float store to selected gate/up output",
      "oracle_match": "direct row output for gate and up roles",
    },
  ]

  dnr2_gates = {
    "dnr1_oracle_binding_passed": dnr1.get("gate_pass") is True,
    "fresh_native_run_present": "missing" not in fresh,
    "native_gate_correct": gates.get("gate_correct_lte_2e_3") is True,
    "native_up_correct": gates.get("up_correct_lte_2e_3") is True,
    "native_owned_no_external_artifact": gates.get("no_external_artifact") is True,
    "full_authority_shape": fresh.get("rows") == 12288 and fresh.get("q8_bytes") == 4608,
    "performance_not_claimed": gates.get("consumer_lte_60us") is not True,
  }

  perf_gap = None
  if historical_median_us and oracle_consumer_us:
    perf_gap = historical_median_us - oracle_consumer_us

  result = {
    "date": "2026-06-20",
    "phase": "DNR-2_DECODE_NATIVE_ADDRESS_DATA_FORMAT_LOWERING",
    "schema": "decode_native_renderer_dnr2_lowering_closeout_v1",
    "verdict": "PASS_DNR2_NATIVE_LOWERING_CORRECT_BLOCKED_DNR3_SCHEDULER_RESOURCE" if all(dnr2_gates.values()) else "BLOCKED_DNR2_NATIVE_LOWERING_CORRECTNESS",
    "gate_pass": all(dnr2_gates.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "native_route": "tinygrad Ops.PROGRAM AMD DSL q8_b2b_fullrow_reduce",
    "address_data_format_contract": address_data_format_contract,
    "correctness": correctness,
    "timing": {
      "fresh_short_median_us": fresh_median_us,
      "historical_median_us": historical_median_us,
      "oracle_consumer_us": oracle_consumer_us,
      "tinygrad_asm_oracle_record_us": tinygrad_asm_us,
      "historical_native_minus_oracle_us": perf_gap,
      "consumer_lte_60us": gates.get("consumer_lte_60us"),
    },
    "gates": dnr2_gates,
    "blocked_at": {
      "next_phase": "DNR-3 scheduler/resource model",
      "reason": "Native address/data-format lowering is correct, but the tinygrad AMD DSL body is much slower than the hipcc/LLD oracle.",
      "not_blocked_by": ["Q4_K block addressing", "block_q8_1 addressing", "scale/min decode", "dot4 selection", "gate/up output selection", "numeric correctness"],
      "open_work": ["s_clause/s_delay_alu semantic placement", "global_load_b128/coalesced load shape without correctness drift", "register live-range/resource policy", "instruction ordering and wait policy", "branch/resource policy matching oracle"],
    },
    "input_artifacts": [
      "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr1_oracle_binding_result.json",
      "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr2_gateup_full_fresh.json",
      "bench/q8-ffn-codegen-transfer/asm_gateup_full.json",
      "bench/q8-ffn-amd-scheduler-project/oracle_contract.json",
    ],
    "next_action": "Start DNR-3 only as broad scheduler/resource work. Do not reopen address lowering, BEAM/search, or one-off wait/load/reduction patches from DNR-2.",
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "gate_pass": result["gate_pass"],
    "correctness": correctness,
    "timing": result["timing"],
    "blocked_at": result["blocked_at"],
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
