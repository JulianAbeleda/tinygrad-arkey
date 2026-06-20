#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from typing import Any

from tinygrad import Tensor
from tinygrad.dtype import dtypes

from extra.q8_ffn_asm_fullrow_reduce import HIDDEN, Q4_WORDS, Q8_BYTES, build_fullrow_reduce
from extra.qk_decode_native_renderer_dnr3b_compound_emitter_probe import grouped, inst_name, insts_from_program


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c1_load_shape_result.json"


def read_json(rel: str) -> dict[str, Any]:
  with (ROOT / rel).open() as f:
    return json.load(f)


def inst_text(inst: Any) -> str:
  return str(inst)


def mentions_reg(text: str, reg: str) -> bool:
  return reg in text


def build_stream() -> list[Any]:
  gate = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  up = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  gate_words = Tensor.empty(Q4_WORDS, dtype=dtypes.uint32, device="AMD").contiguous()
  up_words = Tensor.empty(Q4_WORDS, dtype=dtypes.uint32, device="AMD").contiguous()
  q8 = Tensor.empty(Q8_BYTES, dtype=dtypes.uint8, device="AMD").contiguous()
  return insts_from_program(build_fullrow_reduce(gate.uop, up.uop, gate_words.uop, up_words.uop, q8.uop))


def first_scalar_pair_index(insts: list[Any]) -> int:
  for i in range(len(insts)-1):
    if inst_name(insts[i]) == "global_load_b32" and inst_name(insts[i+1]) == "global_load_b32":
      return i
  raise RuntimeError("could not find scalar b32 load pair")


def classify_load_pairs(insts: list[Any]) -> list[dict[str, Any]]:
  pairs: list[dict[str, Any]] = []
  for i in range(len(insts)-1):
    t0, t1 = inst_text(insts[i]), inst_text(insts[i+1])
    if inst_name(insts[i]) != "global_load_b32" or inst_name(insts[i+1]) != "global_load_b32":
      continue
    if not ("v[8]" in t0 and "s[16:17]" in t0 and "v[9]" in t1 and "s[18:19]" in t1):
      continue
    next_pair = next((j for j in range(i+2, len(insts)-1)
                      if inst_name(insts[j]) == "global_load_b32" and inst_name(insts[j+1]) == "global_load_b32"), len(insts))
    window = [inst_text(x) for x in insts[i:next_pair]]
    pairs.append({
      "iteration": len(pairs),
      "q4_load_index": i,
      "q8_load_index": i+1,
      "q4_dest": "v[8]",
      "q8_dest": "v[9]",
      "wait_index": i+2 if i+2 < len(insts) and inst_name(insts[i+2]) == "s_waitcnt" else None,
      "first_dot4_index": next((i+k for k, text in enumerate(window) if text.startswith("v_dot4_i32_iu8")), None),
      "mutates_q4_dest_before_dot": any(text.startswith(("v_and_b32", "v_cndmask_b32")) and mentions_reg(text, "v[8]") for text in window),
      "uses_q8_dest_in_dot": any(text.startswith("v_dot4_i32_iu8") and mentions_reg(text, "v[9]") for text in window),
      "address_increment_indices": [
        i+k for k, text in enumerate(window)
        if text.startswith("v_add_nc_u32_e32") and (mentions_reg(text, "v[23]") or mentions_reg(text, "v[24]"))
      ],
      "span_until_next_pair": next_pair - i,
    })
  return pairs


def main() -> int:
  dnr3a = read_json("bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3a_scheduler_resource_plan_result.json")
  dnr3b = read_json("bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3b_compound_emitter_result.json")
  oracle = read_json("bench/q8-ffn-amd-scheduler-project/oracle_contract.json")
  insts = build_stream()
  texts = [inst_text(i) for i in insts]
  names = [inst_name(i) for i in insts]
  first_pair = first_scalar_pair_index(insts)
  preheader_global_loads = [
    {"index": i, "name": names[i], "text": texts[i]}
    for i in range(first_pair) if names[i].startswith("global_load")
  ]
  pairs = classify_load_pairs(insts)
  isa = importlib.import_module("tinygrad.runtime.autogen.amd.rdna3.ins")
  native_grouped = grouped(insts)
  oracle_contract = oracle["instruction_contract"]
  oracle_grouped = oracle_contract["oracle_grouped"]
  consumer_coupling = all(
    p["wait_index"] is not None and p["first_dot4_index"] is not None and
    p["mutates_q4_dest_before_dot"] and p["uses_q8_dest_in_dot"] and len(p["address_increment_indices"]) == 2
    for p in pairs
  )
  local_substitution_safe = False
  hoist_scalar_safe_but_not_oracle = consumer_coupling and len(pairs) == 8
  coalesced_needs_dataflow = consumer_coupling and hasattr(isa, "global_load_b128")
  gates = {
    "dnr3a_plan_present": dnr3a.get("gate_pass") is True,
    "dnr3b_generic_emitter_blocked_as_expected": dnr3b.get("verdict") == "BLOCKED_DNR3B_GENERIC_EMITTER_CORRECT_BUT_NOT_ORACLE_SHAPED",
    "global_load_b128_opcode_available": hasattr(isa, "global_load_b128"),
    "native_grouped_global_load_is_22": native_grouped.get("global_load") == 22,
    "oracle_grouped_global_load_is_11": oracle_grouped.get("global_load") == 11,
    "eight_scalar_loop_pairs_found": len(pairs) == 8,
    "consumer_coupling_proven": consumer_coupling,
    "local_b32_to_b128_substitution_safe": local_substitution_safe,
    "performance_claim": False,
  }
  verdict = (
    "BLOCKED_DNR3C1_LOAD_SHAPE_NEEDS_REGISTER_DATAFLOW_EMITTER"
    if all(v for k, v in gates.items() if k not in ("local_b32_to_b128_substitution_safe", "performance_claim")) and not local_substitution_safe
    else "BLOCKED_DNR3C1_LOAD_SHAPE_STREAM_EVIDENCE_INCOMPLETE"
  )
  result = {
    "date": "2026-06-20",
    "phase": "DNR-3C1_DECODE_LOAD_SHAPE_REWRITE_AUDIT",
    "schema": "decode_native_renderer_dnr3c1_load_shape_probe_v1",
    "verdict": verdict,
    "gate_pass": False,
    "default_behavior_changed": False,
    "performance_claim": False,
    "stream": {
      "instruction_count": len(insts),
      "native_grouped": native_grouped,
      "oracle_grouped": oracle_grouped,
      "oracle_key_load_shape": oracle_contract["key_load_shape"],
      "preheader_global_load_count": len(preheader_global_loads),
      "preheader_global_loads": preheader_global_loads,
      "scalar_loop_load_pair_count": len(pairs),
      "scalar_loop_load_pairs": pairs,
      "consumer_registers": {"q4_word": "v[8]", "q8_word": "v[9]", "accum": "v[4]", "sumq": "v[5]"},
    },
    "rewrite_options": [
      {
        "name": "local_b32_to_b128_substitution",
        "status": "blocked",
        "reason": "The current loop consumes and mutates v[8]/v[9] immediately after each scalar pair; replacing one load with b128 would define multiple VGPRs without remapping the eight dot4 consumers.",
      },
      {
        "name": "hoist_same_scalar_loads",
        "status": "possible_but_not_oracle_shaped" if hoist_scalar_safe_but_not_oracle else "not_proven",
        "reason": "It could preload the same eight q4/q8 words into distinct registers, but grouped global_load remains 22 and live ranges increase.",
      },
      {
        "name": "coalesced_b128_q4_q8_preload",
        "status": "requires_register_dataflow_emitter" if coalesced_needs_dataflow else "blocked_opcode_missing",
        "reason": "The opcode exists, but correctness requires allocating multiword load destinations, remapping each dot4 input, preserving q4 nibble select mutation, and updating wait/live-range policy.",
      },
    ],
    "gates": gates,
    "blocked_at": {
      "next_phase": "DNR-3C2 register/dataflow emitter for decode load shape",
      "reason": "The first oracle-shape rewrite is not a local scheduler edit; it changes producer registers, consumer operands, wait placement, and live ranges.",
      "minimum_unblock": [
        "build a decode dataflow object for the eight q4/q8 dot4 lanes",
        "allocate distinct VGPR ranges for coalesced q4 and q8 preload results",
        "rewrite q4 unpack/select operations to consume per-lane preloaded registers without clobbering later lanes",
        "rewrite q8 dot4 operands to the matching preloaded registers",
        "emit waits from the new producer-consumer edges",
        "run synthetic gate/up correctness before adding markers, branches, or LDS/reduction edits",
      ],
    },
    "input_artifacts": [
      "extra/q8_ffn_asm_fullrow_reduce.py",
      "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3a_scheduler_resource_plan_result.json",
      "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3b_compound_emitter_result.json",
      "bench/q8-ffn-amd-scheduler-project/oracle_contract.json",
    ],
    "next_action": "Implement DNR-3C2 as a small register/dataflow emitter; do not tune s_clause/s_delay_alu or BEAM before this pass exists.",
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "gates": gates,
    "native_grouped_global_load": native_grouped.get("global_load"),
    "oracle_grouped_global_load": oracle_grouped.get("global_load"),
    "scalar_loop_load_pairs": len(pairs),
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
