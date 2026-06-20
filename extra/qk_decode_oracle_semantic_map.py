#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import re
from collections import Counter
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DISASM = ROOT / "bench/qk-decode-primitive-transfer/oracle/q8_mmvq_gateup.disasm.txt"
EXTRACT = ROOT / "bench/qk-decode-primitive-transfer/decode_oracle_gateup_extract_result.json"
CONTRACT = ROOT / "bench/q8-ffn-amd-scheduler-project/oracle_contract.json"
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_oracle_semantic_map_result.json"


STAGES = [
  {
    "id": "S0_setup_bounds_addresses",
    "pc_start": 0x1600,
    "pc_end": 0x16cf,
    "semantic": "load kernargs, derive row/lane ids, select gate/up base pointers, handle edge lanes",
    "native_question": "Does native spend extra issue slots deriving addresses or avoid oracle exec-mask branches?",
  },
  {
    "id": "S1_scale_min_byte_select",
    "pc_start": 0x16d0,
    "pc_end": 0x177f,
    "semantic": "load q4 scale/min bytes, unpack lane-local scale and min selectors",
    "native_question": "Does native reproduce the oracle's compact five-byte selector path or over-load unpack metadata?",
  },
  {
    "id": "S2_q4_vector_load_prefetch",
    "pc_start": 0x1780,
    "pc_end": 0x17ff,
    "semantic": "compute q4 block addresses and issue four b128 data loads plus scale/min payload loads",
    "native_question": "Does native overlap q4 vector loads with subsequent unpack/dot work, or serialize before dot4?",
  },
  {
    "id": "S3_interleaved_unpack_dot4_scale",
    "pc_start": 0x1800,
    "pc_end": 0x1983,
    "semantic": "interleave q4 nibble select, waitcnt ladder, 16 dot4 ops, q8/q4 scaling, and final fma",
    "native_question": "This is the central body objective: preserve oracle waitcnt-to-dot4 interleaving and avoid static similarity as the target.",
  },
  {
    "id": "S4_cross_lane_reduce_partial",
    "pc_start": 0x1984,
    "pc_end": 0x1a63,
    "semantic": "five ds_bpermute/add reduction steps, lane-0 partial store to LDS",
    "native_question": "Does native match the oracle's exact bpermute topology and LDS handoff count?",
  },
  {
    "id": "S5_final_writeback",
    "pc_start": 0x1a64,
    "pc_end": 0x1ad4,
    "semantic": "barrier, lane-0 ds_load_b128, four-float final reduce, one global store",
    "native_question": "Does native pay extra final-reduction or store overhead relative to oracle?",
  },
]


def load_json(path: pathlib.Path) -> dict[str, Any]:
  with path.open() as f:
    return json.load(f)


def parse_symbol_disasm(text: str, symbol: str = "q8_mmvq_gateup") -> list[dict[str, Any]]:
  in_symbol = False
  insts: list[dict[str, Any]] = []
  for line in text.splitlines():
    if re.search(rf"<{re.escape(symbol)}>:", line):
      in_symbol = True
      continue
    if in_symbol and re.search(r"^0*[0-9a-fA-F]+ <", line):
      break
    if not in_symbol:
      continue
    m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_.$]*)\b.*//\s*([0-9a-fA-F]+):", line)
    if not m:
      continue
    insts.append({"pc": int(m.group(2), 16), "mnemonic": m.group(1), "text": line.strip()})
  return insts


def group_counts(insts: list[dict[str, Any]]) -> dict[str, int]:
  c = Counter(row["mnemonic"] for row in insts)
  return {
    "dot4": sum(v for k, v in c.items() if "dot4" in k),
    "fma": sum(v for k, v in c.items() if k.startswith("v_fma") or k.startswith("v_mad") or "mad_mix" in k),
    "convert": sum(v for k, v in c.items() if k.startswith("v_cvt")),
    "valu": sum(v for k, v in c.items() if k.startswith("v_")),
    "salu": sum(v for k, v in c.items() if k.startswith("s_")),
    "ds": sum(v for k, v in c.items() if k.startswith("ds_")),
    "barrier": c["s_barrier"],
    "global_load": sum(v for k, v in c.items() if k.startswith("global_load")),
    "global_store": sum(v for k, v in c.items() if k.startswith("global_store")),
    "shuffle": c["ds_bpermute_b32"],
    "branch": sum(v for k, v in c.items() if "branch" in k),
    "waitcnt": c["s_waitcnt"],
    "s_clause": c["s_clause"],
    "s_delay_alu": c["s_delay_alu"],
  }


def stage_for(pc: int) -> dict[str, Any] | None:
  for stage in STAGES:
    if stage["pc_start"] <= pc <= stage["pc_end"]:
      return stage
  return None


def main() -> int:
  extract = load_json(EXTRACT)
  contract = load_json(CONTRACT)
  insts = parse_symbol_disasm(DISASM.read_text())
  stage_rows = []
  for stage in STAGES:
    rows = [row for row in insts if stage["pc_start"] <= row["pc"] <= stage["pc_end"]]
    stage_rows.append({
      **stage,
      "pc_range": f"0x{stage['pc_start']:x}-0x{stage['pc_end']:x}",
      "instruction_count": len(rows),
      "grouped": group_counts(rows),
      "top_mnemonics": Counter(row["mnemonic"] for row in rows).most_common(20),
    })

  uncovered = [row for row in insts if stage_for(row["pc"]) is None]
  total_grouped = group_counts(insts)
  oracle_grouped = contract["instruction_contract"]["oracle_grouped"]
  native_grouped = contract["instruction_contract"]["tinygrad_asm_grouped"]
  grouped_matches = {k: total_grouped.get(k) == oracle_grouped.get(k) for k in oracle_grouped}

  static_diffs = {
    k: {
      "oracle": oracle_grouped.get(k),
      "native": native_grouped.get(k),
      "delta_native_minus_oracle": (native_grouped.get(k, 0) - oracle_grouped.get(k, 0)),
    }
    for k in sorted(set(oracle_grouped) | set(native_grouped))
  }
  uncovered_is_padding = all(row["mnemonic"] == "s_code_end" and row["pc"] > STAGES[-1]["pc_end"] for row in uncovered)

  decision = {
    "resume_native_now": False,
    "reason": "OES-4 names the central body shape but does not attribute time. Native should remain parked until OES-5 joins PCs/stalls or a runtime launch delta is proven.",
    "first_dynamic_question": "Is the native gap dominated by S3 wait/load/dot/scale issue serialization, S4/S5 reduction handoff, or launch/runtime outside the body?",
    "next_phase": "OES-5 PC timeline and stall attribution",
  }

  gates = {
    "extraction_passed": extract.get("gate_pass") is True,
    "all_body_instructions_covered": len(uncovered) == 0 or uncovered_is_padding,
    "uncovered_only_trailing_code_end_padding": uncovered_is_padding,
    "six_stages_present": len(stage_rows) == 6,
    "dot4_all_in_s3": stage_rows[3]["grouped"]["dot4"] == 16 and total_grouped["dot4"] == 16,
    "reduction_topology_named": stage_rows[4]["grouped"]["shuffle"] == 5 and stage_rows[5]["grouped"]["ds"] == 1,
    "single_global_store_in_s5": stage_rows[5]["grouped"]["global_store"] == 1,
    "matches_oracle_contract": all(grouped_matches.values()),
    "native_reopen_blocked_on_timeline": decision["resume_native_now"] is False,
  }

  result = {
    "date": "2026-06-20",
    "phase": "DECODE_ORACLE_SEMANTIC_ISA_MAP",
    "schema": "decode_oracle_semantic_map_v1",
    "verdict": "PASS_DECODE_ORACLE_SEMANTIC_MAP_STATIC_COMPLETE_OES5_REQUIRED" if all(gates.values()) else "BLOCKED_DECODE_ORACLE_SEMANTIC_MAP_INCOMPLETE",
    "gate_pass": all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "artifact": extract["artifact"],
    "metadata": extract["metadata"],
    "total_instruction_count": len(insts),
    "total_grouped": total_grouped,
    "grouped_matches_oracle_contract": grouped_matches,
    "stages": stage_rows,
    "static_native_diff": static_diffs,
    "uncovered": uncovered[:20],
    "uncovered_count": len(uncovered),
    "decision": decision,
    "gates": gates,
  }

  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "gate_pass": result["gate_pass"],
    "stage_summary": [(row["id"], row["instruction_count"], row["grouped"]) for row in stage_rows],
    "decision": decision,
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
