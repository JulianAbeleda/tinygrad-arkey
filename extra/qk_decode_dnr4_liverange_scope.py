#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import re
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
ORACLE_DISASM = ROOT / "bench/qk-decode-primitive-transfer/oracle/q8_mmvq_gateup.disasm.txt"
NATIVE_LEDGER = ROOT / "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7a_resource_ledger_result.json"
COARSE = ROOT / "bench/qk-decode-primitive-transfer/decode_oracle_coarse_attribution_result.json"
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_dnr4_liverange_scope_result.json"


STAGES = [
  ("S0_setup_bounds_addresses", 0x1600, 0x16cf),
  ("S1_scale_min_byte_select", 0x16d0, 0x177f),
  ("S2_q4_vector_load_prefetch", 0x1780, 0x17ff),
  ("S3_interleaved_unpack_dot4_scale", 0x1800, 0x1983),
  ("S4_cross_lane_reduce_partial", 0x1984, 0x1a63),
  ("S5_final_writeback", 0x1a64, 0x1ad4),
]


def load(path: pathlib.Path, default: Any = None) -> Any:
  return json.loads(path.read_text()) if path.exists() else default


def regs_from_line(line: str) -> set[int]:
  regs: set[int] = set()
  for a, b in re.findall(r"v\[(\d+):(\d+)\]", line):
    regs.update(range(int(a), int(b) + 1))
  for a in re.findall(r"(?<![A-Za-z0-9_\[])v(\d+)\b", line):
    regs.add(int(a))
  return regs


def oracle_stage_regs() -> list[dict[str, Any]]:
  text = ORACLE_DISASM.read_text()
  rows: list[dict[str, Any]] = []
  for name, lo, hi in STAGES:
    regs: set[int] = set()
    insts = 0
    for line in text.splitlines():
      m = re.search(r"//\s*([0-9a-fA-F]+):", line)
      if not m:
        continue
      pc = int(m.group(1), 16)
      if lo <= pc <= hi:
        insts += 1
        regs |= regs_from_line(line)
    rows.append({
      "stage": name,
      "pc_range": f"0x{lo:x}-0x{hi:x}",
      "instruction_count": insts,
      "unique_vgpr": len(regs),
      "max_vgpr_index": max(regs) if regs else None,
      "vgpr_set": sorted(regs),
    })
  return rows


def main() -> int:
  native = load(NATIVE_LEDGER, {})
  coarse = load(COARSE, {})
  oracle_rows = oracle_stage_regs()
  native_regs = (((native.get("native") or {}).get("registers") or {}))
  native_desc = ((native.get("native") or {}).get("descriptor") or {})

  oracle_max = max(row["max_vgpr_index"] for row in oracle_rows if row["max_vgpr_index"] is not None)
  native_alloc = native_desc.get("allocated_vgpr_per_workitem")
  native_unique = native_regs.get("unique_vgpr_count")
  native_bands = native_regs.get("vgpr_bands")
  native_phase_pressure = native_regs.get("vgpr_phase_pressure")

  compression_targets = [
    {
      "id": "DNR4-T1-reduction-band-reuse",
      "evidence": "Oracle S4/S5 reduction uses v0-v6/v0-v4 after the dot body; native keeps a separate v50-v54 reduction/store band.",
      "required_change": "Make native reduction/tail reuse dead dot/address registers instead of reserving high v50-v54 temporaries.",
      "gate": "allocated VGPR decreases from 56 toward <=40 without correctness regression.",
    },
    {
      "id": "DNR4-T2-dot-body-vector-band-compression",
      "evidence": "Oracle S2/S3 stays within v0-v25 while native q4_q8_dot_body spans disjoint bands through v37.",
      "required_change": "Pack q4 vector payload, scale/min selectors, and accumulators into one low-register lane-local schedule.",
      "gate": "S3-equivalent native body max VGPR index <=31 and 16 dot4 preserved.",
    },
    {
      "id": "DNR4-T3-live-interval-expiry-check",
      "evidence": "Native ledger has long accumulator/reduction spans crossing q4 body into tail; oracle explicitly restarts low registers for reduction/writeback.",
      "required_change": "Add a per-stage live interval assertion before timing any new candidate.",
      "gate": "probe can prove which registers are dead at S3->S4 and S4->S5 boundaries.",
    },
  ]

  gates = {
    "coarse_attribution_passed": coarse.get("gate_pass") is True,
    "oracle_max_vgpr_index_25": oracle_max == 25,
    "native_allocated_vgpr_known": isinstance(native_alloc, int),
    "native_allocated_vgpr_above_oracle": isinstance(native_alloc, int) and native_alloc > 32,
    "native_high_reduction_band_present": bool(native_bands) and [50, 54] in native_bands,
    "compression_targets_named": len(compression_targets) == 3,
    "no_perf_claim": True,
  }

  result = {
    "date": "2026-06-20",
    "phase": "DNR4_RESOURCE_LIVERANGE_SCOPE",
    "schema": "decode_dnr4_liverange_scope_v1",
    "verdict": "PASS_DNR4_RESOURCE_LIVERANGE_SCOPE_READY" if all(gates.values()) else "BLOCKED_DNR4_RESOURCE_LIVERANGE_SCOPE_INCOMPLETE",
    "gate_pass": all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "oracle_stage_vgpr": oracle_rows,
    "oracle_summary": {
      "max_vgpr_index": oracle_max,
      "profiled_vgpr_count": (((coarse.get("resource_delta") or {}).get("oracle_trace_vgpr"))),
      "metadata_vgpr_count": (((coarse.get("resource_delta") or {}).get("oracle_metadata_vgpr"))),
    },
    "native_summary": {
      "allocated_vgpr_per_workitem": native_alloc,
      "unique_vgpr_count": native_unique,
      "vgpr_bands": native_bands,
      "vgpr_phase_pressure": native_phase_pressure,
    },
    "compression_targets": compression_targets,
    "decision": {
      "next": "Implement DNR4-T1 as a structural/live-range candidate first; do not tune issue order until allocated VGPR and stage liveness are measured.",
      "promotion_gate": "correctness plus same-run timing movement; <=40 allocated VGPR is a structural gate, not a performance claim.",
      "blocked_without": "a native emitter variant that can intentionally reuse reduction/tail registers and dump live intervals.",
    },
    "gates": gates,
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "gates": gates,
    "oracle_max_vgpr_index": oracle_max,
    "native_allocated_vgpr": native_alloc,
    "targets": [row["id"] for row in compression_targets],
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())

