#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tinygrad import Tensor
from tinygrad.dtype import dtypes
from tinygrad.renderer.amd.dsl import FixedBitField, Reg
from tinygrad.renderer.amd.elf import assemble_linear, kernel_descriptor_from_elf

from extra.q8_ffn_asm_fullrow_reduce import HIDDEN, Q4_WORDS, Q8_BYTES, build_fullrow_reduce
from extra.qk_decode_native_renderer_dnr3b_compound_emitter_probe import grouped, inst_name, insts_from_program
from extra.qk_decode_native_renderer_dnr3c4_semantic_reduction_probe import build_dnr3c4_candidate


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7a_resource_ledger_result.json"


def read_json(rel: str) -> dict[str, Any]:
  with (ROOT / rel).open() as f:
    return json.load(f)


def compact_bands(vals: list[int]) -> list[list[int]]:
  if not vals: return []
  bands: list[list[int]] = []
  start = prev = vals[0]
  for val in vals[1:]:
    if val == prev + 1:
      prev = val
      continue
    bands.append([start, prev])
    start = prev = val
  bands.append([start, prev])
  return bands


def reg_uses(insts: list[Any]) -> dict[str, dict[int, dict[str, Any]]]:
  ret: dict[str, dict[int, dict[str, Any]]] = {"vgpr": {}, "sgpr": {}}
  for idx, inst in enumerate(insts):
    for name, field in inst._fields:
      if isinstance(field, FixedBitField): continue
      val = getattr(inst, name)
      if not isinstance(val, Reg): continue
      if 256 <= val.offset < 512:
        kind, base = "vgpr", val.offset - 256
      elif val.offset < 106:
        kind, base = "sgpr", val.offset
      else:
        continue
      for reg in range(base, base + val.sz):
        row = ret[kind].setdefault(reg, {"first": idx, "last": idx, "uses": 0, "ops": set()})
        row["first"] = min(row["first"], idx)
        row["last"] = max(row["last"], idx)
        row["uses"] += 1
        row["ops"].add(inst_name(inst))
  for table in ret.values():
    for row in table.values(): row["ops"] = sorted(row["ops"])
  return ret


def phase(idx: int) -> str:
  if idx < 58: return "address_scale_preheader"
  if idx < 135: return "q4_q8_dot_body"
  if idx < 178: return "wave_and_cross_wave_reduce"
  return "store_tail"


def phase_pressure(uses: dict[int, dict[str, Any]]) -> dict[str, Any]:
  phases = ("address_scale_preheader", "q4_q8_dot_body", "wave_and_cross_wave_reduce", "store_tail")
  rows: dict[str, set[int]] = {p: set() for p in phases}
  for reg, span in uses.items():
    for idx in range(span["first"], span["last"] + 1):
      rows[phase(idx)].add(reg)
  return {p: {"count": len(vals), "bands": compact_bands(sorted(vals))} for p, vals in rows.items()}


def high_value_spans(uses: dict[int, dict[str, Any]], regs: list[int]) -> list[dict[str, Any]]:
  rows = []
  for reg in regs:
    if reg not in uses: continue
    span = uses[reg]
    rows.append({
      "reg": f"v[{reg}]",
      "first": span["first"],
      "last": span["last"],
      "span_len": span["last"] - span["first"] + 1,
      "first_phase": phase(span["first"]),
      "last_phase": phase(span["last"]),
      "uses": span["uses"],
      "ops": span["ops"][:8],
    })
  return rows


def build_program(fxn: Any) -> Any:
  gate = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  up = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  gate_words = Tensor.empty(Q4_WORDS, dtype=dtypes.uint32, device="AMD").contiguous()
  up_words = Tensor.empty(Q4_WORDS, dtype=dtypes.uint32, device="AMD").contiguous()
  q8 = Tensor.empty(Q8_BYTES, dtype=dtypes.uint8, device="AMD").contiguous()
  return fxn(gate.uop, up.uop, gate_words.uop, up_words.uop, q8.uop)


def resource_row(name: str, fxn: Any) -> dict[str, Any]:
  program = build_program(fxn)
  insts = insts_from_program(program)
  binary = assemble_linear(program, program.src[2], "gfx1100")
  desc = kernel_descriptor_from_elf(binary)
  uses = reg_uses(insts)
  vgprs = sorted(uses["vgpr"])
  sgprs = sorted(uses["sgpr"])
  allocated_vgpr = ((int(desc.compute_pgm_rsrc1) & 0x3f) + 1) * 8
  allocated_sgpr = ((((int(desc.compute_pgm_rsrc1) >> 6) & 0xf) + 1) * 8) if ((int(desc.compute_pgm_rsrc1) >> 6) & 0xf) else None
  return {
    "name": name,
    "instruction_count": len(insts),
    "grouped": grouped(insts),
    "registers": {
      "max_vgpr": max(vgprs) + 1 if vgprs else 0,
      "max_sgpr": max(sgprs) + 1 if sgprs else 0,
      "unique_vgpr_count": len(vgprs),
      "unique_sgpr_count": len(sgprs),
      "vgpr_bands": compact_bands(vgprs),
      "sgpr_bands": compact_bands(sgprs),
      "vgpr_phase_pressure": phase_pressure(uses["vgpr"]),
      "sgpr_phase_pressure": phase_pressure(uses["sgpr"]),
      "preload_band_spans": high_value_spans(uses["vgpr"], list(range(80, 96))),
      "accumulator_and_reduction_spans": high_value_spans(uses["vgpr"], [4, 5, 10, 11, 12, 13, 50, 51, 52, 53, 54]),
    },
    "descriptor": {
      "group_segment_fixed_size": int(desc.group_segment_fixed_size),
      "private_segment_fixed_size": int(getattr(desc, "private_segment_fixed_size", 0)),
      "kernarg_size": int(desc.kernarg_size),
      "compute_pgm_rsrc1": int(desc.compute_pgm_rsrc1),
      "compute_pgm_rsrc2": int(desc.compute_pgm_rsrc2),
      "allocated_vgpr_per_workitem": allocated_vgpr,
      "allocated_sgpr_estimate": allocated_sgpr,
    },
    "launch": {
      "local_size": [128, 1, 1],
      "wave_size": 32,
      "waves_per_workgroup": 4,
    },
  }


def occupancy_estimate(allocated_vgpr: int) -> dict[str, Any]:
  # RDNA3 public occupancy details depend on exact CU/SIMD resources. Keep this as a sensitivity estimate, not a claim.
  return {
    "model": "sensitivity_estimate_not_hardware_counter",
    "wave32_max_waves_per_simd": 16,
    "if_vgpr_slots_per_simd_1536": min(16, 1536 // max(1, allocated_vgpr)),
    "if_vgpr_slots_per_simd_1024": min(16, 1024 // max(1, allocated_vgpr)),
  }


def main() -> int:
  dnr3c6 = read_json("bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c6_attribution_scope_result.json")
  dnr3c8 = read_json("bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c8_tooling_inventory_result.json")
  loader = read_json("bench/q8-ffn-amd-scheduler-project/artifact_loader.json")
  oracle = read_json("bench/q8-ffn-amd-scheduler-project/oracle_contract.json")

  native = resource_row("native_dnr2", build_fullrow_reduce)
  c4 = resource_row("dnr3c4_static_shape", build_dnr3c4_candidate)
  best_name = dnr3c6["timing_context"]["best_variant"]
  native_alloc, c4_alloc = native["descriptor"]["allocated_vgpr_per_workitem"], c4["descriptor"]["allocated_vgpr_per_workitem"]
  local_static_explained_us = dnr3c6["timing_context"]["best_static_improvement_us"]
  remaining_gap_us = dnr3c6["timing_context"]["best_variant_us"] - dnr3c6["timing_context"]["oracle_us"]

  comparisons = {
    "allocated_vgpr_delta": c4_alloc - native_alloc,
    "allocated_vgpr_ratio": c4_alloc / native_alloc if native_alloc else None,
    "unique_vgpr_delta": c4["registers"]["unique_vgpr_count"] - native["registers"]["unique_vgpr_count"],
    "private_segment_same_zero": native["descriptor"]["private_segment_fixed_size"] == c4["descriptor"]["private_segment_fixed_size"] == 0,
    "lds_same": native["descriptor"]["group_segment_fixed_size"] == c4["descriptor"]["group_segment_fixed_size"] == 16,
    "launch_shape_same": native["launch"] == c4["launch"],
    "best_static_variant": best_name,
    "local_static_explained_us": local_static_explained_us,
    "remaining_gap_us": remaining_gap_us,
  }

  gates = {
    "dnr3c8_inventory_ready": dnr3c8.get("gate_pass") is True,
    "native_resource_extracted": native["descriptor"]["allocated_vgpr_per_workitem"] > 0,
    "c4_resource_extracted": c4["descriptor"]["allocated_vgpr_per_workitem"] > 0,
    "private_spill_absent": comparisons["private_segment_same_zero"],
    "c4_vgpr_pressure_higher": c4_alloc > native_alloc,
    "resource_ledger_names_possible_risk": comparisons["allocated_vgpr_ratio"] is not None and comparisons["allocated_vgpr_ratio"] >= 1.5,
    "resource_ledger_names_30us_cause": False,
    "oracle_vgpr_sgpr_known": False,
  }
  verdict = (
    "PASS_DNR3C7A_RESOURCE_LEDGER_BUILT_BLOCKED_ON_PMC_AND_ORACLE_RESOURCE_GAPS"
    if all(v for k, v in gates.items() if k not in ("resource_ledger_names_30us_cause", "oracle_vgpr_sgpr_known")) else
    "BLOCKED_DNR3C7A_RESOURCE_LEDGER_INCOMPLETE"
  )

  result = {
    "date": "2026-06-20",
    "phase": "DNR-3C7A_DECODE_STATIC_RESOURCE_LEDGER",
    "schema": "decode_native_renderer_dnr3c7a_resource_ledger_v1",
    "verdict": verdict,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": False,
    "native": {**native, "occupancy_estimate": occupancy_estimate(native_alloc)},
    "dnr3c4": {**c4, "occupancy_estimate": occupancy_estimate(c4_alloc)},
    "oracle_partial": {
      "runtime_name": loader["loader"]["gateup"]["runtime_name"],
      "global_size": loader["loader"]["gateup"]["global_size"],
      "local_size": loader["loader"]["gateup"]["local_size"],
      "group_segment_size": loader["loader"]["gateup"]["group_segment_size"],
      "private_segment_size": loader["loader"]["gateup"]["private_segment_size"],
      "kernarg_size": loader["loader"]["gateup"]["kernarg_size"],
      "grouped": oracle["instruction_contract"]["oracle_grouped"],
      "missing": ["VGPR count", "SGPR count", "live intervals", "occupancy estimate from artifact metadata"],
    },
    "comparison": comparisons,
    "interpretation": {
      "what_it_rules_out": [
        "native and DNR-3C4 do not spill private/scratch",
        "native and DNR-3C4 use the same tiny 16-byte LDS allocation",
        "launch wave shape is not changed by the static C4 rewrite",
      ],
      "what_it_suggests": [
        "DNR-3C4 raises allocated VGPR per workitem from 56 to 96 via the v[80:95] preload band",
        "higher VGPR pressure is a plausible reason the static load/LDS count win stays small",
      ],
      "why_it_is_not_enough": [
        "the best static variant still improves native by only single-digit microseconds",
        "oracle VGPR/SGPR/live-range data is missing",
        "no counter evidence yet connects VGPR pressure, memory wait, or issue occupancy to the remaining gap",
      ],
    },
    "gates": gates,
    "blocked_at": {
      "next_phase": "DNR-3C7B same-harness PMC counter ladder",
      "reason": "Static resource ledger finds a plausible VGPR-pressure risk but not a proven >=30us cause.",
      "minimum_unblock": [
        "run native vs DNR-3C4/best-static PMC with SQ/GL2C/SQC counters",
        "compare memory wait/cache/VALU/SALU/LDS-conflict direction",
        "decide whether remaining gap is resource/issue-bound or route should stay parked",
      ],
    },
    "input_artifacts": [
      "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c6_attribution_scope_result.json",
      "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c8_tooling_inventory_result.json",
      "bench/q8-ffn-amd-scheduler-project/artifact_loader.json",
      "bench/q8-ffn-amd-scheduler-project/oracle_contract.json",
    ],
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "native_alloc_vgpr": native_alloc,
    "dnr3c4_alloc_vgpr": c4_alloc,
    "comparison": comparisons,
    "gates": gates,
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
