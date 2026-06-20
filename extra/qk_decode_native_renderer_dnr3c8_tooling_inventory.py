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
from extra.qk_decode_native_renderer_dnr3b_compound_emitter_probe import grouped, insts_from_program
from extra.qk_decode_native_renderer_dnr3c4_semantic_reduction_probe import build_dnr3c4_candidate


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c8_tooling_inventory_result.json"


def read_json(rel: str) -> dict[str, Any]:
  with (ROOT / rel).open() as f:
    return json.load(f)


def scan_regs(insts: list[Any]) -> dict[str, Any]:
  max_vgpr = max_sgpr = max_accvgpr = 0
  regs: dict[str, set[int]] = {"vgpr": set(), "sgpr": set(), "accvgpr": set()}
  for inst in insts:
    for name, field in inst._fields:
      if isinstance(field, FixedBitField): continue
      val = getattr(inst, name)
      if not isinstance(val, Reg): continue
      if 256 <= val.offset < 512:
        # The decode kernels here do not use accvgpr encodings; keep the bucket for matrix parity.
        regs["vgpr"].update(range(val.offset - 256, val.offset - 256 + val.sz))
        max_vgpr = max(max_vgpr, val.offset - 256 + val.sz)
      elif val.offset < 106:
        regs["sgpr"].update(range(val.offset, val.offset + val.sz))
        max_sgpr = max(max_sgpr, val.offset + val.sz)
  return {
    "max_vgpr": max_vgpr,
    "max_sgpr": max_sgpr,
    "max_accvgpr": max_accvgpr,
    "unique_vgpr_count": len(regs["vgpr"]),
    "unique_sgpr_count": len(regs["sgpr"]),
    "vgpr_bands": compact_bands(sorted(regs["vgpr"])),
    "sgpr_bands": compact_bands(sorted(regs["sgpr"])),
  }


def compact_bands(vals: list[int]) -> list[list[int]]:
  if not vals: return []
  bands: list[list[int]] = []
  start = prev = vals[0]
  for v in vals[1:]:
    if v == prev + 1:
      prev = v
      continue
    bands.append([start, prev])
    start = prev = v
  bands.append([start, prev])
  return bands


def program_resource(name: str, fxn: Any) -> dict[str, Any]:
  gate = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  up = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  gate_words = Tensor.empty(Q4_WORDS, dtype=dtypes.uint32, device="AMD").contiguous()
  up_words = Tensor.empty(Q4_WORDS, dtype=dtypes.uint32, device="AMD").contiguous()
  q8 = Tensor.empty(Q8_BYTES, dtype=dtypes.uint8, device="AMD").contiguous()
  program = fxn(gate.uop, up.uop, gate_words.uop, up_words.uop, q8.uop)
  insts = insts_from_program(program)
  binary = assemble_linear(program, program.src[2], "gfx1100")
  desc = kernel_descriptor_from_elf(binary)
  return {
    "name": name,
    "instruction_count": len(insts),
    "grouped": grouped(insts),
    "register_scan": scan_regs(insts),
    "kernel_descriptor": {
      "group_segment_fixed_size": int(desc.group_segment_fixed_size),
      "private_segment_fixed_size": int(getattr(desc, "private_segment_fixed_size", 0)),
      "kernarg_size": int(desc.kernarg_size),
      "compute_pgm_rsrc1": int(desc.compute_pgm_rsrc1),
      "compute_pgm_rsrc2": int(desc.compute_pgm_rsrc2),
    },
    "tool_status": "ready_for_native_programs",
  }


def main() -> int:
  dnr3c6 = read_json("bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c6_attribution_scope_result.json")
  dnr3c7 = read_json("bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7_issue_resource_scope_result.json")
  n1 = read_json("bench/q8-ffn-amd-scheduler-project/n1_attribution.json")
  oracle = read_json("bench/q8-ffn-amd-scheduler-project/oracle_contract.json")
  loader = read_json("bench/q8-ffn-amd-scheduler-project/artifact_loader.json")
  tooling = read_json("bench/amd-scheduler-tooling-backend/execution.json")

  native = program_resource("native_dnr2", build_fullrow_reduce)
  dnr3c4 = program_resource("dnr3c4_static_shape", build_dnr3c4_candidate)
  oracle_gateup = loader["loader"]["gateup"]

  tools = [
    {
      "tool": "static instruction grouping",
      "answers": "opcode class/count deltas for native, candidates, oracle",
      "status": "ready",
      "evidence": "DNR-3C1..C6 grouped counts",
      "gap": None,
    },
    {
      "tool": "same-harness timing ladder",
      "answers": "whether a candidate actually moves latency",
      "status": "ready",
      "evidence": "DNR-3C6 ladder refuted local static attribution",
      "gap": None,
    },
    {
      "tool": "native PROGRAM resource descriptor",
      "answers": "LDS/group segment, private/scratch, kernarg, rsrc flags for native-generated kernels",
      "status": "ready_for_native_programs",
      "evidence": "DNR-3C8 assemble_linear + kernel_descriptor_from_elf succeeded",
      "gap": "does not expose full oracle VGPR/SGPR occupancy unless artifact metadata/disasm provides it",
    },
    {
      "tool": "native register scanner",
      "answers": "max/unique VGPR and SGPR use in generated instruction streams",
      "status": "ready_for_native_programs",
      "evidence": "DNR-3C8 scans Reg operands for native and DNR-3C4",
      "gap": "not yet a true live-interval allocator timeline; no per-phase pressure peak",
    },
    {
      "tool": "oracle resource metadata",
      "answers": "oracle launch size, group/private segment, kernarg",
      "status": "partial",
      "evidence": "artifact_loader gateup metadata exists",
      "gap": "no oracle VGPR/SGPR/live-range metadata in current JSON",
    },
    {
      "tool": "PMC counters",
      "answers": "memory wait/cache/VALU/SALU/LDS-conflict direction when selected counters are available",
      "status": "partial_runnable",
      "evidence": "N1 says PMC runnable; Track T parsed PMC structurally",
      "gap": "not yet same-harness native-vs-DNR3C4-vs-oracle attribution; SQ_WAIT_ANY availability must be checked",
    },
    {
      "tool": "SQTT body timeline",
      "answers": "PC-level issue/stall timeline and interleaving",
      "status": "blocked",
      "evidence": "capture exists but local RDNA3 HCQ decode/body mapping unusable",
      "gap": "needs ROCprofiler/AQLprofile-compatible body packets or decoder repair",
    },
    {
      "tool": "issue/interleaving model",
      "answers": "whether loads/unpack/dot/scale/reduction overlap or serialize",
      "status": "missing",
      "evidence": "static count matching explains only ~10.7% of gap",
      "gap": "needs dependency-stage model plus resource/liveness objective",
    },
    {
      "tool": "search objective",
      "answers": "what BEAM/search should optimize before timing every candidate",
      "status": "blocked",
      "evidence": "local static shape similarity was refuted by DNR-3C6",
      "gap": "requires C7A/C7B attribution to define objective",
    },
  ]

  gates = {
    "dnr3c7_scope_ready": dnr3c7.get("gate_pass") is True,
    "native_descriptor_tool_ready": native["kernel_descriptor"]["private_segment_fixed_size"] == 0,
    "dnr3c4_descriptor_tool_ready": dnr3c4["kernel_descriptor"]["private_segment_fixed_size"] == 0,
    "oracle_launch_resource_partial": oracle_gateup.get("group_segment_size") == 16 and oracle_gateup.get("private_segment_size") == 0,
    "pmc_partial_available": n1["gate"].get("pmc_profile_runnable") is True,
    "sqtt_body_blocked_recorded": n1["gate"].get("sqtt_decode_usable") is False,
    "search_objective_blocked": dnr3c6.get("verdict") == "BLOCKED_DNR3C6_STATIC_LADDER_REFUTES_LOCAL_COUNT_ATTRIBUTION",
  }

  result = {
    "date": "2026-06-20",
    "phase": "DNR-3C8_DECODE_TOOLING_INVENTORY",
    "schema": "decode_native_renderer_dnr3c8_tooling_inventory_v1",
    "verdict": "SCOPE_DNR3C8_TOOLING_INVENTORY_READY_PARTIAL_ATTRIBUTION_TOOLS" if all(gates.values()) else "BLOCKED_DNR3C8_TOOLING_INPUTS_INCONSISTENT",
    "gate_pass": all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "resource_probe": {
      "native_dnr2": native,
      "dnr3c4": dnr3c4,
      "oracle_gateup_partial": {
        "global_size": oracle_gateup["global_size"],
        "local_size": oracle_gateup["local_size"],
        "kernarg_size": oracle_gateup["kernarg_size"],
        "group_segment_size": oracle_gateup["group_segment_size"],
        "private_segment_size": oracle_gateup["private_segment_size"],
        "grouped": oracle["instruction_contract"]["oracle_grouped"],
      },
    },
    "tools": tools,
    "tooling_decision": {
      "have_enough_to_continue_C7A": True,
      "have_enough_for_full_attribution": False,
      "next_tool_to_build": "DNR-3C7A native/C4/oracle resource ledger with live-range bands and occupancy estimate",
      "next_tool_to_run_after_C7A": "DNR-3C7B same-harness PMC native-vs-DNR3C4 counter ladder",
      "do_not_start": ["BEAM/search", "branch-count patches", "marker-count tuning", "native promotion"],
    },
    "gates": gates,
    "input_artifacts": [
      "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c6_attribution_scope_result.json",
      "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7_issue_resource_scope_result.json",
      "bench/q8-ffn-amd-scheduler-project/n1_attribution.json",
      "bench/q8-ffn-amd-scheduler-project/oracle_contract.json",
      "bench/q8-ffn-amd-scheduler-project/artifact_loader.json",
      "bench/amd-scheduler-tooling-backend/execution.json",
    ],
  }

  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "gate_pass": result["gate_pass"],
    "native_regs": native["register_scan"],
    "dnr3c4_regs": dnr3c4["register_scan"],
    "tool_status": {row["tool"]: row["status"] for row in tools},
    "next_tool": result["tooling_decision"]["next_tool_to_build"],
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
