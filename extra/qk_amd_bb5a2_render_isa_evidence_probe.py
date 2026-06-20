#!/usr/bin/env python3
from __future__ import annotations

import hashlib, json, pathlib, sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.renderer.amd.elf import assemble_linear, group_segment_fixed_size_from_elf
from tinygrad.renderer.amd.schedule import AMDLDSLoweredSlot
from tinygrad.runtime.autogen.amd.rdna3.ins import s_endpgm
from tinygrad.uop.ops import KernelInfo, Ops, UOp

OUT = ROOT / "bench/amd-broad-backend-roadmap"


def read_json(rel: str, default: Any = None) -> Any:
  path = ROOT / rel
  if not path.exists(): return default
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def lowered_slots(layer2: dict[str, Any]) -> list[AMDLDSLoweredSlot]:
  rows = ((layer2.get("lds_lowering") or {}).get("lowered_slots") or [])
  return [AMDLDSLoweredSlot(
    slot=int(row["slot"]),
    define_local_slot=int(row["define_local_slot"]),
    offset_bytes=int(row["offset_bytes"]),
    size_bytes=int(row["size_bytes"]),
    element_count=int(row["element_count"]),
    dtype=str(row["dtype"]),
    addrspace=str(row["addrspace"]),
    roles=tuple(str(x) for x in row["roles"]),
    lowering_status=str(row.get("lowering_status", "lowered_define_local")),
  ) for row in rows]


def make_program(name: str, slots: list[AMDLDSLoweredSlot]) -> UOp:
  locals_ = tuple(UOp.placeholder((row.element_count,), dtypes.half, row.define_local_slot, AddrSpace.LOCAL) for row in slots)
  sink = UOp.sink(*locals_, arg=KernelInfo(name=name))
  lin = UOp(Ops.LINEAR, src=(UOp(Ops.INS, arg=s_endpgm()),))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), lin))


def elf_evidence(prg: UOp, arch: str) -> dict[str, Any]:
  binary = assemble_linear(prg, prg.src[2], arch)
  inst_bytes = b"".join(u.arg.to_bytes() for u in prg.src[2].src)
  return {
    "elf_sha256": hashlib.sha256(binary).hexdigest(),
    "instruction_sha256": hashlib.sha256(inst_bytes).hexdigest(),
    "elf_size": len(binary),
    "instruction_bytes": len(inst_bytes),
    "group_segment_fixed_size": group_segment_fixed_size_from_elf(binary),
    "define_local_slots": [u.arg for u in prg.src[0].src if u.op is Ops.DEFINE_LOCAL],
  }


def main() -> int:
  layer2 = read_json("bench/amd-broad-backend-roadmap/bb5a2_lowering_hook_result.json", {})
  slots = lowered_slots(layer2)
  candidate = make_program("bb5a2_two_slot_lds_candidate", slots)
  # Serialized baseline is the current single LDS planning unit before the second stage is materialized.
  baseline = make_program("bb5a2_single_slot_lds_baseline", [slots[0]]) if slots else make_program("bb5a2_empty_baseline", [])
  arch = "gfx1100"
  candidate_ev = elf_evidence(candidate, arch)
  baseline_ev = elf_evidence(baseline, arch)
  required_local_bytes = ((layer2.get("lds_lowering") or {}).get("summary") or {}).get("required_local_bytes")
  gate = {
    "input_lowering_hook_pass": layer2.get("verdict") == "PASS_DEFINE_LOCAL_LOWERING_HOOK" and bool(layer2.get("gate_pass")),
    "candidate_has_two_define_local_slots": len(candidate_ev["define_local_slots"]) == 2,
    "candidate_elf_lds_bytes_match_plan": candidate_ev["group_segment_fixed_size"] == required_local_bytes,
    "baseline_elf_lds_bytes_is_single_slot": bool(slots) and baseline_ev["group_segment_fixed_size"] == slots[0].size_bytes,
    "elf_hash_non_byte_identical": candidate_ev["elf_sha256"] != baseline_ev["elf_sha256"],
    "instruction_stream_identical": candidate_ev["instruction_sha256"] == baseline_ev["instruction_sha256"],
    "default_behavior_changed": False,
    "performance_claim": False,
  }
  positive = [
    "input_lowering_hook_pass", "candidate_has_two_define_local_slots", "candidate_elf_lds_bytes_match_plan",
    "baseline_elf_lds_bytes_is_single_slot", "elf_hash_non_byte_identical",
  ]
  gate_pass = all(gate[x] for x in positive) and not gate["default_behavior_changed"] and not gate["performance_claim"]
  result = {
    "date": "2026-06-19",
    "phase": "BB-5a.2_layer_3_render_elf_evidence",
    "schema": "amd_bb5a2_render_isa_evidence_result_v1",
    "verdict": "PASS_RENDER_ELF_LDS_EVIDENCE" if gate_pass else "FAIL_RENDER_ELF_LDS_EVIDENCE",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "arch": arch,
    "input_artifact": "bench/amd-broad-backend-roadmap/bb5a2_lowering_hook_result.json",
    "candidate": candidate_ev,
    "baseline": baseline_ev,
    "gate": gate,
    "decision": (
      "Layer 3 passes as AMD ELF descriptor evidence: the two-slot DEFINE_LOCAL candidate changes the assembled ELF "
      "and group_segment_fixed_size. The instruction stream remains intentionally identical, so full BB-5a.2 still "
      "needs integration into the real lowering/render path before claiming pipelined ISA movement."
    ),
    "next_action": "Integrate the gated LDS plan/lowering path into real postrange or AMD renderer lowering.",
  }
  write_json("bb5a2_render_isa_evidence_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a2_render_isa_evidence_result.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "candidate_lds": candidate_ev["group_segment_fixed_size"],
    "baseline_lds": baseline_ev["group_segment_fixed_size"],
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
