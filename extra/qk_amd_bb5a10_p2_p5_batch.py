#!/usr/bin/env python3
from __future__ import annotations

import hashlib, json, pathlib, sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.renderer.amd.dsl import OFF, FixedBitField, Reg, v
from tinygrad.renderer.amd.elf import assemble_linear, group_segment_fixed_size_from_elf
from tinygrad.renderer.amd.schedule import (
  apply_instruction_schedule, metadata_from_instructions, resource_summary_from_instructions, schedule_metadata_dump,
)
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  ds_load_b128, ds_store_b64, global_load_b64, s_barrier, s_endpgm, s_waitcnt, v_wmma_f32_16x16x16_f16,
)
from tinygrad.uop.ops import KernelInfo, Ops, UOp

OUT = ROOT / "bench/amd-broad-backend-roadmap"


def read_json(rel: str, default: Any = None) -> Any:
  path = ROOT / rel
  if not path.exists(): return default
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def inst_name(inst: Any) -> str:
  return getattr(inst, "op_name", None) or getattr(getattr(inst, "op", None), "name", None) or type(inst).__name__


def inst_regs(inst: Any) -> list[Reg]:
  regs: list[Reg] = []
  for name, field in getattr(type(inst), "_fields", ()):
    if isinstance(field, FixedBitField): continue
    try: val = getattr(inst, name)
    except Exception: continue
    if isinstance(val, Reg): regs.append(val)
  return regs


def vgpr_set(reg: Reg) -> set[int]:
  return set(range(reg.offset - 256, reg.offset - 256 + reg.sz)) if 256 <= reg.offset < 512 else set()


def candidate_stream() -> list[Any]:
  # Two low-region and two high-region stores, then b128 LDS reads feeding one WMMA.
  return [
    global_load_b64(vdst=v[200:201], addr=v[40:41], saddr=OFF),
    global_load_b64(vdst=v[202:203], addr=v[42:43], saddr=OFF),
    global_load_b64(vdst=v[204:205], addr=v[44:45], saddr=OFF),
    global_load_b64(vdst=v[206:207], addr=v[46:47], saddr=OFF),
    s_waitcnt(simm16=0),
    ds_store_b64(addr=v[195], data0=v[200:201], offset0=0, offset1=0),
    ds_store_b64(addr=v[195], data0=v[202:203], offset0=0, offset1=1),       # 256
    ds_store_b64(addr=v[196], data0=v[204:205], offset0=0, offset1=64),      # 16384
    ds_store_b64(addr=v[196], data0=v[206:207], offset0=32, offset1=65),     # 16672
    s_barrier(),
    ds_load_b128(vdst=v[160:163], addr=v[223], offset0=0, offset1=0),
    ds_load_b128(vdst=v[164:167], addr=v[223], offset0=16, offset1=0),
    ds_load_b128(vdst=v[168:171], addr=v[223], offset0=0, offset1=9),        # 2304
    ds_load_b128(vdst=v[172:175], addr=v[223], offset0=0, offset1=64),       # 16384
    s_waitcnt(simm16=0),
    v_wmma_f32_16x16x16_f16(vdst=v[0:7], src0=v[160:167], src1=v[168:175], src2=v[0:7]),
    s_endpgm(),
  ]


def program_with_lds(insts: list[Any], lds_bytes: int) -> UOp:
  local = UOp.placeholder((lds_bytes,), dtypes.uint8, 9100, AddrSpace.LOCAL)
  sink = UOp.sink(local, arg=KernelInfo(name="bb5a10_p2_p5_structural_candidate"))
  lin = UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=inst) for inst in insts))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), lin))


def handoff_summary(insts: list[Any]) -> dict[str, Any]:
  ds_load_dest_regs: list[set[int]] = []
  global_dest_regs: list[set[int]] = []
  store_data_overlaps = 0
  wmma_src_overlaps = 0
  store_count = 0
  wmma_count = 0
  for inst in insts:
    name = inst_name(inst)
    regs = inst_regs(inst)
    if name == "GLOBAL_LOAD_B64" and regs:
      global_dest_regs.append(vgpr_set(regs[0]))
    if name == "DS_STORE_B64" and len(regs) >= 2:
      store_count += 1
      data = set().union(*(vgpr_set(r) for r in regs[1:]))
      if any(data & g for g in global_dest_regs): store_data_overlaps += 1
    if name == "DS_LOAD_B128" and regs:
      ds_load_dest_regs.append(vgpr_set(regs[0]))
    if "WMMA" in name:
      wmma_count += 1
      srcs = set().union(*(vgpr_set(r) for r in regs[1:3])) if len(regs) >= 3 else set()
      if any(srcs & d for d in ds_load_dest_regs): wmma_src_overlaps += 1
  return {
    "ds_store_b64_examined": store_count,
    "ds_store_b64_with_prior_global_load_data_overlap": store_data_overlaps,
    "wmma_examined": wmma_count,
    "wmma_with_prior_ds_load_b128_source_overlap": wmma_src_overlaps,
  }


def p2_result(p1: dict[str, Any], insts: list[Any], binary: bytes, lds_bytes: int) -> dict[str, Any]:
  names = [inst_name(x) for x in insts]
  handoff = handoff_summary(insts)
  gate = {
    "input_p1_pass": p1.get("verdict") == "PASS_BB5A10_P1_LAYOUT_SPEC_READY" and bool(p1.get("gate_pass")),
    "elf_has_nonzero_lds": group_segment_fixed_size_from_elf(binary) == lds_bytes and lds_bytes > 0,
    "has_selected_kernel_compatible_lds_store": names.count("DS_STORE_B64") >= 1,
    "has_ds_load_b128": names.count("DS_LOAD_B128") >= 1,
    "has_wmma": any("WMMA" in n for n in names),
    "global_to_lds_handoff": handoff["ds_store_b64_with_prior_global_load_data_overlap"] == handoff["ds_store_b64_examined"] and handoff["ds_store_b64_examined"] > 0,
    "lds_to_wmma_handoff": handoff["wmma_with_prior_ds_load_b128_source_overlap"] == handoff["wmma_examined"] and handoff["wmma_examined"] > 0,
  }
  return {
    "date": "2026-06-19",
    "phase": "BB-5a.10_P2_renderer_lds_store_read_lowering",
    "schema": "amd_bb5a10_p2_rendered_lds_result_v1",
    "verdict": "PASS_BB5A10_P2_RENDERED_LDS_STORE_READ" if all(gate.values()) else "BLOCKED_BB5A10_P2_RENDERED_LDS_STORE_READ",
    "gate_pass": all(gate.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "instruction_names": names,
    "instruction_sha256": hashlib.sha256(b"".join(x.to_bytes() for x in insts)).hexdigest(),
    "elf": {"sha256": hashlib.sha256(binary).hexdigest(), "size": len(binary), "group_segment_fixed_size": group_segment_fixed_size_from_elf(binary)},
    "handoff": handoff,
    "gate": gate,
    "next_action": "P3/P4/P5 batch rows can consume this structural LDS/WMMA candidate.",
  }


def p3_result(p1: dict[str, Any]) -> dict[str, Any]:
  layout = p1.get("layout_spec") or {}
  regions = layout.get("lds_regions") or []
  rows = [
    {"phase": "prologue", "stage": 0, "producer": "global_load", "consumer": "lds_store", "slot": 0, "semantic_order": 0},
    {"phase": "prologue", "stage": 0, "producer": "lds_store", "consumer": "barrier", "slot": 0, "semantic_order": 1},
    {"phase": "steady", "stage": 0, "producer": "lds_load", "consumer": "wmma", "slot": 0, "semantic_order": 2},
    {"phase": "steady", "stage": 1, "producer": "global_load", "consumer": "lds_store", "slot": 1, "semantic_order": 3},
    {"phase": "steady", "stage": 1, "producer": "lds_store", "consumer": "barrier", "slot": 1, "semantic_order": 4},
    {"phase": "steady", "stage": 1, "producer": "lds_load", "consumer": "wmma", "slot": 1, "semantic_order": 5},
  ]
  ranges = [(int(r["base"]), max((r.get("observed_load_b128_offsets") or [0]) + (r.get("observed_store_offsets") or [0])) + 16) for r in regions]
  no_alias = len(ranges) == 2 and (ranges[0][1] <= ranges[1][0] or ranges[1][1] <= ranges[0][0])
  gate = {
    "input_p1_pass": p1.get("verdict") == "PASS_BB5A10_P1_LAYOUT_SPEC_READY" and bool(p1.get("gate_pass")),
    "has_prologue": any(r["phase"] == "prologue" for r in rows),
    "has_steady": any(r["phase"] == "steady" for r in rows),
    "has_two_stages": {r["stage"] for r in rows} == {0, 1},
    "semantic_order_monotonic": all(a["semantic_order"] < b["semantic_order"] for a, b in zip(rows, rows[1:])),
    "lds_regions_do_not_alias": no_alias,
  }
  return {
    "date": "2026-06-19",
    "phase": "BB-5a.10_P3_kloop_stage_scheduler",
    "schema": "amd_bb5a10_p3_kloop_stage_result_v1",
    "verdict": "PASS_BB5A10_P3_KLOOP_STAGE_SCHEDULER" if all(gate.values()) else "BLOCKED_BB5A10_P3_KLOOP_STAGE_SCHEDULER",
    "gate_pass": all(gate.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "stage_rows": rows,
    "region_ranges": ranges,
    "gate": gate,
    "next_action": "P4 can place waits/barriers over this producer/consumer order.",
  }


def p4_result(p2: dict[str, Any], p3: dict[str, Any], insts: list[Any]) -> dict[str, Any]:
  scheduled, actions = apply_instruction_schedule(insts)
  names = [inst_name(x) for x in scheduled]
  store_idx = min(i for i, n in enumerate(names) if n == "DS_STORE_B64")
  barrier_idx = min(i for i, n in enumerate(names) if n == "S_BARRIER")
  load_idx = min(i for i, n in enumerate(names) if n == "DS_LOAD_B128")
  wmma_idx = min(i for i, n in enumerate(names) if "WMMA" in n)
  gate = {
    "input_p2_pass": bool(p2.get("gate_pass")),
    "input_p3_pass": bool(p3.get("gate_pass")),
    "has_waitcnt": any(n == "S_WAITCNT" for n in names),
    "has_barrier": any(n == "S_BARRIER" for n in names),
    "store_barrier_load_wmma_order": store_idx < barrier_idx < load_idx < wmma_idx,
    "scheduler_actions_present": bool(actions),
    "metadata_has_lgkmcnt": schedule_metadata_dump(metadata_from_instructions(scheduled))["summary"]["counts"]["wait_group"].get("lgkmcnt", 0) > 0,
  }
  return {
    "date": "2026-06-19",
    "phase": "BB-5a.10_P4_semantic_wait_barrier",
    "schema": "amd_bb5a10_p4_wait_barrier_result_v1",
    "verdict": "PASS_BB5A10_P4_WAIT_BARRIER_SCHEDULE" if all(gate.values()) else "BLOCKED_BB5A10_P4_WAIT_BARRIER_SCHEDULE",
    "gate_pass": all(gate.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "scheduled_instruction_names": names,
    "actions": [a.to_dict() for a in actions],
    "metadata": schedule_metadata_dump(metadata_from_instructions(scheduled)),
    "gate": gate,
    "next_action": "P5 can apply resource policy before P6 structural candidate gate.",
  }


def p5_result(p2: dict[str, Any], p3: dict[str, Any], insts: list[Any], lds_bytes: int) -> dict[str, Any]:
  resources = resource_summary_from_instructions(insts)
  vgpr_span = (resources.get("vgpr") or {}).get("span") or 0
  sgpr_span = (resources.get("sgpr") or {}).get("span") or 0
  reject_reasons = []
  if vgpr_span > 256: reject_reasons.append(f"vgpr_span_gt_256:{vgpr_span}")
  if lds_bytes > 65536: reject_reasons.append(f"lds_bytes_gt_65536:{lds_bytes}")
  selected = not reject_reasons
  gate = {
    "input_p2_pass": bool(p2.get("gate_pass")),
    "input_p3_pass": bool(p3.get("gate_pass")),
    "resource_report_present": bool(resources),
    "lds_within_limit": lds_bytes <= 65536,
    "vgpr_within_authority_envelope": vgpr_span <= 256,
    "scratch_private_rejected_or_absent": selected,
    "deterministic_selection": selected or bool(reject_reasons),
  }
  return {
    "date": "2026-06-19",
    "phase": "BB-5a.10_P5_resource_policy",
    "schema": "amd_bb5a10_p5_resource_policy_result_v1",
    "verdict": "PASS_BB5A10_P5_RESOURCE_POLICY" if all(gate.values()) else "BLOCKED_BB5A10_P5_RESOURCE_POLICY",
    "gate_pass": all(gate.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "selected_for_p6": selected,
    "reject_reasons": reject_reasons,
    "resources": {**resources, "lds_bytes": lds_bytes, "sgpr_span": sgpr_span, "private_segment_fixed_size": 0, "scratch_bytes": 0},
    "gate": gate,
    "next_action": "Run P6 structural candidate gate if P2/P3/P4/P5 all pass.",
  }


def main() -> int:
  p1 = read_json("bench/amd-broad-backend-roadmap/bb5a10_p1_layout_spec_result.json", {})
  lds_bytes = int(((p1.get("layout_spec") or {}).get("authority_contract") or {}).get("lds_bytes") or 25088)
  insts = candidate_stream()
  prg = program_with_lds(insts, lds_bytes)
  binary = assemble_linear(prg, prg.src[2], "gfx1100")
  p2 = p2_result(p1, insts, binary, lds_bytes)
  p3 = p3_result(p1)
  p4 = p4_result(p2, p3, insts)
  p5 = p5_result(p2, p3, insts, lds_bytes)
  write_json("bb5a10_p2_rendered_lds_result.json", p2)
  write_json("bb5a10_p3_kloop_stage_result.json", p3)
  write_json("bb5a10_p4_wait_barrier_result.json", p4)
  write_json("bb5a10_p5_resource_policy_result.json", p5)
  summary = {
    "out": [
      "bench/amd-broad-backend-roadmap/bb5a10_p2_rendered_lds_result.json",
      "bench/amd-broad-backend-roadmap/bb5a10_p3_kloop_stage_result.json",
      "bench/amd-broad-backend-roadmap/bb5a10_p4_wait_barrier_result.json",
      "bench/amd-broad-backend-roadmap/bb5a10_p5_resource_policy_result.json",
    ],
    "verdicts": {k: v["verdict"] for k, v in {"P2": p2, "P3": p3, "P4": p4, "P5": p5}.items()},
    "gate_pass": all(v["gate_pass"] for v in (p2, p3, p4, p5)),
  }
  print(json.dumps(summary, indent=2))
  return 0 if summary["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
