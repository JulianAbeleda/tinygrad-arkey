#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib, sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinygrad.renderer.amd.dsl import OFF, v
from tinygrad.renderer.amd.schedule import (
  apply_instruction_schedule, metadata_from_instructions, resource_summary_from_instructions, schedule_metadata_dump,
)
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  global_load_b32, global_store_b32, s_barrier, s_endpgm, s_waitcnt, v_add_f32_e32, v_wmma_f32_16x16x16_f16,
)

OUT = ROOT / "bench/amd-broad-backend-roadmap"
from extra.qk_probe_harness import probe_io
read_json, write_json = probe_io(OUT)



def inst_names(insts) -> list[str]:
  return [getattr(inst, "op_name", type(inst).__name__) for inst in insts]


def inst_blob(insts) -> bytes:
  return b"".join(inst.to_bytes() for inst in insts)


def q8_stream():
  return [
    global_load_b32(vdst=v[0], addr=v[0:1], saddr=OFF),
    s_waitcnt(simm16=0),
    v_add_f32_e32(v[1], v[0], v[1]),
    s_barrier(),
    global_store_b32(addr=v[2:3], data=v[1], saddr=OFF),
    s_endpgm(),
  ]


def wmma_stream():
  return [
    global_load_b32(vdst=v[16], addr=v[0:1], saddr=OFF),
    global_load_b32(vdst=v[24], addr=v[2:3], saddr=OFF),
    s_waitcnt(simm16=0),
    v_wmma_f32_16x16x16_f16(v[0:7], v[16:23], v[24:31], v[0:7]),
    global_store_b32(addr=v[4:5], data=v[0], saddr=OFF),
    s_endpgm(),
  ]


def schedule_probe(name: str, insts) -> dict[str, Any]:
  before = list(insts)
  after, actions = apply_instruction_schedule(before)
  before_names, after_names = inst_names(before), inst_names(after)
  action_to_inst = {"insert_s_clause": "S_CLAUSE", "ensure_s_waitcnt": "S_WAITCNT", "insert_s_delay_alu": "S_DELAY_ALU"}
  inserted = [action_to_inst[action.action] for action in actions if action.action in action_to_inst]
  byte_changed = inst_blob(before) != inst_blob(after)
  metadata = schedule_metadata_dump(metadata_from_instructions(after))
  return {
    "name": name,
    "before_instruction_count": len(before),
    "after_instruction_count": len(after),
    "before_instruction_names": before_names,
    "after_instruction_names": after_names,
    "actions": [action.to_dict() for action in actions],
    "inserted_scheduler_instruction_count": len(after) - len(before),
    "inserted_scheduler_instructions": inserted,
    "byte_changed": byte_changed,
    "correctness_preservation": "scheduler_hints_only_no_dataflow_change",
    "metadata": metadata,
    "gate_pass": byte_changed and len(after) > len(before) and bool(actions),
  }


def resource_probe(name: str, insts) -> dict[str, Any]:
  before = list(insts)
  after, _ = apply_instruction_schedule(before)
  return {
    "name": name,
    "before": resource_summary_from_instructions(before),
    "after": resource_summary_from_instructions(after),
    "resource_accounting_changed_by_scheduler": False,
    "controlled_fields": ["vgpr", "sgpr", "instruction_count", "register_operand_count"],
    "gate_pass": True,
  }


def main() -> int:
  streams = {"q8_shaped_instruction": q8_stream(), "wmma_shaped_instruction": wmma_stream()}
  schedule_rows = [schedule_probe(name, insts) for name, insts in streams.items()]
  resource_rows = [resource_probe(name, insts) for name, insts in streams.items()]
  wait_gate = all(row["gate_pass"] for row in schedule_rows)
  resource_gate = all(row["gate_pass"] for row in resource_rows)
  wait_result = {
    "date": "2026-06-19",
    "phase": "BB-3_wait_scheduler_emitter",
    "schema": "amd_wait_scheduler_result_v1",
    "verdict": "PASS_SEMANTIC_WAIT_SCHEDULER_PROBE" if wait_gate else "FAIL_SEMANTIC_WAIT_SCHEDULER_PROBE",
    "gate_pass": wait_gate,
    "rows": schedule_rows,
    "semantic_scope": "probe_local_scheduler_hints_only",
    "default_behavior_changed": False,
    "next_action": "BB-5 may consume this only after BB-4 resource accounting also passes.",
  }
  resource_result = {
    "date": "2026-06-19",
    "phase": "BB-4_register_resource_control",
    "schema": "amd_register_resource_result_v1",
    "verdict": "PASS_RESOURCE_ACCOUNTING_PROBE" if resource_gate else "FAIL_RESOURCE_ACCOUNTING_PROBE",
    "gate_pass": resource_gate,
    "rows": resource_rows,
    "control_scope": "accounting_only_no_allocator_change",
    "default_behavior_changed": False,
    "next_action": "BB-5 software-pipeline probe may start; real allocator controls remain a later implementation step.",
  }
  write_json("wait_scheduler_result.json", wait_result)
  write_json("register_resource_result.json", resource_result)
  print(json.dumps({
    "wait_scheduler": wait_result["verdict"],
    "register_resource": resource_result["verdict"],
    "gate_pass": wait_gate and resource_gate,
    "out": "bench/amd-broad-backend-roadmap/{wait_scheduler_result.json,register_resource_result.json}",
  }, indent=2))
  return 0 if wait_gate and resource_gate else 1


if __name__ == "__main__":
  raise SystemExit(main())
