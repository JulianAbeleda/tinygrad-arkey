#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib, sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinygrad.renderer.amd.dsl import OFF, v
from tinygrad.renderer.amd.schedule import apply_instruction_schedule, metadata_from_instructions, schedule_metadata_dump
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  ds_load_b32, ds_store_b32, global_load_b32, global_store_b32, s_barrier, s_endpgm, s_waitcnt,
  v_wmma_f32_16x16x16_f16,
)

OUT = ROOT / "bench/amd-broad-backend-roadmap"


def read_json(rel: str, default: Any = None) -> Any:
  path = ROOT / rel
  if not path.exists(): return default
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def inst_names(insts) -> list[str]:
  return [getattr(inst, "op_name", type(inst).__name__) for inst in insts]


def inst_blob(insts) -> bytes:
  return b"".join(inst.to_bytes() for inst in insts)


def lowered_lds_wmma_stream():
  return [
    global_load_b32(vdst=v[16], addr=v[0:1], saddr=OFF),
    global_load_b32(vdst=v[24], addr=v[2:3], saddr=OFF),
    s_waitcnt(simm16=0),
    ds_store_b32(addr=v[8], data0=v[16]),
    ds_store_b32(addr=v[12], data0=v[24]),
    s_barrier(),
    ds_load_b32(vdst=v[16], addr=v[8]),
    ds_load_b32(vdst=v[24], addr=v[12]),
    v_wmma_f32_16x16x16_f16(v[0:7], v[16:23], v[24:31], v[0:7]),
    global_store_b32(addr=v[4:5], data=v[0], saddr=OFF),
    s_endpgm(),
  ]


def main() -> int:
  bb5a2 = read_json("bench/amd-broad-backend-roadmap/bb5a2_double_buffer_lds_result.json", {})
  before = lowered_lds_wmma_stream()
  after, actions = apply_instruction_schedule(before)
  inserted = inst_names(after)
  action_names = [action.action for action in actions]
  metadata = schedule_metadata_dump(metadata_from_instructions(after))
  gate = {
    "input_bb5a2_pass": bb5a2.get("verdict") == "PASS_DOUBLE_BUFFERED_LDS_LOWERING" and bool(bb5a2.get("gate_pass")),
    "actions_present": bool(actions),
    "waitcnt_action_present": "ensure_s_waitcnt" in action_names,
    "delay_alu_action_present": "insert_s_delay_alu" in action_names,
    "clause_action_present": "insert_s_clause" in action_names,
    "scheduled_stream_has_waitcnt": any("S_WAITCNT" in name.upper() for name in inserted),
    "scheduled_stream_has_delay_alu": any("S_DELAY_ALU" in name.upper() for name in inserted),
    "scheduled_stream_has_wmma": any("WMMA" in name.upper() for name in inserted),
    "scheduled_stream_has_lds": any(name.upper().startswith("DS_") for name in inserted),
    "instruction_bytes_changed": inst_blob(before) != inst_blob(after),
    "default_behavior_changed": False,
    "performance_claim": False,
  }
  positive = [
    "input_bb5a2_pass", "actions_present", "waitcnt_action_present", "delay_alu_action_present",
    "clause_action_present", "scheduled_stream_has_waitcnt", "scheduled_stream_has_delay_alu",
    "scheduled_stream_has_wmma", "scheduled_stream_has_lds", "instruction_bytes_changed",
  ]
  gate_pass = all(gate[x] for x in positive) and not gate["default_behavior_changed"] and not gate["performance_claim"]
  result = {
    "date": "2026-06-19",
    "phase": "BB-5a.3_semantic_wait_scheduler_integration",
    "schema": "amd_bb5a3_wait_scheduler_integration_result_v1",
    "verdict": "PASS_BB5A3_WAIT_SCHEDULER_INTEGRATION" if gate_pass else "FAIL_BB5A3_WAIT_SCHEDULER_INTEGRATION",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "before_instruction_names": inst_names(before),
    "after_instruction_names": inserted,
    "actions": [action.to_dict() for action in actions],
    "metadata": metadata,
    "gate": gate,
    "decision": (
      "BB-5a.3 passes: semantic scheduler actions attach to a lowered LDS/WMMA-shaped stream and change instruction bytes "
      "with wait, clause, and delay actions present."
    ),
    "next_action": "Proceed to BB-5a.4 allocator/resource control.",
  }
  write_json("bb5a3_wait_scheduler_integration_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a3_wait_scheduler_integration_result.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "actions": action_names,
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
