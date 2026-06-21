#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib, sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.renderer.amd.schedule import AMDLDSStagePlan, lds_lowering_dump, lower_lds_stage_plan_to_define_locals
from tinygrad.uop.ops import Ops

OUT = ROOT / "bench/amd-broad-backend-roadmap"
from extra.qk_probe_harness import probe_io
read_json, write_json = probe_io(OUT)




def reconstruct_plan(layer1: dict[str, Any]) -> AMDLDSStagePlan:
  plan = ((layer1.get("lds_stage_plan") or {}).get("plan") or {})
  return AMDLDSStagePlan(
    pipeline_id=str(plan["pipeline_id"]),
    stage_count=int(plan["stage_count"]),
    slots=tuple(int(x) for x in plan["slots"]),
    slot_roles={int(k): tuple(str(v) for v in vals) for k, vals in plan["slot_roles"].items()},
    slot_offsets={int(k): int(v) for k, v in plan["slot_offsets"].items()},
    dependency_groups=tuple(str(x) for x in plan["dependency_groups"]),
    required_local_bytes=int(plan["required_local_bytes"]),
    alias_safe=bool(plan["alias_safe"]),
    lowering_status=str(plan.get("lowering_status", "planned")),
  )


def main() -> int:
  layer1 = read_json("bench/amd-broad-backend-roadmap/bb5a2_lds_stage_plan_result.json", {})
  plan = reconstruct_plan(layer1)
  locals_, lowered = lower_lds_stage_plan_to_define_locals(plan, dtypes.half, base_slot=9000)
  dump = lds_lowering_dump(plan, lowered)
  uop_summary = [
    {
      "op": u.op.name,
      "slot": u.arg,
      "dtype": str(u.dtype),
      "addrspace": "lds" if u.addrspace is AddrSpace.LOCAL else str(u.addrspace),
      "nbytes": u.dtype.nbytes(),
    }
    for u in locals_
  ]
  gate = {
    "input_lds_stage_plan_pass": layer1.get("verdict") == "PASS_LDS_STAGE_PLAN" and bool(layer1.get("gate_pass")),
    "lowered_slot_count_is_2": len(lowered) == 2,
    "define_local_uops_emitted": all(u.op is Ops.DEFINE_LOCAL for u in locals_),
    "addrspace_local": all(u.addrspace is AddrSpace.LOCAL for u in locals_),
    "define_local_slots_distinct": len({row.define_local_slot for row in lowered}) == len(lowered),
    "planned_slots_preserved": [row.slot for row in lowered] == list(plan.slots),
    "planned_offsets_preserved": [row.offset_bytes for row in lowered] == [plan.slot_offsets[x] for x in plan.slots],
    "lowered_bytes_match_plan": sum(row.size_bytes for row in lowered) == plan.required_local_bytes,
    "default_behavior_changed": False,
    "performance_claim": False,
  }
  positive = [
    "input_lds_stage_plan_pass", "lowered_slot_count_is_2", "define_local_uops_emitted", "addrspace_local",
    "define_local_slots_distinct", "planned_slots_preserved", "planned_offsets_preserved", "lowered_bytes_match_plan",
  ]
  gate_pass = all(gate[x] for x in positive) and not gate["default_behavior_changed"] and not gate["performance_claim"]
  result = {
    "date": "2026-06-19",
    "phase": "BB-5a.2_layer_2_define_local_lowering_hook",
    "schema": "amd_bb5a2_lowering_hook_result_v1",
    "verdict": "PASS_DEFINE_LOCAL_LOWERING_HOOK" if gate_pass else "FAIL_DEFINE_LOCAL_LOWERING_HOOK",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "input_artifact": "bench/amd-broad-backend-roadmap/bb5a2_lds_stage_plan_result.json",
    "lds_lowering": dump,
    "uop_summary": uop_summary,
    "gate": gate,
    "decision": (
      "Layer 2 passes: planned lds_slot=0/1 stages lower through a gated helper into durable DEFINE_LOCAL UOps. "
      "This still does not prove renderer consumption or non-byte-identical AMD ISA."
    ),
    "next_action": "Implement BB-5a.2 Layer 3: renderer/ISA evidence for the lowered two-slot LDS structure.",
  }
  write_json("bb5a2_lowering_hook_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a2_lowering_hook_result.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "define_local_slots": dump["summary"]["define_local_slots"],
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
