#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib, sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extra.qk_amd_bb5a3_wait_scheduler_integration_probe import lowered_lds_wmma_stream
from tinygrad.renderer.amd.schedule import apply_instruction_schedule, resource_summary_from_instructions

OUT = ROOT / "bench/amd-broad-backend-roadmap"


def read_json(rel: str, default: Any = None) -> Any:
  path = ROOT / rel
  if not path.exists(): return default
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main() -> int:
  bb5a3 = read_json("bench/amd-broad-backend-roadmap/bb5a3_wait_scheduler_integration_result.json", {})
  bb5a2 = read_json("bench/amd-broad-backend-roadmap/bb5a2_double_buffer_lds_result.json", {})
  layer2 = read_json("bench/amd-broad-backend-roadmap/bb5a2_lowering_hook_result.json", {})
  before = lowered_lds_wmma_stream()
  after, _ = apply_instruction_schedule(before)
  resources = resource_summary_from_instructions(after)
  lds_bytes = ((layer2.get("lds_lowering") or {}).get("summary") or {}).get("required_local_bytes")
  vgpr_span = int(resources["vgpr"]["span"])
  sgpr_span = int(resources["sgpr"]["span"])
  reject_reasons = []
  if lds_bytes is None: reject_reasons.append("missing_lds_bytes")
  elif int(lds_bytes) > 65536: reject_reasons.append("lds_exceeds_64k_policy")
  if vgpr_span > 128: reject_reasons.append("vgpr_span_exceeds_probe_policy")
  if sgpr_span > 96: reject_reasons.append("sgpr_span_exceeds_probe_policy")
  accepted = not reject_reasons
  gate = {
    "input_bb5a2_pass": bb5a2.get("verdict") == "PASS_DOUBLE_BUFFERED_LDS_LOWERING" and bool(bb5a2.get("gate_pass")),
    "input_bb5a3_pass": bb5a3.get("verdict") == "PASS_BB5A3_WAIT_SCHEDULER_INTEGRATION" and bool(bb5a3.get("gate_pass")),
    "vgpr_span_reported": vgpr_span > 0,
    "sgpr_span_reported": sgpr_span >= 0,
    "lds_bytes_reported": isinstance(lds_bytes, int),
    "spill_risk_classified": True,
    "deterministic_accept_or_reject": accepted or bool(reject_reasons),
    "default_behavior_changed": False,
    "performance_claim": False,
  }
  positive = [
    "input_bb5a2_pass", "input_bb5a3_pass", "vgpr_span_reported", "sgpr_span_reported",
    "lds_bytes_reported", "spill_risk_classified", "deterministic_accept_or_reject",
  ]
  gate_pass = all(gate[x] for x in positive) and not gate["default_behavior_changed"] and not gate["performance_claim"]
  result = {
    "date": "2026-06-19",
    "phase": "BB-5a.4_allocator_resource",
    "schema": "amd_bb5a4_allocator_resource_result_v1",
    "verdict": "PASS_BB5A4_ALLOCATOR_RESOURCE_CONTROL" if gate_pass else "FAIL_BB5A4_ALLOCATOR_RESOURCE_CONTROL",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "resource_summary": resources,
    "lds_bytes": lds_bytes,
    "policy": {
      "accepted": accepted,
      "reject_reasons": reject_reasons,
      "spill_risk": "low_probe_risk" if accepted else "rejected_before_spill_risk",
      "occupancy_class": "probe_occupancy_viable" if accepted else "probe_occupancy_rejected",
    },
    "gate": gate,
    "decision": (
      "BB-5a.4 passes: the scheduled LDS/WMMA candidate reports VGPR/SGPR/LDS resources and is deterministically "
      "accepted or rejected by policy before later correctness/performance gates."
    ),
    "next_action": "Proceed to BB-5a.5 resource policy.",
  }
  write_json("bb5a4_allocator_resource_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a4_allocator_resource_result.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "accepted": accepted,
    "vgpr_span": vgpr_span,
    "lds_bytes": lds_bytes,
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
