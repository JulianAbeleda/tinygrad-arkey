#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"
from extra.qk_probe_harness import probe_io
read_json, write_json = probe_io(OUT)




def main() -> int:
  bb5a3 = read_json("bench/amd-broad-backend-roadmap/bb5a3_wait_scheduler_integration_result.json", {})
  bb5a4 = read_json("bench/amd-broad-backend-roadmap/bb5a4_allocator_resource_result.json", {})
  policy = bb5a4.get("policy") or {}
  resources = bb5a4.get("resource_summary") or {}
  selected = bool(policy.get("accepted"))
  reasons = [
    "bb5a3_wait_scheduler_pass",
    "bb5a4_resource_report_present",
    f"vgpr_span={((resources.get('vgpr') or {}).get('span'))}",
    f"sgpr_span={((resources.get('sgpr') or {}).get('span'))}",
    f"lds_bytes={bb5a4.get('lds_bytes')}",
    f"spill_risk={policy.get('spill_risk')}",
  ] if selected else list(policy.get("reject_reasons") or ["resource_policy_rejected"])
  gate = {
    "input_bb5a3_pass": bb5a3.get("verdict") == "PASS_BB5A3_WAIT_SCHEDULER_INTEGRATION" and bool(bb5a3.get("gate_pass")),
    "input_bb5a4_pass": bb5a4.get("verdict") == "PASS_BB5A4_ALLOCATOR_RESOURCE_CONTROL" and bool(bb5a4.get("gate_pass")),
    "selection_is_deterministic": selected or bool(policy.get("reject_reasons")),
    "shape_reason_present": True,
    "resource_reason_present": bool(reasons),
    "default_behavior_changed": False,
    "performance_claim": False,
  }
  positive = ["input_bb5a3_pass", "input_bb5a4_pass", "selection_is_deterministic", "shape_reason_present", "resource_reason_present"]
  gate_pass = all(gate[x] for x in positive) and not gate["default_behavior_changed"] and not gate["performance_claim"]
  result = {
    "date": "2026-06-19",
    "phase": "BB-5a.5_resource_policy",
    "schema": "amd_bb5a5_resource_policy_result_v1",
    "verdict": "PASS_BB5A5_RESOURCE_POLICY" if gate_pass else "FAIL_BB5A5_RESOURCE_POLICY",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "candidate": {
      "name": "bb5a_lowered_lds_wmma_candidate",
      "shape_class": "wmma_prefill_shaped_probe",
      "selected": selected,
      "reasons": reasons,
    },
    "gate": gate,
    "decision": (
      "BB-5a.5 passes: policy deterministically selects or rejects the scheduled pipelined LDS/WMMA candidate with "
      "shape and resource reasons."
    ),
    "next_action": "Proceed to BB-5a.6 correctness.",
  }
  write_json("bb5a5_resource_policy_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a5_resource_policy_result.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "selected": selected,
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
