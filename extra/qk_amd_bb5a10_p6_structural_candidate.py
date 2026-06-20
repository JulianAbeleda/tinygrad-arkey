#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"


def read_json(rel: str, default: Any = None) -> Any:
  path = ROOT / rel
  if not path.exists(): return default
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main() -> int:
  p2 = read_json("bench/amd-broad-backend-roadmap/bb5a10_p2_rendered_lds_result.json", {})
  p3 = read_json("bench/amd-broad-backend-roadmap/bb5a10_p3_kloop_stage_result.json", {})
  p4 = read_json("bench/amd-broad-backend-roadmap/bb5a10_p4_wait_barrier_result.json", {})
  p5 = read_json("bench/amd-broad-backend-roadmap/bb5a10_p5_resource_policy_result.json", {})
  names = p2.get("instruction_names") or []
  gate = {
    "input_p2_pass": p2.get("verdict") == "PASS_BB5A10_P2_RENDERED_LDS_STORE_READ" and bool(p2.get("gate_pass")),
    "input_p3_pass": p3.get("verdict") == "PASS_BB5A10_P3_KLOOP_STAGE_SCHEDULER" and bool(p3.get("gate_pass")),
    "input_p4_pass": p4.get("verdict") == "PASS_BB5A10_P4_WAIT_BARRIER_SCHEDULE" and bool(p4.get("gate_pass")),
    "input_p5_pass": p5.get("verdict") == "PASS_BB5A10_P5_RESOURCE_POLICY" and bool(p5.get("gate_pass")),
    "nonzero_lds": ((p2.get("elf") or {}).get("group_segment_fixed_size") or 0) > 0,
    "has_selected_lds_store": "DS_STORE_B64" in names or "DS_STORE_B128" in names,
    "has_ds_load_b128": "DS_LOAD_B128" in names,
    "has_wmma": any("WMMA" in n for n in names),
    "has_wait_barrier_schedule": bool(p4.get("gate", {}).get("has_waitcnt")) and bool(p4.get("gate", {}).get("has_barrier")),
    "resource_policy_selected": bool(p5.get("selected_for_p6")),
  }
  gate_pass = all(gate.values())
  result = {
    "date": "2026-06-19",
    "phase": "BB-5a.10_P6_structural_candidate_gate",
    "schema": "amd_bb5a10_p6_structural_candidate_result_v1",
    "verdict": "PASS_BB5A10_P6_STRUCTURAL_CANDIDATE" if gate_pass else "BLOCKED_BB5A10_P6_STRUCTURAL_CANDIDATE",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "candidate": {
      "class": "structural_isa_elf_candidate",
      "lds_bytes": (p2.get("elf") or {}).get("group_segment_fixed_size"),
      "instruction_names": names,
      "resource_summary": p5.get("resources"),
    },
    "gate": gate,
    "decision": "P6 complete: P2-P5 jointly produce a structural staged-LDS WMMA candidate. Next valid work is P7 executable correctness.",
    "next_action": "Build an executable correctness harness for this structural candidate; do not start P8 performance before P7 passes.",
    "input_artifacts": [
      "bench/amd-broad-backend-roadmap/bb5a10_p2_rendered_lds_result.json",
      "bench/amd-broad-backend-roadmap/bb5a10_p3_kloop_stage_result.json",
      "bench/amd-broad-backend-roadmap/bb5a10_p4_wait_barrier_result.json",
      "bench/amd-broad-backend-roadmap/bb5a10_p5_resource_policy_result.json",
    ],
  }
  write_json("bb5a10_p6_structural_candidate_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a10_p6_structural_candidate_result.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "next": result["next_action"],
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
