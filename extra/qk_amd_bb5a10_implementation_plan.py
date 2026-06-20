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


def phase(pid: str, name: str, track: str, gate: str, artifact: str, depends_on: list[str],
          if_blocked: str, parallel_group: str | None = None, status: str = "planned") -> dict[str, Any]:
  return {
    "id": pid,
    "name": name,
    "track": track,
    "status": status,
    "parallel_group": parallel_group,
    "depends_on": depends_on,
    "minimum_pass": gate,
    "artifact": artifact,
    "if_blocked": if_blocked,
  }


def main() -> int:
  audit = read_json("bench/amd-broad-backend-roadmap/bb5a10_tensile_layout_audit_result.json", {})
  p1_result = read_json("bench/amd-broad-backend-roadmap/bb5a10_p1_layout_spec_result.json", {})
  p1_pass = p1_result.get("verdict") == "PASS_BB5A10_P1_LAYOUT_SPEC_READY" and bool(p1_result.get("gate_pass"))
  p2_result = read_json("bench/amd-broad-backend-roadmap/bb5a10_p2_rendered_lds_result.json", {})
  p3_result = read_json("bench/amd-broad-backend-roadmap/bb5a10_p3_kloop_stage_result.json", {})
  p4_result = read_json("bench/amd-broad-backend-roadmap/bb5a10_p4_wait_barrier_result.json", {})
  p5_result = read_json("bench/amd-broad-backend-roadmap/bb5a10_p5_resource_policy_result.json", {})
  p6_result = read_json("bench/amd-broad-backend-roadmap/bb5a10_p6_structural_candidate_result.json", {})
  p7ab_result = read_json("bench/amd-broad-backend-roadmap/bb5a10_p7a_p7b_correctness_result.json", {})
  p7c_result = read_json("bench/amd-broad-backend-roadmap/bb5a10_p7c_numeric_correctness_result.json", {})
  p7d_result = read_json("bench/amd-broad-backend-roadmap/bb5a10_p7d_authority_correctness_result.json", {})
  p7e_result = read_json("bench/amd-broad-backend-roadmap/bb5a10_p7e_p8_handoff_result.json", {})
  p2_pass = p2_result.get("verdict") == "PASS_BB5A10_P2_RENDERED_LDS_STORE_READ" and bool(p2_result.get("gate_pass"))
  p3_pass = p3_result.get("verdict") == "PASS_BB5A10_P3_KLOOP_STAGE_SCHEDULER" and bool(p3_result.get("gate_pass"))
  p4_pass = p4_result.get("verdict") == "PASS_BB5A10_P4_WAIT_BARRIER_SCHEDULE" and bool(p4_result.get("gate_pass"))
  p5_pass = p5_result.get("verdict") == "PASS_BB5A10_P5_RESOURCE_POLICY" and bool(p5_result.get("gate_pass"))
  p6_pass = p6_result.get("verdict") == "PASS_BB5A10_P6_STRUCTURAL_CANDIDATE" and bool(p6_result.get("gate_pass"))
  p7ab_pass = p7ab_result.get("verdict") == "PASS_BB5A10_P7A_P7B_EXECUTABLE_WRAPPER" and bool(p7ab_result.get("gate_pass"))
  p7c_pass = p7c_result.get("verdict") == "PASS_BB5A10_P7C_SMALL_NUMERIC_CORRECTNESS" and bool(p7c_result.get("gate_pass"))
  p7d_pass = p7d_result.get("verdict") == "PASS_BB5A10_P7D_AUTHORITY_SUBSET_CORRECTNESS" and bool(p7d_result.get("gate_pass"))
  p7e_pass = p7e_result.get("verdict") == "PASS_BB5A10_P7E_P8_HANDOFF_PACKAGE" and bool(p7e_result.get("gate_pass"))
  plan = [
    phase(
      "P0",
      "freeze selected authority contract",
      "contract",
      "selected rocBLAS MT128 function, resource envelope, offset families, and handoff windows are captured",
      "bench/amd-broad-backend-roadmap/bb5a10_tensile_layout_audit_result.json",
      [],
      "rerun BB-5a.10 layout audit; do not implement against aggregate Tensile corpus",
      status="complete" if audit.get("gate_pass") else "blocked",
    ),
    phase(
      "P1",
      "selected-layout lowering spec",
      "B_LDS_layout",
      "derive Tinygrad candidate layout with A/B operand regions, selected-kernel-compatible LDS stores, ds_load_b128 read groups, and nonzero LDS budget",
      "bench/amd-broad-backend-roadmap/bb5a10_p1_layout_spec_result.json",
      ["P0"],
      "fall back to a smaller structural layout probe that preserves the selected offset families; do not require bitexact Tensile byte lanes",
      "layout_scheduler_resource",
      status="complete" if p1_pass else "planned",
    ),
    phase(
      "P2",
      "renderer LDS store/read lowering",
      "B_LDS_layout",
      "rendered source/disasm shows nonzero LDS, LDS stores, ds_load_b128 reads, and WMMA source registers overlap ds_load_b128 destinations",
      "bench/amd-broad-backend-roadmap/bb5a10_p2_rendered_lds_result.json",
      ["P1"],
      "if ds_load_b128 does not appear, split into vector-read lowering first; if LDS stores vanish, split DEFINE_LOCAL/rangeify preservation first",
      "layout_scheduler_resource",
      status="complete" if p2_pass else "planned",
    ),
    phase(
      "P3",
      "K-loop stage scheduler",
      "C_K_loop_scheduler",
      "prologue plus steady-state depthU=16 loop with producer/consumer LDS stages and no current-read/next-write alias",
      "bench/amd-broad-backend-roadmap/bb5a10_p3_kloop_stage_result.json",
      ["P1"],
      "if full loop lowering blocks, emit a two-iteration structural kernel and prove alternating stage order before generalizing",
      "layout_scheduler_resource",
      status="complete" if p3_pass else "planned",
    ),
    phase(
      "P4",
      "semantic waits and barriers",
      "C_wait_barrier_scheduler",
      "vmcnt/lgkmcnt waits and barriers are dependency-derived and preserve LDS store -> barrier -> LDS load -> WMMA ordering",
      "bench/amd-broad-backend-roadmap/bb5a10_p4_wait_barrier_result.json",
      ["P2", "P3"],
      "if wait movement is byte-identical or textual, add dependency-group metadata to the lowered stream and rerun scheduler integration",
      "layout_scheduler_resource",
      status="complete" if p4_pass else "planned",
    ),
    phase(
      "P5",
      "resource policy and rejection",
      "D_resource_policy",
      "candidate reports VGPR/SGPR/LDS/private/scratch envelope and is either scratch-free or rejected before timing",
      "bench/amd-broad-backend-roadmap/bb5a10_p5_resource_policy_result.json",
      ["P2", "P3"],
      "if resources exceed envelope, reduce accumulator/layout ambition or reject deterministically; do not time spill candidates",
      "layout_scheduler_resource",
      status="complete" if p5_pass else "planned",
    ),
    phase(
      "P6",
      "structural candidate gate",
      "E_candidate_gate",
      "single candidate passes P2/P3/P4/P5 together: nonzero LDS, LDS stores, ds_load_b128 feeding WMMA, waits/barriers, scratch-free",
      "bench/amd-broad-backend-roadmap/bb5a10_p6_structural_candidate_result.json",
      ["P2", "P3", "P4", "P5"],
      "if any structural row fails, route back only to the failing row; do not reopen closed LDS tiling or knob sweeps",
      status="complete" if p6_pass else "planned",
    ),
    phase(
      "P7",
      "correctness harness",
      "E_candidate_gate",
      "small WMMA correctness and authority-shape fp16 relative-error gate pass",
      "bench/amd-broad-backend-roadmap/bb5a10_p7_correctness_result.json",
      ["P6"],
      "if small correctness fails, debug layout mapping; if authority fails only, debug edge predicates and launch mapping",
      status="complete" if p7d_pass else "in_progress" if p7c_pass or p7ab_pass else "planned",
    ),
    phase(
      "P8",
      "performance gate",
      "E_candidate_gate",
      "pure tinygrad authority prefill reaches >=60 TFLOPS without scratch/private spill",
      "bench/amd-broad-backend-roadmap/bb5a10_p8_performance_result.json",
      ["P7"],
      "if below 60 TFLOPS with structural pass, classify counters/instruction mix before changing layout; no blind parameter sweeps",
    ),
    phase(
      "P9",
      "q8 transfer reopen decision",
      "F_q8_transfer",
      "only after P8 pass, scope q8 downstream transfer with <=75us continuation gate and <=60us strong pass",
      "bench/amd-broad-backend-roadmap/bb5a10_p9_q8_reopen_result.json",
      ["P8"],
      "if P8 does not pass, keep q8 transfer blocked and keep all q8-native scheduler work disallowed",
    ),
  ]
  gate = {
    "layout_audit_pass": audit.get("verdict") == "PASS_TENSILE_LAYOUT_AUDIT_CANDIDATE_SPEC_READY_NOT_BITEXACT" and bool(audit.get("gate_pass")),
    "phase_count": len(plan),
    "p0_complete": plan[0]["status"] == "complete",
    "p1_status_valid": plan[1]["status"] in {"planned", "complete"},
    "has_blocked_continuation_for_every_phase": all(bool(p["if_blocked"]) for p in plan),
    "q8_last_and_blocked_until_performance": plan[-1]["track"] == "F_q8_transfer" and plan[-1]["depends_on"] == ["P8"],
    "parallel_core_declared": {p["parallel_group"] for p in plan if p["parallel_group"]} == {"layout_scheduler_resource"},
  }
  gate_pass = bool(gate["layout_audit_pass"] and gate["p0_complete"] and gate["has_blocked_continuation_for_every_phase"] and gate["q8_last_and_blocked_until_performance"] and gate["parallel_core_declared"])
  result = {
    "date": "2026-06-19",
    "phase": "BB-5a.10_implementation_plan",
    "schema": "amd_bb5a10_implementation_plan_v1",
    "verdict": "PASS_BB5A10_IMPLEMENTATION_PLAN_READY" if gate_pass else "BLOCKED_BB5A10_IMPLEMENTATION_PLAN_INPUTS",
    "gate_pass": gate_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "phases": plan,
    "execution_order": {
      "complete_now": [p["id"] for p in plan if p["status"] == "complete"],
      "parallel_next": [] if p6_pass else ["P2", "P3", "P4", "P5"] if p1_pass else ["P1", "P2", "P3", "P4", "P5"],
      "serial_gates_after_parallel": ["P8", "P9"] if p7e_pass else ["P7e", "P8", "P9"] if p7d_pass else ["P7d", "P7e", "P8", "P9"] if p7c_pass else ["P7c", "P7d", "P7e", "P8", "P9"] if p7ab_pass else ["P7", "P8", "P9"] if p6_pass else ["P6", "P7", "P8", "P9"],
      "do_not_start": ["P9 before P8 passes", "q8 native transfer before P8 passes"],
    },
    "gate": gate,
    "decision": "BB-5a.10 P7e P8 handoff package passes. Next valid work is P8 full-authority performance gate; P9 remains blocked until P8 passes." if p7e_pass else
                "BB-5a.10 P7d authority-subset correctness passes. Next valid work is P7e P8 handoff package; P8/P9 remain blocked until the handoff exists." if p7d_pass else
                "BB-5a.10 P7c small numeric correctness passes. Next valid work is P7d authority-shape correctness smoke; P8/P9 remain blocked." if p7c_pass else
                "BB-5a.10 P7a/P7b pass. Next valid work is P7c small numeric correctness; P8/P9 remain blocked." if p7ab_pass else
                "BB-5a.10 P6 structural candidate passes. Next valid work is P7 executable correctness; P8/P9 remain blocked." if p6_pass else
                "BB-5a.10 phases are fully scoped. P1 is complete; start P2/P3/P4/P5 as one coordinated implementation batch, then run P6/P7/P8 gates." if p1_pass else
                "BB-5a.10 phases are fully scoped. Start P1/P2/P3/P4/P5 as one coordinated implementation batch, then run P6/P7/P8 gates.",
    "input_artifacts": ["bench/amd-broad-backend-roadmap/bb5a10_tensile_layout_audit_result.json"],
  }
  write_json("bb5a10_implementation_plan_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a10_implementation_plan_result.json",
    "verdict": result["verdict"],
    "gate_pass": gate_pass,
    "phases": [p["id"] for p in plan],
    "parallel_next": result["execution_order"]["parallel_next"],
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
