#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c9_new_info_ledger_result.json"


def read_json(rel: str) -> dict[str, Any]:
  with (ROOT / rel).open() as f:
    return json.load(f)


def main() -> int:
  dnr3c7a = read_json("bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7a_resource_ledger_result.json")
  dnr3c7b = read_json("bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7b_pmc_ladder_result.json")
  dnr3c7c = read_json("bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7c_issue_interleaving_result.json")
  dnr3c7d = read_json("bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7d_confirmation_result.json")
  dnr3c8 = read_json("bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c8_tooling_inventory_result.json")
  oracle = read_json("bench/q8-ffn-amd-scheduler-project/oracle_contract.json")
  loader = read_json("bench/q8-ffn-amd-scheduler-project/artifact_loader.json")
  n1 = read_json("bench/q8-ffn-amd-scheduler-project/n1_attribution.json")

  known_closed = [
    {
      "fact": "native q8 decode correctness path is real",
      "evidence": "DNR-3C2/C4/C6/C7C/C7D all launch correctness-checked variants",
      "decision": "do not reopen Q4_K addressing, q8 addressing, scale/min extraction, dot4 selection, or gate/up correctness as blockers",
    },
    {
      "fact": "static count matching is insufficient",
      "evidence": "DNR-3C6 best static movement was single-digit microseconds in its ladder; DNR-3C7D best static is only 9.613us over native",
      "decision": "do not add branch, wait, marker, load, or LDS-count patches unless they come with a new attribution signal",
    },
    {
      "fact": "resource descriptors do not show a simple scratch/private/LDS explanation",
      "evidence": "DNR-3C7A native and C4 private=0, tiny LDS=16, same waves/workgroup; DNR-3C8 native descriptor tooling works",
      "decision": "do not expect a one-line descriptor flag to explain the oracle gap",
    },
    {
      "fact": "PMC capture is usable directionally",
      "evidence": "DNR-3C7B and DNR-3C7D collect SQ/GL2/SQC counters and preserve correctness",
      "decision": "PMC can confirm a candidate direction, but cannot promote without material timing",
    },
    {
      "fact": "first issue-order experiment is a small local win only",
      "evidence": "DNR-3C7D C7C-best: 264.628us vs best static 270.635us, gain=6.006us; oracle=93.540us",
      "decision": "park local native DNR-3C schedule rewrites unless new information names a larger lever",
    },
  ]

  missing_info = [
    {
      "id": "NINFO-1",
      "name": "oracle VGPR/SGPR/resource envelope",
      "question": "Does the hipcc/LLD oracle use a fundamentally different VGPR/SGPR allocation, occupancy, or launch resource envelope than native?",
      "current_state": {
        "have": {
          "oracle_local_size": loader["loader"]["gateup"].get("local_size"),
          "oracle_group_segment_size": loader["loader"]["gateup"].get("group_segment_size"),
          "oracle_private_segment_size": loader["loader"]["gateup"].get("private_segment_size"),
          "oracle_kernarg_size": loader["loader"]["gateup"].get("kernarg_size"),
        },
        "missing": ["oracle VGPR count", "oracle SGPR count", "oracle occupancy", "oracle full kernel descriptor rsrc fields", "oracle per-phase live range"],
      },
      "tool_needed": "extract oracle code object metadata with llvm-objdump/roc-objdump/amdhsa metadata or artifact-side descriptor decode",
      "minimum_evidence": [
        "oracle allocated VGPR/workitem and SGPR count",
        "oracle private/scratch bytes",
        "oracle LDS/group segment bytes",
        "oracle occupancy estimate under the same wave/workgroup shape",
        "same fields for native, best static, and C7C-best in one table",
      ],
      "reopens_native_if": "oracle resource envelope differs in a way that plausibly explains >=30us and is implementable in native",
      "parks_native_if": "oracle resource envelope is essentially same-class as native/C7C or only differs in fields already ruled out",
      "priority": "P0",
    },
    {
      "id": "NINFO-2",
      "name": "oracle ISA and semantic schedule map",
      "question": "What exact instruction order does the oracle use for load, unpack/select, dot4, scale/min, cross-wave reduction, waits, and stores?",
      "current_state": {
        "have": {
          "oracle_grouped": oracle["instruction_contract"]["oracle_grouped"],
          "native_grouped": dnr3c7d["timing_rows"][0]["grouped"],
          "c7c_best_grouped": dnr3c7d["timing_rows"][2]["grouped"],
        },
        "missing": [
          "ordered oracle disassembly tied to semantic stages",
          "register operand map for q4/q8/scales/accumulators",
          "waitcnt placement by dependency reason",
          "branch/exec predicate purpose",
          "instruction-level stage overlap model",
        ],
      },
      "tool_needed": "disassemble oracle code object and annotate PC ranges into semantic stages",
      "minimum_evidence": [
        "stage-labeled oracle ISA table",
        "native/C7C equivalent stage table",
        "stage-by-stage delta that names one unimplemented mechanism",
        "proof the mechanism is not just count matching",
      ],
      "reopens_native_if": "oracle has a stage ordering native has not tried, with a clear dependency-safe construction path",
      "parks_native_if": "oracle ordering reduces to already-tested local patterns or uncopyable compiler/runtime mechanics",
      "priority": "P0",
    },
    {
      "id": "NINFO-3",
      "name": "SQTT/body timeline mapped to q8 PCs",
      "question": "Where does native spend cycles inside the q8 kernel body, and does the oracle avoid those stalls?",
      "current_state": {
        "have": {
          "pmc_runnable": n1["gate"].get("pmc_profile_runnable"),
          "sqtt_decode_usable": n1["gate"].get("sqtt_decode_usable"),
          "dnr3c8_sqtt_status": next(row for row in dnr3c8["tools"] if row["tool"] == "SQTT body timeline")["status"],
        },
        "missing": [
          "PC-level native body timeline",
          "PC-level oracle body timeline",
          "stall reason per PC or stage",
          "mapping from PCs to disassembled instructions",
          "same-run alignment between timing and trace capture",
        ],
      },
      "tool_needed": "repair tinygrad HCQ SQTT body decode or use ROCprofiler/AQLprofile ATT body packets and join PCs to disassembly",
      "minimum_evidence": [
        "nonzero body packets for q8 kernel",
        "PC-to-ISA join for native and oracle or native plus C7C-best",
        "stage-level stall histogram",
        "one stall class with plausible >=30us movement",
      ],
      "reopens_native_if": "a specific PC/stage stall dominates and maps to a native schedule transform",
      "parks_native_if": "timeline cannot be recovered or shows diffuse stalls with no actionable stage",
      "priority": "P0",
    },
    {
      "id": "NINFO-4",
      "name": "true live-range pressure and allocator model",
      "question": "Do local rewrites lose time because VGPR live ranges, not instruction counts, reduce occupancy or cause issue pressure?",
      "current_state": {
        "have": {
          "native_allocated_vgpr": dnr3c7a["native"]["descriptor"]["allocated_vgpr_per_workitem"],
          "dnr3c4_allocated_vgpr": dnr3c7a["dnr3c4"]["descriptor"]["allocated_vgpr_per_workitem"],
          "native_private": dnr3c7a["native"]["descriptor"]["private_segment_fixed_size"],
          "dnr3c4_private": dnr3c7a["dnr3c4"]["descriptor"]["private_segment_fixed_size"],
        },
        "missing": [
          "per-instruction live intervals",
          "peak VGPR by semantic stage",
          "C7C-best resource ledger row",
          "oracle live interval or allocator-equivalent metadata",
          "occupancy calculation linked to timing",
        ],
      },
      "tool_needed": "static live-interval builder over AMD DSL Reg operands plus oracle metadata/disassembly",
      "minimum_evidence": [
        "native, best-static, C7C-best live-range charts",
        "peak VGPR/SGPR by stage",
        "occupancy estimate for each",
        "delta tied to measured timing or PMC issue counters",
      ],
      "reopens_native_if": "a specific live-range split/reuse policy can recover >=30us without breaking correctness",
      "parks_native_if": "live ranges explain only small C7C-level movement or need unavailable oracle allocator behavior",
      "priority": "P1",
    },
    {
      "id": "NINFO-5",
      "name": "counter-to-time calibration",
      "question": "Which PMC deltas actually predict wall-time movement for this kernel?",
      "current_state": {
        "have": {
          "c7d_pmc_confirms_wait_or_busy": dnr3c7d["pmc_attribution"]["pmc_confirms_wait_or_busy"],
          "c7d_timing_material": dnr3c7d["gates"]["timing_material"],
        },
        "missing": [
          "counter repeatability over multiple clock-fair runs",
          "correlation between SQ_WAIT_ANY/SQ_BUSY and latency",
          "counter perturbation estimate",
          "per-counter confidence interval",
        ],
      },
      "tool_needed": "multi-run PMC/timing calibration matrix over native, best-static, C7C-best, and oracle if possible",
      "minimum_evidence": [
        "at least 3 same-harness timing runs",
        "at least 3 PMC runs per counter family",
        "counter deltas that monotonically track latency deltas",
        "known perturbation bounds",
      ],
      "reopens_native_if": "a counter family reliably predicts >=30us movement and points to a modifiable stage",
      "parks_native_if": "PMC remains directional only and cannot distinguish small local wins from real route wins",
      "priority": "P1",
    },
    {
      "id": "NINFO-6",
      "name": "oracle runtime/launch integration differences",
      "question": "Is the oracle advantage partly outside the q8 body, such as launch shape, loader, cache state, code object flags, or dispatch contract?",
      "current_state": {
        "have": {
          "oracle_known_us": oracle["known_timings_us"].get("hipcc_lld_gateup_current_loader"),
          "oracle_local_size": loader["loader"]["gateup"].get("local_size"),
          "native_local_resource_known": True,
        },
        "missing": [
          "same-clock interleaved oracle-vs-native-vs-C7C timing",
          "oracle code object flags beyond partial loader metadata",
          "cache warm/cold policy parity",
          "dispatch packet parity",
          "kernel arg layout parity beyond size",
        ],
      },
      "tool_needed": "one-clock interleaved harness including oracle artifact and native candidates plus descriptor dump",
      "minimum_evidence": [
        "native, best-static, C7C-best, oracle in one interleaved run",
        "clock provenance",
        "descriptor/dispatch metadata for each row",
        "same input and correctness policy",
      ],
      "reopens_native_if": "the oracle gap shrinks materially under fair harness or a launch/descriptor field names an implementable difference",
      "parks_native_if": "same-run oracle gap remains >100us with no launch/descriptor explanation",
      "priority": "P1",
    },
    {
      "id": "NINFO-7",
      "name": "new decode primitive route",
      "question": "Is local native q8 MMVQ the wrong primitive compared with q8 artifact reuse, two-lane route, or a fused decode primitive?",
      "current_state": {
        "have": {
          "native_dnr3c_parked": True,
          "q8_artifact_oracle_practical": True,
        },
        "missing": [
          "promotion-grade quality policy for q8 activation reuse",
          "full decode route timing with q8 artifact integration",
          "Llama-relative primitive comparison for decode after DNR-3C closeout",
          "clear acceptance threshold for small local native wins",
        ],
      },
      "tool_needed": "route-level decode promotion audit rather than another native kernel rewrite",
      "minimum_evidence": [
        "route-level latency",
        "quality/error policy",
        "coverage over gate/up/down or target roles",
        "fallback behavior",
        "promotion threshold accepted by project owner",
      ],
      "reopens_native_if": "route audit requires native q8 schedule as a component and defines a smaller acceptable win",
      "parks_native_if": "q8 artifact/oracle route is the only route with meaningful decode movement",
      "priority": "P2",
    },
  ]

  reopen_gates = [
    {
      "gate": "resource_reopen",
      "requires": ["NINFO-1", "NINFO-4"],
      "pass_condition": "resource/live-range delta names a specific implementable native change with credible >=30us upside",
    },
    {
      "gate": "timeline_reopen",
      "requires": ["NINFO-2", "NINFO-3"],
      "pass_condition": "PC/stage timeline names a dominant stall and a dependency-safe schedule transform",
    },
    {
      "gate": "counter_reopen",
      "requires": ["NINFO-5"],
      "pass_condition": "PMC family correlates with wall time and predicts a material candidate before implementation",
    },
    {
      "gate": "fair_oracle_reopen",
      "requires": ["NINFO-6"],
      "pass_condition": "same-run oracle comparison changes the target or exposes an implementable launch/code-object difference",
    },
    {
      "gate": "route_reopen",
      "requires": ["NINFO-7"],
      "pass_condition": "decode route-level policy accepts a small native win or requires native as one part of a broader route",
    },
  ]

  do_not_do = [
    "do not add more load-count-only, branch-count-only, wait-count-only, marker-count-only, or LDS-count-only patches",
    "do not start BEAM/search from static shape similarity; it was refuted",
    "do not promote DNR-3C native schedule from the current C7D result",
    "do not treat PMC direction as timing authority without calibration",
    "do not reopen Q4_K/q8 address/correctness as if they are still the blocker",
  ]

  gates = {
    "dnr3c7d_parked": dnr3c7d["verdict"] == "BLOCKED_DNR3C7D_C7C_SIGNAL_NOT_REPRODUCED_PARK_NATIVE_ROUTE",
    "missing_info_exhaustive_count": len(missing_info) == 7,
    "all_missing_info_has_reopen_and_park_condition": all(row.get("reopens_native_if") and row.get("parks_native_if") for row in missing_info),
    "reopen_gates_defined": len(reopen_gates) == 5,
    "closed_facts_defined": len(known_closed) >= 5,
    "do_not_do_defined": len(do_not_do) >= 5,
  }

  result = {
    "date": "2026-06-20",
    "phase": "DNR-3C9_DECODE_NEW_INFORMATION_LEDGER",
    "schema": "decode_native_renderer_dnr3c9_new_info_ledger_v1",
    "verdict": "SCOPE_DNR3C9_NEW_INFORMATION_EXHAUSTED_NATIVE_PARKED" if all(gates.values()) else "BLOCKED_DNR3C9_LEDGER_INCOMPLETE",
    "gate_pass": all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "known_closed": known_closed,
    "missing_info": missing_info,
    "reopen_gates": reopen_gates,
    "do_not_do": do_not_do,
    "current_decision": {
      "native_dnr3c": "parked",
      "reason": "No current local native schedule lever clears material timing gates after resource, PMC, and issue-order confirmation.",
      "minimum_to_resume": "one reopen gate must pass; otherwise continue route-level decode work or oracle/tooling extraction.",
    },
    "gates": gates,
    "input_artifacts": [
      "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7a_resource_ledger_result.json",
      "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7b_pmc_ladder_result.json",
      "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7c_issue_interleaving_result.json",
      "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7d_confirmation_result.json",
      "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c8_tooling_inventory_result.json",
      "bench/q8-ffn-amd-scheduler-project/oracle_contract.json",
      "bench/q8-ffn-amd-scheduler-project/artifact_loader.json",
      "bench/q8-ffn-amd-scheduler-project/n1_attribution.json",
    ],
  }

  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "gate_pass": result["gate_pass"],
    "missing_info": [row["id"] + ":" + row["name"] for row in missing_info],
    "reopen_gates": [row["gate"] for row in reopen_gates],
    "decision": result["current_decision"],
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
