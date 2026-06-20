#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_dnr4_t3_issue_latency_scope_result.json"


def load(rel: str, default: Any = None) -> Any:
  p = ROOT / rel
  return json.loads(p.read_text()) if p.exists() else default


def main() -> int:
  t2 = load("bench/qk-decode-primitive-transfer/decode_dnr4_t2_lowband_preload_result.json", {})
  c7d = load("bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7d_confirmation_result.json", {})
  c7b = load("bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7b_pmc_ladder_result.json", {})
  semantic = load("bench/qk-decode-primitive-transfer/decode_oracle_semantic_map_result.json", {})
  att = load("bench/qk-decode-primitive-transfer/decode_oracle_att_result.json", {})
  ledger = load("bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c9_new_info_ledger_result.json", {})

  available_tools = [
    {
      "id": "same_harness_timing",
      "status": "ready",
      "answers": "candidate-relative wall-time movement across native, best static, C7C, T2, and combined candidates",
    },
    {
      "id": "native_pmc_issue_wait_cache",
      "status": "ready_directional",
      "answers": "SQ_WAIT_ANY, SQ_BUSY, VALU/SALU, cache, and LDS movement for native-side candidates",
    },
    {
      "id": "oracle_semantic_stage_map",
      "status": "ready_static",
      "answers": "which instructions belong to S0-S5, especially S3 load/unpack/dot/scale/fma",
    },
    {
      "id": "att_pc_timeline",
      "status": "blocked_decoder_library_missing",
      "answers": "exact PC/stage stall attribution for oracle and native",
    },
  ]

  experiments = [
    {
      "id": "T3A-candidate-grid",
      "purpose": "Stop evaluating one schedule in isolation; compare native, best static, C7C, T2, and T2+C7C in one matrix.",
      "measure": ["correctness", "same-harness median_us", "delta_vs_native", "delta_vs_best_static", "delta_vs_c7c"],
      "promotion_gate": "correct and >=30us vs native, >=15us vs best static, or >=10us vs C7C",
    },
    {
      "id": "T3B-native-PMC-correlation",
      "purpose": "Check whether issue/wait counters move with time or merely move directionally.",
      "measure": ["SQ_WAIT_ANY", "SQ_BUSY_CYCLES", "VALU", "SALU", "LDS_ACTIVE", "LDS_BANK_CONFLICT", "GL2 cache counters when available"],
      "pass_condition": "at least one counter family orders the timing winners monotonically enough to predict a material candidate",
    },
    {
      "id": "T3C-combined-issue-shape",
      "purpose": "Test the only native-side constructive candidate left by T2: low-band preload plus C7C unpack-all-then-dot ordering plus T1 low reduction.",
      "blocked_if": "register aliases break correctness, VGPR pressure rises materially, or timing remains within noise of T2/C7C",
    },
    {
      "id": "T3D-ATT-unblock",
      "purpose": "If the candidate grid is not material, install/provide the ATT decoder library and rerun PC-stage timeline attribution before more rewrites.",
      "blocked_on": "decode_oracle_att_result.decoder_library_present == false",
    },
  ]

  do_not_do = [
    "do not start BEAM/search from static shape similarity",
    "do not add more load-count, wait-count, branch-count, LDS-count, or marker-count patches without counter/timing attribution",
    "do not reopen Q4_K addressing, q8 addressing, scale/min extraction, dot4 selection, or gate/up correctness",
    "do not promote DNR4-T2 from structural correctness; its real timing was not material",
    "do not claim oracle PC-level attribution until ATT produces decoded timeline packets",
  ]

  gates = {
    "t2_structural_correct_timing_not_material": (
      t2.get("verdict") == "BLOCKED_DNR4_T2_LOWBAND_CORRECT_TIMING_NOT_MATERIAL"
      and t2.get("gates", {}).get("real_timing_all_correct") is True
      and t2.get("gates", {}).get("real_timing_material") is False
    ),
    "c7d_issue_signal_not_material": (
      c7d.get("verdict") == "BLOCKED_DNR3C7D_C7C_SIGNAL_NOT_REPRODUCED_PARK_NATIVE_ROUTE"
      and c7d.get("gates", {}).get("pmc_confirms_wait_or_busy") is True
      and c7d.get("gates", {}).get("timing_material") is False
    ),
    "pmc_ladder_available": c7b.get("gate_pass") is True,
    "semantic_s3_map_available": semantic.get("gate_pass") is True,
    "att_blocker_recorded": (
      att.get("verdict") == "BLOCKED_DECODE_ORACLE_ATT_DECODER_LIBRARY_MISSING"
      and att.get("gates", {}).get("decoder_library_present") is False
    ),
    "new_info_ledger_exhausted": ledger.get("gate_pass") is True,
    "experiments_named": len(experiments) == 4,
    "search_blocked_until_objective": True,
  }

  result = {
    "date": "2026-06-20",
    "phase": "DNR4_T3_ISSUE_LATENCY_ATTRIBUTION_SCOPE",
    "schema": "decode_dnr4_t3_issue_latency_scope_v1",
    "verdict": "PASS_DNR4_T3_ISSUE_LATENCY_SCOPE_READY" if all(gates.values()) else "BLOCKED_DNR4_T3_ISSUE_LATENCY_SCOPE_INCOMPLETE",
    "gate_pass": all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "why_now": [
      "DNR4-T2 made the q4/q8 low-band preload correct but timing was not material",
      "DNR3-C7D confirmed C7C issue-order PMC movement but not enough wall-time movement",
      "the next useful decode step is attribution/correlation, not another static count-matching patch",
    ],
    "timing_context": {
      "t2": t2.get("timing_context", {}),
      "c7d": c7d.get("timing_context", {}),
    },
    "available_tools": available_tools,
    "experiments": experiments,
    "next_executable_probe": "extra/qk_decode_dnr4_t3_candidate_grid_probe.py",
    "next_probe_contract": {
      "variants": ["native", "best_static", "c7c", "dnr4_t2", "dnr4_t2_plus_c7c_if_buildable"],
      "required_outputs": ["correctness", "median_us", "pmc_issue_wait_cache", "candidate_delta_table"],
      "hard_stop": "if no timing/counter family points to a material lever, stop native rewrites and unblock ATT",
    },
    "blocked_until_att_for": [
      "oracle PC-level stall attribution",
      "native-vs-oracle stage timeline",
      "exact S3 wait/load/dot/scale PC blame",
    ],
    "do_not_do": do_not_do,
    "gates": gates,
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "gates": gates,
    "next_executable_probe": result["next_executable_probe"],
    "experiments": [x["id"] for x in experiments],
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
