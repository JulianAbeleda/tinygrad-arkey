#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7_issue_resource_scope_result.json"


def read_json(rel: str) -> dict[str, Any]:
  with (ROOT / rel).open() as f:
    return json.load(f)


def main() -> int:
  dnr3c6 = read_json("bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c6_attribution_scope_result.json")
  n1 = read_json("bench/q8-ffn-amd-scheduler-project/n1_attribution.json")
  oracle = read_json("bench/q8-ffn-amd-scheduler-project/oracle_contract.json")
  tooling = read_json("bench/amd-scheduler-tooling-backend/execution.json")

  timing = dnr3c6["timing_context"]
  variants = {row["name"]: row for row in dnr3c6["variants"]}
  oracle_grouped = oracle["instruction_contract"]["oracle_grouped"]
  best_static_gap_us = timing["best_variant_us"] - timing["oracle_us"]
  native_gap_us = timing["native_us"] - timing["oracle_us"]
  local_static_explained_us = timing["native_us"] - timing["best_variant_us"]
  local_static_explained_pct = local_static_explained_us / native_gap_us if native_gap_us else 0.0

  tracks = [
    {
      "track": "DNR-3C7A static resource ledger",
      "purpose": "Build a comparable native/C4/oracle resource ledger before more codegen.",
      "questions": [
        "Are VGPR/SGPR/private/LDS usage, occupancy, and wave/workgroup shape materially different?",
        "Does DNR-3C4 add enough live VGPR pressure to erase its static count wins?",
      ],
      "required_artifacts": [
        "native and DNR-3C4 disassembly/resource metadata",
        "oracle resource metadata from artifact loader/oracle contract",
        "register live-range summary for q4/q8 preload regs, accumulators, LDS reduction temps",
      ],
      "exit_gate": "Names a resource mismatch with credible >=30us movement, or rules out static resource metadata as sufficient.",
      "status": "next",
    },
    {
      "track": "DNR-3C7B PMC counter attribution",
      "purpose": "Use counters where local SQTT body decode is not usable.",
      "questions": [
        "Is the remaining gap memory wait, issue occupancy, VALU/SALU pressure, cache locality, or LDS conflict?",
        "Do DNR-3C4 counters move in the direction predicted by its static changes?",
      ],
      "required_artifacts": [
        "same-process native vs DNR-3C4 PMC runs with SQ_BUSY_CYCLES, SQ_WAIT_ANY if available, SQ_INSTS_VALU/SALU, GL2C_HIT/MISS, SQC_LDS_*",
        "counter normalization by dispatch count and correctness check",
      ],
      "exit_gate": "One counter family plausibly explains >=30us, or counters show native/C4 remain the same class.",
      "status": "after_C7A",
    },
    {
      "track": "DNR-3C7C issue/interleaving candidate",
      "purpose": "Only after C7A/C7B name a cause, build a schedule that changes issue behavior, not just counts.",
      "questions": [
        "Can q4/q8 loads, unpack/select, dot4, scale conversion, and reduction be interleaved to hide latency?",
        "Can a branch/wait policy help once tied to real dependency/resource evidence?",
      ],
      "required_artifacts": [
        "candidate schedule object with dependency groups and live-range budget",
        "correctness on real GGUF gate/up",
        "same-harness timing versus native, C4, and oracle",
      ],
      "exit_gate": "Correct candidate improves >=30us or reaches <=110% oracle; otherwise native renderer route is parked.",
      "status": "blocked_on_C7A_C7B",
    },
    {
      "track": "DNR-3C7D SQTT/body tooling reopen",
      "purpose": "Repair trace attribution only if PMC/static ledgers cannot answer the issue/resource question.",
      "questions": [
        "Can ROCprofiler/AQLprofile-style body packet capture be bridged back to tinygrad HCQ?",
        "Can PC-level body mapping identify stalls/issue bubbles for the q8 kernel?",
      ],
      "required_artifacts": [
        "body instruction packets for q8_b2b_fullrow_reduce or a formal reason local HCQ SQTT cannot provide them",
        "PC-to-disasm join for native and DNR-3C4",
      ],
      "exit_gate": "Body-level attribution names a >=30us lever; otherwise tooling route stays parked.",
      "status": "optional_reopen_only",
    },
  ]

  do_not_do = [
    "do not add dead branches to match oracle branch count",
    "do not tune s_clause/s_delay_alu counts without a measured marker-placement win",
    "do not reopen load-shape or LDS-reduction count patches as standalone work",
    "do not start BEAM/search: the legal search space is still missing an issue/resource objective",
    "do not promote native DNR-3C4: it remains far behind the oracle despite correctness",
  ]

  gates = {
    "dnr3c6_static_ladder_refuted": dnr3c6.get("verdict") == "BLOCKED_DNR3C6_STATIC_LADDER_REFUTES_LOCAL_COUNT_ATTRIBUTION",
    "local_static_explains_lt_30us": local_static_explained_us < 30.0,
    "best_static_still_gt_oracle_110pct": timing["best_variant_us"] > timing["oracle_us"] * 1.10,
    "n1_sqtt_decode_unusable": n1["gate"].get("sqtt_decode_usable") is False,
    "n1_pmc_runnable": n1["gate"].get("pmc_profile_runnable") is True,
    "oracle_gap_remains_ge_30us": best_static_gap_us >= 30.0,
    "no_renderer_default_change": True,
  }

  result = {
    "date": "2026-06-20",
    "phase": "DNR-3C7_DECODE_ISSUE_RESOURCE_ATTRIBUTION_SCOPE",
    "schema": "decode_native_renderer_dnr3c7_issue_resource_scope_v1",
    "verdict": "SCOPE_DNR3C7_ISSUE_RESOURCE_ATTRIBUTION_READY" if all(gates.values()) else "BLOCKED_DNR3C7_SCOPE_INPUTS_INCONSISTENT",
    "gate_pass": all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "evidence_summary": {
      "native_us": timing["native_us"],
      "oracle_us": timing["oracle_us"],
      "best_static_variant": timing["best_variant"],
      "best_static_variant_us": timing["best_variant_us"],
      "native_gap_us": native_gap_us,
      "best_static_gap_us": best_static_gap_us,
      "local_static_explained_us": local_static_explained_us,
      "local_static_explained_pct": local_static_explained_pct,
      "native_grouped": variants["native_dnr2"]["grouped"],
      "best_static_grouped": variants[timing["best_variant"]]["grouped"],
      "oracle_grouped": oracle_grouped,
      "pmc_available": n1["gate"].get("pmc_profile_runnable"),
      "sqtt_decode_usable": n1["gate"].get("sqtt_decode_usable"),
      "tooling_track_verdict": tooling.get("verdict"),
    },
    "tracks": tracks,
    "promotion_policy": {
      "native_renderer": "not_promotable_from_DNR3C4_or_C6",
      "q8_artifact_oracle": "remains_practical_decode_route",
      "reopen_native_only_if": [
        "C7A/C7B names a credible >=30us issue/resource lever",
        "or a funded backend project accepts broader compiler value beyond this decode primitive",
      ],
    },
    "do_not_do": do_not_do,
    "gates": gates,
    "blocked_at": {
      "next_phase": "DNR-3C7A static resource ledger",
      "reason": "Local static count matching explains only a small fraction of the gap; the next question is issue/resource behavior.",
      "minimum_unblock": [
        "native/C4/oracle resource metadata ledger",
        "register live-range and occupancy comparison",
        "decision whether PMC counter attribution is necessary before any new emitter work",
      ],
    },
    "input_artifacts": [
      "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c6_attribution_scope_result.json",
      "bench/q8-ffn-amd-scheduler-project/n1_attribution.json",
      "bench/q8-ffn-amd-scheduler-project/oracle_contract.json",
      "bench/amd-scheduler-tooling-backend/execution.json",
    ],
  }

  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "gate_pass": result["gate_pass"],
    "evidence": result["evidence_summary"],
    "next_phase": result["blocked_at"]["next_phase"],
    "tracks": [row["track"] for row in tracks],
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
