#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_dual_track_next_scope_result.json"


def load(rel: str, default: Any = None) -> Any:
  p = ROOT / rel
  return json.loads(p.read_text()) if p.exists() else default


def main() -> int:
  att = load("bench/qk-decode-primitive-transfer/decode_oracle_att_result.json", {})
  route = load("bench/qk-decode-primitive-transfer/decode_route_decision_closeout_result.json", {})
  t3 = load("bench/qk-decode-primitive-transfer/decode_dnr4_t3_candidate_grid_result.json", {})
  primitive = load("bench/qk-decode-primitive-transfer/decode_primitive_transfer_result.json", {})
  tools = load("bench/qk-decode-primitive-transfer/decode_tools_pending_scope_result.json", {})

  tracks = [
    {
      "id": "ATT_PC_TIMELINE",
      "status": "external_tooling_blocked",
      "purpose": "turn oracle/native decode from static stage maps into PC/stage stall attribution",
      "can_progress_now": False,
      "current_blocker": "rocprof trace decoder library missing",
      "local_next": "enhance the ATT probe to record decoder search paths and rerun when the decoder .so is installed",
      "unblock_gate": "decode_oracle_att_result.gates.decoder_library_present == true and att_outputs_present == true",
      "output_needed": [
        "decoded ATT packets",
        "PC to ISA join",
        "PC to S0-S5 semantic stage join",
        "dominant stall class with credible >=30us native upside",
      ],
    },
    {
      "id": "ROUTE_LEVEL_DECODE_PRIMITIVE",
      "status": "local_work_ready",
      "purpose": "avoid more local native schedule count-matching by auditing/promoting a route-level primitive",
      "can_progress_now": True,
      "current_blocker": "owned primitive contract and promotion/readiness ledger are split across q8 artifact, default decode, and imported Q4 route docs",
      "local_next": "build a route-level primitive ledger that compares current default, q8 artifact opt-in, imported Q4, native MMVQ, and future owned q8 lifecycle",
      "unblock_gate": "one route has quality policy, lifecycle costs, fallback behavior, and W==D/token timing sufficient for promotion or explicit rejection",
      "output_needed": [
        "route table with lifecycle, quality, timing, ownership, default policy",
        "promotion gates for q8 artifact and any owned successor",
        "rejection gates for imported Q4 and native local-schedule-only work",
        "search objective if a route becomes lowerable and measurable",
      ],
    },
  ]

  execution_order = [
    {
      "step": "D5A-route-primitive-ledger",
      "track": "ROUTE_LEVEL_DECODE_PRIMITIVE",
      "do_now": True,
      "why": "local artifacts are present and this can decide whether decode has a non-native-schedule route to promote",
      "probe": "extra/qk_decode_route_level_primitive_ledger.py",
    },
    {
      "step": "D5B-att-unblock-audit",
      "track": "ATT_PC_TIMELINE",
      "do_now": True,
      "why": "we can verify local ROCm state and document the exact missing decoder dependency, but cannot produce ATT packets without the library",
      "probe": "extra/qk_decode_att_unblock_audit.py",
    },
    {
      "step": "D5C-route-or-att-decision",
      "track": "BOTH",
      "do_now": False,
      "why": "depends on D5A/D5B: promote/reject route primitive if enough evidence, otherwise require ATT PC timeline before native rewrites",
      "probe": "extra/qk_decode_dual_track_decision.py",
    },
  ]

  gates = {
    "att_blocked_on_decoder_recorded": (
      att.get("verdict") == "BLOCKED_DECODE_ORACLE_ATT_DECODER_LIBRARY_MISSING"
      and att.get("gates", {}).get("decoder_library_present") is False
    ),
    "route_closeout_passed": route.get("gate_pass") is True,
    "t3_native_local_rewrite_exhausted": (
      t3.get("verdict") == "BLOCKED_DNR4_T3_NO_MATERIAL_NATIVE_LEVER_UNBLOCK_ATT"
      and t3.get("gates", {}).get("all_variants_correct") is True
      and t3.get("gates", {}).get("counter_predictive_signal") is False
    ),
    "primitive_transfer_scoped": primitive.get("gate_pass") is True,
    "tooling_scope_ready": tools.get("gate_pass") is True,
    "two_tracks_named": len(tracks) == 2,
    "route_can_progress_while_att_blocked": tracks[1]["can_progress_now"] is True and tracks[0]["can_progress_now"] is False,
  }

  result = {
    "date": "2026-06-20",
    "phase": "DECODE_DUAL_TRACK_NEXT_SCOPE",
    "schema": "decode_dual_track_next_scope_v1",
    "verdict": "PASS_DECODE_DUAL_TRACK_NEXT_SCOPE_READY" if all(gates.values()) else "BLOCKED_DECODE_DUAL_TRACK_NEXT_SCOPE_INCOMPLETE",
    "gate_pass": all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "answer": "yes_do_both_but_with_separate_gates",
    "tracks": tracks,
    "execution_order": execution_order,
    "do_not_do": [
      "do not resume native local schedule rewrites without ATT or a route-level primitive gate",
      "do not start BEAM/search until a lowerable primitive has a measurable objective",
      "do not block route-level decode work on the missing ATT decoder library",
      "do not promote q8 artifact default-on without quality/fallback/policy acceptance",
    ],
    "gates": gates,
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "gates": gates,
    "execution_order": [x["step"] for x in execution_order],
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
