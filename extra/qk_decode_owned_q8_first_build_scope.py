#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_owned_q8_first_build_scope_result.json"


def load(rel: str, default: Any = None) -> Any:
  p = ROOT / rel
  return json.loads(p.read_text()) if p.exists() else default


def main() -> int:
  parity = load("bench/qk-decode-primitive-transfer/decode_owned_q8_artifact_parity_harness_result.json", {})
  t3 = load("bench/qk-decode-primitive-transfer/decode_dnr4_t3_candidate_grid_result.json", {})
  att = load("bench/qk-decode-primitive-transfer/decode_att_unblock_audit_result.json", {})
  successor = load("bench/qk-decode-primitive-transfer/decode_owned_q8_lifecycle_successor_object_result.json", {})

  build_tracks = [
    {
      "id": "owned_producer_cache",
      "decision": "DO_FIRST",
      "why": "route-level q8 lifecycle is defined by producer/cache reuse, quality, and fallback; it can be built and validated before native consumer scheduling is reopened",
      "minimum_probe": "extra/qk_decode_owned_q8_producer_cache_scope.py",
      "exit_gate": "byte/scale semantics scoped, reuse=2 contract preserved, dNLL/fallback gates named",
      "blocked_by_att": False,
    },
    {
      "id": "owned_gateup_consumer",
      "decision": "PARK_FOR_NOW",
      "why": "local native consumer schedule work just failed material timing and PMC did not form a search objective",
      "minimum_reopen": "ATT PC timeline or new route-level lowerable objective",
      "blocked_by_att": True,
    },
  ]

  gates = {
    "parity_harness_ready": parity.get("gate_pass") is True,
    "successor_object_ready": successor.get("gate_pass") is True,
    "native_consumer_schedule_parked": t3.get("verdict") == "BLOCKED_DNR4_T3_NO_MATERIAL_NATIVE_LEVER_UNBLOCK_ATT",
    "att_blocked_recorded": att.get("verdict") == "BLOCKED_DECODE_ATT_DECODER_SO_MISSING",
    "producer_selected_first": build_tracks[0]["decision"] == "DO_FIRST",
    "consumer_parked_with_reopen_gate": build_tracks[1]["decision"] == "PARK_FOR_NOW",
  }
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_OWNED_Q8_FIRST_BUILD_SCOPE",
    "schema": "decode_owned_q8_first_build_scope_v1",
    "verdict": "PASS_DECODE_OWNED_Q8_FIRST_BUILD_SCOPE_READY" if all(gates.values()) else "BLOCKED_DECODE_OWNED_Q8_FIRST_BUILD_SCOPE_INCOMPLETE",
    "gate_pass": all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "build_tracks": build_tracks,
    "next_executable_probe": "extra/qk_decode_owned_q8_producer_cache_scope.py",
    "do_not_do": [
      "do not restart consumer schedule count matching",
      "do not search consumer schedules before ATT or a lowerable objective",
      "do not default-on q8 artifact",
      "do not conflate producer/cache ownership with artifact ownership",
    ],
    "gates": gates,
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "gates": gates,
    "next_executable_probe": result["next_executable_probe"],
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
