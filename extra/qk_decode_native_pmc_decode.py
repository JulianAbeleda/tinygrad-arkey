#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-native-tooling/pmc_decode.json"


def load(rel: str) -> Any:
  p = ROOT / rel
  return json.loads(p.read_text()) if p.exists() else None


def main() -> int:
  evidence = load("bench/q8-ffn-amd-scheduler-project/pmu_sqtt_evidence.json") or {}
  pmc_timing = load("bench/q8-ffn-amd-scheduler-project/pmu_sqtt_pmc_q8_gateup_full.json") or {}
  dso_pmc = load("bench/q8-ffn-dynamic-scheduler-observability/pmc_q8_gateup_full.json") or {}

  pmc_rows = (((evidence.get("runs") or {}).get("pmc") or {}).get("profile") or {}).get("pmc") or []
  events = []
  for row in pmc_rows:
    for ev in row.get("sample_layout") or []:
      events.append({
        "name": ev.get("name"),
        "raw_values": [],
        "decoded": False,
        "unit": "unknown",
        "blob_bytes": row.get("blob_bytes"),
        "layout": ev,
      })

  result = {
    "schema": "decode_native_pmc_decode_v1",
    "date": "2026-06-19",
    "verdict": "BLOCKED_COUNTER_DECODE",
    "inputs": [
      "bench/q8-ffn-amd-scheduler-project/pmu_sqtt_evidence.json",
      "bench/q8-ffn-amd-scheduler-project/pmu_sqtt_pmc_q8_gateup_full.json",
      "bench/q8-ffn-dynamic-scheduler-observability/pmc_q8_gateup_full.json",
    ],
    "program": {
      "name": "q8_b2b_fullrow_reduce",
      "hash_or_tag": None,
    },
    "events": events,
    "derived": {
      "valu_per_busy_cycle": None,
      "salu_per_busy_cycle": None,
      "l2_hit_rate": None,
      "lds_bank_conflict_rate": None,
    },
    "profile_runnable": bool(((evidence.get("classification") or {}).get("pmc_profile_runnable"))),
    "profile_overhead_timing_ms": ((pmc_timing.get("timing") or {}).get("median_ms")),
    "dso_profile_attempt": dso_pmc.get("verdict"),
    "blocker": "blocked_counter_decode",
    "blocker_detail": (
      "PMC profile events record sample layouts and blob sizes, but the persisted artifacts do not include raw PMC "
      "counter blobs or decoded values. Current tooling cannot compute counter-grade rates from these summaries."
    ),
    "feature_implications": [
      {
        "feature": "register_lifetime",
        "authority": "blocked_counter_decode",
        "movement_us": None,
        "decision": "blocked_counter_decode",
      },
      {
        "feature": "scheduler_markers",
        "authority": "blocked_counter_decode",
        "movement_us": None,
        "decision": "blocked_counter_decode",
      },
    ],
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"out": str(OUT.relative_to(ROOT)), "verdict": result["verdict"]}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
