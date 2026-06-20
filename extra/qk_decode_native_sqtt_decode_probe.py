#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-native-tooling/timeline_attribution.json"


def load(rel: str) -> Any:
  p = ROOT / rel
  return json.loads(p.read_text()) if p.exists() else None


def main() -> int:
  evidence = load("bench/q8-ffn-amd-scheduler-project/pmu_sqtt_evidence.json") or {}
  ffn_gate = load("bench/qk-att-inmodel-role-join/ffn_gate.json") or {}
  sqtt_events = (((evidence.get("runs") or {}).get("sqtt") or {}).get("profile") or {}).get("sqtt") or []
  decode_attempts = [ev.get("decode_summary") for ev in sqtt_events if ev.get("decode_summary") is not None]
  errors = sorted({d.get("error") for d in decode_attempts if d and not d.get("ok") and d.get("error")})
  ok = any(d and d.get("ok") for d in decode_attempts)
  main_programs = [p for p in (((ffn_gate.get("programs") or {}).get("variants")) or [])
                   if p.get("program_name") == "q4k_gemv_partial_12288_4096_1"]
  result = {
    "schema": "decode_native_timeline_attribution_v1",
    "date": "2026-06-19",
    "verdict": "PASS_TIMELINE_ATTRIBUTION" if ok else "BLOCKED_TIMELINE_DECODE",
    "program_join": {
      "role": "ffn_gate/up",
      "program_name": "q4k_gemv_partial_12288_4096_1",
      "lib_sha16": (main_programs[0].get("lib_sha16") if main_programs else "236fd9e8841b577f"),
    },
    "decoder": {
      "path": "local_sqtt",
      "ok": ok,
      "error": "; ".join(errors) if errors else None,
    },
    "timeline": [],
    "sqtt_event_count": len(sqtt_events),
    "itrace_event_count": sum(1 for ev in sqtt_events if ev.get("itrace")),
    "total_blob_bytes": sum(ev.get("blob_bytes", 0) for ev in sqtt_events),
    "blocker": None if ok else "blocked_timeline_decode",
    "blocker_detail": None if ok else (
      "SQTT capture is runnable, but persisted decode attempts fail on RDNA3 instruction-trace blobs. "
      "No decoded q8 instruction/resource timeline is available for feature attribution."
    ),
    "feature_implications": [
      {
        "feature": "scheduler_markers",
        "authority": "blocked_timeline_decode" if not ok else "counter_grade",
        "movement_us": None,
        "decision": "blocked_timeline_decode" if not ok else "needs_feature_budget",
      },
      {
        "feature": "register_lifetime",
        "authority": "blocked_timeline_decode" if not ok else "counter_grade",
        "movement_us": None,
        "decision": "blocked_timeline_decode" if not ok else "needs_feature_budget",
      },
    ],
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"out": str(OUT.relative_to(ROOT)), "verdict": result["verdict"]}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
