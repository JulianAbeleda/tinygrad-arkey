#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib
from typing import Any

from extra.amd_scheduler_tooling_backend_execute import OUTDIR, attempt_sqtt_decode, run_capture

OUT = OUTDIR / "t1_body_mapping_proof.json"

CONFIGS: list[dict[str, Any]] = [
  {"name": "baseline", "env": {}},
  {"name": "detail_mode", "env": {"SQTT_MODE": "3"}},
  {"name": "ttrace_exec", "env": {"SQTT_TTRACE_EXEC": "1"}},
  {"name": "detail_mode_ttrace_exec", "env": {"SQTT_MODE": "3", "SQTT_TTRACE_EXEC": "1"}},
]

def summarize(name: str, env: dict[str, str], capture_run: dict[str, Any], decode: dict[str, Any]) -> dict[str, Any]:
  cap = capture_run.get("capture") if capture_run.get("ok") else None
  rows = decode.get("rows", []) if decode.get("attempted") else []
  return {
    "name": name,
    "env": env,
    "capture_ok": capture_run.get("ok", False),
    "returncode": capture_run.get("returncode"),
    "sqtt_events": 0 if cap is None else len(cap.get("sqtt", [])),
    "itrace_events": 0 if cap is None else sum(1 for e in cap.get("sqtt", []) if e.get("itrace")),
    "total_sqtt_blob_bytes": 0 if cap is None else sum(e.get("blob_bytes", 0) for e in cap.get("sqtt", [])),
    "decode_ok_count": decode.get("decode_ok_count"),
    "mapped_instruction_events": decode.get("mapped_instruction_events"),
    "body_instruction_events": decode.get("body_instruction_events"),
    "raw_body_packet_events_top20": decode.get("raw_body_packet_events_top20"),
    "gate_pass": decode.get("gate_pass", False),
    "itrace_packet_tops": [
      {
        "idx": r.get("idx"),
        "raw_packet_counts_top": r.get("raw_packet_counts_top"),
        "mapped_instruction_counts_top": r.get("instruction_counts_top"),
        "error": r.get("error"),
      }
      for r in rows if r.get("itrace")
    ],
    "capture_error": None if cap is not None else (capture_run.get("capture") or {}).get("status", {}).get("error") or capture_run.get("stderr_tail"),
  }

def main() -> int:
  OUTDIR.mkdir(parents=True, exist_ok=True)
  rows = []
  for cfg in CONFIGS:
    capture = run_capture(env_extra=cfg["env"], timeout_s=360)
    decode = attempt_sqtt_decode(capture.get("capture") if capture.get("ok") else None)
    rows.append(summarize(cfg["name"], cfg["env"], capture, decode))

  passing = [r for r in rows if r["gate_pass"]]
  result = {
    "date": "2026-06-19",
    "phase": "T1_sqtt_body_mapping_proof",
    "purpose": "Prove whether local RDNA3 HCQ SQTT register knobs can emit q8 body instruction packets.",
    "configs": rows,
    "gate": {
      "required": "raw body packets and mapped non-S_ENDPGM body instructions for q8_b2b_fullrow_reduce",
      "passing_configs": [r["name"] for r in passing],
    },
    "verdict": "PASS_BODY_MAPPING" if passing else "NO_LOCAL_REGISTER_KNOB_BODY_MAPPING",
    "decision": (
      "Local SQTT mode/ttrace_exec knobs do not produce q8 body instruction packets. The next fix is not another "
      "decoder tweak; use ROCprofiler/AQLprofile packet generation or reverse the missing register sequence from it."
      if not passing else "Use the passing config for T4 attribution."
    ),
  }
  OUT.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps({"out": str(OUT), "verdict": result["verdict"], "passing_configs": result["gate"]["passing_configs"]}, indent=2))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
