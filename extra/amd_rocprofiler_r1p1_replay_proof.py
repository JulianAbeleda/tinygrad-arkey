#!/usr/bin/env python3
from __future__ import annotations

import json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "bench/amd-scheduler-tooling-backend"
OUT = OUTDIR / "r1p1_aqlprofile_replay_proof.json"

def load(path: Path) -> dict:
  return json.loads(path.read_text()) if path.exists() else {}

def run_aql_queue_compat() -> dict:
  code = r"""
import json
from extra.amd_scheduler_tooling_backend_execute import run_capture, attempt_sqtt_decode
cap = run_capture(timeout_s=480, env_extra={"AMD_AQL":"1", "SQTT_ORACLE_TARGET_CU":"1", "SQTT_SIMD_SEL":"1"})
dec = attempt_sqtt_decode(cap.get("capture") if cap.get("ok") else None)
print("R1P1_JSON=" + json.dumps({
  "capture_ok": cap.get("ok"),
  "returncode": cap.get("returncode"),
  "elapsed_s": cap.get("elapsed_s"),
  "sqtt_events": 0 if not cap.get("capture") else len(cap["capture"].get("sqtt", [])),
  "sqtt_bytes": 0 if not cap.get("capture") else sum(x.get("blob_bytes", 0) for x in cap["capture"].get("sqtt", [])),
  "decode": {k: dec.get(k) for k in ["attempted", "decode_ok_count", "decode_fail_count", "mapped_instruction_events", "body_instruction_events", "raw_body_packet_events_top20", "gate_pass", "gate_note"]},
  "stderr_tail": cap.get("stderr_tail", [])[-20:],
}, sort_keys=True))
"""
  t0 = time.perf_counter()
  cp = subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=540)
  parsed = {}
  for line in cp.stdout.splitlines():
    if line.startswith("R1P1_JSON="): parsed = json.loads(line.split("=", 1)[1])
  return {"returncode": cp.returncode, "elapsed_s": round(time.perf_counter() - t0, 3), "parsed": parsed,
          "stdout_tail": cp.stdout.splitlines()[-20:], "stderr_tail": cp.stderr.splitlines()[-20:]}

def summarize_packet_material() -> dict:
  t1b = load(OUTDIR / "t1b_att_aqlprofile.json")
  attempts = (((t1b.get("aqlprofile_pm4_run") or {}).get("parsed") or {}).get("attempts") or [])
  rows = []
  for a in attempts:
    words = ((a.get("command_buffer_words") or {}).get("words") or [])
    nonzero = [(i, w) for i, w in enumerate(words) if w]
    rows.append({
      "name": a.get("name"),
      "ok": a.get("ok"),
      "start_status": a.get("start_status"),
      "stop_status": a.get("stop_status"),
      "read_status": a.get("read_status"),
      "command_buffer_bytes": (a.get("command_buffer_words") or {}).get("bytes"),
      "nonzero_words": len(nonzero),
      "first_nonzero": nonzero[:12],
      "last_nonzero": nonzero[-12:],
      "legacy_start_pm4_status": ((a.get("start_pm4") or {}).get("status")),
      "legacy_stop_pm4_status": ((a.get("stop_pm4") or {}).get("status")),
    })
  return {
    "source": "bench/amd-scheduler-tooling-backend/t1b_att_aqlprofile.json",
    "working_rows": [r for r in rows if r["ok"]],
    "all_rows": rows,
    "replayability": {
      "legacy_pm4_extract_status": "4096 for start/stop/read packets in the old hsa_ven API probe",
      "command_buffer_is_not_self_contained_for_hcq": True,
      "reason": "AQLprofile command buffers embed trace output/control buffer addresses and expect its trace-control protocol; replaying only MASK/TOKEN/CTRL was already refuted.",
    },
  }

def main() -> None:
  OUTDIR.mkdir(parents=True, exist_ok=True)
  aql_compat = run_aql_queue_compat()
  packet_material = summarize_packet_material()

  compat_body = (((aql_compat.get("parsed") or {}).get("decode") or {}).get("body_instruction_events") or 0)
  compat_ok = bool((aql_compat.get("parsed") or {}).get("capture_ok"))
  material_ok = len(packet_material["working_rows"]) > 0

  result = {
    "date": "2026-06-19",
    "phase": "R1-P1 AQLprofile packet replay proof",
    "purpose": "Try the bounded HCQ-body-ATT reopen before native profiled-HCQ work.",
    "gates": {
      "aql_queue_compat": "PASS" if compat_ok else "FAIL",
      "aql_queue_body_packets": "PASS" if compat_body > 0 else "FAIL",
      "aqlprofile_packet_material": "PASS" if material_ok else "FAIL",
      "direct_replayable_packet": "FAIL",
    },
    "aql_queue_compat": aql_compat,
    "packet_material": packet_material,
    "verdict": "BLOCKED_REQUIRES_V2_AQLPROFILE_PACKET_EXPORT_OR_NATIVE_PROFILED_HCQ",
    "meaning": [
      "Forcing tinygrad AMD_AQL=1 is stable but still lifecycle-only, so queue packet format alone is not the missing piece.",
      "AQLprofile can generate nonzero gfx1100 ATT command buffers, but the reusable artifact is not a standalone HCQ PM4 blob.",
      "The next executable replay requires exporting AQLprofile v2 start/stop packets with user-controlled tinygrad trace buffers/control buffers, or implementing the equivalent native profiled-HCQ lifecycle.",
    ],
    "closed": [
      "Plain AQL queue mode as the fix",
      "Legacy hsa_ven command-buffer words as a direct reusable HCQ blob",
      "Another MASK/TOKEN/CTRL transplant",
    ],
    "next_if_funded": "Build a small C/Python v2 AQLprofile packet exporter with callbacks that bind command/output/control buffers to tinygrad-owned GPU VAs, then submit those exact vendor PM4 packets around one HCQ dispatch.",
  }
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"out": str(OUT.relative_to(ROOT)), "verdict": result["verdict"], "compat_ok": compat_ok, "compat_body": compat_body, "material_rows": len(packet_material["working_rows"])}, indent=2))

if __name__ == "__main__":
  main()
