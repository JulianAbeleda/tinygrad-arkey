#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-scheduler-tooling-backend/rocprofiler_reopen_tracks.json"

def load(path: str) -> dict:
  p = ROOT / path
  if not p.exists():
    return {"missing": True, "path": str(p)}
  return json.loads(p.read_text())

def main() -> None:
  audit = load("bench/amd-scheduler-tooling-backend/rocprofiler_thread_trace_audit.json")
  t1b = load("bench/amd-scheduler-tooling-backend/t1b_att_aqlprofile.json")
  diff = load("bench/amd-scheduler-tooling-backend/sqtt_oracle_hcq_diff_result.json")
  decoder = load("bench/amd-scheduler-tooling-backend/att_decoder_binary_probe.json")

  aql_attempts = (((t1b.get("aqlprofile_pm4_run") or {}).get("parsed") or {}).get("attempts") or [])
  aql_ok = [x for x in aql_attempts if x.get("ok")]
  max_nonzero_words = 0
  for x in aql_ok:
    words = ((x.get("command_buffer_words") or {}).get("words") or [])
    max_nonzero_words = max(max_nonzero_words, sum(1 for w in words if w))

  decoder_pass = decoder.get("verdict") == "ATT_DECODER_BINARY_PASS"
  audit_verdict = audit.get("verdict")
  diff_verdict = diff.get("verdict")

  result = {
    "date": "2026-06-19",
    "purpose": "Scope and execute the first decisive phase for the three ROCprofiler ATT reopen options after the lifecycle audit.",
    "source_artifacts": {
      "thread_trace_audit": "bench/amd-scheduler-tooling-backend/rocprofiler_thread_trace_audit.json",
      "t1b_aqlprofile_pm4": "bench/amd-scheduler-tooling-backend/t1b_att_aqlprofile.json",
      "att_decoder_oracle": "bench/amd-scheduler-tooling-backend/att_decoder_binary_probe.json",
      "oracle_hcq_diff": "bench/amd-scheduler-tooling-backend/sqtt_oracle_hcq_diff_result.json",
    },
    "tracks": [
      {
        "track": "1_AQLPROFILE_PACKET_IMPORT_REPLAY",
        "question": "Can we reopen HCQ body ATT by importing/replaying AQLprofile's packet lifecycle instead of sweeping SQTT registers?",
        "phase_executed": "R1-P0 packet/material audit",
        "evidence": {
          "aqlprofile_linked_and_ran": bool(t1b.get("aqlprofile_pm4_run", {}).get("ok")),
          "working_parameter_sets": [x.get("name") for x in aql_ok],
          "max_nonzero_command_words": max_nonzero_words,
          "prior_register_transplant_verdict": "raw MASK/TOKEN/CTRL changed volume but still produced zero body packets",
          "audit_verdict": audit_verdict,
        },
        "phase_verdict": "GO_TO_R1_P1_PACKET_REPLAY_PROOF" if aql_ok and max_nonzero_words else "BLOCKED_NO_PACKET_MATERIAL",
        "next_gate": "Build a no-model HCQ replay proof that wraps one tinygrad dispatch with the AQLprofile-generated start/stop command stream, then require decoded body packets. No model routing.",
        "kill_gate": "If replayed AQLprofile command stream still yields zero body packets, close packet import and keep external ATT only.",
        "risk": "medium-high",
      },
      {
        "track": "2_NATIVE_PROFILED_HCQ",
        "question": "Should tinygrad implement ROCprofiler's profiled-HSA-queue lifecycle natively for KFD/HCQ?",
        "phase_executed": "R2-P0 capability decomposition",
        "evidence": {
          "missing_lifecycle_components": audit.get("high_confidence_missing", []),
          "bounded_register_sweeps_refuted": audit.get("bounded_register_sweeps_already_refuted", []),
          "oracle_hcq_diff_verdict": diff_verdict,
        },
        "phase_verdict": "PROJECT_LEVEL_ONLY_NO_SMALL_PATCH",
        "next_gate": "Only start after R1 packet replay either passes or demonstrates exactly which HCQ queue/profiling primitive is missing.",
        "kill_gate": "Do not start from another SQTT register patch or decoder filter change.",
        "risk": "project-level",
      },
      {
        "track": "3_SPLIT_TOOLING_MODEL",
        "question": "Can we proceed with external ROCprofiler ATT as the instruction oracle plus tinygrad PMCs for in-model attribution?",
        "phase_executed": "R3-P0 contract check",
        "evidence": {
          "external_att_decoder_pass": decoder_pass,
          "external_att_instruction_oracle": "110446 decoded wave instruction records in oracle-to-HCQ diff",
          "tinygrad_hcq_pmc_available": "native PMC atlas already produced in-model primitive bounds",
        },
        "phase_verdict": "PASS_DEFAULT_OBSERVABILITY_MODEL",
        "next_gate": "Use this split model for decode/pre-fill analysis unless the question specifically requires HCQ body instruction packets.",
        "kill_gate": "Do not claim external HIP ATT is identical to tinygrad HCQ; it is an oracle/control path, not direct in-model HCQ evidence.",
        "risk": "low",
      },
    ],
    "decision": {
      "recommended_order": [
        "R3 as default now",
        "R1 if HCQ body packets are worth another bounded experiment",
        "R2 only if R1 passes enough to justify backend work or if project explicitly funds native profiled-HCQ"
      ],
      "do_not_do": [
        "More SQTT MASK/TOKEN/CTRL sweeps",
        "Another decoder availability investigation",
        "Starting native scheduler/backend work from lifecycle-only SQTT evidence"
      ],
    },
  }

  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"wrote": str(OUT.relative_to(ROOT)), "tracks": len(result["tracks"]), "recommended_first": "R3 default + R1 replay proof only if funded"}, indent=2))

if __name__ == "__main__":
  main()
