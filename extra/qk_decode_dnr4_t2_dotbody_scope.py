#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_dnr4_t2_dotbody_scope_result.json"


def load(rel: str, default: Any = None) -> Any:
  p = ROOT / rel
  return json.loads(p.read_text()) if p.exists() else default


def main() -> int:
  t1 = load("bench/qk-decode-primitive-transfer/decode_dnr4_t1_timing_result.json", {})
  dnr3c2 = load("bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c2_dataflow_emitter_result.json", {})
  dnr4 = load("bench/qk-decode-primitive-transfer/decode_dnr4_liverange_scope_result.json", {})
  coarse = load("bench/qk-decode-primitive-transfer/decode_oracle_coarse_attribution_result.json", {})

  targets = [
    {
      "id": "T2A-low-band-b128-preload",
      "problem": "DNR-3C2 closed global-load count by using v80-v95 for q4/q8 words, increasing allocated VGPR pressure.",
      "plan": "Reuse low dead registers after scale/min setup for the 8 q4 words and 8 q8 words.",
      "candidate_register_map": {
        "q4_lanes_0_7": "v12-v19",
        "q8_lanes_0_4": "v25-v29",
        "q8_lanes_5_7": "v33-v35",
        "accumulators": "v4-v5",
        "scale_min": "v30-v32 and v36-v37",
        "scratch": "v10-v11",
      },
      "structural_gate": "no v80-v95 band, max static VGPR index <=41, 16 dot4, <=11 grouped global loads",
    },
    {
      "id": "T2B-combine-with-T1-reduction-reuse",
      "problem": "Even if the dot body is packed low, old reduction v50-v54 would reintroduce a high band.",
      "plan": "Use DNR4-T1 low reduction/tail reuse after the packed dot body.",
      "structural_gate": "reduction uses v1-v6/v10-v13, not v50-v54",
    },
    {
      "id": "T2C-real-GGUF-correctness-and-timing",
      "problem": "Synthetic correctness was insufficient for T1 timing; real GGUF correctness must be part of the T2 gate.",
      "plan": "Run synthetic structural correctness first, then same-harness real GGUF timing/correctness.",
      "promotion_gate": "correct plus >=30us vs native, >=15us vs best static, or >=10us vs C7C",
    },
  ]

  gates = {
    "t1_structural_only_recorded": t1.get("verdict") == "BLOCKED_DNR4_T1_STRUCTURAL_ONLY_TIMING_NOT_MATERIAL",
    "dnr3c2_load_shape_correct": dnr3c2.get("gate_pass") is True,
    "dnr4_scope_ready": dnr4.get("gate_pass") is True,
    "oracle_resource_gap_known": coarse.get("gate_pass") is True,
    "targets_named": len(targets) == 3,
    "no_perf_claim": True,
  }
  result = {
    "date": "2026-06-20",
    "phase": "DNR4_T2_DOTBODY_COMPRESSION_SCOPE",
    "schema": "decode_dnr4_t2_dotbody_scope_v1",
    "verdict": "PASS_DNR4_T2_DOTBODY_SCOPE_READY" if all(gates.values()) else "BLOCKED_DNR4_T2_DOTBODY_SCOPE_INCOMPLETE",
    "gate_pass": all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "why_now": [
      "oracle profiled VGPR is 32 while native allocates 56",
      "T1 removed high reduction registers but timing did not move",
      "remaining plausible native-owned resource gap is S2/S3 dot-body vector/live-range packing",
    ],
    "targets": targets,
    "blocked_if": [
      "low-band b128 preload aliases a still-live scale/min or accumulator register",
      "real GGUF correctness fails even if synthetic correctness passes",
      "same-harness timing does not move materially",
    ],
    "next_probe": "extra/qk_decode_dnr4_t2_lowband_preload_probe.py",
    "gates": gates,
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "gates": gates,
    "targets": [x["id"] for x in targets],
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())

