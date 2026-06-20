#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
from types import SimpleNamespace
from typing import Any

from tinygrad.device import Device
from extra.q8_ffn_artifact_import_route import FixedLaunchRunner
from extra.q8_ffn_fast_artifact_probe import HIP_MMVQ_GATEUP_SOURCE, compile_hipcc_linked, perf_gateup
from extra.q8_ffn_hcq_artifact import NORM_SOURCE


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_owned_q8_mixed_lifecycle_result.json"


def load(rel: str, default: Any = None) -> Any:
  p = ROOT / rel
  return json.loads(p.read_text()) if p.exists() else default


def main() -> int:
  ap = argparse.ArgumentParser(description="Mixed owned-producer + artifact gate/up lifecycle probe")
  ap.add_argument("--gguf", type=pathlib.Path, default=pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  ap.add_argument("--arch", default="gfx1100")
  ap.add_argument("--rows", type=int, default=12288)
  ap.add_argument("--seed", type=int, default=7)
  ap.add_argument("--warmups", type=int, default=8)
  ap.add_argument("--iters", type=int, default=20)
  ap.add_argument("--out", type=pathlib.Path, default=OUT)
  args = ap.parse_args()

  scope = load("bench/qk-decode-primitive-transfer/decode_owned_q8_mixed_lifecycle_scope_result.json", {})
  artifact = load("bench/q8-ffn-amd-scheduler-project/artifact_loader.json", {})
  target_us = (scope.get("candidate") or {}).get("target_lifecycle_us") or ((artifact.get("perf_gateup") or {}).get("gate_up_lifecycle_us"))

  dev = Device["AMD"]
  prod_prg = FixedLaunchRunner(dev.runtime("owned_q8_mixed_producer_comgr", dev.compiler.compile(NORM_SOURCE)), (1, 1, 1), (256, 1, 1))
  gateup_blob = compile_hipcc_linked(HIP_MMVQ_GATEUP_SOURCE, args.arch)
  gateup_prg = FixedLaunchRunner(dev.runtime("owned_q8_mixed_gateup_hcq_artifact", gateup_blob), (args.rows, 2, 1), (32, 4, 1))
  perf_args = SimpleNamespace(gguf=args.gguf, rows=args.rows, seed=args.seed, warmups=args.warmups, iters=args.iters, producer_threads=256)
  perf = perf_gateup(perf_args, prod_prg, gateup_prg)
  lifecycle_us = perf["gate_up_lifecycle_us"]
  gates = {
    "scope_ready": scope.get("gate_pass") is True,
    "producer_correct": perf["gates"].get("producer_correct") is True,
    "gate_correct": perf["gates"].get("gate_correct") is True,
    "up_correct": perf["gates"].get("up_correct") is True,
    "mixed_lifecycle_lte_artifact": lifecycle_us <= target_us,
  }
  if not all(v for k, v in gates.items() if k != "mixed_lifecycle_lte_artifact"):
    verdict = "BLOCKED_DECODE_OWNED_Q8_MIXED_LIFECYCLE_INCORRECT"
  elif gates["mixed_lifecycle_lte_artifact"]:
    verdict = "PASS_DECODE_OWNED_Q8_MIXED_LIFECYCLE_BEATS_ARTIFACT"
  else:
    verdict = "BLOCKED_DECODE_OWNED_Q8_MIXED_LIFECYCLE_NOT_MATERIAL"
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_OWNED_Q8_MIXED_LIFECYCLE",
    "schema": "decode_owned_q8_mixed_lifecycle_v1",
    "verdict": verdict,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": True,
    "candidate": {
      "producer": "owned COMGR NORM_SOURCE via tinygrad HCQ",
      "consumer": "hipcc/LLD q8_mmvq_gateup via tinygrad HCQ",
      "ownership": "mixed",
    },
    "perf_gateup": perf,
    "target_lifecycle_us": target_us,
    "delta_vs_artifact_lifecycle_us": target_us - lifecycle_us,
    "gates": gates,
    "decision": {
      "if_pass": "Producer ownership improves HCQ artifact lifecycle; next scope fully-owned consumer separately.",
      "if_blocked": "Keep owned producer as standalone HCQ-parity row; mixed lifecycle did not improve route target.",
    },
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "lifecycle_us": lifecycle_us,
    "target_lifecycle_us": target_us,
    "delta_vs_artifact_lifecycle_us": result["delta_vs_artifact_lifecycle_us"],
    "gates": gates,
    "out": str(args.out.relative_to(ROOT) if args.out.is_absolute() and args.out.is_relative_to(ROOT) else args.out),
  }, indent=2))
  return 0 if gates["scope_ready"] and gates["producer_correct"] and gates["gate_correct"] and gates["up_correct"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
