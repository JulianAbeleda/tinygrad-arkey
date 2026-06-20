#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, random, statistics, subprocess
from typing import Any

from tinygrad.device import Device
from extra.q8_ffn_artifact_import_route import FixedLaunchRunner
from extra.q8_ffn_fast_artifact_probe import HIP_MMVQ_GATEUP_SOURCE, compile_hipcc_linked
from extra.qk_decode_owned_q8_interleaved_lifecycle_gate import Fixture, clock_sample, gateup_once, producer_once, rel, summarize
from extra.qk_decode_q8_producer_delta_variants import comgr_norm_source


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_q8_nt1024_lifecycle_gate_result.json"


def git_sha() -> str:
  try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
  except Exception: return "unknown"


def main() -> int:
  ap = argparse.ArgumentParser(description="Decode q8 mixed lifecycle gate with COMGR NT1024 producer")
  ap.add_argument("--gguf", type=pathlib.Path, default=pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  ap.add_argument("--arch", default="gfx1100")
  ap.add_argument("--rows", type=int, default=12288)
  ap.add_argument("--seed", type=int, default=101)
  ap.add_argument("--warmups", type=int, default=8)
  ap.add_argument("--rounds", type=int, default=24)
  ap.add_argument("--target-us", type=float, default=115.24)
  ap.add_argument("--producer-target-us", type=float, default=21.72)
  ap.add_argument("--producer-material-delta-us", type=float, default=2.0)
  ap.add_argument("--out", type=pathlib.Path, default=OUT)
  args = ap.parse_args()

  dev = Device["AMD"]
  prod_prg = FixedLaunchRunner(dev.runtime("decode_q8_nt1024_lifecycle_producer", dev.compiler.compile(comgr_norm_source(1024))),
                               (1, 1, 1), (1024, 1, 1))
  gateup_blob = compile_hipcc_linked(HIP_MMVQ_GATEUP_SOURCE, args.arch)
  fx = Fixture(args.gguf, args.rows, args.seed)
  gateup_prg = FixedLaunchRunner(dev.runtime("decode_q8_nt1024_lifecycle_gateup", gateup_blob), (fx.rows, 2, 1), (32, 4, 1))

  for _ in range(args.warmups):
    producer_once(prod_prg, fx)
    gateup_once(gateup_prg, fx)
  _, q8_x0, producer_corr_before = producer_once(prod_prg, fx, check=True)
  _, consumer_corr_before = gateup_once(gateup_prg, fx, q8_x0, check=True)

  clock_before = clock_sample()
  rng = random.Random(args.seed)
  row_types = ["producer_only", "lifecycle"]
  rows: list[dict[str, Any]] = []
  samples = {"producer_only": [], "lifecycle_producer": [], "lifecycle_consumer": [], "lifecycle_total": []}
  for r in range(args.rounds):
    order = row_types[:]
    rng.shuffle(order)
    for label in order:
      prod_us, _, _ = producer_once(prod_prg, fx)
      if label == "producer_only":
        samples["producer_only"].append(prod_us)
        rows.append({"round": r, "label": label, "producer_us": prod_us})
      else:
        cons_us, _ = gateup_once(gateup_prg, fx)
        total_us = prod_us + cons_us
        samples["lifecycle_producer"].append(prod_us)
        samples["lifecycle_consumer"].append(cons_us)
        samples["lifecycle_total"].append(total_us)
        rows.append({"round": r, "label": label, "producer_us": prod_us, "consumer_us": cons_us, "total_us": total_us})
  clock_after = clock_sample()
  _, q8_x1, producer_corr_after = producer_once(prod_prg, fx, check=True)
  _, consumer_corr_after = gateup_once(gateup_prg, fx, q8_x1, check=True)

  summaries = {k: summarize(v) for k, v in samples.items()}
  producer_delta_us = summaries["lifecycle_producer"]["median_us"] - args.producer_target_us
  gates = {
    "rows_present": all(summaries[k]["n"] == args.rounds for k in summaries),
    "producer_correct": bool(producer_corr_before and producer_corr_before["producer_correct"] and producer_corr_before["q8_dequant_bounded"] and
                             producer_corr_after and producer_corr_after["producer_correct"] and producer_corr_after["q8_dequant_bounded"]),
    "consumer_correct": bool(consumer_corr_before and consumer_corr_before["gate_correct"] and consumer_corr_before["up_correct"] and
                             consumer_corr_after and consumer_corr_after["gate_correct"] and consumer_corr_after["up_correct"]),
    "producer_recovered": abs(producer_delta_us) <= args.producer_material_delta_us,
    "lifecycle_total_lte_target": summaries["lifecycle_total"]["median_us"] <= args.target_us,
  }
  if not gates["producer_correct"] or not gates["consumer_correct"]:
    verdict = "BLOCKED_DECODE_Q8_NT1024_LIFECYCLE_INCORRECT"
  elif gates["lifecycle_total_lte_target"]:
    verdict = "PASS_DECODE_Q8_NT1024_LIFECYCLE_GATE"
  elif gates["producer_recovered"]:
    verdict = "BLOCKED_DECODE_Q8_NT1024_LIFECYCLE_CONSUMER_SESSION_DEBT"
  else:
    verdict = "BLOCKED_DECODE_Q8_NT1024_LIFECYCLE_PRODUCER_NOT_RECOVERED"

  result = {
    "date": "2026-06-20",
    "phase": "DECODE_Q8_NT1024_LIFECYCLE_GATE",
    "schema": "decode_q8_nt1024_lifecycle_gate_v1",
    "verdict": verdict,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": True,
    "commit": git_sha(),
    "producer": {"compiler": "tinygrad_COMGR", "threads": 1024},
    "consumer": {"compiler": "hipcc_lld", "kernel": "q8_mmvq_gateup"},
    "target_lifecycle_us": args.target_us,
    "producer_target_us": args.producer_target_us,
    "producer_delta_us": producer_delta_us,
    "summaries": summaries,
    "correctness": {
      "producer_before": producer_corr_before,
      "consumer_before": consumer_corr_before,
      "producer_after": producer_corr_after,
      "consumer_after": consumer_corr_after,
    },
    "gates": gates,
    "clock": {"before": clock_before, "after": clock_after},
    "rows": rows,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "producer_only_us": summaries["producer_only"]["median_us"],
    "lifecycle_producer_us": summaries["lifecycle_producer"]["median_us"],
    "lifecycle_consumer_us": summaries["lifecycle_consumer"]["median_us"],
    "lifecycle_total_us": summaries["lifecycle_total"]["median_us"],
    "target_lifecycle_us": args.target_us,
    "producer_delta_us": producer_delta_us,
    "gates": gates,
    "out": rel(args.out),
  }, indent=2))
  return 0 if gates["producer_correct"] and gates["consumer_correct"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
