#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, random, statistics, subprocess, sys
from typing import Any

from tinygrad.device import Device
from extra.q8_ffn_artifact_import_route import FixedLaunchRunner
from extra.q8_ffn_fast_artifact_probe import HIP_MMVQ_GATEUP_SOURCE, compile_hipcc_linked
from extra.qk_decode_owned_q8_interleaved_lifecycle_gate import Fixture, clock_sample, gateup_once, producer_once, rel, summarize
from extra.qk_decode_q8_producer_delta_variants import comgr_norm_source


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_q8_consumer_only_reconciliation_result.json"


def git_sha() -> str:
  try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
  except Exception: return "unknown"


def median(xs: list[float]) -> float:
  return float(statistics.median(xs))


def simple_summary(xs: list[float]) -> dict[str, Any]:
  if not xs: return {"n": 0}
  sx = sorted(xs)
  return {
    "n": len(xs),
    "min_us": min(xs),
    "p10_us": sx[max(0, int(len(xs) * 0.10) - 1)],
    "median_us": median(xs),
    "mean_us": float(statistics.fmean(xs)),
    "max_us": max(xs),
  }


def child_out(parent_out: pathlib.Path, session: int) -> pathlib.Path:
  return parent_out.parent / f"decode_q8_consumer_only_reconciliation_session_{session:02d}.json"


def run_child(args: argparse.Namespace) -> int:
  dev = Device["AMD"]
  prod_prg = FixedLaunchRunner(dev.runtime("decode_q8_consumer_only_setup_producer", dev.compiler.compile(comgr_norm_source(1024))),
                               (1, 1, 1), (1024, 1, 1))
  gateup_blob = compile_hipcc_linked(HIP_MMVQ_GATEUP_SOURCE, args.arch)
  fx = Fixture(args.gguf, args.rows, args.seed)
  gateup_prg = FixedLaunchRunner(dev.runtime("decode_q8_consumer_only_gateup", gateup_blob), (fx.rows, 2, 1), (32, 4, 1))

  for _ in range(args.producer_warmups):
    producer_once(prod_prg, fx)
  _, q8_x0, producer_corr = producer_once(prod_prg, fx, check=True)
  for _ in range(args.warmups):
    gateup_once(gateup_prg, fx)
  _, consumer_corr_before = gateup_once(gateup_prg, fx, q8_x0, check=True)

  clock_before = clock_sample()
  rng = random.Random(args.seed)
  rows: list[dict[str, Any]] = []
  samples = {"consumer_only": [], "consumer_after_dummy": []}
  labels = list(samples)
  for r in range(args.rounds):
    order = labels[:]
    rng.shuffle(order)
    for label in order:
      if label == "consumer_after_dummy":
        # Extra consumer dispatch gives an order/context row without changing buffers.
        gateup_once(gateup_prg, fx)
      cons_us, _ = gateup_once(gateup_prg, fx)
      samples[label].append(cons_us)
      rows.append({"round": r, "label": label, "consumer_us": cons_us})
  clock_after = clock_sample()
  _, consumer_corr_after = gateup_once(gateup_prg, fx, q8_x0, check=True)

  summaries = {k: summarize(v) for k, v in samples.items()}
  gates = {
    "rows_present": all(summaries[k]["n"] == args.rounds for k in summaries),
    "producer_correct": bool(producer_corr and producer_corr["producer_correct"] and producer_corr["q8_dequant_bounded"]),
    "consumer_correct": bool(consumer_corr_before and consumer_corr_before["gate_correct"] and consumer_corr_before["up_correct"] and
                             consumer_corr_after and consumer_corr_after["gate_correct"] and consumer_corr_after["up_correct"]),
  }
  verdict = "PASS_DECODE_Q8_CONSUMER_ONLY_CHILD" if all(gates.values()) else "BLOCKED_DECODE_Q8_CONSUMER_ONLY_INCORRECT"
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_Q8_CONSUMER_ONLY_RECONCILIATION_CHILD",
    "schema": "decode_q8_consumer_only_reconciliation_child_v1",
    "verdict": verdict,
    "gate_pass": verdict.startswith("PASS_"),
    "commit": git_sha(),
    "producer_setup": {"compiler": "tinygrad_COMGR", "threads": 1024},
    "consumer": {"compiler": "hipcc_lld", "kernel": "q8_mmvq_gateup"},
    "summaries": summaries,
    "correctness": {
      "producer_setup": producer_corr,
      "consumer_before": consumer_corr_before,
      "consumer_after": consumer_corr_after,
    },
    "gates": gates,
    "clock": {"before": clock_before, "after": clock_after},
    "rows": rows,
  }
  args.child_out.parent.mkdir(parents=True, exist_ok=True)
  args.child_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "consumer_only_us": summaries["consumer_only"]["median_us"],
    "consumer_after_dummy_us": summaries["consumer_after_dummy"]["median_us"],
    "gates": gates,
    "out": rel(args.child_out),
  }, indent=2))
  return 0 if all(gates.values()) else 1


def run_parent(args: argparse.Namespace) -> int:
  children: list[dict[str, Any]] = []
  sessions: list[dict[str, Any]] = []
  for session in range(args.sessions):
    out = child_out(args.out, session)
    cmd = [
      sys.executable, rel(pathlib.Path(__file__).resolve()),
      "--child-out", rel(out),
      "--rounds", str(args.rounds),
      "--warmups", str(args.warmups),
      "--producer-warmups", str(args.producer_warmups),
      "--seed", str(args.seed + session),
      "--rows", str(args.rows),
      "--gguf", str(args.gguf),
      "--arch", args.arch,
    ]
    p = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    child = {"session": session, "cmd": cmd, "returncode": p.returncode, "stdout": p.stdout[-4000:], "artifact": rel(out)}
    children.append(child)
    if not out.exists():
      sessions.append({"session": session, "artifact": rel(out), "returncode": p.returncode, "missing_artifact": True})
      continue
    d = json.loads(out.read_text())
    summaries = d.get("summaries", {})
    sessions.append({
      "session": session,
      "artifact": rel(out),
      "returncode": p.returncode,
      "verdict": d.get("verdict"),
      "gates": d.get("gates", {}),
      "consumer_only": simple_summary([float(r["consumer_us"]) for r in d.get("rows", []) if r.get("label") == "consumer_only"]),
      "consumer_after_dummy": simple_summary([float(r["consumer_us"]) for r in d.get("rows", []) if r.get("label") == "consumer_after_dummy"]),
      "consumer_only_median_us": (summaries.get("consumer_only") or {}).get("median_us"),
      "consumer_after_dummy_median_us": (summaries.get("consumer_after_dummy") or {}).get("median_us"),
      "clock": d.get("clock", {}),
    })

  medians = [float(s["consumer_only_median_us"]) for s in sessions if s.get("consumer_only_median_us") is not None]
  dummy_medians = [float(s["consumer_after_dummy_median_us"]) for s in sessions if s.get("consumer_after_dummy_median_us") is not None]
  all_artifacts_present = len(medians) == args.sessions
  all_correct = all((s.get("gates") or {}).get("producer_correct") and (s.get("gates") or {}).get("consumer_correct") for s in sessions)
  reconciled_us = median(medians) if medians else float("inf")
  dummy_reconciled_us = median(dummy_medians) if dummy_medians else float("inf")
  fast_sessions = sum(m <= args.fast_band_us for m in medians)
  slow_sessions = sum(m >= args.slow_band_us for m in medians)

  if not all_artifacts_present or not all_correct:
    verdict, classification = "BLOCKED_DECODE_Q8_CONSUMER_ONLY_INCORRECT", "INCOMPLETE_OR_INCORRECT"
  elif reconciled_us <= args.fast_band_us:
    verdict, classification = "PASS_DECODE_Q8_CONSUMER_ONLY_RECONCILED_FAST", "FAST_BAND"
  elif fast_sessions and slow_sessions:
    verdict, classification = "BLOCKED_DECODE_Q8_CONSUMER_ONLY_BIMODAL", "BIMODAL"
  elif reconciled_us >= args.slow_band_us:
    verdict, classification = "BLOCKED_DECODE_Q8_CONSUMER_ONLY_SLOW_BAND", "SLOW_BAND"
  else:
    verdict, classification = "BLOCKED_DECODE_Q8_CONSUMER_ONLY_MID_BAND", "MID_BAND"

  result = {
    "date": "2026-06-20",
    "phase": "DECODE_Q8_CONSUMER_ONLY_RECONCILIATION",
    "schema": "decode_q8_consumer_only_reconciliation_v1",
    "verdict": verdict,
    "classification": classification,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": True,
    "commit": git_sha(),
    "sessions_requested": args.sessions,
    "rounds_per_session": args.rounds,
    "fast_band_us": args.fast_band_us,
    "slow_band_us": args.slow_band_us,
    "summary": {
      "consumer_only_median_of_session_medians_us": reconciled_us,
      "consumer_after_dummy_median_of_session_medians_us": dummy_reconciled_us,
      "fast_sessions": fast_sessions,
      "slow_sessions": slow_sessions,
      "best_observed_session_us": min(medians) if medians else None,
      "worst_observed_session_us": max(medians) if medians else None,
    },
    "gates": {
      "all_artifacts_present": all_artifacts_present,
      "all_correct": all_correct,
      "consumer_reconciled_lte_fast_band": reconciled_us <= args.fast_band_us,
      "bimodal": bool(fast_sessions and slow_sessions),
    },
    "sessions": sessions,
    "children": children,
    "decision": {
      "FAST_BAND": "consumer-only clears the fast band; lifecycle composition or producer interleaving caused prior slow sessions",
      "MID_BAND": "consumer-only misses the fast band but does not reproduce the lifecycle slow band; compare lifecycle composition",
      "BIMODAL": "consumer alone is bimodal; next attribution is clock/session/launch state",
      "SLOW_BAND": "consumer alone is stable slow; reopen consumer issue/resource attribution",
    },
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "classification": classification,
    **result["summary"],
    "gates": result["gates"],
    "out": rel(args.out),
  }, indent=2))
  return 0 if all_artifacts_present and all_correct else 1


def main() -> int:
  ap = argparse.ArgumentParser(description="Decode q8 consumer-only repeated reconciliation")
  ap.add_argument("--sessions", type=int, default=5)
  ap.add_argument("--rounds", type=int, default=32)
  ap.add_argument("--warmups", type=int, default=8)
  ap.add_argument("--producer-warmups", type=int, default=4)
  ap.add_argument("--seed", type=int, default=151)
  ap.add_argument("--fast-band-us", type=float, default=91.0)
  ap.add_argument("--slow-band-us", type=float, default=100.0)
  ap.add_argument("--rows", type=int, default=12288)
  ap.add_argument("--gguf", type=pathlib.Path, default=pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  ap.add_argument("--arch", default="gfx1100")
  ap.add_argument("--out", type=pathlib.Path, default=OUT)
  ap.add_argument("--child-out", type=pathlib.Path)
  args = ap.parse_args()
  return run_child(args) if args.child_out is not None else run_parent(args)


if __name__ == "__main__":
  raise SystemExit(main())
