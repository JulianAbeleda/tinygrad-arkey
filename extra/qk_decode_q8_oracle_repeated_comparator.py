#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, random, statistics, subprocess, sys
from typing import Any

from tinygrad.device import Device
from extra.q8_ffn_artifact_import_route import FixedLaunchRunner
from extra.q8_ffn_fast_artifact_probe import HIP_MMVQ_GATEUP_SOURCE, compile_hipcc_linked, hip_norm_source
from extra.q8_ffn_hcq_artifact import NORM_SOURCE
from extra.qk_decode_owned_q8_interleaved_lifecycle_gate import Fixture, clock_sample, gateup_once, producer_once, rel, summarize


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_q8_oracle_repeated_comparator_result.json"


def git_sha() -> str:
  try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
  except Exception: return "unknown"


def median(xs: list[float]) -> float:
  return float(statistics.median(xs))


def child_out(parent_out: pathlib.Path, session: int) -> pathlib.Path:
  return parent_out.parent / f"decode_q8_oracle_repeated_comparator_session_{session:02d}.json"


def run_child(args: argparse.Namespace) -> int:
  dev = Device["AMD"]
  owned_prod = FixedLaunchRunner(dev.runtime("q8_cmp_owned_producer", dev.compiler.compile(NORM_SOURCE)), (1, 1, 1), (256, 1, 1))
  oracle_prod_blob = compile_hipcc_linked(hip_norm_source(args.oracle_producer_threads), args.arch)
  oracle_prod = FixedLaunchRunner(dev.runtime("q8_cmp_oracle_producer", oracle_prod_blob), (1, 1, 1), (args.oracle_producer_threads, 1, 1))
  gateup_blob = compile_hipcc_linked(HIP_MMVQ_GATEUP_SOURCE, args.arch)
  fx = Fixture(args.gguf, args.rows, args.seed)
  gateup = FixedLaunchRunner(dev.runtime("q8_cmp_oracle_gateup", gateup_blob), (fx.rows, 2, 1), (32, 4, 1))

  for _ in range(args.warmups):
    producer_once(owned_prod, fx)
    gateup_once(gateup, fx)
    producer_once(oracle_prod, fx)
    gateup_once(gateup, fx)

  correctness: dict[str, Any] = {}
  for name, prod in (("owned", owned_prod), ("oracle", oracle_prod)):
    _, q8_x, pcorr = producer_once(prod, fx, check=True)
    _, ccorr = gateup_once(gateup, fx, q8_x, check=True)
    correctness[name] = {"producer": pcorr, "consumer": ccorr}

  clock_before = clock_sample()
  rng = random.Random(args.seed)
  route_prgs = {"owned_lifecycle": owned_prod, "oracle_lifecycle": oracle_prod}
  rows: list[dict[str, Any]] = []
  samples = {k: {"producer": [], "consumer": [], "total": []} for k in route_prgs}
  for r in range(args.rounds):
    labels = list(route_prgs)
    rng.shuffle(labels)
    for label in labels:
      prod_us, _, _ = producer_once(route_prgs[label], fx)
      cons_us, _ = gateup_once(gateup, fx)
      total_us = prod_us + cons_us
      samples[label]["producer"].append(prod_us)
      samples[label]["consumer"].append(cons_us)
      samples[label]["total"].append(total_us)
      rows.append({"round": r, "label": label, "producer_us": prod_us, "consumer_us": cons_us, "total_us": total_us})
  clock_after = clock_sample()

  summaries = {route: {part: summarize(vals) for part, vals in route_samples.items()} for route, route_samples in samples.items()}
  gates = {
    "rows_present": all(summaries[route]["total"]["n"] == args.rounds for route in route_prgs),
    "owned_correct": bool(correctness["owned"]["producer"]["producer_correct"] and correctness["owned"]["producer"]["q8_dequant_bounded"] and
                          correctness["owned"]["consumer"]["gate_correct"] and correctness["owned"]["consumer"]["up_correct"]),
    "oracle_correct": bool(correctness["oracle"]["producer"]["producer_correct"] and correctness["oracle"]["producer"]["q8_dequant_bounded"] and
                           correctness["oracle"]["consumer"]["gate_correct"] and correctness["oracle"]["consumer"]["up_correct"]),
  }
  verdict = "PASS_DECODE_Q8_ORACLE_COMPARATOR_CHILD" if all(gates.values()) else "BLOCKED_DECODE_Q8_ORACLE_COMPARATOR_INCORRECT"
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_Q8_ORACLE_REPEATED_COMPARATOR_CHILD",
    "schema": "decode_q8_oracle_repeated_comparator_child_v1",
    "verdict": verdict,
    "gate_pass": verdict.startswith("PASS_"),
    "commit": git_sha(),
    "target_us": args.target_us,
    "routes": {
      "owned_lifecycle": {"producer": "tinygrad_COMGR_NORM_SOURCE", "producer_threads": 256, "consumer": "hipcc_lld_gateup"},
      "oracle_lifecycle": {"producer": "hipcc_lld_q8_rmsnorm_side", "producer_threads": args.oracle_producer_threads, "consumer": "hipcc_lld_gateup"},
    },
    "summaries": summaries,
    "correctness": correctness,
    "rows": rows,
    "clock": {"before": clock_before, "after": clock_after},
    "gates": gates,
  }
  args.child_out.parent.mkdir(parents=True, exist_ok=True)
  args.child_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "owned_total_us": summaries["owned_lifecycle"]["total"]["median_us"],
    "oracle_total_us": summaries["oracle_lifecycle"]["total"]["median_us"],
    "owned_producer_us": summaries["owned_lifecycle"]["producer"]["median_us"],
    "oracle_producer_us": summaries["oracle_lifecycle"]["producer"]["median_us"],
    "owned_consumer_us": summaries["owned_lifecycle"]["consumer"]["median_us"],
    "oracle_consumer_us": summaries["oracle_lifecycle"]["consumer"]["median_us"],
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
      "--seed", str(args.seed + session),
      "--target-us", str(args.target_us),
      "--rows", str(args.rows),
      "--gguf", str(args.gguf),
      "--arch", args.arch,
      "--oracle-producer-threads", str(args.oracle_producer_threads),
    ]
    p = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    child = {"session": session, "cmd": cmd, "returncode": p.returncode, "stdout": p.stdout[-4000:], "artifact": rel(out)}
    children.append(child)
    if not out.exists():
      sessions.append({"session": session, "artifact": rel(out), "returncode": p.returncode, "missing_artifact": True})
      continue
    d = json.loads(out.read_text())
    summaries = d.get("summaries", {})
    owned, oracle = summaries.get("owned_lifecycle", {}), summaries.get("oracle_lifecycle", {})
    sessions.append({
      "session": session,
      "artifact": rel(out),
      "returncode": p.returncode,
      "verdict": d.get("verdict"),
      "gates": d.get("gates", {}),
      "owned_producer_us": (owned.get("producer") or {}).get("median_us"),
      "owned_consumer_us": (owned.get("consumer") or {}).get("median_us"),
      "owned_total_us": (owned.get("total") or {}).get("median_us"),
      "owned_min_total_us": (owned.get("total") or {}).get("min_ms", 0.0) * 1000.0,
      "oracle_producer_us": (oracle.get("producer") or {}).get("median_us"),
      "oracle_consumer_us": (oracle.get("consumer") or {}).get("median_us"),
      "oracle_total_us": (oracle.get("total") or {}).get("median_us"),
      "oracle_min_total_us": (oracle.get("total") or {}).get("min_ms", 0.0) * 1000.0,
      "clock": d.get("clock", {}),
    })

  owned_totals = [float(s["owned_total_us"]) for s in sessions if s.get("owned_total_us") is not None]
  oracle_totals = [float(s["oracle_total_us"]) for s in sessions if s.get("oracle_total_us") is not None]
  owned_producers = [float(s["owned_producer_us"]) for s in sessions if s.get("owned_producer_us") is not None]
  oracle_producers = [float(s["oracle_producer_us"]) for s in sessions if s.get("oracle_producer_us") is not None]
  owned_consumers = [float(s["owned_consumer_us"]) for s in sessions if s.get("owned_consumer_us") is not None]
  oracle_consumers = [float(s["oracle_consumer_us"]) for s in sessions if s.get("oracle_consumer_us") is not None]

  all_artifacts = len(owned_totals) == args.sessions and len(oracle_totals) == args.sessions
  all_correct = all((s.get("gates") or {}).get("owned_correct") and (s.get("gates") or {}).get("oracle_correct") for s in sessions)
  owned_reconciled = median(owned_totals) if owned_totals else float("inf")
  oracle_reconciled = median(oracle_totals) if oracle_totals else float("inf")
  delta_owned_minus_oracle = owned_reconciled - oracle_reconciled

  if not all_artifacts or not all_correct:
    verdict = "BLOCKED_DECODE_Q8_ORACLE_COMPARATOR_INCORRECT"
    classification = "INCOMPLETE_OR_INCORRECT"
  else:
    verdict = "PASS_DECODE_Q8_ORACLE_REPEATED_COMPARATOR_ATTRIBUTED"
    if oracle_reconciled <= args.target_us or delta_owned_minus_oracle > args.material_delta_us:
      classification = "OWNED_PRODUCER_DEBT"
    elif oracle_reconciled > args.target_us and abs(delta_owned_minus_oracle) <= args.material_delta_us:
      classification = "SHARED_CONSUMER_OR_SESSION_DEBT"
    else:
      classification = "MIXED_SESSION_VARIANCE"

  result = {
    "date": "2026-06-20",
    "phase": "DECODE_Q8_ORACLE_REPEATED_COMPARATOR",
    "schema": "decode_q8_oracle_repeated_comparator_v1",
    "verdict": verdict,
    "classification": classification,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": True,
    "commit": git_sha(),
    "target_us": args.target_us,
    "material_delta_us": args.material_delta_us,
    "sessions_requested": args.sessions,
    "rounds_per_session": args.rounds,
    "summary": {
      "owned_total_median_of_session_medians_us": owned_reconciled,
      "oracle_total_median_of_session_medians_us": oracle_reconciled,
      "owned_minus_oracle_total_us": delta_owned_minus_oracle,
      "owned_producer_median_of_session_medians_us": median(owned_producers) if owned_producers else None,
      "oracle_producer_median_of_session_medians_us": median(oracle_producers) if oracle_producers else None,
      "owned_consumer_median_of_session_medians_us": median(owned_consumers) if owned_consumers else None,
      "oracle_consumer_median_of_session_medians_us": median(oracle_consumers) if oracle_consumers else None,
    },
    "gates": {
      "all_artifacts_present": all_artifacts,
      "all_correct": all_correct,
      "oracle_reconciled_lte_target": oracle_reconciled <= args.target_us,
      "owned_materially_slower_than_oracle": delta_owned_minus_oracle > args.material_delta_us,
    },
    "sessions": sessions,
    "children": children,
    "decision": {
      "owned_producer_debt": "oracle route clears target or materially beats owned route; focus producer/lifecycle ownership",
      "shared_consumer_or_session_debt": "same hipcc/LLD consumer slow band appears with both producers; compare clock/launch/consumer issue",
      "mixed_session_variance": "repeat under stricter clock control before changing code",
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
  return 0 if all_artifacts and all_correct else 1


def main() -> int:
  ap = argparse.ArgumentParser(description="Repeated decode q8 owned-vs-hipcc/LLD oracle lifecycle comparator")
  ap.add_argument("--sessions", type=int, default=5)
  ap.add_argument("--rounds", type=int, default=24)
  ap.add_argument("--warmups", type=int, default=8)
  ap.add_argument("--seed", type=int, default=71)
  ap.add_argument("--target-us", type=float, default=115.24)
  ap.add_argument("--material-delta-us", type=float, default=2.0)
  ap.add_argument("--rows", type=int, default=12288)
  ap.add_argument("--gguf", type=pathlib.Path, default=pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  ap.add_argument("--arch", default="gfx1100")
  ap.add_argument("--oracle-producer-threads", type=int, default=1024)
  ap.add_argument("--out", type=pathlib.Path, default=OUT)
  ap.add_argument("--child-out", type=pathlib.Path)
  args = ap.parse_args()
  return run_child(args) if args.child_out is not None else run_parent(args)


if __name__ == "__main__":
  raise SystemExit(main())
