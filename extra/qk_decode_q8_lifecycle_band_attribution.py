#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, statistics, subprocess, sys
from typing import Any

from tinygrad.device import Device
from extra.q8_ffn_artifact_import_route import FixedLaunchRunner
from extra.q8_ffn_fast_artifact_probe import HIP_MMVQ_GATEUP_SOURCE, compile_hipcc_linked
from extra.qk_decode_owned_q8_interleaved_lifecycle_gate import Fixture, clock_sample, gateup_once, producer_once, summarize
from extra.qk_decode_q8_producer_delta_variants import comgr_norm_source


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_q8_lifecycle_band_attribution_result.json"


def rel(p: pathlib.Path) -> str:
  return str(p.relative_to(ROOT)) if p.is_absolute() and p.is_relative_to(ROOT) else str(p)


def git_sha() -> str:
  try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
  except Exception: return "unknown"


def median(xs: list[float]) -> float:
  return float(statistics.median(xs))


def classify_band(us: float, fast_us: float, slow_us: float) -> str:
  if us <= fast_us: return "fast"
  if us >= slow_us: return "slow"
  return "mid"


def child_out(parent_out: pathlib.Path, session: int) -> pathlib.Path:
  return parent_out.parent / f"decode_q8_lifecycle_band_attribution_session_{session:02d}.json"


def protocol_summary(session_rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
  xs = [float(r[key]) for r in session_rows if key in r]
  return summarize(xs)


def run_child(args: argparse.Namespace) -> int:
  dev = Device["AMD"]
  prod_prg = FixedLaunchRunner(dev.runtime("decode_q8_lifecycle_band_producer", dev.compiler.compile(comgr_norm_source(1024))),
                               (1, 1, 1), (1024, 1, 1))
  gateup_blob = compile_hipcc_linked(HIP_MMVQ_GATEUP_SOURCE, args.arch)
  fx = Fixture(args.gguf, args.rows, args.seed)
  gateup_prg = FixedLaunchRunner(dev.runtime("decode_q8_lifecycle_band_gateup", gateup_blob), (fx.rows, 2, 1), (32, 4, 1))

  for _ in range(args.producer_warmups):
    producer_once(prod_prg, fx)
  _, q8_x0, producer_corr_before = producer_once(prod_prg, fx, check=True)
  _, consumer_corr_before = gateup_once(gateup_prg, fx, q8_x0, check=True)

  clock_before = clock_sample()
  rows: list[dict[str, Any]] = []

  prebuilt_consumer_rows = []
  for r in range(args.prebuilt_rounds):
    cons_us, _ = gateup_once(gateup_prg, fx)
    row = {"round": r, "label": "prebuilt_consumer", "consumer_us": cons_us}
    rows.append(row)
    prebuilt_consumer_rows.append(row)

  lifecycle_rows = []
  for r in range(args.lifecycle_rounds):
    prod_us, _, _ = producer_once(prod_prg, fx)
    cons_us, _ = gateup_once(gateup_prg, fx)
    row = {"round": r, "label": "lifecycle", "producer_us": prod_us, "consumer_us": cons_us, "total_us": prod_us + cons_us}
    rows.append(row)
    lifecycle_rows.append(row)

  producer_rows = []
  for r in range(args.producer_rounds):
    prod_us, _, _ = producer_once(prod_prg, fx)
    row = {"round": r, "label": "producer_only", "producer_us": prod_us}
    rows.append(row)
    producer_rows.append(row)

  after_dummy_rows = []
  for r in range(args.after_dummy_rounds):
    prod_us, _, _ = producer_once(prod_prg, fx)
    dummy_us, _ = gateup_once(gateup_prg, fx)
    cons_us, _ = gateup_once(gateup_prg, fx)
    row = {"round": r, "label": "lifecycle_after_dummy", "producer_us": prod_us, "dummy_consumer_us": dummy_us,
           "consumer_us": cons_us, "total_us": prod_us + cons_us}
    rows.append(row)
    after_dummy_rows.append(row)

  clock_after = clock_sample()
  _, q8_x1, producer_corr_after = producer_once(prod_prg, fx, check=True)
  _, consumer_corr_after = gateup_once(gateup_prg, fx, q8_x1, check=True)

  first_n = lifecycle_rows[:args.first_n]
  steady = lifecycle_rows[args.first_n:]
  summaries = {
    "prebuilt_consumer": protocol_summary(prebuilt_consumer_rows, "consumer_us"),
    "lifecycle_first_n_producer": protocol_summary(first_n, "producer_us"),
    "lifecycle_first_n_consumer": protocol_summary(first_n, "consumer_us"),
    "lifecycle_first_n_total": protocol_summary(first_n, "total_us"),
    "lifecycle_steady_producer": protocol_summary(steady, "producer_us"),
    "lifecycle_steady_consumer": protocol_summary(steady, "consumer_us"),
    "lifecycle_steady_total": protocol_summary(steady, "total_us"),
    "producer_only": protocol_summary(producer_rows, "producer_us"),
    "after_dummy_consumer": protocol_summary(after_dummy_rows, "consumer_us"),
    "after_dummy_total": protocol_summary(after_dummy_rows, "total_us"),
  }
  gates = {
    "rows_present": len(prebuilt_consumer_rows) == args.prebuilt_rounds and len(lifecycle_rows) == args.lifecycle_rounds and
                    len(producer_rows) == args.producer_rounds and len(after_dummy_rows) == args.after_dummy_rounds,
    "producer_correct": bool(producer_corr_before and producer_corr_before["producer_correct"] and producer_corr_before["q8_dequant_bounded"] and
                             producer_corr_after and producer_corr_after["producer_correct"] and producer_corr_after["q8_dequant_bounded"]),
    "consumer_correct": bool(consumer_corr_before and consumer_corr_before["gate_correct"] and consumer_corr_before["up_correct"] and
                             consumer_corr_after and consumer_corr_after["gate_correct"] and consumer_corr_after["up_correct"]),
    "steady_lifecycle_lte_target": summaries["lifecycle_steady_total"].get("median_us", float("inf")) <= args.target_us,
  }
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_Q8_LIFECYCLE_BAND_ATTRIBUTION_CHILD",
    "schema": "decode_q8_lifecycle_band_attribution_child_v1",
    "verdict": "PASS_DECODE_Q8_LIFECYCLE_BAND_ATTRIBUTION_CHILD" if all(gates[k] for k in ("rows_present", "producer_correct", "consumer_correct")) else
               "BLOCKED_DECODE_Q8_LIFECYCLE_BAND_ATTRIBUTION_INCORRECT",
    "gate_pass": all(gates[k] for k in ("rows_present", "producer_correct", "consumer_correct")),
    "commit": git_sha(),
    "target_us": args.target_us,
    "first_n": args.first_n,
    "producer": {"compiler": "tinygrad_COMGR", "threads": 1024},
    "consumer": {"compiler": "hipcc_lld", "kernel": "q8_mmvq_gateup"},
    "summaries": summaries,
    "correctness": {"producer_before": producer_corr_before, "consumer_before": consumer_corr_before,
                    "producer_after": producer_corr_after, "consumer_after": consumer_corr_after},
    "clock": {"before": clock_before, "after": clock_after},
    "gates": gates,
    "rows": rows,
  }
  args.child_out.parent.mkdir(parents=True, exist_ok=True)
  args.child_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "prebuilt_consumer_us": summaries["prebuilt_consumer"].get("median_us"),
    "steady_consumer_us": summaries["lifecycle_steady_consumer"].get("median_us"),
    "steady_total_us": summaries["lifecycle_steady_total"].get("median_us"),
    "after_dummy_consumer_us": summaries["after_dummy_consumer"].get("median_us"),
    "out": rel(args.child_out),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


def collect_medians(sessions: list[dict[str, Any]], label: str) -> list[float]:
  out = []
  for s in sessions:
    v = s.get(label)
    if v is not None: out.append(float(v))
  return out


def run_parent(args: argparse.Namespace) -> int:
  children: list[dict[str, Any]] = []
  sessions: list[dict[str, Any]] = []
  for session in range(args.sessions):
    out = child_out(args.out, session)
    cmd = [
      sys.executable, rel(pathlib.Path(__file__).resolve()),
      "--child-out", rel(out),
      "--seed", str(args.seed + session),
      "--rows", str(args.rows),
      "--gguf", str(args.gguf),
      "--arch", args.arch,
      "--target-us", str(args.target_us),
      "--producer-warmups", str(args.producer_warmups),
      "--prebuilt-rounds", str(args.prebuilt_rounds),
      "--lifecycle-rounds", str(args.lifecycle_rounds),
      "--producer-rounds", str(args.producer_rounds),
      "--after-dummy-rounds", str(args.after_dummy_rounds),
      "--first-n", str(args.first_n),
      "--consumer-fast-us", str(args.consumer_fast_us),
      "--consumer-slow-us", str(args.consumer_slow_us),
    ]
    p = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    children.append({"session": session, "cmd": cmd, "returncode": p.returncode, "stdout": p.stdout[-4000:], "artifact": rel(out)})
    if not out.exists():
      sessions.append({"session": session, "artifact": rel(out), "returncode": p.returncode, "missing_artifact": True})
      continue
    d = json.loads(out.read_text())
    summaries = d.get("summaries", {})
    row: dict[str, Any] = {"session": session, "artifact": rel(out), "returncode": p.returncode,
                           "verdict": d.get("verdict"), "gates": d.get("gates", {}), "clock": d.get("clock", {})}
    for name in ("prebuilt_consumer", "lifecycle_first_n_producer", "lifecycle_first_n_consumer", "lifecycle_first_n_total",
                 "lifecycle_steady_producer", "lifecycle_steady_consumer", "lifecycle_steady_total", "producer_only",
                 "after_dummy_consumer", "after_dummy_total"):
      row[f"{name}_median_us"] = (summaries.get(name) or {}).get("median_us")
    for name in ("prebuilt_consumer", "lifecycle_first_n_consumer", "lifecycle_steady_consumer", "after_dummy_consumer"):
      med = row.get(f"{name}_median_us")
      row[f"{name}_band"] = classify_band(float(med), args.consumer_fast_us, args.consumer_slow_us) if med is not None else "missing"
    sessions.append(row)

  summary: dict[str, Any] = {}
  for name in ("prebuilt_consumer", "lifecycle_first_n_producer", "lifecycle_first_n_consumer", "lifecycle_first_n_total",
               "lifecycle_steady_producer", "lifecycle_steady_consumer", "lifecycle_steady_total", "producer_only",
               "after_dummy_consumer", "after_dummy_total"):
    medians = collect_medians(sessions, f"{name}_median_us")
    summary[name] = {
      "median_of_session_medians_us": median(medians) if medians else None,
      "best_session_us": min(medians) if medians else None,
      "worst_session_us": max(medians) if medians else None,
    }
  for name in ("prebuilt_consumer", "lifecycle_first_n_consumer", "lifecycle_steady_consumer", "after_dummy_consumer"):
    medians = collect_medians(sessions, f"{name}_median_us")
    summary[name]["fast_sessions"] = sum(m <= args.consumer_fast_us for m in medians)
    summary[name]["slow_sessions"] = sum(m >= args.consumer_slow_us for m in medians)
    summary[name]["band"] = classify_band(float(summary[name]["median_of_session_medians_us"]), args.consumer_fast_us, args.consumer_slow_us) if medians else "missing"

  all_artifacts_present = len([s for s in sessions if not s.get("missing_artifact")]) == args.sessions
  all_correct = all((s.get("gates") or {}).get("producer_correct") and (s.get("gates") or {}).get("consumer_correct") for s in sessions)
  steady_total = summary["lifecycle_steady_total"]["median_of_session_medians_us"] or float("inf")
  prebuilt_consumer = summary["prebuilt_consumer"]["median_of_session_medians_us"] or float("inf")
  lifecycle_consumer = summary["lifecycle_steady_consumer"]["median_of_session_medians_us"] or float("inf")
  after_dummy_consumer = summary["after_dummy_consumer"]["median_of_session_medians_us"] or float("inf")

  if not all_artifacts_present or not all_correct:
    verdict, classification = "BLOCKED_DECODE_Q8_LIFECYCLE_BAND_INCORRECT", "INCOMPLETE_OR_INCORRECT"
  elif steady_total <= args.target_us:
    verdict, classification = "PASS_DECODE_Q8_LIFECYCLE_STEADY_RECONCILED", "STEADY_TARGET_PASS"
  elif prebuilt_consumer <= args.consumer_fast_us and lifecycle_consumer >= args.consumer_slow_us:
    verdict, classification = "BLOCKED_DECODE_Q8_LIFECYCLE_PRODUCER_CONSUMER_ADJACENCY", "PRODUCER_CONSUMER_ADJACENCY"
  elif prebuilt_consumer <= args.consumer_fast_us and lifecycle_consumer <= args.consumer_fast_us and after_dummy_consumer <= args.consumer_fast_us:
    verdict, classification = "BLOCKED_DECODE_Q8_LIFECYCLE_POLICY_OR_TARGET", "STEADY_COMPONENTS_FAST_TARGET_POLICY"
  else:
    verdict, classification = "BLOCKED_DECODE_Q8_LIFECYCLE_SESSION_STATE", "SESSION_STATE"

  result = {
    "date": "2026-06-20",
    "phase": "DECODE_Q8_LIFECYCLE_BAND_ATTRIBUTION",
    "schema": "decode_q8_lifecycle_band_attribution_v1",
    "verdict": verdict,
    "classification": classification,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": True,
    "commit": git_sha(),
    "target_us": args.target_us,
    "consumer_fast_us": args.consumer_fast_us,
    "consumer_slow_us": args.consumer_slow_us,
    "summary": summary,
    "gates": {
      "all_artifacts_present": all_artifacts_present,
      "all_correct": all_correct,
      "steady_lifecycle_lte_target": steady_total <= args.target_us,
      "prebuilt_consumer_fast": prebuilt_consumer <= args.consumer_fast_us,
      "lifecycle_steady_consumer_fast": lifecycle_consumer <= args.consumer_fast_us,
      "after_dummy_consumer_fast": after_dummy_consumer <= args.consumer_fast_us,
    },
    "sessions": sessions,
    "children": children,
    "decision": {
      "if_adjacency": "prebuilt consumer is fast but consumer-after-producer is slow; isolate producer->consumer dependency, cache, or launch adjacency before consumer body rewrite",
      "if_policy_or_target": "steady components are fast but lifecycle still misses target; promotion now depends on policy/target or launch accounting, not kernel body",
      "if_session_state": "consumer band follows whole-session state; clock/session policy must be made explicit",
      "if_pass": "steady lifecycle clears target; promotion policy can reopen",
    },
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "classification": classification,
    "summary": summary,
    "gates": result["gates"],
    "out": rel(args.out),
  }, indent=2))
  return 0 if all_artifacts_present and all_correct else 1


def main() -> int:
  ap = argparse.ArgumentParser(description="Decode q8 NT1024 lifecycle first-N/steady band attribution")
  ap.add_argument("--sessions", type=int, default=5)
  ap.add_argument("--producer-warmups", type=int, default=4)
  ap.add_argument("--prebuilt-rounds", type=int, default=24)
  ap.add_argument("--lifecycle-rounds", type=int, default=32)
  ap.add_argument("--producer-rounds", type=int, default=24)
  ap.add_argument("--after-dummy-rounds", type=int, default=16)
  ap.add_argument("--first-n", type=int, default=4)
  ap.add_argument("--seed", type=int, default=191)
  ap.add_argument("--target-us", type=float, default=115.24)
  ap.add_argument("--consumer-fast-us", type=float, default=91.0)
  ap.add_argument("--consumer-slow-us", type=float, default=100.0)
  ap.add_argument("--rows", type=int, default=12288)
  ap.add_argument("--gguf", type=pathlib.Path, default=pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  ap.add_argument("--arch", default="gfx1100")
  ap.add_argument("--out", type=pathlib.Path, default=OUT)
  ap.add_argument("--child-out", type=pathlib.Path)
  args = ap.parse_args()
  return run_child(args) if args.child_out is not None else run_parent(args)


if __name__ == "__main__":
  raise SystemExit(main())
