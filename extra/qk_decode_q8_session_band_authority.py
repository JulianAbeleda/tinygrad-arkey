#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, re, statistics, subprocess, sys
from typing import Any

from tinygrad.device import Device
from extra.q8_ffn_artifact_import_route import FixedLaunchRunner
from extra.q8_ffn_fast_artifact_probe import HIP_MMVQ_GATEUP_SOURCE, compile_hipcc_linked
from extra.qk_decode_owned_q8_interleaved_lifecycle_gate import Fixture, clock_sample, gateup_once, producer_once, summarize
from extra.qk_decode_q8_producer_delta_variants import comgr_norm_source


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_q8_session_band_authority_result.json"
SCLK_RE = re.compile(r"sclk clock level:\s+\d+\s+\((\d+)Mhz\)")


def rel(p: pathlib.Path) -> str:
  return str(p.relative_to(ROOT)) if p.is_absolute() and p.is_relative_to(ROOT) else str(p)


def git_sha() -> str:
  try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
  except Exception: return "unknown"


def median(xs: list[float]) -> float:
  return float(statistics.median(xs))


def sclk_mhz(sample: dict[str, Any]) -> int | None:
  text = sample.get("stdout") or ""
  m = SCLK_RE.search(text)
  return int(m.group(1)) if m else None


def tagged_clock(tag: str) -> dict[str, Any]:
  s = clock_sample()
  return {"tag": tag, "sclk_mhz": sclk_mhz(s), **s}


def classify_us(us: float, fast_us: float, slow_us: float) -> str:
  if us <= fast_us: return "fast"
  if us >= slow_us: return "slow"
  return "mid"


def child_out(parent_out: pathlib.Path, protocol: str, session: int) -> pathlib.Path:
  return parent_out.parent / f"decode_q8_session_band_authority_{protocol}_session_{session:02d}.json"


def warm_protocol(protocol: str, prod_prg: FixedLaunchRunner, gateup_prg: FixedLaunchRunner, fx: Fixture, n: int) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  if protocol == "cold":
    return rows
  if protocol == "producer_warm":
    for i in range(n):
      us, _, _ = producer_once(prod_prg, fx)
      rows.append({"warmup": i, "label": "producer_warm", "producer_us": us})
  elif protocol == "consumer_warm":
    for i in range(n):
      us, _ = gateup_once(gateup_prg, fx)
      rows.append({"warmup": i, "label": "consumer_warm", "consumer_us": us})
  elif protocol == "lifecycle_warm":
    for i in range(n):
      prod_us, _, _ = producer_once(prod_prg, fx)
      cons_us, _ = gateup_once(gateup_prg, fx)
      rows.append({"warmup": i, "label": "lifecycle_warm", "producer_us": prod_us, "consumer_us": cons_us,
                   "total_us": prod_us + cons_us})
  elif protocol == "producer_then_consumer_warm":
    for i in range(n):
      prod_us, _, _ = producer_once(prod_prg, fx)
      rows.append({"warmup": i, "label": "producer_warm", "producer_us": prod_us})
    for i in range(n):
      cons_us, _ = gateup_once(gateup_prg, fx)
      rows.append({"warmup": i, "label": "consumer_warm", "consumer_us": cons_us})
  else:
    raise ValueError(f"unknown protocol {protocol!r}")
  return rows


def measure_block(prod_prg: FixedLaunchRunner, gateup_prg: FixedLaunchRunner, fx: Fixture, rounds: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  samples = {"producer": [], "prebuilt_consumer": [], "lifecycle_producer": [], "lifecycle_consumer": [], "lifecycle_total": []}
  for r in range(rounds):
    cons_us, _ = gateup_once(gateup_prg, fx)
    samples["prebuilt_consumer"].append(cons_us)
    rows.append({"round": r, "label": "prebuilt_consumer", "consumer_us": cons_us})

    prod_us, _, _ = producer_once(prod_prg, fx)
    samples["producer"].append(prod_us)
    rows.append({"round": r, "label": "producer_only", "producer_us": prod_us})

    prod2_us, _, _ = producer_once(prod_prg, fx)
    cons2_us, _ = gateup_once(gateup_prg, fx)
    samples["lifecycle_producer"].append(prod2_us)
    samples["lifecycle_consumer"].append(cons2_us)
    samples["lifecycle_total"].append(prod2_us + cons2_us)
    rows.append({"round": r, "label": "lifecycle", "producer_us": prod2_us, "consumer_us": cons2_us,
                 "total_us": prod2_us + cons2_us})
  return rows, {k: summarize(v) for k, v in samples.items()}


def run_child(args: argparse.Namespace) -> int:
  dev = Device["AMD"]
  prod_prg = FixedLaunchRunner(dev.runtime(f"decode_q8_session_band_{args.protocol}_producer",
                                           dev.compiler.compile(comgr_norm_source(1024))), (1, 1, 1), (1024, 1, 1))
  gateup_blob = compile_hipcc_linked(HIP_MMVQ_GATEUP_SOURCE, args.arch)
  fx = Fixture(args.gguf, args.rows, args.seed)
  gateup_prg = FixedLaunchRunner(dev.runtime(f"decode_q8_session_band_{args.protocol}_gateup", gateup_blob),
                                 (fx.rows, 2, 1), (32, 4, 1))

  clocks = [tagged_clock("after_compile")]
  setup_us, q8_x0, producer_corr_before = producer_once(prod_prg, fx, check=True)
  consumer_setup_us, consumer_corr_before = gateup_once(gateup_prg, fx, q8_x0, check=True)
  clocks.append(tagged_clock("after_setup_correctness"))

  warm_rows = warm_protocol(args.protocol, prod_prg, gateup_prg, fx, args.warmup_rounds)
  clocks.append(tagged_clock("after_protocol_warmup"))

  measure_rows, summaries = measure_block(prod_prg, gateup_prg, fx, args.rounds)
  clocks.append(tagged_clock("after_measure"))

  _, q8_x1, producer_corr_after = producer_once(prod_prg, fx, check=True)
  _, consumer_corr_after = gateup_once(gateup_prg, fx, q8_x1, check=True)
  clocks.append(tagged_clock("after_final_correctness"))

  gates = {
    "producer_correct": bool(producer_corr_before and producer_corr_before["producer_correct"] and producer_corr_before["q8_dequant_bounded"] and
                             producer_corr_after and producer_corr_after["producer_correct"] and producer_corr_after["q8_dequant_bounded"]),
    "consumer_correct": bool(consumer_corr_before and consumer_corr_before["gate_correct"] and consumer_corr_before["up_correct"] and
                             consumer_corr_after and consumer_corr_after["gate_correct"] and consumer_corr_after["up_correct"]),
    "lifecycle_lte_target": summaries["lifecycle_total"]["median_us"] <= args.target_us,
  }
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_Q8_SESSION_BAND_AUTHORITY_CHILD",
    "schema": "decode_q8_session_band_authority_child_v1",
    "verdict": "PASS_DECODE_Q8_SESSION_BAND_AUTHORITY_CHILD" if gates["producer_correct"] and gates["consumer_correct"] else
               "BLOCKED_DECODE_Q8_SESSION_BAND_AUTHORITY_INCORRECT",
    "gate_pass": gates["producer_correct"] and gates["consumer_correct"],
    "commit": git_sha(),
    "protocol": args.protocol,
    "target_us": args.target_us,
    "setup": {"producer_us": setup_us, "consumer_us": consumer_setup_us},
    "summaries": summaries,
    "correctness": {"producer_before": producer_corr_before, "consumer_before": consumer_corr_before,
                    "producer_after": producer_corr_after, "consumer_after": consumer_corr_after},
    "clocks": clocks,
    "gates": gates,
    "warmup_rows": warm_rows,
    "rows": measure_rows,
  }
  args.child_out.parent.mkdir(parents=True, exist_ok=True)
  args.child_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "protocol": args.protocol,
    "verdict": result["verdict"],
    "prebuilt_consumer_us": summaries["prebuilt_consumer"]["median_us"],
    "lifecycle_consumer_us": summaries["lifecycle_consumer"]["median_us"],
    "lifecycle_total_us": summaries["lifecycle_total"]["median_us"],
    "sclk_mhz": [c.get("sclk_mhz") for c in clocks],
    "out": rel(args.child_out),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


def run_parent(args: argparse.Namespace) -> int:
  protocols = args.protocols.split(",")
  children: list[dict[str, Any]] = []
  sessions: list[dict[str, Any]] = []
  for protocol in protocols:
    for session in range(args.sessions):
      out = child_out(args.out, protocol, session)
      cmd = [
        sys.executable, rel(pathlib.Path(__file__).resolve()),
        "--child-out", rel(out),
        "--protocol", protocol,
        "--seed", str(args.seed + session),
        "--rows", str(args.rows),
        "--gguf", str(args.gguf),
        "--arch", args.arch,
        "--target-us", str(args.target_us),
        "--rounds", str(args.rounds),
        "--warmup-rounds", str(args.warmup_rounds),
        "--fast-total-us", str(args.fast_total_us),
        "--slow-total-us", str(args.slow_total_us),
      ]
      p = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
      children.append({"protocol": protocol, "session": session, "cmd": cmd, "returncode": p.returncode,
                       "stdout": p.stdout[-4000:], "artifact": rel(out)})
      if not out.exists():
        sessions.append({"protocol": protocol, "session": session, "artifact": rel(out), "returncode": p.returncode,
                         "missing_artifact": True})
        continue
      d = json.loads(out.read_text())
      summaries = d.get("summaries", {})
      total = (summaries.get("lifecycle_total") or {}).get("median_us")
      row: dict[str, Any] = {"protocol": protocol, "session": session, "artifact": rel(out), "returncode": p.returncode,
                             "verdict": d.get("verdict"), "gates": d.get("gates", {}),
                             "sclk_mhz": [c.get("sclk_mhz") for c in d.get("clocks", [])]}
      for name in ("producer", "prebuilt_consumer", "lifecycle_producer", "lifecycle_consumer", "lifecycle_total"):
        row[f"{name}_median_us"] = (summaries.get(name) or {}).get("median_us")
      row["total_band"] = classify_us(float(total), args.fast_total_us, args.slow_total_us) if total is not None else "missing"
      sessions.append(row)

  summary: dict[str, Any] = {}
  for protocol in protocols:
    ps = [s for s in sessions if s.get("protocol") == protocol and not s.get("missing_artifact")]
    totals = [float(s["lifecycle_total_median_us"]) for s in ps if s.get("lifecycle_total_median_us") is not None]
    consumers = [float(s["lifecycle_consumer_median_us"]) for s in ps if s.get("lifecycle_consumer_median_us") is not None]
    prebuilt = [float(s["prebuilt_consumer_median_us"]) for s in ps if s.get("prebuilt_consumer_median_us") is not None]
    producers = [float(s["lifecycle_producer_median_us"]) for s in ps if s.get("lifecycle_producer_median_us") is not None]
    sclk_vals = [mhz for s in ps for mhz in (s.get("sclk_mhz") or []) if mhz is not None]
    summary[protocol] = {
      "sessions": len(ps),
      "total_median_of_session_medians_us": median(totals) if totals else None,
      "total_best_session_us": min(totals) if totals else None,
      "total_worst_session_us": max(totals) if totals else None,
      "total_fast_sessions": sum(t <= args.fast_total_us for t in totals),
      "total_slow_sessions": sum(t >= args.slow_total_us for t in totals),
      "consumer_median_of_session_medians_us": median(consumers) if consumers else None,
      "prebuilt_consumer_median_of_session_medians_us": median(prebuilt) if prebuilt else None,
      "producer_median_of_session_medians_us": median(producers) if producers else None,
      "sclk_mhz_min": min(sclk_vals) if sclk_vals else None,
      "sclk_mhz_median": median([float(x) for x in sclk_vals]) if sclk_vals else None,
      "sclk_mhz_max": max(sclk_vals) if sclk_vals else None,
    }

  all_artifacts_present = len([s for s in sessions if not s.get("missing_artifact")]) == len(protocols) * args.sessions
  all_correct = all((s.get("gates") or {}).get("producer_correct") and (s.get("gates") or {}).get("consumer_correct")
                    for s in sessions if not s.get("missing_artifact"))
  best_protocol = min((p for p in protocols if summary[p]["total_median_of_session_medians_us"] is not None),
                      key=lambda p: float(summary[p]["total_median_of_session_medians_us"]), default=None)
  best_total = float(summary[best_protocol]["total_median_of_session_medians_us"]) if best_protocol else float("inf")
  pass_protocols = [p for p in protocols if (summary[p]["total_median_of_session_medians_us"] or float("inf")) <= args.target_us]
  if not all_artifacts_present or not all_correct:
    verdict, classification = "BLOCKED_DECODE_Q8_SESSION_BAND_AUTHORITY_INCORRECT", "INCOMPLETE_OR_INCORRECT"
  elif pass_protocols:
    verdict, classification = "PASS_DECODE_Q8_SESSION_BAND_WARM_POLICY_FOUND", "WARM_POLICY_FOUND"
  elif best_total <= args.slow_total_us:
    verdict, classification = "BLOCKED_DECODE_Q8_SESSION_BAND_NEAR_TARGET_POLICY", "NEAR_TARGET_POLICY"
  else:
    verdict, classification = "BLOCKED_DECODE_Q8_SESSION_BAND_NO_AUTHORITY_POLICY", "NO_AUTHORITY_POLICY"

  result = {
    "date": "2026-06-20",
    "phase": "DECODE_Q8_SESSION_BAND_AUTHORITY",
    "schema": "decode_q8_session_band_authority_v1",
    "verdict": verdict,
    "classification": classification,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": True,
    "commit": git_sha(),
    "target_us": args.target_us,
    "fast_total_us": args.fast_total_us,
    "slow_total_us": args.slow_total_us,
    "protocols": protocols,
    "sessions_per_protocol": args.sessions,
    "rounds": args.rounds,
    "warmup_rounds": args.warmup_rounds,
    "best_protocol": best_protocol,
    "summary": summary,
    "gates": {"all_artifacts_present": all_artifacts_present, "all_correct": all_correct,
              "pass_protocols": pass_protocols, "best_total_lte_target": best_total <= args.target_us},
    "sessions": sessions,
    "children": children,
    "decision": {
      "if_warm_policy_found": "Use this protocol as the q8 lifecycle timing authority candidate; rerun at 5+ sessions before promotion",
      "if_near_target_policy": "Warm-state policy narrows the gap but does not clear target; decide target/policy or continue session-state work",
      "if_no_authority_policy": "None of the bounded warm-state controls force the fast band; promotion remains blocked",
      "clock_boundary": "rocm-smi boundary samples are recorded, but they are not counter-grade timing telemetry",
    },
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "classification": classification,
    "best_protocol": best_protocol,
    "summary": summary,
    "gates": result["gates"],
    "out": rel(args.out),
  }, indent=2))
  return 0 if all_artifacts_present and all_correct else 1


def main() -> int:
  ap = argparse.ArgumentParser(description="Decode q8 session-band warm-state authority matrix")
  ap.add_argument("--sessions", type=int, default=3)
  ap.add_argument("--protocols", default="cold,producer_warm,consumer_warm,lifecycle_warm,producer_then_consumer_warm")
  ap.add_argument("--rounds", type=int, default=16)
  ap.add_argument("--warmup-rounds", type=int, default=16)
  ap.add_argument("--seed", type=int, default=211)
  ap.add_argument("--target-us", type=float, default=115.24)
  ap.add_argument("--fast-total-us", type=float, default=115.24)
  ap.add_argument("--slow-total-us", type=float, default=121.0)
  ap.add_argument("--rows", type=int, default=12288)
  ap.add_argument("--gguf", type=pathlib.Path, default=pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  ap.add_argument("--arch", default="gfx1100")
  ap.add_argument("--out", type=pathlib.Path, default=OUT)
  ap.add_argument("--child-out", type=pathlib.Path)
  ap.add_argument("--protocol", default="cold")
  args = ap.parse_args()
  return run_child(args) if args.child_out is not None else run_parent(args)


if __name__ == "__main__":
  raise SystemExit(main())
