#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, pathlib, random, statistics, subprocess, sys
from typing import Any

from tinygrad.device import Device
from extra.q8_ffn_artifact_import_route import FixedLaunchRunner
from extra.q8_ffn_codegen_transfer_audit import inspect_blob
from extra.q8_ffn_fast_artifact_probe import HIP_MMVQ_GATEUP_SOURCE, compile_hipcc_linked
from extra.qk_decode_owned_q8_interleaved_lifecycle_gate import Fixture, clock_sample, gateup_once, producer_once, summarize
from extra.qk_decode_q8_producer_delta_variants import comgr_norm_source


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_q8_consumer_band_attribution_result.json"


def rel(p: pathlib.Path) -> str:
  return str(p.relative_to(ROOT)) if p.is_absolute() and p.is_relative_to(ROOT) else str(p)


def git_sha() -> str:
  try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
  except Exception: return "unknown"


def sha256_bytes(x: bytes) -> str:
  return hashlib.sha256(x).hexdigest()


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


def classify_band(us: float, fast_band_us: float, slow_band_us: float) -> str:
  if us <= fast_band_us: return "fast"
  if us >= slow_band_us: return "slow"
  return "mid"


def child_out(parent_out: pathlib.Path, session: int) -> pathlib.Path:
  return parent_out.parent / f"decode_q8_consumer_band_attribution_session_{session:02d}.json"


def static_issue_summary(static: dict[str, Any]) -> dict[str, Any]:
  dis = static.get("disasm", {})
  groups = dis.get("grouped_counts", {})
  runtime = static.get("runtime", {})
  top = dis.get("top_mnemonics", [])
  return {
    "instruction_count": dis.get("instruction_count"),
    "grouped_counts": groups,
    "top_mnemonics": top[:20],
    "runtime": runtime,
    "issue_hints": [
      "consumer has dot4 body present" if groups.get("dot4") == 16 else "dot4 body count unexpected",
      "consumer uses shuffle reduction" if (groups.get("shuffle") or 0) > 0 else "no shuffle reduction counted",
      "consumer has one tiny LDS allocation" if runtime.get("group_segment_size") == 16 else "unexpected LDS allocation",
      "consumer has no private spill" if runtime.get("private_segment_size") == 0 else "private spill present",
      "PMC counter decode is still needed for cycle-level cause; static issue analysis only classifies the artifact shape",
    ],
  }


def compile_static(arch: str) -> dict[str, Any]:
  blob = compile_hipcc_linked(HIP_MMVQ_GATEUP_SOURCE, arch)
  static = inspect_blob("decode_q8_consumer_band_gateup", blob, "decode_q8_consumer_band_static_gateup")
  return {
    "compiler": "hipcc_lld",
    "kernel": "q8_mmvq_gateup",
    "arch": arch,
    "source": "extra.q8_ffn_fast_artifact_probe.HIP_MMVQ_GATEUP_SOURCE",
    "source_sha256": hashlib.sha256(HIP_MMVQ_GATEUP_SOURCE.encode()).hexdigest(),
    "hsaco_sha256": sha256_bytes(blob),
    "hsaco_bytes": len(blob),
    "launch": {"global_size": [12288, 2, 1], "local_size": [32, 4, 1]},
    **static,
  }


def time_consumer(prg: FixedLaunchRunner, fx: Fixture, count: int, label: str, start_round: int = 0) -> tuple[list[float], list[dict[str, Any]]]:
  samples: list[float] = []
  rows: list[dict[str, Any]] = []
  for i in range(count):
    us, _ = gateup_once(prg, fx)
    samples.append(us)
    rows.append({"round": start_round + i, "label": label, "consumer_us": us})
  return samples, rows


def run_child(args: argparse.Namespace) -> int:
  dev = Device["AMD"]
  prod_prg = FixedLaunchRunner(dev.runtime("decode_q8_consumer_band_setup_producer", dev.compiler.compile(comgr_norm_source(1024))),
                               (1, 1, 1), (1024, 1, 1))
  gateup_blob = compile_hipcc_linked(HIP_MMVQ_GATEUP_SOURCE, args.arch)
  fx = Fixture(args.gguf, args.rows, args.seed)
  gateup_prg = FixedLaunchRunner(dev.runtime("decode_q8_consumer_band_gateup", gateup_blob), (fx.rows, 2, 1), (32, 4, 1))

  for _ in range(args.producer_warmups):
    producer_once(prod_prg, fx)
  _, q8_x0, producer_corr = producer_once(prod_prg, fx, check=True)
  _, consumer_corr_before = gateup_once(gateup_prg, fx, q8_x0, check=True)

  clock_before = clock_sample()
  rows: list[dict[str, Any]] = []
  samples: dict[str, list[float]] = {"first_n": [], "repeat_same": [], "dummy": [], "after_dummy": [], "mixed": []}

  first, first_rows = time_consumer(gateup_prg, fx, args.first_n, "first_n")
  samples["first_n"].extend(first)
  rows.extend(first_rows)

  repeat, repeat_rows = time_consumer(gateup_prg, fx, args.repeat_n, "repeat_same", len(rows))
  samples["repeat_same"].extend(repeat)
  rows.extend(repeat_rows)

  for r in range(args.dummy_rounds):
    dummy_us, _ = gateup_once(gateup_prg, fx)
    after_us, _ = gateup_once(gateup_prg, fx)
    samples["dummy"].append(dummy_us)
    samples["after_dummy"].append(after_us)
    rows.append({"round": r, "label": "dummy", "consumer_us": dummy_us})
    rows.append({"round": r, "label": "after_dummy", "consumer_us": after_us})

  rng = random.Random(args.seed)
  labels = ["repeat_same", "after_dummy"]
  for r in range(args.mixed_rounds):
    order = labels[:]
    rng.shuffle(order)
    for label in order:
      if label == "after_dummy":
        dummy_us, _ = gateup_once(gateup_prg, fx)
        samples["dummy"].append(dummy_us)
        rows.append({"round": r, "label": "mixed_dummy", "consumer_us": dummy_us})
      us, _ = gateup_once(gateup_prg, fx)
      samples["mixed"].append(us)
      rows.append({"round": r, "label": f"mixed_{label}", "consumer_us": us})

  clock_after = clock_sample()
  _, consumer_corr_after = gateup_once(gateup_prg, fx, q8_x0, check=True)
  summaries = {k: summarize(v) for k, v in samples.items()}
  gates = {
    "producer_correct": bool(producer_corr and producer_corr["producer_correct"] and producer_corr["q8_dequant_bounded"]),
    "consumer_correct": bool(consumer_corr_before and consumer_corr_before["gate_correct"] and consumer_corr_before["up_correct"] and
                             consumer_corr_after and consumer_corr_after["gate_correct"] and consumer_corr_after["up_correct"]),
    "rows_present": len(samples["first_n"]) == args.first_n and len(samples["repeat_same"]) == args.repeat_n and
                    len(samples["after_dummy"]) >= args.dummy_rounds and len(samples["mixed"]) == args.mixed_rounds * 2,
  }
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_Q8_CONSUMER_BAND_ATTRIBUTION_CHILD",
    "schema": "decode_q8_consumer_band_attribution_child_v1",
    "verdict": "PASS_DECODE_Q8_CONSUMER_BAND_ATTRIBUTION_CHILD" if all(gates.values()) else "BLOCKED_DECODE_Q8_CONSUMER_BAND_ATTRIBUTION_INCORRECT",
    "gate_pass": all(gates.values()),
    "commit": git_sha(),
    "producer_setup": {"compiler": "tinygrad_COMGR", "threads": 1024},
    "consumer": {"compiler": "hipcc_lld", "kernel": "q8_mmvq_gateup", "hsaco_sha256": sha256_bytes(gateup_blob)},
    "summaries": summaries,
    "correctness": {"producer": producer_corr, "consumer_before": consumer_corr_before, "consumer_after": consumer_corr_after},
    "clock": {"before": clock_before, "after": clock_after},
    "gates": gates,
    "rows": rows,
  }
  args.child_out.parent.mkdir(parents=True, exist_ok=True)
  args.child_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "repeat_same_us": summaries["repeat_same"]["median_us"],
    "after_dummy_us": summaries["after_dummy"]["median_us"],
    "mixed_us": summaries["mixed"]["median_us"],
    "out": rel(args.child_out),
  }, indent=2))
  return 0 if all(gates.values()) else 1


def run_parent(args: argparse.Namespace) -> int:
  args.out.parent.mkdir(parents=True, exist_ok=True)
  static = compile_static(args.arch)
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
      "--producer-warmups", str(args.producer_warmups),
      "--first-n", str(args.first_n),
      "--repeat-n", str(args.repeat_n),
      "--dummy-rounds", str(args.dummy_rounds),
      "--mixed-rounds", str(args.mixed_rounds),
      "--fast-band-us", str(args.fast_band_us),
      "--slow-band-us", str(args.slow_band_us),
    ]
    p = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    children.append({"session": session, "cmd": cmd, "returncode": p.returncode, "stdout": p.stdout[-4000:], "artifact": rel(out)})
    if not out.exists():
      sessions.append({"session": session, "artifact": rel(out), "returncode": p.returncode, "missing_artifact": True})
      continue
    d = json.loads(out.read_text())
    summaries = d.get("summaries", {})
    session_row: dict[str, Any] = {
      "session": session,
      "artifact": rel(out),
      "returncode": p.returncode,
      "verdict": d.get("verdict"),
      "gates": d.get("gates", {}),
      "clock": d.get("clock", {}),
    }
    for label in ("first_n", "repeat_same", "dummy", "after_dummy", "mixed"):
      med = (summaries.get(label) or {}).get("median_us")
      session_row[f"{label}_median_us"] = med
      session_row[f"{label}_band"] = classify_band(float(med), args.fast_band_us, args.slow_band_us) if med is not None else "missing"
    sessions.append(session_row)

  protocol_labels = ("first_n", "repeat_same", "after_dummy", "mixed")
  protocol_summary: dict[str, Any] = {}
  for label in protocol_labels:
    medians = [float(s[f"{label}_median_us"]) for s in sessions if s.get(f"{label}_median_us") is not None]
    reconciled = median(medians) if medians else float("inf")
    protocol_summary[label] = {
      "median_of_session_medians_us": reconciled,
      "best_session_us": min(medians) if medians else None,
      "worst_session_us": max(medians) if medians else None,
      "fast_sessions": sum(m <= args.fast_band_us for m in medians),
      "slow_sessions": sum(m >= args.slow_band_us for m in medians),
      "band": classify_band(reconciled, args.fast_band_us, args.slow_band_us),
    }

  all_artifacts_present = len([s for s in sessions if not s.get("missing_artifact")]) == args.sessions
  all_correct = all((s.get("gates") or {}).get("producer_correct") and (s.get("gates") or {}).get("consumer_correct") for s in sessions)
  stable_slow = all(protocol_summary[label]["band"] == "slow" for label in protocol_labels)
  any_bimodal = any(protocol_summary[label]["fast_sessions"] and protocol_summary[label]["slow_sessions"] for label in protocol_labels)
  if not all_artifacts_present or not all_correct:
    verdict, classification = "BLOCKED_DECODE_Q8_CONSUMER_BAND_INCORRECT", "INCOMPLETE_OR_INCORRECT"
  elif stable_slow:
    verdict, classification = "BLOCKED_DECODE_Q8_CONSUMER_BAND_STABLE_SLOW_STATIC_ISSUE_NEXT", "STABLE_SLOW"
  elif any_bimodal:
    verdict, classification = "BLOCKED_DECODE_Q8_CONSUMER_BAND_BIMODAL_SESSION_STATE", "BIMODAL_SESSION_STATE"
  elif all(protocol_summary[label]["band"] == "fast" for label in protocol_labels):
    verdict, classification = "PASS_DECODE_Q8_CONSUMER_BAND_FAST", "FAST"
  else:
    verdict, classification = "BLOCKED_DECODE_Q8_CONSUMER_BAND_MIXED_PROTOCOL", "MIXED_PROTOCOL"

  result = {
    "date": "2026-06-20",
    "phase": "DECODE_Q8_CONSUMER_BAND_ATTRIBUTION",
    "schema": "decode_q8_consumer_band_attribution_v1",
    "verdict": verdict,
    "classification": classification,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": True,
    "commit": git_sha(),
    "fast_band_us": args.fast_band_us,
    "slow_band_us": args.slow_band_us,
    "static_artifact": static,
    "static_issue_analysis": static_issue_summary(static),
    "protocol_summary": protocol_summary,
    "gates": {
      "all_artifacts_present": all_artifacts_present,
      "all_correct": all_correct,
      "stable_slow_all_protocols": stable_slow,
      "bimodal_any_protocol": any_bimodal,
      "pmc_counter_decode_available": False,
    },
    "sessions": sessions,
    "children": children,
    "decision": {
      "if_stable_slow": "consumer slow band survives first-N, repeated, after-dummy, and mixed ordering; static artifact has no spill/LDS surprise, so next useful work is PMC/SQTT counter decode or a consumer issue experiment",
      "if_bimodal": "band tracks session/clock/launch state; make timing policy explicit before rewriting consumer",
      "if_fast": "consumer-only no longer blocks; return to lifecycle composition",
      "if_mixed_protocol": "ordering affects band; isolate the protocol transition before codegen work",
      "pmc_boundary": "existing PMC summaries do not contain decoded counter values, so this pass records static issue evidence but not counter-grade rates",
    },
  }
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "classification": classification,
    "protocol_summary": protocol_summary,
    "gates": result["gates"],
    "out": rel(args.out),
  }, indent=2))
  return 0 if all_artifacts_present and all_correct else 1


def main() -> int:
  ap = argparse.ArgumentParser(description="Decode q8 fused gate/up consumer band attribution")
  ap.add_argument("--sessions", type=int, default=5)
  ap.add_argument("--producer-warmups", type=int, default=4)
  ap.add_argument("--first-n", type=int, default=16)
  ap.add_argument("--repeat-n", type=int, default=32)
  ap.add_argument("--dummy-rounds", type=int, default=16)
  ap.add_argument("--mixed-rounds", type=int, default=16)
  ap.add_argument("--seed", type=int, default=171)
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
