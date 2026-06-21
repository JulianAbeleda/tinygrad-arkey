#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, re, statistics, subprocess, sys, threading, time
from typing import Any

from tinygrad.device import Device
from extra.q8_ffn_artifact_import_route import FixedLaunchRunner
from extra.q8_ffn_fast_artifact_probe import HIP_MMVQ_GATEUP_SOURCE, compile_hipcc_linked
from extra.qk_decode_owned_q8_interleaved_lifecycle_gate import Fixture, gateup_once, producer_once, summarize
from extra.qk_decode_q8_producer_delta_variants import comgr_norm_source


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_q8_clock_authority_result.json"
DEV = pathlib.Path("/sys/class/drm/card0/device")
HWMON = next((DEV / "hwmon").glob("hwmon*"), None) if (DEV / "hwmon").exists() else None
SCLK_RE = re.compile(r"(\d+)Mhz")


def rel(p: pathlib.Path) -> str:
  return str(p.relative_to(ROOT)) if p.is_absolute() and p.is_relative_to(ROOT) else str(p)


def git_sha() -> str:
  try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
  except Exception: return "unknown"


def read_text(p: pathlib.Path, default: str = "") -> str:
  try: return p.read_text().strip()
  except OSError: return default


def active_mhz(name: str) -> int:
  for line in read_text(DEV / name).splitlines():
    if "*" in line:
      m = SCLK_RE.search(line)
      return int(m.group(1)) if m else 0
  return 0


def sample() -> dict[str, Any]:
  row: dict[str, Any] = {
    "t": time.perf_counter(),
    "sclk": active_mhz("pp_dpm_sclk"),
    "mclk": active_mhz("pp_dpm_mclk"),
    "fclk": active_mhz("pp_dpm_fclk"),
    "socclk": active_mhz("pp_dpm_socclk"),
    "gpu_busy": int(read_text(DEV / "gpu_busy_percent", "0") or 0),
    "mem_busy": int(read_text(DEV / "mem_busy_percent", "0") or 0),
    "perf_level": read_text(DEV / "power_dpm_force_performance_level"),
  }
  if HWMON is not None:
    row["power_w"] = round(int(read_text(HWMON / "power1_average", "0") or 0) / 1e6, 1)
    row["temp_c"] = round(int(read_text(HWMON / "temp1_input", "0") or 0) / 1e3, 1)
  return row


class Sampler(threading.Thread):
  def __init__(self, interval_s: float):
    super().__init__(daemon=True)
    self.interval_s = interval_s
    self.rows: list[dict[str, Any]] = []
    self._stop_flag = False

  def run(self) -> None:
    while not self._stop_flag:
      self.rows.append(sample())
      time.sleep(self.interval_s)

  def finish(self) -> list[dict[str, Any]]:
    self._stop_flag = True
    self.join(timeout=2)
    return self.rows


def stats(vals: list[float]) -> dict[str, Any]:
  return {
    "n": len(vals),
    "min": min(vals) if vals else None,
    "median": statistics.median(vals) if vals else None,
    "max": max(vals) if vals else None,
  }


def telemetry_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
  active = [r for r in rows if r.get("gpu_busy", 0) > 0 or r.get("sclk", 0) > 800 or r.get("mem_busy", 0) > 0]
  def block(rs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
      "n": len(rs),
      "sclk_mhz": stats([float(r.get("sclk", 0)) for r in rs]),
      "mclk_mhz": stats([float(r.get("mclk", 0)) for r in rs]),
      "gpu_busy_pct": stats([float(r.get("gpu_busy", 0)) for r in rs]),
      "mem_busy_pct": stats([float(r.get("mem_busy", 0)) for r in rs]),
      "power_w": stats([float(r.get("power_w", 0)) for r in rs if "power_w" in r]),
      "temp_c": stats([float(r.get("temp_c", 0)) for r in rs if "temp_c" in r]),
      "perf_level_last": rs[-1].get("perf_level", "") if rs else "",
    }
  return {"all": block(rows), "active": block(active), "rows": rows}


def sudo(cmd: str) -> dict[str, Any]:
  p = subprocess.run(["sudo", "-n", "bash", "-c", cmd], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout[-1000:], "ok": p.returncode == 0}


def set_lane(lane: str, det_mhz: int | None = None) -> dict[str, Any]:
  if lane == "auto":
    return sudo(f"echo auto > {DEV}/power_dpm_force_performance_level")
  if lane == "high":
    return sudo(f"echo high > {DEV}/power_dpm_force_performance_level")
  if lane == "profile_peak":
    return sudo(f"echo profile_peak > {DEV}/power_dpm_force_performance_level")
  if lane == "manual_peak":
    return sudo(f"echo manual > {DEV}/power_dpm_force_performance_level && echo 2 > {DEV}/pp_dpm_sclk && echo 3 > {DEV}/pp_dpm_mclk")
  if lane == "determinism":
    mhz = det_mhz or 2304
    p = subprocess.run(["sudo", "-n", "rocm-smi", "--setperfdeterminism", str(mhz)],
                       text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {"cmd": f"rocm-smi --setperfdeterminism {mhz}", "returncode": p.returncode, "stdout": p.stdout[-1000:], "ok": p.returncode == 0}
  raise ValueError(lane)


def restore_lane() -> list[dict[str, Any]]:
  out = []
  p = subprocess.run(["sudo", "-n", "rocm-smi", "--resetperfdeterminism"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  out.append({"cmd": "rocm-smi --resetperfdeterminism", "returncode": p.returncode, "stdout": p.stdout[-1000:], "ok": p.returncode == 0})
  out.append(sudo(f"echo auto > {DEV}/power_dpm_force_performance_level"))
  return out


def inventory() -> dict[str, Any]:
  return {
    "paths": {
      "power_dpm_force_performance_level": str(DEV / "power_dpm_force_performance_level"),
      "pp_dpm_sclk": str(DEV / "pp_dpm_sclk"),
      "pp_dpm_mclk": str(DEV / "pp_dpm_mclk"),
      "gpu_busy_percent": str(DEV / "gpu_busy_percent"),
      "mem_busy_percent": str(DEV / "mem_busy_percent"),
      "hwmon": str(HWMON) if HWMON else None,
    },
    "values": {
      "perf_level": read_text(DEV / "power_dpm_force_performance_level"),
      "pp_dpm_sclk": read_text(DEV / "pp_dpm_sclk"),
      "pp_dpm_mclk": read_text(DEV / "pp_dpm_mclk"),
      "idle_sample": sample(),
    },
    "sudo_n_true": subprocess.run(["sudo", "-n", "true"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0,
  }


def measure(prod_prg: FixedLaunchRunner, gateup_prg: FixedLaunchRunner, fx: Fixture, rounds: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
    total_us = prod2_us + cons2_us
    samples["lifecycle_producer"].append(prod2_us)
    samples["lifecycle_consumer"].append(cons2_us)
    samples["lifecycle_total"].append(total_us)
    rows.append({"round": r, "label": "lifecycle", "producer_us": prod2_us, "consumer_us": cons2_us, "total_us": total_us})
  return rows, {k: summarize(v) for k, v in samples.items()}


def run_child(args: argparse.Namespace) -> int:
  set_res = set_lane(args.lane, args.det_mhz)
  time.sleep(args.settle_s)
  try:
    dev = Device["AMD"]
    prod_prg = FixedLaunchRunner(dev.runtime(f"decode_q8_clock_{args.lane}_producer", dev.compiler.compile(comgr_norm_source(1024))),
                                 (1, 1, 1), (1024, 1, 1))
    gateup_blob = compile_hipcc_linked(HIP_MMVQ_GATEUP_SOURCE, args.arch)
    fx = Fixture(args.gguf, args.rows, args.seed)
    gateup_prg = FixedLaunchRunner(dev.runtime(f"decode_q8_clock_{args.lane}_gateup", gateup_blob), (fx.rows, 2, 1), (32, 4, 1))

    for _ in range(args.warmups):
      producer_once(prod_prg, fx)
      gateup_once(gateup_prg, fx)
    _, q8_x0, producer_corr_before = producer_once(prod_prg, fx, check=True)
    _, consumer_corr_before = gateup_once(gateup_prg, fx, q8_x0, check=True)

    sampler = Sampler(args.sample_interval_s)
    sampler.start()
    measure_rows, summaries = measure(prod_prg, gateup_prg, fx, args.rounds)
    telem_rows = sampler.finish()

    _, q8_x1, producer_corr_after = producer_once(prod_prg, fx, check=True)
    _, consumer_corr_after = gateup_once(gateup_prg, fx, q8_x1, check=True)
    gates = {
      "lane_set_ok": bool(set_res.get("ok")),
      "producer_correct": bool(producer_corr_before and producer_corr_before["producer_correct"] and producer_corr_before["q8_dequant_bounded"] and
                               producer_corr_after and producer_corr_after["producer_correct"] and producer_corr_after["q8_dequant_bounded"]),
      "consumer_correct": bool(consumer_corr_before and consumer_corr_before["gate_correct"] and consumer_corr_before["up_correct"] and
                               consumer_corr_after and consumer_corr_after["gate_correct"] and consumer_corr_after["up_correct"]),
      "lifecycle_lte_target": summaries["lifecycle_total"]["median_us"] <= args.target_us,
    }
    result = {
      "date": "2026-06-20",
      "phase": "DECODE_Q8_CLOCK_AUTHORITY_CHILD",
      "schema": "decode_q8_clock_authority_child_v1",
      "verdict": "PASS_DECODE_Q8_CLOCK_AUTHORITY_CHILD" if gates["producer_correct"] and gates["consumer_correct"] else
                 "BLOCKED_DECODE_Q8_CLOCK_AUTHORITY_CHILD_INCORRECT",
      "gate_pass": gates["producer_correct"] and gates["consumer_correct"],
      "commit": git_sha(),
      "lane": args.lane,
      "set_lane": set_res,
      "target_us": args.target_us,
      "summaries": summaries,
      "telemetry": telemetry_summary(telem_rows),
      "correctness": {"producer_before": producer_corr_before, "consumer_before": consumer_corr_before,
                      "producer_after": producer_corr_after, "consumer_after": consumer_corr_after},
      "gates": gates,
      "rows": measure_rows,
    }
  finally:
    restore = restore_lane()
  result["restore"] = restore
  args.child_out.parent.mkdir(parents=True, exist_ok=True)
  args.child_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "lane": args.lane,
    "verdict": result["verdict"],
    "lifecycle_total_us": result["summaries"]["lifecycle_total"]["median_us"],
    "consumer_us": result["summaries"]["lifecycle_consumer"]["median_us"],
    "producer_us": result["summaries"]["lifecycle_producer"]["median_us"],
    "telemetry_active": result["telemetry"]["active"],
    "out": rel(args.child_out),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


def child_out(parent_out: pathlib.Path, lane: str, session: int) -> pathlib.Path:
  return parent_out.parent / f"decode_q8_clock_authority_{lane}_session_{session:02d}.json"


def collect(vals: list[float]) -> dict[str, Any]:
  return {"n": len(vals), "min": min(vals) if vals else None, "median": statistics.median(vals) if vals else None, "max": max(vals) if vals else None}


def run_parent(args: argparse.Namespace) -> int:
  lanes = args.lanes.split(",")
  inv = inventory()
  children: list[dict[str, Any]] = []
  sessions: list[dict[str, Any]] = []
  for lane in lanes:
    for session in range(args.sessions):
      out = child_out(args.out, lane, session)
      cmd = [
        sys.executable, rel(pathlib.Path(__file__).resolve()), "--child-out", rel(out), "--lane", lane,
        "--seed", str(args.seed + session), "--rounds", str(args.rounds), "--warmups", str(args.warmups),
        "--settle-s", str(args.settle_s), "--sample-interval-s", str(args.sample_interval_s),
        "--target-us", str(args.target_us), "--rows", str(args.rows), "--gguf", str(args.gguf), "--arch", args.arch,
      ]
      if args.det_mhz is not None:
        cmd += ["--det-mhz", str(args.det_mhz)]
      p = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
      children.append({"lane": lane, "session": session, "cmd": cmd, "returncode": p.returncode, "stdout": p.stdout[-4000:], "artifact": rel(out)})
      if not out.exists():
        sessions.append({"lane": lane, "session": session, "artifact": rel(out), "returncode": p.returncode, "missing_artifact": True})
        continue
      d = json.loads(out.read_text())
      active = d.get("telemetry", {}).get("active", {})
      sessions.append({
        "lane": lane,
        "session": session,
        "artifact": rel(out),
        "returncode": p.returncode,
        "verdict": d.get("verdict"),
        "gates": d.get("gates", {}),
        "lifecycle_total_us": (d.get("summaries", {}).get("lifecycle_total") or {}).get("median_us"),
        "lifecycle_consumer_us": (d.get("summaries", {}).get("lifecycle_consumer") or {}).get("median_us"),
        "lifecycle_producer_us": (d.get("summaries", {}).get("lifecycle_producer") or {}).get("median_us"),
        "active_sclk_median": ((active.get("sclk_mhz") or {}).get("median")),
        "active_sclk_min": ((active.get("sclk_mhz") or {}).get("min")),
        "active_sclk_max": ((active.get("sclk_mhz") or {}).get("max")),
        "active_power_max": ((active.get("power_w") or {}).get("max")),
        "active_temp_max": ((active.get("temp_c") or {}).get("max")),
        "active_samples": active.get("n"),
      })

  summary: dict[str, Any] = {}
  for lane in lanes:
    ls = [s for s in sessions if s.get("lane") == lane and not s.get("missing_artifact")]
    totals = [float(s["lifecycle_total_us"]) for s in ls if s.get("lifecycle_total_us") is not None]
    consumers = [float(s["lifecycle_consumer_us"]) for s in ls if s.get("lifecycle_consumer_us") is not None]
    producers = [float(s["lifecycle_producer_us"]) for s in ls if s.get("lifecycle_producer_us") is not None]
    sclk = [float(s["active_sclk_median"]) for s in ls if s.get("active_sclk_median") is not None]
    summary[lane] = {
      "sessions": len(ls),
      "lane_set_ok_sessions": sum(bool((s.get("gates") or {}).get("lane_set_ok")) for s in ls),
      "correct_sessions": sum(bool((s.get("gates") or {}).get("producer_correct") and (s.get("gates") or {}).get("consumer_correct")) for s in ls),
      "total_us": collect(totals),
      "consumer_us": collect(consumers),
      "producer_us": collect(producers),
      "target_pass_sessions": sum(t <= args.target_us for t in totals),
      "active_sclk_median_mhz": collect(sclk),
    }
  pass_lanes = [lane for lane, s in summary.items() if (s["total_us"]["median"] or float("inf")) <= args.target_us and
                s["correct_sessions"] == s["sessions"] and s["lane_set_ok_sessions"] == s["sessions"]]
  all_present = len([s for s in sessions if not s.get("missing_artifact")]) == len(lanes) * args.sessions
  all_correct = all((s.get("gates") or {}).get("producer_correct") and (s.get("gates") or {}).get("consumer_correct")
                    for s in sessions if not s.get("missing_artifact"))
  if not all_present or not all_correct:
    verdict = "BLOCKED_DECODE_Q8_CLOCK_AUTHORITY_INCORRECT"
  elif pass_lanes:
    verdict = "PASS_DECODE_Q8_CLOCK_AUTHORITY_CONTROLLED_FAST"
  else:
    verdict = "BLOCKED_DECODE_Q8_CLOCK_AUTHORITY_NO_CONTROLLED_FAST"
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_Q8_CLOCK_AUTHORITY",
    "schema": "decode_q8_clock_authority_v1",
    "verdict": verdict,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": True,
    "commit": git_sha(),
    "inventory": inv,
    "target_us": args.target_us,
    "lanes": lanes,
    "sessions_per_lane": args.sessions,
    "rounds": args.rounds,
    "pass_lanes": pass_lanes,
    "summary": summary,
    "gates": {"all_artifacts_present": all_present, "all_correct": all_correct, "pass_lanes": pass_lanes},
    "sessions": sessions,
    "children": children,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"verdict": verdict, "pass_lanes": pass_lanes, "summary": summary, "out": rel(args.out)}, indent=2))
  return 0 if all_present and all_correct else 1


def main() -> int:
  ap = argparse.ArgumentParser(description="Decode q8 clock/DPM authority audit")
  ap.add_argument("--lanes", default="auto,high,profile_peak,manual_peak")
  ap.add_argument("--sessions", type=int, default=5)
  ap.add_argument("--rounds", type=int, default=16)
  ap.add_argument("--warmups", type=int, default=16)
  ap.add_argument("--settle-s", type=float, default=0.5)
  ap.add_argument("--sample-interval-s", type=float, default=0.02)
  ap.add_argument("--seed", type=int, default=251)
  ap.add_argument("--target-us", type=float, default=115.24)
  ap.add_argument("--det-mhz", type=int)
  ap.add_argument("--rows", type=int, default=12288)
  ap.add_argument("--gguf", type=pathlib.Path, default=pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  ap.add_argument("--arch", default="gfx1100")
  ap.add_argument("--out", type=pathlib.Path, default=OUT)
  ap.add_argument("--child-out", type=pathlib.Path)
  ap.add_argument("--lane", default="auto")
  args = ap.parse_args()
  return run_child(args) if args.child_out is not None else run_parent(args)


if __name__ == "__main__":
  raise SystemExit(main())
