#!/usr/bin/env python3
from __future__ import annotations

import argparse, contextlib, io, json, os, pathlib, re, statistics, subprocess, sys, threading, time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_q8_model_route_timing_audit_result.json"
DEV = pathlib.Path("/sys/class/drm/card0/device")
HWMON = next((DEV / "hwmon").glob("hwmon*"), None) if (DEV / "hwmon").exists() else None
ANSI = re.compile(r"\x1b\[[0-9;]*m")
LINE = re.compile(r"\*\*\*\s+\S+\s+\d+\s+(.+?)\s+arg\s+\d+\s+mem")
MHZ = re.compile(r"(\d+)Mhz")


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
      m = MHZ.search(line)
      return int(m.group(1)) if m else 0
  return 0


def sample() -> dict[str, Any]:
  row: dict[str, Any] = {
    "t": time.perf_counter(),
    "sclk": active_mhz("pp_dpm_sclk"),
    "mclk": active_mhz("pp_dpm_mclk"),
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
    self.stop_flag = False

  def run(self) -> None:
    while not self.stop_flag:
      self.rows.append(sample())
      time.sleep(self.interval_s)

  def finish(self) -> list[dict[str, Any]]:
    self.stop_flag = True
    self.join(timeout=2)
    return self.rows


def stat(vals: list[float]) -> dict[str, Any]:
  return {"n": len(vals), "min": min(vals) if vals else None, "median": statistics.median(vals) if vals else None, "max": max(vals) if vals else None}


def telemetry(rows: list[dict[str, Any]]) -> dict[str, Any]:
  active = [r for r in rows if r.get("gpu_busy", 0) > 0 or r.get("sclk", 0) > 800 or r.get("mem_busy", 0) > 0]
  def block(rs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
      "n": len(rs),
      "sclk_mhz": stat([float(r.get("sclk", 0)) for r in rs]),
      "mclk_mhz": stat([float(r.get("mclk", 0)) for r in rs]),
      "gpu_busy_pct": stat([float(r.get("gpu_busy", 0)) for r in rs]),
      "mem_busy_pct": stat([float(r.get("mem_busy", 0)) for r in rs]),
      "power_w": stat([float(r.get("power_w", 0)) for r in rs if "power_w" in r]),
      "temp_c": stat([float(r.get("temp_c", 0)) for r in rs if "temp_c" in r]),
      "perf_level_last": rs[-1].get("perf_level", "") if rs else "",
    }
  return {"all": block(rows), "active": block(active)}


# The privileged perf-state mutations are the SAME dangerous-power operations as qk_clock_pin's boundary; this
# script reuses the boundary's command constants (PIN_PEAK_CMD/SET_AUTO_CMD/RESET_PERF_DETERMINISM) so the sysfs
# writes are spelled in exactly one place. Only the provenance dict SHAPE differs ({cmd,returncode,stdout[-1000:],
# ok} vs the boundary's {cmd,rc,ok,out}) -- that formatting stays local because this script bakes it into its
# artifact bytes, but the operation itself is no longer duplicated.
from extra.qk_clock_pin import PIN_PEAK_CMD, SET_AUTO_CMD, RESET_PERF_DETERMINISM

def sudo(cmd: str) -> dict[str, Any]:
  p = subprocess.run(["sudo", "-n", "bash", "-c", cmd], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout[-1000:], "ok": p.returncode == 0}


def set_lane(lane: str) -> dict[str, Any]:
  if lane == "auto":
    return sudo(SET_AUTO_CMD)
  if lane == "manual_peak":
    return sudo(PIN_PEAK_CMD)
  raise ValueError(lane)


def restore_lane() -> list[dict[str, Any]]:
  p = subprocess.run(RESET_PERF_DETERMINISM, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  return [
    {"cmd": "rocm-smi --resetperfdeterminism", "returncode": p.returncode, "stdout": p.stdout[-1000:], "ok": p.returncode == 0},
    sudo(SET_AUTO_CMD),
  ]


def run_child(args: argparse.Namespace) -> int:
  set_res = set_lane(args.lane)
  time.sleep(args.settle_s)
  try:
    from tinygrad import Tensor, UOp, TinyJit, Context, GlobalCounters, Device
    from extra.llm_generate import load_model_and_tokenizer

    dev = Device[Device.DEFAULT]
    model, tok = load_model_and_tokenizer(args.model, args.max_context, seed=args.seed)
    for lin in (getattr(model, "_q4k_linears", None).linears if getattr(model, "_q4k_linears", None) else []):
      lin.decode_enabled = True
    ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox jumps. " * 800)
    ids = (ids * (1 + args.max_context // max(1, len(ids))))[:args.max_context]
    v_sp = UOp.variable("start_pos", 0, args.max_context - 1)
    temp = Tensor([0.0])
    sampler = Sampler(args.sample_interval_s)
    sampler.start()
    rows = []
    for ck in args.ckpts:
      use_flash = ck >= 1024
      for block in model.blk:
        block._use_flash, block._prefill_v2 = use_flash, False
      step = TinyJit(model.forward)
      tokid = int(ids[ck])
      out = Tensor([[tokid]], dtype="int32").contiguous()
      for i in range(args.warmups):
        out = step(out, v_sp.bind(ck + i), temp).realize()
      out = Tensor([[tokid]], dtype="int32").contiguous()
      W = []
      for i in range(args.nmeas):
        t0 = time.perf_counter()
        out = step(out, v_sp.bind(ck + i), temp)
        _ = int(out.item())
        W.append(time.perf_counter() - t0)
      out = Tensor([[tokid]], dtype="int32").contiguous()
      dev.synchronize()
      t0 = time.perf_counter()
      for i in range(args.nmeas):
        out = step(out, v_sp.bind(ck + i), temp)
      dev.synchronize()
      D = (time.perf_counter() - t0) / args.nmeas
      buf = io.StringIO()
      with contextlib.redirect_stdout(buf), Context(DEBUG=2):
        GlobalCounters.reset()
        step(out, v_sp.bind(ck + args.nmeas), temp).realize()
        gpu_dbg = GlobalCounters.time_sum_s
      progs = sum(1 for line in buf.getvalue().splitlines() if LINE.search(ANSI.sub("", line)))
      w_ms, d_ms = statistics.median(W) * 1e3, D * 1e3
      host = max(0.0, w_ms - d_ms)
      rows.append({
        "ctx": ck,
        "flash": use_flash,
        "wall_ms_W": round(w_ms, 3),
        "dispatch_ms_D": round(d_ms, 3),
        "host_sync_residual_ms": round(host, 3),
        "host_sync_pct_of_wall": round(100 * host / w_ms, 1),
        "tok_s_W": round(1000 / w_ms, 1),
        "tok_s_D_ceiling": round(1000 / d_ms, 1),
        "programs_per_token": progs,
        "debug2_unbatched_gpu_ms": round(gpu_dbg * 1e3, 2),
      })
    telem = telemetry(sampler.finish())
  finally:
    restore = restore_lane()
  med_host = statistics.median([r["host_sync_pct_of_wall"] for r in rows])
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_Q8_MODEL_ROUTE_TIMING_AUDIT_CHILD",
    "schema": "decode_q8_model_route_timing_audit_child_v1",
    "verdict": "PASS_DECODE_Q8_MODEL_ROUTE_TIMING_AUDIT_CHILD",
    "gate_pass": True,
    "commit": git_sha(),
    "mode": args.mode,
    "lane": args.lane,
    "q8_enabled": args.mode == "q8",
    "set_lane": set_res,
    "restore": restore,
    "ckpts": args.ckpts,
    "nmeas": args.nmeas,
    "method": "W=real decode (.item/token), D=dispatch-only graph replay with one final sync",
    "rows": rows,
    "median_host_sync_pct": round(med_host, 1),
    "telemetry": telem,
    "default_behavior_changed": False,
  }
  args.child_out.parent.mkdir(parents=True, exist_ok=True)
  args.child_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "mode": args.mode,
    "lane": args.lane,
    "median_tok_s_W": statistics.median([r["tok_s_W"] for r in rows]),
    "median_host_sync_pct": round(med_host, 1),
    "out": rel(args.child_out),
  }, indent=2))
  return 0


def aggregate_artifacts(args: argparse.Namespace, children: list[dict[str, Any]] | None = None) -> int:
  children = children or []
  artifacts = {}
  for lane in args.lanes.split(","):
    for mode in args.modes.split(","):
      out = args.out.parent / f"decode_q8_model_route_timing_{mode}_{lane}.json"
      if out.exists():
        artifacts[(mode, lane)] = json.loads(out.read_text())
      elif args.aggregate_existing:
        children.append({"mode": mode, "lane": lane, "artifact": rel(out), "missing_artifact": True})
  rows = []
  for lane in args.lanes.split(","):
    base, q8 = artifacts.get(("baseline", lane), {}), artifacts.get(("q8", lane), {})
    base_by = {r["ctx"]: r for r in base.get("rows", [])}
    q8_by = {r["ctx"]: r for r in q8.get("rows", [])}
    for ctx in sorted(set(base_by) & set(q8_by)):
      b, q = base_by[ctx], q8_by[ctx]
      rows.append({
        "lane": lane,
        "ctx": ctx,
        "baseline_tok_s_W": b["tok_s_W"],
        "q8_tok_s_W": q["tok_s_W"],
        "speedup_W": round(q["tok_s_W"] / b["tok_s_W"], 4) if b["tok_s_W"] else None,
        "baseline_dispatch_tok_s_D": b["tok_s_D_ceiling"],
        "q8_dispatch_tok_s_D": q["tok_s_D_ceiling"],
        "q8_host_sync_pct": q["host_sync_pct_of_wall"],
        "q8_programs_per_token": q["programs_per_token"],
      })
  summary: dict[str, Any] = {}
  for lane in args.lanes.split(","):
    lr = [r for r in rows if r["lane"] == lane]
    summary[lane] = {
      "ctxs": [r["ctx"] for r in lr],
      "median_speedup_W": statistics.median([r["speedup_W"] for r in lr]) if lr else None,
      "min_speedup_W": min([r["speedup_W"] for r in lr], default=None),
      "median_q8_host_sync_pct": statistics.median([r["q8_host_sync_pct"] for r in lr]) if lr else None,
      "q8_median_tok_s_W": statistics.median([r["q8_tok_s_W"] for r in lr]) if lr else None,
    }
  gates = {
    "all_children_present": len(artifacts) == len(args.lanes.split(",")) * len(args.modes.split(",")),
    "manual_peak_q8_speedup_positive": (summary.get("manual_peak", {}).get("min_speedup_W") or 0) > 1.0,
    "manual_peak_host_sync_not_target": (
      (v := summary.get("manual_peak", {}).get("median_q8_host_sync_pct")) is not None and v < 10
    ),
  }
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_Q8_MODEL_ROUTE_TIMING_AUDIT",
    "schema": "decode_q8_model_route_timing_audit_v1",
    "verdict": "PASS_DECODE_Q8_MODEL_ROUTE_TIMING_AUDIT" if all(gates.values()) else "BLOCKED_DECODE_Q8_MODEL_ROUTE_TIMING_AUDIT",
    "gate_pass": all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": True,
    "commit": git_sha(),
    "rows": rows,
    "summary": summary,
    "gates": gates,
    "children": children,
    "artifacts": {f"{m}_{l}": rel(args.out.parent / f"decode_q8_model_route_timing_{m}_{l}.json")
                  for (m, l) in artifacts},
    "decision": {
      "if_pass": "Actual in-model q8 graph route has no material host-sync wait lever under manual_peak; primitive fusion is not justified by host-wait evidence.",
      "if_blocked": "Inspect child artifacts before deciding whether route install, clock lane, or timing path failed.",
    },
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"verdict": result["verdict"], "summary": summary, "gates": gates, "out": rel(args.out)}, indent=2))
  return 0 if all(gates.values()) else 1


def run_parent(args: argparse.Namespace) -> int:
  children, artifacts = [], {}
  for lane in args.lanes.split(","):
    for mode in args.modes.split(","):
      out = args.out.parent / f"decode_q8_model_route_timing_{mode}_{lane}.json"
      env = os.environ.copy()
      env.setdefault("DEV", "AMD")
      env.setdefault("JIT", "1")
      env["PYTHONPATH"] = str(ROOT)
      if mode == "q8":
        env["Q8_FFN_HANDWRITTEN"] = "1"
      else:
        env.pop("Q8_FFN_HANDWRITTEN", None)
      cmd = [
        sys.executable, rel(pathlib.Path(__file__).resolve()), "--child-out", rel(out), "--mode", mode, "--lane", lane,
        "--nmeas", str(args.nmeas), "--warmups", str(args.warmups), "--sample-interval-s", str(args.sample_interval_s),
        "--settle-s", str(args.settle_s), "--model", args.model, "--max-context", str(args.max_context),
        "--seed", str(args.seed), "--ckpts", *[str(x) for x in args.ckpts],
      ]
      p = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
      children.append({"mode": mode, "lane": lane, "cmd": cmd, "returncode": p.returncode, "stdout": p.stdout[-4000:], "artifact": rel(out)})
      if out.exists():
        artifacts[(mode, lane)] = json.loads(out.read_text())
  return aggregate_artifacts(args, children)


def main() -> int:
  ap = argparse.ArgumentParser(description="In-model q8 graph route W/D timing under clock lanes")
  ap.add_argument("--lanes", default="auto,manual_peak")
  ap.add_argument("--modes", default="baseline,q8")
  ap.add_argument("--ckpts", nargs="+", type=int, default=[512, 1024])
  ap.add_argument("--nmeas", type=int, default=20)
  ap.add_argument("--warmups", type=int, default=8)
  ap.add_argument("--sample-interval-s", type=float, default=0.02)
  ap.add_argument("--settle-s", type=float, default=0.5)
  ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--max-context", type=int, default=4608)
  ap.add_argument("--seed", type=int, default=20260620)
  ap.add_argument("--out", type=pathlib.Path, default=OUT)
  ap.add_argument("--aggregate-existing", action="store_true")
  ap.add_argument("--child-out", type=pathlib.Path)
  ap.add_argument("--mode", choices=["baseline", "q8"], default="baseline")
  ap.add_argument("--lane", choices=["auto", "manual_peak"], default="auto")
  args = ap.parse_args()
  if args.child_out is not None: return run_child(args)
  if args.aggregate_existing: return aggregate_artifacts(args)
  return run_parent(args)


if __name__ == "__main__":
  raise SystemExit(main())
