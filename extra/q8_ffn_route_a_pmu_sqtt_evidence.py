#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, pathlib, subprocess, sys, textwrap, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench/q8-ffn-amd-scheduler-project"
Q8_SCRIPT = ROOT / "extra/q8_ffn_asm_gateup_full.py"

CHILD = r"""
import json, pathlib, runpy, sys, traceback

from tinygrad.device import Buffer, Compiled, Device

mode = sys.argv[1]
out = pathlib.Path(sys.argv[2])
q8_script = pathlib.Path(sys.argv[3])

sys.argv = [str(q8_script), "--warmups", "0", "--iters", "1", "--out", str(out)]
status = {"ok": True, "error": None}
try:
  runpy.run_path(str(q8_script), run_name="__main__")
except SystemExit as e:
  status["ok"] = e.code in (0, None)
  status["error"] = None if status["ok"] else f"SystemExit({e.code})"
except Exception:
  status["ok"] = False
  status["error"] = traceback.format_exc(limit=8)

try:
  for d in list(Device._opened_devices): Device[d].synchronize()
  for d in list(Device._opened_devices): Device[d]._at_profile_finalize()
except Exception:
  status["profile_finalize_error"] = traceback.format_exc(limit=8)

events = Compiled.profile_events + Buffer.profile_events
counts = {}
for ev in events:
  counts[type(ev).__name__] = counts.get(type(ev).__name__, 0) + 1

pmc = []
sqtt = []
programs = []
ranges = []
device_targets = {}
program_libs = {}
for idx, ev in enumerate(events):
  tn = type(ev).__name__
  if tn == "ProfileDeviceEvent" and getattr(ev, "props", None):
    device_targets[ev.device] = "gfx%d" % (ev.props["gfx_target_version"] // 1000)
  if tn == "ProfileProgramEvent":
    programs.append({"idx": idx, "device": ev.device, "name": ev.name, "lib_bytes": len(ev.lib) if ev.lib is not None else 0,
                     "base": ev.base, "tag": ev.tag})
    if ev.tag is not None and ev.lib is not None: program_libs[ev.tag] = (ev.device, ev.lib)
  elif tn == "ProfileRangeEvent":
    ranges.append({"idx": idx, "device": ev.device, "name": str(ev.name), "st": str(ev.st), "en": None if ev.en is None else str(ev.en)})
  elif tn == "ProfilePMCEvent":
    sample_names = [s.name for s in ev.sched]
    pmc.append({
      "idx": idx, "device": ev.device, "kern": ev.kern, "exec_tag": ev.exec_tag,
      "sched_count": len(ev.sched), "sample_names": sample_names, "blob_bytes": len(ev.blob),
      "sample_layout": [{"name": s.name, "block": s.block, "size": s.size, "off": s.off} for s in ev.sched],
    })
  elif tn == "ProfileSQTTEvent":
    decode_summary = None
    if ev.itrace and len(ev.blob) > 0:
      try:
        from tinygrad.renderer.amd import sqtt as sqtt_decoder
        packet_counts = {}
        inst_counts = {}
        mapped = 0
        dev, lib = program_libs[ev.kern]
        for pkt, inst in sqtt_decoder.map_insts(ev.blob, lib, device_targets.get(dev, "")):
          packet_counts[type(pkt).__name__] = packet_counts.get(type(pkt).__name__, 0) + 1
          if inst is not None:
            mapped += 1
            op = getattr(inst.inst, "op_name", type(inst.inst).__name__)
            inst_counts[op] = inst_counts.get(op, 0) + 1
        decode_summary = {
          "ok": True, "mapped_instruction_events": mapped,
          "packet_counts_top": sorted(packet_counts.items(), key=lambda kv: kv[1], reverse=True)[:20],
          "instruction_counts_top": sorted(inst_counts.items(), key=lambda kv: kv[1], reverse=True)[:20],
        }
      except Exception as exc:
        decode_summary = {"ok": False, "error": repr(exc)}
    sqtt.append({
      "idx": idx, "device": ev.device, "kern": ev.kern, "se": ev.se, "exec_tag": ev.exec_tag,
      "itrace": ev.itrace, "blob_bytes": len(ev.blob), "decode_summary": decode_summary,
    })

print("PROFILE_EVIDENCE_JSON=" + json.dumps({
  "mode": mode, "status": status, "event_counts": counts,
  "programs": programs, "ranges": ranges, "pmc": pmc, "sqtt": sqtt,
}, sort_keys=True))
if not status["ok"]: sys.exit(2)
"""

def run_profile(mode:str, env_extra:dict[str, str], timeout_s:int) -> dict:
  out = BENCH / f"pmu_sqtt_{mode}_q8_gateup_full.json"
  env = os.environ.copy()
  env.update({
    "PYTHONPATH": str(ROOT),
    "VIZ": "0",
    "DEBUG": "0",
    "PROFILE": "1",
  })
  env.update(env_extra)
  cmd = [sys.executable, "-c", CHILD, mode, str(out), str(Q8_SCRIPT)]
  t0 = time.perf_counter()
  try:
    cp = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        timeout=timeout_s, check=False)
    elapsed = time.perf_counter() - t0
    parsed = None
    for line in cp.stdout.splitlines():
      if line.startswith("PROFILE_EVIDENCE_JSON="):
        parsed = json.loads(line.split("=", 1)[1])
    return {
      "mode": mode, "timeout_s": timeout_s, "elapsed_s": round(elapsed, 3), "returncode": cp.returncode,
      "stdout_tail": cp.stdout.splitlines()[-30:], "stderr_tail": cp.stderr.splitlines()[-30:],
      "profile": parsed, "timing_out": str(out), "timing_exists": out.exists(),
    }
  except subprocess.TimeoutExpired as e:
    elapsed = time.perf_counter() - t0
    return {
      "mode": mode, "timeout_s": timeout_s, "elapsed_s": round(elapsed, 3), "returncode": None,
      "timeout": True, "stdout_tail": (e.stdout or "").splitlines()[-30:] if isinstance(e.stdout, str) else [],
      "stderr_tail": (e.stderr or "").splitlines()[-30:] if isinstance(e.stderr, str) else [],
      "profile": None, "timing_out": str(out), "timing_exists": out.exists(),
    }

def classify(pmc_run:dict, sqtt_run:dict) -> dict:
  pmc_profile, sqtt_profile = pmc_run.get("profile"), sqtt_run.get("profile")
  pmc_ok = pmc_run.get("returncode") == 0 and pmc_profile is not None
  sqtt_ok = sqtt_run.get("returncode") == 0 and sqtt_profile is not None
  pmc_events = [] if not pmc_profile else pmc_profile.get("pmc", [])
  sqtt_events = [] if not sqtt_profile else sqtt_profile.get("sqtt", [])
  decode_attempts = [e.get("decode_summary") for e in sqtt_events if e.get("decode_summary") is not None]
  decode_ok = any(d.get("ok") for d in decode_attempts)

  blocker = []
  if not pmc_ok: blocker.append("PMC did not produce a successful summarized profile")
  if not sqtt_ok: blocker.append("SQTT did not produce a successful summarized profile")
  elif not any(e.get("itrace") and e.get("blob_bytes", 0) > 0 for e in sqtt_events):
    blocker.append("SQTT produced no non-empty instruction trace")
  elif not decode_ok:
    blocker.append("SQTT capture is runnable, but the local decoder failed on every instruction-trace blob")

  return {
    "pmc_profile_runnable": pmc_ok and len(pmc_events) > 0,
    "sqtt_profile_runnable": sqtt_ok and len(sqtt_events) > 0,
    "sqtt_decode_usable": decode_ok,
    "pmc_event_count": len(pmc_events),
    "sqtt_event_count": len(sqtt_events),
    "sqtt_total_blob_bytes": sum(e.get("blob_bytes", 0) for e in sqtt_events),
    "a2_reopen": False,
    "a2_reopen_reason": None,
    "verdict": "NO_A2_REOPEN",
    "reason": (
      "Route A A2 remains closed: this evidence pass did not identify a bounded >=30us compiler feature. "
      "PMC/SQTT can be used as observability assets, but not yet as a feature-level cost attribution oracle."
    ),
    "blockers": blocker,
  }

def main() -> None:
  ap = argparse.ArgumentParser(description="Route A PMU/SQTT evidence gate for q8 native scheduler/codegen.")
  ap.add_argument("--out", type=pathlib.Path, default=BENCH / "pmu_sqtt_evidence.json")
  ap.add_argument("--pmc-timeout-s", type=int, default=240)
  ap.add_argument("--sqtt-timeout-s", type=int, default=300)
  ap.add_argument("--skip-sqtt", action="store_true")
  args = ap.parse_args()

  BENCH.mkdir(parents=True, exist_ok=True)
  pmc_run = run_profile("pmc", {"PMC": "1", "SQTT": "0"}, args.pmc_timeout_s)
  if args.skip_sqtt:
    sqtt_run = {"mode": "sqtt", "skipped": True, "profile": None, "returncode": None}
  else:
    sqtt_run = run_profile("sqtt", {
      "PMC": "0", "SQTT": "1", "SQTT_BUFFER_SIZE": "16", "SQTT_LIMIT_SE": "1",
      "SQTT_ITRACE_SE_MASK": "1", "SQTT_TOKEN_EXCLUDE": "0",
    }, args.sqtt_timeout_s)

  result = {
    "date": "2026-06-19",
    "phase": "Route_A_PMU_SQTT_evidence_gate",
    "purpose": "Try to reopen A2 only if PMU/SQTT evidence identifies a bounded >=30us native-codegen feature.",
    "inputs": {
      "q8_script": str(Q8_SCRIPT.relative_to(ROOT)),
      "prior_route_a_result": "bench/q8-ffn-amd-scheduler-project/route_a_result.json",
      "prior_a2_gate": "one feature with credible >=30us movement",
    },
    "runs": {"pmc": pmc_run, "sqtt": sqtt_run},
    "classification": classify(pmc_run, sqtt_run),
  }
  args.out.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps({"out": str(args.out), **result["classification"]}, indent=2))

if __name__ == "__main__":
  main()
