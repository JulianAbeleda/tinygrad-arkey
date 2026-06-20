#!/usr/bin/env python3
"""Prefill clock/DPM authority — telemetry sampler + clock-lane control + probe/matrix drivers + classifier.

Scope: docs/prefill-clock-dpm-authority-scope-20260619.md. Settles whether tinygrad WMMA / Tensile / llama prefill
can be compared under a TELEMETRY-VERIFIED clock lane on this consumer RX 7900 XTX (gfx1100), or whether all claims
must be telemetry-binned by measured sclk. amd-smi is ABSENT here -> rocm-smi + sysfs only.

Usage:
  P0 inventory : python extra/qk_prefill_clock_dpm_authority.py inventory
  P2 probes    : python extra/qk_prefill_clock_dpm_authority.py probe --lane auto
  P3 matrix    : DEV=AMD PREFILL_V2=1 python extra/qk_prefill_clock_dpm_authority.py matrix --lanes auto,manual_peak
Clock writes need root (passwordless sudo on this box). NO kernel/route/default changes here -- measure only.
"""
from __future__ import annotations
import os, sys, json, time, threading, subprocess, statistics, argparse, re, pathlib
ART = pathlib.Path("bench/qk-prefill-clock-dpm-authority"); ART.mkdir(parents=True, exist_ok=True)
DEV = "/sys/class/drm/card0/device"
HWMON = next((p for p in pathlib.Path(f"{DEV}/hwmon").glob("hwmon*")), None)

def _read(p, default=""):
  try: return open(p).read().strip()
  except OSError: return default
def _active_mhz(dpm_file):
  for ln in _read(f"{DEV}/{dpm_file}").splitlines():
    if "*" in ln:
      m = re.search(r"(\d+)Mhz", ln)
      return int(m.group(1)) if m else 0
  return 0
def sample():
  s = {"t": time.perf_counter(),
       "sclk": _active_mhz("pp_dpm_sclk"), "mclk": _active_mhz("pp_dpm_mclk"),
       "fclk": _active_mhz("pp_dpm_fclk"), "socclk": _active_mhz("pp_dpm_socclk"),
       "gpu_busy": int(_read(f"{DEV}/gpu_busy_percent", "0") or 0),
       "mem_busy": int(_read(f"{DEV}/mem_busy_percent", "0") or 0),
       "perf_level": _read(f"{DEV}/power_dpm_force_performance_level")}
  if HWMON:
    s["power_w"] = round(int(_read(f"{HWMON}/power1_average", "0") or 0) / 1e6, 1)
    s["temp_c"] = round(int(_read(f"{HWMON}/temp1_input", "0") or 0) / 1e3, 1)
  return s

class Sampler(threading.Thread):
  """Background telemetry sampler. Reads sysfs (fast, ~no perturbation). Writes nothing until stop()."""
  def __init__(self, interval=0.06, run_id=""):
    super().__init__(daemon=True); self.interval=interval; self.run_id=run_id; self.rows=[]; self._stopflag=False
  def run(self):
    while not self._stopflag:
      r = sample(); r["run_id"] = self.run_id; self.rows.append(r); time.sleep(self.interval)
  def stop(self):
    self._stopflag = True; self.join(timeout=2); return self.rows
  def summary(self, busy_only=True):
    rs = [r for r in self.rows if (not busy_only or r["gpu_busy"] > 0)] or self.rows
    sclk=[r["sclk"] for r in rs]; mclk=[r["mclk"] for r in rs]
    return {"n": len(rs), "sclk_med": statistics.median(sclk) if sclk else 0,
            "sclk_min": min(sclk) if sclk else 0, "sclk_max": max(sclk) if sclk else 0,
            "mclk_med": statistics.median(mclk) if mclk else 0,
            "power_max": max((r.get("power_w",0) for r in rs), default=0),
            "temp_max": max((r.get("temp_c",0) for r in rs), default=0),
            "perf_level": rs[-1]["perf_level"] if rs else ""}

# ---- clock lane control (root) + verification ----
def _sudo(cmd): return subprocess.run(["sudo","-n","bash","-c",cmd], capture_output=True, text=True).returncode==0
LANES = {
  "auto":        lambda: _sudo(f"echo auto > {DEV}/power_dpm_force_performance_level"),
  "high":        lambda: _sudo(f"echo high > {DEV}/power_dpm_force_performance_level"),
  "profile_peak":lambda: _sudo(f"echo profile_peak > {DEV}/power_dpm_force_performance_level"),
  "manual_peak": lambda: _sudo(f"echo manual > {DEV}/power_dpm_force_performance_level && echo 2 > {DEV}/pp_dpm_sclk && echo 3 > {DEV}/pp_dpm_mclk"),
}
def set_lane(lane, det_mhz=None):
  if lane == "determinism" and det_mhz:
    return subprocess.run(["sudo","-n","rocm-smi","--setperfdeterminism",str(det_mhz)],capture_output=True).returncode==0
  return LANES.get(lane, LANES["auto"])()
def restore():
  subprocess.run(["sudo","-n","rocm-smi","--resetperfdeterminism"],capture_output=True)
  _sudo(f"echo auto > {DEV}/power_dpm_force_performance_level")

# ---- classifier (P4) ----
def classify(summ, intended_lane, intended_sclk=None):
  if intended_lane == "auto": return "user-realistic-AUTO"
  if intended_sclk and summ["sclk_med"] and abs(summ["sclk_med"]-intended_sclk)/intended_sclk > 0.05:
    return "unsupported-control(clock-did-not-hold)"
  if summ["sclk_max"] and summ["sclk_min"] and (summ["sclk_max"]-summ["sclk_min"])/max(summ["sclk_max"],1) > 0.05:
    return "clock-confounded(variance>5%)"
  if summ["power_max"] >= 320 and intended_sclk and summ["sclk_med"] < intended_sclk*0.95:
    return "thermal/power-throttled"
  return "controlled-clock-authority"

# ---- P0 inventory ----
def inventory():
  out = {"amd_smi_present": subprocess.run(["which","amd-smi"],capture_output=True).returncode==0,
         "rocm_smi": _read("/dev/null") or "/usr/bin/rocm-smi",
         "dpm_sclk": _read(f"{DEV}/pp_dpm_sclk"), "dpm_mclk": _read(f"{DEV}/pp_dpm_mclk"),
         "pp_od_present": pathlib.Path(f"{DEV}/pp_od_clk_voltage").exists(),
         "idle_sample": sample()}
  (ART/"supported_controls_live.json").write_text(json.dumps(out, indent=2))
  print(json.dumps(out, indent=2)); return out

# ---- P2 probe: does workload SHAPE change DPM state? (idle / sustained matmul loop) ----
def probe(lane="auto", det_mhz=None, secs=5):
  set_lane(lane, det_mhz); time.sleep(0.5)
  res={}
  # idle
  sp=Sampler(run_id=f"{lane}:idle"); sp.start(); time.sleep(secs); res["idle"]=sp.stop() and sp.summary(busy_only=False)
  # sustained WMMA-shaped matmul loop (no model)
  from tinygrad import Tensor, dtypes, Device
  A=Tensor.randn(12288,4096,dtype=dtypes.float16,device='AMD').realize(); B=Tensor.randn(4096,512,dtype=dtypes.float16,device='AMD').realize()
  for _ in range(5): (A@B).realize(); Device['AMD'].synchronize()
  sp=Sampler(run_id=f"{lane}:wmma_loop"); sp.start()
  end=time.time()+secs
  while time.time()<end: (A@B).realize(); Device['AMD'].synchronize()
  res["wmma_loop"]=sp.stop() and sp.summary()
  restore()
  out={"lane":lane,"det_mhz":det_mhz,"idle":res["idle"],"wmma_loop":res["wmma_loop"],
       "verdict":classify(res["wmma_loop"], lane, intended_sclk=(det_mhz or (2304 if "peak" in lane else None)))}
  (ART/"dpm_probe_matrix.json").write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2)); return out

# ---- P3 matrix: reuse the reconciliation harness shape (model.forward, interleaved) with telemetry+lanes ----
def matrix(lanes):
  # NOTE: the per-engine interleaved A/B is in /tmp/prefill_recon.py (reconciliation harness, model.forward).
  # This driver wraps it per clock lane with the Sampler, then classifies. Full impl = the project's P3 execution.
  print("P3 matrix driver: for each lane set_lane()->verify via Sampler->run the model.forward interleaved A/B "
        "(symbolic|concrete|+TensileFFN|+TensileFFN+qo)->record tok/s+telemetry->classify. "
        "llama.cpp run separately under the SAME lane (VRAM exclusive). See scope P3. Lanes=%s" % lanes)

if __name__ == "__main__":
  ap=argparse.ArgumentParser(); ap.add_argument("cmd",choices=["inventory","probe","matrix"])
  ap.add_argument("--lane",default="auto"); ap.add_argument("--det-mhz",type=int,default=None)
  ap.add_argument("--lanes",default="auto"); a=ap.parse_args()
  try:
    if a.cmd=="inventory": inventory()
    elif a.cmd=="probe": probe(a.lane,a.det_mhz)
    elif a.cmd=="matrix": matrix(a.lanes.split(","))
  finally:
    if a.cmd in ("probe","matrix"): restore()
