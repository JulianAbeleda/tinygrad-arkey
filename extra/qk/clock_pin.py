#!/usr/bin/env python3
"""Reusable GPU clock pin for reproducible decode timing (RX 7900 XTX / gfx1100).

`auto` perf-state is clock-VOLATILE for short decode kernels: the GPU drops to idle (500MHz) between measurement
windows and an 8-iter warmup may not ramp it back to peak (2304MHz), so early-ctx measurements read 2-3x slow
(documented: decode-q8-clock-authority-result-20260620.md). `pin_peak()` forces sclk level 2 (2304MHz) + mclk
level 3 (1249MHz) via passwordless sudo sysfs; `restore_auto()` resets perf determinism and returns to `auto`.

Measurement policy: pin for the timing window, ALWAYS restore `auto` in a finally. Report the pinned lane.
Use as a context manager:  `with pinned_peak(enabled=True): ...measure...`
"""
from __future__ import annotations

import contextlib, pathlib, subprocess

DEV = "/sys/class/drm/card0/device"
DEV_SYS = f"{DEV}/power_dpm_force_performance_level"  # the perf-state leaf (single source of truth for the path)

# Canonical privileged perf-state mutations. The dangerous-power sysfs/rocm-smi strings live ONCE here (the
# boundary). A caller that needs a different provenance dict shape may wrap these with its own subprocess +
# formatting, but must NOT re-spell the sysfs writes -- see coding-principles "Contain Dangerous Power".
# (qk_decode_q8_model_route_timing_audit.py reuses these constants with its own artifact provenance shape.)
PIN_PEAK_CMD = f"echo manual > {DEV}/power_dpm_force_performance_level && echo 2 > {DEV}/pp_dpm_sclk && echo 3 > {DEV}/pp_dpm_mclk"
SET_AUTO_CMD = f"echo auto > {DEV}/power_dpm_force_performance_level"
RESET_PERF_DETERMINISM = ["sudo", "-n", "rocm-smi", "--resetperfdeterminism"]

def read_perf_state() -> str:
  """Read the GPU perf-state ('auto'/'manual'/...) -- the READ side of the perf-state boundary (no sudo)."""
  try: return pathlib.Path(DEV_SYS).read_text().strip()
  except OSError: return "unknown"

def _sudo(cmd: str) -> dict:
  p = subprocess.run(["sudo", "-n", "bash", "-c", cmd], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  return {"cmd": cmd, "rc": p.returncode, "ok": p.returncode == 0, "out": p.stdout[-300:]}

def pin_peak() -> dict:
  """force near-peak sclk(2304MHz idx2) + mclk(1249MHz idx3). Returns provenance dict."""
  return _sudo(PIN_PEAK_CMD)

def restore_auto() -> list[dict]:
  r = subprocess.run(RESET_PERF_DETERMINISM, text=True,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  return [{"cmd": "rocm-smi --resetperfdeterminism", "rc": r.returncode, "ok": r.returncode == 0},
          _sudo(SET_AUTO_CMD)]

@contextlib.contextmanager
def pinned_peak(enabled: bool = True):
  """pin peak clock for the duration; always restore auto. yields the pin provenance (or None if disabled)."""
  prov = pin_peak() if enabled else None
  try:
    yield prov
  finally:
    if enabled: restore_auto()

# --- the coarser `rocm-smi --setperflevel` idiom (distinct from the manual+sclk/mclk pin above) -------------------
def perflevel(level: str) -> subprocess.CompletedProcess:
  """Set the rocm-smi perf level ('high'/'auto'/...). The single boundary for the setperflevel mechanism."""
  return subprocess.run(["rocm-smi", "--setperflevel", level], capture_output=True, text=True)

@contextlib.contextmanager
def pinned_perflevel(level: str = "high", restore: str = "auto"):
  """Hold a rocm-smi perf level for the duration; ALWAYS restore in finally (the leak-safe boundary)."""
  perflevel(level)
  try:
    yield
  finally:
    perflevel(restore)
