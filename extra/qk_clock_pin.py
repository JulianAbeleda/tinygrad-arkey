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

import contextlib, subprocess

DEV = "/sys/class/drm/card0/device"

def _sudo(cmd: str) -> dict:
  p = subprocess.run(["sudo", "-n", "bash", "-c", cmd], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  return {"cmd": cmd, "rc": p.returncode, "ok": p.returncode == 0, "out": p.stdout[-300:]}

def pin_peak() -> dict:
  """force near-peak sclk(2304MHz idx2) + mclk(1249MHz idx3). Returns provenance dict."""
  return _sudo(f"echo manual > {DEV}/power_dpm_force_performance_level && echo 2 > {DEV}/pp_dpm_sclk && echo 3 > {DEV}/pp_dpm_mclk")

def restore_auto() -> list[dict]:
  r = subprocess.run(["sudo", "-n", "rocm-smi", "--resetperfdeterminism"], text=True,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  return [{"cmd": "rocm-smi --resetperfdeterminism", "rc": r.returncode, "ok": r.returncode == 0},
          _sudo(f"echo auto > {DEV}/power_dpm_force_performance_level")]

@contextlib.contextmanager
def pinned_peak(enabled: bool = True):
  """pin peak clock for the duration; always restore auto. yields the pin provenance (or None if disabled)."""
  prov = pin_peak() if enabled else None
  try:
    yield prov
  finally:
    if enabled: restore_auto()
