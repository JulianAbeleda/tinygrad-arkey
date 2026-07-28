#!/usr/bin/env python3
"""Reusable GPU clock pin for reproducible decode/prefill timing on AMD gfx1100.

`auto` perf-state is clock-volatile for short kernels: the GPU can drop to idle between measurement windows, so
short benchmark runs can read 2-3x slow. `pin_peak()` forces fixed sclk/mclk levels via the authorized ROCm SMI
interface, with passwordless sudo sysfs as a fallback; `restore_auto()` returns to `auto`.

Measurement policy: pin for the timing window, always restore `auto` in a finally. Report the pinned lane.
Use as a context manager: `with pinned_peak(enabled=True): ...measure...`
"""
from __future__ import annotations

import contextlib
import pathlib
import subprocess
from collections.abc import Iterator
from typing import Any

DEV = "/sys/class/drm/card0/device"
DEV_SYS = f"{DEV}/power_dpm_force_performance_level"

# Canonical privileged perf-state mutations. Keep the sysfs/rocm-smi strings centralized here.
PIN_PEAK_CMD = f"echo manual > {DEV}/power_dpm_force_performance_level && echo 2 > {DEV}/pp_dpm_sclk && echo 3 > {DEV}/pp_dpm_mclk"
SET_AUTO_CMD = f"echo auto > {DEV}/power_dpm_force_performance_level"
RESET_PERF_DETERMINISM = ["sudo", "-n", "rocm-smi", "--resetperfdeterminism"]
ROCM_SMI_PIN_CMD = ["rocm-smi", "--setperflevel", "manual", "--setsclk", "2", "--setmclk", "3"]
ROCM_SMI_AUTO_CMD = ["rocm-smi", "--setperflevel", "auto"]


def read_perf_state() -> str:
  """Read the GPU perf-state ('auto'/'manual'/...) without sudo."""
  try:
    return pathlib.Path(DEV_SYS).read_text().strip()
  except OSError:
    return "unknown"


def _sudo(cmd: str) -> dict[str, Any]:
  p = subprocess.run(["sudo", "-n", "bash", "-c", cmd], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  return {"cmd": cmd, "rc": p.returncode, "ok": p.returncode == 0, "out": p.stdout[-300:]}


def pin_peak() -> dict[str, Any]:
  """Force near-peak sclk/mclk. Returns a provenance dict."""
  p = subprocess.run(ROCM_SMI_PIN_CMD, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  if p.returncode == 0:
    return {"cmd": " ".join(ROCM_SMI_PIN_CMD), "rc": p.returncode, "ok": True, "out": p.stdout[-300:], "transport": "rocm_smi"}
  fallback = _sudo(PIN_PEAK_CMD)
  fallback["rocm_smi"] = {"cmd": " ".join(ROCM_SMI_PIN_CMD), "rc": p.returncode, "out": p.stdout[-300:]}
  fallback["transport"] = "sudo_sysfs"
  return fallback


def restore_auto() -> list[dict[str, Any]]:
  """Reset perf determinism and return the device to auto perf-state."""
  r = subprocess.run(ROCM_SMI_AUTO_CMD, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  if r.returncode == 0:
    return [{"cmd": " ".join(ROCM_SMI_AUTO_CMD), "rc": r.returncode, "ok": True, "out": r.stdout[-300:], "transport": "rocm_smi"}]
  reset = subprocess.run(RESET_PERF_DETERMINISM, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  return [{"cmd": "rocm-smi --resetperfdeterminism", "rc": reset.returncode, "ok": reset.returncode == 0}, _sudo(SET_AUTO_CMD)]


@contextlib.contextmanager
def pinned_peak(enabled: bool = True) -> Iterator[dict[str, Any] | None]:
  """Pin peak clocks for the duration; always restore auto. Yields pin provenance, or None if disabled."""
  prov = pin_peak() if enabled else None
  try:
    yield prov
  finally:
    if enabled:
      restore_auto()


def perflevel(level: str) -> subprocess.CompletedProcess[str]:
  """Set the rocm-smi perf level ('high'/'auto'/...)."""
  return subprocess.run(["rocm-smi", "--setperflevel", level], capture_output=True, text=True)


@contextlib.contextmanager
def pinned_perflevel(level: str = "high", restore: str = "auto") -> Iterator[None]:
  """Hold a rocm-smi perf level for the duration; always restore in finally."""
  perflevel(level)
  try:
    yield
  finally:
    perflevel(restore)
