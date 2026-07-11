#!/usr/bin/env python3
"""Best-effort clock, power, temperature, and VRAM telemetry with explicit failures."""
from __future__ import annotations

import errno
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable

SCHEMA = "tinygrad.amd_telemetry_trace.v1"
DEFAULT_SENSORS = {
  "performance_level": "/sys/class/drm/card0/device/power_dpm_force_performance_level",
  "gpu_busy_percent": "/sys/class/drm/card0/device/gpu_busy_percent",
  "vram_total_bytes": "/sys/class/drm/card0/device/mem_info_vram_total",
  "vram_used_bytes": "/sys/class/drm/card0/device/mem_info_vram_used",
  "core_dpm": "/sys/class/drm/card0/device/pp_dpm_sclk",
  "memory_dpm": "/sys/class/drm/card0/device/pp_dpm_mclk",
  "power_uw": "/sys/class/drm/card0/device/hwmon/hwmon1/power1_average",
  "temperature_mc": "/sys/class/drm/card0/device/hwmon/hwmon1/temp1_input",
  "core_clock_hz": "/sys/class/drm/card0/device/hwmon/hwmon1/freq1_input",
  "memory_clock_hz": "/sys/class/drm/card0/device/hwmon/hwmon1/freq2_input",
}


def read_sensor(path: str | Path) -> dict[str, Any]:
  try:
    text = Path(path).read_text().strip()
    if text == "": return {"status": "zero_suspect", "value": None, "reason": "empty sensor value"}
    try: value: Any = int(text)
    except ValueError: value = text
    return {"status": "live", "value": value}
  except FileNotFoundError as exc:
    return {"status": "unsupported", "value": None, "error": str(exc), "errno": errno.ENOENT}
  except PermissionError as exc:
    return {"status": "blocked", "value": None, "error": str(exc), "errno": errno.EACCES}
  except OSError as exc:
    return {"status": "blocked", "value": None, "error": str(exc), "errno": exc.errno}


def collect_telemetry(process_or_window: str, *, samples: int = 1, interval_s: float = 0.0,
                      sensors: dict[str, str] | None = None, system_snapshot_id: str | None = None,
                      experiment_id: str | None = None) -> dict[str, Any]:
  if samples < 1: raise ValueError("samples must be positive")
  if not isinstance(interval_s, (int, float)) or isinstance(interval_s, bool) or not math.isfinite(interval_s) or interval_s < 0:
    raise ValueError("interval_s must be finite and non-negative")
  configured = sensors or DEFAULT_SENSORS
  rows = []
  for idx in range(samples):
    rows.append({"sample": idx, "monotonic_ns": time.monotonic_ns(),
                 "sensors": {name: {"path": path, **read_sensor(path)} for name, path in configured.items()}})
    if idx + 1 < samples and interval_s: time.sleep(interval_s)
  return {"schema": SCHEMA, "window": process_or_window, "system_snapshot_id": system_snapshot_id,
          "experiment_id": experiment_id, "sample_count": samples, "interval_s": interval_s, "samples": rows}


def collect_process_telemetry(command: list[str], *, interval_s: float = 0.01, sensors: dict[str, str] | None = None,
                              system_snapshot_id: str, experiment_id: str, candidate_id: str,
                              binary_sha256: str, timeout: float = 120.0) -> dict[str, Any]:
  if not command or not all(isinstance(arg, str) and arg for arg in command): raise ValueError("command must be a non-empty argv list")
  if len(binary_sha256) != 64 or any(c not in "0123456789abcdef" for c in binary_sha256):
    raise ValueError("binary_sha256 must be lowercase SHA-256")
  configured, rows, started = sensors or DEFAULT_SENSORS, [], time.monotonic()
  try:
    proc = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  except OSError as exc:
    return {"schema": SCHEMA, "window": "process", "status": "blocked", "error": f"{type(exc).__name__}: {exc}",
            "command": command, "system_snapshot_id": system_snapshot_id, "experiment_id": experiment_id,
            "candidate_id": candidate_id, "binary_sha256": binary_sha256, "sample_count": 0, "samples": []}
  timed_out = False
  while proc.poll() is None:
    rows.append({"sample": len(rows), "monotonic_ns": time.monotonic_ns(),
                 "sensors": {name: {"path": path, **read_sensor(path)} for name, path in configured.items()}})
    if time.monotonic() - started > timeout:
      proc.kill(); timed_out = True; break
    time.sleep(interval_s)
  stdout, stderr = proc.communicate()
  return {"schema": SCHEMA, "window": "process", "status": "blocked" if timed_out or proc.returncode else "live",
          "system_snapshot_id": system_snapshot_id, "experiment_id": experiment_id, "candidate_id": candidate_id,
          "binary_sha256": binary_sha256, "command": command, "returncode": proc.returncode,
          "sample_count": len(rows), "interval_s": interval_s, "samples": rows,
          "stdout": stdout[-4000:], "stderr": stderr[-4000:], "timed_out": timed_out}


def collect_mmq_kernel_window_telemetry(writeback_mode: str, *, repetitions: int, interval_s: float,
                                        system_snapshot_id: str, experiment_id: str, candidate_id: str,
                                        binary_sha256: str, sensors: dict[str, str] | None = None,
                                        timeout: float = 120.0) -> dict[str, Any]:
  if writeback_mode not in ("gated_matrix_v0", "direct_owner_v0"): raise ValueError("writeback_mode is invalid")
  if repetitions < 1: raise ValueError("repetitions must be positive")
  if len(binary_sha256) != 64 or any(c not in "0123456789abcdef" for c in binary_sha256):
    raise ValueError("binary_sha256 must be lowercase SHA-256")
  root = Path(__file__).resolve().parents[2]
  command = ["/usr/bin/env", f"PYTHONPATH={root}", "PROFILE=0", "PMC=0", sys.executable,
             str(root / "extra/qk/mmq_amd_pmc.py"), "--mmq-loop", writeback_mode, "0", str(repetitions)]
  configured, rows, started = sensors or DEFAULT_SENSORS, [], time.monotonic()
  proc = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  assert proc.stdout is not None
  ready = proc.stdout.readline().strip()
  if ready != "MMQ_KERNEL_WINDOW_READY":
    stdout, stderr = proc.communicate()
    return {"schema": SCHEMA, "window": "kernel_loop", "status": "blocked", "reason": "child did not announce kernel window",
            "ready_line": ready, "stdout": stdout[-4000:], "stderr": stderr[-4000:], "sample_count": 0, "samples": [],
            "system_snapshot_id": system_snapshot_id, "experiment_id": experiment_id, "candidate_id": candidate_id,
            "binary_sha256": binary_sha256}
  while proc.poll() is None:
    rows.append({"sample": len(rows), "monotonic_ns": time.monotonic_ns(),
                 "sensors": {name: {"path": path, **read_sensor(path)} for name, path in configured.items()}})
    if time.monotonic() - started > timeout: proc.kill(); break
    time.sleep(interval_s)
  stdout, stderr = proc.communicate()
  return {"schema": SCHEMA, "window": "kernel_loop", "status": "live" if proc.returncode == 0 and rows else "blocked",
          "writeback_mode": writeback_mode, "repetitions": repetitions, "system_snapshot_id": system_snapshot_id,
          "experiment_id": experiment_id, "candidate_id": candidate_id, "binary_sha256": binary_sha256,
          "sample_count": len(rows), "interval_s": interval_s, "samples": rows, "returncode": proc.returncode,
          "stdout": stdout[-4000:], "stderr": stderr[-4000:]}


def validate_telemetry(artifact: dict[str, Any]) -> None:
  if artifact.get("schema") != SCHEMA: raise ValueError(f"schema must be {SCHEMA}")
  if artifact.get("sample_count") != len(artifact.get("samples", [])): raise ValueError("sample_count mismatch")
  for idx, row in enumerate(artifact.get("samples", [])):
    for name, sensor in row.get("sensors", {}).items():
      if sensor.get("status") not in ("live", "zero_suspect", "unsupported", "blocked"):
        raise ValueError(f"samples[{idx}].sensors.{name}.status is invalid")


if __name__ == "__main__": print(json.dumps(collect_telemetry("manual"), indent=2, sort_keys=True))
