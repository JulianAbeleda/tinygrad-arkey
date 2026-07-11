#!/usr/bin/env python3
"""Best-effort clock, power, temperature, and VRAM telemetry with explicit failures."""
from __future__ import annotations

import errno
import json
import math
from pathlib import Path
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


def validate_telemetry(artifact: dict[str, Any]) -> None:
  if artifact.get("schema") != SCHEMA: raise ValueError(f"schema must be {SCHEMA}")
  if artifact.get("sample_count") != len(artifact.get("samples", [])): raise ValueError("sample_count mismatch")
  for idx, row in enumerate(artifact.get("samples", [])):
    for name, sensor in row.get("sensors", {}).items():
      if sensor.get("status") not in ("live", "zero_suspect", "unsupported", "blocked"):
        raise ValueError(f"samples[{idx}].sensors.{name}.status is invalid")


if __name__ == "__main__": print(json.dumps(collect_telemetry("manual"), indent=2, sort_keys=True))
