"""Diagnostic canary for the large preloaded-Q4 transfer lifecycle.

The strict target harness preloads roughly 50 MiB of Q4 bytes before its first
kernel.  This module tests that host-to-device transfer in a *fresh spawned
process*, then runs only the known-safe tiny add.  It is deliberately separate
from the generated MMQ program: a pass says that allocation/copy/readback and a
small queue submission survived, not that the target kernel is promotable.

The parent never constructs a GPU runtime.  ``run_large_preload_canary`` has a
single hard deadline, performs no retry, and fails closed on a child timeout,
kernel-log fault, or failed health callback.  Tests can inject the child runner,
log reader, and health callback, so the CPU suite never touches a device.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Callable

import numpy as np

from tinygrad.runtime.process_isolated import IsolatedResult, run_isolated

SCHEMA = "tinygrad.mmq_q4k_q8_1.preloaded_buffer_canary.v1"
DEVICE = "AMD"
TARGET_Q4_BYTES = 17_408 * 20 * 36 * 4
DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_TINY_SIZE = 256
_GPU_FAULT_MARKERS = (
  "sq_intr", "page fault", "sqc instruction", "mes failed", "gpu reset",
  "device wedged", "vram is lost", "ring gfx timeout",
)


def _deterministic_payload(nbytes: int) -> np.ndarray:
  _validate_nbytes(nbytes)
  # The modulo pattern is deterministic, nonconstant, and cheap to construct;
  # it also exercises the same uint32-backed byte capacity as Q4_K storage.
  return np.arange(nbytes, dtype=np.uint8)


def _validate_nbytes(nbytes: int) -> None:
  if not isinstance(nbytes, int) or nbytes <= 0:
    raise ValueError("nbytes must be a positive integer")


def parse_kernel_faults(text: str) -> list[str]:
  """Extract unique AMD health-fault rows from a kernel-log window."""
  rows: list[str] = []
  for raw in str(text).splitlines():
    row, lowered = raw.strip(), raw.lower()
    if row and any(marker in lowered for marker in _GPU_FAULT_MARKERS) and row not in rows:
      rows.append(row)
  return rows


def _timeline_snapshot(device: str) -> dict[str, int | None]:
  """Read timeline counters when the backend exposes them; never fail a probe."""
  try:
    from tinygrad.device import Device
    dev = Device[device]
    signal = getattr(dev, "timeline_signal", None)
    value = getattr(signal, "value", None)
    return {
      "timeline_value": int(getattr(dev, "timeline_value")) if hasattr(dev, "timeline_value") else None,
      "signal_value": int(value) if isinstance(value, (int, np.integer)) else None,
    }
  except BaseException:
    return {"timeline_value": None, "signal_value": None}


def _run_large_preload_worker(nbytes: int, tiny_size: int, device: str) -> dict[str, Any]:
  """Child-only worker: preload/read back bytes, then run and verify tiny add."""
  from tinygrad import Tensor, dtypes

  if not isinstance(tiny_size, int) or tiny_size <= 0:
    raise ValueError("tiny_size must be a positive integer")
  payload = _deterministic_payload(nbytes)
  payload_sha = hashlib.sha256(payload.tobytes()).hexdigest()
  started = time.perf_counter()
  checkpoints: dict[str, dict[str, int | None]] = {"start": _timeline_snapshot(device)}

  # Constructing a Tensor from the host array forces the large allocator/copyin
  # path.  No target/generated kernel is built or dispatched here.
  large = Tensor(payload, dtype=dtypes.uint8, device=device).realize()
  large_buf = getattr(getattr(large, "uop", None), "buffer", None)
  handle = large_buf.get_buf(device) if large_buf is not None else None
  checkpoints["after_preload"] = _timeline_snapshot(device)

  a = np.arange(tiny_size, dtype=np.float32)
  b = np.arange(tiny_size, dtype=np.float32)[::-1].copy()
  tiny = (Tensor(a, device=device) + Tensor(b, device=device)).numpy()
  tiny_reference = (a + b).astype(np.float32)
  tiny_passed = bool(tiny.shape == (tiny_size,) and np.allclose(tiny, tiny_reference, rtol=1e-3, atol=1e-3))
  checkpoints["after_tiny_add"] = _timeline_snapshot(device)

  roundtrip = large.numpy()
  roundtrip_passed = bool(roundtrip.shape == payload.shape and np.array_equal(roundtrip, payload))
  checkpoints["after_readback"] = _timeline_snapshot(device)
  return {
    "schema": f"{SCHEMA}.child", "passed": bool(tiny_passed and roundtrip_passed),
    "nbytes": nbytes, "payload_sha256": payload_sha,
    "roundtrip_sha256": hashlib.sha256(np.ascontiguousarray(roundtrip).tobytes()).hexdigest(),
    "roundtrip_passed": roundtrip_passed, "tiny_add_passed": tiny_passed,
    "tiny_size": tiny_size, "device": device, "timeline": checkpoints,
    "buffer_va": int(getattr(handle, "va_addr")) if getattr(handle, "va_addr", None) is not None else None,
    "buffer_nbytes": int(getattr(handle, "size")) if getattr(handle, "size", None) is not None else None,
    "elapsed_seconds": time.perf_counter() - started, "no_target_dispatch": True,
  }


def _default_fault_reader(since_timestamp: float) -> str:
  import subprocess
  proc = subprocess.run(
    ["journalctl", "-k", "--since", f"@{since_timestamp:.6f}", "--no-pager", "--output=short-monotonic"],
    text=True, capture_output=True, timeout=10.0, check=False,
  )
  if proc.returncode != 0:
    raise RuntimeError(f"journalctl failed ({proc.returncode}): {proc.stderr[-500:]}")
  return proc.stdout


def run_large_preload_canary(*, nbytes: int = TARGET_Q4_BYTES, tiny_size: int = DEFAULT_TINY_SIZE,
                             device: str = DEVICE, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
                             runner: Callable[..., IsolatedResult] = run_isolated,
                             fault_reader: Callable[[float], str] = _default_fault_reader,
                             health_probe: Callable[[], bool] | None = None) -> dict[str, Any]:
  """Run one diagnostic child and return a fail-closed lifecycle record.

  ``health_probe`` is optional because the child already verifies a tiny add;
  when supplied it is called exactly once in the parent (typically a fresh
  process-isolated health canary).  No retry or target-kernel dispatch occurs.
  """
  if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
    raise ValueError("timeout_seconds must be positive")
  # Validate before spawning so malformed requests never touch a device.
  _validate_nbytes(nbytes)
  if not isinstance(tiny_size, int) or tiny_size <= 0: raise ValueError("tiny_size must be positive")
  started = time.time()
  base: dict[str, Any] = {
    "schema": SCHEMA, "diagnostic_only": True, "promotion_eligible": False,
    "production_dispatch_changed": False, "no_target_dispatch": True,
    "nbytes": nbytes, "tiny_size": tiny_size, "device": device,
    "kernel_faults": [], "health_after": None, "child": None,
  }
  isolated = runner(_run_large_preload_worker, args=(nbytes, tiny_size, device),
                    timeout_seconds=float(timeout_seconds), start_method="spawn")
  base["child_status"] = isolated.status
  base["child_error"] = isolated.error
  if isolated.status == "passed" and isinstance(isolated.result, dict): base["child"] = isolated.result
  try:
    base["kernel_faults"] = parse_kernel_faults(fault_reader(started))
  except BaseException as exc:
    base["exact_blocker"] = f"kernel-log scan failed: {type(exc).__name__}: {exc}"
    return {**base, "status": "BLOCKED", "passed": False}
  if base["kernel_faults"]:
    base["exact_blocker"] = "AMD kernel fault/reset marker observed"
    return {**base, "status": "BLOCKED", "passed": False}
  if isolated.status != "passed" or not isinstance(isolated.result, dict):
    base["exact_blocker"] = isolated.error or "isolated preload child returned no result"
    return {**base, "status": "BLOCKED", "passed": False}
  if not isolated.result.get("passed"):
    base["exact_blocker"] = "preload roundtrip or tiny health check failed"
    return {**base, "status": "BLOCKED", "passed": False}
  if health_probe is not None:
    try: base["health_after"] = bool(health_probe())
    except BaseException as exc:
      base["exact_blocker"] = f"health probe failed: {type(exc).__name__}: {exc}"
      return {**base, "status": "BLOCKED", "passed": False}
    if not base["health_after"]:
      base["exact_blocker"] = "post-run health probe reported device unhealthy"
      return {**base, "status": "BLOCKED", "passed": False}
  base["exact_blocker"] = None
  return {**base, "status": "PASS", "passed": True}


def main() -> int:
  import argparse
  import json
  from pathlib import Path
  from extra.qk.mmq_target_epoch_orchestrator import spawned_tiny_health_probe
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--nbytes", type=int, default=TARGET_Q4_BYTES)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  result = run_large_preload_canary(nbytes=args.nbytes, health_probe=spawned_tiny_health_probe)
  encoded = json.dumps(result, indent=2, sort_keys=True)
  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n")
  print(encoded)
  return 0 if result["passed"] else 1


if __name__ == "__main__": raise SystemExit(main())


__all__ = ["SCHEMA", "TARGET_Q4_BYTES", "parse_kernel_faults", "run_large_preload_canary"]
