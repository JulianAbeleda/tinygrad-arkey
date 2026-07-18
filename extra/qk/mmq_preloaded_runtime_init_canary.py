"""No-dispatch discriminator for exact target-buffer preload + AMD runtime init.

This is intentionally narrower than the target-role harness.  The parent
compiles and pickles the exact target K=256 PROGRAM without constructing an AMD
runtime.  A fresh spawned child then copies the strict 20-epoch Q4/Q8 inputs
and zero accumulator, constructs ``get_runtime("AMD", program)`` (including
code upload), but never calls that target runtime.  Only the independent tiny
add/readback health operation is dispatched.

A pass therefore isolates allocation/copy and program-upload lifecycle from
target-kernel execution.  The result is permanently diagnostic-only.
"""
from __future__ import annotations

import hashlib
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tinygrad.runtime.process_isolated import IsolatedResult, run_isolated

SCHEMA = "tinygrad.mmq_q4k_q8_1.preloaded_runtime_init_canary.v1"
DEVICE = "AMD"
TARGET_SHAPE = (512, 17_408, 5_120)
TOTAL_EPOCHS = TARGET_SHAPE[2] // 256
DEFAULT_TIMEOUT_SECONDS = 240.0
_GPU_FAULT_MARKERS = (
  "sq_intr", "page fault", "sqc instruction", "mes failed", "gpu reset",
  "device wedged", "vram is lost", "ring gfx timeout",
)


def parse_kernel_faults(text: str) -> list[str]:
  rows: list[str] = []
  for raw in str(text).splitlines():
    row, lowered = raw.strip(), raw.lower()
    if row and any(marker in lowered for marker in _GPU_FAULT_MARKERS) and row not in rows:
      rows.append(row)
  return rows


def _timeline_snapshot(device: str) -> dict[str, int | None]:
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


def _hash_array(value: np.ndarray) -> str:
  return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _target_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Regenerate the exact full preloaded arrays used by the target harness."""
  from extra.qk.mmq_llama_five_buffer_gpu_harness import _pack_q4_epochs_contiguous, _random_q4_words
  from extra.qk.mmq_llama_five_buffer_gpu_harness import TARGET_ROLE_PROBE_SHAPE
  from extra.qk.mmq_q4k_q8_reference import q8_1_mmq_ds4_quantize_reference

  m, n, k = TARGET_ROLE_PROBE_SHAPE
  words = _random_q4_words(n, k, 20260721)
  source = np.random.default_rng(20260722).standard_normal((m, k), dtype=np.float32)
  values, scales, sums = q8_1_mmq_ds4_quantize_reference(source)
  q4_blocks = words.view(np.uint8).reshape(n, TOTAL_EPOCHS, 144)
  return _pack_q4_epochs_contiguous(q4_blocks), values.reshape(-1), scales.reshape(-1), sums.reshape(-1)


def _buffer_record(name: str, tensor: Any, device: str) -> dict[str, Any]:
  buf = getattr(getattr(tensor, "uop", None), "buffer", None)
  handle = buf.get_buf(device) if buf is not None else None
  va = getattr(handle, "va_addr", None)
  numel = getattr(tensor, "numel", None)
  elements = int(numel() if callable(numel) else getattr(tensor, "size", 0))
  return {"name": name, "elements": elements,
          "nbytes": int(getattr(handle, "size", 0)) if handle is not None else None,
          "va_addr": int(va) if va is not None else None}


def _run_runtime_init_worker(program_path: str, device: str = DEVICE) -> dict[str, Any]:
  """Child-only exact preload + target runtime construction, with no target call."""
  from tinygrad import Tensor, dtypes
  from tinygrad.engine.realize import get_runtime
  from tinygrad.uop.ops import Ops

  started = time.perf_counter()
  with Path(program_path).open("rb") as handle: program = pickle.load(handle)
  if getattr(program, "op", None) is not Ops.PROGRAM:
    raise RuntimeError("serialized artifact is not a PROGRAM")
  if tuple(getattr(program.arg, "globals", ())) != tuple(range(5)):
    raise RuntimeError(f"target PROGRAM ABI changed: globals={getattr(program.arg, 'globals', None)}")
  q4, values, scales, sums = _target_inputs()
  m, n, _ = TARGET_SHAPE
  checkpoints: dict[str, dict[str, int | None]] = {"start": _timeline_snapshot(device)}

  # Tensor realization forces the same host-to-device copies and full shapes as
  # strict preloaded target mode.  No target kernel is invoked by these copies.
  q4_tensor = Tensor(q4, dtype=dtypes.uint32, device=device).realize()
  values_tensor = Tensor(values, dtype=dtypes.int8, device=device).realize()
  scales_tensor = Tensor(scales, dtype=dtypes.float32, device=device).realize()
  sums_tensor = Tensor(sums, dtype=dtypes.float32, device=device).realize()
  # Match strict mode's device-side zero accumulator instead of introducing a
  # large host-zero SDMA copy that the target adapter never performs.
  accum_tensor = Tensor.zeros(m*n, dtype=dtypes.float32, device=device).realize()
  checkpoints["after_preload"] = _timeline_snapshot(device)

  # This constructs/uploads the exact AMD code object.  Do not call `runtime`:
  # the only dispatch in this child is the independent tiny health operation.
  target_runtime = get_runtime(device, program)
  checkpoints["after_target_runtime_init"] = _timeline_snapshot(device)

  tiny_size = 256
  a = np.arange(tiny_size, dtype=np.float32)
  b = np.arange(tiny_size, dtype=np.float32)[::-1].copy()
  tiny = (Tensor(a, device=device) + Tensor(b, device=device)).numpy()
  tiny_passed = bool(tiny.shape == (tiny_size,) and np.allclose(tiny, a + b, rtol=1e-3, atol=1e-3))
  checkpoints["after_tiny_health"] = _timeline_snapshot(device)
  return {
    "schema": f"{SCHEMA}.child", "passed": tiny_passed,
    "target_runtime_constructed": target_runtime is not None,
    "target_runtime_called": False, "tiny_add_passed": tiny_passed,
    "timeline": checkpoints, "buffers": [
      _buffer_record("q4_preloaded", q4_tensor, device),
      _buffer_record("q8_values_preloaded", values_tensor, device),
      _buffer_record("q8_scales_preloaded", scales_tensor, device),
      _buffer_record("q8_sums_preloaded", sums_tensor, device),
      _buffer_record("accumulator_zero", accum_tensor, device),
    ],
    "input_hashes": {"q4": _hash_array(q4), "values": _hash_array(values),
                      "scales": _hash_array(scales), "sums": _hash_array(sums)},
    "shape": list(TARGET_SHAPE), "epochs": TOTAL_EPOCHS,
    "elapsed_seconds": time.perf_counter() - started, "no_target_dispatch": True,
  }


def _default_fault_reader(since_timestamp: float) -> str:
  import subprocess
  proc = subprocess.run(
    ["journalctl", "-k", "--since", f"@{since_timestamp:.6f}", "--no-pager", "--output=short-monotonic"],
    text=True, capture_output=True, timeout=10.0, check=False,
  )
  if proc.returncode != 0: raise RuntimeError(f"journalctl failed ({proc.returncode}): {proc.stderr[-500:]}")
  return proc.stdout


def run_runtime_init_canary(*, compile_fn: Callable[[str | Path], tuple[str, dict[str, Any]]] | None = None,
                            device: str = DEVICE, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
                            runner: Callable[..., IsolatedResult] = run_isolated,
                            fault_reader: Callable[[float], str] = _default_fault_reader,
                            health_probe: Callable[[], bool] | None = None) -> dict[str, Any]:
  """Compile in parent, run one fresh no-target-dispatch child, fail closed."""
  if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
    raise ValueError("timeout_seconds must be positive")
  if compile_fn is None:
    from extra.qk.mmq_target_epoch_orchestrator import compile_target_program_artifact
    compile_fn = compile_target_program_artifact
  started = time.time()
  base: dict[str, Any] = {"schema": SCHEMA, "diagnostic_only": True,
    "promotion_eligible": False, "production_dispatch_changed": False,
    "no_target_dispatch": True, "kernel_faults": [], "compile": None,
    "child": None, "health_after": None}
  with tempfile.TemporaryDirectory(prefix="tinygrad-mmq-runtime-init-") as temp_dir:
    try: artifact_path, evidence = compile_fn(temp_dir)
    except BaseException as exc:
      return {**base, "status": "BLOCKED", "passed": False,
              "exact_blocker": f"parent compile failed: {type(exc).__name__}: {exc}"}
    base["compile"] = evidence
    isolated = runner(_run_runtime_init_worker, args=(artifact_path, device),
                      timeout_seconds=float(timeout_seconds), start_method="spawn")
    base["child_status"], base["child_error"] = isolated.status, isolated.error
    if isolated.status == "passed" and isinstance(isolated.result, dict): base["child"] = isolated.result
    try: base["kernel_faults"] = parse_kernel_faults(fault_reader(started))
    except BaseException as exc:
      return {**base, "status": "BLOCKED", "passed": False,
              "exact_blocker": f"kernel-log scan failed: {type(exc).__name__}: {exc}"}
    if base["kernel_faults"]:
      return {**base, "status": "BLOCKED", "passed": False,
              "exact_blocker": "AMD kernel fault/reset marker observed"}
    if isolated.status != "passed" or not isinstance(isolated.result, dict):
      return {**base, "status": "BLOCKED", "passed": False,
              "exact_blocker": isolated.error or "runtime-init child returned no result"}
    if not isolated.result.get("passed") or not isolated.result.get("target_runtime_constructed"):
      return {**base, "status": "BLOCKED", "passed": False,
              "exact_blocker": "runtime initialization or tiny health failed"}
    if health_probe is not None:
      try: base["health_after"] = bool(health_probe())
      except BaseException as exc:
        return {**base, "status": "BLOCKED", "passed": False,
                "exact_blocker": f"health probe failed: {type(exc).__name__}: {exc}"}
      if not base["health_after"]:
        return {**base, "status": "BLOCKED", "passed": False,
                "exact_blocker": "post-run health probe reported device unhealthy"}
    return {**base, "status": "PASS", "passed": True, "exact_blocker": None}


def main() -> int:
  import argparse
  import json
  from extra.qk.mmq_target_epoch_orchestrator import spawned_tiny_health_probe
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  result = run_runtime_init_canary(health_probe=spawned_tiny_health_probe)
  encoded = json.dumps(result, indent=2, sort_keys=True)
  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n")
  print(encoded)
  return 0 if result["passed"] else 1


if __name__ == "__main__": raise SystemExit(main())


__all__ = ["SCHEMA", "TARGET_SHAPE", "parse_kernel_faults", "run_runtime_init_canary"]
