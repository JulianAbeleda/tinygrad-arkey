"""Fresh-process one-shot exact-target epoch canary.

The parent reuses the established CPU-only target PROGRAM compiler.  A fresh
child regenerates the complete epoch-major Q4 preload and full Q8 buffers,
allocates a zero output, builds the exact views, and dispatches exactly one
selected K=256 epoch.  It compares that partial against the independent DS4
oracle, then the parent performs log and independent-health checks.  This is
diagnostic evidence only and never changes the production route.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tinygrad.runtime.process_isolated import IsolatedResult, run_isolated

SCHEMA = "tinygrad.mmq_q4k_q8_1.single_epoch_canary.v1"
DEVICE = "AMD"
TARGET_SHAPE = (512, 17_408, 5_120)
TOTAL_EPOCHS = TARGET_SHAPE[2] // 256
DEFAULT_EPOCH = 0
DEFAULT_CHILD_TIMEOUT_SECONDS = 120.0
DEFAULT_RUNTIME_TIMEOUT_MS = 30_000
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
    return {"timeline_value": int(getattr(dev, "timeline_value")) if hasattr(dev, "timeline_value") else None,
            "signal_value": int(value) if isinstance(value, (int, np.integer)) else None}
  except BaseException:
    return {"timeline_value": None, "signal_value": None}


def _hash_array(value: np.ndarray) -> str:
  return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _make_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Create corrected epoch-major Q4 plus full deterministic Q8."""
  from extra.qk.mmq_llama_five_buffer_gpu_harness import _pack_q4_epochs_contiguous, _random_q4_words
  from extra.qk.mmq_q4k_q8_reference import q8_1_mmq_ds4_quantize_reference

  m, n, k = TARGET_SHAPE
  words = _random_q4_words(n, k, 20260721)
  q4_blocks = words.view(np.uint8).reshape(n, TOTAL_EPOCHS, 144)
  q4_packed = _pack_q4_epochs_contiguous(q4_blocks)
  source = np.random.default_rng(20260722).standard_normal((m, k), dtype=np.float32)
  values, scales, sums = q8_1_mmq_ds4_quantize_reference(source)
  return q4_packed, np.ascontiguousarray(values.reshape(-1)), np.ascontiguousarray(scales.reshape(-1)), \
         np.ascontiguousarray(sums.reshape(-1))


def _buffer_record(name: str, tensor: Any, device: str) -> dict[str, Any]:
  buf = getattr(getattr(tensor, "uop", None), "buffer", None)
  handle = buf.get_buf(device) if buf is not None else None
  va = getattr(handle, "va_addr", None)
  numel = getattr(tensor, "numel", None)
  elements = int(numel() if callable(numel) else getattr(tensor, "size", 0))
  return {"name": name, "elements": elements,
          "nbytes": int(getattr(handle, "size", 0)) if handle is not None else None,
          "va_addr": int(va) if va is not None else None}


def _run_epoch_worker(program_path: str, epoch_start: int, epoch_count: int = 1, device: str = DEVICE,
                      runtime_timeout_ms: int = DEFAULT_RUNTIME_TIMEOUT_MS,
                      fresh_output_each_launch: bool = False,
                      epoch_sequence: tuple[int, ...] | None = None) -> dict[str, Any]:
  """Child-only persistent-preload target prefix and final oracle comparison."""
  from tinygrad import Tensor, dtypes
  from tinygrad.engine.realize import get_runtime
  from tinygrad.uop.ops import Ops
  from extra.qk.mmq_q4k_q8_reference import (
    Q81MMQDS4Activation, Q81MMQDS4ActivationSpec, Q4KQ81MMQTileSpec,
    Q8_1_MMQ_DS4_LAYOUT, q4k_q8_1_mmq_ds4_tile_reference,
  )

  if epoch_sequence is None:
    if not isinstance(epoch_count, int) or epoch_count <= 0: raise ValueError("epoch_count must be positive")
    if not 0 <= epoch_start < TOTAL_EPOCHS or epoch_start + epoch_count > TOTAL_EPOCHS:
      raise ValueError(f"epoch range must fit [0,{TOTAL_EPOCHS - 1}]")
    epochs = tuple(range(epoch_start, epoch_start + epoch_count))
  else:
    if not epoch_sequence or any(not isinstance(e, int) or not 0 <= e < TOTAL_EPOCHS for e in epoch_sequence):
      raise ValueError(f"epoch_sequence must be nonempty and use epochs in [0,{TOTAL_EPOCHS - 1}]")
    epochs = tuple(epoch_sequence)
    epoch_start, epoch_count = epochs[0], len(epochs)
  if not isinstance(runtime_timeout_ms, int) or runtime_timeout_ms <= 0: raise ValueError("runtime timeout must be positive")
  started = time.perf_counter()
  with Path(program_path).open("rb") as handle: program = pickle.load(handle)
  if getattr(program, "op", None) is not Ops.PROGRAM: raise RuntimeError("serialized artifact is not a PROGRAM")
  if tuple(getattr(program.arg, "globals", ())) != tuple(range(5)):
    raise RuntimeError(f"target PROGRAM ABI changed: globals={getattr(program.arg, 'globals', None)}")
  q4_packed, values, scales, sums = _make_inputs()
  m, n, _ = TARGET_SHAPE
  checkpoints: dict[str, dict[str, int | None]] = {"start": _timeline_snapshot(device)}
  # Match strict target harness partial allocation: output is not host-zeroed
  # (the full-N target dispatch writes every element), avoiding an extra SDMA
  # copy that would obscure the preload/runtime lifecycle. Keep the first
  # output's allocation order identical in both modes, then retain it and add
  # fresh held outputs only for subsequent launches in fresh mode.
  output = Tensor.empty(m * n, dtype=dtypes.float32, device=device).realize()
  held_outputs: list[Any] = [output] if fresh_output_each_launch else []
  q4_tensor = Tensor(q4_packed, dtype=dtypes.uint32, device=device).realize()
  values_tensor = Tensor(values, dtype=dtypes.int8, device=device).realize()
  scales_tensor = Tensor(scales, dtype=dtypes.float32, device=device).realize()
  sums_tensor = Tensor(sums, dtype=dtypes.float32, device=device).realize()
  checkpoints["after_preload"] = _timeline_snapshot(device)

  runtime = get_runtime(device, program)
  dispatches: list[dict[str, Any]] = []
  gpu_ms_total = 0.0
  completed_epochs: list[int] = []
  for ordinal, epoch in enumerate(epochs):
    if fresh_output_each_launch and ordinal:
      output = Tensor.empty(m * n, dtype=dtypes.float32, device=device).realize()
      held_outputs.append(output)
    q4_view = q4_tensor.uop.buffer.view(n * 36, dtypes.uint32, epoch * n * 36 * dtypes.uint32.itemsize)
    values_view = values_tensor.uop.buffer.view(2 * m * 128, dtypes.int8,
                                                 epoch * 2 * m * 128 * dtypes.int8.itemsize)
    scales_view = scales_tensor.uop.buffer.view(2 * m * 4, dtypes.float32,
                                                 epoch * 2 * m * 4 * dtypes.float32.itemsize)
    sums_view = sums_tensor.uop.buffer.view(2 * m * 4, dtypes.float32,
                                             epoch * 2 * m * 4 * dtypes.float32.itemsize)
    args = tuple((output.uop.buffer, q4_view, values_view, scales_view, sums_view)[slot].get_buf(device)
                 for slot in program.arg.globals)
    gpu_seconds = runtime(*args, global_size=(n // 128, m // 128, 1), local_size=program.arg.local_size,
                          vals=program.arg.vals({}), wait=True, timeout=runtime_timeout_ms)
    gpu_ms_total += float(gpu_seconds) * 1000.0 if gpu_seconds is not None else 0.0
    completed_epochs.append(epoch)
    dispatches.append({"epoch": epoch, "timeline": _timeline_snapshot(device)})
  got = output.numpy().reshape(m, n)

  # Recreate the independent oracle from the same deterministic arrays, with
  # FP16-rounded metadata exactly as the target harness uses.
  epoch = epochs[-1]
  q4_bytes = q4_packed.view(np.uint8).reshape(TOTAL_EPOCHS, n, 144)[epoch].reshape(-1)
  ep_values = values.reshape(TOTAL_EPOCHS * 2, m, 128)[epoch * 2:(epoch + 1) * 2]
  ep_scales = scales.reshape(TOTAL_EPOCHS * 2, m, 4)[epoch * 2:(epoch + 1) * 2]
  ep_sums = sums.reshape(TOTAL_EPOCHS * 2, m, 4)[epoch * 2:(epoch + 1) * 2]
  ds4 = Q81MMQDS4Activation(ep_values, ep_scales.astype(np.float16).astype(np.float32),
                            ep_sums.astype(np.float16).astype(np.float32),
                            Q81MMQDS4ActivationSpec(m=m, k=256, m_tile=m))
  spec = Q4KQ81MMQTileSpec(role="single_epoch_canary", m=m, n=n, k=256, m_tile=m, n_tile=n,
                           activation_layout=Q8_1_MMQ_DS4_LAYOUT)
  reference = q4k_q8_1_mmq_ds4_tile_reference(q4_bytes.view(np.uint8), ds4, spec)
  from extra.qk.mmq_llama_five_buffer_gpu_harness import _numeric_comparison
  comparison = _numeric_comparison(got, reference)
  return {"schema": f"{SCHEMA}.child", "status": "PASS" if comparison["status"] == "pass" else "BLOCKED",
          "passed": comparison["status"] == "pass", "epoch": epoch,
          "epoch_start": epoch_start, "epoch_count": epoch_count, "epoch_sequence": list(epochs),
          "completed_epochs": completed_epochs,
          "shape": [m, n, 256], "comparison": comparison, "gpu_ms": gpu_ms_total,
          "target_dispatches": epoch_count, "dispatches": dispatches,
          "timeline": checkpoints, "output_mode": "fresh_held" if fresh_output_each_launch else "persistent",
          "output_count": len(held_outputs) if fresh_output_each_launch else 1,
          "output_buffers": [_buffer_record(f"output_{i}", tensor, device)
                             for i, tensor in enumerate(held_outputs if fresh_output_each_launch else [output])],
          "buffers": [_buffer_record("q4_preloaded", q4_tensor, device),
            _buffer_record("q8_values", values_tensor, device),
            _buffer_record("q8_scales", scales_tensor, device), _buffer_record("q8_sums", sums_tensor, device)],
          "input_hashes": {"q4_epoch_major": _hash_array(q4_packed), "values": _hash_array(values),
                           "scales": _hash_array(scales), "sums": _hash_array(sums)},
          "output_initialization": "empty_full_target_partial",
          "no_fallback": True, "elapsed_seconds": time.perf_counter() - started}


def _default_fault_reader(since_timestamp: float) -> str:
  import subprocess
  proc = subprocess.run(["journalctl", "-k", "--since", f"@{since_timestamp:.6f}", "--no-pager", "--output=short-monotonic"],
                        text=True, capture_output=True, timeout=10.0, check=False)
  if proc.returncode != 0: raise RuntimeError(f"journalctl failed ({proc.returncode}): {proc.stderr[-500:]}")
  return proc.stdout


def _default_health_probe() -> bool:
  from extra.qk.mmq_target_epoch_orchestrator import spawned_tiny_health_probe
  return spawned_tiny_health_probe()


def run_single_epoch_canary(*, epoch: int = DEFAULT_EPOCH, epoch_start: int | None = None,
                            epoch_count: int = 1, output_path: str | Path | None = None,
                            fresh_output_each_launch: bool = False,
                            epoch_sequence: tuple[int, ...] | list[int] | None = None,
                            compile_fn: Callable[[str | Path], tuple[str, dict[str, Any]]] | None = None,
                            device: str = DEVICE, timeout_seconds: float = DEFAULT_CHILD_TIMEOUT_SECONDS,
                            runtime_timeout_ms: int = DEFAULT_RUNTIME_TIMEOUT_MS,
                            runner: Callable[..., IsolatedResult] = run_isolated,
                            fault_reader: Callable[[float], str] = _default_fault_reader,
                            health_probe: Callable[[], bool] | None = _default_health_probe) -> dict[str, Any]:
  if epoch_sequence is not None:
    if not epoch_sequence or any(not isinstance(e, int) or not 0 <= e < TOTAL_EPOCHS for e in epoch_sequence):
      raise ValueError(f"epoch_sequence must be nonempty and use epochs in [0,{TOTAL_EPOCHS - 1}]")
    sequence = tuple(epoch_sequence)
    epoch_start, epoch_count = sequence[0], len(sequence)
  else:
    if epoch_start is None: epoch_start = epoch
    if not isinstance(epoch_count, int) or epoch_count <= 0: raise ValueError("epoch_count must be positive")
    if not 0 <= epoch_start < TOTAL_EPOCHS or epoch_start + epoch_count > TOTAL_EPOCHS:
      raise ValueError(f"epoch range must fit [0,{TOTAL_EPOCHS - 1}]")
    sequence = None
  if timeout_seconds <= 0 or runtime_timeout_ms <= 0: raise ValueError("timeouts must be positive")
  if compile_fn is None:
    from extra.qk.mmq_target_epoch_orchestrator import compile_target_program_artifact
    compile_fn = compile_target_program_artifact
  started = time.time()
  base: dict[str, Any] = {"schema": SCHEMA, "diagnostic_only": True, "promotion_eligible": False,
    "production_dispatch_changed": False, "no_fallback": True, "target_dispatches": epoch_count,
    "epoch": (sequence[-1] if sequence is not None else epoch_start + epoch_count - 1),
    "epoch_start": epoch_start, "epoch_count": epoch_count,
    "epoch_sequence": list(sequence) if sequence is not None else list(range(epoch_start, epoch_start + epoch_count)),
    "fresh_output_each_launch": bool(fresh_output_each_launch),
    "output_mode": "fresh_held" if fresh_output_each_launch else "persistent",
    "kernel_faults": [], "compile": None, "child": None, "health_after": None}
  with tempfile.TemporaryDirectory(prefix="tinygrad-mmq-single-epoch-") as temp_dir:
    try: artifact_path, evidence = compile_fn(temp_dir)
    except BaseException as exc:
      result = {**base, "status": "BLOCKED", "passed": False,
                "exact_blocker": f"parent compile failed: {type(exc).__name__}: {exc}"}
      return _write_result(result, output_path)
    base["compile"] = evidence
    isolated = runner(_run_epoch_worker, args=(artifact_path, epoch_start, epoch_count, device, runtime_timeout_ms,
                                               bool(fresh_output_each_launch), sequence),
                      timeout_seconds=float(timeout_seconds), start_method="spawn")
    base["child_status"], base["child_error"] = isolated.status, isolated.error
    if isolated.status == "passed" and isinstance(isolated.result, dict): base["child"] = isolated.result
    try: base["kernel_faults"] = parse_kernel_faults(fault_reader(started))
    except BaseException as exc:
      return _write_result({**base, "status": "BLOCKED", "passed": False,
                            "exact_blocker": f"kernel-log scan failed: {type(exc).__name__}: {exc}"}, output_path)
    if base["kernel_faults"]:
      return _write_result({**base, "status": "BLOCKED", "passed": False,
                            "exact_blocker": "AMD kernel fault/reset marker observed"}, output_path)
    if isolated.status != "passed" or not isinstance(isolated.result, dict):
      return _write_result({**base, "status": "BLOCKED", "passed": False,
                            "exact_blocker": isolated.error or "epoch child returned no result"}, output_path)
    if not isolated.result.get("passed"):
      return _write_result({**base, "status": "BLOCKED", "passed": False,
                            "exact_blocker": "single epoch numerical comparison failed"}, output_path)
    if health_probe is not None:
      try: base["health_after"] = bool(health_probe())
      except BaseException as exc:
        return _write_result({**base, "status": "BLOCKED", "passed": False,
                              "exact_blocker": f"health probe failed: {type(exc).__name__}: {exc}"}, output_path)
      if not base["health_after"]:
        return _write_result({**base, "status": "BLOCKED", "passed": False,
                              "exact_blocker": "post-run health probe reported device unhealthy"}, output_path)
    return _write_result({**base, "status": "PASS", "passed": True, "exact_blocker": None}, output_path)


def _write_result(result: dict[str, Any], output_path: str | Path | None) -> dict[str, Any]:
  if output_path is not None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  return result


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="diagnostic one-shot exact Qwen3 target epoch canary")
  parser.add_argument("--epoch", type=int, default=DEFAULT_EPOCH,
                      help="legacy alias for --epoch-start when no prefix is requested")
  parser.add_argument("--epoch-start", type=int, default=None)
  parser.add_argument("--epoch-count", type=int, default=1)
  parser.add_argument("--epoch-sequence", type=str, default=None,
                      help="comma-separated epoch indices; overrides --epoch-start/--epoch-count")
  parser.add_argument("--fresh-output-each-launch", action="store_true",
                      help="allocate and retain a distinct output for each target launch")
  parser.add_argument("--output", type=Path)
  args = parser.parse_args(argv)
  try:
    sequence = None if args.epoch_sequence is None else tuple(int(part.strip()) for part in args.epoch_sequence.split(",") if part.strip())
  except ValueError as exc:
    parser.error(f"invalid --epoch-sequence: {exc}")
  result = run_single_epoch_canary(epoch=args.epoch, epoch_start=args.epoch_start,
                                   epoch_count=args.epoch_count, output_path=args.output,
                                   fresh_output_each_launch=args.fresh_output_each_launch,
                                   epoch_sequence=sequence)
  print(json.dumps(result, sort_keys=True))
  return 0 if result.get("passed") else 1


if __name__ == "__main__": raise SystemExit(main())


__all__ = ["SCHEMA", "TARGET_SHAPE", "TOTAL_EPOCHS", "parse_kernel_faults", "run_single_epoch_canary"]
