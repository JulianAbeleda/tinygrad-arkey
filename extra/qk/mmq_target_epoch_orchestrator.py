"""Fail-closed process-per-epoch diagnostic for the Qwen3-14B prefill target.

The generated K=256 PROGRAM is compiled once without constructing an AMD
runtime, serialized, and loaded by one fresh spawned process per K epoch.  A
partial is admitted to the host aggregate only after:

* the worker reports an exact numerical comparison against the independent
  FP16-rounded DS4 oracle;
* the kernel log window contains no AMD queue/fault/reset marker; and
* a known-safe tiny add passes in a second fresh spawned process.

This deliberately is not promotion evidence.  Process-per-epoch execution
isolates queue lifetime and diagnoses all 20 K epochs safely; the strict R6
same-process health gate remains separate.

The v1 top-level record is extended additively with a deterministic fixture
identity and an ``epoch_health`` attestation.  The nested attestation and
fixture schemas are versioned independently so existing v1 readers can ignore
the extra audit fields while R6 can require them explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import tempfile
import time
from typing import Any, Callable, Iterable

import numpy as np

from extra.qk.mmq_llama_five_buffer_gpu_harness import (
  ATOL, FULL_GRID_BACKEND_ID, RTOL, TARGET_ROLE_PROBE_SHAPE, _artifact_evidence,
  _numeric_comparison, _random_q4_words,
)


SCHEMA = "tinygrad.mmq_q4k_q8_1_target_epoch_orchestrator.v1"
ATTESTATION_SCHEMA = "tinygrad.mmq_q4k_q8_1_target_epoch_attestation.v1"
FIXTURE_SCHEMA = "tinygrad.mmq_q4k_q8_1_target_fixture.v1"
TOTAL_EPOCHS = TARGET_ROLE_PROBE_SHAPE[2] // 256
DEFAULT_WORKER_TIMEOUT_SECONDS = 120.0
DEFAULT_RUNTIME_TIMEOUT_MS = 30_000
_GPU_FAULT_MARKERS = (
  "sq_intr",
  "[gfxhub] page fault",
  "sqc instruction",
  "mes failed",
  "gpu reset",
  "device wedged",
  "vram is lost",
  "ring gfx timeout",
)


def parse_kernel_faults(text: str) -> list[str]:
  """Return unique kernel-log rows that indicate an AMD queue/GPU health fault."""
  rows: list[str] = []
  for raw in str(text).splitlines():
    row, lowered = raw.strip(), raw.lower()
    if row and any(marker in lowered for marker in _GPU_FAULT_MARKERS) and row not in rows: rows.append(row)
  return rows


def read_kernel_log_since(since_timestamp: float) -> str:
  """Read only kernel messages emitted after the epoch's wall-clock start."""
  if not isinstance(since_timestamp, (int, float)) or since_timestamp <= 0:
    raise ValueError("since_timestamp must be a positive Unix timestamp")
  proc = subprocess.run(
    ["journalctl", "-k", "--since", f"@{since_timestamp:.6f}", "--no-pager", "--output=short-monotonic"],
    text=True, capture_output=True, timeout=10.0, check=False,
  )
  if proc.returncode != 0:
    raise RuntimeError(f"journalctl failed ({proc.returncode}): {proc.stderr[-500:]}")
  return proc.stdout


def compile_target_program_artifact(temp_dir: str | Path) -> tuple[str, dict[str, Any]]:
  """Compile the target K=256 PROGRAM CPU-side and serialize it for spawn workers."""
  from tinygrad.uop.ops import Ops
  from extra.qk.amdgpu_metadata import parse_amdgpu_metadata
  from extra.qk.mmq_llama_five_buffer_full_kernel import (
    build_llama_five_buffer_full_kernel, compile_llama_five_buffer_full_kernel,
  )
  from extra.qk.mmq_llama_runtime_contract import LLAMA_SOURCE_COMMIT

  m, n, _ = TARGET_ROLE_PROBE_SHAPE
  compiled = compile_llama_five_buffer_full_kernel(build_llama_five_buffer_full_kernel(m, n, 256))
  if not compiled.emitted or compiled.program is None:
    raise RuntimeError(compiled.blocker or "target K=256 program did not emit")
  program = compiled.program
  if program.op is not Ops.PROGRAM: raise RuntimeError(f"compile result is not PROGRAM: {program.op}")
  programs = [u for u in program.toposort() if u.op is Ops.PROGRAM]
  if programs != [program]: raise RuntimeError(f"expected one closed PROGRAM, found {len(programs)}")
  if tuple(program.arg.globals) != tuple(range(5)):
    raise RuntimeError(f"target PROGRAM ABI changed: globals={program.arg.globals}")
  binary, source, evidence = _artifact_evidence(program, parse_amdgpu_metadata)
  if not isinstance(binary, bytes) or not isinstance(source, str):
    raise RuntimeError("compiled PROGRAM lacks distinct source/binary identity")

  path = Path(temp_dir) / "target_k256_program.pkl"
  with path.open("wb") as handle:
    pickle.dump(program, handle, protocol=pickle.HIGHEST_PROTOCOL)
  serialized_sha = hashlib.sha256(path.read_bytes()).hexdigest()
  return str(path), {
    **evidence,
    "backend_id": FULL_GRID_BACKEND_ID,
    "serialized_program_sha256": serialized_sha,
    "serialized_program_nbytes": path.stat().st_size,
    "compile_only_parent": True,
    "source_revision": LLAMA_SOURCE_COMMIT,
    "program_globals": list(program.arg.globals),
    "program_global_size": list(program.arg.global_size),
    "program_local_size": list(program.arg.local_size),
    "distinct_binary_identity": True,
    "no_fallback": True,
  }


def _target_epoch_inputs(epoch: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Regenerate the target harness's deterministic inputs and one epoch oracle."""
  from extra.qk.mmq_q4k_q8_reference import (
    Q81MMQDS4Activation, Q81MMQDS4ActivationSpec, Q4KQ81MMQTileSpec,
    Q8_1_MMQ_DS4_LAYOUT, q4k_q8_1_mmq_ds4_tile_reference,
    q8_1_mmq_ds4_quantize_reference,
  )

  if not 0 <= epoch < TOTAL_EPOCHS: raise ValueError(f"epoch must be in [0,{TOTAL_EPOCHS-1}]")
  m, n, k = TARGET_ROLE_PROBE_SHAPE
  words = _random_q4_words(n, k, 20260721)
  source = np.random.default_rng(20260722).standard_normal((m, k), dtype=np.float32)
  values, scales, sums = q8_1_mmq_ds4_quantize_reference(source)
  q4_blocks = words.view(np.uint8).reshape(n, TOTAL_EPOCHS, 144)
  q4_epoch = np.ascontiguousarray(q4_blocks[:, epoch:epoch+1, :].reshape(-1).view(np.uint32))
  values_epoch = np.ascontiguousarray(values[epoch*2:(epoch+1)*2].reshape(-1))
  scales_epoch = np.ascontiguousarray(scales[epoch*2:(epoch+1)*2].reshape(-1))
  sums_epoch = np.ascontiguousarray(sums[epoch*2:(epoch+1)*2].reshape(-1))
  ds4 = Q81MMQDS4Activation(
    values[epoch*2:(epoch+1)*2],
    scales[epoch*2:(epoch+1)*2].astype(np.float16).astype(np.float32),
    sums[epoch*2:(epoch+1)*2].astype(np.float16).astype(np.float32),
    Q81MMQDS4ActivationSpec(m=m, k=256, m_tile=m),
  )
  spec = Q4KQ81MMQTileSpec(
    role="isolated_target_epoch", m=m, n=n, k=256, m_tile=m, n_tile=n,
    activation_layout=Q8_1_MMQ_DS4_LAYOUT,
  )
  reference = q4k_q8_1_mmq_ds4_tile_reference(
    q4_blocks[:, epoch:epoch+1, :].reshape(-1).view(np.uint8), ds4, spec,
  )
  return q4_epoch, values_epoch, scales_epoch, sums_epoch, reference


def target_fixture_evidence() -> dict[str, Any]:
  """Return deterministic input/repack identity shared with the strict target harness.

  This is CPU-only and intentionally records byte hashes of the complete
  epoch-major Q4/Q8 sources.  The epoch orchestrator's overwrite binaries may
  differ from an in-place accumulator binary, but both must bind this fixture.
  """
  from extra.qk.mmq_q4k_q8_reference import q8_1_mmq_ds4_quantize_reference

  m, n, k = TARGET_ROLE_PROBE_SHAPE
  words = _random_q4_words(n, k, 20260721)
  source = np.random.default_rng(20260722).standard_normal((m, k), dtype=np.float32)
  values, scales, sums = q8_1_mmq_ds4_quantize_reference(source)
  q4_blocks = words.view(np.uint8).reshape(n, TOTAL_EPOCHS, 144)
  return {
    "schema": FIXTURE_SCHEMA,
    "role": "ffn_gate_up", "shape": [m, n, k], "total_epochs": TOTAL_EPOCHS,
    "seeds": {"q4": 20260721, "q8_source": 20260722},
    "repack": {
      "q4_sha256": hashlib.sha256(np.ascontiguousarray(q4_blocks).tobytes()).hexdigest(),
      "q8_values_sha256": hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),
      "q8_scales_sha256": hashlib.sha256(np.ascontiguousarray(scales).tobytes()).hexdigest(),
      "q8_sums_sha256": hashlib.sha256(np.ascontiguousarray(sums).tobytes()).hexdigest(),
      "q4_layout": "q4_k_bytes[n, k_epoch, 144]",
      "q8_layout": "q8_ds4[epoch, m, groups]",
    },
    "source_sha256": hashlib.sha256(np.ascontiguousarray(source).tobytes()).hexdigest(),
  }


def _run_target_epoch_worker(program_path: str, output_path: str, epoch: int,
                             runtime_timeout_ms: int = DEFAULT_RUNTIME_TIMEOUT_MS) -> dict[str, Any]:
  """Load and dispatch exactly one target epoch in a fresh spawned process."""
  from tinygrad import Tensor, dtypes
  from tinygrad.engine.realize import get_runtime
  from tinygrad.uop.ops import Ops

  started = time.perf_counter()
  with Path(program_path).open("rb") as handle: program = pickle.load(handle)
  if program.op is not Ops.PROGRAM: raise RuntimeError("serialized artifact is not a PROGRAM")
  if tuple(program.arg.globals) != tuple(range(5)):
    raise RuntimeError(f"serialized target PROGRAM ABI changed: globals={program.arg.globals}")
  q4, values, scales, sums, reference = _target_epoch_inputs(epoch)
  m, n, _ = TARGET_ROLE_PROBE_SHAPE
  output = Tensor.empty(m*n, dtype=dtypes.float32, device="AMD").realize()
  q4_tensor = Tensor(q4, device="AMD").realize()
  values_tensor = Tensor(values, device="AMD").realize()
  scales_tensor = Tensor(scales, device="AMD").realize()
  sums_tensor = Tensor(sums, device="AMD").realize()
  buffers = (output.uop.buffer, q4_tensor.uop.buffer, values_tensor.uop.buffer,
             scales_tensor.uop.buffer, sums_tensor.uop.buffer)
  runtime = get_runtime("AMD", program)
  args = tuple(buffers[slot].get_buf("AMD") for slot in program.arg.globals)
  gpu_seconds = runtime(
    *args, global_size=program.arg.global_size, local_size=program.arg.local_size,
    vals=program.arg.vals({}), wait=True, timeout=runtime_timeout_ms,
  )
  got = output.numpy().reshape(m, n)
  comparison = _numeric_comparison(got, reference, rtol=RTOL, atol=ATOL)
  passed = comparison["status"] == "pass"
  output_sha = hashlib.sha256(np.ascontiguousarray(got).tobytes()).hexdigest()
  if passed: np.save(output_path, got, allow_pickle=False)
  return {
    "schema": f"{SCHEMA}.epoch", "passed": passed, "status": "PASS" if passed else "BLOCKED",
    "epoch": epoch, "shape": [m, n, 256], "comparison": comparison,
    "output_sha256": output_sha, "gpu_ms": float(gpu_seconds)*1000.0 if gpu_seconds is not None else None,
    "elapsed_seconds": time.perf_counter() - started, "no_fallback": True,
  }


def run_isolated_target_epoch(program_path: str, output_path: str, epoch: int,
                              *, timeout_seconds: float = DEFAULT_WORKER_TIMEOUT_SECONDS,
                              runtime_timeout_ms: int = DEFAULT_RUNTIME_TIMEOUT_MS) -> dict[str, Any]:
  """Run one epoch worker under a hard spawned-process deadline."""
  from tinygrad.runtime.process_isolated import run_isolated
  isolated = run_isolated(
    _run_target_epoch_worker, args=(program_path, output_path, epoch, runtime_timeout_ms),
    timeout_seconds=timeout_seconds, start_method="spawn",
  )
  if isolated.status != "passed" or not isinstance(isolated.result, dict):
    return {
      "schema": f"{SCHEMA}.epoch", "passed": False, "status": "BLOCKED", "epoch": epoch,
      "exact_blocker": "isolated epoch worker did not return a result",
      "isolated_status": isolated.status, "error": isolated.error,
      "stderr_tail": isolated.stderr[-1000:], "stdout_tail": isolated.stdout[-1000:],
      "elapsed_seconds": isolated.elapsed_seconds,
    }
  return isolated.result


def spawned_tiny_health_probe() -> bool:
  """Run the independent tiny-add health canary in a fresh process."""
  from extra.qk.prefill.host_safety_canary import make_tiny_health_probe
  from tinygrad.runtime.process_isolated import run_isolated
  result = run_isolated(
    make_tiny_health_probe(), timeout_seconds=10.0, start_method="spawn",
  )
  return bool(result.status == "passed" and result.result is True)


def _health_passed(value: Any) -> bool:
  if isinstance(value, dict): return bool(value.get("passed"))
  return value is True


def _base_result(program_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
  return {
    "schema": SCHEMA, "passed": False, "status": "BLOCKED",
    "shape": list(TARGET_ROLE_PROBE_SHAPE), "role": "ffn_gate_up",
    "diagnostic_only": True, "promotion_eligible": False,
    "production_dispatch_changed": False, "default_route": "direct_packed",
    "completed_epochs": [], "failed_epoch": None, "stop_reason": None,
    "kernel_faults": [], "epoch_results": [], "program": program_evidence or {},
    "no_fallback": True, "fixture": None, "epoch_health": [],
    "health_attestation": {"schema": ATTESTATION_SCHEMA, "preflight_passed": None,
                            "epochs": [], "status": "BLOCKED"},
  }


def orchestrate_epoch_sweep(
  *,
  epoch_indices: Iterable[int],
  compile_artifact: Callable[[str | Path], tuple[str, dict[str, Any]]] = compile_target_program_artifact,
  epoch_runner: Callable[[str, str, int], dict[str, Any]] = run_isolated_target_epoch,
  health_probe: Callable[[], Any] = spawned_tiny_health_probe,
  fault_reader: Callable[[float], str] = read_kernel_log_since,
  expected_partial_shape: tuple[int, ...] = TARGET_ROLE_PROBE_SHAPE[:2],
) -> dict[str, Any]:
  """Run a fail-closed isolated sequence and host-aggregate only verified partials."""
  epochs = tuple(int(x) for x in epoch_indices)
  if not epochs: raise ValueError("at least one epoch is required")
  if len(set(epochs)) != len(epochs): raise ValueError("epoch indices must be unique")
  if any(not 0 <= epoch < TOTAL_EPOCHS for epoch in epochs):
    raise ValueError(f"epoch indices must be in [0,{TOTAL_EPOCHS-1}]")

  with tempfile.TemporaryDirectory(prefix="tinygrad-mmq-target-epochs-") as temp_dir:
    try:
      artifact_path, program_evidence = compile_artifact(temp_dir)
    except BaseException as exc:
      out = _base_result()
      out["stop_reason"] = f"program compile/serialization failed: {type(exc).__name__}: {exc}"
      return out
    out = _base_result(program_evidence)
    out["fixture"] = target_fixture_evidence()

    def finish() -> dict[str, Any]:
      health = out["health_attestation"]
      health["epochs"] = list(out["epoch_health"])
      health["all_post_epoch_healthy"] = bool(
        health.get("preflight_passed") is True and
        all(row.get("post_health") is True for row in out["epoch_health"]))
      health["all_kernel_faults_clear"] = not out["kernel_faults"] and all(
        not row.get("kernel_faults") for row in out["epoch_health"])
      health["status"] = "PASS" if (
        health.get("preflight_passed") is True and
        health["all_post_epoch_healthy"] and health["all_kernel_faults_clear"] and
        len(out["epoch_health"]) == len(out["completed_epochs"]) == len(epochs)
      ) else "BLOCKED"
      return out

    try: preflight = health_probe()
    except BaseException as exc:
      out["stop_reason"] = f"preflight health probe failed: {type(exc).__name__}: {exc}"
      return finish()
    out["preflight_health"] = _health_passed(preflight)
    out["health_attestation"]["preflight_passed"] = out["preflight_health"]
    if not out["preflight_health"]:
      out["stop_reason"] = "preflight GPU health canary failed"
      return finish()

    aggregate = np.zeros(expected_partial_shape, dtype=np.float32)
    for epoch in epochs:
      output_path = str(Path(temp_dir) / f"epoch_{epoch:02d}.npy")
      epoch_started = time.time()
      attestation = {
        "epoch": epoch, "worker_passed": None, "kernel_faults": [],
        "kernel_log_checked": False, "post_health": None, "post_health_checked": False,
        "partial_verified": False,
        "status": "BLOCKED", "stop_stage": None,
      }
      out["epoch_health"].append(attestation)
      try: result = epoch_runner(artifact_path, output_path, epoch)
      except BaseException as exc:
        result = {"passed": False, "epoch": epoch, "status": "BLOCKED",
                  "exact_blocker": f"{type(exc).__name__}: {exc}"}
      out["epoch_results"].append(result)
      attestation["worker_passed"] = bool(isinstance(result, dict) and result.get("passed"))
      if not isinstance(result, dict) or not result.get("passed"):
        attestation["stop_stage"] = "worker"
        out["failed_epoch"] = epoch
        out["stop_reason"] = f"epoch {epoch} worker or numerical comparison failed"
        return finish()

      try: faults = parse_kernel_faults(fault_reader(epoch_started))
      except BaseException as exc:
        attestation["stop_stage"] = "kernel_log"
        out["failed_epoch"] = epoch
        out["stop_reason"] = f"kernel health log unavailable after epoch {epoch}: {type(exc).__name__}: {exc}"
        return finish()
      attestation["kernel_log_checked"] = True
      attestation["kernel_faults"] = list(faults)
      if faults:
        attestation["stop_stage"] = "kernel_fault"
        out["failed_epoch"] = epoch
        out["kernel_faults"].extend(faults)
        out["stop_reason"] = f"kernel health fault detected after epoch {epoch}"
        return finish()

      try: post_health = health_probe()
      except BaseException as exc:
        attestation["stop_stage"] = "post_health"
        out["failed_epoch"] = epoch
        out["stop_reason"] = f"post-epoch health probe failed after epoch {epoch}: {type(exc).__name__}: {exc}"
        return finish()
      attestation["post_health_checked"] = True
      attestation["post_health"] = _health_passed(post_health)
      if not attestation["post_health"]:
        attestation["stop_stage"] = "post_health"
        out["failed_epoch"] = epoch
        out["stop_reason"] = f"post-epoch GPU health canary failed after epoch {epoch}"
        return finish()

      try:
        partial = np.load(output_path, allow_pickle=False)
      except BaseException as exc:
        attestation["stop_stage"] = "partial_load"
        out["failed_epoch"] = epoch
        out["stop_reason"] = f"verified epoch {epoch} partial could not be loaded: {type(exc).__name__}: {exc}"
        return finish()
      if partial.shape != expected_partial_shape or partial.dtype != np.float32 or not np.all(np.isfinite(partial)):
        attestation["stop_stage"] = "partial_validation"
        out["failed_epoch"] = epoch
        out["stop_reason"] = f"verified epoch {epoch} partial has invalid shape/dtype/finiteness"
        return finish()
      attestation["partial_verified"] = True
      attestation["status"] = "PASS"
      aggregate += partial
      out["completed_epochs"].append(epoch)

    out.update({
      "passed": True, "status": "PASS", "stop_reason": None,
      "aggregate_shape": list(aggregate.shape),
      "aggregate_sum": float(np.sum(aggregate, dtype=np.float64)),
      "aggregate_sha256": hashlib.sha256(np.ascontiguousarray(aggregate).tobytes()).hexdigest(),
      "coverage": {"verified_epochs": list(epochs), "verified_k": 256*len(epochs),
                   "target_epochs": TOTAL_EPOCHS, "complete_target": set(epochs) == set(range(TOTAL_EPOCHS))},
    })
    return finish()


def _parse_epochs(value: str) -> tuple[int, ...]:
  if value.strip().lower() == "all": return tuple(range(TOTAL_EPOCHS))
  try: epochs = tuple(int(piece.strip()) for piece in value.split(",") if piece.strip())
  except ValueError as exc: raise argparse.ArgumentTypeError("epochs must be comma-separated integers or 'all'") from exc
  if not epochs or any(not 0 <= epoch < TOTAL_EPOCHS for epoch in epochs):
    raise argparse.ArgumentTypeError(f"epochs must be in [0,{TOTAL_EPOCHS-1}]")
  return epochs


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--epochs", type=_parse_epochs, default=(0,),
                      help="comma-separated K=256 epoch indices, or 'all' (default: 0)")
  parser.add_argument("--output", type=Path, help="optional JSON evidence path")
  args = parser.parse_args()
  result = orchestrate_epoch_sweep(epoch_indices=args.epochs)
  encoded = json.dumps(result, indent=2, sort_keys=True)
  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + os.linesep)
  print(encoded)
  return 0 if result["passed"] else 1


if __name__ == "__main__": raise SystemExit(main())


__all__ = [
  "SCHEMA", "ATTESTATION_SCHEMA", "FIXTURE_SCHEMA", "TOTAL_EPOCHS", "compile_target_program_artifact",
  "target_fixture_evidence", "orchestrate_epoch_sweep",
  "parse_kernel_faults", "read_kernel_log_since", "run_isolated_target_epoch",
  "spawned_tiny_health_probe",
]
