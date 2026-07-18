"""Fresh-child, no-target-dispatch canary for a frozen v2 AMDProgram family.

The child loads one already-frozen epoch PROGRAM binding and preconstructs an
admitted prefix (or the complete family) through tinygrad's existing
``get_runtime`` cache.  It never builds or realizes a target Tensor CALL and
never calls an MMQ runtime.  Only an independent tiny-add health kernel runs
after successful preconstruction.

This is a diagnostic discriminator for code-object upload/runtime lifetime.
It cannot qualify correctness, performance, or production promotion.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from tinygrad.runtime.process_isolated import IsolatedResult, run_isolated

from extra.qk.mmq_exact_role_spec import (
  DEFAULT_EXACT_ROLE_SPEC, ExactRoleSpec, admit_exact_role_spec, exact_role_spec,
)


SCHEMA = "tinygrad.mmq_frozen_epoch_runtime_preconstruction_canary.v2"
CHILD_SCHEMA = f"{SCHEMA}.child"
DEVICE = "AMD"
DEFAULT_TIMEOUT_SECONDS = 240.0


def _validate_prefix_epochs(role_spec: ExactRoleSpec, prefix_epochs: int | None) -> int:
  """Admit the same bounded diagnostic prefixes as the v2 GPU harness."""
  selected = role_spec.epochs if prefix_epochs is None else prefix_epochs
  allowed = tuple(sorted({1, 2, 3, role_spec.epochs}))
  if not isinstance(selected, int) or isinstance(selected, bool) or selected not in allowed:
    raise ValueError(
      f"frozen v2 runtime preconstruction prefix must be one of {allowed} "
      f"for role {role_spec.role!r}")
  return selected


def _tiny_health(device: str) -> bool:
  """Dispatch one unrelated known-safe add after target runtime construction."""
  from tinygrad import Tensor

  size = 256
  left = np.arange(size, dtype=np.float32)
  right = np.arange(size, dtype=np.float32)[::-1].copy()
  actual = (Tensor(left, device=device) + Tensor(right, device=device)).numpy()
  return bool(actual.shape == (size,) and np.allclose(actual, left + right, rtol=1e-3, atol=1e-3))


def _load_binding(role_spec: ExactRoleSpec, frozen_bundle: str) -> Any:
  from extra.qk.mmq_frozen_epoch_program_set import load_frozen_epoch_program_set_binding
  return load_frozen_epoch_program_set_binding(role_spec, frozen_bundle)


def _target_identities(binding: Any, prefix_epochs: int) -> tuple[dict[str, str], ...]:
  programs = tuple(binding.artifact.programs[:prefix_epochs])
  binaries = tuple(binding.artifact.binaries[:prefix_epochs])
  if len(programs) != prefix_epochs or len(binaries) != prefix_epochs:
    raise ValueError("frozen v2 binding lacks the requested PROGRAM identity prefix")
  return tuple({
    "function_name": program.arg.function_name,
    "binary_sha256": hashlib.sha256(binary).hexdigest(),
  } for program, binary in zip(programs, binaries))


def _run_frozen_epoch_runtime_preconstruction_worker(
    frozen_bundle: str, role: str, shape: tuple[int, int, int],
    prefix_epochs: int, device: str = DEVICE) -> dict[str, Any]:
  """Load, preconstruct, and health-check in one spawn-safe child."""
  from extra.qk.mmq_frozen_epoch_program_set import load_frozen_epoch_program_set_binding
  from extra.qk.mmq_llama_five_buffer_gpu_harness import _preconstruct_frozen_program_runtimes

  started = time.perf_counter()
  role_spec = exact_role_spec(role, shape=shape)
  prefix_epochs = _validate_prefix_epochs(role_spec, prefix_epochs)
  binding = load_frozen_epoch_program_set_binding(role_spec, frozen_bundle)
  if len(binding.artifact.programs) != role_spec.epochs or \
     len(binding.artifact.binaries) != role_spec.epochs or \
     len(binding.program_keys) != role_spec.epochs:
    raise RuntimeError("frozen v2 binding does not retain its complete admitted family")

  programs = tuple(binding.artifact.programs[:prefix_epochs])
  program_keys = tuple(binding.program_keys[:prefix_epochs])
  target_identities = _target_identities(binding, prefix_epochs)
  base = {
    "schema": CHILD_SCHEMA,
    "role": role_spec.role, "shape": list(role_spec.shape),
    "family_identity": binding.family_identity,
    "prefix_epochs": prefix_epochs, "complete_family": prefix_epochs == role_spec.epochs,
    "program_keys": list(program_keys),
    "target_program_identities": [dict(identity) for identity in target_identities],
    "compile_performed": False, "requires_recompile": False,
    "hip_used": False, "no_fallback": True,
    "target_tensor_call_constructed": False,
    "target_runtime_called": False, "target_dispatch_count": 0,
    "no_target_dispatch": True,
  }
  try:
    preconstruction = _preconstruct_frozen_program_runtimes(
      programs, program_keys, target_identities, device=device)
  except BaseException as exc:
    partial = getattr(exc, "runtime_preconstruction", None)
    return {
      **base, "status": "BLOCKED", "passed": False,
      "exact_blocker": f"frozen runtime preconstruction failed: {type(exc).__name__}: {exc}",
      "runtime_preconstruction": dict(partial) if isinstance(partial, dict) else None,
      "tiny_health_passed": None,
      "elapsed_seconds": time.perf_counter() - started,
    }

  preconstruction_passed = bool(
    preconstruction.get("status") == "PASS" and
    preconstruction.get("count") == prefix_epochs and
    preconstruction.get("ordered_program_keys") == list(program_keys) and
    preconstruction.get("no_compute_dispatch_during_preconstruction") is True and
    preconstruction.get("all_checks_pass") is True)
  if not preconstruction_passed:
    return {
      **base, "status": "BLOCKED", "passed": False,
      "exact_blocker": "runtime preconstruction evidence did not close its exact no-dispatch contract",
      "runtime_preconstruction": preconstruction,
      "tiny_health_passed": None,
      "elapsed_seconds": time.perf_counter() - started,
    }

  tiny_health_passed = _tiny_health(device)
  passed = preconstruction_passed and tiny_health_passed
  return {
    **base, "status": "PASS" if passed else "BLOCKED", "passed": passed,
    "exact_blocker": None if passed else "independent tiny health failed after runtime preconstruction",
    "runtime_preconstruction": preconstruction,
    "tiny_health_passed": tiny_health_passed,
    "elapsed_seconds": time.perf_counter() - started,
  }


def _default_fault_reader(since_timestamp: float) -> str:
  from extra.qk.mmq_target_epoch_orchestrator import read_kernel_log_since
  return read_kernel_log_since(since_timestamp)


def _default_health_probe() -> bool:
  from extra.qk.mmq_target_epoch_orchestrator import spawned_tiny_health_probe
  return bool(spawned_tiny_health_probe())


def run_frozen_epoch_runtime_preconstruction_canary(
    frozen_bundle: str | Path, *,
    role_spec: ExactRoleSpec = DEFAULT_EXACT_ROLE_SPEC,
    prefix_epochs: int | None = None,
    device: str = DEVICE,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    runner: Callable[..., IsolatedResult] = run_isolated,
    fault_reader: Callable[[float], str] = _default_fault_reader,
    health_probe: Callable[[], bool] = _default_health_probe) -> dict[str, Any]:
  """Run one health-guarded, fail-closed runtime-only family discriminator."""
  base: dict[str, Any] = {
    "schema": SCHEMA, "status": "BLOCKED", "passed": False,
    "research_only": True, "diagnostic_only": True, "promotion_eligible": False,
    "production_dispatch_changed": False, "default_route": "direct_packed",
    "compile_performed": False, "requires_recompile": False,
    "hip_used": False, "no_fallback": True,
    "target_tensor_call_constructed": False,
    "target_runtime_called": False, "target_dispatch_count": 0,
    "no_target_dispatch": True,
    "health_before": None, "health_after": None,
    "kernel_faults": [], "child": None,
  }
  try:
    role_spec = admit_exact_role_spec(role_spec)
    prefix_epochs = _validate_prefix_epochs(role_spec, prefix_epochs)
  except (TypeError, ValueError) as exc:
    return {**base, "exact_blocker": str(exc)}
  if device != DEVICE:
    return {**base, "exact_blocker": "frozen v2 runtime preconstruction canary only admits device='AMD'"}
  if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
    return {**base, "exact_blocker": "timeout_seconds must be positive"}
  bundle = str(Path(frozen_bundle).resolve())
  try:
    binding = _load_binding(role_spec, bundle)
    expected_program_keys = tuple(binding.program_keys[:prefix_epochs])
    expected_target_identities = _target_identities(binding, prefix_epochs)
    if len(binding.program_keys) != role_spec.epochs or len(expected_program_keys) != prefix_epochs:
      raise ValueError("frozen v2 binding does not retain its complete admitted family")
  except BaseException as exc:
    return {**base, "exact_blocker":
      f"parent frozen binding validation failed: {type(exc).__name__}: {exc}"}
  base.update({
    "role": role_spec.role, "shape": list(role_spec.shape),
    "prefix_epochs": prefix_epochs, "complete_family": prefix_epochs == role_spec.epochs,
    "frozen_bundle": bundle, "family_identity": binding.family_identity,
    "program_keys": list(expected_program_keys),
    "target_program_identities": [dict(row) for row in expected_target_identities],
  })

  started = time.time()
  try: base["health_before"] = bool(health_probe())
  except BaseException as exc:
    return {**base, "exact_blocker": f"pre-run health probe failed: {type(exc).__name__}: {exc}"}
  if not base["health_before"]:
    return {**base, "exact_blocker": "pre-run GPU health probe reported device unhealthy"}

  isolated = None
  try:
    isolated = runner(
      _run_frozen_epoch_runtime_preconstruction_worker,
      args=(bundle, role_spec.role, role_spec.shape, prefix_epochs, device),
      timeout_seconds=float(timeout_seconds), start_method="spawn")
    base["child_status"], base["child_error"] = isolated.status, isolated.error
    if isolated.status == "passed" and isinstance(isolated.result, dict):
      base["child"] = isolated.result
  except BaseException as exc:
    isolated = None
    base["child_status"], base["child_error"] = "runner_error", f"{type(exc).__name__}: {exc}"

  try:
    from extra.qk.mmq_target_epoch_orchestrator import parse_kernel_faults
    base["kernel_faults"] = parse_kernel_faults(fault_reader(started))
  except BaseException as exc:
    base["kernel_fault_scan_error"] = f"{type(exc).__name__}: {exc}"
  try: base["health_after"] = bool(health_probe())
  except BaseException as exc:
    base["health_after_error"] = f"{type(exc).__name__}: {exc}"

  if "kernel_fault_scan_error" in base:
    return {**base, "exact_blocker": f"kernel-log scan failed: {base['kernel_fault_scan_error']}"}
  if base["kernel_faults"]:
    return {**base, "exact_blocker": "AMD kernel fault/reset marker observed"}
  if base["health_after"] is not True:
    return {**base, "exact_blocker": base.get("health_after_error", "post-run GPU health probe reported device unhealthy")}
  if isolated is None:
    return {**base, "exact_blocker": f"runtime-preconstruction runner failed: {base['child_error']}"}
  if isolated.status != "passed" or not isinstance(isolated.result, dict):
    return {**base, "exact_blocker": isolated.error or "runtime-preconstruction child returned no result"}
  child = isolated.result
  preconstruction = child.get("runtime_preconstruction")
  runtimes = preconstruction.get("runtimes") if isinstance(preconstruction, Mapping) else None
  runtime_rows_match = isinstance(runtimes, list) and len(runtimes) == prefix_epochs and all(
    isinstance(row, Mapping) and row.get("epoch") == epoch and
    row.get("program_key") == expected_program_keys[epoch] and
    row.get("program_identity") == expected_target_identities[epoch] and
    row.get("expected_program_identity") == expected_target_identities[epoch] and
    row.get("all_checks_pass") is True
    for epoch, row in enumerate(runtimes))
  if child.get("schema") != CHILD_SCHEMA or child.get("status") != "PASS" or child.get("passed") is not True or \
     child.get("tiny_health_passed") is not True or \
     child.get("no_target_dispatch") is not True or child.get("target_dispatch_count") != 0 or \
     child.get("target_runtime_called") is not False or child.get("target_tensor_call_constructed") is not False or \
     child.get("role") != role_spec.role or child.get("shape") != list(role_spec.shape) or \
     child.get("prefix_epochs") != prefix_epochs or child.get("family_identity") != binding.family_identity or \
     child.get("complete_family") is not (prefix_epochs == role_spec.epochs) or \
     child.get("program_keys") != list(expected_program_keys) or \
     child.get("target_program_identities") != [dict(row) for row in expected_target_identities] or \
     child.get("compile_performed") is not False or child.get("requires_recompile") is not False or \
     child.get("hip_used") is not False or child.get("no_fallback") is not True or \
     not isinstance(preconstruction, Mapping) or preconstruction.get("enabled") is not True or \
     preconstruction.get("status") != "PASS" or preconstruction.get("device") != DEVICE or \
     preconstruction.get("count") != prefix_epochs or \
     preconstruction.get("ordered_program_keys") != list(expected_program_keys) or \
     preconstruction.get("no_compute_dispatch_during_preconstruction") is not True or \
     preconstruction.get("runtime_cache_retains_code_allocations") is not True or \
     preconstruction.get("all_checks_pass") is not True or not runtime_rows_match:
    return {**base, "exact_blocker": child.get("exact_blocker") or
      "runtime-preconstruction child did not close its exact no-target contract"}
  return {**base, "status": "PASS", "passed": True, "exact_blocker": None}


def main() -> int:
  import argparse
  import json

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("frozen_bundle", type=Path)
  parser.add_argument("--role", default=DEFAULT_EXACT_ROLE_SPEC.role)
  parser.add_argument("--shape", type=int, nargs=3, metavar=("M", "N", "K"),
                      default=DEFAULT_EXACT_ROLE_SPEC.shape)
  parser.add_argument("--prefix-epochs", type=int)
  parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  role_spec = exact_role_spec(args.role, shape=tuple(args.shape))
  result = run_frozen_epoch_runtime_preconstruction_canary(
    args.frozen_bundle, role_spec=role_spec, prefix_epochs=args.prefix_epochs,
    timeout_seconds=args.timeout_seconds)
  encoded = json.dumps(result, indent=2, sort_keys=True)
  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n")
  print(encoded)
  return 0 if result["passed"] else 1


if __name__ == "__main__": raise SystemExit(main())


__all__ = [
  "CHILD_SCHEMA", "SCHEMA", "run_frozen_epoch_runtime_preconstruction_canary",
]
