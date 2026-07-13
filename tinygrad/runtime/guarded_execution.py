"""Transport-neutral guarded execution and correctness lifecycle.

This module owns the safety mechanics shared by every compiled transport:
health preflight, guarded allocation callbacks, input immutability, full output
comparison, finite-value checks, and release.  It deliberately knows nothing
about LDS, registers, AMD ISA, or candidate selection.  A caller supplies the
transport-specific executable and allocation/dispatch hooks.
"""
from __future__ import annotations

from dataclasses import dataclass
import math, time
from typing import Any, Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class GuardPolicy:
  prefix_bytes: int = 4096
  suffix_bytes: int = 4096
  pattern: str = "deterministic_per_buffer"
  check_inputs_unchanged: bool = True
  check_before_and_after_each_launch: bool = True
  timeout_seconds: float = 10.0
  rtol: float = 2e-2
  atol: float = 2e-2

  def __post_init__(self):
    if not isinstance(self.prefix_bytes, int) or self.prefix_bytes <= 0: raise ValueError("prefix_bytes must be positive")
    if not isinstance(self.suffix_bytes, int) or self.suffix_bytes <= 0: raise ValueError("suffix_bytes must be positive")
    if not isinstance(self.pattern, str) or not self.pattern: raise ValueError("guard pattern is required")
    if not isinstance(self.timeout_seconds, (int, float)) or self.timeout_seconds <= 0: raise ValueError("timeout must be positive")
    if not isinstance(self.rtol, (int, float)) or self.rtol < 0: raise ValueError("rtol must be non-negative")
    if not isinstance(self.atol, (int, float)) or self.atol < 0: raise ValueError("atol must be non-negative")


@dataclass(frozen=True)
class GuardedBuffer:
  """Opaque allocation returned by a transport-independent allocator hook."""
  name: str
  resource: Any
  prefix_bytes: int
  suffix_bytes: int


@dataclass(frozen=True)
class GuardedExecutionHooks:
  """Adapters for allocation and dispatch; no transport policy lives here."""
  allocate: Callable[[str, np.ndarray, GuardPolicy], GuardedBuffer]
  upload: Callable[[GuardedBuffer, np.ndarray], None]
  readback: Callable[[GuardedBuffer], np.ndarray]
  guards_intact: Callable[[GuardedBuffer], bool]
  dispatch: Callable[[Any, Mapping[str, GuardedBuffer]], float | int | None]
  health: Callable[[], bool]
  release: Callable[[GuardedBuffer], None]


def _guard_bytes(name: str, count: int) -> bytes:
  seed = sum((idx + 1) * ord(char) for idx, char in enumerate(name)) & 0xff
  return bytes((seed + idx * 29) & 0xff for idx in range(count))


def make_tinygrad_guarded_hooks(device: str, dispatch: Callable[[Any, Mapping[str, GuardedBuffer]], float | int | None],
                                health: Callable[[], bool]) -> GuardedExecutionHooks:
  """Build the standard Buffer-backed hooks for a compiled tinygrad PROGRAM.

  The dispatch callback remains transport/ABI-specific; allocation, guard
  placement, host transfer, readback, and release are shared.  A payload is a
  view into one larger byte buffer, so the kernel receives the payload view
  while prefix/suffix bytes remain available for corruption checks.
  """
  from tinygrad.device import Buffer
  from tinygrad.dtype import _from_np_dtype, dtypes

  def allocate(name: str, value: np.ndarray, policy: GuardPolicy) -> GuardedBuffer:
    array = np.ascontiguousarray(value)
    raw = Buffer(device, policy.prefix_bytes + array.nbytes + policy.suffix_bytes, dtypes.uint8, preallocate=True)
    prefix, suffix = _guard_bytes(name + ":prefix", policy.prefix_bytes), _guard_bytes(name + ":suffix", policy.suffix_bytes)
    initial = prefix + bytes(array.nbytes) + suffix
    raw.copyin(memoryview(bytearray(initial)))
    payload = raw.view(array.size, _from_np_dtype(array.dtype), policy.prefix_bytes)
    payload.allocate()
    return GuardedBuffer(name, {"raw": raw, "payload": payload, "shape": array.shape,
                                "prefix": prefix, "suffix": suffix}, policy.prefix_bytes, policy.suffix_bytes)

  def upload(buffer: GuardedBuffer, value: np.ndarray) -> None:
    buffer.resource["payload"].copyin(memoryview(np.ascontiguousarray(value)))

  def readback(buffer: GuardedBuffer) -> np.ndarray:
    return buffer.resource["payload"].numpy().reshape(buffer.resource["shape"]).copy()

  def guards_intact(buffer: GuardedBuffer) -> bool:
    raw, resource = buffer.resource["raw"], buffer.resource
    data = bytearray(raw.nbytes); raw.copyout(memoryview(data))
    return bytes(data[:buffer.prefix_bytes]) == resource["prefix"] and bytes(data[-buffer.suffix_bytes:]) == resource["suffix"]

  def release(buffer: GuardedBuffer) -> None:
    payload, raw = buffer.resource["payload"], buffer.resource["raw"]
    if payload.is_allocated(): payload.deallocate()
    if raw.is_allocated(): raw.deallocate()

  return GuardedExecutionHooks(allocate, upload, readback, guards_intact, dispatch, health, release)


def make_tinygrad_executable_hooks(device: str, health: Callable[[], bool],
                                   argument_order: tuple[str, ...] = ("a", "b", "output")) -> GuardedExecutionHooks:
  """Bind logical guarded buffers to an :class:`ExecutableHandle` ABI.

  The default order is the shared dense GEMM contract.  Other workloads can
  provide a different logical order without adding transport branches.
  """
  if not argument_order or len(set(argument_order)) != len(argument_order): raise ValueError("argument order must be unique")

  def dispatch(executable: Any, buffers: Mapping[str, GuardedBuffer]) -> float | int | None:
    try: payloads = tuple(buffers[name].resource["payload"].get_buf(device) for name in argument_order)
    except KeyError as exc: raise ValueError(f"compiled ABI buffer is missing: {exc.args[0]}") from exc
    return executable.dispatch(*payloads)

  return make_tinygrad_guarded_hooks(device, dispatch, health)


def run_tinygrad_executable_guarded(*, executable: Any, device: str, inputs: Mapping[str, np.ndarray],
                                    reference: np.ndarray, health: Callable[[], bool],
                                    policy: GuardPolicy = GuardPolicy(), identity: Mapping[str, Any] | None = None,
                                    argument_order: tuple[str, ...] = ("a", "b", "output"), output_dtype: Any = np.float16) -> dict[str, Any]:
  """Run a compiled tinygrad executable through the shared guarded lifecycle."""
  return run_guarded_execution(executable=executable, inputs=inputs, reference=reference,
    hooks=make_tinygrad_executable_hooks(device, health, argument_order), policy=policy,
    identity=identity, output_dtype=output_dtype)


def _finite(value: Any) -> bool:
  try: return bool(np.all(np.isfinite(np.asarray(value))))
  except (TypeError, ValueError): return False


def _nonconstant(inputs: Mapping[str, np.ndarray]) -> bool:
  return bool(inputs) and any(np.asarray(value).size > 1 and np.ptp(np.asarray(value)) != 0 for value in inputs.values())


def _blocked(identity: Mapping[str, Any], *, error: str, policy: GuardPolicy,
             before: bool = False, after: bool = False, dispatch_performed: bool = False) -> dict[str, Any]:
  return {"schema": "guarded-execution.v1", "status": "blocked", "passed": False,
          "errors": [error], "identity": dict(identity), "device_healthy_before": before,
          "device_healthy_after": after, "guards_intact": False, "inputs_unchanged": False,
          "numerics_passed": False, "full_output_compared": False, "nonconstant_inputs": False,
          "device_fault": not after if dispatch_performed else False, "dispatch_performed": dispatch_performed,
          "elapsed_seconds": 0.0, "rtol": policy.rtol, "atol": policy.atol}


def run_guarded_execution(*, executable: Any, inputs: Mapping[str, np.ndarray], reference: np.ndarray,
                          hooks: GuardedExecutionHooks, policy: GuardPolicy = GuardPolicy(),
                          identity: Mapping[str, Any] | None = None, output_dtype: Any = np.float16) -> dict[str, Any]:
  """Run one guarded transport execution through caller-owned hooks.

  The function is intentionally the only lifecycle owner: no dispatch occurs
  until health and guarded allocations pass, and a result cannot pass without
  post-dispatch health, intact guards, unchanged inputs, finite output, and a
  full `allclose` comparison against the supplied reference.  Process-level
  hard timeouts belong around this function via :func:`run_isolated`.
  """
  identity = {} if identity is None else dict(identity)
  if not isinstance(inputs, Mapping) or not inputs: return _blocked(identity, error="non-empty inputs are required", policy=policy)
  if not isinstance(reference, np.ndarray) or not reference.size: return _blocked(identity, error="non-empty ndarray reference is required", policy=policy)
  if not isinstance(hooks, GuardedExecutionHooks): raise TypeError("GuardedExecutionHooks are required")
  if not _nonconstant(inputs): return _blocked(identity, error="nonconstant inputs are required", policy=policy)
  try: before = bool(hooks.health())
  except Exception: before = False
  if not before: return _blocked(identity, error="device health preflight failed", policy=policy, before=False)

  allocations: dict[str, GuardedBuffer] = {}
  snapshots = {name: np.array(value, copy=True) for name, value in inputs.items()}
  started = time.monotonic(); dispatch_performed = False; errors: list[str] = []
  output = None; elapsed = None
  def release_all() -> None:
    for buffer in allocations.values():
      try: hooks.release(buffer)
      except Exception: pass
  try:
    for name, value in inputs.items():
      array = np.asarray(value)
      allocations[name] = hooks.allocate(name, array, policy)
      hooks.upload(allocations[name], array)
    allocations["output"] = hooks.allocate("output", np.zeros(reference.shape, dtype=output_dtype), policy)
    if policy.check_before_and_after_each_launch and not all(hooks.guards_intact(buf) for buf in allocations.values()):
      release_all()
      return _blocked(identity, error="guard corruption detected before dispatch", policy=policy, before=True)
    dispatch_performed = True
    value = hooks.dispatch(executable, allocations)
    if value is not None: elapsed = float(value)
    output = np.asarray(hooks.readback(allocations["output"]))
  except Exception as exc:
    errors.append(f"execution failed: {type(exc).__name__}: {exc}")
  after = False
  try: after = bool(hooks.health())
  except Exception as exc: errors.append(f"device health postflight failed: {type(exc).__name__}: {exc}")
  if not after: errors.append("device health postflight failed")
  guards = all(hooks.guards_intact(buf) for buf in allocations.values()) if allocations else False
  if not guards: errors.append("guard corruption detected after dispatch")
  unchanged = True
  if policy.check_inputs_unchanged:
    for name, expected in snapshots.items():
      try: unchanged = unchanged and bool(np.array_equal(hooks.readback(allocations[name]), expected))
      except Exception: unchanged = False
    if not unchanged: errors.append("input buffer mutation detected")
  finite = _finite(output) if output is not None else False
  if not finite: errors.append("output is missing or contains non-finite values")
  full = output is not None and output.shape == reference.shape
  if not full: errors.append("full output shape comparison was not possible")
  numerics = bool(full and finite and np.allclose(output, reference, rtol=policy.rtol, atol=policy.atol))
  if not numerics: errors.append("full output numerical comparison failed")
  wall = time.monotonic() - started
  if elapsed is None: elapsed = wall
  if not math.isfinite(elapsed) or elapsed < 0 or elapsed > policy.timeout_seconds:
    errors.append("dispatch exceeded the execution timeout")
  for buffer in allocations.values():
    try: hooks.release(buffer)
    except Exception as exc: errors.append(f"buffer release failed: {type(exc).__name__}: {exc}")
  return {"schema": "guarded-execution.v1", "status": "passed" if not errors else "failed", "passed": not errors,
          "errors": errors, "identity": identity, "device_healthy_before": before,
          "device_healthy_after": after, "guards_intact": guards, "inputs_unchanged": unchanged,
          "numerics_passed": numerics, "full_output_compared": full, "nonconstant_inputs": True,
          "device_fault": dispatch_performed and not after, "dispatch_performed": dispatch_performed,
          "elapsed_seconds": float(elapsed), "rtol": policy.rtol, "atol": policy.atol,
          "output_shape": list(output.shape) if output is not None else None,
          "max_abs_error": float(np.max(np.abs(output - reference))) if output is not None and output.shape == reference.shape else None}


__all__ = ["GuardPolicy", "GuardedBuffer", "GuardedExecutionHooks", "make_tinygrad_guarded_hooks",
           "make_tinygrad_executable_hooks", "run_guarded_execution", "run_tinygrad_executable_guarded"]
