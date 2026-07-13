"""One isolated guarded executor: the single safety boundary for GPU dispatch.

This module composes the four grounded pieces of the arc into ONE entry point:

  * :func:`tinygrad.runtime.process_isolated.run_isolated` -- the hard process
    timeout and process-group cleanup;
  * :func:`extra.qk.prefill.guarded_execution.run_guarded_execution` -- health
    preflight, guarded allocation, input immutability, full-output correctness,
    finite checks, and buffer release;
  * :class:`tinygrad.runtime.bridge.ExecutableHandle` -- the single PROGRAM-geometry
    dispatch path (constructed by the child, never the parent);
  * the canonical typed dispatch states in
    :mod:`tinygrad.runtime.execution_bridge_contracts`.

P0-2 (dangerous execution must not be caller-owned).  The PARENT function
:func:`run_isolated_guarded_execution` constructs NO live GPU runtime.  It only
validates the request, hands an opaque ``builder`` callable to ``run_isolated``,
records the child's terminal state, runs an INDEPENDENT health probe in its own
isolated child, and persists failures.  The runtime is constructed strictly
inside the child, by the ``builder`` -- a grep of the parent code path finds no
``Device[...]``, ``.runtime(``, ``get_runtime``, ``prepare_executable``, or
``Buffer(``.  The production builder :func:`make_tinygrad_bundle_builder` holds
that construction and is invoked only from inside the child.

P0-7 (device-loss recovery is unproved).  Any timeout, vanished child, device
fault, guard corruption, or numerical failure produces a truthful terminal
state and STOPS.  There is no automatic reset, retry, or continuation; a new
session begins with a fresh call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from extra.qk.prefill.guarded_execution import (GuardedExecutionHooks, GuardPolicy,
                                                make_tinygrad_executable_hooks, run_guarded_execution)
from tinygrad.runtime.execution_bridge_contracts import DISPATCH_STATES, dispatch_state
from tinygrad.runtime.process_isolated import IsolatedResult, run_isolated

SCHEMA = "isolated-guarded-executor.v1"


@dataclass(frozen=True)
class ExecutableBundle:
  """What a builder hands the child: the dispatch target and its guarded hooks.

  Both are constructed INSIDE the child.  ``executable`` is any object with a
  PROGRAM-geometry ``dispatch`` (an :class:`ExecutableHandle`) and an optional
  ``close``.  ``hooks`` own allocation/upload/readback/guards/dispatch/health/
  release for that executable.
  """
  executable: Any
  hooks: GuardedExecutionHooks


@dataclass(frozen=True)
class ExecutionRequest:
  """The serializable-ish description of ONE guarded run (no live runtime)."""
  inputs: Mapping[str, np.ndarray]
  reference: np.ndarray
  policy: GuardPolicy = GuardPolicy()
  identity: Mapping[str, Any] | None = None
  output_dtype: Any = np.float16


@dataclass(frozen=True)
class IsolatedExecutionResult:
  """One truthful, typed terminal outcome.  The parent NEVER auto-retries."""
  dispatch_state: str
  passed: bool
  terminal: Mapping[str, Any]
  health_after: bool
  errors: tuple[str, ...] = ()
  guarded: Mapping[str, Any] | None = None
  health_terminal: Mapping[str, Any] | None = None
  identity: Mapping[str, Any] = field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    return {"schema": SCHEMA, "dispatch_state": self.dispatch_state, "passed": self.passed,
            "terminal": dict(self.terminal), "health_after": self.health_after,
            "errors": list(self.errors), "guarded": dict(self.guarded) if self.guarded is not None else None,
            "health_terminal": dict(self.health_terminal) if self.health_terminal is not None else None,
            "identity": dict(self.identity)}


# --- Child-side entry points (RUN IN THE ISOLATED CHILD, never the parent) ---

def _child_execute(builder: Callable[[], ExecutableBundle], request: ExecutionRequest) -> dict[str, Any]:
  """Construct the runtime, run the guarded lifecycle, and close -- all in-child.

  Every controlled outcome returns a dict; only a genuinely vanished child
  (crash/kill/hard-timeout) fails to return one, which the parent reads as
  device loss.  Runtime construction failures and dispatch/ABI errors are
  controlled and come back as a dict with ``passed=False``.
  """
  try:
    bundle = builder()
  except BaseException as exc:  # runtime construction is a controlled failure, not a crash
    return {"schema": SCHEMA, "status": "failed", "passed": False, "dispatch_performed": False,
            "device_fault": False, "errors": [f"runtime construction failed: {type(exc).__name__}: {exc}"]}
  cleanup_errors: list[str] = []
  try:
    result = run_guarded_execution(executable=bundle.executable, inputs=request.inputs,
      reference=request.reference, hooks=bundle.hooks, policy=request.policy,
      identity=request.identity, output_dtype=request.output_dtype)
  finally:
    close = getattr(bundle.executable, "close", None)
    if close is not None:
      try: close()
      except Exception as exc: cleanup_errors.append(f"executable close failed: {type(exc).__name__}: {exc}")
  if cleanup_errors:
    result = {**result, "status": "failed", "passed": False,
              "errors": [*result.get("errors", ()), *cleanup_errors]}
  return result


def _child_health(probe: Callable[[], bool]) -> bool:
  return bool(probe())


# --- Parent orchestration (constructs NO runtime) ----------------------------

def _classify(child: IsolatedResult, health_after: bool) -> tuple[str, bool, dict[str, Any] | None, list[str]]:
  """Map the child's terminal state + independent health to a typed outcome."""
  errors: list[str] = []
  guarded: dict[str, Any] | None = None
  if child.timed_out:
    state = "timed_out"
    errors.append("isolated child exceeded the hard timeout")
  elif child.status != "passed" or not isinstance(child.result, Mapping):
    # The child callback ALWAYS returns a dict on controlled paths, so a missing
    # dict means the process vanished (crash/kill): treat it as device loss.
    state = "device_lost"
    errors.append(f"isolated child produced no result: {child.error or 'unknown child termination'}")
  else:
    guarded = dict(child.result)
    errors.extend(str(e) for e in guarded.get("errors", ()))
    if guarded.get("device_fault") is True: state = "device_lost"
    elif guarded.get("passed") is True: state = "completed"
    else: state = "failed"
  if not health_after:
    # P0-7: independent post-run health loss ends the session as device loss.
    if state not in ("timed_out", "device_lost"): errors.append("independent health check failed after dispatch")
    state = "device_lost"
  return dispatch_state(state), state == "completed" and health_after, guarded, errors


def run_isolated_guarded_execution(*, builder: Callable[[], ExecutableBundle], request: ExecutionRequest,
                                   health_probe: Callable[[], bool] | None = None,
                                   timeout_seconds: float | None = None, terminate_grace_seconds: float = 0.25,
                                   health_timeout_seconds: float = 10.0,
                                   persist: Callable[[dict[str, Any]], None] | None = None) -> IsolatedExecutionResult:
  """Run ONE guarded GPU dispatch behind the process-isolation boundary.

  The parent validates, launches the child (which owns all dangerous behavior),
  records the child's terminal state, runs an INDEPENDENT isolated health probe,
  classifies a single truthful typed dispatch state, and persists any failure.
  It constructs no runtime and never retries.
  """
  if not callable(builder): raise TypeError("builder must be callable and is invoked only inside the child")
  if not isinstance(request, ExecutionRequest): raise TypeError("an ExecutionRequest is required")
  if health_probe is not None and not callable(health_probe): raise TypeError("health_probe must be callable")
  identity = dict(request.identity) if request.identity else {}

  # Pre-launch validation MUST fail closed without ever launching or dispatching.
  if not isinstance(request.inputs, Mapping) or not request.inputs:
    return _terminal("not_attempted", identity, ["non-empty inputs are required"], persist)
  if not isinstance(request.reference, np.ndarray) or not request.reference.size:
    return _terminal("not_attempted", identity, ["a non-empty ndarray reference is required"], persist)

  timeout = float(timeout_seconds if timeout_seconds is not None else request.policy.timeout_seconds)
  # The builder is passed as OPAQUE data into the child; the parent never calls it.
  child = run_isolated(_child_execute, args=(builder, request), timeout_seconds=timeout,
                       terminate_grace_seconds=terminate_grace_seconds)

  # Independent, separately-timed health probe -- in ITS OWN isolated child, so
  # the parent still constructs no runtime and cannot itself be wedged.
  health_after, health_terminal = True, None
  if health_probe is not None:
    hp = run_isolated(_child_health, args=(health_probe,), timeout_seconds=health_timeout_seconds,
                      terminate_grace_seconds=terminate_grace_seconds)
    health_after = bool(hp.status == "passed" and hp.result is True)
    health_terminal = {"status": hp.status, "result": bool(hp.result is True), "timed_out": hp.timed_out,
                       "error": hp.error, "elapsed_seconds": hp.elapsed_seconds}

  state, passed, guarded, errors = _classify(child, health_after)
  terminal = {"child_status": child.status, "timed_out": child.timed_out, "error": child.error,
              "produced_result": isinstance(child.result, Mapping), "elapsed_seconds": child.elapsed_seconds,
              "stdout": child.stdout, "stderr": child.stderr}
  outcome = IsolatedExecutionResult(dispatch_state=state, passed=passed, terminal=terminal,
                                    health_after=health_after, errors=tuple(errors), guarded=guarded,
                                    health_terminal=health_terminal, identity=identity)
  if not passed and persist is not None: persist(outcome.to_dict())
  return outcome


def _terminal(state: str, identity: dict[str, Any], errors: list[str],
              persist: Callable[[dict[str, Any]], None] | None) -> IsolatedExecutionResult:
  outcome = IsolatedExecutionResult(dispatch_state=dispatch_state(state), passed=False,
                                    terminal={"child_status": "not_launched", "produced_result": False},
                                    health_after=False, errors=tuple(errors), guarded=None, identity=identity)
  if persist is not None: persist(outcome.to_dict())
  return outcome


# --- Production builder (RUN IN THE CHILD ONLY) ------------------------------

def make_tinygrad_bundle_builder(*, program: Any, compile_evidence: Mapping[str, Any], device: str = "AMD",
                                 health: Callable[[], bool], argument_order: tuple[str, ...] = ("a", "b", "output")
                                 ) -> Callable[[], ExecutableBundle]:
  """Return a builder that constructs the real runtime bundle INSIDE the child.

  The returned closure -- and ONLY the returned closure, invoked by
  ``_child_execute`` in the isolated child -- imports the bridge, resolves the
  live :class:`ExecutableHandle` via ``prepare_executable``, and binds the
  Buffer-backed guarded hooks.  Nothing here runs in the parent.
  """
  def build() -> ExecutableBundle:
    from tinygrad.runtime.bridge import prepare_executable
    executable = prepare_executable(program, compile_evidence, device=device)
    hooks = make_tinygrad_executable_hooks(device, health, argument_order)
    return ExecutableBundle(executable, hooks)
  return build


__all__ = ["SCHEMA", "DISPATCH_STATES", "ExecutableBundle", "ExecutionRequest", "IsolatedExecutionResult",
           "run_isolated_guarded_execution", "make_tinygrad_bundle_builder"]
