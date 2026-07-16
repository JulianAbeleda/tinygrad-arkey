"""C4 harmless host safety-boundary canary (P0-2 shape, P0-7 health authority).

This supersedes the callback-owned canary in
``attn_qo_direct_l2_hardware_canary_20260712.py`` for the SAFETY probe.  It owns
no dangerous behaviour itself: every GPU operation flows through the one process
boundary in :func:`extra.qk.prefill.isolated_guarded_executor.run_isolated_guarded_execution`.

Two things live here:

  * an INDEPENDENT health probe (P0-7) that runs a KNOWN-SAFE tiny elementwise
    add on tiny buffers and returns a bool "the device is alive"; it is the
    separately-timed health canary the parent runs after any dispatch, and it is
    invokable standalone by the lead;
  * a host safety-boundary harness :func:`run_host_safety_canary`.  In the
    default ``execute=False`` mode it compiles/audits ONLY and prints the exact
    bounded GPU work plan -- it dispatches nothing.  In ``execute=True`` mode it
    runs the tiny-safe-op through the isolated guarded executor and then the
    independent health probe, returning one truthful typed terminal record.

The safe op is a float32 ``a + b`` on ``size`` elements.  Its buffer ABI is the
standard tinygrad order (output slot first, then the two inputs), and the add is
commutative so input order is immaterial to correctness.

No dispatch, runtime construction, or Device construction happens at import time
or in ``execute=False`` mode's control flow beyond the compile-only step; the
real op runs only when the lead calls ``execute=True``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from extra.qk.prefill.guarded_execution import GuardPolicy
from extra.qk.prefill.isolated_guarded_executor import (ExecutableBundle, ExecutionRequest, IsolatedExecutionResult,
                                                        build_tinygrad_bundle, make_tinygrad_bundle_builder,
                                                        run_isolated_guarded_execution)
from extra.qk.prefill.execution_bridge_contracts import dispatch_state
from tinygrad.runtime.process_isolated import run_isolated

SCHEMA = "host-safety-canary.v1"
SAFE_SIZE = 256
SAFE_DEVICE = "AMD"
# Standard tinygrad kernel ABI for `out = a + b`: output buffer slot first, then
# the input buffers.  The add is commutative, so a<->b order never changes truth.
SAFE_ARGUMENT_ORDER = ("output", "a", "b")
SAFE_TIMEOUT_SECONDS = 10.0


def _safe_arrays(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  # Deterministic, NONCONSTANT inputs so the guarded lifecycle admits them.
  a = np.arange(size, dtype=np.float32)
  b = np.arange(size, dtype=np.float32)[::-1].copy()
  return a, b, (a + b).astype(np.float32)


def _safe_request(size: int, *, identity: Mapping[str, Any]) -> ExecutionRequest:
  a, b, reference = _safe_arrays(size)
  return ExecutionRequest(inputs={"a": a, "b": b}, reference=reference,
                          policy=GuardPolicy(timeout_seconds=SAFE_TIMEOUT_SECONDS),
                          identity=dict(identity), output_dtype=np.float32)


# --- The known-safe tiny GPU op + the independent health probe (P0-7) --------

def _tiny_add_is_alive(size: int, device: str) -> bool:
  """Run one tiny elementwise add and confirm it is numerically correct.

  RUNS IN THE ISOLATED CHILD (via the executor's health seam or ``run_isolated``);
  the parent never calls this directly.  A tiny op that reads back the expected
  sum is a truthful "the device is alive" signal.
  """
  from tinygrad import Tensor, dtypes
  from tinygrad.helpers import Context
  a_np, b_np, reference = _safe_arrays(size)
  with Context(DEV=device):
    a = Tensor(a_np, dtype=dtypes.float32)
    b = Tensor(b_np, dtype=dtypes.float32)
    out = (a + b).numpy()
  return bool(getattr(out, "shape", None) == (size,) and np.allclose(out, reference, rtol=1e-3, atol=1e-3))


@dataclass(frozen=True)
class TinyHealthProbe:
  """A PICKLABLE independent health probe (``() -> bool``).

  The probe runs in its OWN spawned child, which re-initializes the GPU fresh, so
  it must survive pickling -- hence a frozen dataclass, not a closure/lambda.
  """
  size: int = SAFE_SIZE
  device: str = SAFE_DEVICE

  def __call__(self) -> bool:
    return _tiny_add_is_alive(self.size, self.device)


def make_tiny_health_probe(*, size: int = SAFE_SIZE, device: str = SAFE_DEVICE) -> Callable[[], bool]:
  """Return the independent, PICKLABLE health probe the parent hands the executor.

  The returned ``() -> bool`` is invoked inside an isolated spawned child;
  standalone the lead can call ``make_tiny_health_probe()()`` directly.
  """
  return TinyHealthProbe(size, device)


def _always_alive() -> bool:
  """Module-level (picklable) in-child health hook for the tiny-add bundle."""
  return True


def build_tiny_add_bundle(*, size: int = SAFE_SIZE, device: str = SAFE_DEVICE,
                          argument_order: tuple[str, ...] = SAFE_ARGUMENT_ORDER) -> ExecutableBundle:
  """Recompile the tiny ``a + b`` PROGRAM and construct its runtime FRESH in-child.

  RUNS IN THE SPAWNED CHILD.  A UOp is not picklable, so the parent hands only
  picklable descriptors (``size``/``device``); this function re-runs
  :func:`compile_tiny_safe_add` and :func:`build_tinygrad_bundle` in-process so
  the child owns a freshly-compiled program and a freshly-initialized device.
  """
  program, evidence = compile_tiny_safe_add(size=size, device=device)
  return build_tinygrad_bundle(program=program, compile_evidence=evidence, device=device,
                               argument_order=argument_order, health=_always_alive)


def tiny_device_health(*, size: int = SAFE_SIZE, device: str = SAFE_DEVICE,
                       timeout_seconds: float = SAFE_TIMEOUT_SECONDS) -> bool:
  """Independently timed, process-isolated device-alive check (lead-invokable)."""
  res = run_isolated(_tiny_add_is_alive, args=(size, device), timeout_seconds=timeout_seconds)
  return bool(res.status == "passed" and res.result is True)


# --- Compile-only tiny-safe-op preparation (no dispatch) ---------------------

def compile_tiny_safe_add(*, size: int = SAFE_SIZE, device: str = SAFE_DEVICE) -> tuple[Any, dict[str, Any]]:
  """Compile ``a + b`` to a real PROGRAM UOp + minimal passing compile evidence.

  This is compile-only: it produces the PROGRAM's final binary but never
  dispatches.  The runtime handle is constructed later, INSIDE the spawned child,
  by :func:`build_tiny_add_bundle` (which re-runs this compile in-child).
  """
  from tinygrad import Tensor, dtypes
  from tinygrad.engine.realize import compile_linear
  from tinygrad.helpers import Context
  from tinygrad.uop.ops import Ops, ProgramInfo
  with Context(DEV=device):
    a = Tensor.empty(size, dtype=dtypes.float32)
    b = Tensor.empty(size, dtype=dtypes.float32)
    compiled = compile_linear((a + b).schedule_linear())
  program = next((u for u in compiled.toposort() if u.op is Ops.PROGRAM and isinstance(u.arg, ProgramInfo)
                  and len(u.src) >= 5 and u.src[4].op is Ops.BINARY), None)
  if program is None: raise RuntimeError("tiny safe add did not lower to a source-bound binary PROGRAM")
  binary = next(u.arg for u in program.src if u.op is Ops.BINARY and isinstance(u.arg, bytes))
  evidence = {"schema": f"{SCHEMA}.compile", "passed": True, "synthetic": False,
              "binary_sha256": hashlib.sha256(binary).hexdigest()}
  return program, evidence


def _work_plan(program: Any, evidence: Mapping[str, Any], size: int, device: str) -> dict[str, Any]:
  info = getattr(program, "arg", None)
  bytes_each = size * 4  # float32
  return {"schema": f"{SCHEMA}.plan", "device": device, "op": "elementwise_add_f32", "elements": size,
          "argument_order": list(SAFE_ARGUMENT_ORDER),
          "buffers_bytes": {"a": bytes_each, "b": bytes_each, "output": bytes_each},
          "kernel": getattr(info, "function_name", None),
          "global_size": list(getattr(info, "global_size", ()) or ()),
          "local_size": list(getattr(info, "local_size", None) or ()) or None,
          "binary_sha256": evidence.get("binary_sha256"),
          "guard_prefix_bytes": GuardPolicy().prefix_bytes, "guard_suffix_bytes": GuardPolicy().suffix_bytes,
          "dispatch": "none (compile/audit-only)"}


def _format_plan(plan: Mapping[str, Any]) -> str:
  return ("host safety canary -- bounded GPU work plan (AUDIT ONLY, NO DISPATCH):\n"
          f"  device        : {plan['device']}\n"
          f"  operation     : {plan['op']} on {plan['elements']} elements\n"
          f"  kernel        : {plan['kernel']}\n"
          f"  argument order: {plan['argument_order']}\n"
          f"  buffer bytes  : {plan['buffers_bytes']} (+ {plan['guard_prefix_bytes']}/{plan['guard_suffix_bytes']} guard bytes)\n"
          f"  launch        : global={plan['global_size']} local={plan['local_size']}\n"
          f"  binary sha256 : {plan['binary_sha256']}\n"
          f"  dispatch      : {plan['dispatch']}")


# --- Typed terminal record ---------------------------------------------------

@dataclass(frozen=True)
class HostSafetyRecord:
  """One truthful terminal record for a canary invocation (never auto-retried)."""
  mode: str  # "audit" | "execute"
  dispatch_state: str
  executed: bool
  passed: bool
  plan: Mapping[str, Any]
  result: Mapping[str, Any] | None = None
  errors: tuple[str, ...] = ()
  schema: str = SCHEMA

  def to_dict(self) -> dict[str, Any]:
    return {"schema": self.schema, "mode": self.mode, "dispatch_state": self.dispatch_state,
            "executed": self.executed, "passed": self.passed, "plan": dict(self.plan),
            "result": dict(self.result) if self.result is not None else None, "errors": list(self.errors)}


# --- The host safety-boundary harness ----------------------------------------

def run_host_safety_canary(*, execute: bool = False, size: int = SAFE_SIZE, device: str = SAFE_DEVICE,
                           compile_fn: Callable[[], tuple[Any, Mapping[str, Any]]] | None = None,
                           builder: Callable[[], Any] | None = None,
                           runner: Callable[..., IsolatedExecutionResult] | None = None,
                           health_probe: Callable[[], bool] | None = None,
                           timeout_seconds: float | None = None, terminate_grace_seconds: float = 0.25,
                           health_timeout_seconds: float = SAFE_TIMEOUT_SECONDS,
                           persist: Callable[[dict[str, Any]], None] | None = None,
                           printer: Callable[[str], None] | None = print) -> HostSafetyRecord:
  """Run the host safety boundary for the tiny-safe-op.

  ``execute=False`` (default): compile/audit-only.  Prints the exact bounded GPU
  work plan and returns a ``not_attempted`` record; NOTHING dispatches and no
  runtime handle is constructed.  ``execute=True`` (the LEAD invokes this): runs
  the tiny op through :func:`run_isolated_guarded_execution` behind the process
  boundary, then the INDEPENDENT health probe, and returns one typed terminal
  record.  Any timeout/device loss/guard/numeric fault STOPS -- no reset/retry.

  ``compile_fn``/``builder``/``runner``/``health_probe`` are injectable seams so
  the fake-runtime tests can drive every path without a real GPU.
  """
  compile_fn = compile_fn or (lambda: compile_tiny_safe_add(size=size, device=device))
  program, evidence = compile_fn()
  plan = _work_plan(program, evidence, size, device)

  if not execute:
    if printer is not None: printer(_format_plan(plan))
    return HostSafetyRecord("audit", dispatch_state("not_attempted"), executed=False, passed=False, plan=plan)

  runner = runner or run_isolated_guarded_execution
  probe = health_probe if health_probe is not None else make_tiny_health_probe(size=size, device=device)
  if builder is None:
    # A PICKLABLE spec: the child RECOMPILES the program + inits the device fresh
    # under spawn from these descriptors (the parent's compiled ``program`` above
    # is a UOp, not picklable, and is used only for the audit plan).
    builder = make_tinygrad_bundle_builder(build=build_tiny_add_bundle, size=size, device=device,
                                           argument_order=SAFE_ARGUMENT_ORDER)
  request = _safe_request(size, identity={"canary": SCHEMA, "op": "tiny_safe_add", "size": size, "device": device})
  out = runner(builder=builder, request=request, health_probe=probe,
               timeout_seconds=timeout_seconds if timeout_seconds is not None else request.policy.timeout_seconds,
               terminate_grace_seconds=terminate_grace_seconds, health_timeout_seconds=health_timeout_seconds,
               persist=persist)
  return HostSafetyRecord("execute", out.dispatch_state, executed=True, passed=out.passed, plan=plan,
                          result=out.to_dict(), errors=out.errors)


__all__ = ["SCHEMA", "SAFE_SIZE", "SAFE_DEVICE", "SAFE_ARGUMENT_ORDER", "HostSafetyRecord", "TinyHealthProbe",
           "make_tiny_health_probe", "tiny_device_health", "compile_tiny_safe_add", "build_tiny_add_bundle",
           "run_host_safety_canary"]
