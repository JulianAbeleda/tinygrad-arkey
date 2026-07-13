"""C3 fake-runtime matrix for the one isolated guarded executor.

Every test drives :func:`run_isolated_guarded_execution` with a FAKE runtime
bundle injected at the child-only construction seam (the ``builder``).  No real
Device is constructed and nothing dispatches to hardware.  The matrix covers
success, ABI error, guard corruption, buffer mutation, numerical error, in-child
device fault, timeout, child crash, no-result, cleanup failure (close and
release), independent-health loss, and pre-launch validation.  In every case the
PARENT survives and reports a truthful, typed terminal dispatch state.
"""
from __future__ import annotations

import inspect
import os
import signal
import time

import numpy as np

from extra.qk.prefill.guarded_execution import GuardedBuffer, GuardedExecutionHooks, GuardPolicy
from extra.qk.prefill import isolated_guarded_executor as ige
from extra.qk.prefill.isolated_guarded_executor import (ExecutableBundle, ExecutionRequest,
                                                        make_tinygrad_bundle_builder,
                                                        run_isolated_guarded_execution)
from tinygrad.runtime.bridge import ExecutableHandle


class FakeGpu:
  """A fake runtime + guarded hooks.  ``mode`` selects the failure to inject."""
  def __init__(self, mode: str): self.mode = mode; self.data = {}; self.faulted = False; self.closed = False

  # --- executable interface (the single dispatch target + cleanup) ---
  def close(self):
    self.closed = True
    if self.mode == "cleanup_close": raise RuntimeError("executable close failed on purpose")

  # --- guarded hooks ---
  def allocate(self, name, value, policy):
    self.data[name] = np.zeros_like(value)
    return GuardedBuffer(name, name, policy.prefix_bytes, policy.suffix_bytes)
  def upload(self, buffer, value): self.data[buffer.name][...] = value
  def readback(self, buffer): return self.data[buffer.name].copy()
  def guards_intact(self, buffer): return not (self.mode == "guard_corruption" and buffer.name == "output")
  def health(self): return not self.faulted
  def release(self, buffer):
    if self.mode == "cleanup_release" and buffer.name == "output": raise RuntimeError("buffer release failed on purpose")

  def dispatch(self, executable, buffers):
    if self.mode == "abi": raise ValueError("compiled ABI buffer is missing: 'output'")
    if self.mode == "timeout": time.sleep(5.0)
    if self.mode == "crash": os.kill(os.getpid(), signal.SIGKILL)
    if self.mode == "no_result": os._exit(0)
    if self.mode == "numerical": self.data["output"][...] = self.data["a"] * self.data["b"] + 999
    else: self.data["output"][...] = self.data["a"] + self.data["b"]
    if self.mode == "mutation": self.data["a"][...] = self.data["a"] + 7  # corrupt an input buffer
    if self.mode == "device_fault": self.faulted = True  # health goes bad after dispatch
    return 0.001


def _builder(mode: str):
  # Invoked ONLY inside the isolated child: this is the runtime-construction seam.
  def build() -> ExecutableBundle:
    fake = FakeGpu(mode)
    hooks = GuardedExecutionHooks(fake.allocate, fake.upload, fake.readback, fake.guards_intact,
                                  fake.dispatch, fake.health, fake.release)
    return ExecutableBundle(fake, hooks)
  return build


def _request():
  return ExecutionRequest(inputs={"a": np.array([1, 2], dtype=np.float16), "b": np.array([3, 4], dtype=np.float16)},
                          reference=np.array([4, 6], dtype=np.float32), policy=GuardPolicy(), identity={"case": "fake"})


def _run(mode: str, *, health_probe=None, timeout_seconds=None, persist=None):
  return run_isolated_guarded_execution(builder=_builder(mode), request=_request(), health_probe=health_probe,
                                        timeout_seconds=timeout_seconds, persist=persist)


# --- success + cleanup on the happy path -------------------------------------

def test_success_completes_with_independent_health_and_clean_cleanup():
  out = _run("success", health_probe=lambda: True)
  assert out.dispatch_state == "completed" and out.passed is True
  assert out.health_after is True and out.guarded["passed"] is True
  assert out.guarded["guards_intact"] is True and out.guarded["inputs_unchanged"] is True
  assert out.health_terminal["result"] is True


# --- the failure-mode matrix (each STOPS at one truthful terminal state) -----

def test_abi_error_is_a_controlled_failure_after_an_attempt():
  out = _run("abi")
  assert out.dispatch_state == "failed" and out.passed is False
  assert out.guarded is not None and out.guarded["dispatch_performed"] is True
  assert any("ABI buffer is missing" in e for e in out.errors)


def test_guard_corruption_fails_closed():
  out = _run("guard_corruption")
  assert out.dispatch_state == "failed" and out.passed is False
  assert out.guarded["guards_intact"] is False


def test_input_buffer_mutation_is_detected():
  out = _run("mutation")
  assert out.dispatch_state == "failed" and out.guarded["inputs_unchanged"] is False


def test_numerical_error_fails_full_output_comparison():
  out = _run("numerical")
  assert out.dispatch_state == "failed" and out.guarded["numerics_passed"] is False


def test_in_child_device_fault_becomes_device_lost():
  out = _run("device_fault")
  assert out.dispatch_state == "device_lost" and out.passed is False
  assert out.guarded["device_fault"] is True


def test_hard_timeout_is_typed_and_the_parent_survives():
  out = _run("timeout", timeout_seconds=0.4)
  assert out.dispatch_state == "timed_out" and out.passed is False
  assert out.terminal["timed_out"] is True and out.guarded is None


def test_child_crash_is_reported_as_device_lost_without_a_result():
  out = _run("crash")
  assert out.dispatch_state == "device_lost" and out.passed is False
  assert out.terminal["produced_result"] is False


def test_child_no_result_is_reported_as_device_lost():
  out = _run("no_result")
  assert out.dispatch_state == "device_lost" and out.passed is False
  assert out.terminal["produced_result"] is False


def test_cleanup_close_failure_is_recorded_and_fails_the_run():
  out = _run("cleanup_close")
  assert out.dispatch_state == "failed" and out.passed is False
  assert any("close failed" in e for e in out.errors)


def test_cleanup_release_failure_is_recorded_and_fails_the_run():
  out = _run("cleanup_release")
  assert out.dispatch_state == "failed" and out.passed is False
  assert any("release failed" in e for e in out.errors)


def test_independent_health_loss_downgrades_a_success_to_device_lost():
  out = _run("success", health_probe=lambda: False)
  assert out.dispatch_state == "device_lost" and out.passed is False
  assert out.health_after is False and any("health check failed" in e for e in out.errors)


# --- pre-launch validation + persistence -------------------------------------

def test_empty_inputs_never_launch_and_report_not_attempted():
  bad = ExecutionRequest(inputs={}, reference=np.array([1.0]))
  out = run_isolated_guarded_execution(builder=_builder("success"), request=bad)
  assert out.dispatch_state == "not_attempted" and out.passed is False
  assert out.terminal["child_status"] == "not_launched"


def test_failures_are_persisted_and_success_is_not():
  persisted = []
  ok = _run("success", health_probe=lambda: True, persist=persisted.append)
  assert ok.passed is True and persisted == []
  bad = _run("numerical", persist=persisted.append)
  assert bad.passed is False and len(persisted) == 1
  assert persisted[0]["dispatch_state"] == "failed" and persisted[0]["schema"] == ige.SCHEMA


# --- P0-2 / P1-2 verification by construction ---------------------------------

def test_parent_path_constructs_no_runtime_p0_2():
  # The PARENT orchestration must never construct a live runtime/Device/buffer.
  parent_src = "".join(inspect.getsource(fn) for fn in
                       (run_isolated_guarded_execution, ige._classify, ige._terminal))
  for forbidden in ("Device[", ".runtime(", "get_runtime", "prepare_executable", "Buffer("):
    assert forbidden not in parent_src, f"parent path must not reference {forbidden!r} (P0-2)"
  # Runtime construction lives ONLY in the child-invoked builder.
  assert "prepare_executable" in inspect.getsource(make_tinygrad_bundle_builder)


def test_executable_call_is_identical_to_dispatch_p1_2():
  src = inspect.getsource(ExecutableHandle.__call__)
  assert "self.dispatch(" in src and "self.runtime(" not in src, "__call__ must delegate to dispatch (P1-2)"
