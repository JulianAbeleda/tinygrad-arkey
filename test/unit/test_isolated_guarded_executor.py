"""C3 fake-runtime matrix for the one isolated guarded executor.

Every test drives :func:`run_isolated_guarded_execution` with a FAKE runtime
bundle injected at the child-only construction seam (the ``builder``).  No real
Device is constructed and nothing dispatches to hardware.  The matrix covers
success, ABI error, guard corruption, buffer mutation, numerical error, in-child
device fault, timeout, child crash, no-result, cleanup failure (close and
release), independent-health loss, and pre-launch validation.  In every case the
PARENT survives and reports a truthful, typed terminal dispatch state.

C3.5: the executor now runs its children under SPAWN, so the injected builder is
a picklable :class:`BundleSpec` over a MODULE-LEVEL build function
(``build_fake_bundle``) and the health probes are MODULE-LEVEL functions -- never
closures/lambdas, which cannot cross the spawn boundary.  spawn re-imports this
module in the child, so every child target lives at module scope.
"""
from __future__ import annotations

import inspect
import os
import pickle
import signal
import time

import numpy as np

from extra.qk.prefill.guarded_execution import GuardedBuffer, GuardedExecutionHooks, GuardPolicy
from extra.qk.prefill import isolated_guarded_executor as ige
from extra.qk.prefill.isolated_guarded_executor import (BundleSpec, ExecutableBundle, ExecutionRequest,
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


def build_fake_bundle(*, mode: str) -> ExecutableBundle:
  # MODULE-LEVEL so it survives spawn pickling; invoked ONLY inside the isolated
  # child (the runtime-construction seam).  The FakeGpu + hooks are built here.
  fake = FakeGpu(mode)
  hooks = GuardedExecutionHooks(fake.allocate, fake.upload, fake.readback, fake.guards_intact,
                                fake.dispatch, fake.health, fake.release)
  return ExecutableBundle(fake, hooks)


def _construction_failure(*, mode: str) -> ExecutableBundle:
  # A SOFTWARE runtime construction/init failure (stand-in for ImportError /
  # OSError bad-fd from a fork()ed GPU init): raised in-child before any bundle.
  raise ImportError(f"fake runtime could not initialize ({mode})")


def _builder(mode: str) -> BundleSpec:
  return make_tinygrad_bundle_builder(build=build_fake_bundle, mode=mode)


def _healthy() -> bool: return True
def _unhealthy() -> bool: return False


def _request():
  return ExecutionRequest(inputs={"a": np.array([1, 2], dtype=np.float16), "b": np.array([3, 4], dtype=np.float16)},
                          reference=np.array([4, 6], dtype=np.float32), policy=GuardPolicy(), identity={"case": "fake"})


def _run(mode: str, *, health_probe=None, timeout_seconds=None, persist=None):
  return run_isolated_guarded_execution(builder=_builder(mode), request=_request(), health_probe=health_probe,
                                        timeout_seconds=timeout_seconds, persist=persist)


# --- success + cleanup on the happy path -------------------------------------

def test_success_completes_with_independent_health_and_clean_cleanup():
  out = _run("success", health_probe=_healthy)
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
  out = _run("success", health_probe=_unhealthy)
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
  ok = _run("success", health_probe=_healthy, persist=persisted.append)
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
  # Runtime construction lives ONLY in the child-invoked construction helper, and
  # the parent's builder factory just wraps a picklable spec (no construction).
  assert "prepare_executable" in inspect.getsource(ige.build_tinygrad_bundle)
  assert "prepare_executable" not in inspect.getsource(make_tinygrad_bundle_builder)


def test_executable_call_is_identical_to_dispatch_p1_2():
  src = inspect.getsource(ExecutableHandle.__call__)
  assert "self.dispatch(" in src and "self.runtime(" not in src, "__call__ must delegate to dispatch (P1-2)"


# --- C3.5: spawn/picklable boundary + classifier correctness ------------------

def test_builder_and_request_are_picklable_for_spawn():
  # spawn pickles the child target + args; the injected builder MUST survive it.
  spec = _builder("success")
  assert isinstance(spec, BundleSpec)
  restored = pickle.loads(pickle.dumps(spec))
  bundle = restored()  # the child reconstructs FakeGpu + hooks from the spec alone
  assert isinstance(bundle, ExecutableBundle)
  pickle.dumps(_request())  # the request crosses the boundary too
  pickle.dumps(_healthy)    # module-level probes are picklable; lambdas would not be


def test_make_builder_rejects_a_closure_that_cannot_survive_spawn():
  def local_build(**kw): return build_fake_bundle(**kw)  # a closure: unpicklable
  try:
    make_tinygrad_bundle_builder(build=local_build, mode="success")
    assert False, "a non-module-level build must be rejected"
  except ValueError as exc:
    assert "module-level" in str(exc)


def test_runtime_construction_failure_is_failed_not_device_lost():
  # A SOFTWARE init failure (ImportError/OSError bad-fd) must be typed ``failed``.
  # Even when the equally-broken health probe also fails, it is NOT device loss.
  spec = make_tinygrad_bundle_builder(build=_construction_failure, mode="import")
  out = run_isolated_guarded_execution(builder=spec, request=_request(), health_probe=_unhealthy)
  assert out.dispatch_state == "failed" and out.passed is False
  assert out.guarded is not None and out.guarded.get("construction_failed") is True
  assert any("could not initialize" in e for e in out.errors)
