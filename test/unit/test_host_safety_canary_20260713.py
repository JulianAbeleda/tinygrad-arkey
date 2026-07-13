"""C4 synthetic (fake-runtime) matrix for the host safety-boundary canary.

Every test drives :func:`run_host_safety_canary` through its injectable seams so
NOTHING touches a real GPU: the compile step is faked, the ExecutableBundle is
the C3 fake runtime injected at the child-only ``builder`` seam, and the health
probe is a module-level (picklable) function.  No real ``Device`` is constructed and nothing
dispatches to hardware.  The matrix proves: audit mode never dispatches and
prints the bounded plan; a normal tiny op completes and does NOT false-trigger
the timeout; a synthetic child hang leaves the PARENT alive with a typed
``timed_out`` and cleanup run; and post-run independent health loss downgrades a
success to ``device_lost`` (P0-7).
"""
from __future__ import annotations

import os
import signal
import time
import types

import numpy as np

from extra.qk.prefill.guarded_execution import GuardedBuffer, GuardedExecutionHooks, GuardPolicy
from extra.qk.prefill import host_safety_canary as hsc
from extra.qk.prefill.host_safety_canary import HostSafetyRecord, run_host_safety_canary
from extra.qk.prefill.isolated_guarded_executor import BundleSpec, ExecutableBundle, make_tinygrad_bundle_builder


# --- the C3 fake runtime, reused at the child-only builder seam ---------------
# C3.5: the executor spawns its children, so the build fn and health probes are
# MODULE-LEVEL (picklable) -- closures/lambdas cannot cross the spawn boundary.

class FakeGpu:
  def __init__(self, mode: str): self.mode = mode; self.data = {}; self.faulted = False; self.closed = False
  def close(self): self.closed = True
  def allocate(self, name, value, policy):
    self.data[name] = np.zeros_like(value)
    return GuardedBuffer(name, name, policy.prefix_bytes, policy.suffix_bytes)
  def upload(self, buffer, value): self.data[buffer.name][...] = value
  def readback(self, buffer): return self.data[buffer.name].copy()
  def guards_intact(self, buffer): return True
  def health(self): return not self.faulted
  def release(self, buffer): pass
  def dispatch(self, executable, buffers):
    if self.mode == "timeout": time.sleep(5.0)
    if self.mode == "crash": os.kill(os.getpid(), signal.SIGKILL)
    if self.mode == "numerical": self.data["output"][...] = self.data["a"] + self.data["b"] + 999
    else: self.data["output"][...] = self.data["a"] + self.data["b"]
    return 0.001


def build_fake_bundle(*, mode: str) -> ExecutableBundle:
  # MODULE-LEVEL runtime-construction seam, invoked ONLY inside the isolated child.
  fake = FakeGpu(mode)
  hooks = GuardedExecutionHooks(fake.allocate, fake.upload, fake.readback, fake.guards_intact,
                                fake.dispatch, fake.health, fake.release)
  return ExecutableBundle(fake, hooks)


def _fake_builder(mode: str) -> BundleSpec:
  return make_tinygrad_bundle_builder(build=build_fake_bundle, mode=mode)


def _healthy() -> bool: return True
def _unhealthy() -> bool: return False


def _fake_compile():
  # A stand-in for compile_tiny_safe_add: NO real Device, NO real compile.
  arg = types.SimpleNamespace(function_name="E_256_add", global_size=(1, 1, 1), local_size=(256, 1, 1))
  program = types.SimpleNamespace(arg=arg)
  return program, {"schema": "fake", "passed": True, "binary_sha256": "ab" * 32, "synthetic": True}


def _run(mode: str, *, execute=True, health_probe=None, timeout_seconds=None, size=8, **kw):
  return run_host_safety_canary(execute=execute, size=size, compile_fn=_fake_compile,
                                builder=_fake_builder(mode), health_probe=health_probe,
                                timeout_seconds=timeout_seconds, printer=None, **kw)


# --- default execute=False audit mode: compile/audit-only, NEVER dispatches ---

def test_audit_mode_prints_plan_and_never_dispatches():
  printed: list[str] = []
  def _no_dispatch(**kwargs): raise AssertionError("audit mode must never invoke the executor")
  out = run_host_safety_canary(execute=False, size=8, compile_fn=_fake_compile,
                               runner=_no_dispatch, builder=_fake_builder("boom"), printer=printed.append)
  assert isinstance(out, HostSafetyRecord)
  assert out.mode == "audit" and out.dispatch_state == "not_attempted"
  assert out.executed is False and out.passed is False
  assert out.plan["op"] == "elementwise_add_f32" and out.plan["argument_order"] == ["output", "a", "b"]
  assert out.plan["dispatch"].startswith("none") and out.plan["binary_sha256"] == "ab" * 32
  assert printed and "NO DISPATCH" in printed[0]


# --- execute=True through the REAL isolated executor with the FAKE runtime -----

def test_normal_tiny_op_completes_and_does_not_false_trigger_timeout():
  out = _run("success", health_probe=_healthy, timeout_seconds=10.0)
  assert out.mode == "execute" and out.executed is True
  assert out.dispatch_state == "completed" and out.passed is True
  assert out.result["health_after"] is True and out.result["guarded"]["passed"] is True


def test_synthetic_child_hang_times_out_and_parent_survives_with_cleanup():
  out = _run("timeout", timeout_seconds=0.4)
  # The PARENT survives; the terminal is typed timed_out and no guarded result
  # leaked out (the hung child's cleanup boundary was enforced by termination).
  assert out.dispatch_state == "timed_out" and out.passed is False
  assert out.result["terminal"]["timed_out"] is True and out.result["guarded"] is None


def test_post_run_health_loss_downgrades_success_to_device_lost():
  out = _run("success", health_probe=_unhealthy)
  assert out.dispatch_state == "device_lost" and out.passed is False
  assert out.result["health_after"] is False
  assert any("health check failed" in e for e in out.errors)


def test_numerical_fault_fails_closed_and_persists():
  persisted: list[dict] = []
  out = _run("numerical", health_probe=_healthy, persist=persisted.append)
  assert out.dispatch_state == "failed" and out.passed is False
  assert out.result["guarded"]["numerics_passed"] is False
  assert len(persisted) == 1 and persisted[0]["passed"] is False


def test_child_crash_is_reported_as_device_lost():
  out = _run("crash", health_probe=_healthy)
  assert out.dispatch_state == "device_lost" and out.passed is False
  assert out.result["terminal"]["produced_result"] is False


# --- the independent health probe (P0-7) --------------------------------------

def test_health_probe_is_an_independent_picklable_callable():
  import pickle
  probe = hsc.make_tiny_health_probe(size=8)
  assert callable(probe)  # NOT invoked here: calling it would run a real tiny GPU op.
  # It crosses the spawn boundary into its own child, so it MUST pickle.
  restored = pickle.loads(pickle.dumps(probe))
  assert isinstance(restored, hsc.TinyHealthProbe) and restored.size == 8


def test_default_builder_and_probe_are_picklable_for_spawn():
  # The production execute=True path hands the executor a picklable spec + probe
  # (a UOp/runtime is never pickled: build_tiny_add_bundle recompiles in-child).
  import pickle
  spec = make_tinygrad_bundle_builder(build=hsc.build_tiny_add_bundle, size=8, device=hsc.SAFE_DEVICE,
                                      argument_order=hsc.SAFE_ARGUMENT_ORDER)
  pickle.dumps(spec)  # no real GPU touched: we only serialize the descriptor
  pickle.dumps(hsc.make_tiny_health_probe(size=8))


def test_harness_uses_the_isolated_executor_and_never_builds_a_runtime_in_the_parent():
  # The audit/parent control flow references no live Device/runtime/Buffer; the
  # only runtime construction is delegated to compile_fn/builder (child seam).
  import inspect
  src = inspect.getsource(run_host_safety_canary) + inspect.getsource(hsc._work_plan)
  for forbidden in ("Device[", ".runtime(", "get_runtime", "prepare_executable", "Buffer("):
    assert forbidden not in src, f"safety-canary parent path must not reference {forbidden!r}"
