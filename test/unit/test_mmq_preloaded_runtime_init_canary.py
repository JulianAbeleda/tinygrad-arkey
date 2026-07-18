"""CPU-only orchestration contracts for the no-target-dispatch canary."""
from __future__ import annotations

from pathlib import Path

from tinygrad.runtime.process_isolated import IsolatedResult
from extra.qk import mmq_preloaded_runtime_init_canary as canary


def _compile(temp_dir):
  path = Path(temp_dir) / "target.pkl"
  path.write_bytes(b"fake-program")
  return str(path), {"binary_sha256": "ab" * 32, "no_fallback": True}


def _child_pass():
  return {"passed": True, "target_runtime_constructed": True, "target_runtime_called": False}


def _runner_pass(callback, *, args=(), timeout_seconds=0, start_method=None, **kwargs):
  assert callback is canary._run_runtime_init_worker
  assert Path(args[0]).name == "target.pkl"
  assert start_method == "spawn" and timeout_seconds > 0
  return IsolatedResult("passed", _child_pass())


def test_fault_parser_deduplicates():
  assert canary.parse_kernel_faults("GPU reset\nGPU reset\nquiet") == ["GPU reset"]
  assert canary.parse_kernel_faults("quiet") == []


def test_success_compiles_parent_and_is_diagnostic():
  health_calls: list[int] = []
  out = canary.run_runtime_init_canary(
    compile_fn=_compile, timeout_seconds=1, runner=_runner_pass,
    fault_reader=lambda _: "", health_probe=lambda: health_calls.append(1) or True,
  )
  assert out["status"] == "PASS" and out["passed"] is True
  assert out["compile"]["no_fallback"] is True
  assert out["diagnostic_only"] is True and out["promotion_eligible"] is False
  assert out["no_target_dispatch"] is True and health_calls == [1]


def test_child_timeout_fails_closed_without_health_retry():
  calls: list[int] = []
  def runner(*args, **kwargs): return IsolatedResult("timed_out", error="deadline", timed_out=True)
  out = canary.run_runtime_init_canary(
    compile_fn=_compile, timeout_seconds=1, runner=runner,
    fault_reader=lambda _: "", health_probe=lambda: calls.append(1) or True,
  )
  assert out["status"] == "BLOCKED" and out["passed"] is False
  assert "deadline" in out["exact_blocker"] and calls == []


def test_kernel_fault_blocks_before_health_probe():
  calls: list[int] = []
  out = canary.run_runtime_init_canary(
    compile_fn=_compile, runner=_runner_pass,
    fault_reader=lambda _: "amdgpu: page fault", health_probe=lambda: calls.append(1) or True,
  )
  assert out["status"] == "BLOCKED" and out["passed"] is False
  assert out["kernel_faults"] and calls == []


def test_compile_failure_fails_closed_without_child():
  calls: list[int] = []
  def bad_compile(_): raise RuntimeError("compile blocked")
  def runner(*args, **kwargs): calls.append(1); raise AssertionError("must not run")
  out = canary.run_runtime_init_canary(compile_fn=bad_compile, runner=runner)
  assert out["status"] == "BLOCKED" and "compile blocked" in out["exact_blocker"]
  assert calls == []
