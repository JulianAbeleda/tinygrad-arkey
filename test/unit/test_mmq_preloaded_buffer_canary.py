"""CPU-only contract tests for the large-preload lifecycle canary."""
from __future__ import annotations

from tinygrad.runtime.process_isolated import IsolatedResult
from extra.qk import mmq_preloaded_buffer_canary as canary


def _child_pass() -> dict:
  return {"passed": True, "tiny_add_passed": True, "roundtrip_passed": True}


def _runner_pass(callback, *, args=(), timeout_seconds=0, start_method=None, **kwargs):
  assert callback is canary._run_large_preload_worker
  assert start_method == "spawn" and timeout_seconds > 0
  return IsolatedResult("passed", _child_pass())


def test_payload_is_deterministic_and_q4_capacity_matches_target():
  left = canary._deterministic_payload(32)
  right = canary._deterministic_payload(32)
  assert left.dtype.name == "uint8" and left.tobytes() == right.tobytes()
  assert canary.TARGET_Q4_BYTES == 17_408 * 20 * 36 * 4


def test_fault_parser_deduplicates_health_markers():
  faults = canary.parse_kernel_faults("quiet\nGPU reset succeeded\nGPU reset succeeded\nordinary warning")
  assert faults == ["GPU reset succeeded"]
  assert canary.parse_kernel_faults("quiet") == []


def test_success_is_diagnostic_and_calls_health_once():
  calls: list[int] = []
  out = canary.run_large_preload_canary(
    nbytes=32, tiny_size=8, runner=_runner_pass, fault_reader=lambda _: "",
    health_probe=lambda: calls.append(1) or True,
  )
  assert out["status"] == "PASS" and out["passed"] is True
  assert out["diagnostic_only"] is True and out["promotion_eligible"] is False
  assert out["no_target_dispatch"] is True and calls == [1]


def test_child_timeout_fails_closed_without_health_retry():
  calls: list[int] = []
  def runner(*args, **kwargs): return IsolatedResult("timed_out", error="deadline", timed_out=True)
  out = canary.run_large_preload_canary(
    nbytes=32, runner=runner, fault_reader=lambda _: "", health_probe=lambda: calls.append(1) or True,
  )
  assert out["status"] == "BLOCKED" and out["passed"] is False
  assert "deadline" in out["exact_blocker"] and calls == []


def test_kernel_fault_fails_closed_before_health_probe():
  calls: list[int] = []
  out = canary.run_large_preload_canary(
    nbytes=32, runner=_runner_pass, fault_reader=lambda _: "amdgpu: GPU reset succeeded",
    health_probe=lambda: calls.append(1) or True,
  )
  assert out["status"] == "BLOCKED" and out["passed"] is False
  assert out["kernel_faults"] and calls == []


def test_invalid_request_never_invokes_runner():
  calls: list[int] = []
  def runner(*args, **kwargs): calls.append(1); raise AssertionError("must not run")
  try: canary.run_large_preload_canary(nbytes=0, runner=runner)
  except ValueError: pass
  else: raise AssertionError("invalid nbytes must raise")
  assert calls == []
