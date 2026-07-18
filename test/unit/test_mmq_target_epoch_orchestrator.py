"""CPU-only tests for the fail-closed, process-per-epoch target diagnostic.

The real worker compiles/dispatches on AMD.  These tests replace every hardware
boundary with an injected fake and exercise only the parent-side state machine.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from extra.qk.mmq_target_epoch_orchestrator import orchestrate_epoch_sweep, parse_kernel_faults


def _compile_artifact(temp_dir: str | Path) -> tuple[str, dict]:
  path = Path(temp_dir) / "program.pkl"
  path.write_bytes(b"fake-program")
  return str(path), {"binary_sha256": "ab" * 32, "passed": True}


def _health_sequence(*values: bool) -> tuple[Callable[[], bool], list[bool]]:
  remaining, calls = list(values), []

  def probe() -> bool:
    assert remaining, "health probe called more often than expected"
    value = remaining.pop(0)
    calls.append(value)
    return value

  return probe, calls


def _epoch_runner(*, failing_epoch: int | None = None, calls: list[int] | None = None):
  def run(artifact_path: str, output_path: str, epoch: int) -> dict:
    assert Path(artifact_path).read_bytes() == b"fake-program"
    if calls is not None: calls.append(epoch)
    if epoch == failing_epoch:
      return {"passed": False, "epoch": epoch, "status": "NUMERICAL_MISMATCH"}
    # Different constant per epoch makes the host-side FP32 sum observable.
    partial = np.full((2, 3), epoch + 1, dtype=np.float32)
    np.save(output_path, partial)
    return {"passed": True, "epoch": epoch, "output_sha256": f"epoch-{epoch}"}

  return run


def _no_faults(_since: float) -> str:
  return ""


def _assert_diagnostic_only(out: dict) -> None:
  assert out["promotion_eligible"] is False
  assert out["production_dispatch_changed"] is False
  assert out["diagnostic_only"] is True


def test_parse_kernel_faults_finds_only_relevant_gpu_health_markers():
  text = """
    harmless amdgpu informational row
    amdgpu 0000:03:00.0: amdgpu: [gfxhub] page fault
    amdgpu: MES failed to remove queue
    amdgpu: GPU reset(42) succeeded!
    unrelated application timeout
  """
  faults = parse_kernel_faults(text)
  assert len(faults) == 3
  assert any("page fault" in row for row in faults)
  assert any("MES failed" in row for row in faults)
  assert any("GPU reset" in row for row in faults)
  assert parse_kernel_faults("all quiet\nordinary compiler warning") == []


def test_all_pass_runs_epochs_in_order_aggregates_fp32_and_is_never_promotable():
  calls: list[int] = []
  health, health_calls = _health_sequence(True, True, True, True)  # preflight + after each epoch
  out = orchestrate_epoch_sweep(
    epoch_indices=[0, 1, 2],
    compile_artifact=_compile_artifact,
    epoch_runner=_epoch_runner(calls=calls),
    health_probe=health,
    fault_reader=_no_faults,
    expected_partial_shape=(2, 3),
  )

  assert out["passed"] is True
  assert calls == [0, 1, 2]
  assert health_calls == [True, True, True, True]
  assert out["completed_epochs"] == [0, 1, 2]
  assert out["aggregate_shape"] == [2, 3]
  # 1 + 2 + 3 in every cell; expose a small deterministic witness in evidence.
  assert out["aggregate_sum"] == 36.0
  assert len(out["aggregate_sha256"]) == 64
  _assert_diagnostic_only(out)


def test_epoch_failure_stops_before_later_epochs_and_keeps_only_verified_partials():
  calls: list[int] = []
  health, _ = _health_sequence(True, True)  # preflight and epoch zero
  out = orchestrate_epoch_sweep(
    epoch_indices=[0, 1, 2],
    compile_artifact=_compile_artifact,
    epoch_runner=_epoch_runner(failing_epoch=1, calls=calls),
    health_probe=health,
    fault_reader=_no_faults,
    expected_partial_shape=(2, 3),
  )

  assert out["passed"] is False
  assert calls == [0, 1]
  assert out["completed_epochs"] == [0]
  assert out["failed_epoch"] == 1
  assert "epoch" in out["stop_reason"].lower()
  _assert_diagnostic_only(out)


def test_kernel_fault_invalidates_current_epoch_and_stops_immediately():
  calls: list[int] = []
  health, health_calls = _health_sequence(True)  # only preflight; fault wins before post-canary

  def fault_after_first(_since: float) -> str:
    return "amdgpu: sq_intr: inst access fault, GPU reset required"

  out = orchestrate_epoch_sweep(
    epoch_indices=[0, 1],
    compile_artifact=_compile_artifact,
    epoch_runner=_epoch_runner(calls=calls),
    health_probe=health,
    fault_reader=fault_after_first,
    expected_partial_shape=(2, 3),
  )

  assert out["passed"] is False
  assert calls == [0]
  assert health_calls == [True]
  assert out["completed_epochs"] == []
  assert out["failed_epoch"] == 0
  assert out["kernel_faults"] and "sq_intr" in out["kernel_faults"][0]
  assert "kernel" in out["stop_reason"].lower()
  _assert_diagnostic_only(out)


def test_health_canary_failure_invalidates_current_epoch_and_stops_immediately():
  calls: list[int] = []
  health, health_calls = _health_sequence(True, True, False)
  out = orchestrate_epoch_sweep(
    epoch_indices=[0, 1, 2],
    compile_artifact=_compile_artifact,
    epoch_runner=_epoch_runner(calls=calls),
    health_probe=health,
    fault_reader=_no_faults,
    expected_partial_shape=(2, 3),
  )

  assert out["passed"] is False
  assert calls == [0, 1]
  assert health_calls == [True, True, False]
  assert out["completed_epochs"] == [0]
  assert out["failed_epoch"] == 1
  assert "health" in out["stop_reason"].lower()
  _assert_diagnostic_only(out)
