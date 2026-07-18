"""CPU-only tests for the compile-once metadata storage bisect parent."""
from __future__ import annotations

import hashlib
from pathlib import Path

from tinygrad.runtime.process_isolated import IsolatedResult
from extra.qk import mmq_metadata_storage_bisect as bisect


def _compile_factory(calls: list[str], *, bad_serialized_hash: bool = False):
  def compile_fn(temp_dir):
    calls.append(str(temp_dir))
    path = Path(temp_dir) / "target.pkl"
    path.write_bytes(b"one serialized target program")
    serialized = hashlib.sha256(path.read_bytes()).hexdigest()
    return str(path), {
      "binary_sha256": "ab" * 32,
      "source_sha256": "cd" * 32,
      "serialized_program_sha256": ("ef" * 32) if bad_serialized_hash else serialized,
    }
  return compile_fn


def _child(mode: str, *, passed: bool = True) -> dict:
  return {
    "metadata_storage_mode": mode,
    "target_dispatches": 3,
    "no_fallback": True,
    "q4_epoch_sequence": [0, 0, 0],
    "q8_epoch_sequence": [0, 0, 0],
    "q8_values_epoch_sequence": [0, 0, 0],
    "q8_metadata_epoch_sequence": [0, 1, 2],
    "passed": passed,
    "status": "PASS" if passed else "BLOCKED",
    "comparison": {"status": "pass" if passed else "mismatch", "mismatch_count": 0 if passed else 1},
  }


def _runner_factory(calls: list[tuple], *, failing_mode: str | None = None,
                    wrong_hash_mode: str | None = None, timeout_mode: str | None = None):
  def runner(callback, *, args=(), timeout_seconds=0, start_method=None, **_kwargs):
    program_path, expected_sha, mode, device, runtime_timeout_ms = args
    calls.append((callback, program_path, expected_sha, mode, device, runtime_timeout_ms,
                  timeout_seconds, start_method))
    if mode == timeout_mode:
      return IsolatedResult("timed_out", error="deadline", timed_out=True)
    worker_sha = "00" * 32 if mode == wrong_hash_mode else expected_sha
    return IsolatedResult("passed", {
      "metadata_storage_mode": mode,
      "serialized_program_sha256": worker_sha,
      "child": _child(mode, passed=mode != failing_mode),
    })
  return runner


def test_compile_once_all_modes_use_same_serialized_artifact_and_stay_diagnostic():
  compile_calls: list[str] = []
  worker_calls: list[tuple] = []
  out = bisect.run_metadata_storage_bisect(
    compile_fn=_compile_factory(compile_calls),
    runner=_runner_factory(worker_calls),
    timeout_seconds=7,
    runtime_timeout_ms=1234,
  )

  assert out["status"] == "PASS" and out["passed"] is True
  assert out["diagnostic_complete"] is True and out["same_serialized_artifact"] is True
  assert out["diagnostic_only"] is True and out["promotion_eligible"] is False
  assert out["production_dispatch_changed"] is False and out["no_fallback"] is True
  assert len(compile_calls) == 1 and len(worker_calls) == len(bisect.METADATA_STORAGE_MODES)
  assert [call[3] for call in worker_calls] == list(bisect.METADATA_STORAGE_MODES)
  assert len({call[1] for call in worker_calls}) == 1
  assert len({call[2] for call in worker_calls}) == 1
  assert all(call[0] is bisect._run_metadata_mode_worker and call[4:] == ("AMD", 1234, 7, "spawn")
             for call in worker_calls)
  assert {row["worker_serialized_program_sha256"] for row in out["modes"]} == {
    out["program_identity"]["serialized_program_sha256"]}


def test_numerical_failure_is_preserved_per_mode_and_blocks_aggregate():
  out = bisect.run_metadata_storage_bisect(
    compile_fn=_compile_factory([]),
    runner=_runner_factory([], failing_mode="preloaded_views"),
  )
  rows = {row["metadata_storage_mode"]: row for row in out["modes"]}
  assert out["status"] == "BLOCKED" and out["passed"] is False
  assert out["diagnostic_complete"] is True and out["same_serialized_artifact"] is True
  assert rows["preloaded_views"]["status"] == "BLOCKED"
  assert rows["dedicated_preloaded"]["status"] == "PASS"
  assert "preloaded_views" in out["exact_blocker"]


def test_worker_artifact_identity_drift_fails_closed():
  out = bisect.run_metadata_storage_bisect(
    compile_fn=_compile_factory([]),
    runner=_runner_factory([], wrong_hash_mode="fixed_refreshed"),
  )
  row = next(row for row in out["modes"] if row["metadata_storage_mode"] == "fixed_refreshed")
  assert out["status"] == "BLOCKED" and out["same_serialized_artifact"] is False
  assert out["diagnostic_complete"] is False
  assert row["contract_valid"] is False and "different serialized" in row["exact_blocker"]


def test_timeout_is_mode_local_but_aggregate_remains_fail_closed():
  worker_calls: list[tuple] = []
  out = bisect.run_metadata_storage_bisect(
    compile_fn=_compile_factory([]),
    runner=_runner_factory(worker_calls, timeout_mode="dedicated_preloaded"),
  )
  assert len(worker_calls) == len(bisect.METADATA_STORAGE_MODES)
  row = next(row for row in out["modes"] if row["metadata_storage_mode"] == "dedicated_preloaded")
  assert row["status"] == "BLOCKED" and row["isolated_status"] == "timed_out"
  assert out["status"] == "BLOCKED" and out["diagnostic_complete"] is False


def test_parent_serialized_hash_mismatch_blocks_before_any_worker():
  worker_calls: list[tuple] = []
  out = bisect.run_metadata_storage_bisect(
    compile_fn=_compile_factory([], bad_serialized_hash=True),
    runner=_runner_factory(worker_calls),
  )
  assert out["status"] == "BLOCKED" and worker_calls == []
  assert "parent serialized" in out["exact_blocker"]


def test_runner_exception_is_mode_local_and_fails_closed():
  calls: list[str] = []
  def runner(_callback, *, args=(), **_kwargs):
    calls.append(args[2])
    if args[2] == "dedicated_preloaded": raise RuntimeError("spawn unavailable")
    return IsolatedResult("passed", {
      "metadata_storage_mode": args[2],
      "serialized_program_sha256": args[1],
      "child": _child(args[2]),
    })
  out = bisect.run_metadata_storage_bisect(
    compile_fn=_compile_factory([]), runner=runner,
  )
  assert calls == list(bisect.METADATA_STORAGE_MODES)
  row = next(row for row in out["modes"] if row["metadata_storage_mode"] == "dedicated_preloaded")
  assert row["status"] == "BLOCKED" and "spawn unavailable" in row["exact_blocker"]
  assert out["status"] == "BLOCKED" and out["diagnostic_complete"] is False
