from tinygrad.runtime.process_isolated import run_isolated


def _success(value): return {"value": value}
def _failure(): raise RuntimeError("boom")
def _failure_with_census():
  exc = RuntimeError("delayed synchronize")
  exc.pm4_dispatch_census = {"status": "REALIZATION_ERROR", "accepted_target_call_count": 1,
                             "calls": [{"kernarg_qwords": [1, 2, 3, 4, 5]}]}
  raise exc
def _failure_with_preparation():
  exc = RuntimeError("producer synchronize")
  exc.preparation_phase = {
    "status": "SYNCHRONIZATION_ERROR",
    "synchronize": {"began": True, "returned": False},
  }
  raise exc
def _hang():
  import time; time.sleep(10)
def _no_result():
  import os; os._exit(0)


def test_isolated_success_and_exception_are_structured():
  assert run_isolated(_success, args=(3,), timeout_seconds=2).result == {"value": 3}
  failed = run_isolated(_failure, timeout_seconds=2)
  assert failed.status == "failed" and "boom" in (failed.error or "")


def test_isolated_failure_preserves_typed_dispatch_evidence():
  failed = run_isolated(_failure_with_census, timeout_seconds=2)
  assert failed.status == "failed" and "delayed synchronize" in (failed.error or "")
  assert failed.evidence == {"pm4_dispatch_census": {
    "status": "REALIZATION_ERROR", "accepted_target_call_count": 1,
    "calls": [{"kernarg_qwords": [1, 2, 3, 4, 5]}]}}


def test_isolated_failure_preserves_typed_preparation_evidence():
  failed = run_isolated(_failure_with_preparation, timeout_seconds=2)
  assert failed.evidence == {"preparation_phase": {
    "status": "SYNCHRONIZATION_ERROR",
    "synchronize": {"began": True, "returned": False},
  }}


def test_isolated_timeout_is_hard_and_fail_closed():
  result = run_isolated(_hang, timeout_seconds=0.1, terminate_grace_seconds=0.05)
  assert result.status == "timed_out" and result.timed_out is True


def test_isolated_missing_result_fails_closed():
  result = run_isolated(_no_result, timeout_seconds=2)
  assert result.status == "failed" and "without a result" in (result.error or "")


def test_isolated_runs_under_spawn_with_picklable_target():
  # The GPU path uses spawn so the child initializes the device FRESH; a
  # module-level target + picklable args survive the pickle spawn requires.
  assert run_isolated(_success, args=(5,), timeout_seconds=30, start_method="spawn").result == {"value": 5}
