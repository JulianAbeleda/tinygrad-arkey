import os

from extra.qk.prefill.attn_qo_direct_l2_hardware_executor_20260712 import (
  ENABLE_ENV, ENABLE_VALUE, exact_pair_metadata, run_hardware_executor)


def test_executor_is_default_closed_and_does_not_call_callbacks(monkeypatch):
  monkeypatch.delenv(ENABLE_ENV, raising=False)
  called = []
  result = run_hardware_executor(candidate={}, compile_artifact={}, route_binding={},
    stage_dispatch=lambda _: called.append("stage"), paired_benchmark=lambda _: called.append("pair"))
  assert result["status"] == "blocked"
  assert result["dispatch_state"] == "not_attempted"
  assert called == []


def test_executor_requires_both_callbacks_after_opt_in(monkeypatch):
  monkeypatch.setenv(ENABLE_ENV, ENABLE_VALUE)
  result = run_hardware_executor(candidate={}, compile_artifact={},
    route_binding={"storage": "direct_l2"}, stage_dispatch=None, paired_benchmark=None)
  assert result["status"] == "blocked"
  assert result["blockers"] == ["stage dispatch and paired benchmark callbacks are required"]


def test_pair_metadata_is_cpu_only_and_explicitly_dispatch_free():
  result = exact_pair_metadata()
  assert result["status"] == "prepared"
  assert result["dispatch_state"] == "not_attempted"
  assert result["role"] == "attn_qo"


def test_bad_route_is_rejected_before_stage_callback(monkeypatch):
  monkeypatch.setenv(ENABLE_ENV, ENABLE_VALUE)
  called = []
  result = run_hardware_executor(candidate={}, compile_artifact={}, route_binding={"storage": "lds"},
    stage_dispatch=lambda _: called.append(1), paired_benchmark=lambda _: {})
  assert result["status"] == "blocked"
  assert called == []
