from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from extra.qk.mmq_ffn_gate_up_guarded_sessions import (
  CHILD_SCHEMA, SCHEMA, GuardedChildFactories, GuardedQueueRequest,
  QueueRuntimeEnvironment, run_guarded_ffn_gate_up_queue_child,
  run_guarded_ffn_gate_up_sessions,
)


def _sid(char: str) -> str:
  return "sha256:" + char * 64


def _loaded(queue: str = "PM4"):
  composition = {
    "composition_identity": _sid("1"),
    "candidate_binding": {
      "candidate_executable_identity": _sid("2")},
    "direct_bindings_by_queue": {
      "PM4": {"executable_identity": _sid("3")},
      "AQL": {"executable_identity": _sid("4")}},
  }
  return SimpleNamespace(
    family=SimpleNamespace(family_identity=_sid("5")),
    fixture=SimpleNamespace(
      input_identity=_sid("6"), workload_identity=_sid("7")),
    composition=composition,
    matched_timing_contract={"evidence_identity": _sid("8")},
    contract_validation_kwargs={"authority": "common"})


class _Census(AbstractContextManager):
  def __init__(self, events):
    self.events = events

  def __enter__(self):
    self.events.append("census_enter")
    return self

  def __exit__(self, *_args):
    self.events.append("census_exit")

  def snapshot(self):
    self.events.append("snapshot")
    return {
      "authority": "test", "provider_identity": "test",
      "readback_count": 0, "copyout_count": 0, "copyout_bytes": 0}


def _child_request(events, *, queue="PM4", worker_status="PASS"):
  loaded = _loaded(queue)

  def load(config):
    assert config == {"config": "v2"}
    events.append("load")
    return loaded

  def runtime(*, queue_mode):
    events.append(f"runtime:{queue_mode}")
    return QueueRuntimeEnvironment(
      effective_queue_attestation={"queue": queue_mode},
      pre_sync=lambda: None, post_sync=lambda: None)

  routes = object()

  def compose(_loaded_value, **kwargs):
    assert _loaded_value is loaded
    assert kwargs["queue_mode"] == queue
    events.append("compose")
    kwargs["candidate_route_builder"]()
    kwargs["direct_route_builder"]()
    return routes

  adapted = SimpleNamespace(candidate=object(), direct_packed=object())

  def adapt(value, **kwargs):
    assert value is routes
    assert kwargs["input_identity"] == _sid("6")
    events.append("adapt")
    return adapted

  def worker(**kwargs):
    assert kwargs["candidate_callbacks"] is adapted.candidate
    assert kwargs["direct_callbacks"] is adapted.direct_packed
    assert kwargs["host_io_census"]()["readback_count"] == 0
    events.append("worker")
    if worker_status == "PASS":
      return {"schema": "session.v2", "status": "PASS",
              "queue_mode": queue}
    return {"schema": "blocked.v2", "status": "BLOCKED",
            "exact_blocker": "worker stopped"}

  factories = GuardedChildFactories(
    runtime_config_loader=load, queue_runtime_factory=runtime,
    route_composer=compose, route_adapter=adapt,
    persistent_worker=worker, clock_ns=lambda: 1)
  return GuardedQueueRequest(
    runtime_config={"config": "v2"}, queue_mode=queue,
    session_identity=_sid("9" if queue == "PM4" else "a"),
    clock_identity="clock", warmups=3, rounds=10, seed=7,
    candidate_route_builder=lambda **_kwargs: None,
    direct_route_builder=lambda **_kwargs: None,
    host_io_census_factory=lambda: _Census(events),
    child_factories=factories)


def test_queue_child_selects_queue_then_builds_routes_once_inside_one_census(
    monkeypatch,
    ):
  events = []
  request = _child_request(events)
  for factory in (
      request.child_factories.runtime_config_loader,
      request.child_factories.queue_runtime_factory):
    original = factory

    def checked(*args, _original=original, **kwargs):
      assert __import__("os").environ["AMD_AQL"] == "0"
      return _original(*args, **kwargs)

    if factory is request.child_factories.runtime_config_loader:
      request = GuardedQueueRequest(
        **{**request.__dict__, "child_factories": GuardedChildFactories(
          runtime_config_loader=checked,
          queue_runtime_factory=request.child_factories.queue_runtime_factory,
          route_composer=request.child_factories.route_composer,
          route_adapter=request.child_factories.route_adapter,
          persistent_worker=request.child_factories.persistent_worker,
          clock_ns=request.child_factories.clock_ns)})
    else:
      request = GuardedQueueRequest(
        **{**request.__dict__, "child_factories": GuardedChildFactories(
          runtime_config_loader=request.child_factories.runtime_config_loader,
          queue_runtime_factory=checked,
          route_composer=request.child_factories.route_composer,
          route_adapter=request.child_factories.route_adapter,
          persistent_worker=request.child_factories.persistent_worker,
          clock_ns=request.child_factories.clock_ns)})
  monkeypatch.delenv("AMD_AQL", raising=False)
  result = run_guarded_ffn_gate_up_queue_child(request)
  assert result["status"] == "PASS"
  assert result["schema"] == CHILD_SCHEMA
  assert result["route_composer_invocations"] == 1
  assert result["candidate_route_builder_invocations"] == 1
  assert result["direct_route_builder_invocations"] == 1
  assert result["host_io_census_scope_count"] == 1
  assert events == [
    "load", "census_enter", "runtime:PM4", "compose", "adapt",
    "snapshot", "worker", "census_exit"]


def test_queue_child_preserves_worker_blocker_without_retry():
  result = run_guarded_ffn_gate_up_queue_child(
    _child_request([], worker_status="BLOCKED"))
  assert result["status"] == "BLOCKED"
  assert result["exact_blocker"] == "worker stopped"
  assert result["no_retry"] is True and result["retry_count"] == 0


def test_queue_child_fails_closed_if_composer_invokes_builder_twice():
  events = []
  request = _child_request(events)

  def compose_twice(_loaded_value, **kwargs):
    kwargs["candidate_route_builder"]()
    kwargs["candidate_route_builder"]()

  request = replace(
    request, child_factories=replace(
      request.child_factories, route_composer=compose_twice))
  result = run_guarded_ffn_gate_up_queue_child(request)
  assert result["status"] == "BLOCKED"
  assert "builder invoked more than once" in result["exact_blocker"]
  assert result["candidate_route_builder_invocations"] == 2
  assert events[-1] == "census_exit"


def _child_result(queue):
  loaded = _loaded(queue)
  session = {"schema": "session.v2", "status": "PASS", "queue_mode": queue}
  payload = {
    "schema": CHILD_SCHEMA, "status": "PASS", "exact_blocker": None,
    "queue_mode": queue,
    "session_identity": _sid("9" if queue == "PM4" else "a"),
    "clock_identity": "clock",
    "no_retry": True, "retry_count": 0, "no_queue_fallback": True,
    "route_composer_invocations": 1,
    "candidate_route_builder_invocations": 1,
    "direct_route_builder_invocations": 1,
    "host_io_census_scope_count": 1,
    "family_identity": loaded.family.family_identity,
    "input_identity": loaded.fixture.input_identity,
    "workload_identity": loaded.fixture.workload_identity,
    "candidate_executable_identity":
      loaded.composition["candidate_binding"][
        "candidate_executable_identity"],
    "direct_executable_identity":
      loaded.composition["direct_bindings_by_queue"][queue][
        "executable_identity"],
    "contract": loaded.matched_timing_contract,
    "contract_validation_kwargs": loaded.contract_validation_kwargs,
    "contract_identity": _sid("8"),
    "composition_identity": _sid("1"),
    "queue_session": session,
  }
  from extra.qk.mmq_ffn_gate_up_guarded_sessions import _identity
  return {**payload, "evidence_identity": _identity(payload)}


def _run_parent(
    *, result_for=None, health=None, faults=None, isolated_override=None,
    collector_override=None):
  calls = []
  result_for = result_for or _child_result
  health_values = iter(health or (True, True, True, True))
  fault_values = iter(faults or (([], {}), ([], {})))

  def isolated(callback, *, args, **kwargs):
    request = args[0]
    calls.append(("spawn", request.queue_mode, callback, kwargs))
    if isolated_override is not None:
      return isolated_override(request)
    return SimpleNamespace(
      status="passed", result=result_for(request.queue_mode),
      timed_out=False, error=None, elapsed_seconds=0.1)

  def collector(**kwargs):
    calls.append(("collect", tuple(kwargs["queue_sessions"])))
    if collector_override is not None:
      return collector_override(kwargs)
    return {"status": "PASS", "decision": {"status": "EVALUATED_NO_WIN"}}

  result = run_guarded_ffn_gate_up_sessions(
    runtime_config={"config": "v2"},
    candidate_route_builder=lambda **_kwargs: None,
    direct_route_builder=lambda **_kwargs: None,
    host_io_census_factory=lambda: _Census([]),
    paired_collector=collector, isolated_runner=isolated,
    health_probe=lambda env: calls.append(
      ("health", env["AMD_AQL"])) or next(health_values),
    fault_collector=lambda _started: next(fault_values),
    session_identity_factory=lambda queue:
      _sid("9" if queue == "PM4" else "a"),
    clock_identity="clock")
  return result, calls


def test_parent_runs_pm4_then_aql_and_collects_only_after_both_guarded():
  result, calls = _run_parent()
  assert result["status"] == "PASS" and result["schema"] == SCHEMA
  assert [row[1] for row in calls if row[0] == "spawn"] == ["PM4", "AQL"]
  assert [row for row in calls if row[0] == "health"] == [
    ("health", "0"), ("health", "0"),
    ("health", "1"), ("health", "1")]
  assert calls[-1] == ("collect", ("PM4", "AQL"))
  assert result["protocol"]["no_retry"] is True


def test_parent_stops_before_aql_after_pm4_fault():
  result, calls = _run_parent(faults=(
    (["amdgpu: GPU reset"], {"blocks": [1]}),))
  assert result["status"] == "BLOCKED"
  assert [row[1] for row in calls if row[0] == "spawn"] == ["PM4"]
  assert result["queue_sessions"]["PM4"]["kernel_faults"]
  assert result["queue_sessions"]["AQL"]["child_status"] == \
    "not_attempted_after_prior_blocker"
  assert not any(row[0] == "collect" for row in calls)


def test_parent_stops_before_aql_after_blocked_pm4_child():
  def blocked(queue):
    row = _child_result(queue)
    row["status"] = "BLOCKED"
    row["exact_blocker"] = "PM4 worker blocked"
    from extra.qk.mmq_ffn_gate_up_guarded_sessions import _identity
    row["evidence_identity"] = _identity({
      key: value for key, value in row.items() if key != "evidence_identity"})
    return row

  result, calls = _run_parent(result_for=blocked)
  assert result["status"] == "BLOCKED"
  assert "PM4 worker blocked" in result["exact_blocker"]
  assert [row[1] for row in calls if row[0] == "spawn"] == ["PM4"]


def test_parent_preflight_failure_launches_nothing_but_runs_post_health_and_faults():
  result, calls = _run_parent(
    health=(False, True), faults=(([], {"checked": True}),))
  assert result["status"] == "BLOCKED"
  assert not any(row[0] == "spawn" for row in calls)
  assert [row for row in calls if row[0] == "health"] == [
    ("health", "0"), ("health", "0")]
  assert result["queue_sessions"]["PM4"]["spawn_count"] == 0
  assert result["queue_sessions"]["AQL"]["child_status"] == \
    "not_attempted_after_prior_blocker"


def test_parent_postflight_failure_blocks_after_one_spawn():
  result, calls = _run_parent(health=(True, False))
  assert result["status"] == "BLOCKED"
  assert result["exact_blocker"] == "PM4 postflight health failed"
  assert [row[1] for row in calls if row[0] == "spawn"] == ["PM4"]
  assert result["queue_sessions"]["AQL"]["child_status"] == \
    "not_attempted_after_prior_blocker"


def test_parent_timeout_blocks_without_retry_or_aql_fallback():
  def timed_out(_request):
    return SimpleNamespace(
      status="timed_out", result=None, timed_out=True,
      error="deadline", elapsed_seconds=1800.0)

  result, calls = _run_parent(isolated_override=timed_out)
  assert result["status"] == "BLOCKED"
  assert result["exact_blocker"] == "PM4 persistent child timed out"
  assert [row[1] for row in calls if row[0] == "spawn"] == ["PM4"]
  assert result["queue_sessions"]["PM4"]["spawn_count"] == 1
  assert result["queue_sessions"]["PM4"]["retry_count"] == 0


def test_parent_collector_exception_fails_closed_after_both_children():
  def fail_collector(_kwargs):
    raise RuntimeError("collector failed")

  result, calls = _run_parent(collector_override=fail_collector)
  assert result["status"] == "BLOCKED"
  assert "collector failed" in result["exact_blocker"]
  assert [row[1] for row in calls if row[0] == "spawn"] == ["PM4", "AQL"]
  assert calls[-1][0] == "collect"


def test_parent_rejects_child_with_tampered_unrehashed_payload():
  def tampered(queue):
    row = _child_result(queue)
    row["session_identity"] = _sid("c")
    return row

  result, calls = _run_parent(result_for=tampered)
  assert result["status"] == "BLOCKED"
  assert "content identity differs" in result["exact_blocker"]
  assert [row[1] for row in calls if row[0] == "spawn"] == ["PM4"]


@pytest.mark.parametrize("field,value", (
  ("session_identity", _sid("c")), ("clock_identity", "wrong-clock"),
))
def test_parent_rejects_rehashed_child_session_or_clock_drift(field, value):
  def drifted(queue):
    row = _child_result(queue)
    row[field] = value
    from extra.qk.mmq_ffn_gate_up_guarded_sessions import _identity
    row["evidence_identity"] = _identity({
      key: item for key, item in row.items() if key != "evidence_identity"})
    return row

  result, calls = _run_parent(result_for=drifted)
  assert result["status"] == "BLOCKED"
  assert "child contract differs" in result["exact_blocker"]
  assert [row[1] for row in calls if row[0] == "spawn"] == ["PM4"]


def test_parent_rejects_cross_queue_candidate_identity_drift():
  def drifted(queue):
    row = _child_result(queue)
    if queue == "AQL":
      row["candidate_executable_identity"] = _sid("c")
    from extra.qk.mmq_ffn_gate_up_guarded_sessions import _identity
    row["evidence_identity"] = _identity({
      key: item for key, item in row.items() if key != "evidence_identity"})
    return row

  result, calls = _run_parent(result_for=drifted)
  assert result["status"] == "BLOCKED"
  assert "runtime authorities differ" in result["exact_blocker"]
  assert [row[1] for row in calls if row[0] == "spawn"] == ["PM4", "AQL"]
  assert not any(row[0] == "collect" for row in calls)
