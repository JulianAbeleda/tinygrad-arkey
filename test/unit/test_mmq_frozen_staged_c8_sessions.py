from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from extra.qk.mmq_frozen_epoch_runtime_preconstruction_canary import QUEUE_CLASSES
from extra.qk.mmq_frozen_staged_c8_sessions import (
  CLOCK_IDENTITY, QUEUE_ATTESTATION_SCHEMA, QUEUE_SESSION_SCHEMA,
  ROUTE_SEQUENCE_SCHEMA, SCHEMA,
  _atomic_write_json, run_persistent_c8_queue_session_worker,
  run_guarded_persistent_c8_route_sequence,
  run_persistent_c8_route_sequence_worker, run_persistent_c8_sessions,
)
from extra.qk.mmq_frozen_staged_c8_timing import QueueTimingRunners
from extra.qk.mmq_frozen_staged_family import (
  FrozenStagedFamily, load_frozen_staged_family_manifest,
)
from extra.qk.mmq_staged_c7_c8_contract import staged_c8_randomized_orders
from test.unit.test_mmq_frozen_staged_c8_timing import (
  _candidate_receipt, _fallback_receipt,
)
from test.unit.test_mmq_frozen_staged_family import _loader, _produce
from test.unit.test_mmq_staged_c7_c8_contract import _c6, _c7
from tinygrad.runtime.process_isolated import run_isolated


@pytest.fixture
def family(tmp_path) -> FrozenStagedFamily:
  role_spec, binding, output, _ = _produce(tmp_path)
  return load_frozen_staged_family_manifest(
    output, role_spec=role_spec, frozen_bundle="/frozen/bundle",
    binding_loader=_loader(binding))


def _attestation(queue: str) -> dict:
  expected_aql = "1" if queue == "AQL" else "0"
  return {
    "schema": QUEUE_ATTESTATION_SCHEMA,
    "authority": "instantiated_device_state", "device": "CPU-mock",
    "requested_queue_mode": queue, "effective_queue_mode": queue,
    "effective_queue_class": QUEUE_CLASSES[queue],
    "expected_queue_class": QUEUE_CLASSES[queue],
    "environment_amd_aql": expected_aql,
    "checks": {
      "environment_matches_requested": True,
      "requested_matches_effective": True,
      "queue_class_matches_effective": True,
    },
    "all_checks_pass": True,
  }


def _spawn_attestor_factory(*, queue_mode, config):
  return _attestation(queue_mode)


def _spawn_runner_factory(*, queue_mode, family, c6_correctness_evidence,
                          clock_identity, clock_ns, config):
  return QueueTimingRunners(
    lambda **_kwargs: _candidate_receipt(
      family, c6_correctness_evidence, queue_mode, clock=clock_identity),
    lambda **_kwargs: _fallback_receipt(
      c6_correctness_evidence, queue_mode, clock=clock_identity))


def _factories(family: FrozenStagedFamily, c6: dict, events: list[tuple]):
  def attestor_factory(*, queue_mode, config):
    events.append(("attestor_factory", queue_mode, dict(config)))
    return lambda: _attestation(queue_mode)

  def runner_factory(*, queue_mode, family, c6_correctness_evidence,
                     clock_identity, clock_ns, config):
    assert family.family_identity
    assert c6_correctness_evidence == c6
    assert clock_identity == CLOCK_IDENTITY and callable(clock_ns)
    events.append(("runner_factory", queue_mode, dict(config)))

    def candidate(**kwargs):
      events.append((
        "invoke", queue_mode, kwargs["phase"], kwargs["pair_index"],
        "staged_candidate", kwargs["invocation_index"]))
      return _candidate_receipt(
        family, c6, queue_mode, clock=clock_identity)

    def fallback(**kwargs):
      events.append((
        "invoke", queue_mode, kwargs["phase"], kwargs["pair_index"],
        "direct_packed", kwargs["invocation_index"]))
      return _fallback_receipt(
        c6, queue_mode, clock=clock_identity)

    return QueueTimingRunners(candidate, fallback)
  return runner_factory, attestor_factory


def _direct_isolated(events):
  def run(callback, *, args, timeout_seconds, start_method):
    events.append(("spawn", args[2], timeout_seconds, start_method))
    try:
      result = callback(*args)
      return SimpleNamespace(
        status="passed", result=result, timed_out=False, error=None,
        elapsed_seconds=0.5)
    except BaseException as exc:
      return SimpleNamespace(
        status="failed", result=None, timed_out=False,
        error=f"{type(exc).__name__}: {exc}", elapsed_seconds=0.5)
  return run


def test_one_persistent_child_per_queue_collects_same_session_clock_and_order(family):
  c6, events = _c6(family), []
  runner_factory, attestor_factory = _factories(family, c6, events)
  health_calls, fault_calls = [], []

  def health(env):
    health_calls.append(dict(env))
    return True

  def faults(started):
    assert isinstance(started, float)
    fault_calls.append(started)
    return [], {"clean": True}

  result = run_persistent_c8_sessions(
    family=family, c7_memory_ledger=_c7(family),
    c6_correctness_evidence=c6, runner_factory=runner_factory,
    attestor_factory=attestor_factory, warmups=3, rounds=10, seed=77,
    required_speedup=1.05, timeout_seconds=12,
    runner_config={"fixture": "runner"}, attestor_config={"fixture": "attestor"},
    isolated_runner=_direct_isolated(events), health_probe=health,
    fault_collector=faults,
    session_identity_factory=lambda queue: f"{queue}-persistent-session")
  assert result["schema"] == SCHEMA and result["status"] == "PASS", result["exact_blocker"]
  assert result["c8_collection"]["status"] == "PASS"
  assert result["c8_collection"]["c8_contract"]["decision"]["status"] == "CERTIFIED_WIN"
  assert [row for row in events if row[0] == "spawn"] == [
    ("spawn", "PM4", 12.0, "spawn"), ("spawn", "AQL", 12.0, "spawn")]
  assert [row[:2] for row in events if row[0] == "runner_factory"] == [
    ("runner_factory", "PM4"), ("runner_factory", "AQL")]
  assert health_calls == [
    {"AMD_AQL": "0"}, {"AMD_AQL": "0"},
    {"AMD_AQL": "1"}, {"AMD_AQL": "1"}]
  assert len(fault_calls) == 2

  for queue in ("PM4", "AQL"):
    session = result["queue_sessions"][queue]
    child = session["child"]
    assert session["status"] == "PASS" and session["spawn_count"] == 1
    assert session["launched"] is True
    assert session["target_dispatch_attempted"] is True
    assert session["target_dispatch_attempted_authority"] == \
      "passing_child_queue_session"
    assert child["schema"] == QUEUE_SESSION_SCHEMA
    assert child["effective_queue_attestation"]["effective_queue_mode"] == queue
    assert child["runner_factory_invocations"] == 1
    assert child["candidate_runner_instance_count"] == 1
    assert child["fallback_runner_instance_count"] == 1
    assert child["invocation_counts"] == {
      "staged_candidate": 13, "direct_packed": 13}
    assert child["orders"] == staged_c8_randomized_orders(seed=77, round_count=10)
    invocations = [
      row for row in events
      if row[:2] == ("invoke", queue) and row[2] == "round"]
    for pair_index, order in enumerate(child["orders"]):
      assert [row[4] for row in invocations if row[3] == pair_index] == order


def test_worker_rejects_environment_only_or_wrong_effective_queue_attestation(family):
  c6, events = _c6(family), []
  runner_factory, _ = _factories(family, c6, events)

  def wrong_attestor(*, queue_mode, config):
    row = _attestation(queue_mode)
    row["effective_queue_mode"] = "AQL"
    return row

  with pytest.raises(ValueError, match="did not close"):
    run_persistent_c8_queue_session_worker(
      family, c6, "PM4", 3, 10, 1, "session", CLOCK_IDENTITY,
      runner_factory, wrong_attestor)
  assert not [row for row in events if row[0] == "runner_factory"]


def test_worker_and_factories_are_spawn_serializable_without_a_device(family):
  c6 = _c6(family)
  isolated = run_isolated(
    run_persistent_c8_queue_session_worker,
    args=(
      family, c6, "PM4", 3, 10, 5, "PM4-spawn-session",
      CLOCK_IDENTITY, _spawn_runner_factory, _spawn_attestor_factory, {}, {},
    ), timeout_seconds=10, start_method="spawn")
  assert isolated.status == "passed", isolated.error
  assert isolated.result["status"] == "PASS"
  assert isolated.result["invocation_counts"] == {
    "staged_candidate": 13, "direct_packed": 13}


def test_worker_localizes_failed_invocation_and_preserves_partial_schedule(family):
  c6, events = _c6(family), []
  runner_factory, attestor_factory = _factories(family, c6, events)

  def failing_factory(**kwargs):
    runners = runner_factory(**kwargs)
    candidate_calls = 0

    def candidate(**invoke_kwargs):
      nonlocal candidate_calls
      if candidate_calls == 1:
        raise RuntimeError("classified candidate failure")
      candidate_calls += 1
      return runners.candidate(**invoke_kwargs)

    return QueueTimingRunners(candidate, runners.direct_packed)

  result = run_persistent_c8_queue_session_worker(
    family, c6, "PM4", 3, 10, 5, "PM4-spawn-session",
    CLOCK_IDENTITY, failing_factory, attestor_factory)
  assert result["schema"] == QUEUE_SESSION_SCHEMA
  assert result["status"] == "BLOCKED"
  assert result["invocation_failure"] == {
    "route": "staged_candidate", "phase": "warmup",
    "invocation_index": 1, "pair_index": None,
    "exception": "RuntimeError", "error": "classified candidate failure",
    "nested_failure": None,
  }
  assert result["invocation_counts"] == {
    "staged_candidate": 1, "direct_packed": 1}
  assert result["completed_warmups"] == {
    "staged_candidate": 1, "direct_packed": 1}
  assert result["completed_paired_rounds"] == 0
  assert "warmup staged_candidate[1]" in result["exact_blocker"]
  assert len([row for row in events if row[0] == "invoke"]) == 2


def test_worker_localizes_fallback_round_failure(family):
  c6, events = _c6(family), []
  runner_factory, attestor_factory = _factories(family, c6, events)

  def failing_factory(**kwargs):
    runners = runner_factory(**kwargs)

    def fallback(**invoke_kwargs):
      if invoke_kwargs["phase"] == "round":
        raise RuntimeError("classified fallback failure")
      return runners.direct_packed(**invoke_kwargs)

    return QueueTimingRunners(runners.candidate, fallback)

  result = run_persistent_c8_queue_session_worker(
    family, c6, "PM4", 3, 10, 5, "PM4-spawn-session",
    CLOCK_IDENTITY, failing_factory, attestor_factory)
  assert result["status"] == "BLOCKED"
  assert result["invocation_failure"]["route"] == "direct_packed"
  assert result["invocation_failure"]["phase"] == "round"
  assert result["invocation_failure"]["invocation_index"] == 3
  assert result["invocation_failure"]["pair_index"] == 0
  assert result["invocation_counts"]["direct_packed"] == 3
  assert result["completed_paired_rounds"] == 0
  assert "round direct_packed[3] pair[0]" in result["exact_blocker"]


def test_bounded_route_sequence_reuses_one_candidate_owner(family):
  c6, events = _c6(family), []
  runner_factory, attestor_factory = _factories(family, c6, events)

  def persistent_factory(**kwargs):
    runners = runner_factory(**kwargs)
    state = {}
    preconstruct_calls = []

    def preconstruct(*, queue_mode):
      preconstruct_calls.append(queue_mode)
      state.update({
        "signature": {"role": "fixture"}, "initialization_count": 1,
        "invocation_count": 0, "buffer_ranges": {"output": [1, 2]},
        "runtime_identity": {"object_id": 7},
      })
      return {"status": "PASS", "construct_only": True, "target_dispatch_count": 0}

    def candidate(**invoke_kwargs):
      assert preconstruct_calls == ["PM4"]
      state["invocation_count"] += 1
      return runners.candidate(**invoke_kwargs)

    candidate.persistent_session_state = state
    candidate.preconstruct = preconstruct
    return QueueTimingRunners(candidate, runners.direct_packed)

  result = run_persistent_c8_route_sequence_worker(
    family, c6, "PM4", ("staged_candidate", "staged_candidate"),
    "session", CLOCK_IDENTITY, persistent_factory, attestor_factory)
  assert result["schema"] == ROUTE_SEQUENCE_SCHEMA
  assert result["status"] == "PASS"
  assert result["completed_positions"] == 2
  assert result["invocation_counts"] == {
    "staged_candidate": 2, "direct_packed": 0}
  assert [row["route"] for row in result["breadcrumbs"]] == [
    "staged_candidate", "staged_candidate"]
  assert [row["persistent_session_lifecycle"]["invocation_count"]
          for row in result["breadcrumbs"]] == [1, 2]


def test_route_sequence_worker_preconstructs_candidate_before_first_route(family):
  c6, events = _c6(family), []
  runner_factory, attestor_factory = _factories(family, c6, events)

  def tracking_factory(**kwargs):
    runners = runner_factory(**kwargs)
    preconstruct_calls = []

    def preconstruct(*, queue_mode):
      assert not preconstruct_calls, "preconstruct must be invoked exactly once"
      preconstruct_calls.append(queue_mode)
      events.append(("preconstruct", queue_mode))
      return {"status": "PASS", "construct_only": True, "target_dispatch_count": 0}

    def candidate(**invoke_kwargs):
      assert preconstruct_calls == ["PM4"], \
        "candidate route ran before preconstruction completed"
      return runners.candidate(**invoke_kwargs)

    candidate.preconstruct = preconstruct
    return QueueTimingRunners(candidate, runners.direct_packed)

  result = run_persistent_c8_route_sequence_worker(
    family, c6, "PM4", ("staged_candidate",),
    "session", CLOCK_IDENTITY, tracking_factory, attestor_factory)
  assert result["schema"] == ROUTE_SEQUENCE_SCHEMA and result["status"] == "PASS"
  ordered = [row[0] for row in events if row[0] in ("preconstruct", "invoke")]
  assert ordered == ["preconstruct", "invoke"]


def test_route_sequence_worker_preconstructs_candidate_before_direct_packed_first_route(family):
  c6, events = _c6(family), []
  runner_factory, attestor_factory = _factories(family, c6, events)

  def tracking_factory(**kwargs):
    runners = runner_factory(**kwargs)
    preconstruct_calls = []

    def preconstruct(*, queue_mode):
      assert not preconstruct_calls, "preconstruct must be invoked exactly once"
      preconstruct_calls.append(queue_mode)
      events.append(("preconstruct", queue_mode))
      return {"status": "PASS", "construct_only": True, "target_dispatch_count": 0}

    def direct_packed(**invoke_kwargs):
      assert preconstruct_calls == ["PM4"], \
        "direct_packed route ran before the candidate was preconstructed"
      return runners.direct_packed(**invoke_kwargs)

    runners.candidate.preconstruct = preconstruct
    return QueueTimingRunners(runners.candidate, direct_packed)

  result = run_persistent_c8_route_sequence_worker(
    family, c6, "PM4", ("direct_packed", "staged_candidate"),
    "session", CLOCK_IDENTITY, tracking_factory, attestor_factory)
  assert result["schema"] == ROUTE_SEQUENCE_SCHEMA and result["status"] == "PASS"
  ordered = [row[0] for row in events if row[0] in ("preconstruct", "invoke")]
  assert ordered == ["preconstruct", "invoke", "invoke"]
  assert [row[4] for row in events if row[0] == "invoke"] == \
    ["direct_packed", "staged_candidate"]


def test_route_sequence_worker_fails_closed_without_candidate_preconstruct(family):
  c6, events = _c6(family), []
  runner_factory, attestor_factory = _factories(family, c6, events)

  with pytest.raises(TypeError, match="preconstruct"):
    run_persistent_c8_route_sequence_worker(
      family, c6, "PM4", ("staged_candidate",),
      "session", CLOCK_IDENTITY, runner_factory, attestor_factory)
  assert not [row for row in events if row[0] == "invoke"]


def test_route_sequence_worker_does_not_preconstruct_direct_packed_only_sequence(family):
  c6, events = _c6(family), []
  runner_factory, attestor_factory = _factories(family, c6, events)

  result = run_persistent_c8_route_sequence_worker(
    family, c6, "PM4", ("direct_packed",),
    "session", CLOCK_IDENTITY, runner_factory, attestor_factory)
  assert result["schema"] == ROUTE_SEQUENCE_SCHEMA and result["status"] == "PASS"


def test_guarded_route_sequence_stops_after_one_faulting_child(family):
  c6, events = _c6(family), []
  runner_factory, attestor_factory = _factories(family, c6, events)

  def failing_factory(**kwargs):
    runners = runner_factory(**kwargs)
    calls = 0

    def candidate(**invoke_kwargs):
      nonlocal calls
      if calls == 1: raise RuntimeError("second candidate failed")
      calls += 1
      return runners.candidate(**invoke_kwargs)

    candidate.preconstruct = lambda *, queue_mode: {
      "status": "PASS", "construct_only": True, "target_dispatch_count": 0}
    return QueueTimingRunners(candidate, runners.direct_packed)

  result = run_guarded_persistent_c8_route_sequence(
    family=family, c6_correctness_evidence=c6, queue_mode="PM4",
    sequence=("staged_candidate", "staged_candidate"),
    runner_factory=failing_factory, attestor_factory=attestor_factory,
    isolated_runner=_direct_isolated(events),
    health_probe=lambda _env: True,
    fault_collector=lambda _started: (["synthetic fault"], {"clean": False}))
  assert result["status"] == "BLOCKED"
  assert result["spawn_count"] == 1 and result["no_retry"] is True
  assert result["child"]["completed_positions"] == 1
  assert result["child"]["invocation_failure"]["position"] == 1
  assert result["kernel_faults"] == ["synthetic fault"]


def test_parent_preserves_structured_child_blocker_and_fault_evidence(family):
  c6, events = _c6(family), []
  runner_factory, attestor_factory = _factories(family, c6, events)

  def failing_factory(**kwargs):
    runners = runner_factory(**kwargs)

    def candidate(**_invoke_kwargs):
      raise RuntimeError("instruction fault boundary")

    return QueueTimingRunners(candidate, runners.direct_packed)

  result = run_persistent_c8_sessions(
    family=family, c7_memory_ledger=_c7(family),
    c6_correctness_evidence=c6, runner_factory=failing_factory,
    attestor_factory=attestor_factory, isolated_runner=_direct_isolated(events),
    health_probe=lambda _env: True,
    fault_collector=lambda _started: (["synthetic SQ instruction fault"], {"clean": False}),
    session_identity_factory=lambda queue: f"{queue}-session")
  pm4 = result["queue_sessions"]["PM4"]
  assert result["status"] == "BLOCKED"
  assert pm4["child_status"] == "blocked"
  assert pm4["child"]["invocation_failure"]["route"] == "staged_candidate"
  assert pm4["kernel_faults"] == ["synthetic SQ instruction fault"]
  assert "warmup staged_candidate[0]" in pm4["exact_blocker"]
  assert result["queue_sessions"]["AQL"]["child_status"] == \
    "not_attempted_after_prior_blocker"


def test_parent_contains_timeout_without_retry_or_queue_fallback(family):
  c6, events = _c6(family), []
  runner_factory, attestor_factory = _factories(family, c6, events)
  calls = []

  def timed_out(_callback, *, args, timeout_seconds, start_method):
    calls.append((args[2], timeout_seconds, start_method))
    return SimpleNamespace(
      status="timed_out", result=None, timed_out=True,
      error="deadline", elapsed_seconds=timeout_seconds)

  result = run_persistent_c8_sessions(
    family=family, c7_memory_ledger=_c7(family),
    c6_correctness_evidence=c6, runner_factory=runner_factory,
    attestor_factory=attestor_factory, isolated_runner=timed_out,
    health_probe=lambda _env: True,
    fault_collector=lambda _started: ([], {}),
    session_identity_factory=lambda queue: f"{queue}-session")
  assert result["status"] == "BLOCKED" and result["c8_collection"] is None
  assert calls == [("PM4", 1800.0, "spawn")]
  assert result["queue_sessions"]["PM4"]["spawn_count"] == 1
  assert result["queue_sessions"]["PM4"]["launched"] is True
  assert result["queue_sessions"]["PM4"]["target_dispatch_attempted"] is None
  assert result["queue_sessions"]["AQL"]["spawn_count"] == 0
  assert result["queue_sessions"]["AQL"]["child_status"] == \
    "not_attempted_after_prior_blocker"
  assert result["queue_sessions"]["AQL"]["launched"] is False
  assert result["queue_sessions"]["AQL"]["target_dispatch_attempted"] is False
  assert result["queue_sessions"]["AQL"]["target_dispatch_attempted_authority"] == \
    "not_attempted_after_prior_blocker"
  assert all(row["no_retry"] is True
             for row in result["queue_sessions"].values())
  assert all(row["no_queue_fallback"] is True
             for row in result["queue_sessions"].values())


def test_preflight_failure_prevents_all_later_queue_children(family):
  c6, events = _c6(family), []
  runner_factory, attestor_factory = _factories(family, c6, events)
  result = run_persistent_c8_sessions(
    family=family, c7_memory_ledger=_c7(family),
    c6_correctness_evidence=c6, runner_factory=runner_factory,
    attestor_factory=attestor_factory,
    isolated_runner=_direct_isolated(events),
    health_probe=lambda env: env["AMD_AQL"] == "1",
    fault_collector=lambda _started: ([], {}),
    session_identity_factory=lambda queue: f"{queue}-session")
  assert result["status"] == "BLOCKED"
  assert result["queue_sessions"]["PM4"]["spawn_count"] == 0
  assert result["queue_sessions"]["PM4"]["target_dispatch_attempted"] is False
  assert result["queue_sessions"]["AQL"]["spawn_count"] == 0
  assert result["queue_sessions"]["AQL"]["child_status"] == \
    "not_attempted_after_prior_blocker"
  assert not [row for row in events if row[0] == "spawn"]


def test_runner_error_preserves_unknown_launch_and_dispatch_state(family):
  c6, events = _c6(family), []
  runner_factory, attestor_factory = _factories(family, c6, events)

  def runner_error(*_args, **_kwargs):
    raise RuntimeError("spawn failed before returning child authority")

  result = run_persistent_c8_sessions(
    family=family, c7_memory_ledger=_c7(family),
    c6_correctness_evidence=c6, runner_factory=runner_factory,
    attestor_factory=attestor_factory, isolated_runner=runner_error,
    health_probe=lambda _env: True,
    fault_collector=lambda _started: ([], {}),
    session_identity_factory=lambda queue: f"{queue}-session")
  pm4, aql = result["queue_sessions"]["PM4"], result["queue_sessions"]["AQL"]
  assert result["status"] == "BLOCKED"
  assert pm4["launched"] is None and pm4["target_dispatch_attempted"] is None
  assert pm4["target_dispatch_attempted_authority"] == \
    "unknown_without_structured_child_queue_session"
  assert aql["launched"] is False and aql["target_dispatch_attempted"] is False


def test_session_output_is_atomic(tmp_path, monkeypatch):
  output = tmp_path / "nested" / "c8-session.json"
  original_replace = os.replace
  calls = []

  def replace(source, destination):
    calls.append((source, destination))
    return original_replace(source, destination)

  monkeypatch.setattr(
    "extra.qk.mmq_frozen_staged_c8_sessions.os.replace", replace)
  _atomic_write_json(output, {"status": "PASS"})
  assert output.read_text() == '{\n  "status": "PASS"\n}\n'
  assert len(calls) == 1 and calls[0][1] == output
  assert not list(output.parent.glob(f".{output.name}.*.tmp"))
