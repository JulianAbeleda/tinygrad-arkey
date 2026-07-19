from __future__ import annotations

import gc
import hashlib
import weakref

import pytest

from extra.qk.mmq_ffn_gate_up_c8_paired_sessions import (
  collect_ffn_gate_up_c8_paired_sessions,
  ffn_gate_up_c8_randomized_orders,
)
from extra.qk.mmq_ffn_gate_up_c8_session_worker import (
  BLOCKED_SCHEMA, LowLevelRouteCallbacks,
  adapt_ffn_gate_up_runtime_routes,
  run_ffn_gate_up_c8_persistent_queue_session_worker,
)
from extra.qk.mmq_ffn_gate_up_c8_runtime import (
  FfnGateUpNoReadbackOutputRealizer, FfnGateUpOuterWallRoutes,
  FfnGateUpRouteCallback, OUTPUT_REALIZATION_SEMANTICS,
  OUTER_WALL_WRAPPER,
)
from extra.qk.mmq_ffn_gate_up_matched_timing_contract import (
  build_ffn_gate_up_matched_complete_role_timing_contract,
)
from extra.qk.mmq_ffn_gate_up_outer_wall_runner import (
  CANDIDATE_TRACE_SCHEMA, RouteInvocation,
  build_ffn_gate_up_post_sync_execution_attestation,
)
from extra.qk.mmq_frozen_staged_c8_sessions import QUEUE_ATTESTATION_SCHEMA
from extra.qk.mmq_frozen_epoch_runtime_preconstruction_canary import QUEUE_CLASSES


def _sid(label: str) -> str:
  return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


def _raw(label: str) -> str:
  return hashlib.sha256(label.encode()).hexdigest()


def _effective_queue(queue: str) -> dict:
  expected_aql = "1" if queue == "AQL" else "0"
  return {
    "schema": QUEUE_ATTESTATION_SCHEMA,
    "authority": "instantiated_device_state", "device": "AMD",
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


def _authority() -> dict:
  workload_identity, input_identity = _sid("workload"), _sid("inputs")
  return {
    "workload_identity": workload_identity,
    "input_identity": input_identity,
    "logical_q4_identity": _sid("logical-q4"),
    "resident_fp16_activation_identity": _sid("resident-fp16-activation"),
    "candidate_binding": {
      "family_identity": _sid("candidate-family"),
      "candidate_executable_identity": _sid("candidate-executable"),
      "program_key": _raw("candidate-program"),
      "binary_sha256": _raw("candidate-binary"),
    },
    "direct_bindings_by_queue": {
      queue: {
        "qualification_identity": _sid(f"{queue}-direct-qualification"),
        "executable_identity": _sid(f"{queue}-direct-executable"),
        "binary_sha256": _raw(f"{queue}-direct-binary"),
      } for queue in ("PM4", "AQL")
    },
    "joint_session_c7_identity": _sid("joint-session-c7"),
    "c6_bindings_by_queue": {
      queue: {
        "evidence_identity": _sid(f"{queue}-c6"),
        "candidate_correctness_identity":
          _sid(f"{queue}-candidate-correctness"),
        "comparator_identity": _sid(f"{queue}-comparator"),
        "workload_identity": workload_identity,
        "input_identity": input_identity,
      } for queue in ("PM4", "AQL")
    },
    "transition_preflight_bindings_by_queue": {
      queue: {
        name: _sid(f"{queue}-{name}") for name in (
          "candidate_candidate", "direct_direct",
          "direct_candidate_prefix1", "direct_candidate_full_role",
          "candidate_direct_candidate",
        )
      } for queue in ("PM4", "AQL")
    },
  }


def _contract(authority: dict) -> dict:
  return build_ffn_gate_up_matched_complete_role_timing_contract(**authority)


class PersistentClock:
  def __init__(self, *, candidate_ns: int, direct_ns: int):
    self.durations = {
      "staged_candidate": candidate_ns,
      "direct_packed": direct_ns,
    }
    self.base = 100
    self.position = 0
    self.start = None
    self.duration = None
    self.events = []

  def select(self, route: str) -> None:
    assert self.position == 1
    self.duration = self.durations[route]

  def __call__(self) -> int:
    self.events.append("clock")
    if self.position == 0:
      self.start = self.base
      value = self.start
    else:
      assert self.start is not None and self.duration is not None
      value = (
        self.start + self.duration - 100 if self.position == 1 else
        self.start + self.duration - 50 if self.position == 2 else
        self.start + self.duration)
    self.position += 1
    if self.position == 4:
      self.position = 0
      self.base += 10_000
      self.duration = None
    return value


class Output:
  def __init__(self):
    self.realized = False
    self.readback_called = False

  def numpy(self):
    self.readback_called = True
    raise AssertionError("persistent worker must not read output")


def _candidate_trace(start: int) -> dict:
  cursor = start + 10

  def interval():
    nonlocal cursor
    row = {"start_ns": cursor, "end_ns": cursor + 2}
    cursor += 2
    return row

  trace = {
    "schema": CANDIDATE_TRACE_SCHEMA,
    "activation_producer": interval(),
    "route_setup": interval(),
    "output_initialization": interval(),
    "epochs": [],
  }
  for ordinal in range(20):
    row = {"ordinal": ordinal}
    for phase in (
        "gather", "q4_transfer", "q8_values_transfer",
        "q8_scales_transfer", "q8_sums_transfer", "staging_sync",
        "dispatch", "dispatch_sync"):
      row[phase] = interval()
    trace["epochs"].append(row)
  return trace


def _worker_inputs(
    *, queue: str = "PM4", candidate_ns: int = 1000,
    direct_ns: int = 2000, output_refs: list | None = None,
    events: list | None = None,
    ):
  authority = _authority()
  contract = _contract(authority)
  clock = PersistentClock(
    candidate_ns=candidate_ns, direct_ns=direct_ns)
  events = [] if events is None else events
  output_refs = [] if output_refs is None else output_refs
  host_io_counts = {
    "readback_count": 0, "copyout_count": 0, "copyout_bytes": 0}

  def candidate_invoke():
    events.append("staged_candidate")
    clock.select("staged_candidate")
    output = Output()
    output_refs.append(weakref.ref(output))
    return RouteInvocation(output, _candidate_trace(clock.start))

  def direct_invoke():
    events.append("direct_packed")
    clock.select("direct_packed")
    output = Output()
    output_refs.append(weakref.ref(output))
    return RouteInvocation(output)

  def realize(output):
    assert output.readback_called is False
    output.realized = True

  candidate_executable = \
    authority["candidate_binding"]["candidate_executable_identity"]
  direct_executable = \
    authority["direct_bindings_by_queue"][queue]["executable_identity"]
  def callbacks(route, executable, invoke):
    return LowLevelRouteCallbacks(
      queue_mode=queue, input_identity=authority["input_identity"],
      executable_identity=executable, invoke=invoke, realize_output=realize,
      attest_post_sync=lambda _output, observed_queue:
        build_ffn_gate_up_post_sync_execution_attestation(
          observation_authority="instrumented_test_execution",
          queue_mode=observed_queue, route_id=route,
          executable_identity=executable,
          input_identity=authority["input_identity"]))

  def host_io_census():
    return {
      "authority": "instrumented_test_host_io",
      "provider_identity": f"{queue}-host-io-provider", **host_io_counts}
  host_io_census.counts = host_io_counts
  return {
    "contract": contract, "contract_validation_kwargs": authority,
    "queue_mode": queue, "session_identity": _sid(f"{queue}-session"),
    "clock_identity": f"{queue}:persistent-monotonic-clock",
    "candidate_executable_identity": candidate_executable,
    "direct_executable_identity": direct_executable,
    "pre_sync": lambda: None, "post_sync": lambda: None,
    "clock_ns": clock,
    "effective_queue_attestation": _effective_queue(queue),
    "host_io_census": host_io_census,
    "candidate_callbacks": callbacks(
      "staged_candidate", candidate_executable, candidate_invoke),
    "direct_callbacks": callbacks(
      "direct_packed", direct_executable, direct_invoke),
    "warmups": 3, "rounds": 10, "seed": 20260719,
  }, events, output_refs


def _adapt_typed_runtime_routes(inputs: dict):
  input_identity = inputs["contract_validation_kwargs"]["input_identity"]
  candidate = inputs["candidate_callbacks"]
  direct = inputs["direct_callbacks"]
  candidate_route = FfnGateUpRouteCallback(
    route_id="staged_candidate", queue_mode=inputs["queue_mode"],
    input_identity=input_identity,
    executable_identity=inputs["candidate_executable_identity"],
    invoke=candidate.invoke,
    realize_output=FfnGateUpNoReadbackOutputRealizer(
      callback=candidate.realize_output,
      semantics=OUTPUT_REALIZATION_SEMANTICS,
      readback_performed=False),
    attest_post_sync=candidate.attest_post_sync,
    outer_wall_wrapper=OUTER_WALL_WRAPPER,
    emits_timing_receipt=False)
  direct_route = FfnGateUpRouteCallback(
    route_id="direct_packed", queue_mode=inputs["queue_mode"],
    input_identity=input_identity,
    executable_identity=inputs["direct_executable_identity"],
    invoke=direct.invoke,
    realize_output=FfnGateUpNoReadbackOutputRealizer(
      callback=direct.realize_output,
      semantics=OUTPUT_REALIZATION_SEMANTICS,
      readback_performed=False),
    attest_post_sync=direct.attest_post_sync,
    outer_wall_wrapper=OUTER_WALL_WRAPPER,
    emits_timing_receipt=False)
  routes = FfnGateUpOuterWallRoutes(candidate_route, direct_route)
  adapted = adapt_ffn_gate_up_runtime_routes(
    routes, queue_mode=inputs["queue_mode"], input_identity=input_identity,
    candidate_executable_identity=inputs[
      "candidate_executable_identity"],
    direct_executable_identity=inputs["direct_executable_identity"])
  assert adapted.candidate.invoke is candidate_route.invoke
  assert adapted.candidate.realize_output is candidate_route.realize_output
  assert adapted.direct_packed.invoke is direct_route.invoke
  assert adapted.direct_packed.realize_output is direct_route.realize_output
  return routes, adapted


def test_worker_runs_exact_persistent_sequence_and_collector_accepts_capture():
  pm4_inputs, pm4_events, _ = _worker_inputs(queue="PM4")
  aql_inputs, aql_events, _ = _worker_inputs(queue="AQL")
  pm4 = run_ffn_gate_up_c8_persistent_queue_session_worker(**pm4_inputs)
  aql = run_ffn_gate_up_c8_persistent_queue_session_worker(**aql_inputs)
  assert pm4["status"] == aql["status"] == "PASS"
  expected = ["staged_candidate", "direct_packed"] * 3
  for order in ffn_gate_up_c8_randomized_orders(
      seed=20260719, round_count=10):
    expected.extend(order)
  assert pm4_events == aql_events == expected
  assert pm4["invocation_counts"] == {
    "staged_candidate": 13, "direct_packed": 13}
  assert len(pm4["invocation_order_census"]) == 26
  assert pm4["no_retry"] is True
  assert pm4["no_queue_fallback"] is True
  assert pm4["readback_performed"] is False

  collected = collect_ffn_gate_up_c8_paired_sessions(
    contract=pm4_inputs["contract"],
    contract_validation_kwargs=pm4_inputs["contract_validation_kwargs"],
    queue_sessions={"PM4": pm4, "AQL": aql},
    warmups=3, rounds=10, seed=20260719, required_speedup=1.0)
  assert collected["decision"]["status"] == "EVALUATED_WIN"
  assert collected["decision"]["queue_wins"] == {
    "PM4": True, "AQL": True}


def test_typed_runtime_routes_feed_both_workers_and_paired_collector_without_readback():
  sessions, refs_by_queue = {}, {}
  first_routes = None
  for queue in ("PM4", "AQL"):
    inputs, _, refs = _worker_inputs(queue=queue)
    routes, adapted = _adapt_typed_runtime_routes(inputs)
    first_routes = routes if first_routes is None else first_routes
    inputs["candidate_callbacks"] = adapted.candidate
    inputs["direct_callbacks"] = adapted.direct_packed
    sessions[queue] = \
      run_ffn_gate_up_c8_persistent_queue_session_worker(**inputs)
    refs_by_queue[queue] = refs

  assert first_routes is not None
  pm4_inputs, _, _ = _worker_inputs(queue="PM4")
  collected = collect_ffn_gate_up_c8_paired_sessions(
    contract=pm4_inputs["contract"],
    contract_validation_kwargs=pm4_inputs["contract_validation_kwargs"],
    queue_sessions=sessions, warmups=3, rounds=10, seed=20260719,
    required_speedup=1.0)
  assert collected["decision"]["status"] == "EVALUATED_WIN"
  assert collected["decision"]["queue_wins"] == {
    "PM4": True, "AQL": True}
  assert all(
    session["readback_performed"] is False
    for session in sessions.values())
  gc.collect()
  assert all(len(refs) == 26 for refs in refs_by_queue.values())
  assert all(
    ref() is None for refs in refs_by_queue.values() for ref in refs)


def test_runtime_route_adapter_rejects_untyped_routes_before_callbacks():
  inputs, events, _ = _worker_inputs()
  with pytest.raises(TypeError, match="FfnGateUpOuterWallRoutes"):
    adapt_ffn_gate_up_runtime_routes(
      {"candidate": inputs["candidate_callbacks"]},
      queue_mode=inputs["queue_mode"],
      input_identity=inputs["contract_validation_kwargs"]["input_identity"],
      candidate_executable_identity=inputs[
        "candidate_executable_identity"],
      direct_executable_identity=inputs["direct_executable_identity"])
  assert events == []


def test_runtime_route_adapter_rejects_declared_readback_before_callbacks():
  inputs, events, _ = _worker_inputs()
  routes, _ = _adapt_typed_runtime_routes(inputs)
  invalid = FfnGateUpOuterWallRoutes(
    FfnGateUpRouteCallback(
      route_id=routes.candidate.route_id,
      queue_mode=routes.candidate.queue_mode,
      input_identity=routes.candidate.input_identity,
      executable_identity=routes.candidate.executable_identity,
      invoke=routes.candidate.invoke,
      realize_output=FfnGateUpNoReadbackOutputRealizer(
        callback=routes.candidate.realize_output.callback,
        semantics=OUTPUT_REALIZATION_SEMANTICS,
        readback_performed=True),
      attest_post_sync=routes.candidate.attest_post_sync,
      outer_wall_wrapper=routes.candidate.outer_wall_wrapper,
      emits_timing_receipt=False),
    routes.direct_packed)
  with pytest.raises(ValueError, match="no-readback"):
    adapt_ffn_gate_up_runtime_routes(
      invalid,
      queue_mode=inputs["queue_mode"],
      input_identity=inputs["contract_validation_kwargs"]["input_identity"],
      candidate_executable_identity=inputs[
        "candidate_executable_identity"],
      direct_executable_identity=inputs["direct_executable_identity"])
  assert events == []


def test_worker_releases_output_references_after_outer_wall_without_readback():
  inputs, _, refs = _worker_inputs()
  result = run_ffn_gate_up_c8_persistent_queue_session_worker(**inputs)
  assert result["status"] == "PASS"
  gc.collect()
  assert len(refs) == 26
  assert all(ref() is None for ref in refs)


def test_first_failure_stops_session_without_retry():
  inputs, events, _ = _worker_inputs()
  original = inputs["direct_callbacks"]
  calls = 0

  def fail_direct():
    nonlocal calls
    calls += 1
    events.append("direct_packed")
    raise RuntimeError("injected direct failure")

  inputs["direct_callbacks"] = LowLevelRouteCallbacks(
    queue_mode=original.queue_mode, input_identity=original.input_identity,
    executable_identity=original.executable_identity, invoke=fail_direct,
    realize_output=original.realize_output,
    attest_post_sync=original.attest_post_sync)
  result = run_ffn_gate_up_c8_persistent_queue_session_worker(**inputs)
  assert result["schema"] == BLOCKED_SCHEMA
  assert result["status"] == "BLOCKED"
  assert result["collector_eligible"] is False
  assert result["no_retry"] is True and result["retry_count"] == 0
  assert result["no_queue_fallback"] is True
  assert result["readback_performed"] is False
  assert result["failed_invocation"]["route_id"] == "direct_packed"
  assert result["failed_invocation"]["invocation_index"] == 0
  assert result["completed_invocation_counts"] == {
    "staged_candidate": 1, "direct_packed": 0}
  assert events == ["staged_candidate", "direct_packed"]
  assert calls == 1


def test_blocked_diagnostic_cannot_feed_paired_collector():
  pm4_inputs, _, _ = _worker_inputs(queue="PM4")
  aql_inputs, _, _ = _worker_inputs(queue="AQL")
  original = pm4_inputs["candidate_callbacks"]
  pm4_inputs["candidate_callbacks"] = LowLevelRouteCallbacks(
    queue_mode=original.queue_mode, input_identity=original.input_identity,
    executable_identity=original.executable_identity,
    invoke=lambda: (_ for _ in ()).throw(RuntimeError("blocked")),
    realize_output=original.realize_output,
    attest_post_sync=original.attest_post_sync)
  blocked = run_ffn_gate_up_c8_persistent_queue_session_worker(**pm4_inputs)
  aql = run_ffn_gate_up_c8_persistent_queue_session_worker(**aql_inputs)
  with pytest.raises(ValueError, match="fields differ"):
    collect_ffn_gate_up_c8_paired_sessions(
      contract=pm4_inputs["contract"],
      contract_validation_kwargs=pm4_inputs["contract_validation_kwargs"],
      queue_sessions={"PM4": blocked, "AQL": aql},
      warmups=3, rounds=10, seed=20260719)


def test_legacy_callback_omission_is_rejected_before_any_callback():
  inputs, events, _ = _worker_inputs()
  sync_calls = []
  inputs["pre_sync"] = lambda: sync_calls.append("pre_sync")
  inputs["candidate_callbacks"] = None
  with pytest.raises(TypeError, match="legacy receipt-runner defaults"):
    run_ffn_gate_up_c8_persistent_queue_session_worker(**inputs)
  assert events == [] and sync_calls == []


@pytest.mark.parametrize("route", ("candidate", "direct"))
def test_callback_executable_identity_drift_is_rejected_before_sync(route):
  inputs, events, _ = _worker_inputs()
  sync_calls = []
  inputs["pre_sync"] = lambda: sync_calls.append("pre_sync")
  key = f"{route}_callbacks"
  original = inputs[key]
  inputs[key] = LowLevelRouteCallbacks(
    queue_mode=original.queue_mode, input_identity=original.input_identity,
    executable_identity=_sid(f"wrong-{route}-executable"),
    invoke=original.invoke, realize_output=original.realize_output,
    attest_post_sync=original.attest_post_sync)
  with pytest.raises(ValueError, match="executable identity drifted"):
    run_ffn_gate_up_c8_persistent_queue_session_worker(**inputs)
  assert events == [] and sync_calls == []


def test_declared_executable_identity_drift_is_rejected_before_sync():
  inputs, events, _ = _worker_inputs()
  sync_calls = []
  inputs["pre_sync"] = lambda: sync_calls.append("pre_sync")
  inputs["candidate_executable_identity"] = _sid("wrong-declared-candidate")
  with pytest.raises(ValueError, match="differs from contract"):
    run_ffn_gate_up_c8_persistent_queue_session_worker(**inputs)
  assert events == [] and sync_calls == []


def test_invalid_contract_blocks_before_any_low_level_callback():
  inputs, events, _ = _worker_inputs()
  sync_calls = []
  inputs["pre_sync"] = lambda: sync_calls.append("pre_sync")
  inputs["contract"]["joint_session_c7"]["status"] = "BLOCKED"
  with pytest.raises(ValueError, match="legacy, missing, or mismatched"):
    run_ffn_gate_up_c8_persistent_queue_session_worker(**inputs)
  assert events == [] and sync_calls == []


def test_effective_queue_attestation_and_callback_queue_fail_before_sync():
  inputs, events, _ = _worker_inputs(queue="AQL")
  sync_calls = []
  inputs["pre_sync"] = lambda: sync_calls.append("pre_sync")
  inputs["effective_queue_attestation"] = _effective_queue("PM4")
  with pytest.raises(ValueError, match="effective queue attestation"):
    run_ffn_gate_up_c8_persistent_queue_session_worker(**inputs)
  assert events == [] and sync_calls == []

  inputs, events, _ = _worker_inputs(queue="AQL")
  original = inputs["candidate_callbacks"]
  inputs["candidate_callbacks"] = LowLevelRouteCallbacks(
    queue_mode="PM4", input_identity=original.input_identity,
    executable_identity=original.executable_identity,
    invoke=original.invoke, realize_output=original.realize_output,
    attest_post_sync=original.attest_post_sync)
  with pytest.raises(ValueError, match="queue mode drifted"):
    run_ffn_gate_up_c8_persistent_queue_session_worker(**inputs)
  assert events == []


def test_readback_then_none_is_blocked_with_retained_host_io_census():
  inputs, _, _ = _worker_inputs()
  original = inputs["candidate_callbacks"]

  def readback_then_none(output):
    inputs["host_io_census"].counts["readback_count"] += 1
    output.readback_called = True
    return None

  inputs["candidate_callbacks"] = LowLevelRouteCallbacks(
    queue_mode=original.queue_mode, input_identity=original.input_identity,
    executable_identity=original.executable_identity, invoke=original.invoke,
    realize_output=readback_then_none,
    attest_post_sync=original.attest_post_sync)
  result = run_ffn_gate_up_c8_persistent_queue_session_worker(**inputs)
  assert result["status"] == "BLOCKED"
  assert result["collector_eligible"] is False
  assert result["readback_performed"] is True
  assert result["failed_invocation"]["exception"] == "HostIoCensusViolation"
  evidence = result["failed_invocation"]["audit_evidence"]
  assert evidence["status"] == "BLOCKED"
  assert evidence["delta"]["readback_count"] == 1
  assert evidence["no_readback"] is False


def test_wrong_post_sync_observation_and_reset_clock_block_session():
  inputs, _, _ = _worker_inputs()
  original = inputs["candidate_callbacks"]
  inputs["candidate_callbacks"] = LowLevelRouteCallbacks(
    queue_mode=original.queue_mode, input_identity=original.input_identity,
    executable_identity=original.executable_identity, invoke=original.invoke,
    realize_output=original.realize_output,
    attest_post_sync=lambda _output, queue:
      build_ffn_gate_up_post_sync_execution_attestation(
        observation_authority="instrumented_test_execution",
        queue_mode=queue, route_id="direct_packed",
        executable_identity=original.executable_identity,
        input_identity=original.input_identity))
  result = run_ffn_gate_up_c8_persistent_queue_session_worker(**inputs)
  assert result["status"] == "BLOCKED"
  assert "post-sync execution attestation differs" in \
    result["failed_invocation"]["error"]

  inputs, _, _ = _worker_inputs()
  clock = inputs["clock_ns"]

  def resetting_clock():
    value = clock()
    if clock.position == 0:
      clock.base = 100
    return value

  inputs["clock_ns"] = resetting_clock
  result = run_ffn_gate_up_c8_persistent_queue_session_worker(**inputs)
  assert result["status"] == "BLOCKED"
  assert "clock did not advance across invocations" in \
    result["failed_invocation"]["error"]
