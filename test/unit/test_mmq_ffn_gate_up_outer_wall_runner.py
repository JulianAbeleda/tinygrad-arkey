from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from extra.qk.mmq_ffn_gate_up_matched_timing_contract import (
  build_ffn_gate_up_matched_complete_role_timing_contract,
)
from extra.qk.mmq_ffn_gate_up_outer_wall_runner import (
  CANDIDATE_TRACE_SCHEMA, RouteInvocation,
  run_ffn_gate_up_outer_synchronized_wall,
  validate_ffn_gate_up_outer_wall_receipt,
)


def _sid(label: str) -> str:
  return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


def _raw(label: str) -> str:
  return hashlib.sha256(label.encode()).hexdigest()


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


def _interval(start: int, duration: int = 2) -> tuple[dict, int]:
  return {"start_ns": start, "end_ns": start + duration}, start + duration


def _candidate_trace() -> dict:
  cursor = 110
  producer, cursor = _interval(cursor)
  setup, cursor = _interval(cursor)
  initialization, cursor = _interval(cursor)
  epochs = []
  for ordinal in range(20):
    row = {"ordinal": ordinal}
    for phase in (
        "gather", "q4_transfer", "q8_values_transfer",
        "q8_scales_transfer", "q8_sums_transfer", "staging_sync",
        "dispatch", "dispatch_sync"):
      row[phase], cursor = _interval(cursor)
    epochs.append(row)
  assert cursor == 436
  return {
    "schema": CANDIDATE_TRACE_SCHEMA,
    "activation_producer": producer,
    "route_setup": setup,
    "output_initialization": initialization,
    "epochs": epochs,
  }


class Clock:
  def __init__(self, values):
    self.values = iter(values)

  def __call__(self):
    return next(self.values)


class Output:
  def __init__(self):
    self.realized = False
    self.readback_called = False

  def numpy(self):
    self.readback_called = True
    raise AssertionError("readback must never be called")


def _reidentify(value: dict) -> None:
  payload = {key: item for key, item in value.items()
             if key != "evidence_identity"}
  encoded = json.dumps(
    payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
  value["evidence_identity"] = \
    "sha256:" + hashlib.sha256(encoded).hexdigest()


def _run_candidate(
    *, authority: dict | None = None, contract: dict | None = None,
    trace: dict | None = None, clock_values=(100, 500, 550, 600),
    readback_requested=False, readback_callback=None,
    executable_identity: str | None = None,
    ):
  authority = _authority() if authority is None else authority
  contract = _contract(authority) if contract is None else contract
  output, events = Output(), []

  def pre_sync(): events.append("pre_sync")
  def invoke():
    events.append("invoke")
    return RouteInvocation(
      output, _candidate_trace() if trace is None else trace)
  def realize(value):
    events.append("realize")
    assert value is output
    value.realized = True
  def post_sync(): events.append("post_sync")

  result = run_ffn_gate_up_outer_synchronized_wall(
    contract=contract, contract_validation_kwargs=authority,
    queue_mode="PM4", route_id="staged_candidate",
    executable_identity=executable_identity or
      authority["candidate_binding"]["candidate_executable_identity"],
    pre_sync=pre_sync, invoke_route=invoke, realize_output=realize,
    post_sync=post_sync, clock_ns=Clock(clock_values),
    readback_requested=readback_requested,
    readback_callback=readback_callback)
  return result, output, events


def test_candidate_outer_wall_sequence_and_exhaustive_receipt():
  authority = _authority()
  result, output, events = _run_candidate(authority=authority)
  receipt, timing = result.receipt, result.receipt["timing"]
  assert events == ["pre_sync", "invoke", "realize", "post_sync"]
  assert result.output is output and output.realized is True
  assert output.readback_called is False
  assert receipt["pre_sync_outside_timed_wall"] is True
  assert receipt["readback_performed"] is False
  assert receipt["route_id"] == "staged_candidate"
  assert receipt["queue_mode"] == "PM4"
  assert receipt["workload_identity"] == authority["workload_identity"]
  assert receipt["input_identity"] == authority["input_identity"]
  assert timing["outer_start_ns"] == 100
  assert timing["route_invoke_end_ns"] == 500
  assert timing["output_realize_end_ns"] == 550
  assert timing["outer_end_ns"] == 600
  assert timing["clock_bookkeeping_gap_ns"] == {
    "before_first_phase_ns": 10, "after_last_phase_ns": 64,
    "total_ns": 74, "only_allowed_unattributed_time": True,
  }
  assert timing["output_realization"] == {
    "start_ns": 500, "end_ns": 550}
  assert timing["output_realization_ns"] == 50
  assert timing["final_sync"] == {"start_ns": 550, "end_ns": 600}
  assert timing["final_sync_ns"] == 50
  assert timing["exhaustive_phase_sum_ns"] == \
    timing["complete_role_ns"] == 500
  assert len(timing["route_phases"]["epochs"]) == 20
  assert validate_ffn_gate_up_outer_wall_receipt(
    receipt, contract=_contract(authority),
    contract_validation_kwargs=authority) == receipt


def test_direct_uses_identical_boundary_and_exact_queue_executable():
  authority = _authority()
  contract = _contract(authority)
  output, events = Output(), []

  def realize(value):
    events.append("realize")
    value.realized = True

  result = run_ffn_gate_up_outer_synchronized_wall(
    contract=contract, contract_validation_kwargs=authority,
    queue_mode="AQL", route_id="direct_packed",
    executable_identity=authority["direct_bindings_by_queue"]["AQL"][
      "executable_identity"],
    pre_sync=lambda: events.append("pre_sync"),
    invoke_route=lambda: (
      events.append("invoke") or RouteInvocation(output)),
    realize_output=realize,
    post_sync=lambda: events.append("post_sync"),
    clock_ns=Clock((100, 500, 550, 600)))
  receipt = result.receipt
  assert events == ["pre_sync", "invoke", "realize", "post_sync"]
  assert output.readback_called is False
  assert receipt["timing_boundary_identity"] == \
    contract["timing_boundary"]["boundary_identity"]
  assert receipt["timing"]["route_phases"] == {
    "production_direct_packed_invocation": {
      "start_ns": 100, "end_ns": 500}}
  assert receipt["timing"]["clock_bookkeeping_gap_ns"]["total_ns"] == 0
  assert receipt["timing"]["exhaustive_phase_sum_ns"] == 500


def test_pre_sync_precedes_start_and_post_sync_precedes_end_clock():
  authority = _authority()
  contract = _contract(authority)
  output, events = Output(), []
  values = iter((100, 500, 550, 600))

  def clock():
    events.append("clock")
    return next(values)

  run_ffn_gate_up_outer_synchronized_wall(
    contract=contract, contract_validation_kwargs=authority,
    queue_mode="PM4", route_id="staged_candidate",
    executable_identity=authority["candidate_binding"][
      "candidate_executable_identity"],
    pre_sync=lambda: events.append("pre_sync"),
    invoke_route=lambda: (
      events.append("invoke") or
      RouteInvocation(output, _candidate_trace())),
    realize_output=lambda _: events.append("realize"),
    post_sync=lambda: events.append("post_sync"), clock_ns=clock)
  assert events == [
    "pre_sync", "clock", "invoke", "clock",
    "realize", "clock", "post_sync", "clock",
  ]


def test_invalid_contract_blocks_every_callback_before_timing():
  authority = _authority()
  contract = _contract(authority)
  contract["joint_session_c7"]["status"] = "BLOCKED"
  _reidentify(contract)
  calls = []
  callback = lambda *args, **kwargs: calls.append((args, kwargs))
  with pytest.raises(ValueError, match="legacy, missing, or mismatched"):
    run_ffn_gate_up_outer_synchronized_wall(
      contract=contract, contract_validation_kwargs=authority,
      queue_mode="PM4", route_id="staged_candidate",
      executable_identity=authority["candidate_binding"][
        "candidate_executable_identity"],
      pre_sync=callback, invoke_route=callback, realize_output=callback,
      post_sync=callback, clock_ns=callback)
  assert calls == []


def test_readback_request_or_callback_is_rejected_before_sync():
  authority = _authority()
  contract = _contract(authority)
  calls = []
  with pytest.raises(ValueError, match="readback is forbidden"):
    run_ffn_gate_up_outer_synchronized_wall(
      contract=contract, contract_validation_kwargs=authority,
      queue_mode="PM4", route_id="staged_candidate",
      executable_identity=authority["candidate_binding"][
        "candidate_executable_identity"],
      pre_sync=lambda: calls.append("pre_sync"),
      invoke_route=lambda: calls.append("invoke"),
      realize_output=lambda _: calls.append("realize"),
      post_sync=lambda: calls.append("post_sync"),
      clock_ns=lambda: calls.append("clock"),
      readback_requested=True)
  assert calls == []

  readback_calls = []
  with pytest.raises(ValueError, match="readback is forbidden"):
    run_ffn_gate_up_outer_synchronized_wall(
      contract=contract, contract_validation_kwargs=authority,
      queue_mode="PM4", route_id="staged_candidate",
      executable_identity=authority["candidate_binding"][
        "candidate_executable_identity"],
      pre_sync=lambda: calls.append("pre_sync"),
      invoke_route=lambda: calls.append("invoke"),
      realize_output=lambda _: calls.append("realize"),
      post_sync=lambda: calls.append("post_sync"),
      clock_ns=lambda: calls.append("clock"),
      readback_callback=lambda output: readback_calls.append(output))
  assert calls == [] and readback_calls == []


def test_candidate_trace_requires_activation_producer():
  trace = _candidate_trace()
  del trace["activation_producer"]
  with pytest.raises(ValueError, match="candidate phase trace fields differ"):
    _run_candidate(trace=trace)


@pytest.mark.parametrize(("phase", "delta", "message"), (
  ("route_setup", 1, "phase gap"),
  ("route_setup", -1, "phase overlap"),
))
def test_candidate_trace_rejects_internal_gap_or_overlap(
    phase, delta, message):
  trace = _candidate_trace()
  trace[phase]["start_ns"] += delta
  with pytest.raises(ValueError, match=message):
    _run_candidate(trace=trace)


def test_candidate_trace_rejects_wrong_epoch_count_or_order():
  short = _candidate_trace()
  short["epochs"].pop()
  with pytest.raises(ValueError, match="exactly 20 epochs"):
    _run_candidate(trace=short)

  unordered = _candidate_trace()
  unordered["epochs"][7]["ordinal"] = 8
  with pytest.raises(ValueError, match="ordinal differs"):
    _run_candidate(trace=unordered)


def test_nonmonotonic_outer_clock_is_rejected():
  with pytest.raises(ValueError, match="strictly monotonic"):
    _run_candidate(clock_values=(100, 500, 500, 600))
  with pytest.raises(ValueError, match="strictly monotonic"):
    _run_candidate(clock_values=(100, 500, 550, 549))


def test_wrong_route_or_executable_is_rejected_before_sync():
  authority = _authority()
  contract = _contract(authority)
  calls = []
  common = {
    "contract": contract, "contract_validation_kwargs": authority,
    "queue_mode": "PM4",
    "executable_identity":
      authority["candidate_binding"]["candidate_executable_identity"],
    "pre_sync": lambda: calls.append("pre_sync"),
    "invoke_route": lambda: calls.append("invoke"),
    "realize_output": lambda _: calls.append("realize"),
    "post_sync": lambda: calls.append("post_sync"),
    "clock_ns": lambda: calls.append("clock"),
  }
  with pytest.raises(ValueError, match="route_id must be"):
    run_ffn_gate_up_outer_synchronized_wall(
      **common, route_id="legacy_r5_candidate")
  assert calls == []
  with pytest.raises(ValueError, match="timed executable identity differs"):
    run_ffn_gate_up_outer_synchronized_wall(
      **{**common, "executable_identity": _sid("wrong-executable")},
      route_id="staged_candidate")
  assert calls == []


@pytest.mark.parametrize(("path", "replacement", "message"), (
  (("contract_identity",), _sid("wrong-contract"),
   "contract/workload/input identity differs"),
  (("workload_identity",), _sid("wrong-workload"),
   "contract/workload/input identity differs"),
  (("input_identity",), _sid("wrong-input"),
   "contract/workload/input identity differs"),
  (("timing_boundary_identity",), _sid("wrong-boundary"),
   "timing boundary differs"),
  (("executable_identity",), _sid("wrong-executable"),
   "executable identity differs"),
  (("route_id",), "legacy_route", "route_id must be"),
))
def test_receipt_validator_rejects_wrong_route_boundary_or_identity(
    path, replacement, message):
  authority = _authority()
  receipt = deepcopy(_run_candidate(authority=authority)[0].receipt)
  cursor = receipt
  for key in path[:-1]: cursor = cursor[key]
  cursor[path[-1]] = replacement
  _reidentify(receipt)
  with pytest.raises(ValueError, match=message):
    validate_ffn_gate_up_outer_wall_receipt(
      receipt, contract=_contract(authority),
      contract_validation_kwargs=authority)


def test_receipt_validator_rejects_bookkeeping_drift_and_omitted_producer():
  authority = _authority()
  receipt = deepcopy(_run_candidate(authority=authority)[0].receipt)
  receipt["timing"]["clock_bookkeeping_gap_ns"]["total_ns"] += 1
  receipt["timing"]["exhaustive_phase_sum_ns"] += 1
  _reidentify(receipt)
  with pytest.raises(ValueError, match="reconstruction"):
    validate_ffn_gate_up_outer_wall_receipt(
      receipt, contract=_contract(authority),
      contract_validation_kwargs=authority)

  receipt = deepcopy(_run_candidate(authority=authority)[0].receipt)
  del receipt["timing"]["route_phases"]["activation_producer"]
  _reidentify(receipt)
  with pytest.raises(ValueError, match="candidate phase trace fields differ"):
    validate_ffn_gate_up_outer_wall_receipt(
      receipt, contract=_contract(authority),
      contract_validation_kwargs=authority)


def test_realize_callback_cannot_return_a_readback_value():
  authority = _authority()
  contract = _contract(authority)
  output = Output()
  with pytest.raises(ValueError, match="must not return a readback value"):
    run_ffn_gate_up_outer_synchronized_wall(
      contract=contract, contract_validation_kwargs=authority,
      queue_mode="PM4", route_id="staged_candidate",
      executable_identity=authority["candidate_binding"][
        "candidate_executable_identity"],
      pre_sync=lambda: None,
      invoke_route=lambda: RouteInvocation(output, _candidate_trace()),
      realize_output=lambda _: object(),
      post_sync=lambda: None, clock_ns=Clock((100, 500)))
  assert output.readback_called is False
