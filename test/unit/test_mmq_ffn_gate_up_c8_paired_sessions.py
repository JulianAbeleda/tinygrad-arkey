from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from extra.qk.mmq_ffn_gate_up_c8_paired_sessions import (
  SESSION_SCHEMA, collect_ffn_gate_up_c8_paired_sessions,
  ffn_gate_up_c8_randomized_orders,
  validate_ffn_gate_up_c8_paired_sessions,
)
from extra.qk.mmq_ffn_gate_up_matched_timing_contract import (
  build_ffn_gate_up_matched_complete_role_timing_contract,
)
from extra.qk.mmq_ffn_gate_up_outer_wall_runner import (
  CANDIDATE_TRACE_SCHEMA, RouteInvocation,
  build_ffn_gate_up_post_sync_execution_attestation,
  run_ffn_gate_up_outer_synchronized_wall as _run_outer,
  seal_ffn_gate_up_effective_queue_attestation,
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


def _sealed_queue(queue: str) -> dict:
  return seal_ffn_gate_up_effective_queue_attestation(
    _effective_queue(queue), queue_mode=queue)


def _host_io() -> dict:
  return {
    "authority": "instrumented_test_host_io",
    "provider_identity": "test-host-io-provider",
    "readback_count": 0, "copyout_count": 0, "copyout_bytes": 0,
  }


def _identity(value) -> str:
  encoded = json.dumps(
    value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
  return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _reidentify(value: dict) -> None:
  payload = {key: item for key, item in value.items()
             if key != "evidence_identity"}
  value["evidence_identity"] = _identity(payload)


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


def _candidate_trace(outer_start: int) -> dict:
  cursor = outer_start + 10

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


def _receipt(
    *, authority: dict, contract: dict, queue: str, route: str,
    elapsed_ns: int, outer_start: int,
    ) -> dict:
  invoke_end = outer_start + elapsed_ns - 100
  realize_end, outer_end = invoke_end + 50, invoke_end + 100
  executable = authority["candidate_binding"]["candidate_executable_identity"] \
    if route == "staged_candidate" else \
    authority["direct_bindings_by_queue"][queue]["executable_identity"]
  result = _run_outer(
    contract=contract, contract_validation_kwargs=authority,
    queue_mode=queue, route_id=route, executable_identity=executable,
    pre_sync=lambda: None,
    invoke_route=lambda: RouteInvocation(
      object(), _candidate_trace(outer_start)
      if route == "staged_candidate" else None),
    realize_output=lambda _output: None, post_sync=lambda: None,
    attest_post_sync=lambda _output, observed_queue:
      build_ffn_gate_up_post_sync_execution_attestation(
        observation_authority="instrumented_test_execution",
        queue_mode=observed_queue, route_id=route,
        executable_identity=executable,
        input_identity=authority["input_identity"]),
    host_io_census=_host_io,
    effective_queue_attestation=_sealed_queue(queue),
    clock_ns=iter((outer_start, invoke_end, realize_end, outer_end)).__next__)
  assert result.receipt["timing"]["complete_role_ns"] == elapsed_ns
  return dict(result.receipt)


def _sample(
    *, receipt: dict, phase: str, route: str, invocation_index: int,
    pair_index: int | None, clock_identity: str,
    ) -> dict:
  return {
    "phase": phase, "route_id": route,
    "invocation_index": invocation_index, "pair_index": pair_index,
    "clock_identity": clock_identity, "receipt": receipt,
  }


def _census(warmups: int, orders: list[list[str]]) -> list[dict]:
  counts = {"staged_candidate": 0, "direct_packed": 0}
  rows = []
  for _ in range(warmups):
    for route in ("staged_candidate", "direct_packed"):
      rows.append({
        "ordinal": len(rows), "phase": "warmup", "pair_index": None,
        "route_id": route, "invocation_index": counts[route],
      })
      counts[route] += 1
  for pair_index, order in enumerate(orders):
    for route in order:
      rows.append({
        "ordinal": len(rows), "phase": "round",
        "pair_index": pair_index, "route_id": route,
        "invocation_index": counts[route],
      })
      counts[route] += 1
  return rows


def _session(
    *, authority: dict, contract: dict, queue: str,
    candidate_round_ns: int, direct_round_ns: int,
    warmups: int = 3, rounds: int = 10, seed: int = 20260719,
    ) -> dict:
  clock_identity = f"{queue}:dependency-injected-monotonic-clock"
  next_outer_start = 100

  def receipt(route: str, elapsed_ns: int) -> dict:
    nonlocal next_outer_start
    result = _receipt(
      authority=authority, contract=contract, queue=queue, route=route,
      elapsed_ns=elapsed_ns, outer_start=next_outer_start)
    next_outer_start = result["timing"]["outer_end_ns"] + 100
    return result

  candidate_warmups, direct_warmups = [], []
  for index in range(warmups):
    candidate_warmups.append(_sample(
      receipt=receipt("staged_candidate", candidate_round_ns + 50),
      phase="warmup", route="staged_candidate", invocation_index=index,
      pair_index=None, clock_identity=clock_identity))
    direct_warmups.append(_sample(
      receipt=receipt("direct_packed", direct_round_ns + 50),
      phase="warmup", route="direct_packed", invocation_index=index,
      pair_index=None, clock_identity=clock_identity))
  orders = ffn_gate_up_c8_randomized_orders(
    seed=seed, round_count=rounds)
  pairs = []
  for pair_index, order in enumerate(orders):
    row = {"pair_index": pair_index, "order": list(order)}
    for route in order:
      key = "candidate" if route == "staged_candidate" else "direct_packed"
      elapsed = candidate_round_ns if route == "staged_candidate" \
        else direct_round_ns
      row[key] = _sample(
        receipt=receipt(route, elapsed + pair_index),
        phase="round", route=route,
        invocation_index=warmups + pair_index, pair_index=pair_index,
        clock_identity=clock_identity)
    pairs.append(row)
  payload = {
    "schema": SESSION_SCHEMA, "status": "PASS", "queue_mode": queue,
    "session_identity": _sid(f"{queue}-session"),
    "clock_identity": clock_identity,
    "effective_queue_attestation": _sealed_queue(queue),
    "warmups_per_route": warmups,
    "paired_round_count": rounds, "seed": seed,
    "candidate_warmups": candidate_warmups,
    "direct_warmups": direct_warmups, "paired_rounds": pairs,
    "invocation_counts": {
      "staged_candidate": warmups + rounds,
      "direct_packed": warmups + rounds,
    },
    "invocation_order_census": _census(warmups, orders),
    "no_retry": True, "no_queue_fallback": True,
    "readback_performed": False,
    "clock_monotonic_across_session": True,
    "unique_receipt_identities": True,
    "promotion_evidence_eligible": False,
  }
  return {**payload, "evidence_identity": _identity(payload)}


def _inputs(*, aql_candidate_ns: int = 1000,
            required_speedup: float = 1.0):
  authority = _authority()
  contract = _contract(authority)
  sessions = {
    "PM4": _session(
      authority=authority, contract=contract, queue="PM4",
      candidate_round_ns=1000, direct_round_ns=2000),
    "AQL": _session(
      authority=authority, contract=contract, queue="AQL",
      candidate_round_ns=aql_candidate_ns, direct_round_ns=2000),
  }
  return {
    "contract": contract, "contract_validation_kwargs": authority,
    "queue_sessions": sessions, "warmups": 3, "rounds": 10,
    "seed": 20260719, "required_speedup": required_speedup,
  }


def test_dual_queue_evaluated_win_retains_raw_samples_and_census():
  inputs = _inputs()
  result = collect_ffn_gate_up_c8_paired_sessions(**inputs)
  assert result["status"] == "PASS"
  assert result["decision"] == {
    "status": "EVALUATED_WIN", "evaluated": True,
    "candidate_wins_both_queues": True, "required_speedup": 1.0,
    "queue_wins": {"PM4": True, "AQL": True},
    "selected_route": "staged_candidate",
    "promotion_evidence_eligible": False,
    "rule": (
      "candidate median complete-role outer-wall time must be lower and "
      "direct/candidate speedup must meet the threshold on both PM4 and AQL"),
  }
  assert result["protocol"]["distinct_queue_sessions"] is True
  for queue in ("PM4", "AQL"):
    row = result["queues"][queue]
    assert row["same_clock_within_queue"] is True
    assert row["invocation_counts"] == {
      "staged_candidate": 13, "direct_packed": 13}
    assert len(row["invocation_order_census"]) == 26
    assert len(row["round_receipt_order_census"]) == 20
    assert row["raw_samples_ns"]["candidate_rounds"] == \
      [1000 + index for index in range(10)]
    assert row["raw_samples_ns"]["direct_rounds"] == \
      [2000 + index for index in range(10)]
    assert row["statistics"]["candidate_median_complete_role_ns"] == 1004.5
    assert row["statistics"]["direct_median_complete_role_ns"] == 2004.5
    assert row["all_receipts_validated"] is True
  assert validate_ffn_gate_up_c8_paired_sessions(
    result, **inputs) == result


def test_one_queue_loss_is_honest_evaluated_no_win():
  result = collect_ffn_gate_up_c8_paired_sessions(
    **_inputs(aql_candidate_ns=2200))
  assert result["decision"]["status"] == "EVALUATED_NO_WIN"
  assert result["decision"]["evaluated"] is True
  assert result["decision"]["queue_wins"] == {
    "PM4": True, "AQL": False}
  assert result["decision"]["selected_route"] == "direct_packed"


def test_required_speedup_applies_to_both_queues():
  result = collect_ffn_gate_up_c8_paired_sessions(
    **_inputs(required_speedup=2.1))
  assert result["decision"]["status"] == "EVALUATED_NO_WIN"
  assert result["decision"]["queue_wins"] == {
    "PM4": False, "AQL": False}


def test_seeded_order_is_balanced_and_deterministic():
  first = ffn_gate_up_c8_randomized_orders(seed=17, round_count=12)
  second = ffn_gate_up_c8_randomized_orders(seed=17, round_count=12)
  assert first == second
  assert all(sorted(order) == ["direct_packed", "staged_candidate"]
             for order in first)
  candidate_first = sum(order[0] == "staged_candidate" for order in first)
  assert candidate_first == len(first) - candidate_first
  with pytest.raises(ValueError, match="must be even"):
    ffn_gate_up_c8_randomized_orders(seed=17, round_count=11)


def test_distinct_pm4_aql_sessions_are_required():
  inputs = _inputs()
  sessions = deepcopy(inputs["queue_sessions"])
  sessions["AQL"]["session_identity"] = sessions["PM4"]["session_identity"]
  _reidentify(sessions["AQL"])
  with pytest.raises(ValueError, match="session identities must be distinct"):
    collect_ffn_gate_up_c8_paired_sessions(
      **{**inputs, "queue_sessions": sessions})


@pytest.mark.parametrize(("field", "replacement", "message"), (
  ("no_retry", False, "retry/fallback/readback"),
  ("no_queue_fallback", False, "retry/fallback/readback"),
  ("readback_performed", True, "retry/fallback/readback"),
))
def test_session_rejects_retry_fallback_or_readback(
    field, replacement, message):
  inputs = _inputs()
  sessions = deepcopy(inputs["queue_sessions"])
  sessions["PM4"][field] = replacement
  _reidentify(sessions["PM4"])
  with pytest.raises(ValueError, match=message):
    collect_ffn_gate_up_c8_paired_sessions(
      **{**inputs, "queue_sessions": sessions})


def test_warmup_round_and_order_protocol_fail_closed():
  inputs = _inputs()
  with pytest.raises(ValueError, match="warmups must be an integer >= 3"):
    collect_ffn_gate_up_c8_paired_sessions(**{**inputs, "warmups": 2})
  with pytest.raises(ValueError, match="rounds must be an integer >= 10"):
    collect_ffn_gate_up_c8_paired_sessions(**{**inputs, "rounds": 9})

  sessions = deepcopy(inputs["queue_sessions"])
  sessions["PM4"]["candidate_warmups"].pop()
  _reidentify(sessions["PM4"])
  with pytest.raises(ValueError, match="equal complete warmup lists"):
    collect_ffn_gate_up_c8_paired_sessions(
      **{**inputs, "queue_sessions": sessions})

  sessions = deepcopy(inputs["queue_sessions"])
  sessions["AQL"]["paired_rounds"][0]["order"].reverse()
  _reidentify(sessions["AQL"])
  with pytest.raises(ValueError, match="seeded schedule"):
    collect_ffn_gate_up_c8_paired_sessions(
      **{**inputs, "queue_sessions": sessions})


def test_invocation_counts_and_order_census_fail_closed():
  inputs = _inputs()
  sessions = deepcopy(inputs["queue_sessions"])
  sessions["PM4"]["invocation_counts"]["staged_candidate"] -= 1
  _reidentify(sessions["PM4"])
  with pytest.raises(ValueError, match="invocation counts differ"):
    collect_ffn_gate_up_c8_paired_sessions(
      **{**inputs, "queue_sessions": sessions})

  sessions = deepcopy(inputs["queue_sessions"])
  sessions["AQL"]["invocation_order_census"][7]["route_id"] = \
    "invalid_route"
  _reidentify(sessions["AQL"])
  with pytest.raises(ValueError, match="invocation order census differs"):
    collect_ffn_gate_up_c8_paired_sessions(
      **{**inputs, "queue_sessions": sessions})


def test_clock_identity_drift_within_queue_fails_closed():
  inputs = _inputs()
  sessions = deepcopy(inputs["queue_sessions"])
  sessions["PM4"]["paired_rounds"][3]["candidate"]["clock_identity"] = \
    "other-clock"
  _reidentify(sessions["PM4"])
  with pytest.raises(ValueError, match="clock_identity differs"):
    collect_ffn_gate_up_c8_paired_sessions(
      **{**inputs, "queue_sessions": sessions})


def test_effective_queue_clock_order_and_unique_receipts_are_revalidated():
  inputs = _inputs()
  sessions = deepcopy(inputs["queue_sessions"])
  sessions["AQL"]["effective_queue_attestation"] = \
    sessions["PM4"]["effective_queue_attestation"]
  _reidentify(sessions["AQL"])
  with pytest.raises(ValueError, match="effective queue attestation"):
    collect_ffn_gate_up_c8_paired_sessions(
      **{**inputs, "queue_sessions": sessions})

  sessions = deepcopy(inputs["queue_sessions"])
  first = sessions["PM4"]["paired_rounds"][2]["candidate"]["receipt"]
  second = sessions["PM4"]["paired_rounds"][3]["candidate"]["receipt"]
  sessions["PM4"]["paired_rounds"][2]["candidate"]["receipt"] = second
  sessions["PM4"]["paired_rounds"][3]["candidate"]["receipt"] = first
  _reidentify(sessions["PM4"])
  with pytest.raises(ValueError, match="not monotonic across the session"):
    collect_ffn_gate_up_c8_paired_sessions(
      **{**inputs, "queue_sessions": sessions})

  sessions = deepcopy(inputs["queue_sessions"])
  sessions["PM4"]["paired_rounds"][3]["candidate"]["receipt"] = deepcopy(
    sessions["PM4"]["paired_rounds"][2]["candidate"]["receipt"])
  _reidentify(sessions["PM4"])
  with pytest.raises(ValueError, match="timing receipt identity repeated"):
    collect_ffn_gate_up_c8_paired_sessions(
      **{**inputs, "queue_sessions": sessions})


@pytest.mark.parametrize(("field", "replacement", "message"), (
  ("contract_identity", _sid("wrong-contract"),
   "contract/workload/input identity differs"),
  ("workload_identity", _sid("wrong-workload"),
   "contract/workload/input identity differs"),
  ("input_identity", _sid("wrong-input"),
   "contract/workload/input identity differs"),
  ("timing_boundary_identity", _sid("wrong-boundary"),
   "timing boundary differs"),
  ("executable_identity", _sid("wrong-executable"),
   "executable identity differs"),
  ("route_id", "direct_packed", "queue/route receipt identity differs"),
))
def test_every_receipt_identity_is_revalidated(
    field, replacement, message):
  inputs = _inputs()
  sessions = deepcopy(inputs["queue_sessions"])
  sample = sessions["PM4"]["paired_rounds"][2]["candidate"]
  sample["receipt"][field] = replacement
  _reidentify(sample["receipt"])
  _reidentify(sessions["PM4"])
  with pytest.raises(ValueError, match=message):
    collect_ffn_gate_up_c8_paired_sessions(
      **{**inputs, "queue_sessions": sessions})


def test_invalid_c6_c7_or_transition_contract_blocks_before_sessions():
  inputs = _inputs()
  contract = deepcopy(inputs["contract"])
  contract["queue_preconditions"]["PM4"]["transition_preflights"][
    "status"] = "BLOCKED"
  _reidentify(contract)
  with pytest.raises(ValueError, match="legacy, missing, or mismatched"):
    collect_ffn_gate_up_c8_paired_sessions(
      **{**inputs, "contract": contract, "queue_sessions": {}})

  contract = deepcopy(inputs["contract"])
  contract["joint_session_c7"]["status"] = "BLOCKED"
  _reidentify(contract)
  with pytest.raises(ValueError, match="legacy, missing, or mismatched"):
    collect_ffn_gate_up_c8_paired_sessions(
      **{**inputs, "contract": contract, "queue_sessions": {}})


def test_collection_validator_rejects_retained_decision_drift():
  inputs = _inputs()
  result = collect_ffn_gate_up_c8_paired_sessions(**inputs)
  result["decision"]["status"] = "EVALUATED_NO_WIN"
  _reidentify(result)
  with pytest.raises(ValueError, match="differs from exact session evidence"):
    validate_ffn_gate_up_c8_paired_sessions(result, **inputs)
