"""Dependency-injected outer synchronized-wall runner for exact ``ffn_gate_up``.

The module is CPU-only and imports no tinygrad device or runtime code.  A
future GPU caller supplies synchronization, route invocation, output
realization, and monotonic-clock callbacks.  Exact C6/C7/transition admission
is revalidated before any callback runs.  This layer never performs readback.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from extra.qk.mmq_ffn_gate_up_matched_timing_contract import (
  CANDIDATE_ROUTE, DIRECT_ROUTE, K_LAUNCHES, QUEUE_MODES,
  validate_ffn_gate_up_matched_complete_role_timing_contract,
)


SCHEMA = "tinygrad.mmq_q4k_q8_1.ffn_gate_up_outer_wall_receipt.v1"
CANDIDATE_TRACE_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.ffn_gate_up_candidate_phase_trace.v1"
MEASUREMENT_SOURCE = "outer_synchronized_wall"

_HEX = frozenset("0123456789abcdef")
_TOP_LEVEL_FIELDS = {
  "schema", "status", "contract_identity", "workload_identity",
  "input_identity", "queue_mode", "route_id", "executable_identity",
  "timing_boundary_identity", "measurement_source",
  "pre_sync_outside_timed_wall", "readback_performed", "timing",
  "evidence_identity",
}
_TIMING_FIELDS = {
  "outer_start_ns", "route_invoke_end_ns", "output_realize_end_ns",
  "outer_end_ns", "route_phases", "clock_bookkeeping_gap_ns",
  "output_realization", "output_realization_ns", "final_sync",
  "final_sync_ns", "exhaustive_phase_sum_ns", "complete_role_ns",
}
_CANDIDATE_TRACE_FIELDS = {
  "schema", "activation_producer", "route_setup",
  "output_initialization", "epochs",
}
_INTERVAL_FIELDS = {"start_ns", "end_ns"}
_EPOCH_FIELDS = {
  "ordinal", "gather", "q4_transfer", "q8_values_transfer",
  "q8_scales_transfer", "q8_sums_transfer", "staging_sync",
  "dispatch", "dispatch_sync",
}
_EPOCH_PHASES = (
  "gather", "q4_transfer", "q8_values_transfer", "q8_scales_transfer",
  "q8_sums_transfer", "staging_sync", "dispatch", "dispatch_sync",
)


@dataclass(frozen=True)
class RouteInvocation:
  """One route output plus candidate-only absolute phase intervals."""

  output: Any
  candidate_phase_trace: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class OuterWallRunResult:
  """The untouched route output and its JSON-safe timing receipt."""

  output: Any
  receipt: Mapping[str, Any]


def _canonical(value: Any) -> bytes:
  return json.dumps(
    value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity(value: Any) -> str:
  return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
  if not isinstance(value, Mapping):
    raise ValueError(f"{label} must be a mapping")
  return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
  if set(value) != expected:
    raise ValueError(
      f"{label} fields differ: expected {sorted(expected)!r}, "
      f"got {sorted(value)!r}")


def _content_identity(value: Any, label: str) -> str:
  if not isinstance(value, str) or not value.startswith("sha256:") or \
     len(value) != 71 or any(char not in _HEX for char in value[7:]):
    raise ValueError(f"{label} must be a sha256 content identity")
  return value


def _timestamp(value: Any, label: str) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value < 0:
    raise ValueError(f"{label} must be a non-negative integer timestamp")
  return value


def _duration(value: Any, label: str) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value < 0:
    raise ValueError(f"{label} must be a non-negative integer duration")
  return value


def _interval(value: Any, label: str) -> dict[str, int]:
  row = _mapping(value, label)
  _exact_keys(row, _INTERVAL_FIELDS, label)
  start = _timestamp(row["start_ns"], f"{label}.start_ns")
  end = _timestamp(row["end_ns"], f"{label}.end_ns")
  if end <= start:
    raise ValueError(f"{label} must have a positive monotonic duration")
  return {"start_ns": start, "end_ns": end}


def _flatten_candidate_trace(value: Any) -> tuple[dict[str, Any], list[tuple[str, dict[str, int]]]]:
  trace = _mapping(value, "candidate phase trace")
  _exact_keys(trace, _CANDIDATE_TRACE_FIELDS, "candidate phase trace")
  if trace["schema"] != CANDIDATE_TRACE_SCHEMA:
    raise ValueError("candidate phase trace schema differs or is legacy")
  normalized: dict[str, Any] = {"schema": CANDIDATE_TRACE_SCHEMA}
  flattened: list[tuple[str, dict[str, int]]] = []
  for name in ("activation_producer", "route_setup", "output_initialization"):
    interval = _interval(trace[name], f"candidate phase trace.{name}")
    normalized[name] = interval
    flattened.append((name, interval))
  epochs = trace["epochs"]
  if not isinstance(epochs, list) or len(epochs) != K_LAUNCHES:
    raise ValueError(
      f"candidate phase trace must enumerate exactly {K_LAUNCHES} epochs")
  normalized_epochs = []
  for expected_ordinal, raw_epoch in enumerate(epochs):
    label = f"candidate phase trace.epochs[{expected_ordinal}]"
    epoch = _mapping(raw_epoch, label)
    _exact_keys(epoch, _EPOCH_FIELDS, label)
    if epoch["ordinal"] != expected_ordinal:
      raise ValueError(f"{label}.ordinal differs from ordered K launches")
    normalized_epoch: dict[str, Any] = {"ordinal": expected_ordinal}
    for phase_name in _EPOCH_PHASES:
      interval = _interval(epoch[phase_name], f"{label}.{phase_name}")
      normalized_epoch[phase_name] = interval
      flattened.append(
        (f"epochs[{expected_ordinal}].{phase_name}", interval))
    normalized_epochs.append(normalized_epoch)
  normalized["epochs"] = normalized_epochs
  for (prior_name, prior), (current_name, current) in zip(
      flattened, flattened[1:]):
    if current["start_ns"] < prior["end_ns"]:
      raise ValueError(
        f"candidate phase overlap between {prior_name} and {current_name}")
    if current["start_ns"] > prior["end_ns"]:
      raise ValueError(
        f"candidate phase gap between {prior_name} and {current_name}")
  return normalized, flattened


def _candidate_timing(
    *, trace: Any, outer_start_ns: int, route_invoke_end_ns: int,
    output_realize_end_ns: int, outer_end_ns: int,
    ) -> dict[str, Any]:
  normalized, flattened = _flatten_candidate_trace(trace)
  first, last = flattened[0][1], flattened[-1][1]
  if first["start_ns"] < outer_start_ns:
    raise ValueError("candidate phase trace begins before the outer wall")
  if last["end_ns"] > route_invoke_end_ns:
    raise ValueError("candidate phase trace ends after route invocation returned")
  before_first = first["start_ns"] - outer_start_ns
  after_last = route_invoke_end_ns - last["end_ns"]
  bookkeeping = before_first + after_last
  route_phase_ns = sum(
    interval["end_ns"] - interval["start_ns"]
    for _, interval in flattened)
  output_realization = output_realize_end_ns - route_invoke_end_ns
  final_sync = outer_end_ns - output_realize_end_ns
  complete = outer_end_ns - outer_start_ns
  exhaustive = route_phase_ns + bookkeeping + output_realization + final_sync
  if exhaustive != complete:
    raise ValueError(
      "candidate phases do not reconcile exhaustively to the outer wall")
  return {
    "outer_start_ns": outer_start_ns,
    "route_invoke_end_ns": route_invoke_end_ns,
    "output_realize_end_ns": output_realize_end_ns,
    "outer_end_ns": outer_end_ns,
    "route_phases": normalized,
    "clock_bookkeeping_gap_ns": {
      "before_first_phase_ns": before_first,
      "after_last_phase_ns": after_last,
      "total_ns": bookkeeping,
      "only_allowed_unattributed_time": True,
    },
    "output_realization": {
      "start_ns": route_invoke_end_ns, "end_ns": output_realize_end_ns},
    "output_realization_ns": output_realization,
    "final_sync": {
      "start_ns": output_realize_end_ns, "end_ns": outer_end_ns},
    "final_sync_ns": final_sync,
    "exhaustive_phase_sum_ns": exhaustive,
    "complete_role_ns": complete,
  }


def _direct_timing(
    *, outer_start_ns: int, route_invoke_end_ns: int,
    output_realize_end_ns: int, outer_end_ns: int,
    ) -> dict[str, Any]:
  route_interval = {
    "start_ns": outer_start_ns, "end_ns": route_invoke_end_ns}
  route_ns = route_invoke_end_ns - outer_start_ns
  output_realization = output_realize_end_ns - route_invoke_end_ns
  final_sync = outer_end_ns - output_realize_end_ns
  complete = outer_end_ns - outer_start_ns
  exhaustive = route_ns + output_realization + final_sync
  if exhaustive != complete:
    raise ValueError("direct phases do not reconcile to the outer wall")
  return {
    "outer_start_ns": outer_start_ns,
    "route_invoke_end_ns": route_invoke_end_ns,
    "output_realize_end_ns": output_realize_end_ns,
    "outer_end_ns": outer_end_ns,
    "route_phases": {"production_direct_packed_invocation": route_interval},
    "clock_bookkeeping_gap_ns": {
      "before_first_phase_ns": 0, "after_last_phase_ns": 0,
      "total_ns": 0, "only_allowed_unattributed_time": True,
    },
    "output_realization": {
      "start_ns": route_invoke_end_ns, "end_ns": output_realize_end_ns},
    "output_realization_ns": output_realization,
    "final_sync": {
      "start_ns": output_realize_end_ns, "end_ns": outer_end_ns},
    "final_sync_ns": final_sync,
    "exhaustive_phase_sum_ns": exhaustive,
    "complete_role_ns": complete,
  }


def _validated_contract(
    contract: Any, contract_validation_kwargs: Any,
    ) -> dict[str, Any]:
  kwargs = _mapping(
    contract_validation_kwargs, "contract validation authorities")
  return validate_ffn_gate_up_matched_complete_role_timing_contract(
    contract, **dict(kwargs))


def _route_executable(
    contract: Mapping[str, Any], *, queue_mode: str, route_id: str,
    ) -> str:
  if queue_mode not in QUEUE_MODES:
    raise ValueError(f"queue_mode must be one of {QUEUE_MODES!r}")
  if route_id == CANDIDATE_ROUTE:
    return _content_identity(
      contract["candidate"]["candidate_executable_identity"],
      "candidate executable identity")
  if route_id == DIRECT_ROUTE:
    return _content_identity(
      contract["direct_packed"]["queue_qualifications"][queue_mode][
        "executable_identity"],
      f"{queue_mode} direct executable identity")
  raise ValueError(
    f"route_id must be {CANDIDATE_ROUTE!r} or {DIRECT_ROUTE!r}")


def run_ffn_gate_up_outer_synchronized_wall(
    *, contract: Mapping[str, Any],
    contract_validation_kwargs: Mapping[str, Any],
    queue_mode: str, route_id: str, executable_identity: str,
    pre_sync: Callable[[], Any], invoke_route: Callable[[], RouteInvocation],
    realize_output: Callable[[Any], Any], post_sync: Callable[[], Any],
    clock_ns: Callable[[], int],
    readback_requested: bool = False, readback_callback: Any | None = None,
    ) -> OuterWallRunResult:
  """Run one admitted route over the exact shared outer wall.

  ``pre_sync`` completes before the start timestamp.  The route output is
  realized and ``post_sync`` completes before the end timestamp.  A readback
  callback is intentionally unsupported.
  """
  validated = _validated_contract(contract, contract_validation_kwargs)
  if type(readback_requested) is not bool:
    raise ValueError("readback_requested must be a bool")
  if readback_requested or readback_callback is not None:
    raise ValueError("readback is forbidden by the matched timing contract")
  if not all(callable(callback) for callback in (
      pre_sync, invoke_route, realize_output, post_sync, clock_ns)):
    raise TypeError("outer-wall synchronization, route, realization, and clock callbacks must be callable")
  expected_executable = _route_executable(
    validated, queue_mode=queue_mode, route_id=route_id)
  executable_identity = _content_identity(
    executable_identity, "timed executable identity")
  if executable_identity != expected_executable:
    raise ValueError("timed executable identity differs from the exact route")

  pre_sync()
  outer_start_ns = _timestamp(clock_ns(), "outer start")
  invocation = invoke_route()
  if not isinstance(invocation, RouteInvocation):
    raise TypeError("invoke_route must return RouteInvocation")
  route_invoke_end_ns = _timestamp(clock_ns(), "route invoke end")
  realize_result = realize_output(invocation.output)
  if realize_result is not None:
    raise ValueError("realize_output must not return a readback value")
  output_realize_end_ns = _timestamp(clock_ns(), "output realize end")
  post_sync()
  outer_end_ns = _timestamp(clock_ns(), "outer end")
  checkpoints = (
    outer_start_ns, route_invoke_end_ns, output_realize_end_ns, outer_end_ns)
  if any(current <= prior for prior, current in zip(
      checkpoints, checkpoints[1:])):
    raise ValueError("outer-wall clock checkpoints must be strictly monotonic")

  timing = _candidate_timing(
    trace=invocation.candidate_phase_trace,
    outer_start_ns=outer_start_ns,
    route_invoke_end_ns=route_invoke_end_ns,
    output_realize_end_ns=output_realize_end_ns,
    outer_end_ns=outer_end_ns) if route_id == CANDIDATE_ROUTE else \
    _direct_timing(
      outer_start_ns=outer_start_ns,
      route_invoke_end_ns=route_invoke_end_ns,
      output_realize_end_ns=output_realize_end_ns,
      outer_end_ns=outer_end_ns)
  if route_id == DIRECT_ROUTE and \
     invocation.candidate_phase_trace is not None:
    raise ValueError("direct route must not carry candidate phase evidence")

  boundary_identity = validated["timing_boundary"]["boundary_identity"]
  payload = {
    "schema": SCHEMA, "status": "PASS",
    "contract_identity": validated["evidence_identity"],
    "workload_identity": validated["workload"]["identity"],
    "input_identity": validated["common_inputs"]["identity"],
    "queue_mode": queue_mode, "route_id": route_id,
    "executable_identity": executable_identity,
    "timing_boundary_identity": boundary_identity,
    "measurement_source": MEASUREMENT_SOURCE,
    "pre_sync_outside_timed_wall": True,
    "readback_performed": False,
    "timing": timing,
  }
  receipt = {**payload, "evidence_identity": _identity(payload)}
  validate_ffn_gate_up_outer_wall_receipt(
    receipt, contract=validated,
    contract_validation_kwargs=contract_validation_kwargs)
  return OuterWallRunResult(invocation.output, receipt)


def _validate_timing(
    value: Any, *, route_id: str,
    ) -> dict[str, Any]:
  timing = _mapping(value, "outer-wall receipt timing")
  _exact_keys(timing, _TIMING_FIELDS, "outer-wall receipt timing")
  outer_start = _timestamp(timing["outer_start_ns"], "timing.outer_start_ns")
  invoke_end = _timestamp(
    timing["route_invoke_end_ns"], "timing.route_invoke_end_ns")
  realize_end = _timestamp(
    timing["output_realize_end_ns"], "timing.output_realize_end_ns")
  outer_end = _timestamp(timing["outer_end_ns"], "timing.outer_end_ns")
  if not outer_start < invoke_end < realize_end < outer_end:
    raise ValueError("receipt outer-wall clock checkpoints are not monotonic")
  if route_id == CANDIDATE_ROUTE:
    expected = _candidate_timing(
      trace=timing["route_phases"], outer_start_ns=outer_start,
      route_invoke_end_ns=invoke_end, output_realize_end_ns=realize_end,
      outer_end_ns=outer_end)
  else:
    expected = _direct_timing(
      outer_start_ns=outer_start, route_invoke_end_ns=invoke_end,
      output_realize_end_ns=realize_end, outer_end_ns=outer_end)
  if dict(timing) != expected:
    raise ValueError("receipt timing differs from its exhaustive outer-wall reconstruction")
  for field in (
      "output_realization_ns", "final_sync_ns",
      "exhaustive_phase_sum_ns", "complete_role_ns"):
    _duration(timing[field], f"timing.{field}")
  return expected


def validate_ffn_gate_up_outer_wall_receipt(
    value: Any, *, contract: Mapping[str, Any],
    contract_validation_kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
  """Validate one JSON-safe receipt against exact pre-timing authorities."""
  validated = _validated_contract(contract, contract_validation_kwargs)
  row = _mapping(value, "outer-wall timing receipt")
  _exact_keys(row, _TOP_LEVEL_FIELDS, "outer-wall timing receipt")
  if row["schema"] != SCHEMA or row["status"] != "PASS":
    raise ValueError("outer-wall receipt schema/status differs or is legacy")
  if row["contract_identity"] != validated["evidence_identity"] or \
     row["workload_identity"] != validated["workload"]["identity"] or \
     row["input_identity"] != validated["common_inputs"]["identity"]:
    raise ValueError("outer-wall receipt contract/workload/input identity differs")
  queue_mode, route_id = row["queue_mode"], row["route_id"]
  expected_executable = _route_executable(
    validated, queue_mode=queue_mode, route_id=route_id)
  if row["executable_identity"] != expected_executable:
    raise ValueError("outer-wall receipt executable identity differs")
  if row["timing_boundary_identity"] != \
     validated["timing_boundary"]["boundary_identity"] or \
     row["measurement_source"] != MEASUREMENT_SOURCE:
    raise ValueError("outer-wall receipt timing boundary differs")
  if row["pre_sync_outside_timed_wall"] is not True or \
     row["readback_performed"] is not False:
    raise ValueError("outer-wall receipt synchronization/readback policy differs")
  _validate_timing(row["timing"], route_id=route_id)
  payload = {key: item for key, item in row.items() if key != "evidence_identity"}
  if row["evidence_identity"] != _identity(payload):
    raise ValueError("outer-wall receipt content identity differs")
  return dict(row)


__all__ = [
  "CANDIDATE_TRACE_SCHEMA", "MEASUREMENT_SOURCE", "OuterWallRunResult",
  "RouteInvocation", "SCHEMA",
  "run_ffn_gate_up_outer_synchronized_wall",
  "validate_ffn_gate_up_outer_wall_receipt",
]
