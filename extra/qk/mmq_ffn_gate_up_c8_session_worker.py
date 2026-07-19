"""CPU-only persistent queue-session worker for matched ``ffn_gate_up`` C8.

Low-level candidate and direct callbacks are mandatory and executable-bound.
The worker validates the complete contract before any callback, then performs
candidate/direct warmups followed by the seeded paired schedule in one
persistent Python session.  Every invocation uses the exact outer-wall runner.
There is no receipt-runner default, retry, fallback, or readback.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from extra.qk.mmq_ffn_gate_up_c8_paired_sessions import (
  SESSION_SCHEMA, ffn_gate_up_c8_randomized_orders,
)
from extra.qk.mmq_ffn_gate_up_matched_timing_contract import (
  CANDIDATE_ROUTE, DIRECT_ROUTE, QUEUE_MODES,
  validate_ffn_gate_up_matched_complete_role_timing_contract,
)
from extra.qk.mmq_ffn_gate_up_outer_wall_runner import (
  RouteInvocation, run_ffn_gate_up_outer_synchronized_wall,
  seal_ffn_gate_up_effective_queue_attestation,
)


BLOCKED_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.ffn_gate_up_c8_queue_session_blocked.v2"
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class LowLevelRouteCallbacks:
  """Explicit callbacks for one executable-bound low-level route."""

  queue_mode: str
  input_identity: str
  executable_identity: str
  invoke: Callable[[], RouteInvocation]
  realize_output: Callable[[Any], Any]
  attest_post_sync: Callable[[Any, str], Mapping[str, Any]]


@dataclass(frozen=True)
class AdaptedLowLevelRoutes:
  """Worker callbacks retained directly from validated runtime route types."""

  candidate: LowLevelRouteCallbacks
  direct_packed: LowLevelRouteCallbacks


def _canonical(value: Any) -> bytes:
  return json.dumps(
    value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity(value: Any) -> str:
  return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
  if not isinstance(value, Mapping):
    raise ValueError(f"{label} must be a mapping")
  return value


def _content_identity(value: Any, label: str) -> str:
  if not isinstance(value, str) or not value.startswith("sha256:") or \
     len(value) != 71 or any(char not in _HEX for char in value[7:]):
    raise ValueError(f"{label} must be a sha256 content identity")
  return value


def _nonempty(value: Any, label: str) -> str:
  if not isinstance(value, str) or not value:
    raise ValueError(f"{label} must be a non-empty string")
  return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
    raise ValueError(f"{label} must be an integer >= {minimum}")
  return value


def _callbacks(
    value: Any, *, label: str, queue_mode: str, input_identity: str,
    expected_executable_identity: str,
    ) -> LowLevelRouteCallbacks:
  if not isinstance(value, LowLevelRouteCallbacks):
    raise TypeError(
      f"{label} must be explicit LowLevelRouteCallbacks; "
      "legacy receipt-runner defaults are unsupported")
  if value.queue_mode != queue_mode:
    raise ValueError(f"{label} queue mode drifted")
  if _content_identity(
      value.input_identity, f"{label} input identity") != input_identity:
    raise ValueError(f"{label} input identity drifted")
  identity = _content_identity(
    value.executable_identity, f"{label} executable identity")
  if identity != expected_executable_identity:
    raise ValueError(f"{label} executable identity drifted")
  if not callable(value.invoke) or not callable(value.realize_output) or \
     not callable(value.attest_post_sync):
    raise TypeError(
      f"{label} invoke, realize_output, and attest_post_sync callbacks must "
      "be callable")
  return value


def _census_row(
    *, ordinal: int, phase: str, pair_index: int | None,
    route_id: str, invocation_index: int,
    ) -> dict[str, Any]:
  return {
    "ordinal": ordinal, "phase": phase, "pair_index": pair_index,
    "route_id": route_id, "invocation_index": invocation_index,
  }


def adapt_ffn_gate_up_runtime_routes(
    routes: Any, *, queue_mode: str, input_identity: str,
    candidate_executable_identity: str,
    direct_executable_identity: str,
    ) -> AdaptedLowLevelRoutes:
  """Fail-closed adapter from runtime-owned typed routes to worker callbacks.

  The local import keeps this worker importable without loading the broader
  runtime composition.  No wrapper lambda or default realizer is inserted:
  the runtime-owned ``invoke`` and typed no-readback ``realize_output`` objects
  are retained by identity.
  """
  from extra.qk.mmq_ffn_gate_up_c8_runtime import (
    FfnGateUpOuterWallRoutes, FfnGateUpRouteCallback,
  )
  if not isinstance(routes, FfnGateUpOuterWallRoutes):
    raise TypeError(
      "runtime routes must be validated FfnGateUpOuterWallRoutes")
  input_identity = _content_identity(input_identity, "runtime input identity")
  candidate_executable_identity = _content_identity(
    candidate_executable_identity, "runtime candidate executable identity")
  direct_executable_identity = _content_identity(
    direct_executable_identity, "runtime direct executable identity")
  validated = routes.validate(
    queue_mode=queue_mode, input_identity=input_identity,
    candidate_executable_identity=candidate_executable_identity,
    direct_executable_identity=direct_executable_identity)

  def adapt(
      route: Any, *, route_id: str, executable_identity: str,
      ) -> LowLevelRouteCallbacks:
    if not isinstance(route, FfnGateUpRouteCallback):
      raise TypeError(
        f"{route_id} runtime route must be FfnGateUpRouteCallback")
    route.validate(
      route_id=route_id, queue_mode=queue_mode, input_identity=input_identity,
      executable_identity=executable_identity)
    adapted = LowLevelRouteCallbacks(
      queue_mode=route.queue_mode, input_identity=route.input_identity,
      executable_identity=route.executable_identity,
      invoke=route.invoke, realize_output=route.realize_output,
      attest_post_sync=route.attest_post_sync)
    if adapted.invoke is not route.invoke or \
       adapted.realize_output is not route.realize_output or \
       adapted.attest_post_sync is not route.attest_post_sync:
      raise RuntimeError(
        f"{route_id} runtime callback identity changed during adaptation")
    return adapted

  return AdaptedLowLevelRoutes(
    candidate=adapt(
      validated.candidate, route_id=CANDIDATE_ROUTE,
      executable_identity=candidate_executable_identity),
    direct_packed=adapt(
      validated.direct_packed, route_id=DIRECT_ROUTE,
      executable_identity=direct_executable_identity))


def run_ffn_gate_up_c8_persistent_queue_session_worker(
    *, contract: Mapping[str, Any],
    contract_validation_kwargs: Mapping[str, Any],
    queue_mode: str, session_identity: str, clock_identity: str,
    candidate_executable_identity: str, direct_executable_identity: str,
    pre_sync: Callable[[], Any], post_sync: Callable[[], Any],
    clock_ns: Callable[[], int],
    effective_queue_attestation: Mapping[str, Any],
    host_io_census: Callable[[], Mapping[str, Any]],
    candidate_callbacks: LowLevelRouteCallbacks,
    direct_callbacks: LowLevelRouteCallbacks,
    warmups: int, rounds: int, seed: int,
    ) -> dict[str, Any]:
  """Produce one sealed queue capture accepted by the paired C8 collector."""
  authorities = dict(_mapping(
    contract_validation_kwargs, "contract validation authorities"))
  validated = validate_ffn_gate_up_matched_complete_role_timing_contract(
    contract, **authorities)
  if queue_mode not in QUEUE_MODES:
    raise ValueError(f"queue_mode must be one of {QUEUE_MODES!r}")
  sealed_queue_attestation = seal_ffn_gate_up_effective_queue_attestation(
    effective_queue_attestation, queue_mode=queue_mode)
  session_identity = _content_identity(session_identity, "session identity")
  clock_identity = _nonempty(clock_identity, "clock identity")
  warmups = _integer(warmups, "warmups", minimum=3)
  rounds = _integer(rounds, "rounds", minimum=10)
  seed = _integer(seed, "seed")
  if not all(callable(callback) for callback in (
      pre_sync, post_sync, clock_ns, host_io_census)):
    raise TypeError(
      "pre_sync, post_sync, clock_ns, and host_io_census must be callable")

  expected_input = validated["common_inputs"]["identity"]
  expected_candidate = validated["candidate"]["candidate_executable_identity"]
  expected_direct = validated["direct_packed"]["queue_qualifications"][
    queue_mode]["executable_identity"]
  candidate_executable_identity = _content_identity(
    candidate_executable_identity, "candidate executable identity")
  direct_executable_identity = _content_identity(
    direct_executable_identity, "direct executable identity")
  if candidate_executable_identity != expected_candidate:
    raise ValueError("candidate executable identity differs from contract")
  if direct_executable_identity != expected_direct:
    raise ValueError("direct executable identity differs from contract")
  route_callbacks = {
    CANDIDATE_ROUTE: _callbacks(
      candidate_callbacks, label="candidate callbacks",
      queue_mode=queue_mode, input_identity=expected_input,
      expected_executable_identity=expected_candidate),
    DIRECT_ROUTE: _callbacks(
      direct_callbacks, label="direct callbacks",
      queue_mode=queue_mode, input_identity=expected_input,
      expected_executable_identity=expected_direct),
  }
  orders = ffn_gate_up_c8_randomized_orders(
    seed=seed, round_count=rounds)

  counts = {CANDIDATE_ROUTE: 0, DIRECT_ROUTE: 0}
  candidate_warmups: list[dict[str, Any]] = []
  direct_warmups: list[dict[str, Any]] = []
  pairs: list[dict[str, Any]] = []
  census: list[dict[str, Any]] = []
  failed_invocation: dict[str, Any] | None = None
  prior_outer_end_ns: int | None = None
  receipt_identities: set[str] = set()

  def invoke_one(
      *, route_id: str, phase: str, pair_index: int | None,
      ) -> dict[str, Any] | None:
    nonlocal failed_invocation, prior_outer_end_ns
    invocation_index = counts[route_id]
    callbacks = route_callbacks[route_id]
    attempted = _census_row(
      ordinal=len(census), phase=phase, pair_index=pair_index,
      route_id=route_id, invocation_index=invocation_index)
    try:
      result = run_ffn_gate_up_outer_synchronized_wall(
        contract=validated, contract_validation_kwargs=authorities,
        queue_mode=queue_mode, route_id=route_id,
        executable_identity=callbacks.executable_identity,
        pre_sync=pre_sync, invoke_route=callbacks.invoke,
        realize_output=callbacks.realize_output, post_sync=post_sync,
        attest_post_sync=callbacks.attest_post_sync,
        host_io_census=host_io_census,
        effective_queue_attestation=sealed_queue_attestation,
        clock_ns=clock_ns)
      receipt = dict(result.receipt)
      outer_start = receipt["timing"]["outer_start_ns"]
      outer_end = receipt["timing"]["outer_end_ns"]
      receipt_identity = receipt["evidence_identity"]
      if receipt_identity in receipt_identities:
        raise ValueError("persistent session timing receipt identity repeated")
      if prior_outer_end_ns is not None and outer_start <= prior_outer_end_ns:
        raise ValueError(
          "persistent session clock did not advance across invocations")
      prior_outer_end_ns = outer_end
      receipt_identities.add(receipt_identity)
      # Drop the worker's output reference only after the outer end timestamp.
      # No readback or release callback is invoked.
      del result
    except Exception as exc:
      failed_invocation = {
        **attempted, "exception": type(exc).__name__, "error": str(exc),
        "audit_evidence":
          dict(exc.evidence) if isinstance(
            getattr(exc, "evidence", None), Mapping) else None,
      }
      return None
    census.append(attempted)
    counts[route_id] += 1
    return {
      "phase": phase, "route_id": route_id,
      "invocation_index": invocation_index, "pair_index": pair_index,
      "clock_identity": clock_identity, "receipt": receipt,
    }

  def blocked() -> dict[str, Any]:
    assert failed_invocation is not None
    audit_evidence = failed_invocation.get("audit_evidence")
    observed_readback = bool(
      isinstance(audit_evidence, Mapping) and
      audit_evidence.get("delta", {}).get("readback_count", 0))
    payload = {
      "schema": BLOCKED_SCHEMA, "status": "BLOCKED",
      "exact_blocker": (
        f"{queue_mode} persistent C8 failed at "
        f"{failed_invocation['phase']} "
        f"{failed_invocation['route_id']}"
        f"[{failed_invocation['invocation_index']}]: "
        f"{failed_invocation['exception']}: "
        f"{failed_invocation['error']}"),
      "queue_mode": queue_mode, "session_identity": session_identity,
      "clock_identity": clock_identity,
      "effective_queue_attestation": sealed_queue_attestation,
      "contract_identity": validated["evidence_identity"],
      "failed_invocation": dict(failed_invocation),
      "completed_invocation_counts": dict(counts),
      "completed_invocation_order_census": list(census),
      "completed_candidate_warmups": list(candidate_warmups),
      "completed_direct_warmups": list(direct_warmups),
      "completed_paired_rounds": list(pairs),
      "warmups_per_route": warmups, "paired_round_count": rounds,
      "seed": seed, "no_retry": True, "retry_count": 0,
      "no_queue_fallback": True, "readback_performed": observed_readback,
      "clock_monotonic_across_session": False,
      "unique_receipt_identities": False,
      "collector_eligible": False,
    }
    return {**payload, "evidence_identity": _identity(payload)}

  for _warmup_index in range(warmups):
    sample = invoke_one(
      route_id=CANDIDATE_ROUTE, phase="warmup", pair_index=None)
    if sample is None: return blocked()
    candidate_warmups.append(sample)
    sample = invoke_one(
      route_id=DIRECT_ROUTE, phase="warmup", pair_index=None)
    if sample is None: return blocked()
    direct_warmups.append(sample)

  for pair_index, order in enumerate(orders):
    row: dict[str, Any] = {
      "pair_index": pair_index, "order": list(order)}
    for route_id in order:
      sample = invoke_one(
        route_id=route_id, phase="round", pair_index=pair_index)
      if sample is None: return blocked()
      row["candidate" if route_id == CANDIDATE_ROUTE
          else "direct_packed"] = sample
    pairs.append(row)

  payload = {
    "schema": SESSION_SCHEMA, "status": "PASS",
    "queue_mode": queue_mode, "session_identity": session_identity,
    "clock_identity": clock_identity,
    "effective_queue_attestation": sealed_queue_attestation,
    "warmups_per_route": warmups,
    "paired_round_count": rounds, "seed": seed,
    "candidate_warmups": candidate_warmups,
    "direct_warmups": direct_warmups, "paired_rounds": pairs,
    "invocation_counts": counts,
    "invocation_order_census": census,
    "no_retry": True, "no_queue_fallback": True,
    "readback_performed": False,
    "clock_monotonic_across_session": True,
    "unique_receipt_identities": True,
    "promotion_evidence_eligible": False,
  }
  return {**payload, "evidence_identity": _identity(payload)}


__all__ = [
  "AdaptedLowLevelRoutes", "BLOCKED_SCHEMA", "LowLevelRouteCallbacks",
  "adapt_ffn_gate_up_runtime_routes",
  "run_ffn_gate_up_c8_persistent_queue_session_worker",
]
