"""CPU-only dual-queue paired C8 collector for exact ``ffn_gate_up`` receipts.

The collector consumes already-captured outer synchronized-wall receipts.  It
revalidates the complete-role contract and every receipt before calculating a
median or decision.  PM4 and AQL remain distinct persistent sessions with one
clock identity per session, equal warmups, seeded balanced pair order, and no
retry, fallback, or readback.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import random
import statistics
from typing import Any

from extra.qk.mmq_ffn_gate_up_matched_timing_contract import (
  CANDIDATE_ROUTE, DIRECT_ROUTE, QUEUE_MODES,
  validate_ffn_gate_up_matched_complete_role_timing_contract,
)
from extra.qk.mmq_ffn_gate_up_outer_wall_runner import (
  validate_ffn_gate_up_effective_queue_attestation,
  validate_ffn_gate_up_outer_wall_receipt,
)


SCHEMA = "tinygrad.mmq_q4k_q8_1.ffn_gate_up_c8_paired_sessions.v2"
SESSION_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.ffn_gate_up_c8_queue_session_capture.v2"
RANDOMIZATION_ALGORITHM = "python_random_v1_balanced_pair_order"

_HEX = frozenset("0123456789abcdef")
_SESSION_FIELDS = {
  "schema", "status", "queue_mode", "session_identity", "clock_identity",
  "effective_queue_attestation",
  "warmups_per_route", "paired_round_count", "seed",
  "candidate_warmups", "direct_warmups", "paired_rounds",
  "invocation_counts", "invocation_order_census", "no_retry",
  "no_queue_fallback", "readback_performed", "evidence_identity",
  "clock_monotonic_across_session", "unique_receipt_identities",
  "promotion_evidence_eligible",
}
_SAMPLE_FIELDS = {
  "phase", "route_id", "invocation_index", "pair_index",
  "clock_identity", "receipt",
}
_PAIR_FIELDS = {"pair_index", "order", "candidate", "direct_packed"}
_CENSUS_FIELDS = {
  "ordinal", "phase", "pair_index", "route_id", "invocation_index",
}
_COLLECTION_FIELDS = {
  "schema", "status", "contract_identity", "workload_identity",
  "input_identity", "protocol", "queues", "decision",
  "production_dispatch_changed", "promotion_evidence_eligible",
  "evidence_identity",
}


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


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
    raise ValueError(f"{label} must be an integer >= {minimum}")
  return value


def _number(value: Any, label: str) -> float:
  if not isinstance(value, (int, float)) or isinstance(value, bool) or \
     not math.isfinite(value):
    raise ValueError(f"{label} must be a finite number")
  return float(value)


def _nonempty(value: Any, label: str) -> str:
  if not isinstance(value, str) or not value:
    raise ValueError(f"{label} must be a non-empty string")
  return value


def ffn_gate_up_c8_randomized_orders(
    *, seed: int, round_count: int,
    ) -> list[list[str]]:
  """Return the existing C8 alternating-then-seeded-shuffle pair schedule."""
  seed = _integer(seed, "seed")
  round_count = _integer(round_count, "round_count", minimum=10)
  if round_count % 2:
    raise ValueError("round_count must be even for balanced pair order")
  orders = [
    [CANDIDATE_ROUTE, DIRECT_ROUTE] if index % 2 == 0
    else [DIRECT_ROUTE, CANDIDATE_ROUTE]
    for index in range(round_count)
  ]
  random.Random(seed).shuffle(orders)
  return orders


def _validated_contract(
    contract: Any, contract_validation_kwargs: Any,
    ) -> dict[str, Any]:
  kwargs = _mapping(
    contract_validation_kwargs, "contract validation authorities")
  return validate_ffn_gate_up_matched_complete_role_timing_contract(
    contract, **dict(kwargs))


def _sample(
    value: Any, *, label: str, queue_mode: str, session_clock_identity: str,
    route_id: str, phase: str, invocation_index: int,
    pair_index: int | None, contract: Mapping[str, Any],
    contract_validation_kwargs: Mapping[str, Any],
    effective_queue_attestation: Mapping[str, Any],
    ) -> tuple[dict[str, Any], int]:
  row = _mapping(value, label)
  _exact_keys(row, _SAMPLE_FIELDS, label)
  expected = {
    "phase": phase, "route_id": route_id,
    "invocation_index": invocation_index, "pair_index": pair_index,
    "clock_identity": session_clock_identity,
  }
  for field, expected_value in expected.items():
    if row[field] != expected_value:
      raise ValueError(f"{label}.{field} differs from the invocation census")
  raw_receipt = _mapping(row["receipt"], f"{label}.receipt")
  if raw_receipt.get("queue_mode") != queue_mode or \
     raw_receipt.get("route_id") != route_id or \
     raw_receipt.get("effective_queue_attestation") != \
       effective_queue_attestation:
    raise ValueError(f"{label} queue/route receipt identity differs")
  receipt = validate_ffn_gate_up_outer_wall_receipt(
    raw_receipt, contract=contract,
    contract_validation_kwargs=contract_validation_kwargs)
  if receipt["queue_mode"] != queue_mode or receipt["route_id"] != route_id:
    raise ValueError(f"{label} queue/route receipt identity differs")
  elapsed = _integer(
    receipt["timing"]["complete_role_ns"],
    f"{label}.receipt.timing.complete_role_ns", minimum=1)
  return {
    **expected, "receipt": receipt,
  }, elapsed


def _expected_census(
    *, warmups: int, orders: list[list[str]],
    ) -> list[dict[str, Any]]:
  counts = {CANDIDATE_ROUTE: 0, DIRECT_ROUTE: 0}
  rows: list[dict[str, Any]] = []
  for _warmup_index in range(warmups):
    for route_id in (CANDIDATE_ROUTE, DIRECT_ROUTE):
      rows.append({
        "ordinal": len(rows), "phase": "warmup", "pair_index": None,
        "route_id": route_id, "invocation_index": counts[route_id],
      })
      counts[route_id] += 1
  for pair_index, order in enumerate(orders):
    for route_id in order:
      rows.append({
        "ordinal": len(rows), "phase": "round", "pair_index": pair_index,
        "route_id": route_id, "invocation_index": counts[route_id],
      })
      counts[route_id] += 1
  return rows


def _session(
    value: Any, *, queue_mode: str, contract: Mapping[str, Any],
    contract_validation_kwargs: Mapping[str, Any],
    warmups: int, rounds: int, seed: int,
    ) -> dict[str, Any]:
  label = f"{queue_mode} C8 session"
  row = _mapping(value, label)
  _exact_keys(row, _SESSION_FIELDS, label)
  payload = {key: item for key, item in row.items() if key != "evidence_identity"}
  if row["schema"] != SESSION_SCHEMA or row["status"] != "PASS" or \
     row["queue_mode"] != queue_mode:
    raise ValueError(f"{label} schema/status/queue differs")
  if row["evidence_identity"] != _identity(payload):
    raise ValueError(f"{label} content identity differs")
  session_identity = _content_identity(
    row["session_identity"], f"{label} identity")
  clock_identity = _nonempty(row["clock_identity"], f"{label} clock identity")
  effective_queue_attestation = \
    validate_ffn_gate_up_effective_queue_attestation(
      row["effective_queue_attestation"], queue_mode=queue_mode)
  if row["warmups_per_route"] != warmups or \
     row["paired_round_count"] != rounds or row["seed"] != seed:
    raise ValueError(f"{label} warmup/round/seed protocol differs")
  if row["no_retry"] is not True or row["no_queue_fallback"] is not True or \
     row["readback_performed"] is not False or \
     row["clock_monotonic_across_session"] is not True or \
     row["unique_receipt_identities"] is not True or \
     row["promotion_evidence_eligible"] is not False:
    raise ValueError(f"{label} retry/fallback/readback policy differs")
  candidate_warmups, direct_warmups = \
    row["candidate_warmups"], row["direct_warmups"]
  pairs = row["paired_rounds"]
  if not isinstance(candidate_warmups, list) or \
     not isinstance(direct_warmups, list) or \
     len(candidate_warmups) != warmups or len(direct_warmups) != warmups:
    raise ValueError(f"{label} requires equal complete warmup lists")
  if not isinstance(pairs, list) or len(pairs) != rounds:
    raise ValueError(f"{label} paired-round list is incomplete")
  orders = ffn_gate_up_c8_randomized_orders(
    seed=seed, round_count=rounds)
  expected_counts = {
    CANDIDATE_ROUTE: warmups + rounds,
    DIRECT_ROUTE: warmups + rounds,
  }
  if row["invocation_counts"] != expected_counts:
    raise ValueError(f"{label} invocation counts differ")
  expected_census = _expected_census(warmups=warmups, orders=orders)
  census = row["invocation_order_census"]
  if not isinstance(census, list) or census != expected_census or \
     any(not isinstance(item, Mapping) or set(item) != _CENSUS_FIELDS
         for item in census):
    raise ValueError(f"{label} invocation order census differs")

  candidate_warmup_ns, direct_warmup_ns = [], []
  validated_candidate_warmups, validated_direct_warmups = [], []
  for index in range(warmups):
    sample, elapsed = _sample(
      candidate_warmups[index],
      label=f"{label}.candidate_warmups[{index}]",
      queue_mode=queue_mode, session_clock_identity=clock_identity,
      route_id=CANDIDATE_ROUTE, phase="warmup",
      invocation_index=index, pair_index=None, contract=contract,
      contract_validation_kwargs=contract_validation_kwargs,
      effective_queue_attestation=effective_queue_attestation)
    validated_candidate_warmups.append(sample)
    candidate_warmup_ns.append(elapsed)
    sample, elapsed = _sample(
      direct_warmups[index], label=f"{label}.direct_warmups[{index}]",
      queue_mode=queue_mode, session_clock_identity=clock_identity,
      route_id=DIRECT_ROUTE, phase="warmup", invocation_index=index,
      pair_index=None, contract=contract,
      contract_validation_kwargs=contract_validation_kwargs,
      effective_queue_attestation=effective_queue_attestation)
    validated_direct_warmups.append(sample)
    direct_warmup_ns.append(elapsed)

  validated_pairs, candidate_round_ns, direct_round_ns = [], [], []
  receipt_order_census = []
  for pair_index, raw_pair in enumerate(pairs):
    pair_label = f"{label}.paired_rounds[{pair_index}]"
    pair = _mapping(raw_pair, pair_label)
    _exact_keys(pair, _PAIR_FIELDS, pair_label)
    if pair["pair_index"] != pair_index or pair["order"] != orders[pair_index]:
      raise ValueError(f"{pair_label} index/order differs from seeded schedule")
    candidate, candidate_ns = _sample(
      pair["candidate"], label=f"{pair_label}.candidate",
      queue_mode=queue_mode, session_clock_identity=clock_identity,
      route_id=CANDIDATE_ROUTE, phase="round",
      invocation_index=warmups + pair_index, pair_index=pair_index,
      contract=contract,
      contract_validation_kwargs=contract_validation_kwargs,
      effective_queue_attestation=effective_queue_attestation)
    direct, direct_ns = _sample(
      pair["direct_packed"], label=f"{pair_label}.direct_packed",
      queue_mode=queue_mode, session_clock_identity=clock_identity,
      route_id=DIRECT_ROUTE, phase="round",
      invocation_index=warmups + pair_index, pair_index=pair_index,
      contract=contract,
      contract_validation_kwargs=contract_validation_kwargs,
      effective_queue_attestation=effective_queue_attestation)
    candidate_round_ns.append(candidate_ns)
    direct_round_ns.append(direct_ns)
    samples = {
      CANDIDATE_ROUTE: candidate,
      DIRECT_ROUTE: direct,
    }
    validated_pairs.append({
      "pair_index": pair_index, "order": list(orders[pair_index]),
      "candidate": candidate, "direct_packed": direct,
    })
    for route_id in orders[pair_index]:
      receipt_order_census.append({
        "pair_index": pair_index, "route_id": route_id,
        "invocation_index": warmups + pair_index,
        "receipt_identity": samples[route_id]["receipt"]["evidence_identity"],
      })

  candidate_median = statistics.median(candidate_round_ns)
  direct_median = statistics.median(direct_round_ns)
  speedup = direct_median / candidate_median
  samples_by_key = {}
  for sample in (
      validated_candidate_warmups + validated_direct_warmups +
      [pair[route] for pair in validated_pairs
       for route in ("candidate", "direct_packed")]):
    key = (
      sample["phase"], sample["pair_index"], sample["route_id"],
      sample["invocation_index"])
    if key in samples_by_key:
      raise ValueError(f"{label} duplicate invocation sample key")
    samples_by_key[key] = sample
  prior_outer_end = None
  receipt_identities = set()
  for census_row in expected_census:
    key = (
      census_row["phase"], census_row["pair_index"], census_row["route_id"],
      census_row["invocation_index"])
    receipt = samples_by_key[key]["receipt"]
    outer_start = receipt["timing"]["outer_start_ns"]
    outer_end = receipt["timing"]["outer_end_ns"]
    if receipt["evidence_identity"] in receipt_identities:
      raise ValueError(f"{label} timing receipt identity repeated")
    if prior_outer_end is not None and outer_start <= prior_outer_end:
      raise ValueError(f"{label} clock is not monotonic across the session")
    prior_outer_end = outer_end
    receipt_identities.add(receipt["evidence_identity"])
  return {
    "queue_mode": queue_mode, "session_identity": session_identity,
    "clock_identity": clock_identity, "same_clock_within_queue": True,
    "effective_queue_attestation": effective_queue_attestation,
    "warmups_per_route": warmups, "paired_round_count": rounds,
    "seed": seed, "randomization_algorithm": RANDOMIZATION_ALGORITHM,
    "randomized_orders": orders,
    "invocation_counts": expected_counts,
    "invocation_order_census": expected_census,
    "round_receipt_order_census": receipt_order_census,
    "raw_samples_ns": {
      "candidate_warmups": candidate_warmup_ns,
      "direct_warmups": direct_warmup_ns,
      "candidate_rounds": candidate_round_ns,
      "direct_rounds": direct_round_ns,
    },
    "statistics": {
      "statistic": "median",
      "candidate_median_complete_role_ns": candidate_median,
      "direct_median_complete_role_ns": direct_median,
      "direct_over_candidate_speedup": speedup,
    },
    "validated_receipts": {
      "candidate_warmups": validated_candidate_warmups,
      "direct_warmups": validated_direct_warmups,
      "paired_rounds": validated_pairs,
    },
    "all_receipts_validated": True,
    "no_retry": True, "no_queue_fallback": True,
    "readback_performed": False,
    "clock_monotonic_across_session": True,
    "unique_receipt_identities": True,
    "promotion_evidence_eligible": False,
  }


def collect_ffn_gate_up_c8_paired_sessions(
    *, contract: Mapping[str, Any],
    contract_validation_kwargs: Mapping[str, Any],
    queue_sessions: Mapping[str, Mapping[str, Any]],
    warmups: int, rounds: int, seed: int,
    required_speedup: int | float = 1.0,
    ) -> dict[str, Any]:
  """Validate both queue sessions, then emit one honest evaluated decision."""
  validated_contract = _validated_contract(
    contract, contract_validation_kwargs)
  warmups = _integer(warmups, "warmups", minimum=3)
  rounds = _integer(rounds, "rounds", minimum=10)
  seed = _integer(seed, "seed")
  required_speedup = _number(required_speedup, "required_speedup")
  if required_speedup < 1.0:
    raise ValueError("required_speedup must be at least 1.0")
  sessions = _mapping(queue_sessions, "queue sessions")
  if set(sessions) != set(QUEUE_MODES):
    raise ValueError(f"queue sessions must contain exactly {QUEUE_MODES!r}")

  queues = {
    queue: _session(
      sessions[queue], queue_mode=queue, contract=validated_contract,
      contract_validation_kwargs=contract_validation_kwargs,
      warmups=warmups, rounds=rounds, seed=seed)
    for queue in QUEUE_MODES
  }
  if queues["PM4"]["session_identity"] == queues["AQL"]["session_identity"]:
    raise ValueError("PM4 and AQL session identities must be distinct")
  if queues["PM4"]["effective_queue_attestation"]["evidence_identity"] == \
     queues["AQL"]["effective_queue_attestation"]["evidence_identity"]:
    raise ValueError("PM4 and AQL effective queue attestations must be distinct")
  queue_wins = {
    queue: (
      queues[queue]["statistics"]["candidate_median_complete_role_ns"] <
      queues[queue]["statistics"]["direct_median_complete_role_ns"] and
      queues[queue]["statistics"]["direct_over_candidate_speedup"] >=
      required_speedup)
    for queue in QUEUE_MODES
  }
  evaluated_win = all(queue_wins.values())
  decision = {
    "status": "EVALUATED_WIN" if evaluated_win else "EVALUATED_NO_WIN",
    "evaluated": True,
    "candidate_wins_both_queues": evaluated_win,
    "required_speedup": required_speedup,
    "queue_wins": queue_wins,
    "selected_route": CANDIDATE_ROUTE if evaluated_win else DIRECT_ROUTE,
    "promotion_evidence_eligible": False,
    "rule": (
      "candidate median complete-role outer-wall time must be lower and "
      "direct/candidate speedup must meet the threshold on both PM4 and AQL"),
  }
  payload = {
    "schema": SCHEMA, "status": "PASS",
    "contract_identity": validated_contract["evidence_identity"],
    "workload_identity": validated_contract["workload"]["identity"],
    "input_identity": validated_contract["common_inputs"]["identity"],
    "protocol": {
      "distinct_queue_sessions": True,
      "same_clock_identity_within_each_queue": True,
      "warmups_per_route": warmups, "paired_rounds": rounds, "seed": seed,
      "randomization_algorithm": RANDOMIZATION_ALGORITHM,
      "no_retry": True, "no_queue_fallback": True,
      "readback_performed": False,
      "effective_queue_attestation_bound": True,
      "clock_monotonic_across_each_session": True,
      "unique_receipt_identities": True,
    },
    "queues": queues, "decision": decision,
    "production_dispatch_changed": False,
    "promotion_evidence_eligible": False,
  }
  return {**payload, "evidence_identity": _identity(payload)}


def validate_ffn_gate_up_c8_paired_sessions(
    value: Any, *, contract: Mapping[str, Any],
    contract_validation_kwargs: Mapping[str, Any],
    queue_sessions: Mapping[str, Mapping[str, Any]],
    warmups: int, rounds: int, seed: int,
    required_speedup: int | float = 1.0,
    ) -> dict[str, Any]:
  """Rebuild a collection from exact inputs and reject any retained drift."""
  row = _mapping(value, "paired C8 collection")
  _exact_keys(row, _COLLECTION_FIELDS, "paired C8 collection")
  if row["schema"] != SCHEMA:
    raise ValueError("paired C8 collection schema differs or is legacy")
  expected = collect_ffn_gate_up_c8_paired_sessions(
    contract=contract,
    contract_validation_kwargs=contract_validation_kwargs,
    queue_sessions=queue_sessions, warmups=warmups, rounds=rounds,
    seed=seed, required_speedup=required_speedup)
  if dict(row) != expected:
    raise ValueError("paired C8 collection differs from exact session evidence")
  return dict(row)


__all__ = [
  "RANDOMIZATION_ALGORITHM", "SCHEMA", "SESSION_SCHEMA",
  "collect_ffn_gate_up_c8_paired_sessions",
  "ffn_gate_up_c8_randomized_orders",
  "validate_ffn_gate_up_c8_paired_sessions",
]
