"""Matched live-timing collector for one frozen staged MMQ role.

The collector owns measurement order and evidence validation, while execution
remains injectable.  A live candidate runner must use the frozen staged-family
executor and return an exhaustive timing receipt.  A live fallback runner must
wrap tinygrad's production ``_run_direct_packed_baseline`` path and return the
exact executable evidence used for that synchronized call.

This module does not import or initialize a Device.  Unit tests can exercise
the complete collection and certification path with CPU-only runner stubs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

from extra.qk.mmq_exact_role_spec import DEFAULT_INVENTORY, exact_role_spec
from extra.qk.mmq_frozen_staged_family import (
  QUEUE_MODES, FrozenStagedFamily, load_frozen_staged_family_manifest,
)
from extra.qk.mmq_staged_c7_c8_contract import (
  build_staged_c8_timing_contract, staged_c8_randomized_orders,
  staged_logical_memory_requirements,
)


SCHEMA = "tinygrad.mmq_q4k_q8_1.frozen_staged_c8_timing_collection.v1"
CANDIDATE_RECEIPT_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.frozen_staged_candidate_timing_receipt.v1"
FALLBACK_RECEIPT_SCHEMA = "tinygrad.direct_packed.complete_role_timing_receipt.v1"
MEASUREMENT_SOURCE = "synchronized_wall"
RANDOMIZATION_ALGORITHM = "python_random_v1_balanced_pair_order"
TRANSFER_NAMES = ("q4", "q8_values", "q8_scales", "q8_original_sums")
TRANSFER_CATEGORIES = (
  "compact_q4_stage", "compact_q8_values_stage",
  "compact_q8_scales_stage", "compact_q8_sums_stage",
)
PHASE_SEMANTICS = {
  "gather": "all per-epoch host/source-view work outside the four transfer calls before staging synchronization",
  "transfers": "four separately timed AMD SDMA enqueue calls in ABI slot order",
  "staging_sync": "Device.synchronize returning before target submission",
  "dispatch": "runtime invocation with wait=True plus target argument/accumulation work",
  "dispatch_sync": "post-dispatch Device.synchronize plus phase-isolation receipt closure",
  "final_sync": "explicit final Device.synchronize after the complete epoch sequence",
}
STAGED_CANDIDATE_FAILURE_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.frozen_staged_candidate_failure.v1"

TimingRunner = Callable[..., Mapping[str, Any]]


class StagedCandidateExecutionError(ValueError):
  """Compact JSON-safe evidence for one failed staged timing invocation."""

  def __init__(self, message: str, failure_evidence: Mapping[str, Any]):
    super().__init__(message)
    self.failure_evidence = dict(failure_evidence)


def _canonical(value: Any) -> bytes:
  return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity(value: Any) -> str:
  return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _nonempty(value: Any, label: str) -> str:
  if not isinstance(value, str) or not value:
    raise ValueError(f"{label} must be a non-empty string")
  return value


def _number(value: Any, label: str, *, positive: bool = False) -> float:
  if not isinstance(value, (int, float)) or isinstance(value, bool) or \
     not math.isfinite(value) or value < 0 or (positive and value <= 0):
    qualifier = "positive" if positive else "non-negative"
    raise ValueError(f"{label} must be a finite {qualifier} number")
  return float(value)


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
    raise ValueError(f"{label} must be an integer >= {minimum}")
  return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
  if set(value) != expected:
    raise ValueError(
      f"{label} fields differ: expected {sorted(expected)!r}, got {sorted(value)!r}")


@dataclass(frozen=True)
class QueueTimingAuthority:
  """Session and clock identities for one queue-specific matched run."""

  session_identity: str
  clock_identity: str

  def validate(self, queue_mode: str) -> "QueueTimingAuthority":
    if queue_mode not in QUEUE_MODES:
      raise ValueError(f"queue_mode must be one of {QUEUE_MODES!r}")
    _nonempty(self.session_identity, f"{queue_mode}.session_identity")
    _nonempty(self.clock_identity, f"{queue_mode}.clock_identity")
    return self


@dataclass(frozen=True)
class QueueTimingRunners:
  """Candidate and direct-packed execution callables for one queue mode."""

  candidate: TimingRunner
  direct_packed: TimingRunner

  def validate(self, queue_mode: str) -> "QueueTimingRunners":
    if not callable(self.candidate) or not callable(self.direct_packed):
      raise TypeError(f"{queue_mode} timing runners must be callable")
    return self


def _logical(family: FrozenStagedFamily) -> tuple[int, dict[str, int]]:
  logical = staged_logical_memory_requirements(family)
  components = logical["components"]
  compact = {name: components[name] for name in TRANSFER_CATEGORIES}
  return logical["epoch_count"], compact


def _candidate_receipt(
    value: Any, *, family: FrozenStagedFamily, queue_mode: str,
    clock_identity: str, candidate_executable_identity: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
  label = f"{queue_mode} staged candidate timing receipt"
  if not isinstance(value, Mapping): raise ValueError(f"{label} must be a mapping")
  _exact_keys(value, {
    "schema", "status", "family_identity", "program_key", "binary_sha256",
    "candidate_executable_identity", "queue_mode", "measurement_source",
    "clock_identity", "phase_semantics", "output_initialization_ms", "epochs", "final_sync_ms",
    "complete_role_ms",
  }, label)
  checks = {
    "schema": value["schema"] == CANDIDATE_RECEIPT_SCHEMA,
    "status": value["status"] == "PASS",
    "family_identity": value["family_identity"] == family.family_identity,
    "program_key": value["program_key"] == family.binding.program_key,
    "binary_sha256": value["binary_sha256"] == family.binding.binary_sha256,
    "candidate_executable_identity":
      value["candidate_executable_identity"] == candidate_executable_identity,
    "queue_mode": value["queue_mode"] == queue_mode,
    "measurement_source": value["measurement_source"] == MEASUREMENT_SOURCE,
    "clock_identity": value["clock_identity"] == clock_identity,
    "phase_semantics": value["phase_semantics"] == PHASE_SEMANTICS,
  }
  if not all(checks.values()):
    failed = sorted(name for name, passed in checks.items() if not passed)
    raise ValueError(f"{label} identity or execution checks failed: {failed!r}")

  epochs, compact = _logical(family)
  rows = value["epochs"]
  if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or len(rows) != epochs:
    raise ValueError(f"{label}.epochs must enumerate all {epochs} epochs")
  expected_transfers = tuple(zip(range(1, 5), TRANSFER_NAMES, TRANSFER_CATEGORIES))
  normalized_rows, phase_sum = [], 0.0
  for ordinal, row in enumerate(rows):
    row_label = f"{label}.epochs[{ordinal}]"
    if not isinstance(row, Mapping): raise ValueError(f"{row_label} must be a mapping")
    _exact_keys(row, {
      "ordinal", "gather_ms", "transfers", "staging_sync_ms",
      "dispatch_ms", "dispatch_sync_ms",
    }, row_label)
    if row["ordinal"] != ordinal:
      raise ValueError(f"{row_label}.ordinal differs from the complete epoch order")
    transfers = row["transfers"]
    if not isinstance(transfers, Sequence) or isinstance(transfers, (str, bytes)) or \
       len(transfers) != 4:
      raise ValueError(f"{row_label}.transfers must enumerate exactly four copies")
    normalized_transfers, transfer_ms = [], 0.0
    for index, (transfer, expected) in enumerate(zip(transfers, expected_transfers)):
      transfer_label = f"{row_label}.transfers[{index}]"
      if not isinstance(transfer, Mapping):
        raise ValueError(f"{transfer_label} must be a mapping")
      _exact_keys(transfer, {"slot", "name", "category", "nbytes", "elapsed_ms"}, transfer_label)
      slot, name, category = expected
      if (transfer["slot"], transfer["name"], transfer["category"]) != expected:
        raise ValueError(f"{transfer_label} differs from ABI slot/name/category {expected!r}")
      if transfer["nbytes"] != compact[category]:
        raise ValueError(f"{transfer_label}.nbytes differs from the compact staged family")
      elapsed = _number(transfer["elapsed_ms"], f"{transfer_label}.elapsed_ms")
      transfer_ms += elapsed
      normalized_transfers.append({
        "slot": slot, "name": name, "category": category,
        "nbytes": compact[category], "elapsed_ms": elapsed,
      })
    timings = {
      field: _number(row[field], f"{row_label}.{field}")
      for field in ("gather_ms", "staging_sync_ms", "dispatch_ms", "dispatch_sync_ms")
    }
    phase_sum += sum(timings.values()) + transfer_ms
    normalized_rows.append({
      "ordinal": ordinal, **timings, "transfers": normalized_transfers,
      "transfer_ms": transfer_ms,
    })
  initialization = _number(value["output_initialization_ms"], f"{label}.output_initialization_ms")
  final_sync = _number(value["final_sync_ms"], f"{label}.final_sync_ms")
  complete = _number(value["complete_role_ms"], f"{label}.complete_role_ms", positive=True)
  expected_complete = initialization + phase_sum + final_sync
  if not math.isclose(complete, expected_complete, rel_tol=1e-12, abs_tol=1e-9):
    raise ValueError(f"{label}.complete_role_ms differs from the exhaustive phase sum")

  contract_epochs = [{
    "ordinal": row["ordinal"], "gather_ms": row["gather_ms"],
    "transfer_ms": row["transfer_ms"], "staging_sync_ms": row["staging_sync_ms"],
    "dispatch_ms": row["dispatch_ms"], "dispatch_sync_ms": row["dispatch_sync_ms"],
    "staged_bytes": sum(compact.values()), "staged_components": dict(compact),
    "copy_count": 4,
  } for row in normalized_rows]
  contract = {
    "output_initialization_ms": initialization, "epochs": contract_epochs,
    "final_sync_ms": final_sync, "complete_role_ms": complete,
  }
  detail = {
    **{key: value[key] for key in (
      "schema", "status", "family_identity", "program_key", "binary_sha256",
      "candidate_executable_identity", "queue_mode", "measurement_source",
      "clock_identity", "phase_semantics",
    )},
    "output_initialization_ms": initialization, "epochs": normalized_rows,
    "final_sync_ms": final_sync, "complete_role_ms": complete,
    "exhaustive_phase_sum_verified": True,
  }
  return contract, detail


def _fallback_receipt(
    value: Any, *, queue_mode: str, clock_identity: str,
    ) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
  label = f"{queue_mode} direct_packed timing receipt"
  if not isinstance(value, Mapping): raise ValueError(f"{label} must be a mapping")
  _exact_keys(value, {
    "schema", "status", "queue_mode", "measurement_source", "clock_identity",
    "complete_role_ms", "fallback_evidence",
  }, label)
  checks = {
    "schema": value["schema"] == FALLBACK_RECEIPT_SCHEMA,
    "status": value["status"] == "PASS",
    "queue_mode": value["queue_mode"] == queue_mode,
    "measurement_source": value["measurement_source"] == MEASUREMENT_SOURCE,
    "clock_identity": value["clock_identity"] == clock_identity,
  }
  if not all(checks.values()):
    failed = sorted(name for name, passed in checks.items() if not passed)
    raise ValueError(f"{label} identity or execution checks failed: {failed!r}")
  complete = _number(value["complete_role_ms"], f"{label}.complete_role_ms", positive=True)
  fallback_evidence = value["fallback_evidence"]
  if not isinstance(fallback_evidence, Mapping):
    raise ValueError(f"{label}.fallback_evidence must be a mapping")
  detail = {**dict(value), "complete_role_ms": complete}
  return {"complete_role_ms": complete}, dict(fallback_evidence), detail


def _invoke(
    runner: TimingRunner, *, queue_mode: str, route: str, phase: str,
    invocation_index: int, pair_index: int | None, family: FrozenStagedFamily,
    clock_identity: str, candidate_executable_identity: str,
    ) -> Mapping[str, Any]:
  result = runner(
    queue_mode=queue_mode, route=route, phase=phase,
    invocation_index=invocation_index, pair_index=pair_index,
    family=family, prefix_epochs=family.binding.role_spec.epochs,
    clock_identity=clock_identity,
    candidate_executable_identity=candidate_executable_identity,
  )
  if not isinstance(result, Mapping):
    raise ValueError(f"{queue_mode} {route} runner returned no mapping")
  return result


def _collect_queue(
    *, family: FrozenStagedFamily, queue_mode: str,
    c6: Mapping[str, Any], authority: QueueTimingAuthority,
    runners: QueueTimingRunners, warmups: int, rounds: int, seed: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
  authority.validate(queue_mode)
  runners.validate(queue_mode)
  executable = _nonempty(
    c6.get("candidate_executable_identity"),
    "C6 candidate_executable_identity")
  queue_correctness, comparators = c6.get("queue_correctness"), c6.get("queue_comparators")
  if not isinstance(queue_correctness, Mapping) or not isinstance(comparators, Mapping):
    raise ValueError("C6 queue correctness/comparator identities are missing")
  warmup_details = {"staged_candidate": [], "direct_packed": []}
  fallback_identity = None
  candidate_invocations = fallback_invocations = 0
  for warmup_index in range(warmups):
    candidate_raw = _invoke(
      runners.candidate, queue_mode=queue_mode, route="staged_candidate",
      phase="warmup", invocation_index=candidate_invocations,
      pair_index=None, family=family, clock_identity=authority.clock_identity,
      candidate_executable_identity=executable)
    candidate_invocations += 1
    _, candidate_detail = _candidate_receipt(
      candidate_raw, family=family, queue_mode=queue_mode,
      clock_identity=authority.clock_identity,
      candidate_executable_identity=executable)
    warmup_details["staged_candidate"].append(candidate_detail)

    fallback_raw = _invoke(
      runners.direct_packed, queue_mode=queue_mode, route="direct_packed",
      phase="warmup", invocation_index=fallback_invocations,
      pair_index=None, family=family, clock_identity=authority.clock_identity,
      candidate_executable_identity=executable)
    fallback_invocations += 1
    _, evidence, fallback_detail = _fallback_receipt(
      fallback_raw, queue_mode=queue_mode, clock_identity=authority.clock_identity)
    if fallback_identity is None: fallback_identity = evidence
    elif evidence != fallback_identity:
      raise ValueError(f"{queue_mode} direct_packed executable evidence changed across warmups")
    warmup_details["direct_packed"].append(fallback_detail)

  orders = staged_c8_randomized_orders(seed=seed, round_count=rounds)
  paired_contract, paired_details = [], []
  for pair_index, order in enumerate(orders):
    contract_pair: dict[str, Any] = {"pair_index": pair_index, "order": order}
    detail_pair: dict[str, Any] = {"pair_index": pair_index, "order": order}
    for route in order:
      if route == "staged_candidate":
        raw = _invoke(
          runners.candidate, queue_mode=queue_mode, route=route, phase="round",
          invocation_index=candidate_invocations, pair_index=pair_index, family=family,
          clock_identity=authority.clock_identity,
          candidate_executable_identity=executable)
        candidate_invocations += 1
        contract, detail = _candidate_receipt(
          raw, family=family, queue_mode=queue_mode,
          clock_identity=authority.clock_identity,
          candidate_executable_identity=executable)
        contract_pair["candidate"], detail_pair["candidate"] = contract, detail
      else:
        raw = _invoke(
          runners.direct_packed, queue_mode=queue_mode, route=route, phase="round",
          invocation_index=fallback_invocations, pair_index=pair_index, family=family,
          clock_identity=authority.clock_identity,
          candidate_executable_identity=executable)
        fallback_invocations += 1
        contract, evidence, detail = _fallback_receipt(
          raw, queue_mode=queue_mode, clock_identity=authority.clock_identity)
        if fallback_identity is None: fallback_identity = evidence
        elif evidence != fallback_identity:
          raise ValueError(f"{queue_mode} direct_packed executable evidence changed across rounds")
        contract_pair["fallback"], detail_pair["fallback"] = contract, detail
    paired_contract.append(contract_pair)
    paired_details.append(detail_pair)
  if fallback_identity is None:
    raise ValueError(f"{queue_mode} direct_packed evidence was not observed")

  observation = {
    "family_identity": family.family_identity,
    "candidate_executable_identity": executable,
    "candidate_c6_queue_evidence_identity": queue_correctness[queue_mode],
    "candidate_comparator_identity": comparators[queue_mode],
    "fallback_evidence": fallback_identity,
    "session_identity": authority.session_identity,
    "workload_identity": c6.get("workload_identity"),
    "input_identity": c6.get("input_identity"),
    "device_identity": c6.get("device_identity"),
    "software_identity": c6.get("software_identity"),
    "clock_identity": authority.clock_identity,
    "measurement_source": MEASUREMENT_SOURCE,
    "candidate_warmups": warmups, "fallback_warmups": warmups,
    "randomization_seed": seed,
    "randomization_algorithm": RANDOMIZATION_ALGORITHM,
    "randomized_order_identity": _identity({"seed": seed, "orders": orders}),
    "paired_rounds": paired_contract,
  }
  detail = {
    "queue_mode": queue_mode, "session_identity": authority.session_identity,
    "clock_identity": authority.clock_identity,
    "warmups": warmup_details, "paired_rounds": paired_details,
    "invocation_counts": {
      "staged_candidate": candidate_invocations,
      "direct_packed": fallback_invocations,
    },
    "orders": orders, "all_candidate_epoch_phases_retained": True,
    "direct_packed_evidence_stable": True,
  }
  return observation, detail


def collect_staged_c8_timing(
    *, family: FrozenStagedFamily, c7_memory_ledger: Mapping[str, Any],
    c6_correctness_evidence: Mapping[str, Any],
    queue_authorities: Mapping[str, QueueTimingAuthority],
    queue_runners: Mapping[str, QueueTimingRunners],
    warmups: int = 3, rounds: int = 10, seed: int = 20260719,
    required_speedup: int | float = 1.0,
    ) -> dict[str, Any]:
  """Collect and certify matched candidate/direct-packed timing for PM4 and AQL."""
  if not isinstance(family, FrozenStagedFamily):
    raise TypeError("family must be a loader-validated FrozenStagedFamily")
  warmups = _integer(warmups, "warmups", minimum=3)
  rounds = _integer(rounds, "rounds", minimum=10)
  seed = _integer(seed, "seed")
  if not isinstance(queue_authorities, Mapping) or set(queue_authorities) != set(QUEUE_MODES):
    raise ValueError(f"queue_authorities must contain exactly {QUEUE_MODES}")
  if not isinstance(queue_runners, Mapping) or set(queue_runners) != set(QUEUE_MODES):
    raise ValueError(f"queue_runners must contain exactly {QUEUE_MODES}")
  if queue_authorities["PM4"].session_identity == queue_authorities["AQL"].session_identity:
    raise ValueError("PM4 and AQL timing sessions must be distinct")

  observations, details = {}, {}
  for queue_mode in QUEUE_MODES:
    observations[queue_mode], details[queue_mode] = _collect_queue(
      family=family, queue_mode=queue_mode, c6=c6_correctness_evidence,
      authority=queue_authorities[queue_mode], runners=queue_runners[queue_mode],
      warmups=warmups, rounds=rounds, seed=seed)
  c8 = build_staged_c8_timing_contract(
    family=family, c7_memory_ledger=c7_memory_ledger,
    c6_correctness_evidence=c6_correctness_evidence,
    queue_observations=observations, required_speedup=required_speedup)
  payload = {
    "schema": SCHEMA, "status": "PASS", "family_identity": family.family_identity,
    "protocol": {
      "warmups_per_route": warmups, "paired_rounds": rounds, "seed": seed,
      "randomization_algorithm": RANDOMIZATION_ALGORITHM,
      "measurement_source": MEASUREMENT_SOURCE,
      "queue_modes_measured_separately": True,
    },
    "queue_collections": details, "c8_contract": c8,
    "production_dispatch_changed": False,
  }
  return {**payload, "evidence_identity": _identity(payload)}


def collect_staged_c8_timing_from_samples(
    *, family: FrozenStagedFamily, c7_memory_ledger: Mapping[str, Any],
    c6_correctness_evidence: Mapping[str, Any], samples: Mapping[str, Any],
    required_speedup: int | float = 1.0,
    ) -> dict[str, Any]:
  """Replay captured live receipts through the same collector validation path."""
  if not isinstance(samples, Mapping):
    raise ValueError("samples must be a mapping")
  _exact_keys(samples, {"warmups", "rounds", "seed", "queues"}, "samples")
  queues = samples["queues"]
  if not isinstance(queues, Mapping) or set(queues) != set(QUEUE_MODES):
    raise ValueError(f"samples.queues must contain exactly {QUEUE_MODES}")
  authorities, runners = {}, {}
  for queue_mode in QUEUE_MODES:
    row = queues[queue_mode]
    if not isinstance(row, Mapping):
      raise ValueError(f"samples.queues.{queue_mode} must be a mapping")
    _exact_keys(row, {
      "session_identity", "clock_identity", "candidate_warmups",
      "fallback_warmups", "paired_rounds",
    }, f"samples.queues.{queue_mode}")
    candidate_warmups, fallback_warmups, pairs = (
      row["candidate_warmups"], row["fallback_warmups"], row["paired_rounds"])
    if not all(isinstance(value, list) for value in (candidate_warmups, fallback_warmups, pairs)):
      raise ValueError(f"samples.queues.{queue_mode} receipt lists are malformed")
    candidate_values = list(candidate_warmups)
    fallback_values = list(fallback_warmups)
    for pair in pairs:
      if not isinstance(pair, Mapping) or set(pair) != {"order", "candidate", "fallback"}:
        raise ValueError(f"samples.queues.{queue_mode}.paired_rounds row is malformed")
      candidate_values.append(pair["candidate"])
      fallback_values.append(pair["fallback"])
    expected_orders = staged_c8_randomized_orders(
      seed=samples["seed"], round_count=samples["rounds"])
    if [pair["order"] for pair in pairs] != expected_orders:
      raise ValueError(f"samples.queues.{queue_mode} orders differ from seeded schedule")

    def replay(values: list[Any]) -> TimingRunner:
      cursor = iter(values)
      def run(**_kwargs: Any) -> Mapping[str, Any]:
        try: return next(cursor)
        except StopIteration as exc: raise ValueError("sample runner exhausted") from exc
      return run

    authorities[queue_mode] = QueueTimingAuthority(
      row["session_identity"], row["clock_identity"])
    runners[queue_mode] = QueueTimingRunners(
      replay(candidate_values), replay(fallback_values))
  return collect_staged_c8_timing(
    family=family, c7_memory_ledger=c7_memory_ledger,
    c6_correctness_evidence=c6_correctness_evidence,
    queue_authorities=authorities, queue_runners=runners,
    warmups=samples["warmups"], rounds=samples["rounds"], seed=samples["seed"],
    required_speedup=required_speedup)


def make_frozen_staged_candidate_runner(
    *, role_spec: Any, frozen_bundle: str | Path,
    staged_family_manifest: str | Path, runtime_canary_by_queue: Mapping[str, Mapping[str, Any]],
    probe_runner: TimingRunner | None = None, timeout_seconds: float = 900.0,
    inventory: str | Path | Mapping[str, Any] = DEFAULT_INVENTORY,
    ) -> TimingRunner:
  """Adapt the existing exact staged executor to the collector receipt seam.

  The default is the existing same-process full-grid path.  The surrounding
  queue-specific child must already have its final ``AMD_AQL`` value; spawning
  a fresh target child per invocation would defeat warmed paired measurement.
  An injected ``probe_runner`` is only a test/custom same-process seam and
  must add a ``c8_timing_receipt`` to its otherwise ordinary probe result.
  The staged executor validates the frozen family, C4 binding, full prefix,
  correctness, queue mode, phase isolation, and no-fallback facts before this
  adapter releases that timing receipt.
  """
  if probe_runner is None:
    from extra.qk.mmq_llama_five_buffer_gpu_harness import \
      run_full_grid_target_role_probe
    probe_runner = run_full_grid_target_role_probe
  if not callable(probe_runner):
    raise TypeError("same-process staged timing probe runner must be callable")
  if getattr(probe_runner, "__name__", "") == "run_full_grid_target_role_probe_isolated":
    raise ValueError("C8 warmed pairing cannot spawn one isolated child per candidate invocation")

  def run(*, queue_mode: str, family: FrozenStagedFamily,
          clock_identity: str, candidate_executable_identity: str,
          **_kwargs: Any) -> Mapping[str, Any]:
    from extra.qk.mmq_frozen_staged_family_execution import \
      run_frozen_staged_family_prefix_probe
    from extra.qk.mmq_llama_five_buffer_gpu_harness import \
      FROZEN_STAGED_C8_TIMING_RECEIPT_SCHEMA
    if queue_mode not in runtime_canary_by_queue:
      raise ValueError(f"{queue_mode} runtime canary is unavailable")

    def timing_probe(**kwargs: Any) -> Mapping[str, Any]:
      requested_env = kwargs.pop("child_env_overrides", None)
      # This adapter deliberately runs inside the already-isolated queue
      # child.  The executor's timeout belongs to an outer subprocess wrapper
      # and is not part of the same-process harness signature.
      kwargs.pop("timeout_seconds", None)
      expected_aql = "1" if queue_mode == "AQL" else "0"
      if not isinstance(requested_env, Mapping) or \
         requested_env.get("AMD_AQL") != expected_aql or \
         os.environ.get("AMD_AQL") != expected_aql:
        raise ValueError(
          "same-process staged timing child AMD_AQL mode differs from queue_mode")
      return probe_runner(**kwargs, c8_phase_timing=True)

    result = run_frozen_staged_family_prefix_probe(
      role_spec=role_spec, frozen_bundle=frozen_bundle,
      staged_family_manifest=staged_family_manifest,
      prefix_epochs=role_spec.epochs, queue_mode=queue_mode,
      runtime_canary=runtime_canary_by_queue[queue_mode],
      timeout_seconds=timeout_seconds, inventory=inventory,
      probe_runner=timing_probe,
      persistent_session_parent_containment=True)
    if result.get("status") != "PASS" or result.get("family_identity") != family.family_identity:
      details = [
        value for value in (
          result.get("exact_blocker"), result.get("exception"), result.get("error"))
        if isinstance(value, str) and value
      ]
      suffix = f": {'; '.join(details)}" if details else ""
      raw = result.get("raw_probe")
      raw_failure = {
        key: raw[key] for key in (
          "exact_blocker", "exception", "error", "completed_epochs")
        if isinstance(raw, Mapping) and key in raw and
           isinstance(raw[key], (str, int, float, bool, type(None)))
      }
      failure_evidence = {
        "schema": STAGED_CANDIDATE_FAILURE_SCHEMA,
        "status": "BLOCKED",
        "exact_blocker": result.get("exact_blocker"),
        "exception": result.get("exception"), "error": result.get("error"),
        "raw_probe_failure": raw_failure or None,
      }
      message = f"{queue_mode} frozen staged timing execution did not pass{suffix}"
      raise StagedCandidateExecutionError(message, failure_evidence)
    receipt = result.get("raw_probe", {}).get("c8_timing_receipt")
    if not isinstance(receipt, Mapping) or \
       receipt.get("schema") != FROZEN_STAGED_C8_TIMING_RECEIPT_SCHEMA or \
       receipt.get("status") != "PASS" or \
       receipt.get("gap_free_phase_partition") is not True:
      raise ValueError(f"{queue_mode} frozen staged probe omitted c8_timing_receipt")
    return {
      "schema": CANDIDATE_RECEIPT_SCHEMA, "status": "PASS",
      "family_identity": family.family_identity,
      "program_key": family.binding.program_key,
      "binary_sha256": family.binding.binary_sha256,
      "candidate_executable_identity": candidate_executable_identity,
      "queue_mode": queue_mode, "measurement_source": receipt["measurement_source"],
      "clock_identity": clock_identity, "phase_semantics": receipt["phase_semantics"],
      "output_initialization_ms": receipt["output_initialization_ms"],
      "epochs": receipt["epochs"], "final_sync_ms": receipt["final_sync_ms"],
      "complete_role_ms": receipt["complete_role_ms"],
    }
  return run


def make_direct_packed_fallback_runner(
    *, linear: Any, input_tensor: Any, route_spec: Any,
    fallback_evidence_by_queue: Mapping[str, Mapping[str, Any]],
    execution_attestor: Callable[[Any, str], Mapping[str, Any]],
    synchronize: Callable[[], None] | None = None,
    executor: Callable[[Any, Any, Any], Any] | None = None,
    realize_output: Callable[[Any], None] | None = None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> TimingRunner:
  """Wrap the production direct-packed route in a synchronized-wall receipt.

  PM4 and AQL callers must invoke the returned runner in separate children
  whose effective ``AMD_AQL`` value matches ``queue_mode``.  The default
  executor is the same ``_run_direct_packed_baseline`` used by production and
  the exact research fallback; no bounded test comparator is substituted.
  ``execution_attestor`` must independently identify the executable observed
  by that invocation after its final synchronization.  Its evidence must
  exactly equal the frozen queue evidence, so a caller cannot time one path
  while certifying another.  Dependency injection keeps the seam CPU-testable.
  """
  if not isinstance(fallback_evidence_by_queue, Mapping) or \
     set(fallback_evidence_by_queue) != set(QUEUE_MODES):
    raise ValueError(f"fallback_evidence_by_queue must contain exactly {QUEUE_MODES}")
  if executor is None:
    from tinygrad.llm.prefill_routes import _run_direct_packed_baseline
    executor = _run_direct_packed_baseline
  if synchronize is None:
    from tinygrad.device import Device
    synchronize = Device["AMD"].synchronize
  if realize_output is None:
    def realize_output(value: Any) -> None:
      realize = getattr(value, "realize", None)
      if not callable(realize):
        raise TypeError("direct_packed fallback output has no realize()")
      realize()
  if not all(callable(value) for value in (
      executor, synchronize, realize_output, clock_ns, execution_attestor)):
    raise TypeError("direct_packed timing dependencies must be callable")

  def run(*, queue_mode: str, clock_identity: str, **_kwargs: Any) -> Mapping[str, Any]:
    if queue_mode not in QUEUE_MODES:
      raise ValueError(f"queue_mode must be one of {QUEUE_MODES!r}")
    expected_aql = "1" if queue_mode == "AQL" else "0"
    if os.environ.get("AMD_AQL") != expected_aql:
      raise ValueError("direct_packed timing child AMD_AQL mode differs from queue_mode")
    evidence = fallback_evidence_by_queue[queue_mode]
    if not isinstance(evidence, Mapping):
      raise ValueError(f"{queue_mode} direct_packed fallback evidence must be a mapping")
    synchronize()
    started_ns = clock_ns()
    output = executor(linear, input_tensor, route_spec)
    if output is None:
      raise RuntimeError("production direct_packed baseline returned no output")
    realize_output(output)
    synchronize()
    ended_ns = clock_ns()
    if not isinstance(started_ns, int) or not isinstance(ended_ns, int) or ended_ns <= started_ns:
      raise ValueError("direct_packed synchronized-wall clock did not advance")
    attested = execution_attestor(output, queue_mode)
    if not isinstance(attested, Mapping):
      raise ValueError("direct_packed execution attestor returned no evidence mapping")
    if dict(attested) != dict(evidence):
      raise ValueError(
        f"{queue_mode} timed direct_packed executable differs from frozen fallback evidence")
    return {
      "schema": FALLBACK_RECEIPT_SCHEMA, "status": "PASS",
      "queue_mode": queue_mode, "measurement_source": MEASUREMENT_SOURCE,
      "clock_identity": clock_identity,
      "complete_role_ms": (ended_ns - started_ns) / 1e6,
      "fallback_evidence": dict(evidence),
    }
  return run


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
  value = json.loads(Path(path).read_text())
  if not isinstance(value, dict): raise ValueError(f"{label} must contain one JSON object")
  return value


def _atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
  output = Path(path)
  output.parent.mkdir(parents=True, exist_ok=True)
  fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
  try:
    with os.fdopen(fd, "w") as handle:
      json.dump(dict(value), handle, sort_keys=True, indent=2, allow_nan=False)
      handle.write("\n")
      handle.flush()
      os.fsync(handle.fileno())
    os.replace(temporary, output)
  except BaseException:
    try: os.unlink(temporary)
    except FileNotFoundError: pass
    raise


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--role", required=True)
  parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
  parser.add_argument("--frozen-bundle", type=Path, required=True)
  parser.add_argument("--staged-family-manifest", type=Path, required=True)
  parser.add_argument("--c6", type=Path, required=True)
  parser.add_argument("--c7", type=Path, required=True)
  parser.add_argument("--samples", type=Path, required=True)
  parser.add_argument("--required-speedup", type=float, default=1.0)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args(argv)
  role_spec = exact_role_spec(args.role, inventory=args.inventory)
  family = load_frozen_staged_family_manifest(
    args.staged_family_manifest, role_spec=role_spec,
    frozen_bundle=args.frozen_bundle, inventory=args.inventory)
  result = collect_staged_c8_timing_from_samples(
    family=family, c7_memory_ledger=_read_json(args.c7, "C7 evidence"),
    c6_correctness_evidence=_read_json(args.c6, "C6 evidence"),
    samples=_read_json(args.samples, "timing samples"),
    required_speedup=args.required_speedup)
  _atomic_write_json(args.output, result)
  print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
  return 0


__all__ = [
  "CANDIDATE_RECEIPT_SCHEMA", "FALLBACK_RECEIPT_SCHEMA", "MEASUREMENT_SOURCE",
  "PHASE_SEMANTICS", "STAGED_CANDIDATE_FAILURE_SCHEMA",
  "StagedCandidateExecutionError",
  "QueueTimingAuthority", "QueueTimingRunners", "SCHEMA",
  "collect_staged_c8_timing", "collect_staged_c8_timing_from_samples",
  "make_direct_packed_fallback_runner", "make_frozen_staged_candidate_runner",
]


if __name__ == "__main__": raise SystemExit(main())
