"""Persistent, queue-separated C8 timing-session orchestration.

Each PM4/AQL measurement gets exactly one fresh spawned child.  That child
attests its instantiated queue, initializes the candidate and direct-packed
runner once, then performs equal warmups and a seeded randomized paired
schedule using one persistent process and clock.  The parent surrounds each
child with independent health/fault checks and replays the resulting receipts
through the public frozen-staged C8 collector.

The orchestration layer is device-neutral.  Only the default queue attestor
imports ``Device``, and it runs inside the already queue-selected child.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

from extra.qk.mmq_exact_role_spec import DEFAULT_INVENTORY, exact_role_spec
from extra.qk.mmq_frozen_epoch_runtime_preconstruction_canary import QUEUE_CLASSES
from extra.qk.mmq_frozen_staged_c8_timing import (
  QueueTimingRunners, collect_staged_c8_timing_from_samples,
)
from extra.qk.mmq_frozen_staged_family import (
  QUEUE_MODES, FrozenStagedFamily, load_frozen_staged_family_manifest,
)
from extra.qk.mmq_staged_c7_c8_contract import staged_c8_randomized_orders


SCHEMA = "tinygrad.mmq_q4k_q8_1.frozen_staged_c8_persistent_sessions.v1"
QUEUE_SESSION_SCHEMA = f"{SCHEMA}.queue"
QUEUE_ATTESTATION_SCHEMA = f"{SCHEMA}.effective_queue_attestation"
CLOCK_IDENTITY = "clock-policy-0"
ROUTE_SEQUENCE_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.frozen_staged_c8_route_sequence.v1"

RunnerFactory = Callable[..., QueueTimingRunners]
AttestorFactory = Callable[..., Any]


def _canonical(value: Any) -> bytes:
  return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity(value: Any) -> str:
  return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
    raise ValueError(f"{label} must be an integer >= {minimum}")
  return value


def _positive_number(value: Any, label: str) -> float:
  if not isinstance(value, (int, float)) or isinstance(value, bool) or \
     not math.isfinite(value) or value <= 0:
    raise ValueError(f"{label} must be a finite positive number")
  return float(value)


def _nonempty(value: Any, label: str) -> str:
  if not isinstance(value, str) or not value:
    raise ValueError(f"{label} must be a non-empty string")
  return value


def _callable_class_name(value: Any) -> str:
  value = getattr(value, "func", value)
  typ = value if isinstance(value, type) else type(value)
  return f"{typ.__module__}.{typ.__qualname__}"


def amd_effective_queue_attestor_factory(
    *, queue_mode: str, device: str = "AMD", config: Mapping[str, Any] | None = None,
    ) -> Callable[[], Mapping[str, Any]]:
  """Create the live child attestor without trusting ``AMD_AQL`` as authority."""
  if config:
    if set(config) != {"device"}:
      raise ValueError("default queue attestor config only permits device")
    device = _nonempty(config["device"], "attestor device")

  def attest() -> Mapping[str, Any]:
    from tinygrad.device import Device
    dev = Device[device]
    effective = "AQL" if bool(getattr(dev, "is_aql", False)) else "PM4"
    queue_class = _callable_class_name(getattr(dev, "hw_compute_queue_t", None))
    checks = {
      "environment_matches_requested":
        os.environ.get("AMD_AQL") == ("1" if queue_mode == "AQL" else "0"),
      "requested_matches_effective": effective == queue_mode,
      "queue_class_matches_effective": queue_class == QUEUE_CLASSES[effective],
    }
    return {
      "schema": QUEUE_ATTESTATION_SCHEMA,
      "authority": "instantiated_device_state",
      "device": device,
      "requested_queue_mode": queue_mode,
      "effective_queue_mode": effective,
      "effective_queue_class": queue_class,
      "expected_queue_class": QUEUE_CLASSES[queue_mode],
      "environment_amd_aql": os.environ.get("AMD_AQL"),
      "checks": checks, "all_checks_pass": all(checks.values()),
    }
  return attest


def _validated_queue_attestation(value: Any, queue_mode: str) -> dict[str, Any]:
  if not isinstance(value, Mapping):
    raise ValueError("effective queue attestor returned no mapping")
  expected_keys = {
    "schema", "authority", "device", "requested_queue_mode",
    "effective_queue_mode", "effective_queue_class", "expected_queue_class",
    "environment_amd_aql", "checks", "all_checks_pass",
  }
  if set(value) != expected_keys:
    raise ValueError("effective queue attestation fields differ")
  expected_aql = "1" if queue_mode == "AQL" else "0"
  expected_checks = {
    "environment_matches_requested": True,
    "requested_matches_effective": True,
    "queue_class_matches_effective": True,
  }
  if value.get("schema") != QUEUE_ATTESTATION_SCHEMA or \
     value.get("authority") != "instantiated_device_state" or \
     not isinstance(value.get("device"), str) or not value["device"] or \
     value.get("requested_queue_mode") != queue_mode or \
     value.get("effective_queue_mode") != queue_mode or \
     value.get("effective_queue_class") != QUEUE_CLASSES[queue_mode] or \
     value.get("expected_queue_class") != QUEUE_CLASSES[queue_mode] or \
     value.get("environment_amd_aql") != expected_aql or \
     value.get("checks") != expected_checks or value.get("all_checks_pass") is not True:
    raise ValueError("effective queue attestation did not close")
  return dict(value)


def _invoke(
    runner: Callable[..., Mapping[str, Any]], *, queue_mode: str, route: str,
    phase: str, invocation_index: int, pair_index: int | None,
    family: FrozenStagedFamily, clock_identity: str,
    candidate_executable_identity: str,
    ) -> dict[str, Any]:
  value = runner(
    queue_mode=queue_mode, route=route, phase=phase,
    invocation_index=invocation_index, pair_index=pair_index,
    family=family, prefix_epochs=family.binding.role_spec.epochs,
    clock_identity=clock_identity,
    candidate_executable_identity=candidate_executable_identity)
  if not isinstance(value, Mapping):
    raise ValueError(f"{queue_mode} {route} timing runner returned no mapping")
  return dict(value)


def run_persistent_c8_queue_session_worker(
    family: FrozenStagedFamily, c6_correctness_evidence: Mapping[str, Any],
    queue_mode: str, warmups: int, rounds: int, seed: int,
    session_identity: str, clock_identity: str,
    runner_factory: RunnerFactory, attestor_factory: AttestorFactory,
    runner_config: Mapping[str, Any] | None = None,
    attestor_config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
  """Spawn-child entry: collect one queue's entire paired schedule."""
  if not isinstance(family, FrozenStagedFamily):
    raise TypeError("family must be a loader-validated FrozenStagedFamily")
  if queue_mode not in QUEUE_MODES:
    raise ValueError(f"queue_mode must be one of {QUEUE_MODES!r}")
  warmups, rounds = _integer(warmups, "warmups", minimum=3), \
    _integer(rounds, "rounds", minimum=10)
  seed = _integer(seed, "seed")
  session_identity = _nonempty(session_identity, "session_identity")
  clock_identity = _nonempty(clock_identity, "clock_identity")
  if not callable(runner_factory) or not callable(attestor_factory):
    raise TypeError("runner_factory and attestor_factory must be callable")
  expected_aql = "1" if queue_mode == "AQL" else "0"
  os.environ.update({"AMD_AQL": expected_aql, "DEV": "AMD"})

  attestor = attestor_factory(
    queue_mode=queue_mode, config={} if attestor_config is None else dict(attestor_config))
  attestation = _validated_queue_attestation(
    attestor() if callable(attestor) else attestor, queue_mode)
  runners = runner_factory(
    queue_mode=queue_mode, family=family,
    c6_correctness_evidence=dict(c6_correctness_evidence),
    clock_identity=clock_identity, clock_ns=time.perf_counter_ns,
    config={} if runner_config is None else dict(runner_config))
  if not isinstance(runners, QueueTimingRunners):
    raise TypeError("runner_factory must return QueueTimingRunners")
  runners.validate(queue_mode)
  executable = _nonempty(
    c6_correctness_evidence.get("candidate_executable_identity"),
    "C6 candidate_executable_identity")

  candidate_warmups, fallback_warmups = [], []
  candidate_invocations = fallback_invocations = 0
  orders = staged_c8_randomized_orders(seed=seed, round_count=rounds)
  pairs: list[dict[str, Any]] = []
  failure: dict[str, Any] | None = None

  def invoke_tracked(
      runner: Callable[..., Mapping[str, Any]], *, route: str, phase: str,
      invocation_index: int, pair_index: int | None,
      ) -> dict[str, Any] | None:
    nonlocal failure
    try:
      return _invoke(
        runner, queue_mode=queue_mode, route=route, phase=phase,
        invocation_index=invocation_index, pair_index=pair_index,
        family=family, clock_identity=clock_identity,
        candidate_executable_identity=executable)
    except Exception as exc:
      nested = getattr(exc, "failure_evidence", None)
      if isinstance(nested, Mapping):
        try:
          json.loads(_canonical(nested))
          nested = dict(nested)
        except (TypeError, ValueError):
          nested = None
      failure = {
        "route": route, "phase": phase,
        "invocation_index": invocation_index, "pair_index": pair_index,
        "exception": type(exc).__name__, "error": str(exc),
        "nested_failure": nested,
      }
      return None

  def blocked() -> dict[str, Any]:
    assert failure is not None
    pair_suffix = "" if failure["pair_index"] is None else \
      f" pair[{failure['pair_index']}]"
    exact_blocker = (
      f"{queue_mode} persistent C8 invocation failed at "
      f"{failure['phase']} {failure['route']}[{failure['invocation_index']}]"
      f"{pair_suffix}: "
      f"{failure['exception']}: {failure['error']}")
    sample = {
      "session_identity": session_identity, "clock_identity": clock_identity,
      "candidate_warmups": candidate_warmups,
      "fallback_warmups": fallback_warmups, "paired_rounds": pairs,
    }
    payload = {
      "schema": QUEUE_SESSION_SCHEMA, "status": "BLOCKED",
      "exact_blocker": exact_blocker, "queue_mode": queue_mode,
      "family_identity": family.family_identity,
      "session_identity": session_identity, "clock_identity": clock_identity,
      "effective_queue_attestation": attestation,
      "runner_factory_invocations": 1,
      "candidate_runner_instance_count": 1,
      "fallback_runner_instance_count": 1,
      "invocation_failure": dict(failure),
      "invocation_counts": {
        "staged_candidate": candidate_invocations,
        "direct_packed": fallback_invocations,
      },
      "completed_warmups": {
        "staged_candidate": len(candidate_warmups),
        "direct_packed": len(fallback_warmups),
      },
      "completed_paired_rounds": len(pairs),
      "warmups_per_route": warmups, "paired_rounds": rounds, "seed": seed,
      "orders": orders, "samples": sample,
      "persistent_child_session": True, "same_session": True,
      "same_clock": True, "equal_warmups": True,
      "no_retry": True, "no_queue_fallback": True,
      "production_dispatch_changed": False,
    }
    return {**payload, "evidence_identity": _identity(payload)}

  for _ in range(warmups):
    receipt = invoke_tracked(
      runners.candidate, route="staged_candidate", phase="warmup",
      invocation_index=candidate_invocations, pair_index=None)
    if receipt is None: return blocked()
    candidate_warmups.append(receipt)
    candidate_invocations += 1
    receipt = invoke_tracked(
      runners.direct_packed, route="direct_packed", phase="warmup",
      invocation_index=fallback_invocations, pair_index=None)
    if receipt is None: return blocked()
    fallback_warmups.append(receipt)
    fallback_invocations += 1

  for pair_index, order in enumerate(orders):
    row: dict[str, Any] = {"order": list(order)}
    for route in order:
      if route == "staged_candidate":
        receipt = invoke_tracked(
          runners.candidate, route=route, phase="round",
          invocation_index=candidate_invocations, pair_index=pair_index)
        if receipt is None: return blocked()
        row["candidate"] = receipt
        candidate_invocations += 1
      elif route == "direct_packed":
        receipt = invoke_tracked(
          runners.direct_packed, route=route, phase="round",
          invocation_index=fallback_invocations, pair_index=pair_index)
        if receipt is None: return blocked()
        row["fallback"] = receipt
        fallback_invocations += 1
      else:
        raise ValueError(f"seeded C8 order contains unsupported route {route!r}")
    pairs.append(row)

  sample = {
    "session_identity": session_identity, "clock_identity": clock_identity,
    "candidate_warmups": candidate_warmups,
    "fallback_warmups": fallback_warmups, "paired_rounds": pairs,
  }
  payload = {
    "schema": QUEUE_SESSION_SCHEMA, "status": "PASS",
    "queue_mode": queue_mode, "family_identity": family.family_identity,
    "session_identity": session_identity, "clock_identity": clock_identity,
    "effective_queue_attestation": attestation,
    "runner_factory_invocations": 1,
    "candidate_runner_instance_count": 1,
    "fallback_runner_instance_count": 1,
    "invocation_counts": {
      "staged_candidate": candidate_invocations,
      "direct_packed": fallback_invocations,
    },
    "warmups_per_route": warmups, "paired_rounds": rounds, "seed": seed,
    "orders": orders, "samples": sample,
    "persistent_child_session": True, "same_session": True,
    "same_clock": True, "equal_warmups": True,
    "no_retry": True, "no_queue_fallback": True,
    "production_dispatch_changed": False,
  }
  return {**payload, "evidence_identity": _identity(payload)}


def run_persistent_c8_route_sequence_worker(
    family: FrozenStagedFamily, c6_correctness_evidence: Mapping[str, Any],
    queue_mode: str, sequence: Sequence[str], session_identity: str,
    clock_identity: str, runner_factory: RunnerFactory,
    attestor_factory: AttestorFactory,
    runner_config: Mapping[str, Any] | None = None,
    attestor_config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
  """Run one bounded diagnostic route sequence inside one persistent child."""
  if not isinstance(family, FrozenStagedFamily):
    raise TypeError("family must be a loader-validated FrozenStagedFamily")
  if queue_mode not in QUEUE_MODES:
    raise ValueError(f"queue_mode must be one of {QUEUE_MODES!r}")
  normalized = tuple(sequence)
  supported = {"staged_candidate", "direct_packed"}
  if not 1 <= len(normalized) <= 3 or any(route not in supported for route in normalized):
    raise ValueError("diagnostic route sequence must contain 1-3 supported routes")
  session_identity = _nonempty(session_identity, "session_identity")
  clock_identity = _nonempty(clock_identity, "clock_identity")
  os.environ.update({
    "AMD_AQL": "1" if queue_mode == "AQL" else "0", "DEV": "AMD"})
  attestor = attestor_factory(
    queue_mode=queue_mode,
    config={} if attestor_config is None else dict(attestor_config))
  attestation = _validated_queue_attestation(
    attestor() if callable(attestor) else attestor, queue_mode)
  runners = runner_factory(
    queue_mode=queue_mode, family=family,
    c6_correctness_evidence=dict(c6_correctness_evidence),
    clock_identity=clock_identity, clock_ns=time.perf_counter_ns,
    config={} if runner_config is None else dict(runner_config))
  if not isinstance(runners, QueueTimingRunners):
    raise TypeError("runner_factory must return QueueTimingRunners")
  runners.validate(queue_mode)
  executable = _nonempty(
    c6_correctness_evidence.get("candidate_executable_identity"),
    "C6 candidate_executable_identity")
  counts = {"staged_candidate": 0, "direct_packed": 0}
  breadcrumbs: list[dict[str, Any]] = []
  failure = None
  for position, route in enumerate(normalized):
    runner = runners.candidate if route == "staged_candidate" else runners.direct_packed
    invocation_index = counts[route]
    try:
      receipt = _invoke(
        runner, queue_mode=queue_mode, route=route, phase="diagnostic",
        invocation_index=invocation_index, pair_index=position,
        family=family, clock_identity=clock_identity,
        candidate_executable_identity=executable)
    except Exception as exc:
      nested = getattr(exc, "failure_evidence", None)
      failure = {
        "position": position, "route": route,
        "invocation_index": invocation_index,
        "exception": type(exc).__name__, "error": str(exc),
        "nested_failure": dict(nested) if isinstance(nested, Mapping) else None,
      }
      break
    counts[route] += 1
    lifecycle = None
    if route == "staged_candidate":
      state = getattr(runner, "persistent_session_state", None)
      if isinstance(state, Mapping):
        lifecycle = {
          key: state.get(key) for key in (
            "signature", "initialization_count", "invocation_count",
            "buffer_ranges", "runtime_identity")
        }
    breadcrumbs.append({
      "position": position, "route": route,
      "invocation_index": invocation_index,
      "receipt_schema": receipt.get("schema"),
      "receipt_status": receipt.get("status"),
      "complete_role_ms": receipt.get("complete_role_ms"),
      "persistent_session_lifecycle": lifecycle,
    })
  exact_blocker = None if failure is None else (
    f"{queue_mode} diagnostic route sequence failed at position "
    f"{failure['position']} {failure['route']}[{failure['invocation_index']}]: "
    f"{failure['exception']}: {failure['error']}")
  payload = {
    "schema": ROUTE_SEQUENCE_SCHEMA,
    "status": "PASS" if failure is None else "BLOCKED",
    "exact_blocker": exact_blocker, "queue_mode": queue_mode,
    "family_identity": family.family_identity,
    "session_identity": session_identity, "clock_identity": clock_identity,
    "effective_queue_attestation": attestation,
    "sequence": list(normalized), "completed_positions": len(breadcrumbs),
    "invocation_counts": counts, "breadcrumbs": breadcrumbs,
    "invocation_failure": failure, "persistent_child_session": True,
    "no_retry": True, "no_queue_fallback": True,
    "production_dispatch_changed": False,
  }
  return {**payload, "evidence_identity": _identity(payload)}


def run_guarded_persistent_c8_route_sequence(
    *, family: FrozenStagedFamily,
    c6_correctness_evidence: Mapping[str, Any],
    queue_mode: str, sequence: Sequence[str],
    runner_factory: RunnerFactory,
    attestor_factory: AttestorFactory = amd_effective_queue_attestor_factory,
    runner_config: Mapping[str, Any] | None = None,
    attestor_config: Mapping[str, Any] | None = None,
    timeout_seconds: float = 180.0,
    isolated_runner: Callable[..., Any] | None = None,
    health_probe: Callable[[Mapping[str, str]], bool] | None = None,
    fault_collector: Callable[[float], tuple[list[str], Mapping[str, Any]]] | None = None,
    ) -> dict[str, Any]:
  """Contain one no-retry route-sequence diagnostic with health/fault guards."""
  timeout_seconds = _positive_number(timeout_seconds, "timeout_seconds")
  if isolated_runner is None:
    from tinygrad.runtime.process_isolated import run_isolated
    isolated_runner = run_isolated
  if health_probe is None:
    from extra.qk.mmq_target_epoch_orchestrator import spawned_tiny_health_probe
    health_probe = spawned_tiny_health_probe
  if fault_collector is None:
    from extra.qk.mmq_target_epoch_orchestrator import collect_kernel_fault_evidence
    fault_collector = collect_kernel_fault_evidence
  env = {"AMD_AQL": "1" if queue_mode == "AQL" else "0"}
  try: health_before = bool(health_probe(env))
  except Exception: health_before = False
  started = time.time()
  isolated = None
  if health_before:
    args = (
      family, dict(c6_correctness_evidence), queue_mode, tuple(sequence),
      _identity({
        "schema": f"{ROUTE_SEQUENCE_SCHEMA}.session_identity",
        "queue_mode": queue_mode, "sequence": list(sequence),
        "nonce": time.time_ns(),
      }),
      CLOCK_IDENTITY, runner_factory, attestor_factory,
      {} if runner_config is None else dict(runner_config),
      {} if attestor_config is None else dict(attestor_config),
    )
    isolated = isolated_runner(
      run_persistent_c8_route_sequence_worker, args=args,
      timeout_seconds=timeout_seconds, start_method="spawn")
  try: health_after = bool(health_probe(env))
  except Exception: health_after = False
  try:
    faults, fault_evidence = fault_collector(started)
    faults, fault_evidence = list(faults), dict(fault_evidence)
  except Exception as exc:
    faults = [f"kernel fault collection failed: {type(exc).__name__}: {exc}"]
    fault_evidence = {}
  child = getattr(isolated, "result", None)
  timed_out = bool(getattr(isolated, "timed_out", False))
  error = getattr(isolated, "error", None)
  blocker = None
  if not health_before: blocker = f"{queue_mode} diagnostic preflight health failed"
  elif timed_out: blocker = f"{queue_mode} diagnostic child timed out"
  elif not isinstance(child, Mapping):
    blocker = error or f"{queue_mode} diagnostic child returned no structured result"
  elif child.get("schema") != ROUTE_SEQUENCE_SCHEMA or child.get("status") != "PASS":
    blocker = child.get("exact_blocker") or f"{queue_mode} diagnostic child blocked"
  elif faults: blocker = f"{queue_mode} kernel fault/reset marker observed during diagnostic"
  elif not health_after: blocker = f"{queue_mode} diagnostic postflight health failed"
  payload = {
    "schema": f"{ROUTE_SEQUENCE_SCHEMA}.guarded",
    "status": "PASS" if blocker is None else "BLOCKED",
    "exact_blocker": blocker, "queue_mode": queue_mode,
    "sequence": list(sequence), "health_before": health_before,
    "health_after": health_after, "kernel_faults": faults,
    "kernel_fault_evidence": fault_evidence, "timed_out": timed_out,
    "error": error, "elapsed_seconds": getattr(isolated, "elapsed_seconds", None),
    "child": dict(child) if isinstance(child, Mapping) else None,
    "spawn_count": 1 if isolated is not None else 0,
    "no_retry": True, "no_queue_fallback": True,
  }
  return {**payload, "evidence_identity": _identity(payload)}


def _session_envelope(
    *, family: FrozenStagedFamily, queue_mode: str, isolated_runner: Callable[..., Any],
    health_probe: Callable[[Mapping[str, str]], bool],
    fault_collector: Callable[[float], tuple[list[str], Mapping[str, Any]]],
    worker_args: tuple[Any, ...], timeout_seconds: float,
    ) -> dict[str, Any]:
  env = {"AMD_AQL": "1" if queue_mode == "AQL" else "0"}
  try: health_before = bool(health_probe(env))
  except BaseException: health_before = False
  started = time.time()
  isolated, runner_error = None, None
  if health_before:
    try:
      isolated = isolated_runner(
        run_persistent_c8_queue_session_worker, args=worker_args,
        timeout_seconds=timeout_seconds, start_method="spawn")
    except BaseException as exc:
      runner_error = f"{type(exc).__name__}: {exc}"
  try: health_after = bool(health_probe(env))
  except BaseException: health_after = False
  try:
    faults, fault_evidence = fault_collector(started)
    faults, fault_evidence = list(faults), dict(fault_evidence)
  except BaseException as exc:
    faults = [f"kernel fault collection failed: {type(exc).__name__}: {exc}"]
    fault_evidence = {}

  child = getattr(isolated, "result", None)
  launched: bool | None = False if not health_before else \
    None if runner_error is not None else isolated is not None
  child_status = "not_launched" if launched is False else \
    "runner_error" if runner_error else getattr(isolated, "status", None)
  if isinstance(child, Mapping) and child.get("schema") == QUEUE_SESSION_SCHEMA and \
     child.get("status") == "BLOCKED":
    child_status = "blocked"
  timed_out = bool(getattr(isolated, "timed_out", False))
  error = runner_error or getattr(isolated, "error", None)
  target_dispatch_attempted: bool | None = \
    True if isinstance(child, Mapping) and \
      child.get("schema") == QUEUE_SESSION_SCHEMA and child.get("status") == "PASS" \
    else False if launched is False else None
  blocker = None
  if not health_before: blocker = f"{queue_mode} C8 preflight health failed"
  elif timed_out: blocker = f"{queue_mode} persistent C8 child timed out"
  elif child_status == "blocked" and isinstance(child, Mapping):
    blocker = child.get("exact_blocker") or \
      f"{queue_mode} persistent C8 child reported a blocker"
  elif child_status != "passed" or not isinstance(child, Mapping):
    blocker = error or f"{queue_mode} persistent C8 child returned no structured result"
  elif faults: blocker = f"{queue_mode} kernel fault/reset marker observed during C8"
  elif not health_after: blocker = f"{queue_mode} C8 postflight health failed"
  elif child.get("schema") != QUEUE_SESSION_SCHEMA or \
       child.get("status") != "PASS" or \
       child.get("queue_mode") != queue_mode or \
       child.get("family_identity") != family.family_identity or \
       child.get("persistent_child_session") is not True or \
       child.get("same_session") is not True or child.get("same_clock") is not True or \
       child.get("equal_warmups") is not True or child.get("no_retry") is not True or \
       child.get("no_queue_fallback") is not True:
    blocker = f"{queue_mode} persistent C8 child contract differs"
  return {
    "status": "PASS" if blocker is None else "BLOCKED",
    "exact_blocker": blocker, "queue_mode": queue_mode,
    "health_before": health_before, "health_after": health_after,
    "kernel_faults": faults, "kernel_fault_evidence": fault_evidence,
    "child_status": child_status, "timed_out": timed_out, "error": error,
    "elapsed_seconds": getattr(isolated, "elapsed_seconds", None),
    "child": dict(child) if isinstance(child, Mapping) else None,
    "launched": launched, "target_dispatch_attempted": target_dispatch_attempted,
    "target_dispatch_attempted_authority":
      "passing_child_queue_session" if target_dispatch_attempted is True else
      "preflight_prevented_child_launch" if target_dispatch_attempted is False else
      "unknown_without_structured_child_queue_session",
    "spawn_count": 1 if isolated is not None else 0,
    "no_retry": True, "no_queue_fallback": True,
  }


def run_persistent_c8_sessions(
    *, family: FrozenStagedFamily, c7_memory_ledger: Mapping[str, Any],
    c6_correctness_evidence: Mapping[str, Any],
    runner_factory: RunnerFactory,
    attestor_factory: AttestorFactory = amd_effective_queue_attestor_factory,
    warmups: int = 3, rounds: int = 10, seed: int = 20260719,
    required_speedup: int | float = 1.0, timeout_seconds: float = 1800.0,
    clock_identity: str = CLOCK_IDENTITY,
    runner_config: Mapping[str, Any] | None = None,
    attestor_config: Mapping[str, Any] | None = None,
    isolated_runner: Callable[..., Any] | None = None,
    health_probe: Callable[[Mapping[str, str]], bool] | None = None,
    fault_collector: Callable[[float], tuple[list[str], Mapping[str, Any]]] | None = None,
    session_identity_factory: Callable[[str], str] | None = None,
    ) -> dict[str, Any]:
  """Run exactly one persistent spawned timing child for each queue mode."""
  if not isinstance(family, FrozenStagedFamily):
    raise TypeError("family must be a loader-validated FrozenStagedFamily")
  warmups, rounds = _integer(warmups, "warmups", minimum=3), \
    _integer(rounds, "rounds", minimum=10)
  seed = _integer(seed, "seed")
  timeout_seconds = _positive_number(timeout_seconds, "timeout_seconds")
  clock_identity = _nonempty(clock_identity, "clock_identity")
  if not callable(runner_factory) or not callable(attestor_factory):
    raise TypeError("runner_factory and attestor_factory must be callable")
  if isolated_runner is None:
    from tinygrad.runtime.process_isolated import run_isolated
    isolated_runner = run_isolated
  if health_probe is None:
    from extra.qk.mmq_target_epoch_orchestrator import spawned_tiny_health_probe
    health_probe = spawned_tiny_health_probe
  if fault_collector is None:
    from extra.qk.mmq_target_epoch_orchestrator import collect_kernel_fault_evidence
    fault_collector = collect_kernel_fault_evidence
  if session_identity_factory is None:
    nonce = time.time_ns()
    session_identity_factory = lambda queue: _identity({
      "schema": f"{SCHEMA}.session_identity", "family_identity": family.family_identity,
      "queue_mode": queue, "seed": seed, "parent_pid": os.getpid(), "nonce": nonce,
    })
  if not callable(session_identity_factory):
    raise TypeError("session_identity_factory must be callable")

  sessions: dict[str, Any] = {}
  prior_blocker: str | None = None
  for queue_mode in QUEUE_MODES:
    if prior_blocker is not None:
      sessions[queue_mode] = {
        "status": "BLOCKED",
        "exact_blocker":
          f"{queue_mode} not attempted after prior queue blocker: {prior_blocker}",
        "queue_mode": queue_mode, "health_before": None, "health_after": None,
        "kernel_faults": [], "kernel_fault_evidence": {},
        "child_status": "not_attempted_after_prior_blocker",
        "timed_out": False, "error": None, "elapsed_seconds": None,
        "child": None, "launched": False, "target_dispatch_attempted": False,
        "target_dispatch_attempted_authority":
          "not_attempted_after_prior_blocker",
        "spawn_count": 0, "no_retry": True, "no_queue_fallback": True,
      }
      continue
    session_identity = _nonempty(
      session_identity_factory(queue_mode), f"{queue_mode} session identity")
    args = (
      family, dict(c6_correctness_evidence), queue_mode, warmups, rounds, seed,
      session_identity, clock_identity, runner_factory, attestor_factory,
      {} if runner_config is None else dict(runner_config),
      {} if attestor_config is None else dict(attestor_config),
    )
    sessions[queue_mode] = _session_envelope(
      family=family, queue_mode=queue_mode, isolated_runner=isolated_runner,
      health_probe=health_probe, fault_collector=fault_collector,
      worker_args=args, timeout_seconds=timeout_seconds)
    if sessions[queue_mode]["status"] != "PASS":
      prior_blocker = sessions[queue_mode]["exact_blocker"]

  blocker = next(
    (sessions[queue]["exact_blocker"] for queue in QUEUE_MODES
     if sessions[queue]["status"] != "PASS"), None)
  collection = None
  if blocker is None:
    try:
      samples = {
        "warmups": warmups, "rounds": rounds, "seed": seed,
        "queues": {queue: sessions[queue]["child"]["samples"] for queue in QUEUE_MODES},
      }
      if samples["queues"]["PM4"]["session_identity"] == \
         samples["queues"]["AQL"]["session_identity"]:
        raise ValueError("PM4 and AQL persistent session identities alias")
      collection = collect_staged_c8_timing_from_samples(
        family=family, c7_memory_ledger=c7_memory_ledger,
        c6_correctness_evidence=c6_correctness_evidence,
        samples=samples, required_speedup=required_speedup)
    except BaseException as exc:
      blocker = f"C8 persistent session evidence failed closed: {type(exc).__name__}: {exc}"
  payload = {
    "schema": SCHEMA, "status": "PASS" if blocker is None else "BLOCKED",
    "exact_blocker": blocker, "family_identity": family.family_identity,
    "protocol": {
      "one_spawned_child_per_queue": True, "persistent_child_session": True,
      "warmups_per_route": warmups, "paired_rounds": rounds, "seed": seed,
      "clock_identity": clock_identity, "no_retry": True,
      "no_queue_fallback": True,
    },
    "queue_sessions": sessions, "c8_collection": collection,
    "production_dispatch_changed": False,
  }
  return {**payload, "evidence_identity": _identity(payload)}


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
  value = json.loads(Path(path).read_text())
  if not isinstance(value, dict):
    raise ValueError(f"{label} must contain one JSON object")
  return value


def _atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
  output = Path(path)
  output.parent.mkdir(parents=True, exist_ok=True)
  fd, temporary = tempfile.mkstemp(
    prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
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


def _load_factory(spec: str, label: str) -> Callable[..., Any]:
  if not isinstance(spec, str) or spec.count(":") != 1:
    raise ValueError(f"{label} must use module:attribute syntax")
  module_name, attribute = spec.split(":", 1)
  value = getattr(importlib.import_module(module_name), attribute)
  if not callable(value):
    raise TypeError(f"{label} does not resolve to a callable")
  return value


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--role", required=True)
  parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
  parser.add_argument("--frozen-bundle", type=Path, required=True)
  parser.add_argument("--staged-family-manifest", type=Path, required=True)
  parser.add_argument("--c6", type=Path, required=True)
  parser.add_argument("--c7", type=Path, required=True)
  parser.add_argument("--runner-factory", required=True, metavar="MODULE:ATTRIBUTE")
  parser.add_argument(
    "--attestor-factory",
    default="extra.qk.mmq_frozen_staged_c8_sessions:amd_effective_queue_attestor_factory",
    metavar="MODULE:ATTRIBUTE")
  parser.add_argument("--runner-config", type=Path)
  parser.add_argument("--attestor-config", type=Path)
  parser.add_argument("--warmups", type=int, default=3)
  parser.add_argument("--rounds", type=int, default=10)
  parser.add_argument("--seed", type=int, default=20260719)
  parser.add_argument("--required-speedup", type=float, default=1.0)
  parser.add_argument("--clock-identity", default=CLOCK_IDENTITY)
  parser.add_argument("--timeout-seconds", type=float, default=1800.0)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args(argv)
  role = exact_role_spec(args.role, inventory=args.inventory)
  family = load_frozen_staged_family_manifest(
    args.staged_family_manifest, role_spec=role,
    frozen_bundle=args.frozen_bundle, inventory=args.inventory)
  result = run_persistent_c8_sessions(
    family=family, c7_memory_ledger=_read_json(args.c7, "C7 evidence"),
    c6_correctness_evidence=_read_json(args.c6, "C6 evidence"),
    runner_factory=_load_factory(args.runner_factory, "runner factory"),
    attestor_factory=_load_factory(args.attestor_factory, "attestor factory"),
    warmups=args.warmups, rounds=args.rounds, seed=args.seed,
    required_speedup=args.required_speedup, timeout_seconds=args.timeout_seconds,
    clock_identity=args.clock_identity,
    runner_config={} if args.runner_config is None else
      _read_json(args.runner_config, "runner config"),
    attestor_config={} if args.attestor_config is None else
      _read_json(args.attestor_config, "attestor config"))
  _atomic_write_json(args.output, result)
  print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
  return 0 if result["status"] == "PASS" else 1


__all__ = [
  "CLOCK_IDENTITY", "QUEUE_ATTESTATION_SCHEMA", "QUEUE_SESSION_SCHEMA", "SCHEMA",
  "ROUTE_SEQUENCE_SCHEMA",
  "amd_effective_queue_attestor_factory", "run_persistent_c8_queue_session_worker",
  "run_persistent_c8_route_sequence_worker",
  "run_guarded_persistent_c8_route_sequence", "run_persistent_c8_sessions",
]


if __name__ == "__main__": raise SystemExit(main())
