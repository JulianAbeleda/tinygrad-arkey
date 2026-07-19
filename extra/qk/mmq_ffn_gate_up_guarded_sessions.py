"""Guarded, queue-isolated orchestration for matched ``ffn_gate_up`` C8.

The parent runs PM4 and then AQL in distinct spawned children.  Every child
selects ``AMD_AQL`` before any device import, loads the exact v2 runtime
configuration, installs one host-I/O census for its whole lifetime, constructs
both routes exactly once, and delegates all samples to the persistent v2
worker.  Parent-side health and kernel-fault guards fail closed with no retry
or queue fallback.

All GPU-facing boundaries are dependency injected.  Importing this module does
not import ``Device`` or initialize a runtime.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
import hashlib
import json
import math
import os
import time
from typing import Any


SCHEMA = "tinygrad.mmq_q4k_q8_1.ffn_gate_up_guarded_sessions.v2"
QUEUE_ENVELOPE_SCHEMA = f"{SCHEMA}.queue_envelope"
CHILD_SCHEMA = f"{SCHEMA}.child"
CLOCK_IDENTITY = "python_time_perf_counter_ns_v1"
QUEUE_MODES = ("PM4", "AQL")


def _canonical(value: Any) -> bytes:
  return json.dumps(
    value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity(value: Any) -> str:
  return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
  if not isinstance(value, Mapping):
    raise ValueError(f"{label} must be a mapping")
  return value


def _nonempty(value: Any, label: str) -> str:
  if not isinstance(value, str) or not value:
    raise ValueError(f"{label} must be a non-empty string")
  return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
    raise ValueError(f"{label} must be an integer >= {minimum}")
  return value


def _positive(value: Any, label: str) -> float:
  if not isinstance(value, (int, float)) or isinstance(value, bool) or \
     not math.isfinite(value) or value <= 0:
    raise ValueError(f"{label} must be finite and positive")
  return float(value)


def _default_runtime_config_loader(config: Mapping[str, Any]) -> Any:
  from extra.qk.mmq_ffn_gate_up_c8_runtime import \
    load_ffn_gate_up_c8_runtime_config
  return load_ffn_gate_up_c8_runtime_config(config)


def _default_route_composer(loaded: Any, **kwargs: Any) -> Any:
  from extra.qk.mmq_ffn_gate_up_c8_runtime import \
    compose_ffn_gate_up_queue_runners
  return compose_ffn_gate_up_queue_runners(loaded, **kwargs)


def _default_route_adapter(routes: Any, **kwargs: Any) -> Any:
  from extra.qk.mmq_ffn_gate_up_c8_session_worker import \
    adapt_ffn_gate_up_runtime_routes
  return adapt_ffn_gate_up_runtime_routes(routes, **kwargs)


def _default_persistent_worker(**kwargs: Any) -> Mapping[str, Any]:
  from extra.qk.mmq_ffn_gate_up_c8_session_worker import \
    run_ffn_gate_up_c8_persistent_queue_session_worker
  return run_ffn_gate_up_c8_persistent_queue_session_worker(**kwargs)


def _default_paired_collector(**kwargs: Any) -> Mapping[str, Any]:
  from extra.qk.mmq_ffn_gate_up_c8_paired_sessions import \
    collect_ffn_gate_up_c8_paired_sessions
  return collect_ffn_gate_up_c8_paired_sessions(**kwargs)


def _production_direct_route_builder(**kwargs: Any) -> Any:
  from extra.qk.mmq_ffn_gate_up_low_level_routes import \
    make_ffn_gate_up_direct_route
  return make_ffn_gate_up_direct_route(**kwargs)


def _production_host_io_census_factory() -> AbstractContextManager[Any]:
  from extra.qk.mmq_ffn_gate_up_low_level_routes import \
    AmdAllocatorHostIoCensus
  return AmdAllocatorHostIoCensus.for_live_amd()


@dataclass(frozen=True)
class QueueRuntimeEnvironment:
  """Instantiated queue authority and synchronization callbacks."""

  effective_queue_attestation: Mapping[str, Any]
  pre_sync: Callable[[], Any]
  post_sync: Callable[[], Any]


def amd_queue_runtime_environment(
    *, queue_mode: str, device: str = "AMD",
    ) -> QueueRuntimeEnvironment:
  """Instantiate and attest the selected AMD queue inside the selected child."""
  # The caller has already set AMD_AQL.  Keep both imports here so importing
  # this module in the parent can never construct Device first.
  from extra.qk.mmq_frozen_staged_c8_sessions import \
    amd_effective_queue_attestor_factory
  from tinygrad.device import Device
  dev = Device[device]
  attestor = amd_effective_queue_attestor_factory(
    queue_mode=queue_mode, device=device)
  observed = attestor()
  synchronize = getattr(dev, "synchronize", None)
  if not callable(synchronize):
    raise TypeError("instantiated AMD device lacks synchronize")
  return QueueRuntimeEnvironment(
    effective_queue_attestation=observed,
    pre_sync=synchronize, post_sync=synchronize)


@dataclass(frozen=True)
class GuardedChildFactories:
  """Injectable child seams; defaults remain lazy and production-faithful."""

  runtime_config_loader: Callable[[Mapping[str, Any]], Any] = \
    _default_runtime_config_loader
  queue_runtime_factory: Callable[..., QueueRuntimeEnvironment] = \
    amd_queue_runtime_environment
  route_composer: Callable[..., Any] = _default_route_composer
  route_adapter: Callable[..., Any] = _default_route_adapter
  persistent_worker: Callable[..., Mapping[str, Any]] = \
    _default_persistent_worker
  clock_ns: Callable[[], int] = time.perf_counter_ns


@dataclass(frozen=True)
class GuardedQueueRequest:
  """One picklable spawned-child request."""

  runtime_config: Mapping[str, Any]
  queue_mode: str
  session_identity: str
  clock_identity: str
  warmups: int
  rounds: int
  seed: int
  candidate_route_builder: Callable[..., Any]
  direct_route_builder: Callable[..., Any]
  host_io_census_factory: Callable[[], AbstractContextManager[Any]]
  child_factories: GuardedChildFactories
  direct_object_builder: Callable[..., Any] | None = None


def _loaded_authorities(loaded: Any, queue_mode: str) -> dict[str, Any]:
  family = getattr(loaded, "family", None)
  fixture = getattr(loaded, "fixture", None)
  composition = _mapping(
    getattr(loaded, "composition", None), "runtime composition")
  contract = dict(_mapping(
    getattr(loaded, "matched_timing_contract", None),
    "matched timing contract"))
  kwargs = dict(_mapping(
    getattr(loaded, "contract_validation_kwargs", None),
    "contract validation authorities"))
  family_identity = _nonempty(
    getattr(family, "family_identity", None), "family identity")
  input_identity = _nonempty(
    getattr(fixture, "input_identity", None), "fixture input identity")
  workload_identity = _nonempty(
    getattr(fixture, "workload_identity", None), "fixture workload identity")
  candidate = _mapping(
    composition.get("candidate_binding"), "candidate binding")
  direct_rows = _mapping(
    composition.get("direct_bindings_by_queue"), "direct bindings")
  direct = _mapping(direct_rows.get(queue_mode), f"{queue_mode} direct binding")
  return {
    "family_identity": family_identity,
    "input_identity": input_identity,
    "workload_identity": workload_identity,
    "candidate_executable_identity": _nonempty(
      candidate.get("candidate_executable_identity"),
      "candidate executable identity"),
    "direct_executable_identity": _nonempty(
      direct.get("executable_identity"), "direct executable identity"),
    "contract": contract, "contract_validation_kwargs": kwargs,
    "contract_identity": _nonempty(
      contract.get("evidence_identity"), "contract identity"),
    "composition_identity": _nonempty(
      composition.get("composition_identity"), "composition identity"),
  }


def run_guarded_ffn_gate_up_queue_child(
    request: GuardedQueueRequest,
    ) -> dict[str, Any]:
  """Spawn entrypoint for one queue-selected persistent session."""
  if not isinstance(request, GuardedQueueRequest):
    raise TypeError("guarded queue child requires GuardedQueueRequest")
  if request.queue_mode not in QUEUE_MODES:
    raise ValueError(f"queue_mode must be one of {QUEUE_MODES!r}")
  queue_mode = request.queue_mode
  # This must precede every factory that may transitively import Device.
  os.environ.update({
    "AMD_AQL": "1" if queue_mode == "AQL" else "0", "DEV": "AMD"})
  factories = request.child_factories
  if not isinstance(factories, GuardedChildFactories):
    raise TypeError("child_factories must be GuardedChildFactories")
  if not all(callable(callback) for callback in (
      request.candidate_route_builder, request.direct_route_builder,
      request.host_io_census_factory, factories.runtime_config_loader,
      factories.queue_runtime_factory, factories.route_composer,
      factories.route_adapter, factories.persistent_worker,
      factories.clock_ns)):
    raise TypeError("guarded child factories and builders must be callable")

  base = {
    "schema": CHILD_SCHEMA, "queue_mode": queue_mode,
    "session_identity": request.session_identity,
    "clock_identity": request.clock_identity,
    "no_retry": True, "retry_count": 0,
    "no_queue_fallback": True, "route_composer_invocations": 0,
    "candidate_route_builder_invocations": 0,
    "direct_route_builder_invocations": 0,
    "host_io_census_scope_count": 0,
  }
  try:
    loaded = factories.runtime_config_loader(dict(request.runtime_config))
    authority = _loaded_authorities(loaded, queue_mode)
    census_context = request.host_io_census_factory()
    with census_context as census:
      base["host_io_census_scope_count"] = 1
      snapshot = getattr(census, "snapshot", None)
      if not callable(snapshot):
        raise TypeError("host-I/O census context must expose snapshot()")
      runtime = factories.queue_runtime_factory(queue_mode=queue_mode)
      if not isinstance(runtime, QueueRuntimeEnvironment) or \
         not callable(runtime.pre_sync) or not callable(runtime.post_sync):
        raise TypeError(
          "queue runtime factory must return QueueRuntimeEnvironment")
      builder_counts = {"candidate": 0, "direct": 0}

      def candidate_builder(**kwargs: Any) -> Any:
        builder_counts["candidate"] += 1
        base["candidate_route_builder_invocations"] = \
          builder_counts["candidate"]
        if builder_counts["candidate"] > 1:
          raise RuntimeError("candidate route builder invoked more than once")
        return request.candidate_route_builder(**kwargs)

      def direct_builder(**kwargs: Any) -> Any:
        builder_counts["direct"] += 1
        base["direct_route_builder_invocations"] = builder_counts["direct"]
        if builder_counts["direct"] > 1:
          raise RuntimeError("direct route builder invoked more than once")
        return request.direct_route_builder(**kwargs)

      base["route_composer_invocations"] += 1
      routes = factories.route_composer(
        loaded, queue_mode=queue_mode,
        clock_identity=request.clock_identity,
        object_builder=request.direct_object_builder,
        candidate_route_builder=candidate_builder,
        direct_route_builder=direct_builder)
      if builder_counts != {"candidate": 1, "direct": 1}:
        raise RuntimeError(
          "route composer must invoke each route builder exactly once")
      adapted = factories.route_adapter(
        routes, queue_mode=queue_mode,
        input_identity=authority["input_identity"],
        candidate_executable_identity=
          authority["candidate_executable_identity"],
        direct_executable_identity=authority["direct_executable_identity"])
      session = factories.persistent_worker(
        contract=authority["contract"],
        contract_validation_kwargs=authority["contract_validation_kwargs"],
        queue_mode=queue_mode,
        session_identity=request.session_identity,
        clock_identity=request.clock_identity,
        candidate_executable_identity=
          authority["candidate_executable_identity"],
        direct_executable_identity=authority["direct_executable_identity"],
        pre_sync=runtime.pre_sync, post_sync=runtime.post_sync,
        clock_ns=factories.clock_ns,
        effective_queue_attestation=
          runtime.effective_queue_attestation,
        host_io_census=snapshot,
        candidate_callbacks=getattr(adapted, "candidate", None),
        direct_callbacks=getattr(adapted, "direct_packed", None),
        warmups=request.warmups, rounds=request.rounds, seed=request.seed)
    session = dict(_mapping(session, "persistent queue session"))
    if session.get("status") != "PASS":
      blocker = session.get("exact_blocker") or \
        f"{queue_mode} persistent worker blocked"
      payload = {
        **base, **authority, "status": "BLOCKED",
        "exact_blocker": blocker, "queue_session": session,
      }
    else:
      payload = {
        **base, **authority, "status": "PASS",
        "exact_blocker": None, "queue_session": session,
      }
  except BaseException as exc:
    payload = {
      **base, "status": "BLOCKED",
      "exact_blocker":
        f"{queue_mode} guarded child failed closed: "
        f"{type(exc).__name__}: {exc}",
      "exception": type(exc).__name__, "queue_session": None,
    }
  return {**payload, "evidence_identity": _identity(payload)}


def _not_attempted(queue_mode: str, blocker: str) -> dict[str, Any]:
  payload = {
    "schema": QUEUE_ENVELOPE_SCHEMA, "status": "BLOCKED",
    "exact_blocker":
      f"{queue_mode} not attempted after prior queue blocker: {blocker}",
    "queue_mode": queue_mode, "health_before": None, "health_after": None,
    "kernel_faults": [], "kernel_fault_evidence": {},
    "child_status": "not_attempted_after_prior_blocker",
    "timed_out": False, "error": None, "elapsed_seconds": None,
    "child": None, "launched": False, "spawn_count": 0,
    "no_retry": True, "retry_count": 0, "no_queue_fallback": True,
  }
  return {**payload, "evidence_identity": _identity(payload)}


def _guarded_queue_envelope(
    *, request: GuardedQueueRequest, timeout_seconds: float,
    isolated_runner: Callable[..., Any],
    health_probe: Callable[[Mapping[str, str]], bool],
    fault_collector: Callable[[float], tuple[list[str], Mapping[str, Any]]],
    ) -> dict[str, Any]:
  queue_mode = request.queue_mode
  env = {"AMD_AQL": "1" if queue_mode == "AQL" else "0"}
  try: health_before = bool(health_probe(env))
  except BaseException: health_before = False
  started = time.time()
  isolated, runner_error = None, None
  if health_before:
    try:
      isolated = isolated_runner(
        run_guarded_ffn_gate_up_queue_child, args=(request,),
        timeout_seconds=timeout_seconds, start_method="spawn")
    except BaseException as exc:
      runner_error = f"{type(exc).__name__}: {exc}"
  try: health_after = bool(health_probe(env))
  except BaseException: health_after = False
  try:
    faults, fault_evidence = fault_collector(started)
    faults, fault_evidence = list(faults), dict(fault_evidence)
  except BaseException as exc:
    faults = [
      f"kernel fault collection failed: {type(exc).__name__}: {exc}"]
    fault_evidence = {}

  child = getattr(isolated, "result", None)
  timed_out = bool(getattr(isolated, "timed_out", False))
  child_status = getattr(isolated, "status", None)
  error = runner_error or getattr(isolated, "error", None)
  try:
    child_identity_valid = isinstance(child, Mapping) and \
      child.get("evidence_identity") == _identity({
        key: value for key, value in child.items()
        if key != "evidence_identity"})
  except BaseException:
    child_identity_valid = False
  blocker = None
  if not health_before:
    blocker = f"{queue_mode} preflight health failed"
  elif timed_out:
    blocker = f"{queue_mode} persistent child timed out"
  elif runner_error is not None:
    blocker = runner_error
  elif child_status != "passed" or not isinstance(child, Mapping):
    blocker = error or f"{queue_mode} child returned no structured result"
  elif faults:
    blocker = f"{queue_mode} kernel fault/reset marker observed"
  elif not health_after:
    blocker = f"{queue_mode} postflight health failed"
  elif not child_identity_valid:
    blocker = f"{queue_mode} guarded child content identity differs"
  elif child.get("schema") != CHILD_SCHEMA or \
       child.get("queue_mode") != queue_mode or \
       child.get("session_identity") != request.session_identity or \
       child.get("clock_identity") != request.clock_identity or \
       child.get("status") != "PASS" or \
       child.get("no_retry") is not True or \
       child.get("retry_count") != 0 or \
       child.get("no_queue_fallback") is not True or \
       child.get("route_composer_invocations") != 1 or \
       child.get("candidate_route_builder_invocations") != 1 or \
       child.get("direct_route_builder_invocations") != 1 or \
       child.get("host_io_census_scope_count") != 1:
    blocker = child.get("exact_blocker") or \
      f"{queue_mode} guarded child contract differs"
  payload = {
    "schema": QUEUE_ENVELOPE_SCHEMA,
    "status": "PASS" if blocker is None else "BLOCKED",
    "exact_blocker": blocker, "queue_mode": queue_mode,
    "health_before": health_before, "health_after": health_after,
    "kernel_faults": faults, "kernel_fault_evidence": fault_evidence,
    "child_status": child_status, "timed_out": timed_out,
    "error": error,
    "elapsed_seconds": getattr(isolated, "elapsed_seconds", None),
    "child": dict(child) if isinstance(child, Mapping) else None,
    "launched": isolated is not None,
    "spawn_count": 1 if isolated is not None else 0,
    "no_retry": True, "retry_count": 0, "no_queue_fallback": True,
  }
  return {**payload, "evidence_identity": _identity(payload)}


def run_guarded_ffn_gate_up_sessions(
    *, runtime_config: Mapping[str, Any],
    candidate_route_builder: Callable[..., Any],
    direct_route_builder: Callable[..., Any] = \
      _production_direct_route_builder,
    host_io_census_factory: Callable[[], AbstractContextManager[Any]] = \
      _production_host_io_census_factory,
    warmups: int = 3, rounds: int = 10, seed: int = 20260719,
    required_speedup: int | float = 1.0,
    timeout_seconds: float = 1800.0,
    clock_identity: str = CLOCK_IDENTITY,
    direct_object_builder: Callable[..., Any] | None = None,
    child_factories: GuardedChildFactories = GuardedChildFactories(),
    paired_collector: Callable[..., Mapping[str, Any]] = \
      _default_paired_collector,
    isolated_runner: Callable[..., Any] | None = None,
    health_probe: Callable[[Mapping[str, str]], bool] | None = None,
    fault_collector: Callable[
      [float], tuple[list[str], Mapping[str, Any]]] | None = None,
    session_identity_factory: Callable[[str], str] | None = None,
    ) -> dict[str, Any]:
  """Run exactly one guarded PM4 child, then one guarded AQL child."""
  runtime_config = dict(_mapping(runtime_config, "runtime config"))
  if not all(callable(callback) for callback in (
      candidate_route_builder, direct_route_builder,
      host_io_census_factory, paired_collector)):
    raise TypeError("route, census, and collector factories must be callable")
  warmups = _integer(warmups, "warmups", minimum=3)
  rounds = _integer(rounds, "rounds", minimum=10)
  if rounds % 2:
    raise ValueError("rounds must be even for balanced paired ordering")
  seed = _integer(seed, "seed")
  required_speedup = _positive(required_speedup, "required_speedup")
  if required_speedup < 1.0:
    raise ValueError("required_speedup must be at least 1.0")
  timeout_seconds = _positive(timeout_seconds, "timeout_seconds")
  clock_identity = _nonempty(clock_identity, "clock identity")
  if isolated_runner is None:
    from tinygrad.runtime.process_isolated import run_isolated
    isolated_runner = run_isolated
  if health_probe is None:
    from extra.qk.mmq_target_epoch_orchestrator import \
      spawned_tiny_health_probe
    health_probe = spawned_tiny_health_probe
  if fault_collector is None:
    from extra.qk.mmq_target_epoch_orchestrator import \
      collect_kernel_fault_evidence
    fault_collector = collect_kernel_fault_evidence
  if session_identity_factory is None:
    nonce = time.time_ns()
    session_identity_factory = lambda queue: _identity({
      "schema": f"{SCHEMA}.session_identity", "queue_mode": queue,
      "parent_pid": os.getpid(), "seed": seed, "nonce": nonce})
  if not all(callable(callback) for callback in (
      isolated_runner, health_probe, fault_collector,
      session_identity_factory)):
    raise TypeError("guard factories must be callable")

  queue_envelopes: dict[str, Any] = {}
  prior_blocker: str | None = None
  for queue_mode in QUEUE_MODES:
    if prior_blocker is not None:
      queue_envelopes[queue_mode] = _not_attempted(
        queue_mode, prior_blocker)
      continue
    request = GuardedQueueRequest(
      runtime_config=runtime_config, queue_mode=queue_mode,
      session_identity=_nonempty(
        session_identity_factory(queue_mode),
        f"{queue_mode} session identity"),
      clock_identity=clock_identity, warmups=warmups, rounds=rounds,
      seed=seed, candidate_route_builder=candidate_route_builder,
      direct_route_builder=direct_route_builder,
      host_io_census_factory=host_io_census_factory,
      child_factories=child_factories,
      direct_object_builder=direct_object_builder)
    queue_envelopes[queue_mode] = _guarded_queue_envelope(
      request=request, timeout_seconds=timeout_seconds,
      isolated_runner=isolated_runner, health_probe=health_probe,
      fault_collector=fault_collector)
    if queue_envelopes[queue_mode]["status"] != "PASS":
      prior_blocker = queue_envelopes[queue_mode]["exact_blocker"]

  blocker = next((
    queue_envelopes[queue]["exact_blocker"] for queue in QUEUE_MODES
    if queue_envelopes[queue]["status"] != "PASS"), None)
  collection = None
  if blocker is None:
    try:
      children = {
        queue: queue_envelopes[queue]["child"] for queue in QUEUE_MODES}
      authority_fields = (
        "family_identity", "input_identity", "workload_identity",
        "contract_identity", "composition_identity",
        "candidate_executable_identity")
      if any(
          children["PM4"].get(field) != children["AQL"].get(field)
          for field in authority_fields):
        raise ValueError("PM4/AQL child runtime authorities differ")
      if children["PM4"]["contract"] != children["AQL"]["contract"] or \
         children["PM4"]["contract_validation_kwargs"] != \
           children["AQL"]["contract_validation_kwargs"]:
        raise ValueError("PM4/AQL child contract authorities differ")
      if children["PM4"]["session_identity"] == \
         children["AQL"]["session_identity"]:
        raise ValueError("PM4/AQL session identities alias")
      collection = paired_collector(
        contract=children["PM4"]["contract"],
        contract_validation_kwargs=
          children["PM4"]["contract_validation_kwargs"],
        queue_sessions={
          queue: children[queue]["queue_session"] for queue in QUEUE_MODES},
        warmups=warmups, rounds=rounds, seed=seed,
        required_speedup=required_speedup)
      collection = dict(_mapping(collection, "paired C8 collection"))
      if collection.get("status") != "PASS":
        raise ValueError("paired C8 collector did not pass")
    except BaseException as exc:
      blocker = (
        "guarded ffn_gate_up paired collection failed closed: "
        f"{type(exc).__name__}: {exc}")
  payload = {
    "schema": SCHEMA, "status": "PASS" if blocker is None else "BLOCKED",
    "exact_blocker": blocker,
    "protocol": {
      "queue_order": list(QUEUE_MODES),
      "one_spawned_child_per_attempted_queue": True,
      "persistent_routes_built_once_per_child": True,
      "host_io_census_once_per_child": True,
      "warmups_per_route": warmups, "paired_rounds": rounds, "seed": seed,
      "no_retry": True, "retry_count": 0, "no_queue_fallback": True,
      "pre_post_health_per_attempted_queue": True,
      "fault_capture_per_attempted_queue": True,
    },
    "queue_sessions": queue_envelopes,
    "c8_collection": collection,
    "production_dispatch_changed": False,
    "promotion_evidence_eligible": False,
  }
  return {**payload, "evidence_identity": _identity(payload)}


__all__ = [
  "CHILD_SCHEMA", "CLOCK_IDENTITY", "QUEUE_ENVELOPE_SCHEMA", "SCHEMA",
  "GuardedChildFactories", "GuardedQueueRequest",
  "QueueRuntimeEnvironment", "amd_queue_runtime_environment",
  "run_guarded_ffn_gate_up_queue_child",
  "run_guarded_ffn_gate_up_sessions",
]
