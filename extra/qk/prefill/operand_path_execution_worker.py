#!/usr/bin/env python3
"""Typed JSON provider worker for operand-path candidate execution.

Adapters are explicit provider authority.  This module never derives an adapter
from a route/candidate name and has no default transport (including LDS).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import json, random, statistics, sys, uuid
from typing import Any, Callable, Mapping, Protocol

from extra.qk.prefill.guarded_execution import GuardPolicy
from extra.qk.prefill.isolated_guarded_executor import (ExecutionRequest as GuardedRequest,
  IsolatedExecutionResult, run_isolated_guarded_execution)
from tinygrad.runtime.execution_bridge_contracts import (ExecutionRequest, ExecutionResult, PhaseResult,
  TypedError, UnsupportedOutcome)

PROTOCOL = "tinygrad.operand_path_execution_worker.v1"


@dataclass(frozen=True)
class PreparedExecution:
  """Spawn-safe material prepared by one explicitly selected provider adapter."""
  builder: Callable[[], Any]
  inputs: Mapping[str, Any]
  reference: Any
  compile_evidence: Mapping[str, Any]
  health_probe: Callable[[], bool] | None = None
  output_dtype: Any = None
  counter_support: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


class ProviderAdapter(Protocol):
  def prepare(self, request: ExecutionRequest) -> PreparedExecution: ...


class AdapterRegistry:
  def __init__(self) -> None: self._adapters: dict[str, ProviderAdapter] = {}
  def register(self, adapter_id: str, adapter: ProviderAdapter) -> None:
    if not isinstance(adapter_id, str) or not adapter_id.strip(): raise ValueError("adapter_id must be non-empty")
    if not callable(getattr(adapter, "prepare", None)): raise TypeError("adapter must define prepare(request)")
    self._adapters[adapter_id] = adapter
  def resolve(self, adapter_id: str) -> ProviderAdapter | None: return self._adapters.get(adapter_id)
  def ids(self) -> tuple[str, ...]: return tuple(sorted(self._adapters))


ADAPTERS = AdapterRegistry()
_PRODUCTION_ADAPTER_LOADERS: dict[str, tuple[str, str]] = {
  "tinygrad.amd.gfx1100.current_prefill.v1": (
    "extra.qk.prefill.current_prefill_execution_adapter",
    "register_current_prefill_adapter",
  ),
  "tinygrad.amd.gfx1100.current_qwen3_8b_decode_q4k_g3.compile_only.v1": (
    "extra.qk.decode.current_decode_execution_adapter",
    "register_current_decode_adapter",
  ),
}


def _error(request: ExecutionRequest, phase: str, code: str, *, recoverable: bool, context: Mapping[str, Any]) -> PhaseResult:
  return PhaseResult(phase, "failed", error=TypedError(code, phase, recoverable, candidate=request.candidate_id, context=context))


def register_production_adapter(adapter_id: str, registry: AdapterRegistry = ADAPTERS) -> bool:
  loader = _PRODUCTION_ADAPTER_LOADERS.get(adapter_id)
  if loader is None: return False
  if registry.resolve(adapter_id) is not None: return True
  module_name, register_name = loader
  register = getattr(importlib.import_module(module_name), register_name)
  register(registry)
  return registry.resolve(adapter_id) is not None


def resolve_adapter(adapter_id: Any, registry: AdapterRegistry = ADAPTERS, *, allow_production: bool = False) -> ProviderAdapter | None:
  if not isinstance(adapter_id, str): return None
  adapter = registry.resolve(adapter_id)
  if adapter is not None or not allow_production: return adapter
  register_production_adapter(adapter_id, registry)
  return registry.resolve(adapter_id)


def _guarded_request(request: ExecutionRequest, prepared: PreparedExecution) -> GuardedRequest:
  correctness, guard = request.correctness, request.guard
  policy = GuardPolicy(timeout_seconds=(guard.hard_timeout_ms / 1000 if guard else 30.0),
    rtol=correctness.rtol if correctness else 0.0, atol=correctness.atol if correctness else 0.0,
    check_inputs_unchanged=correctness.immutable_inputs if correctness else True)
  kwargs = {"inputs": prepared.inputs, "reference": prepared.reference, "policy": policy,
            "identity": {"experiment_id": request.experiment_id, "candidate_id": request.candidate_id,
                         "request_digest": request.digest, "schedule_digest": request.schedule_digest}}
  if prepared.output_dtype is not None: kwargs["output_dtype"] = prepared.output_dtype
  return GuardedRequest(**kwargs)


def _run(prepared: PreparedExecution, guarded: GuardedRequest,
         runner: Callable[..., IsolatedExecutionResult]) -> IsolatedExecutionResult:
  return runner(builder=prepared.builder, request=guarded, health_probe=prepared.health_probe,
                timeout_seconds=guarded.policy.timeout_seconds)


def _outcome_status(outcome: IsolatedExecutionResult) -> str:
  if outcome.passed: return "passed"
  return "timed_out" if outcome.dispatch_state == "timed_out" else "failed"


_COMPILE_IDENTITY_KEYS = ("binary_sha256", "source_sha256", "target_id", "canonical_identity",
                          "input_identity", "reference_identity", "abi_digest", "candidate_digest")


def _compile_identity(evidence: Mapping[str, Any]) -> dict[str, Any]:
  identity = {key: evidence[key] for key in _COMPILE_IDENTITY_KEYS if evidence.get(key) is not None}
  if "target_id" not in identity and evidence.get("target") is not None: identity["target"] = evidence["target"]
  return identity


def _result_extensions(request: ExecutionRequest, session_id: str | None = None) -> dict[str, Any]:
  extensions: dict[str, Any] = {}
  if "workload" in request.target_context: extensions["workload"] = request.target_context["workload"]
  identity = {key: request.target_context[key] for key in
              ("target_id", "system_snapshot_id", "reference_identity", "input_identity", "clock_state_id")
              if request.target_context.get(key) is not None}
  supplied_session = session_id if session_id is not None else request.target_context.get("session_id")
  if supplied_session is not None: identity["session_id"] = supplied_session
  if identity: extensions["identity"] = identity
  return extensions


def _execution_evidence(outcome: IsolatedExecutionResult,
                        compile_evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
  guarded = dict(outcome.guarded or {})
  evidence = {"dispatch_state": outcome.dispatch_state, "health": {
    "preflight": guarded.get("device_healthy_before"), "postflight": outcome.health_after,
    "device_fault": guarded.get("device_fault", outcome.dispatch_state == "device_lost"),
    "timeout_contained": outcome.dispatch_state == "timed_out" and outcome.health_after,
  }, "guarded": guarded}
  # This is a provider assertion, not an inference from successful dispatch.
  if compile_evidence is not None and compile_evidence.get("executed_binary_matches_compile") is True:
    binary_sha256 = compile_evidence.get("binary_sha256")
    if isinstance(binary_sha256, str) and binary_sha256: evidence["executed_binary_sha256"] = binary_sha256
  return evidence


def _execution_identity(outcome: IsolatedExecutionResult, compile_evidence: Mapping[str, Any],
                        session_id: str) -> dict[str, Any]:
  identity = {**dict(outcome.identity), **_compile_identity(compile_evidence), "session_id": session_id}
  if compile_evidence.get("executed_binary_matches_compile") is True:
    binary = compile_evidence.get("binary_sha256")
    if isinstance(binary, str) and binary: identity["executed_binary_sha256"] = binary
  return identity


def _counter_phase(request: ExecutionRequest, prepared: PreparedExecution) -> PhaseResult | None:
  if not request.counter_groups: return None
  unsupported, supported = [], {}
  for group in request.counter_groups:
    row = prepared.counter_support.get(group.group_id)
    if row is None:
      unsupported.append(UnsupportedOutcome("counter group unavailable", "counter", group.group_id,
                                            {"counters": list(group.counters)}))
    else: supported[group.group_id] = dict(row)
  return PhaseResult("counter", "passed" if supported else "unsupported",
                     evidence=supported, unsupported=tuple(unsupported))


def _execution_result(request: ExecutionRequest, phases: list[PhaseResult],
                      session_id: str | None = None, extra: Mapping[str, Any] | None = None) -> ExecutionResult:
  extensions = _result_extensions(request, session_id)
  if extra is not None: extensions.update(extra)
  return ExecutionResult(request.experiment_id, request.candidate_id, request.digest, tuple(phases), extensions)


def _element_count(reference: Any) -> int | None:
  size = getattr(reference, "size", None)
  if isinstance(size, int): return size
  try:
    total = 1
    for dim in reference.shape: total *= int(dim)
    return total
  except (AttributeError, TypeError, ValueError): return None


def _correctness_evidence(request: ExecutionRequest, prepared: PreparedExecution,
                          outcome: IsolatedExecutionResult) -> dict[str, Any]:
  guarded = dict(outcome.guarded or {})
  finite = guarded.get("finite_output")
  numerical = guarded.get("numerics_passed")
  return {"scope": "full_gemm", "element_count": _element_count(prepared.reference),
          "max_error": guarded.get("max_abs_error"), "finite_output": finite,
          "nan_fraction": guarded.get("nan_fraction", 0.0 if finite is True else None),
          "inf_fraction": guarded.get("inf_fraction", 0.0 if finite is True else None),
          "finite_min": guarded.get("finite_min"), "finite_max": guarded.get("finite_max"),
          "tolerance_abs": request.correctness.atol if request.correctness else guarded.get("atol"),
          "tolerance_rel": request.correctness.rtol if request.correctness else guarded.get("rtol"),
          "numerical_passed": numerical, "full_output_compared": guarded.get("full_output_compared"),
          "inputs_unchanged": guarded.get("inputs_unchanged")}


def execute(request: ExecutionRequest, *, registry: AdapterRegistry = ADAPTERS,
            runner: Callable[..., IsolatedExecutionResult] = run_isolated_guarded_execution) -> ExecutionResult:
  """Compile, correctness-gate, and time one typed request."""
  phases: list[PhaseResult] = []
  session_id = str(request.target_context.get("session_id") or uuid.uuid4().hex)
  adapter_id = request.compiler_context.get("adapter_id")
  adapter = resolve_adapter(adapter_id, registry)
  if adapter is None:
    detail = {"adapter_id": adapter_id, "registered": list(registry.ids())}
    phases.append(PhaseResult("compile", "unsupported", unsupported=(UnsupportedOutcome(
      "explicit provider adapter is not registered", "compile", "provider_adapter", detail),)))
    return _execution_result(request, phases, session_id)
  try: prepared = adapter.prepare(request)
  except Exception as exc:
    phases.append(_error(request, "compile", "adapter_compile_failed", recoverable=True,
                         context={"adapter_id": adapter_id, "error": f"{type(exc).__name__}: {exc}"}))
    return _execution_result(request, phases, session_id)
  compile_identity = _compile_identity(prepared.compile_evidence)
  phases.append(PhaseResult("compile", "passed", identity=compile_identity, evidence=dict(prepared.compile_evidence)))
  guarded = _guarded_request(request, prepared)
  correctness = _run(prepared, guarded, runner)
  outcome_status = _outcome_status(correctness)
  phases.append(PhaseResult("execution", outcome_status,
    identity=_execution_identity(correctness, prepared.compile_evidence, session_id),
    evidence=_execution_evidence(correctness, prepared.compile_evidence),
    error=None if correctness.passed else TypedError("guarded_execution_failed", "execution", False,
      candidate=request.candidate_id, context={"errors": list(correctness.errors)})))
  phases.append(PhaseResult("correctness", outcome_status,
    identity=correctness.identity, evidence=_correctness_evidence(request, prepared, correctness),
    error=None if correctness.passed else TypedError("guarded_correctness_failed", "correctness", False,
      candidate=request.candidate_id, context={"errors": list(correctness.errors)})))
  if not correctness.passed:
    return _execution_result(request, phases, session_id)

  if request.timing is not None:
    runs: list[float] = []
    total = request.timing.warmups + request.timing.rounds
    # Candidate-order randomization belongs to the multi-candidate session coordinator.
    # Within one request, warmups must precede measured rounds.
    for idx in range(total):
      outcome = _run(prepared, guarded, runner)
      if not outcome.passed:
        phases.append(_error(request, "timing", "guarded_timing_failed", recoverable=False,
                             context={"run": idx, "errors": list(outcome.errors)}))
        return _execution_result(request, phases, session_id)
      if idx >= request.timing.warmups:
        elapsed = (outcome.guarded or {}).get("elapsed_seconds")
        if not isinstance(elapsed, (int, float)):
          phases.append(_error(request, "timing", "missing_elapsed_time", recoverable=True, context={"run": idx}))
          return _execution_result(request, phases, session_id)
        runs.append(float(elapsed))
    value = statistics.median(runs) if request.timing.statistic == "median" else statistics.mean(runs)
    phases.append(PhaseResult("timing", "passed", evidence={"scope": "kernel", "samples": runs, "units": "s",
      "value": value, "statistic": request.timing.statistic, "warmups": request.timing.warmups,
      "repetitions": request.timing.rounds, "inclusion": "kernel_only", "sync": "guarded_dispatch",
      "noise_threshold": request.timing.noise_threshold,
      **({"clock_state_id": request.target_context["clock_state_id"]}
         if request.target_context.get("clock_state_id") is not None else {})}))

  if (counter := _counter_phase(request, prepared)) is not None: phases.append(counter)
  return _execution_result(request, phases, session_id)


def _session_result(request: ExecutionRequest, phases: list[PhaseResult], session_id: str,
                    measured_order: list[list[str]]) -> ExecutionResult:
  return _execution_result(request, phases, session_id,
    {"measured_launch_order": [list(row) for row in measured_order]})


def execute_session(requests: tuple[ExecutionRequest, ...] | list[ExecutionRequest], *,
                    registry: AdapterRegistry = ADAPTERS,
                    runner: Callable[..., IsolatedExecutionResult] = run_isolated_guarded_execution,
                    session_id: str | None = None) -> tuple[ExecutionResult, ...]:
  """Correctness-gate and benchmark two or more candidates in one seeded session."""
  cohort = tuple(requests)
  if len(cohort) < 2 or any(not isinstance(request, ExecutionRequest) for request in cohort):
    raise ValueError("execute_session requires at least two ExecutionRequest values")
  first = cohort[0]
  if any(request.experiment_id != first.experiment_id for request in cohort):
    raise ValueError("session requests must have one experiment_id")
  if any(request.workload_digest != first.workload_digest for request in cohort):
    raise ValueError("session requests must have one workload_digest")
  comparable_context = ("workload", "target_id", "system_snapshot_id", "clock_state_id")
  if any(any(request.target_context.get(key) != first.target_context.get(key) for key in comparable_context)
         for request in cohort):
    raise ValueError("session requests must have one workload, target, system snapshot, and clock state")
  if len({request.candidate_id for request in cohort}) != len(cohort):
    raise ValueError("session candidate_id values must be unique")
  timing = first.timing
  if timing is None or any(request.timing != timing for request in cohort):
    raise ValueError("session requests must have one compatible timing protocol")
  if not timing.same_session:
    raise ValueError("session timing protocol must require same_session")
  sid = session_id or uuid.uuid4().hex
  if not isinstance(sid, str) or not sid.strip(): raise ValueError("session_id must be non-empty")

  phases: dict[str, list[PhaseResult]] = {request.candidate_id: [] for request in cohort}
  prepared: dict[str, PreparedExecution] = {}
  guarded: dict[str, GuardedRequest] = {}
  active: set[str] = set()
  measured_order: list[list[str]] = []

  # Resolve and prepare the complete cohort before allowing any dispatch.
  for request in cohort:
    candidate_phases = phases[request.candidate_id]
    adapter_id = request.compiler_context.get("adapter_id")
    adapter = resolve_adapter(adapter_id, registry)
    if adapter is None:
      detail = {"adapter_id": adapter_id, "registered": list(registry.ids())}
      candidate_phases.append(PhaseResult("compile", "unsupported", unsupported=(UnsupportedOutcome(
        "explicit provider adapter is not registered", "compile", "provider_adapter", detail),)))
      continue
    try: item = adapter.prepare(request)
    except Exception as exc:
      candidate_phases.append(_error(request, "compile", "adapter_compile_failed", recoverable=True,
        context={"adapter_id": adapter_id, "error": f"{type(exc).__name__}: {exc}"}))
      continue
    candidate_phases.append(PhaseResult("compile", "passed", identity=_compile_identity(item.compile_evidence),
                                        evidence=dict(item.compile_evidence)))
    prepared[request.candidate_id] = item
    guarded[request.candidate_id] = _guarded_request(request, item)

  # Gate every prepared candidate before the first warmup or measured launch.
  for request in cohort:
    cid = request.candidate_id
    if cid not in prepared: continue
    try: outcome = _run(prepared[cid], guarded[cid], runner)
    except Exception as exc:
      outcome = IsolatedExecutionResult("failed", False, {}, False, errors=(f"{type(exc).__name__}: {exc}",))
    status = _outcome_status(outcome)
    phases[cid].append(PhaseResult("execution", status,
      identity=_execution_identity(outcome, prepared[cid].compile_evidence, sid),
      evidence=_execution_evidence(outcome, prepared[cid].compile_evidence), error=None if outcome.passed else TypedError(
        "guarded_execution_failed", "execution", False, candidate=cid, context={"errors": list(outcome.errors)})))
    phases[cid].append(PhaseResult("correctness", status, identity={**dict(outcome.identity), "session_id": sid},
      evidence=_correctness_evidence(request, prepared[cid], outcome), error=None if outcome.passed else TypedError(
        "guarded_correctness_failed", "correctness", False, candidate=cid, context={"errors": list(outcome.errors)})))
    if outcome.passed: active.add(cid)

  samples: dict[str, list[float]] = {request.candidate_id: [] for request in cohort}
  by_id = {request.candidate_id: request for request in cohort}

  # Warmups are deliberately complete before measurement begins.
  for cid in [request.candidate_id for request in cohort if request.candidate_id in active]:
    for warmup in range(timing.warmups):
      try: outcome = _run(prepared[cid], guarded[cid], runner)
      except Exception as exc:
        outcome = IsolatedExecutionResult("failed", False, {}, False, errors=(f"{type(exc).__name__}: {exc}",))
      if not outcome.passed:
        phases[cid].append(_error(by_id[cid], "timing", "guarded_timing_failed", recoverable=False,
          context={"stage": "warmup", "run": warmup, "dispatch_state": outcome.dispatch_state,
                   "errors": list(outcome.errors)}))
        active.remove(cid)
        break

  rng = random.Random(timing.randomization_seed)
  for round_index in range(timing.rounds):
    order = [request.candidate_id for request in cohort if request.candidate_id in active]
    rng.shuffle(order)
    measured_order.append(list(order))
    for cid in order:
      try: outcome = _run(prepared[cid], guarded[cid], runner)
      except Exception as exc:
        outcome = IsolatedExecutionResult("failed", False, {}, False, errors=(f"{type(exc).__name__}: {exc}",))
      if not outcome.passed:
        phases[cid].append(_error(by_id[cid], "timing", "guarded_timing_failed", recoverable=False,
          context={"stage": "measurement", "round": round_index, "dispatch_state": outcome.dispatch_state,
                   "errors": list(outcome.errors)}))
        active.remove(cid)
        continue
      elapsed = (outcome.guarded or {}).get("elapsed_seconds")
      if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
        phases[cid].append(_error(by_id[cid], "timing", "missing_elapsed_time", recoverable=True,
          context={"round": round_index}))
        active.remove(cid)
      else: samples[cid].append(float(elapsed))

  for request in cohort:
    cid = request.candidate_id
    if cid in active:
      runs = samples[cid]
      value = statistics.median(runs) if timing.statistic == "median" else statistics.mean(runs)
      phases[cid].append(PhaseResult("timing", "passed", identity={"session_id": sid}, evidence={
        "scope": "kernel", "samples": runs, "units": "s", "value": value, "statistic": timing.statistic,
        "warmups": timing.warmups, "repetitions": timing.rounds, "inclusion": "kernel_only",
        "sync": "guarded_dispatch", "noise_threshold": timing.noise_threshold,
        **({"clock_state_id": request.target_context["clock_state_id"]}
           if request.target_context.get("clock_state_id") is not None else {})}))
    if cid in prepared and (counter := _counter_phase(request, prepared[cid])) is not None:
      phases[cid].append(counter)
  return tuple(_session_result(request, phases[request.candidate_id], sid, measured_order) for request in cohort)


def process(row: Any, *, registry: AdapterRegistry = ADAPTERS,
            runner: Callable[..., IsolatedExecutionResult] = run_isolated_guarded_execution) -> dict[str, Any]:
  request = ExecutionRequest.from_dict(row)
  adapter_id = request.compiler_context.get("adapter_id")
  adapter = resolve_adapter(adapter_id, registry, allow_production=True)
  if adapter is None:
    return execute(request, registry=registry, runner=runner).to_dict()
  scoped = AdapterRegistry()
  scoped.register(adapter_id, adapter)
  return execute(request, registry=scoped, runner=runner).to_dict()


def main() -> int:
  try: response = process(json.loads(sys.stdin.read()))
  except Exception as exc:
    response = {"protocol": PROTOCOL, "ok": False,
                "error": {"code": "malformed_request", "message": f"{type(exc).__name__}: {exc}"}}
  sys.stdout.write(json.dumps(response, sort_keys=True, separators=(",", ":")) + "\n")
  return 0


if __name__ == "__main__": raise SystemExit(main())
