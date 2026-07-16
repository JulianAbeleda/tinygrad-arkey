import json
import numpy as np
import pytest

from extra.qk.prefill.isolated_guarded_executor import IsolatedExecutionResult
from extra.qk.prefill import operand_path_execution_worker as worker
from extra.qk.prefill.operand_path_execution_worker import AdapterRegistry, PreparedExecution, execute, execute_session, process
from extra.qk.prefill.execution_bridge_contracts import (CounterGroupRequest, CorrectnessProtocol, ExecutionRequest,
  GuardProtocol, TimingProtocol, TransportPlan)


class FakeAdapter:
  def prepare(self, request):
    return PreparedExecution(lambda: None, {"a": np.array([1, 2])}, np.array([1, 2]),
                             {"adapter": "fake", "transport": request.transport_plan.transport})


def request(**changes):
  values = dict(experiment_id="exp", candidate_id="cand", comparator_id="base", workload_digest="w",
    schedule_digest="s", transport_plan=TransportPlan("cache_streamed", "s"),
    target_context={"provider": "cpu-test", "session_id": "single-test"}, compiler_context={"adapter_id": "fake"},
    correctness=CorrectnessProtocol("fake", rtol=0, atol=0), guard=GuardProtocol(1000),
    timing=TimingProtocol(1, 2, 7), counter_groups=(CounterGroupRequest("pmc", ("L2_HIT",)),))
  values.update(changes)
  return ExecutionRequest(**values)


def fake_runner(**kwargs):
  return IsolatedExecutionResult("completed", True, {}, True, guarded={"elapsed_seconds": 0.002,
    "device_healthy_before": True, "finite_output": True, "numerics_passed": True,
    "full_output_compared": True, "inputs_unchanged": True, "max_abs_error": 0.0},
                                 identity=dict(kwargs["request"].identity or {}))


def test_explicit_adapter_runs_typed_compile_correctness_timing_and_preserves_timing_when_counters_unsupported():
  registry = AdapterRegistry(); registry.register("fake", FakeAdapter())
  out = execute(request(), registry=registry, runner=fake_runner)
  phases = {phase.phase: phase for phase in out.phases}
  assert [phase.phase for phase in out.phases] == ["compile", "execution", "correctness", "timing", "counter"]
  assert phases["execution"].status == "passed" and phases["execution"].evidence["health"]["postflight"] is True
  assert phases["correctness"].evidence["numerical_passed"] is True
  assert phases["timing"].status == "passed" and phases["timing"].evidence["samples"] == [0.002, 0.002]
  assert phases["counter"].status == "unsupported" and phases["counter"].unsupported[0].feature == "pmc"
  assert json.loads(out.to_json()) == process(request().to_dict(), registry=registry, runner=fake_runner)


def test_unknown_adapter_fails_closed_without_route_or_lds_fallback():
  called = False
  def runner(**kwargs):
    nonlocal called; called = True
  out = execute(request(compiler_context={"adapter_id": "route_named_like_lds"}), registry=AdapterRegistry(), runner=runner)
  assert not called
  assert len(out.phases) == 1 and out.phases[0].phase == "compile" and out.phases[0].status == "unsupported"
  assert out.phases[0].unsupported[0].feature == "provider_adapter"


def test_failed_guarded_correctness_stops_before_timing():
  registry = AdapterRegistry(); registry.register("fake", FakeAdapter())
  calls = 0
  def fail(**kwargs):
    nonlocal calls; calls += 1
    return IsolatedExecutionResult("failed", False, {}, True, errors=("bad output",))
  out = execute(request(), registry=registry, runner=fail)
  assert calls == 1 and [phase.phase for phase in out.phases] == ["compile", "execution", "correctness"]
  assert out.phases[-1].status == "failed"


def test_session_gates_all_candidates_then_warms_up_and_randomizes_each_measured_round():
  registry = AdapterRegistry(); registry.register("fake", FakeAdapter())
  events, calls = [], {}
  def runner(**kwargs):
    cid = kwargs["request"].identity["candidate_id"]
    calls[cid] = calls.get(cid, 0) + 1
    events.append(cid)
    return IsolatedExecutionResult("completed", True, {}, True, guarded={"elapsed_seconds": calls[cid] / 1000,
      "finite_output": True, "numerics_passed": True, "full_output_compared": True,
      "inputs_unchanged": True}, identity=dict(kwargs["request"].identity))
  reqs = [request(candidate_id=cid, timing=TimingProtocol(1, 3, 17)) for cid in ("a", "b", "c")]
  out = execute_session(reqs, registry=registry, runner=runner, session_id="session-test")
  assert events[:3] == ["a", "b", "c"]  # all correctness gates precede warmups
  assert events[3:6] == ["a", "b", "c"]
  orders = out[0].extensions["measured_launch_order"]
  assert events[6:] == [cid for row in orders for cid in row]
  assert len(orders) == 3 and all(sorted(row) == ["a", "b", "c"] for row in orders)
  for result in out:
    timing = next(phase for phase in result.phases if phase.phase == "timing")
    assert result.extensions["identity"]["session_id"] == "session-test"
    assert timing.identity["session_id"] == "session-test"
    assert timing.evidence["units"] == "s" and len(timing.evidence["samples"]) == 3


def test_session_candidate_fault_is_fail_closed_without_stopping_other_candidates():
  registry = AdapterRegistry(); registry.register("fake", FakeAdapter())
  calls = {"good": 0, "bad": 0}
  def runner(**kwargs):
    cid = kwargs["request"].identity["candidate_id"]; calls[cid] += 1
    if cid == "bad" and calls[cid] == 2:
      return IsolatedExecutionResult("timed_out", False, {}, True, errors=("timeout",))
    return fake_runner(**kwargs)
  out = execute_session([request(candidate_id="good"), request(candidate_id="bad")], registry=registry,
                        runner=runner, session_id="fault-session")
  good, bad = out
  assert next(phase for phase in good.phases if phase.phase == "timing").status == "passed"
  bad_timing = next(phase for phase in bad.phases if phase.phase == "timing")
  assert bad_timing.status == "failed" and bad_timing.error.context["dispatch_state"] == "timed_out"
  assert calls == {"good": 4, "bad": 2}


def test_session_rejects_incompatible_cohort_before_prepare_or_dispatch():
  class CountingAdapter(FakeAdapter):
    def __init__(self): self.prepares = 0
    def prepare(self, request): self.prepares += 1; return super().prepare(request)
  adapter = CountingAdapter(); registry = AdapterRegistry(); registry.register("fake", adapter)
  runner_calls = 0
  def runner(**kwargs):
    nonlocal runner_calls; runner_calls += 1
  import pytest
  with pytest.raises(ValueError, match="compatible timing"):
    execute_session([request(candidate_id="a"), request(candidate_id="b", timing=TimingProtocol(1, 3, 7))],
                    registry=registry, runner=runner)
  assert adapter.prepares == 0 and runner_calls == 0


def test_result_propagates_request_and_compile_identity_without_inventing_executed_binary():
  class IdentityAdapter:
    def __init__(self, equality): self.equality = equality
    def prepare(self, request):
      evidence = {"binary_sha256": "b" * 64, "source_sha256": "a" * 64, "target": "gfx1100",
                  "canonical_identity": "c" * 64}
      if self.equality is not None: evidence["executed_binary_matches_compile"] = self.equality
      return PreparedExecution(lambda: None, {"a": np.array([1, 2])}, np.array([1, 2]), evidence)
  enriched = request(target_context={"provider": "cpu-test", "workload": {"shape": [1, 2]},
    "target_id": "target-1", "system_snapshot_id": "snapshot-1", "session_id": "single-session"})
  for equality, has_executed_hash in ((None, False), (False, False), (True, True)):
    registry = AdapterRegistry(); registry.register("fake", IdentityAdapter(equality))
    out = execute(enriched, registry=registry, runner=fake_runner)
    compile_phase, execution_phase = out.phases[:2]
    assert compile_phase.identity == {"binary_sha256": "b" * 64, "source_sha256": "a" * 64,
      "target": "gfx1100", "canonical_identity": "c" * 64}
    assert execution_phase.identity["binary_sha256"] == "b" * 64
    assert ("executed_binary_sha256" in execution_phase.evidence) is has_executed_hash
    assert execution_phase.identity.get("executed_binary_sha256") == (("b" * 64) if has_executed_hash else None)
    assert out.extensions == {"workload": {"shape": [1, 2]}, "identity": {"target_id": "target-1",
      "system_snapshot_id": "snapshot-1", "session_id": "single-session"}}


def test_batch_extensions_include_each_request_workload_and_shared_session():
  registry = AdapterRegistry(); registry.register("fake", FakeAdapter())
  reqs = [request(candidate_id=cid, target_context={"provider": "cpu-test", "workload": {"shape": [1, 2]},
    "target_id": "target-1", "system_snapshot_id": "snapshot-1"}) for cid in ("a", "b")]
  out = execute_session(reqs, registry=registry, runner=fake_runner, session_id="batch-session")
  assert [result.extensions["workload"] for result in out] == [{"shape": [1, 2]}, {"shape": [1, 2]}]
  assert all(result.extensions["identity"]["session_id"] == "batch-session" for result in out)


def test_process_lazily_registers_production_adapter_without_import_time_mutation(monkeypatch):
  production_id = "tinygrad.amd.gfx1100.current_prefill.v1"
  calls = {"imported": 0, "prepared": 0}

  class ProductionAdapter:
    def prepare(self, request):
      calls["prepared"] += 1
      return PreparedExecution(lambda: None, {"a": np.array([1, 2])}, np.array([1, 2]),
                               {"adapter": "production", "transport": request.transport_plan.transport})

  class Module:
    @staticmethod
    def register_current_prefill_adapter(registry):
      calls["imported"] += 1
      registry.register(production_id, ProductionAdapter())

  monkeypatch.setattr(worker.importlib, "import_module", lambda name: Module())
  registry = AdapterRegistry()
  assert registry.ids() == ()
  out = process(request(compiler_context={"adapter_id": production_id}).to_dict(), registry=registry, runner=fake_runner)
  assert calls == {"imported": 1, "prepared": 1}
  assert registry.ids() == (production_id,)
  assert [phase["phase"] for phase in out["phases"]] == ["compile", "execution", "correctness", "timing", "counter"]


def test_main_consumes_stdin_json_via_lazy_production_registry_without_gpu_dispatch(monkeypatch, capsys):
  production_id = "tinygrad.amd.gfx1100.current_prefill.v1"

  class ProductionAdapter:
    def prepare(self, request):
      return PreparedExecution(lambda: None, {"a": np.array([1, 2])}, np.array([1, 2]),
                               {"adapter": "production", "transport": request.transport_plan.transport})

  class Module:
    @staticmethod
    def register_current_prefill_adapter(registry):
      registry.register(production_id, ProductionAdapter())

  payload = json.dumps(request(compiler_context={"adapter_id": production_id}).to_dict())
  original_process = worker.process
  monkeypatch.setattr(worker.importlib, "import_module", lambda name: Module())
  monkeypatch.setattr(worker.sys, "stdin", type("Stdin", (), {"read": lambda self: payload})())
  monkeypatch.setattr(worker, "process", lambda row, registry=worker.ADAPTERS, runner=worker.run_isolated_guarded_execution:
                      original_process(row, registry=registry, runner=fake_runner))
  assert worker.main() == 0
  out = json.loads(capsys.readouterr().out)
  assert out["candidate_id"] == "cand"
  assert [phase["phase"] for phase in out["phases"]] == ["compile", "execution", "correctness", "timing", "counter"]


def test_process_fails_closed_for_unknown_production_adapter(monkeypatch):
  monkeypatch.setattr(worker.importlib, "import_module", pytest.fail)
  out = process(request(compiler_context={"adapter_id": "tinygrad.amd.gfx1100.unknown"}).to_dict(),
                registry=AdapterRegistry(), runner=fake_runner)
  assert [phase["phase"] for phase in out["phases"]] == ["compile"]
  assert out["phases"][0]["status"] == "unsupported"
