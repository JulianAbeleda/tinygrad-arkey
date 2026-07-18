from __future__ import annotations

from types import SimpleNamespace

import pytest

from extra.qk import prefill_research_routes as routes
from extra.qk.prefill.frozen_exact_role_runtime import ABI_NAMES, ABI_DTYPES, EXECUTION_EVIDENCE_SCHEMA
from extra.qk.prefill.six_row_research_selector import (
  GROUPS, RETAINED_POLICY_IDENTITY, TARGET, ResearchPolicyBlocked, load_retained_policy,
)
from tinygrad.llm.prefill_route_observer import (
  PrefillRouteAttachment, observe_prefill_route_executions, observe_prefill_routes, prefill_route_scope,
)


class _Tensor:
  def __init__(self, shape): self.shape, self.device = tuple(shape), "CPU"
  def __getitem__(self, _index): return _Tensor(self.shape[1:])
  def cast(self, *_args): return self
  def contiguous(self): return self
  def reshape(self, *shape): return _Tensor(shape)


class _NoTensorWork(_Tensor):
  def __getitem__(self, _index): raise AssertionError("selection failure reached Tensor work")


def _group(role, quant="Q4_K"):
  return next(group for group in GROUPS if group.workload.role == role and group.workload.quant_format == quant)


def _authority(*, bundles=None, fallbacks=None, policy=None, target=None):
  candidate = _group("ffn_gate_up")
  return routes.ExactResearchRouteAuthority(
    load_retained_policy() if policy is None else policy,
    TARGET if target is None else target,
    {candidate.expected_binding_identity: "/frozen/gate-up.tar"} if bundles is None else bundles,
    {group.expected_binding_identity: f"program:{group.invocation_id}"
     for group in GROUPS if group.expected_binding_identity.startswith("fallback:")} if fallbacks is None else fallbacks)


def _config(**kwargs):
  return routes.PrefillResearchRouteConfig(
    exact_policy_enabled=True, exact_authority=_authority(**kwargs))


def _linear(group):
  workload = group.workload
  quant_attr = "q4k_storage" if workload.quant_format == "Q4_K" else "q6k_storage"
  linear = SimpleNamespace(
    bias=None, in_features=workload.k, out_features=workload.n, parts=1, opts=(),
    name=f"blk.0.{workload.role}.weight", route_role=workload.role,
    prefill_packed_weight=lambda: (_ for _ in ()).throw(AssertionError("test runtime touched packed weight")),
  )
  setattr(linear, quant_attr, SimpleNamespace())
  linear._prefill_route_attachment = PrefillRouteAttachment(
    "actual-model-invocation", "q4k_q8_five_buffer_research" if group is GROUPS[0] else "direct_packed",
    linear.name, {"artifact_identity": RETAINED_POLICY_IDENTITY,
                  "binding_identity": group.expected_binding_identity}, {"target": TARGET})
  return linear


def _observe(call):
  legacy, executions = [], []
  with observe_prefill_routes(legacy.append), \
       observe_prefill_route_executions(lambda linear, event: executions.append((linear, event))), \
       prefill_route_scope(True):
    output = call()
  return output, legacy, executions


def _execution_evidence(role_spec):
  return {
    "schema": EXECUTION_EVIDENCE_SCHEMA,
    "runtime": {
      "device": "AMD", "amd_aql_env": "0", "amd_aql_effective": False, "queue_mode": "PM4",
      "queue_class": "tinygrad.runtime.ops_amd.AMDComputeQueue",
      "runtime_class": "tinygrad.runtime.ops_amd.AMDProgram",
    },
    "staging": {
      "mode": "all_inputs_fixed_va_tinygrad_assign", "fixed_va": True,
      "persistent_buffers": True, "synchronized_before_overwrite": True,
      "transfer": "tinygrad_runtime_lowering",
      "inputs": [
        {"slot": slot, "name": name, "va": 0x100000 + slot * 0x10000,
         "nbytes": elements * dtype.itemsize, "allocation_nbytes": elements * dtype.itemsize}
        for slot, (name, elements, dtype) in enumerate(
          zip(ABI_NAMES[1:], role_spec.program.abi_elements[1:], ABI_DTYPES[1:]), start=1)
      ],
    },
    "dispatch": {
      "mode": "eager_native_runtime", "count": role_spec.epochs, "tinyjit_replay_captured": False,
    },
  }


def test_exact_candidate_runs_frozen_adapter_and_emits_legacy_and_actual_identity(monkeypatch):
  group, output, calls = GROUPS[0], object(), []
  lin = _linear(group)
  def frozen_runner(linear, activation, **kwargs):
    calls.append((linear, activation, kwargs))
    identity, role_spec = group.expected_binding_identity, kwargs["role_spec"]
    return SimpleNamespace(
      output=output, binding=SimpleNamespace(
        candidate_identity=identity, program_key="program-key:exact", role_spec=role_spec),
      evidence={"candidate_identity":identity, "program_key":"program-key:exact",
                "shape":list(role_spec.shape), "program_shape":list(role_spec.program.shape),
                "execution":_execution_evidence(role_spec)})
  monkeypatch.setattr(routes, "run_frozen_exact_q4k_research", frozen_runner)
  monkeypatch.setattr(routes, "_run_direct_packed_baseline",
                      lambda *_args: (_ for _ in ()).throw(AssertionError("candidate silently fell back")))

  result, legacy, executions = _observe(lambda: routes.route_direct_packed_prefill_research(
    lin, _Tensor((1, group.workload.m, group.workload.k)), config=_config()))
  assert result is output and legacy == [lin] and len(executions) == 1
  assert calls[0][0] is lin and calls[0][1].shape == (group.workload.m, group.workload.k)
  assert calls[0][2]["enabled"] is True and calls[0][2]["frozen_bundle"] == "/frozen/gate-up.tar"
  assert calls[0][2]["role_spec"].candidate_canonical_identity == group.expected_binding_identity
  event = executions[0][1]
  assert event.invocation_id == "actual-model-invocation"
  assert event.executed_route_id == "q4k_q8_five_buffer_research"
  assert event.candidate_identity == group.expected_binding_identity
  assert event.program_identity == "program-key:exact"
  assert event.fallback_used is False and event.fallback_reason is None
  assert event.execution_evidence["runtime"]["queue_mode"] == "PM4"
  assert event.execution_evidence["staging"]["fixed_va"] is True
  assert event.execution_evidence["dispatch"]["count"] == group.workload.k // 256
  assert event.execution_evidence["dispatch"]["tinyjit_replay_captured"] is False


@pytest.mark.parametrize("group", [_group("attn_qo"), _group("attn_kv", "Q6_K")])
def test_explicit_q4_or_q6_fallback_calls_existing_baseline_and_emits_actual_fallback(monkeypatch, group):
  output, baseline_calls = object(), []
  lin = _linear(group)
  def baseline(linear, x, spec):
    baseline_calls.append((linear, x, spec))
    routes.notify_prefill_route(linear)  # existing baseline's legacy notification
    return output
  monkeypatch.setattr(routes, "_run_direct_packed_baseline", baseline)
  monkeypatch.setattr(routes, "run_frozen_exact_q4k_research",
                      lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("fallback reached frozen runtime")))

  result, legacy, executions = _observe(lambda: routes.route_direct_packed_prefill_research(
    lin, _Tensor((1, group.workload.m, group.workload.k)), config=_config()))
  assert result is output and legacy == [lin] and len(baseline_calls) == 1
  event = executions[0][1]
  assert event.executed_route_id == "direct_packed"
  assert event.candidate_identity == group.expected_binding_identity
  assert event.program_identity == f"program:{group.invocation_id}"
  assert event.fallback_used is True and "six-row policy" in event.fallback_reason


@pytest.mark.parametrize("mutate, message", [
  (lambda lin, config: setattr(lin._prefill_route_attachment, "route_id", "wrong"), "attachment route"),
  (lambda lin, config: object.__setattr__(config.exact_authority, "target", {**TARGET, "arch":"gfx1200"}), "target authority"),
])
def test_exact_attachment_or_target_drift_fails_before_tensor_work(mutate, message):
  group, config = GROUPS[0], _config()
  lin = _linear(group)
  # PrefillRouteAttachment is frozen; replace it for route drift.
  if "attachment" in message:
    old = lin._prefill_route_attachment
    lin._prefill_route_attachment = PrefillRouteAttachment(
      old.invocation_id, "wrong", old.tensor_identity, old.selected_policy, old.scanned_target_facts)
  else:
    mutate(lin, config)
  with pytest.raises(ResearchPolicyBlocked, match=message):
    routes.route_direct_packed_prefill_research(
      lin, _NoTensorWork((1, group.workload.m, group.workload.k)), config=config)


def test_unknown_workload_and_missing_bundle_fail_before_tensor_work():
  group = GROUPS[0]
  unknown = _linear(group)
  unknown.out_features = group.workload.n + 128
  with pytest.raises(ResearchPolicyBlocked, match="unknown exact"):
    routes.route_direct_packed_prefill_research(
      unknown, _NoTensorWork((1, group.workload.m, group.workload.k)), config=_config())
  with pytest.raises(ResearchPolicyBlocked, match="no frozen bundle"):
    routes.route_direct_packed_prefill_research(
      _linear(group), _NoTensorWork((1, group.workload.m, group.workload.k)),
      config=_config(bundles={}))


def test_candidate_runtime_failure_or_identity_drift_never_silently_falls_back(monkeypatch):
  group, lin = GROUPS[0], _linear(GROUPS[0])
  baseline_calls = []
  monkeypatch.setattr(routes, "_run_direct_packed_baseline",
                      lambda *_args: baseline_calls.append(True))
  monkeypatch.setattr(routes, "run_frozen_exact_q4k_research",
                      lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("candidate failed")))
  with pytest.raises(RuntimeError, match="candidate failed"):
    routes.route_direct_packed_prefill_research(
      lin, _Tensor((1, group.workload.m, group.workload.k)), config=_config())
  assert baseline_calls == []

  monkeypatch.setattr(routes, "run_frozen_exact_q4k_research", lambda *_args, **_kwargs: SimpleNamespace(
    output=object(), binding=SimpleNamespace(candidate_identity="drift", program_key="wrong", role_spec=object()),
    evidence={"candidate_identity":"drift", "program_key":"wrong", "shape":[], "program_shape":[]}))
  with pytest.raises(RuntimeError, match="identity drifted"):
    routes.route_direct_packed_prefill_research(
      lin, _Tensor((1, group.workload.m, group.workload.k)), config=_config())
  assert baseline_calls == []


def test_enabled_exact_policy_rejects_non_route_linear_instead_of_returning_none():
  group = GROUPS[0]
  linear = SimpleNamespace(bias=None, in_features=group.workload.k, out_features=group.workload.n)
  with pytest.raises(ResearchPolicyBlocked, match="does not match"):
    routes.route_direct_packed_prefill_research(
      linear, _NoTensorWork((1, group.workload.m, group.workload.k)), config=_config())


def test_exact_policy_remains_default_off_and_old_research_path_is_unchanged(monkeypatch):
  calls, output = [], object()
  monkeypatch.setattr(routes, "_run_direct_packed_baseline",
                      lambda lin, x, spec: calls.append((lin, x, spec)) or output)
  lin = _linear(_group("attn_qo"))
  # Invalid authority is inert unless the explicit boolean is also enabled.
  config = routes.PrefillResearchRouteConfig(
    exact_policy_enabled=False,
    exact_authority=routes.ExactResearchRouteAuthority({}, {}, {}, {}))
  assert routes.route_direct_packed_prefill_research(
    lin, _Tensor((1, 512, 5120)), config=config) is output
  assert len(calls) == 1
