from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tinygrad import Tensor, dtypes
from tinygrad.engine.realize import get_call_arg_uops
from tinygrad.function import function
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops

from extra.qk.mmq_exact_role_spec import exact_role_spec
from extra.qk.mmq_frozen_target_artifact import FrozenTargetArtifact
from extra.qk.prefill import frozen_exact_role_scheduler as scheduler
from extra.qk.q4k_q8_activation_producer import Q4KQ8ActivationTile
from test.unit.test_frozen_exact_role_runtime import _artifact


def _scheduler_artifact(role_spec, *, outs=(0,), ins=(0, 1, 2, 3, 4)):
  artifact = _artifact(role_spec)
  program = artifact.program.replace(arg=replace(artifact.program.arg, outs=outs, ins=ins))
  manifest = {
    **artifact.manifest,
    "program": {**artifact.manifest["program"], "key": program.key.hex()},
  }
  return FrozenTargetArtifact(
    manifest, program, artifact.binary, artifact.source, artifact.disassembly, artifact.fixture)


def _linear(role_spec, calls):
  packed = Tensor.empty(
    role_spec.n * role_spec.epochs * scheduler.Q4_WORDS_PER_EPOCH_ROW,
    dtype=dtypes.uint32, device="AMD")
  def packed_weight():
    calls.append("packed-weight")
    return packed
  return SimpleNamespace(
    bias=None, out_features=role_spec.n, in_features=role_spec.k,
    q4k_storage=object(), prefill_packed_weight=packed_weight)


def _producer(calls):
  def produce(source, spec):
    calls.append((source, spec))
    return Q4KQ8ActivationTile(
      Tensor.empty(spec.values_shape, dtype=dtypes.int8, device="AMD"),
      Tensor.empty(spec.metadata_shape, dtype=dtypes.float32, device="AMD"),
      Tensor.empty(spec.metadata_shape, dtype=dtypes.float32, device="AMD"),
    )
  return produce


def _build(*, artifact=None):
  role_spec = exact_role_spec("ffn_gate_up")
  weight_calls, producer_calls = [], []
  artifact = _scheduler_artifact(role_spec) if artifact is None else artifact
  result = scheduler.build_frozen_exact_q4k_schedule(
    _linear(role_spec, weight_calls),
    Tensor.empty(role_spec.m, role_spec.k, dtype=dtypes.float16, device="AMD"),
    role_spec=role_spec, frozen_bundle="/frozen/exact.tar", enabled=True,
    artifact_loader=lambda _path: artifact, activation_producer=_producer(producer_calls))
  return role_spec, result, weight_calls, producer_calls


def test_scheduler_builds_exact_lazy_program_chain_without_runtime_or_device_access():
  with patch("tinygrad.engine.realize.get_runtime") as get_runtime, Context(ALLOW_DEVICE_USAGE=0):
    role_spec, result, weight_calls, producer_calls = _build()
  get_runtime.assert_not_called()
  assert result is not None and result.output.shape == (1, role_spec.m, role_spec.n)
  assert weight_calls == ["packed-weight"] and len(producer_calls) == role_spec.epochs == 20

  calls = [u for u in result.output.uop.toposort()
           if u.op is Ops.CALL and u.src[0].op is Ops.PROGRAM]
  assert len(calls) == role_spec.epochs
  # The accumulator zero must be an ordinary scheduler constant. A shaped
  # unique CONST with UNIQUE/DEVICE sources survives graph inspection but
  # fails tensor-function verification during real lowering.
  assert all(not (u.op is Ops.CONST and len(u.src) != 0) for u in result.output.uop.toposort())
  assert {call.src[0] for call in calls} == {result.binding.artifact.program}
  assert all(len(get_call_arg_uops(call)) == 5 for call in calls)
  assert all(tuple(call.src[0].arg.outs) == (0,) and tuple(call.src[0].arg.ins) == tuple(range(5))
             for call in calls)
  # Every new accumulator argument is rooted after the previous PROGRAM call.
  for previous, current in zip(calls, calls[1:]):
    assert previous in get_call_arg_uops(current)[0].toposort()

  execution = result.evidence["execution"]
  assert execution["dispatch"] == {
    "mode": "lazy_ops_program_chain", "count": role_spec.epochs,
    "program_key": result.binding.program_key, "eager_native_runtime": False,
    "scheduler_visible": True, "tinyjit_replay_visible": True,
  }
  assert execution["runtime"]["created_during_graph_build"] is False
  assert execution["runtime"]["gpu_dispatch_during_graph_build"] is False
  assert execution["operands"]["host_fixed_va_staging"] is False


@pytest.mark.parametrize("outs,ins", [
  ((), (0, 1, 2, 3, 4)),
  ((0,), (1, 2, 3, 4)),
  ((1,), (0, 1, 2, 3, 4)),
])
def test_scheduler_rejects_frozen_program_without_exact_accumulator_effects_before_tensor_work(outs, ins):
  role_spec = exact_role_spec("ffn_gate_up")
  artifact = _scheduler_artifact(role_spec, outs=outs, ins=ins)
  weight_calls = []
  with pytest.raises(ValueError, match="side-effect contract"):
    scheduler.build_frozen_exact_q4k_schedule(
      _linear(role_spec, weight_calls),
      Tensor.empty(role_spec.m, role_spec.k, dtype=dtypes.float16, device="AMD"),
      role_spec=role_spec, frozen_bundle="/frozen/exact.tar", enabled=True,
      artifact_loader=lambda _path: artifact, activation_producer=_producer([]))
  assert weight_calls == []


def test_scheduler_is_default_off_before_artifact_or_weight_access():
  role_spec = exact_role_spec("ffn_gate_up")
  lin = SimpleNamespace(
    prefill_packed_weight=lambda: (_ for _ in ()).throw(AssertionError("weight touched")))
  assert scheduler.build_frozen_exact_q4k_schedule(
    lin, object(), role_spec=role_spec, frozen_bundle="/missing", enabled=False,
    artifact_loader=lambda _path: (_ for _ in ()).throw(AssertionError("artifact loaded"))) is None


def test_scheduler_reuses_prevalidated_authority_binding_without_deserializing_bundle_again():
  role_spec, first, _, _ = _build()
  weight_calls, producer_calls = [], []
  second = scheduler.build_frozen_exact_q4k_schedule(
    _linear(role_spec, weight_calls),
    Tensor.empty(role_spec.m, role_spec.k, dtype=dtypes.float16, device="AMD"),
    role_spec=role_spec, frozen_bundle="/frozen/exact.tar", enabled=True,
    binding=first.binding,
    artifact_loader=lambda _path: (_ for _ in ()).throw(AssertionError("artifact reloaded")),
    activation_producer=_producer(producer_calls))
  assert second.binding is first.binding
  assert weight_calls == ["packed-weight"] and len(producer_calls) == role_spec.epochs


def test_scheduler_chain_can_be_callified_inside_function_without_device_escape():
  role_spec, first, _, _ = _build()
  weight_calls, producer_calls = [], []
  lin = _linear(role_spec, weight_calls)

  @function(allow_implicit=True)
  def routed(activation):
    return scheduler.build_frozen_exact_q4k_schedule(
      lin, activation, role_spec=role_spec, frozen_bundle="/frozen/exact.tar", enabled=True,
      binding=first.binding,
      artifact_loader=lambda _path: (_ for _ in ()).throw(AssertionError("artifact reloaded")),
      activation_producer=_producer(producer_calls)).output

  with patch("tinygrad.engine.realize.get_runtime") as get_runtime:
    output = routed(Tensor.empty(role_spec.m, role_spec.k, dtype=dtypes.float16, device="AMD"))
  get_runtime.assert_not_called()
  assert output.shape == (1, role_spec.m, role_spec.n)
  assert weight_calls == ["packed-weight"] and len(producer_calls) == role_spec.epochs
  assert len([u for u in output.uop.toposort()
              if u.op is Ops.CALL and u.src[0].op is Ops.PROGRAM]) == role_spec.epochs


def test_scheduler_evidence_rejects_dispatch_count_or_runtime_drift():
  role_spec, result, _, _ = _build()
  execution = result.evidence["execution"]
  with pytest.raises(ValueError, match="PROGRAM chain"):
    scheduler.validate_frozen_schedule_evidence({
      **execution, "dispatch": {**execution["dispatch"], "count": role_spec.epochs - 1},
    }, role_spec)
  with pytest.raises(ValueError, match="runtime boundary"):
    scheduler.validate_frozen_schedule_evidence({
      **execution, "runtime": {**execution["runtime"], "created_during_graph_build": True},
    }, role_spec)
