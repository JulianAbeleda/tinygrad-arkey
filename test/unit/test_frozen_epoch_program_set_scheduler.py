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
from extra.qk.mmq_frozen_epoch_program_set import (
  BINDING_SCHEMA, LEGACY_SCHEMA, FrozenEpochProgramSetArtifact, FrozenEpochProgramSetBinding,
  SCHEMA as ARTIFACT_SCHEMA,
)
from extra.qk.prefill import frozen_epoch_program_set_scheduler as scheduler
from extra.qk.q4k_q8_activation_producer import Q4KQ8ActivationTile
from test.unit.test_mmq_frozen_epoch_program_set import _family


def _binding(role_spec, *, legacy: bool = False):
  family = _family(role_spec)
  programs = tuple(variant.program.replace(
    arg=replace(variant.program.arg, outs=(0,), ins=tuple(range(5))))
    for variant in family.variants)
  keys = tuple(program.key.hex() for program in programs)
  manifest = {
    "schema": LEGACY_SCHEMA if legacy else ARTIFACT_SCHEMA,
    "c1_certification": ({
      "gate": "C1", "certified": False, "status": "legacy_v2_missing_generation_provenance",
      "content_addressed": False,
    } if legacy else {
      "gate": "C1", "certified": True, "content_addressed": True,
    }),
    "role": {"name": role_spec.role, "shape": list(role_spec.shape),
             "candidate_identity": role_spec.candidate_canonical_identity},
    "variants": [{"epoch": epoch, "program_key": key} for epoch, key in enumerate(keys)],
    "family_identity": "family:" + "a" * 64,
  }
  artifact = FrozenEpochProgramSetArtifact(
    manifest, programs,
    tuple(f"binary-{epoch}".encode() for epoch in range(role_spec.epochs)),
    tuple(f"source-{epoch}" for epoch in range(role_spec.epochs)),
  )
  return FrozenEpochProgramSetBinding(
    BINDING_SCHEMA, role_spec, artifact, role_spec.candidate_canonical_identity,
    manifest["family_identity"], keys)


def _linear(role_spec, calls):
  packed = Tensor.empty(
    role_spec.n, role_spec.epochs, 36, dtype=dtypes.uint32, device="AMD")
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


def _build(prefix_epochs):
  role_spec = exact_role_spec("ffn_gate_up")
  weight_calls, producer_calls = [], []
  binding = _binding(role_spec)
  result = scheduler.build_frozen_epoch_program_set_schedule(
    _linear(role_spec, weight_calls),
    Tensor.empty(role_spec.m, role_spec.k, dtype=dtypes.float16, device="AMD"),
    role_spec=role_spec, frozen_bundle="/frozen/v2.tar", enabled=True,
    prefix_epochs=prefix_epochs, binding=binding,
    binding_loader=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("binding reloaded")),
    activation_producer=_producer(producer_calls))
  return role_spec, binding, result, weight_calls, producer_calls


@pytest.mark.parametrize("prefix_epochs", [1, 2, 20])
def test_v2_scheduler_prefix_uses_one_producer_and_same_five_buffer_identities(prefix_epochs):
  with patch("tinygrad.engine.realize.get_runtime") as get_runtime, Context(ALLOW_DEVICE_USAGE=0):
    role_spec, binding, result, weight_calls, producer_calls = _build(prefix_epochs)
  get_runtime.assert_not_called()
  assert result is not None and result.output.shape == (1, role_spec.m, role_spec.n)
  assert weight_calls == ["packed-weight"] and len(producer_calls) == 1
  assert producer_calls[0][0].shape == (role_spec.m, role_spec.k)
  assert (producer_calls[0][1].m, producer_calls[0][1].k) == (role_spec.m, role_spec.k)

  calls = [node for node in result.output.uop.toposort()
           if node.op is Ops.CALL and node.src[0].op is Ops.PROGRAM]
  assert len(calls) == prefix_epochs
  assert [call.src[0] for call in calls] == list(binding.artifact.programs[:prefix_epochs])
  arguments = [get_call_arg_uops(call) for call in calls]
  assert all(len(row) == 5 for row in arguments)
  assert all(arguments[0][slot].buf_uop is row[slot].buf_uop
             for row in arguments for slot in range(5))
  assert len(result.preparation_outputs) == 5
  assert all(result.preparation_outputs[slot].uop.buf_uop is arguments[0][slot].buf_uop
             for slot in range(5))
  assert all(previous in current[0].toposort()
             for previous, current in zip(calls, arguments[1:]))

  execution = result.evidence["execution"]
  assert execution["operands"]["fixed_base_slots"] == list(range(5))
  assert execution["operands"]["all_calls_share_buffer_identity"] is True
  assert execution["operands"]["full_role_ds4_producer_calls"] == 1
  assert execution["operands"]["host_staging"] is False
  assert execution["dispatch"]["selected_epochs"] == list(range(prefix_epochs))
  assert execution["dispatch"]["program_keys"] == list(binding.program_keys[:prefix_epochs])
  assert result.evidence["prefix_complete"] is True
  assert result.evidence["complete_role"] is (prefix_epochs == role_spec.epochs)


def test_v2_scheduler_is_default_off_before_binding_or_tensor_access():
  role_spec = exact_role_spec("ffn_gate_up")
  lin = SimpleNamespace(
    prefill_packed_weight=lambda: (_ for _ in ()).throw(AssertionError("weight touched")))
  assert scheduler.build_frozen_epoch_program_set_schedule(
    lin, object(), role_spec=role_spec, frozen_bundle="/missing", enabled=False,
    binding_loader=lambda *_args, **_kwargs:
      (_ for _ in ()).throw(AssertionError("binding loaded"))) is None


def test_scheduler_default_accepts_loader_validated_legacy_v2_but_strict_rejects_before_weight_touch():
  role_spec, weight_calls, producer_calls = exact_role_spec("ffn_gate_up"), [], []
  legacy = _binding(role_spec, legacy=True)
  loader_calls = []
  def load_binding(requested, bundle, **kwargs):
    loader_calls.append((requested, bundle, kwargs))
    return legacy
  with Context(ALLOW_DEVICE_USAGE=0):
    result = scheduler.build_frozen_epoch_program_set_schedule(
      _linear(role_spec, weight_calls),
      Tensor.empty(role_spec.m, role_spec.k, dtype=dtypes.float16, device="AMD"),
      role_spec=role_spec, frozen_bundle="/frozen/legacy-v2.tar", enabled=True,
      prefix_epochs=1, binding_loader=load_binding, activation_producer=_producer(producer_calls))
  assert result is not None
  assert loader_calls == [(role_spec, "/frozen/legacy-v2.tar", {
    "inventory": scheduler.DEFAULT_INVENTORY})]
  assert result.binding.artifact.manifest["c1_certification"]["certified"] is False
  assert weight_calls == ["packed-weight"] and len(producer_calls) == 1

  weight_calls.clear()
  with pytest.raises(ValueError, match="complete exact epoch family"):
    scheduler.build_frozen_epoch_program_set_schedule(
      _linear(role_spec, weight_calls),
      Tensor.empty(role_spec.m, role_spec.k, dtype=dtypes.float16, device="AMD"),
      role_spec=role_spec, frozen_bundle="/frozen/legacy-v2.tar", enabled=True,
      prefix_epochs=1, require_c1=True, binding_loader=load_binding, activation_producer=_producer([]))
  assert loader_calls[-1] == (role_spec, "/frozen/legacy-v2.tar", {
    "inventory": scheduler.DEFAULT_INVENTORY, "require_c1": True})
  assert weight_calls == []


def test_v2_scheduler_rejects_forged_role_before_binding_or_weight_touch():
  role_spec, weight_calls = exact_role_spec("ffn_gate_up"), []
  forged = replace(role_spec, candidate_canonical_identity="0" * 64)
  with pytest.raises(ValueError, match="canonical admitted"):
    scheduler.build_frozen_epoch_program_set_schedule(
      _linear(role_spec, weight_calls), object(),
      role_spec=forged, frozen_bundle="/frozen/v2.tar", enabled=True,
      binding_loader=lambda *_args, **_kwargs:
        (_ for _ in ()).throw(AssertionError("binding loaded")))
  assert weight_calls == []


def test_v2_scheduler_rejects_noncontiguous_prefix_and_effect_drift_before_weight_touch():
  role_spec, weight_calls = exact_role_spec("ffn_gate_up"), []
  with pytest.raises(ValueError, match="prefix_epochs"):
    scheduler.build_frozen_epoch_program_set_schedule(
      _linear(role_spec, weight_calls),
      Tensor.empty(role_spec.m, role_spec.k, dtype=dtypes.float16, device="AMD"),
      role_spec=role_spec, frozen_bundle="/frozen/v2.tar", enabled=True,
      prefix_epochs=0, binding=_binding(role_spec), activation_producer=_producer([]))
  assert weight_calls == []

  binding = _binding(role_spec)
  bad_program = binding.artifact.programs[3].replace(
    arg=replace(binding.artifact.programs[3].arg, outs=(), ins=tuple(range(5))))
  artifact = replace(
    binding.artifact,
    programs=binding.artifact.programs[:3] + (bad_program,) + binding.artifact.programs[4:])
  keys = binding.program_keys[:3] + (bad_program.key.hex(),) + binding.program_keys[4:]
  bad_binding = replace(binding, artifact=artifact, program_keys=keys)
  with pytest.raises(ValueError, match="accumulator effects"):
    scheduler.build_frozen_epoch_program_set_schedule(
      _linear(role_spec, weight_calls),
      Tensor.empty(role_spec.m, role_spec.k, dtype=dtypes.float16, device="AMD"),
      role_spec=role_spec, frozen_bundle="/frozen/v2.tar", enabled=True,
      binding=bad_binding, activation_producer=_producer([]))
  assert weight_calls == []


def test_v2_scheduler_route_function_callifies_without_device_or_runtime_access():
  role_spec, binding = exact_role_spec("ffn_gate_up"), None
  binding = _binding(role_spec)
  weight_calls, producer_calls = [], []
  lin = _linear(role_spec, weight_calls)

  @function(allow_implicit=True)
  def routed(activation):
    return scheduler.build_frozen_epoch_program_set_schedule(
      lin, activation, role_spec=role_spec, frozen_bundle="/frozen/v2.tar", enabled=True,
      binding=binding, prefix_epochs=2,
      binding_loader=lambda *_args, **_kwargs:
        (_ for _ in ()).throw(AssertionError("binding reloaded")),
      activation_producer=_producer(producer_calls)).output

  with patch("tinygrad.engine.realize.get_runtime") as get_runtime, Context(ALLOW_DEVICE_USAGE=0):
    output = routed(Tensor.empty(role_spec.m, role_spec.k, dtype=dtypes.float16, device="AMD"))
  get_runtime.assert_not_called()
  assert output.shape == (1, role_spec.m, role_spec.n)
  assert weight_calls == ["packed-weight"] and len(producer_calls) == 1
  calls = [node for node in output.uop.toposort()
           if node.op is Ops.CALL and node.src[0].op is Ops.PROGRAM]
  assert len(calls) == 2
  arguments = [get_call_arg_uops(call) for call in calls]
  assert all(arguments[0][slot].buf_uop is arguments[1][slot].buf_uop for slot in range(5))
  assert calls[0] in arguments[1][0].toposort()
