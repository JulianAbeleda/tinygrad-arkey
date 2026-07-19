from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from extra.qk.mmq_attn_qo_c8_runtime import DirectPackedObjects
from extra.qk.mmq_ffn_gate_up_c8_runtime import (
  COMPOSITION_SCHEMA, FIXTURE_SCHEMA, FP16_INPUT_SEMANTICS,
  OUTER_WALL_WRAPPER, OUTPUT_REALIZATION_SEMANTICS,
  FfnGateUpCandidateInputs, FfnGateUpNoReadbackOutputRealizer,
  FfnGateUpRouteCallback, FfnGateUpV2Fixture,
  build_ffn_gate_up_direct_packed_objects,
  compose_ffn_gate_up_queue_runners, load_ffn_gate_up_c8_runtime_config,
  rebuild_ffn_gate_up_v2_fixture, resident_fp16_roundtrip,
)
from extra.qk.mmq_ffn_gate_up_matched_timing_contract import (
  build_ffn_gate_up_matched_complete_role_timing_contract,
)


def _sha(index: int) -> str:
  return "sha256:" + f"{index:064x}"


def _hex(index: int) -> str:
  return f"{index:064x}"


def _content_identity(value) -> str:
  encoded = json.dumps(
    value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
  return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _seal_composition(value):
  payload = {key: item for key, item in value.items()
             if key != "composition_identity"}
  value["composition_identity"] = _content_identity(payload)
  return value


def _role():
  return SimpleNamespace(
    role="ffn_gate_up", m=512, n=17408, k=5120,
    shape=(512, 17408, 5120), epochs=20)


def _family():
  binding = SimpleNamespace(
    role_spec=_role(), program_key=_hex(10), binary_sha256=_hex(11))
  return SimpleNamespace(binding=binding, family_identity=_sha(12))


def _fixture(role) -> FfnGateUpV2Fixture:
  resident = np.zeros((512, 5120), dtype=np.float16)
  return FfnGateUpV2Fixture(
    role_spec=role, execution_fixture={"schema": FIXTURE_SCHEMA},
    words=np.zeros(1, dtype=np.uint32),
    resident_fp16_activation=resident,
    roundtrip_fp32=resident.astype(np.float32),
    q8_values=np.zeros(1, dtype=np.int8),
    q8_scales=np.zeros(1, dtype=np.float32),
    q8_sums=np.zeros(1, dtype=np.float32),
    q4_epoch_major=np.zeros(1, dtype=np.uint32),
    fixture_identity=_sha(20), workload_identity=_sha(21),
    input_identity=_sha(22), logical_q4_identity=_sha(23),
    resident_fp16_activation_identity=_sha(24))


def _composition(family, fixture):
  candidate = {
    "family_identity": family.family_identity,
    "candidate_executable_identity": _sha(30),
    "program_key": family.binding.program_key,
    "binary_sha256": family.binding.binary_sha256,
  }
  direct = {
    queue: {
      "qualification_identity": _sha(40 + index),
      "executable_identity": _sha(50 + index),
      "binary_sha256": _hex(60 + index),
    } for index, queue in enumerate(("PM4", "AQL"))
  }
  c6 = {
    queue: {
      "status": "PASS", "fixture_schema": FIXTURE_SCHEMA,
      "input_semantics": FP16_INPUT_SEMANTICS,
      "legacy_fp32_prequantized": False,
      "family_identity": family.family_identity,
      "workload_identity": fixture.workload_identity,
      "input_identity": fixture.input_identity,
      "device_identity": f"device-{queue}",
      "software_identity": "software-clean",
      "evidence_identity": _sha(70 + index),
      "candidate_correctness_identity": _sha(80 + index),
      "comparator_identity": _sha(90 + index),
    } for index, queue in enumerate(("PM4", "AQL"))
  }
  transitions = {}
  ordinal = 100
  for queue in ("PM4", "AQL"):
    transitions[queue] = {}
    for field in (
        "candidate_candidate", "direct_direct", "direct_candidate_prefix1",
        "direct_candidate_full_role", "candidate_direct_candidate"):
      transitions[queue][field] = _sha(ordinal)
      ordinal += 1
  return {
    "schema": COMPOSITION_SCHEMA, "status": "READY",
    "role": "ffn_gate_up", "family_identity": family.family_identity,
    "execution_fixture_identity": fixture.fixture_identity,
    "workload_identity": fixture.workload_identity,
    "input_identity": fixture.input_identity,
    "logical_q4_identity": fixture.logical_q4_identity,
    "resident_fp16_activation_identity":
      fixture.resident_fp16_activation_identity,
    "candidate_binding": candidate,
    "direct_bindings_by_queue": direct,
    "c6_by_queue": c6,
    "joint_session_c7_identity": _sha(120),
    "transition_preflight_bindings_by_queue": transitions,
    "runtime_canary_by_queue": {
      queue: {
        "schema":
          "tinygrad.mmq_q4k_q8_1.frozen_staged_runtime_canary.v1",
        "status": "PASS", "queue_mode": queue,
        "family_identity": family.family_identity,
        "all_checks_pass": True,
        "program_key": family.binding.program_key,
        "binary_sha256": family.binding.binary_sha256,
        "compile_performed": False, "requires_recompile": False,
        "amd_aql_effective": queue == "AQL", "exact_blocker": None,
      } for queue in ("PM4", "AQL")
    },
    "matched_timing_contract_identity": "",
    "promotion_eligible_on_candidate_win": False,
    "composition_identity": "",
  }


def _contract_kwargs(composition):
  return {
    "workload_identity": composition["workload_identity"],
    "input_identity": composition["input_identity"],
    "logical_q4_identity": composition["logical_q4_identity"],
    "resident_fp16_activation_identity":
      composition["resident_fp16_activation_identity"],
    "candidate_binding": composition["candidate_binding"],
    "direct_bindings_by_queue": composition["direct_bindings_by_queue"],
    "joint_session_c7_identity": composition["joint_session_c7_identity"],
    "c6_bindings_by_queue": {
      queue: {
        field: composition["c6_by_queue"][queue][field]
        for field in (
          "evidence_identity", "candidate_correctness_identity",
          "comparator_identity", "workload_identity", "input_identity")
      } for queue in ("PM4", "AQL")
    },
    "transition_preflight_bindings_by_queue":
      composition["transition_preflight_bindings_by_queue"],
  }


def _write_json(path: Path, value) -> Path:
  path.write_text(json.dumps(value))
  return path


def _case(tmp_path: Path):
  family, fixture = _family(), None
  fixture = _fixture(family.binding.role_spec)
  composition = _composition(family, fixture)
  contract = build_ffn_gate_up_matched_complete_role_timing_contract(
    **_contract_kwargs(composition))
  composition["matched_timing_contract_identity"] = contract["evidence_identity"]
  _seal_composition(composition)
  paths = {
    "composition": _write_json(tmp_path / "composition.json", composition),
    "execution_fixture_v2": _write_json(
      tmp_path / "fixture-v2.json", {"schema": FIXTURE_SCHEMA}),
    "matched_timing_contract": _write_json(tmp_path / "contract.json", contract),
  }
  for queue, key in (("PM4", "qualification_pm4"),
                     ("AQL", "qualification_aql")):
    binding = composition["direct_bindings_by_queue"][queue]
    paths[key] = _write_json(tmp_path / f"{queue}.json", {
      "status": "PASS", "queue_mode": queue,
      "qualification_identity": binding["qualification_identity"],
      "fallback_evidence": {
        "executable_identity": binding["executable_identity"],
        "binary_sha256": binding["binary_sha256"],
        "workload_identity": fixture.workload_identity,
        "input_identity": fixture.input_identity,
      },
    })
  paths["frozen_bundle"] = tmp_path / "bundle"
  paths["frozen_bundle"].mkdir()
  paths["staged_family_manifest"] = _write_json(
    tmp_path / "family.json", {"unused": True})
  config = {key: str(paths[key]) for key in (
    "composition", "execution_fixture_v2", "matched_timing_contract",
    "frozen_bundle", "staged_family_manifest",
    "qualification_pm4", "qualification_aql")}
  return family, fixture, composition, contract, config


def _load(tmp_path: Path):
  family, fixture, composition, contract, config = _case(tmp_path)
  loaded = load_ffn_gate_up_c8_runtime_config(
    config, family=family,
    fixture_rebuilder=lambda role, raw: fixture)
  return loaded, fixture, composition, contract, config


def test_resident_fp16_roundtrip_is_the_candidate_source():
  source = np.array([[1.0003, -0.3333, 65500.0]], dtype=np.float32)
  resident, roundtrip = resident_fp16_roundtrip(source)
  assert resident.dtype == np.float16
  assert roundtrip.dtype == np.float32
  np.testing.assert_array_equal(roundtrip, resident.astype(np.float32))
  assert not np.array_equal(roundtrip, source)


def test_legacy_bundle_fixture_is_not_accepted_as_v2_execution_evidence():
  calls = []
  legacy = {
    "schema": "tinygrad.mmq_q4k_q8_1_target_fixture.v1",
    "role": "ffn_gate_up", "shape": [512, 17408, 5120],
    "total_epochs": 20,
    "seeds": {"q4": 20260721, "q8_source": 20260722},
    "source_sha256": _hex(1), "repack": {},
  }
  with pytest.raises(ValueError, match="legacy FP32-prequantized"):
    rebuild_ffn_gate_up_v2_fixture(
      _role(), legacy,
      quantizer=lambda source: calls.append(source))
  assert calls == []


def test_runtime_config_cross_binds_v2_c4_c6_c7_c8_without_gpu(tmp_path: Path):
  loaded, fixture, composition, contract, _ = _load(tmp_path)
  assert loaded.fixture is fixture
  assert loaded.matched_timing_contract == contract
  assert loaded.contract_validation_kwargs["input_identity"] == \
    composition["input_identity"]
  assert set(loaded.qualification_paths_by_queue) == {"PM4", "AQL"}
  assert loaded.qualification_paths_by_queue["PM4"] != \
    loaded.qualification_paths_by_queue["AQL"]


def test_runtime_config_loads_family_from_injected_paths(tmp_path: Path):
  family, fixture, _, _, config = _case(tmp_path)
  calls = []

  def family_loader(path, *, role_spec, frozen_bundle):
    calls.append((Path(path), role_spec, Path(frozen_bundle)))
    return family

  loaded = load_ffn_gate_up_c8_runtime_config(
    config, family_loader=family_loader,
    fixture_rebuilder=lambda role, raw: fixture)
  assert loaded.family is family
  assert calls[0][0] == Path(config["staged_family_manifest"]).resolve()
  assert calls[0][1].role == "ffn_gate_up"
  assert calls[0][2] == Path(config["frozen_bundle"]).resolve()


def test_direct_objects_receive_exact_resident_fp16_bytes_via_injected_builder(
    tmp_path: Path,
    ):
  loaded, fixture, *_ = _load(tmp_path)
  calls = []

  def builder(role, words, activation, activation_dtype):
    calls.append((role, words, activation.copy(), activation_dtype))
    return DirectPackedObjects("linear", "activation-object", "route-spec")

  objects = build_ffn_gate_up_direct_packed_objects(
    loaded, object_builder=builder)
  assert objects == DirectPackedObjects(
    "linear", "activation-object", "route-spec")
  assert calls[0][3] == "float16"
  assert calls[0][2].shape == (1, 512, 5120)
  assert calls[0][2].dtype == np.float16
  np.testing.assert_array_equal(
    calls[0][2][0], fixture.resident_fp16_activation)


def test_outer_wall_route_builders_receive_exact_v2_inputs(tmp_path: Path):
  loaded, fixture, *_ = _load(tmp_path)
  calls = {}
  shared_activation = object()

  def object_builder(role, words, activation, activation_dtype):
    calls["objects"] = (role, activation, activation_dtype)
    return DirectPackedObjects("linear", shared_activation, "spec")

  def candidate_route_builder(**kwargs):
    calls["candidate"] = kwargs
    return FfnGateUpRouteCallback(
      route_id="staged_candidate",
      queue_mode="PM4",
      input_identity=loaded.fixture.input_identity,
      executable_identity=loaded.composition["candidate_binding"][
        "candidate_executable_identity"],
      invoke=lambda: {"candidate": True},
      realize_output=FfnGateUpNoReadbackOutputRealizer(
        callback=lambda output: None,
        semantics=OUTPUT_REALIZATION_SEMANTICS,
        readback_performed=False),
      attest_post_sync=lambda output, queue: {},
      outer_wall_wrapper=OUTER_WALL_WRAPPER,
      emits_timing_receipt=False)

  def direct_route_builder(**kwargs):
    calls["direct"] = kwargs
    return FfnGateUpRouteCallback(
      route_id="direct_packed",
      queue_mode="PM4",
      input_identity=loaded.fixture.input_identity,
      executable_identity=loaded.composition["direct_bindings_by_queue"][
        "PM4"]["executable_identity"],
      invoke=lambda: {"direct": True},
      realize_output=FfnGateUpNoReadbackOutputRealizer(
        callback=lambda output: None,
        semantics=OUTPUT_REALIZATION_SEMANTICS,
        readback_performed=False),
      attest_post_sync=lambda output, queue: {},
      outer_wall_wrapper=OUTER_WALL_WRAPPER,
      emits_timing_receipt=False)

  routes = compose_ffn_gate_up_queue_runners(
    loaded, queue_mode="PM4", clock_identity="clock-policy-0",
    object_builder=object_builder,
    candidate_route_builder=candidate_route_builder,
    direct_route_builder=direct_route_builder)
  assert callable(routes.candidate.invoke) and \
    callable(routes.direct_packed.invoke)
  assert routes.candidate.realize_output(object()) is None
  assert routes.direct_packed.realize_output(object()) is None
  assert calls["objects"][2] == "float16"
  np.testing.assert_array_equal(
    calls["objects"][1][0], fixture.resident_fp16_activation)
  candidate_inputs = calls["candidate"]["candidate_inputs"]
  assert isinstance(candidate_inputs, FfnGateUpCandidateInputs)
  assert candidate_inputs.fixture_identity == fixture.fixture_identity
  assert candidate_inputs.input_identity == fixture.input_identity
  assert candidate_inputs.logical_q4_identity == fixture.logical_q4_identity
  assert candidate_inputs.resident_fp16_activation_identity == \
    fixture.resident_fp16_activation_identity
  assert candidate_inputs.resident_fp16_activation is shared_activation
  assert candidate_inputs.q8_producer_semantics == \
    "per_invocation_from_resident_fp16_inside_outer_synchronized_wall"
  assert set(candidate_inputs.q8_reference_sha256) == \
    {"values", "scales", "sums"}
  assert not hasattr(candidate_inputs, "q8_values")
  assert not hasattr(candidate_inputs, "roundtrip_fp32")
  assert calls["candidate"]["matched_timing_contract"] == \
    loaded.matched_timing_contract
  assert calls["candidate"]["frozen_bundle"] == loaded.frozen_bundle
  assert calls["direct"]["direct_objects"].linear == "linear"
  assert calls["direct"]["direct_objects"].activation is shared_activation
  assert calls["candidate"]["candidate_inputs"].resident_fp16_activation is \
    calls["direct"]["direct_objects"].activation
  bindings = calls["direct"]["bindings_by_queue"]
  assert bindings["PM4"].input_identity == loaded.fixture.input_identity
  assert bindings["AQL"].input_identity == loaded.fixture.input_identity


@pytest.mark.parametrize("missing", ["candidate", "direct"])
def test_outer_wall_route_builder_omission_blocks_before_object_or_runtime_calls(
    tmp_path: Path, missing: str,
    ):
  loaded, *_ = _load(tmp_path)
  calls = []

  def object_builder(*args):
    calls.append(("objects", args))
    raise AssertionError("GPU object builder must not be reached")

  def route_builder(**kwargs):
    calls.append(("route", kwargs))
    raise AssertionError("route builder must not be reached")

  with pytest.raises(ValueError, match="explicit production-faithful"):
    compose_ffn_gate_up_queue_runners(
      loaded, queue_mode="PM4", clock_identity="clock-policy-0",
      object_builder=object_builder,
      candidate_route_builder=None if missing == "candidate" else route_builder,
      direct_route_builder=None if missing == "direct" else route_builder)
  assert calls == []


def test_legacy_callable_candidate_runner_is_rejected_after_shared_objects_only(
    tmp_path: Path,
    ):
  loaded, *_ = _load(tmp_path)
  calls = []

  def object_builder(*args):
    calls.append(("objects", args))
    return DirectPackedObjects("linear", object(), "spec")

  def direct_builder(**kwargs):
    calls.append(("direct", kwargs))
    raise AssertionError("direct route builder must not be reached")

  with pytest.raises(TypeError, match="FfnGateUpRouteCallback"):
    compose_ffn_gate_up_queue_runners(
      loaded, queue_mode="PM4", clock_identity="clock-policy-0",
      object_builder=object_builder,
      candidate_route_builder=lambda **kwargs: (
        lambda: {"legacy_timing_receipt": True}),
      direct_route_builder=direct_builder)
  assert [row[0] for row in calls] == ["objects"]


def test_typed_output_realizer_rejects_returned_data_and_missing_contract():
  realizer = FfnGateUpNoReadbackOutputRealizer(
    callback=lambda output: np.array([1], dtype=np.int32),
    semantics=OUTPUT_REALIZATION_SEMANTICS,
    readback_performed=False)
  with pytest.raises(ValueError, match="must return None"):
    realizer(object())
  with pytest.raises(ValueError, match="no-readback"):
    FfnGateUpNoReadbackOutputRealizer(
      callback=lambda output: None,
      semantics=OUTPUT_REALIZATION_SEMANTICS,
      readback_performed=True).validate()
  with pytest.raises(TypeError):
    FfnGateUpRouteCallback(
      route_id="staged_candidate", input_identity=_sha(1),
      executable_identity=_sha(2), invoke=lambda: None)


@pytest.mark.parametrize("field,value", [
  ("fixture_schema", "tinygrad.mmq_q4k_q8_1_target_fixture.v1"),
  ("input_semantics", "prequantized_fp32_source"),
  ("legacy_fp32_prequantized", True),
])
def test_runtime_config_rejects_legacy_c6_before_gpu_builders(
    tmp_path: Path, field, value,
    ):
  family, fixture, composition, contract, config = _case(tmp_path)
  composition["c6_by_queue"]["PM4"][field] = value
  _seal_composition(composition)
  _write_json(Path(config["composition"]), composition)
  with pytest.raises(ValueError, match="legacy FP32-prequantized"):
    load_ffn_gate_up_c8_runtime_config(
      config, family=family,
      fixture_rebuilder=lambda role, raw: fixture)


def test_contract_and_qualification_identity_drift_fail_closed(tmp_path: Path):
  family, fixture, composition, contract, config = _case(tmp_path)
  bad_contract = copy.deepcopy(contract)
  bad_contract["common_inputs"]["identity"] = _sha(999)
  _write_json(Path(config["matched_timing_contract"]), bad_contract)
  with pytest.raises(ValueError, match="legacy, missing, or mismatched"):
    load_ffn_gate_up_c8_runtime_config(
      config, family=family,
      fixture_rebuilder=lambda role, raw: fixture)

  _write_json(Path(config["matched_timing_contract"]), contract)
  qualification = json.loads(Path(config["qualification_pm4"]).read_text())
  qualification["qualification_identity"] = _sha(998)
  _write_json(Path(config["qualification_pm4"]), qualification)
  with pytest.raises(ValueError, match="qualification path binding differs"):
    load_ffn_gate_up_c8_runtime_config(
      config, family=family,
      fixture_rebuilder=lambda role, raw: fixture)


def test_import_does_not_open_a_tinygrad_device():
  script = """
from tinygrad.device import Device
assert not Device._opened_devices
import extra.qk.mmq_ffn_gate_up_c8_runtime
assert not Device._opened_devices
"""
  subprocess.run(
    [sys.executable, "-c", script], check=True, cwd=Path(__file__).parents[2])
