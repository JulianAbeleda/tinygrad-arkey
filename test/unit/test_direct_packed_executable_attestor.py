from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from tinygrad import Tensor

from extra.qk.direct_packed_executable_attestor import (
  ARTIFACT_SCHEMA, EVIDENCE_SCHEMA, OBSERVATION_SCHEMA,
  DirectPackedAttestationBindings, DirectPackedAttestationPrimitives,
  DirectPackedLinearExecutionCapture, FrozenDirectPackedExecutableAttestor,
  build_direct_packed_fallback_evidence,
  build_direct_packed_qualification_artifact,
  load_direct_packed_qualification,
  make_production_direct_packed_attested_runner,
  persist_direct_packed_qualification,
  production_attestation_primitives,
  qualify_and_freeze_production_direct_packed,
  validate_direct_packed_fallback_evidence,
)
from extra.qk.mmq_staged_c7_c8_contract import _fallback_evidence


QUEUES = ("PM4", "AQL")


def _bindings(queue: str, *, prefix: str = "E_") -> DirectPackedAttestationBindings:
  return DirectPackedAttestationBindings(
    queue_mode=queue, workload_identity="attn_qo-m512-n5120-k5120",
    input_identity="sha256:" + "2" * 64,
    device_identity="cpu-test-device", software_identity="tinygrad-test-revision",
    comparator_identity="sha256:" + ("9" if queue == "PM4" else "a") * 64,
    clock_identity="perf-counter-ns-policy-0", required_program_prefix=prefix,
  )


def _bindings_by_queue(*, prefix: str = "E_"):
  return {queue: _bindings(queue, prefix=prefix) for queue in QUEUES}


def _output(delta: int = 1) -> Tensor:
  value = Tensor([1, 2, 3, 4], device="CPU")
  return (value * value + delta).contiguous()


def _freeze_both(monkeypatch):
  bindings = _bindings_by_queue()
  capture = DirectPackedLinearExecutionCapture(bindings_by_queue=bindings)
  evidence, observations = {}, {}
  for queue, aql in (("PM4", "0"), ("AQL", "1")):
    monkeypatch.setenv("AMD_AQL", aql)
    output = _output()
    capture.realize_output(output)
    observations[queue] = capture.observation_post_sync(output, queue)
    evidence[queue] = build_direct_packed_fallback_evidence(
      observations[queue], bindings[queue])
  return bindings, evidence, observations


def _persist_both(tmp_path, bindings, observations):
  paths = {}
  for queue in QUEUES:
    paths[queue] = tmp_path / f"{queue.lower()}-qualification.json"
    artifact = build_direct_packed_qualification_artifact(
      observations[queue], bindings[queue])
    persist_direct_packed_qualification(
      paths[queue], artifact, bindings[queue])
  return paths


def test_real_primitives_capture_exact_linear_and_emit_c8_schema(monkeypatch):
  bindings, evidence, observations = _freeze_both(monkeypatch)
  for queue in QUEUES:
    observation, row = observations[queue], evidence[queue]
    assert observation["schema"] == OBSERVATION_SCHEMA
    assert observation["runtime_cache_join_verified"] is True
    assert observation["post_sync_attestation"] is True
    manifest = observation["manifest"]
    assert manifest["artifact_schema"] == ARTIFACT_SCHEMA
    assert manifest["queue_mode"] == queue
    assert manifest["direct_packed_program_ordinals"]
    assert manifest["programs"]
    assert all(program["function_name"].startswith("E_")
               for program in manifest["programs"])
    assert row["schema"] == EVIDENCE_SCHEMA
    assert row["artifact_schema"] == ARTIFACT_SCHEMA
    assert row["queue_mode"] == queue
    assert row["binary_sha256"] == manifest["aggregate_binary_sha256"]
    assert row["executable_identity"] == manifest["executable_identity"]
    assert validate_direct_packed_fallback_evidence(row, bindings[queue]) == row
    # The existing C8 contract consumes the new producer without a schema
    # translation or weakened identity check.
    assert _fallback_evidence(row, queue=queue) == row


def test_frozen_attestor_accepts_repeat_of_exact_compiled_execution(monkeypatch):
  bindings, evidence, _ = _freeze_both(monkeypatch)
  attestor = FrozenDirectPackedExecutableAttestor(
    expected_evidence_by_queue=evidence, bindings_by_queue=bindings)
  monkeypatch.setenv("AMD_AQL", "0")
  output = _output()
  attestor.realize_output(output)
  observed = attestor.attest_post_sync(output, "PM4")
  assert observed == evidence["PM4"]
  assert attestor.last_observation["runtime_cache_join_verified"] is True
  assert output.numpy().tolist() == [2, 5, 10, 17]


def test_frozen_attestor_rejects_schedule_or_code_object_drift(monkeypatch):
  bindings, evidence, _ = _freeze_both(monkeypatch)
  attestor = FrozenDirectPackedExecutableAttestor(
    expected_evidence_by_queue=evidence, bindings_by_queue=bindings)
  monkeypatch.setenv("AMD_AQL", "0")
  output = _output(delta=2)
  attestor.realize_output(output)
  with pytest.raises(ValueError, match="observed direct_packed executable differs"):
    attestor.attest_post_sync(output, "PM4")
  assert attestor.last_observation["manifest"]["executable_identity"] != \
    evidence["PM4"]["executable_identity"]


def test_capture_fails_closed_when_exact_runtime_key_was_not_observed(monkeypatch):
  bindings, _, _ = _freeze_both(monkeypatch)
  base = production_attestation_primitives()
  primitives = DirectPackedAttestationPrimitives(
    linearize=base.linearize, compile_linear=base.compile_linear,
    execute_compiled=base.execute_compiled,
    runtime_lookup=lambda _key, _device: None,
    profile_events=base.profile_events)
  capture = DirectPackedLinearExecutionCapture(
    bindings_by_queue=bindings, primitives=primitives)
  monkeypatch.setenv("AMD_AQL", "0")
  output = _output()
  capture.realize_output(output)
  with pytest.raises(ValueError, match="absent from exact runtime cache key"):
    capture.observation_post_sync(output, "PM4")


def test_capture_crosschecks_profile_code_object_when_profile_exposes_one(monkeypatch):
  bindings, _, _ = _freeze_both(monkeypatch)
  base, events = production_attestation_primitives(), []

  def execute(compiled, var_vals):
    base.execute_compiled(compiled, var_vals)
    program = next(call.src[0] for call in compiled.src
                   if getattr(call.src[0].op, "name", "") == "PROGRAM")
    events.append(SimpleNamespace(
      device=program.src[1].arg, name=program.arg.function_name,
      lib=b"not-the-compiled-code-object"))

  primitives = DirectPackedAttestationPrimitives(
    linearize=base.linearize, compile_linear=base.compile_linear,
    execute_compiled=execute, runtime_lookup=base.runtime_lookup,
    profile_events=lambda: tuple(events))
  capture = DirectPackedLinearExecutionCapture(
    bindings_by_queue=bindings, primitives=primitives)
  monkeypatch.setenv("AMD_AQL", "0")
  output = _output()
  capture.realize_output(output)
  with pytest.raises(ValueError, match="PROFILE code object differs"):
    capture.observation_post_sync(output, "PM4")


def test_capture_requires_production_direct_packed_program_prefix(monkeypatch):
  capture = DirectPackedLinearExecutionCapture(
    bindings_by_queue=_bindings_by_queue(prefix="prefill_q4k_direct_packed_load_gemm"))
  monkeypatch.setenv("AMD_AQL", "0")
  output = _output()
  capture.realize_output(output)
  with pytest.raises(ValueError, match="no PROGRAM matching"):
    capture.observation_post_sync(output, "PM4")


def test_frozen_evidence_binding_and_content_drift_fail_at_construction(monkeypatch):
  bindings, evidence, _ = _freeze_both(monkeypatch)
  drifted = copy.deepcopy(evidence)
  drifted["PM4"]["binary_sha256"] = "0" * 64
  payload = {key: value for key, value in drifted["PM4"].items()
             if key != "evidence_identity"}
  import hashlib, json
  drifted["PM4"]["evidence_identity"] = "sha256:" + hashlib.sha256(json.dumps(
    payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
  with pytest.raises(ValueError, match="artifact identity differs"):
    FrozenDirectPackedExecutableAttestor(
      expected_evidence_by_queue=drifted, bindings_by_queue=bindings)


def test_pending_execution_and_post_sync_output_queue_are_single_use(monkeypatch):
  bindings = _bindings_by_queue()
  capture = DirectPackedLinearExecutionCapture(bindings_by_queue=bindings)
  monkeypatch.setenv("AMD_AQL", "0")
  output = _output()
  capture.realize_output(output)
  with pytest.raises(RuntimeError, match="prior direct_packed execution"):
    capture.realize_output(_output())
  with pytest.raises(ValueError, match="output or queue differs"):
    capture.observation_post_sync(_output(), "PM4")
  with pytest.raises(RuntimeError, match="no direct_packed execution"):
    capture.observation_post_sync(output, "PM4")


def test_production_runner_requires_frozen_pair_and_composes_existing_boundary(
    monkeypatch, tmp_path):
  bindings, evidence, observations = _freeze_both(monkeypatch)
  paths = _persist_both(tmp_path, bindings, observations)
  monkeypatch.setenv("AMD_AQL", "0")
  events, clock = [], iter((1_000_000, 3_000_000))

  def executor(linear, value, spec):
    events.append(("execute", linear, value, spec))
    return _output()

  runner, attestor = make_production_direct_packed_attested_runner(
    linear="linear", input_tensor="input", route_spec="spec",
    qualification_paths_by_queue=paths, bindings_by_queue=bindings,
    synchronize=lambda: events.append("sync"), executor=executor,
    clock_ns=lambda: next(clock))
  receipt = runner(queue_mode="PM4", clock_identity=bindings["PM4"].clock_identity)
  assert receipt["complete_role_ms"] == 2.0
  assert receipt["fallback_evidence"] == evidence["PM4"]
  assert events == [
    "sync", ("execute", "linear", "input", "spec"), "sync"]
  assert attestor.last_observation["post_sync_attestation"] is True


def test_qualification_bootstrap_is_untimed_atomic_and_no_replace(
    monkeypatch, tmp_path):
  bindings, events = _bindings_by_queue(), []
  monkeypatch.setenv("AMD_AQL", "0")
  output_path = tmp_path / "pm4-qualified.json"

  def executor(linear, value, spec):
    events.append(("execute", linear, value, spec))
    return _output()

  artifact = qualify_and_freeze_production_direct_packed(
    linear="linear", input_tensor="input", route_spec="spec",
    queue_mode="PM4", bindings_by_queue=bindings, output=output_path,
    synchronize=lambda: events.append("sync"), executor=executor)
  assert events == [
    "sync", ("execute", "linear", "input", "spec"), "sync"]
  assert artifact["qualification_only"] is True
  assert artifact["timing_samples_collected"] is False
  assert "complete_role_ms" not in artifact and "samples_ms" not in artifact
  assert load_direct_packed_qualification(
    output_path, bindings["PM4"]) == artifact
  assert not list(tmp_path.glob(".*.tmp"))
  with pytest.raises(FileExistsError, match="refusing to replace"):
    qualify_and_freeze_production_direct_packed(
      linear="linear", input_tensor="input", route_spec="spec",
      queue_mode="PM4", bindings_by_queue=bindings, output=output_path,
      synchronize=lambda: events.append("unexpected-sync"), executor=executor)
  assert "unexpected-sync" not in events


def test_timed_runner_refuses_missing_preexisting_queue_qualification(
    monkeypatch, tmp_path):
  bindings, _, observations = _freeze_both(monkeypatch)
  pm4 = tmp_path / "pm4.json"
  persist_direct_packed_qualification(
    pm4, build_direct_packed_qualification_artifact(
      observations["PM4"], bindings["PM4"]), bindings["PM4"])
  with pytest.raises(FileNotFoundError):
    make_production_direct_packed_attested_runner(
      linear="linear", input_tensor="input", route_spec="spec",
      qualification_paths_by_queue={
        "PM4": pm4, "AQL": tmp_path / "missing-aql.json"},
      bindings_by_queue=bindings, executor=lambda *_args: _output(),
      synchronize=lambda: None)
