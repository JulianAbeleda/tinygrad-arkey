from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from extra.qk.mmq_attn_qo_c8_runtime import DirectPackedObjects
from extra.qk.mmq_ffn_gate_up_c8_runtime import (
  FfnGateUpCandidateInputs, FfnGateUpRouteCallback,
  compose_ffn_gate_up_queue_runners,
)
from extra.qk.mmq_ffn_gate_up_low_level_routes import (
  AmdAllocatorHostIoCensus, make_ffn_gate_up_candidate_route,
  make_ffn_gate_up_direct_route,
)
from extra.qk.mmq_ffn_gate_up_matched_timing_contract import (
  build_ffn_gate_up_matched_complete_role_timing_contract,
)


def _sid(label: str) -> str:
  return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


def _raw(label: str) -> str:
  return hashlib.sha256(label.encode()).hexdigest()


def _authority() -> dict:
  workload_identity, input_identity = _sid("workload"), _sid("inputs")
  return {
    "workload_identity": workload_identity,
    "input_identity": input_identity,
    "logical_q4_identity": _sid("logical-q4"),
    "resident_fp16_activation_identity": _sid("resident-fp16"),
    "candidate_binding": {
      "family_identity": _sid("family"),
      "candidate_executable_identity": _sid("candidate-executable"),
      "program_key": _raw("program"), "binary_sha256": _raw("binary"),
    },
    "direct_bindings_by_queue": {
      queue: {
        "qualification_identity": _sid(f"{queue}-qualification"),
        "executable_identity": _sid(f"{queue}-direct-executable"),
        "binary_sha256": _raw(f"{queue}-direct-binary"),
      } for queue in ("PM4", "AQL")
    },
    "joint_session_c7_identity": _sid("joint-c7"),
    "c6_bindings_by_queue": {
      queue: {
        "evidence_identity": _sid(f"{queue}-c6"),
        "candidate_correctness_identity": _sid(f"{queue}-correctness"),
        "comparator_identity": _sid(f"{queue}-comparator"),
        "workload_identity": workload_identity,
        "input_identity": input_identity,
      } for queue in ("PM4", "AQL")
    },
    "transition_preflight_bindings_by_queue": {
      queue: {
        field: _sid(f"{queue}-{field}") for field in (
          "candidate_candidate", "direct_direct",
          "direct_candidate_prefix1", "direct_candidate_full_role",
          "candidate_direct_candidate")
      } for queue in ("PM4", "AQL")
    },
  }


def _contract(authority: dict) -> dict:
  return build_ffn_gate_up_matched_complete_role_timing_contract(**authority)


def _candidate_inputs(authority: dict) -> FfnGateUpCandidateInputs:
  return FfnGateUpCandidateInputs(
    role_spec=SimpleNamespace(
      role="ffn_gate_up", shape=(512, 17408, 5120), epochs=20),
    words=np.zeros(1, dtype=np.uint32),
    q4_epoch_major=np.zeros(1, dtype=np.uint32),
    resident_fp16_activation=np.zeros((1, 1, 1), dtype=np.float16),
    q8_producer_semantics=
      "per_invocation_from_resident_fp16_inside_outer_synchronized_wall",
    q8_reference_sha256={
      "values": _raw("q8-values"), "scales": _raw("q8-scales"),
      "sums": _raw("q8-sums")},
    fixture_identity=_sid("fixture"),
    workload_identity=authority["workload_identity"],
    input_identity=authority["input_identity"],
    logical_q4_identity=authority["logical_q4_identity"],
    resident_fp16_activation_identity=
      authority["resident_fp16_activation_identity"])


class Output:
  pass


@dataclass
class FakeCandidateInvocation:
  output: object
  candidate_phase_trace: dict
  pending_observation: object


@dataclass
class FakeCandidateAttestation:
  status: str
  queue_mode: str
  family_identity: str
  candidate_executable_identity: str
  input_identity: str
  program_key: str
  binary_sha256: str
  launch_count: int
  observation_identity: str


class FakeCandidateSession:
  prepared_kwargs = None
  last = None

  @classmethod
  def prepare(cls, **kwargs):
    cls.prepared_kwargs = kwargs
    cls.last = cls(
      kwargs["authority"].queue_mode,
      kwargs["authority"].family_identity,
      kwargs["authority"].candidate_executable_identity,
      kwargs["authority"].input_identity,
      kwargs["authority"].program_key,
      kwargs["authority"].binary_sha256)
    return cls.last

  def __init__(
      self, queue, family_identity, executable, input_identity,
      program_key, binary_sha256):
    self.queue, self.family_identity, self.executable, self.input_identity = \
      queue, family_identity, executable, input_identity
    self.program_key, self.binary_sha256 = program_key, binary_sha256
    self.pending_q8 = None
    self.invocation = None

  def invoke(self, prefix_epochs):
    assert prefix_epochs == 20
    self.pending_q8 = Output()
    self.invocation = FakeCandidateInvocation(
      Output(), {"schema": "candidate-trace"}, self.pending_q8)
    return self.invocation

  def attest_post_sync(self, invocation, queue_mode):
    assert invocation is self.invocation
    result = FakeCandidateAttestation(
      "PASS", queue_mode, self.family_identity, self.executable,
      self.input_identity, self.program_key, self.binary_sha256, 20,
      _sid("candidate-observation"))
    self.pending_q8 = self.invocation = None
    return result


def _candidate_route(
    *, queue="PM4", session_class=FakeCandidateSession,
    invocation_type=FakeCandidateInvocation,
    attestation_type=FakeCandidateAttestation,
    ):
  authority = _authority()
  family = SimpleNamespace(
    family_identity=authority["candidate_binding"]["family_identity"],
    binding=SimpleNamespace(
      program_key=authority["candidate_binding"]["program_key"],
      binary_sha256=authority["candidate_binding"]["binary_sha256"]))
  canaries = {
    mode: {
      "status": "PASS", "queue_mode": mode,
      "family_identity": family.family_identity}
    for mode in ("PM4", "AQL")
  }
  route = make_ffn_gate_up_candidate_route(
    queue_mode=queue, matched_timing_contract=_contract(authority),
    contract_validation_kwargs=authority, family=family,
    frozen_bundle="/bundle", staged_family_manifest="/family.json",
    runtime_canary_by_queue=canaries,
    candidate_inputs=_candidate_inputs(authority),
    low_level_authority=SimpleNamespace(
      queue_mode=queue,
      family_identity=family.family_identity,
      candidate_executable_identity=authority["candidate_binding"][
        "candidate_executable_identity"],
      input_identity=authority["input_identity"],
      program_key=family.binding.program_key,
      binary_sha256=family.binding.binary_sha256),
    low_level_dependencies=object(), session_class=session_class,
    invocation_type=invocation_type, attestation_type=attestation_type)
  return authority, route


def _direct_evidence(authority: dict) -> dict:
  return {
    queue: {
      "status": "PASS", "queue_mode": queue,
      "input_identity": authority["input_identity"],
      "executable_identity":
        authority["direct_bindings_by_queue"][queue]["executable_identity"],
      "evidence_identity": _sid(f"{queue}-direct-evidence"),
    } for queue in ("PM4", "AQL")
  }


class FakeDirectAttestor:
  def __init__(self, *, expected_evidence_by_queue, bindings_by_queue, **kwargs):
    self.expected = expected_evidence_by_queue
    self.pending = None

  def realize_output(self, output):
    output.realized = True
    self.pending = output

  def attest_post_sync(self, output, queue):
    if output is not self.pending:
      raise ValueError("output differs")
    self.pending = None
    return self.expected[queue]


class FakePackedQ4:
  def __init__(self):
    self.realize_count = 0

  def realize(self):
    self.realize_count += 1
    return self


def _direct_route(*, queue="PM4", executor=None, evidence=None):
  authority = _authority()
  output = Output()
  output.realized = False
  packed_q4 = FakePackedQ4()
  linear = SimpleNamespace(prefill_packed_weight=lambda: packed_q4)
  objects = DirectPackedObjects(linear, "activation", "route-spec")
  route = make_ffn_gate_up_direct_route(
    queue_mode=queue, matched_timing_contract=_contract(authority),
    contract_validation_kwargs=authority, direct_objects=objects,
    qualification_paths_by_queue={"PM4": "/p", "AQL": "/a"},
    bindings_by_queue={"PM4": object(), "AQL": object()},
    executor=(lambda *args: output) if executor is None else executor,
    evidence_loader=lambda *args: (
      _direct_evidence(authority) if evidence is None else evidence),
    attestor_factory=FakeDirectAttestor)
  return authority, route, output, packed_q4


def test_candidate_route_binds_session_and_retains_output_q8_until_attestation():
  authority, route = _candidate_route()
  assert isinstance(route, FfnGateUpRouteCallback)
  assert route.queue_mode == "PM4"
  assert route.input_identity == authority["input_identity"]
  assert route.executable_identity == authority["candidate_binding"][
    "candidate_executable_identity"]
  invocation = route.invoke()
  output_ref = weakref.ref(invocation.output)
  q8_ref = weakref.ref(FakeCandidateSession.last.pending_q8)
  del invocation
  gc.collect()
  assert output_ref() is not None and q8_ref() is not None
  observed = route.attest_post_sync(output_ref(), "PM4")
  assert observed["queue_mode"] == "PM4"
  assert observed["executable_identity"] == route.executable_identity
  assert observed["input_identity"] == route.input_identity
  gc.collect()
  assert q8_ref() is None


def test_candidate_route_rejects_queue_and_observation_drift():
  _, route = _candidate_route()
  invocation = route.invoke()
  with pytest.raises(ValueError, match="output or queue differs"):
    route.attest_post_sync(invocation.output, "AQL")

  _, route = _candidate_route()
  invocation = route.invoke()
  FakeCandidateSession.last.executable = _sid("wrong-executable")
  with pytest.raises(ValueError, match="candidate low-level observation differs"):
    route.attest_post_sync(invocation.output, "PM4")


class LegacyCandidateSession(FakeCandidateSession):
  def invoke(self, prefix_epochs):
    assert prefix_epochs == 20
    return {"legacy_receipt": True}


def test_candidate_route_rejects_legacy_receipt():
  _, route = _candidate_route(session_class=LegacyCandidateSession)
  with pytest.raises(TypeError, match="legacy receipt"):
    route.invoke()


def test_production_candidate_rejects_reference_numpy_activation():
  with pytest.raises(TypeError, match="shared AMD Tensor resident FP16"):
    _candidate_route(
      session_class=None, invocation_type=None, attestation_type=None)


def test_candidate_route_builds_one_static_q4_object_before_invocations():
  authority = _authority()
  inputs = _candidate_inputs(authority)
  resident_q4 = Output()
  calls = []

  def builder(value):
    calls.append(value)
    return resident_q4

  family = SimpleNamespace(
    family_identity=authority["candidate_binding"]["family_identity"],
    binding=SimpleNamespace(
      program_key=authority["candidate_binding"]["program_key"],
      binary_sha256=authority["candidate_binding"]["binary_sha256"]))
  route = make_ffn_gate_up_candidate_route(
    queue_mode="PM4", matched_timing_contract=_contract(authority),
    contract_validation_kwargs=authority, family=family,
    frozen_bundle="/bundle", staged_family_manifest="/family.json",
    runtime_canary_by_queue={
      "PM4": {
        "status": "PASS", "queue_mode": "PM4",
        "family_identity": family.family_identity}},
    candidate_inputs=inputs,
    low_level_authority=SimpleNamespace(
      queue_mode="PM4", family_identity=family.family_identity,
      candidate_executable_identity=authority["candidate_binding"][
        "candidate_executable_identity"],
      input_identity=authority["input_identity"],
      program_key=family.binding.program_key,
      binary_sha256=family.binding.binary_sha256),
    low_level_dependencies=object(), session_class=FakeCandidateSession,
    invocation_type=FakeCandidateInvocation,
    attestation_type=FakeCandidateAttestation,
    static_q4_builder=builder)
  assert calls == [inputs.q4_epoch_major]
  assert FakeCandidateSession.prepared_kwargs["q4_epoch_major"] is resident_q4
  for _ in range(2):
    invocation = route.invoke()
    route.attest_post_sync(invocation.output, "PM4")
  assert calls == [inputs.q4_epoch_major]
  assert FakeCandidateSession.prepared_kwargs["q4_epoch_major"] is resident_q4


def test_direct_route_uses_production_executor_realizer_and_frozen_observation():
  authority, route, output, packed_q4 = _direct_route(queue="AQL")
  assert packed_q4.realize_count == 1
  invocation = route.invoke()
  assert invocation.output is output and output.realized is False
  assert route.realize_output(output) is None and output.realized is True
  observed = route.attest_post_sync(output, "AQL")
  assert observed["queue_mode"] == "AQL"
  assert observed["executable_identity"] == \
    authority["direct_bindings_by_queue"]["AQL"]["executable_identity"]
  assert observed["input_identity"] == authority["input_identity"]
  route.invoke()
  assert packed_q4.realize_count == 1


def test_direct_route_rejects_legacy_receipt_and_frozen_identity_drift():
  with pytest.raises(TypeError, match="legacy receipt"):
    _direct_route(executor=lambda *args: {"legacy_receipt": True})[1].invoke()
  authority = _authority()
  evidence = _direct_evidence(authority)
  evidence["PM4"]["input_identity"] = _sid("wrong-input")
  with pytest.raises(ValueError, match="differs from matched contract"):
    _direct_route(evidence=evidence)


class FakeAllocator:
  def __init__(self):
    self.calls = []
    self.dev = SimpleNamespace(device="AMD")

  def _copyout(self, destination, source):
    self.calls.append((destination.nbytes, source))
    destination[:] = bytes([source]) * destination.nbytes


def test_amd_allocator_host_io_census_counts_and_restores_copyout():
  allocator = FakeAllocator()
  assert "_copyout" not in allocator.__dict__
  census = AmdAllocatorHostIoCensus(
    allocator, provider_identity="test-provider")
  with census:
    before = census.snapshot()
    allocator._copyout(memoryview(bytearray(7)), 3)
    after = census.snapshot()
    assert before["copyout_count"] == before["copyout_bytes"] == 0
    assert after["readback_count"] == after["copyout_count"] == 1
    assert after["copyout_bytes"] == 7
  assert "_copyout" not in allocator.__dict__
  destination = memoryview(bytearray(2))
  allocator._copyout(destination, 4)
  assert bytes(destination) == b"\x04\x04"
  with pytest.raises(RuntimeError, match="installed"):
    census.snapshot()


def test_amd_allocator_host_io_census_restores_exact_instance_binding():
  allocator = FakeAllocator()
  calls = []

  def prior(destination, source):
    calls.append((destination.nbytes, source))

  allocator._copyout = prior
  census = AmdAllocatorHostIoCensus(allocator)
  with census:
    allocator._copyout(memoryview(bytearray(5)), 8)
  assert allocator.__dict__["_copyout"] is prior
  allocator._copyout(memoryview(bytearray(2)), 9)
  assert calls == [(5, 8), (2, 9)]


def test_runtime_composition_still_has_no_default_route_builders():
  with pytest.raises(ValueError, match="explicit production-faithful"):
    compose_ffn_gate_up_queue_runners(
      object(), queue_mode="PM4", clock_identity="clock-policy-0")
