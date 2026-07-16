import json
import pytest

from extra.qk.prefill.execution_bridge_contracts import (
  ArtifactRequest, CompileArtifactMetadata, CorrectnessProtocol, CounterGroupRequest, DispatchEvidence, DISPATCH_STATES,
  ExecutionRequest, ExecutionResult, GuardProtocol, PhaseResult, SafetyAdmission, SemanticOperandPlan, SemanticScheduleIdentity,
  TimingProtocol, TransportPlan, TypedError, UnsupportedOutcome, WorkloadIdentity, dispatch_state, reject_synthetic,
)


def test_contracts_are_deterministic_and_json_serializable():
  workload = WorkloadIdentity("w1", "attn_qo", (512, 4096, 4096), ("fp16", "fp32"))
  schedule = SemanticScheduleIdentity(workload.digest, "s" * 64, ("a", "b", "out"), "a" * 64)
  plan = TransportPlan("lds", schedule.schedule_digest, {"barriers": 2})
  artifact = CompileArtifactMetadata("c" * 64, schedule.schedule_digest, plan.transport, "AMD:gfx1100", "a" * 64, "b" * 64, "s" * 64)
  assert json.loads(json.dumps(artifact.to_dict()))["schema"].endswith("compile_artifact.v1")
  assert workload.digest == WorkloadIdentity("w1", "attn_qo", (512, 4096, 4096), ("fp16", "fp32")).digest
  assert DispatchEvidence("e", "w", "lds", "completed").state == "completed"


def test_safety_and_dispatch_reject_unsafe_values():
  with pytest.raises(ValueError): SafetyAdmission("w", "e", True, False, "h")
  with pytest.raises(ValueError): DispatchEvidence("e", "w", "lds", "unknown")
  with pytest.raises(ValueError): WorkloadIdentity("w", "r", (0,), ("fp16",))


def test_typed_dispatch_state_vocabulary_and_synthetic_rejection():
  # P0-1: the full lifecycle vocabulary is typed, not a bare boolean.
  assert set(DISPATCH_STATES) == {"not_attempted", "attempted", "submitted", "completed",
                                  "failed", "timed_out", "device_lost"}
  assert dispatch_state("not_attempted") == "not_attempted"
  with pytest.raises(ValueError): dispatch_state("dispatched")
  # P0-4: synthetic evidence is rejected only in production-mode decisions.
  err = TypedError("guard_corruption", "dispatch", recoverable=False, candidate="c" * 64)
  assert err.digest and err.recoverable is False
  reject_synthetic({"synthetic": True}, production=False)  # tolerated offline
  with pytest.raises(ValueError): reject_synthetic({"synthetic": True}, production=True)


def test_operand_execution_request_round_trip_and_compatible_transport_defaults():
  old_plan = TransportPlan("lds", "s" * 64)
  assert old_plan.operands == ()
  operands = (
    SemanticOperandPlan("A", "activation", "arg0", "register_resident"),
    SemanticOperandPlan("B", "weight", "arg1", "lds_staged", {"bytes": 4096}),
  )
  plan = TransportPlan("mixed", "s" * 64, operands=operands)
  request = ExecutionRequest(
    "exp", "candidate", "baseline", "w" * 64, "s" * 64, plan,
    {"provider": "tinygrad", "target": "AMD:gfx1100"}, {"compiler": "tinygrad"},
    artifacts=(ArtifactRequest("final_isa_manifest"), ArtifactRequest("resource_metadata")),
    counter_groups=(CounterGroupRequest("cache", ("GL2C_HIT",), optional_when_unsupported=True),),
    correctness=CorrectnessProtocol("reference", atol=1e-5, rtol=1e-5),
    guard=GuardProtocol(1000), timing=TimingProtocol(2, 9, 7, noise_threshold=0.01),
  )
  restored = ExecutionRequest.from_dict(json.loads(request.to_json()))
  assert restored == request
  assert restored.transport_plan.operands[1].declared_strategy == "lds_staged"
  with pytest.raises(ValueError): SemanticOperandPlan("A", "activation", "arg0", "mall_resident")


def test_typed_unsupported_counter_does_not_replace_timing_result():
  unsupported = UnsupportedOutcome("blocked_gfx11_pmc", "counter", "GL2C_HIT")
  result = ExecutionResult("exp", "candidate", "r" * 64, (
    PhaseResult("correctness", "passed", evidence={"scope": "full_output"}),
    PhaseResult("timing", "passed", evidence={"samples_ns": [10, 11, 10]}),
    PhaseResult("counter", "unsupported", unsupported=(unsupported,)),
  ))
  restored = ExecutionResult.from_dict(json.loads(result.to_json()))
  assert restored == result
  assert restored.phases[1].status == "passed"
  assert restored.phases[2].unsupported[0].reason == "blocked_gfx11_pmc"
  with pytest.raises(ValueError): PhaseResult("counter", "unsupported")


def test_operand_request_rejects_duplicate_abi_and_invalid_numeric_protocols():
  with pytest.raises(ValueError):
    TransportPlan("mixed", "s" * 64, operands=(
      SemanticOperandPlan("A", "activation", "arg0", "register_resident"),
      SemanticOperandPlan("B", "weight", "arg0", "lds_staged"),
    ))
  with pytest.raises(ValueError): CorrectnessProtocol("reference", atol=float("nan"))
  with pytest.raises(ValueError): TimingProtocol(0, 5, 1, noise_threshold=float("inf"))
