import json
import pytest

from tinygrad.runtime.execution_bridge_contracts import (
  CompileArtifactMetadata, DispatchEvidence, DISPATCH_STATES, SafetyAdmission, SemanticScheduleIdentity,
  TransportPlan, TypedError, WorkloadIdentity, dispatch_state, reject_synthetic,
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
