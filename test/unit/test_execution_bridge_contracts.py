import json
import pytest

from tinygrad.runtime.execution_bridge_contracts import (
  CompileArtifactMetadata, DispatchEvidence, SafetyAdmission, SemanticScheduleIdentity,
  TransportPlan, WorkloadIdentity,
)


def test_contracts_are_deterministic_and_json_serializable():
  workload = WorkloadIdentity("w1", "attn_qo", (512, 4096, 4096), ("fp16", "fp32"))
  schedule = SemanticScheduleIdentity(workload.digest, "s" * 64, ("a", "b", "out"), "a" * 64)
  plan = TransportPlan("lds", schedule.schedule_digest, {"barriers": 2})
  artifact = CompileArtifactMetadata("c" * 64, schedule.schedule_digest, plan.transport, "AMD:gfx1100", "a" * 64, "b" * 64, "s" * 64)
  assert json.loads(json.dumps(artifact.to_dict()))["schema"].endswith("compile_artifact.v1")
  assert workload.digest == WorkloadIdentity("w1", "attn_qo", (512, 4096, 4096), ("fp16", "fp32")).digest
  assert DispatchEvidence("e", "w", "lds", "passed").status == "passed"


def test_safety_and_dispatch_reject_unsafe_values():
  with pytest.raises(ValueError): SafetyAdmission("w", "e", True, False, "h")
  with pytest.raises(ValueError): DispatchEvidence("e", "w", "lds", "unknown")
  with pytest.raises(ValueError): WorkloadIdentity("w", "r", (0,), ("fp16",))
