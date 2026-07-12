import pytest

from tinygrad.codegen.opt.amd_resource_artifact import (
  AMDPhysicalInterval, AMDResourceArtifact, AMDResourceFacts, join_amd_resource_artifact, validate_amd_resource_artifact)
from tinygrad.codegen.opt.register_contracts import RegisterBank


def _artifact():
  return join_amd_resource_artifact(
    target="gfx1100", abi="amdgpu_kernel", source="define void @k()", binary=b"hsaco",
    candidate_identity="a" * 64, resources=AMDResourceFacts(vgpr=64, sgpr=32, lds_bytes=0),
    intervals=(AMDPhysicalInterval("A", "vgpr", 0, 8), AMDPhysicalInterval("B", "vgpr", 8, 16),
               AMDPhysicalInterval("accumulator", "vgpr", 16, 24)))


def test_amd_resource_artifact_joins_identity_resources_and_intervals():
  artifact = _artifact()
  assert artifact.intervals[0].bank is RegisterBank.VGPR
  assert artifact.source_sha256 != artifact.binary_sha256
  assert artifact.to_json()["candidate_identity"] == "a" * 64
  assert artifact.to_json()["logical_role_intervals"]["A"][0]["start"] == 0
  assert artifact.from_json(artifact.to_json()) == artifact
  assert validate_amd_resource_artifact(artifact, expected_target="gfx1100") is artifact


@pytest.mark.parametrize("intervals", [
  (AMDPhysicalInterval("A", "vgpr", 0, 8), AMDPhysicalInterval("B", "vgpr", 4, 12)),
  (AMDPhysicalInterval("A", "vgpr", 0, 65),),
])
def test_amd_resource_artifact_rejects_overlap_or_resource_overrun(intervals):
  with pytest.raises(ValueError):
    join_amd_resource_artifact(target="gfx1100", abi="amdgpu_kernel", source=b"s", binary=b"b",
      candidate_identity="b" * 64, resources=AMDResourceFacts(vgpr=64, sgpr=32), intervals=intervals)


def test_amd_resource_artifact_rejects_identity_mismatch_and_unknown_facts():
  with pytest.raises(ValueError, match="target identity"):
    validate_amd_resource_artifact(_artifact(), expected_target="gfx1200")
  with pytest.raises(ValueError):
    AMDResourceFacts(vgpr=-1, sgpr=1)
  with pytest.raises(ValueError):
    join_amd_resource_artifact(target="gfx1100", abi="amdgpu_kernel", source=b"s", binary=b"b",
      candidate_identity="not-a-sha", resources=AMDResourceFacts(vgpr=1, sgpr=1),
      intervals=(AMDPhysicalInterval("A", "vgpr", 0, 1),))
  with pytest.raises(ValueError, match="final_program"):
    row = _artifact().to_json(); row["resource_stage"] = "host_estimate"; AMDResourceArtifact.from_json(row)


def test_amd_resource_artifact_can_require_logical_mapping_roles():
  artifact = _artifact()
  assert validate_amd_resource_artifact(artifact, required_roles=("A", "B")) is artifact
  with pytest.raises(ValueError, match="missing required logical register roles"):
    validate_amd_resource_artifact(artifact, required_roles=("A", "stage_buffer"))
  with pytest.raises(ValueError, match="unique"):
    validate_amd_resource_artifact(artifact, required_roles=("A", "A"))
