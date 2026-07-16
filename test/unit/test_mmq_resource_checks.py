import pytest
from extra.qk.mmq_resource_checks import check_mmq_resource_evidence

def artifact(**changes):
  value = {"schema": "tinygrad.kernel_resource_trace.v1", "candidate_id": "c", "kernel_name": "k",
    "resources": {"vgpr": 96, "lds_bytes": 4096, "scratch_bytes": 0, "vgpr_spills": 0, "sgpr_spills": 0,
      "workgroup_threads": 64, "max_workgroup_threads": 256, "wavefront_size": 32, "occupancy": .5},
    "isa": {"barrier_sites": 2, "mfma_sites": 8}}
  for section, vals in changes.items(): value[section] = {**value[section], **vals}
  return value

def check(a):
  return check_mmq_resource_evidence(a, expected_candidate_id="c", expected_kernel_name="k", max_vgpr=128,
    max_lds_bytes=65536, min_occupancy=.25, expected_wavefront_size=32)

def test_complete_final_artifact_passes(): assert check(artifact())["candidate_id"] == "c"

@pytest.mark.parametrize(("section", "field"), [("resources", "scratch_bytes"), ("resources", "occupancy"),
  ("resources", "vgpr_spills"), ("isa", "mfma_sites")])
def test_missing_evidence_fails_closed(section, field):
  a = artifact(); del a[section][field]
  with pytest.raises(ValueError, match="missing"): check(a)

def test_scratch_occupancy_and_mfma_are_gates():
  for section, change, message in (("resources", {"scratch_bytes": 1}, "scratch"),
                                   ("resources", {"occupancy": .1}, "occupancy"),
                                   ("isa", {"mfma_sites": 0}, "MFMA")):
    with pytest.raises(ValueError, match=message): check(artifact(**{section: change}))

def test_multi_wave_requires_barrier():
  with pytest.raises(ValueError, match="barrier"): check(artifact(resources={"workgroup_threads": 256}, isa={"barrier_sites": 0}))
