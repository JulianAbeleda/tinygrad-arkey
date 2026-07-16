import pytest

from extra.qk.amd_wmma_resource_adapter import build_amd_wmma_resource_bundle, check_amd_wmma_resources

NOTES = {"vgpr": 96, "sgpr": 32, "vgpr_spills": 0, "sgpr_spills": 0, "lds_bytes": 4096,
         "scratch_bytes": 0, "max_workgroup_threads": 64, "wavefront_size": 32, "dynamic_stack": False}
ASM = """
  v_wmma_f32_16x16x16_f16 v0, v1, v2 // 0000: 00000000
  s_barrier // 0008: 00000000
  s_endpgm // 0010: 00000000
"""

def test_adapter_joins_final_numbers_and_isa_sites():
  row = build_amd_wmma_resource_bundle(candidate_id="c", kernel_name="k", binary=b"\x7fELFcode",
                                       disassembly=ASM, metadata=NOTES, occupancy=.5)
  assert row["resources"] == {"vgpr": 96, "sgpr": 32, "lds_bytes": 4096, "scratch_bytes": 0,
    "vgpr_spills": 0, "sgpr_spills": 0, "workgroup_threads": 64, "max_workgroup_threads": 64,
    "wavefront_size": 32, "dynamic_stack": False, "occupancy": .5}
  assert row["isa"] == {"barrier_sites": 1, "mfma_sites": 1}
  assert check_amd_wmma_resources(candidate_id="c", kernel_name="k", binary=b"\x7fELFcode",
    disassembly=ASM, metadata=NOTES, occupancy=.5, max_vgpr=128, max_lds_bytes=65536,
    min_occupancy=.25, expected_wavefront_size=32)["candidate_id"] == "c"

def test_missing_occupancy_fails_closed():
  with pytest.raises(ValueError, match="missing.*occupancy"):
    check_amd_wmma_resources(candidate_id="c", kernel_name="k", binary=b"\x7fELFcode",
      disassembly=ASM, metadata=NOTES, max_vgpr=128, max_lds_bytes=65536,
      min_occupancy=.25, expected_wavefront_size=32)
