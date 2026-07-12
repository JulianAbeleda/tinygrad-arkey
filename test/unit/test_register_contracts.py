import pytest

from tinygrad.codegen.opt.register_contracts import Lease, RegisterBank, RegisterDescriptor, RegisterRole
from tinygrad.renderer.isa.amd_register_contracts import GFX1100_REGISTER_DESCRIPTOR, gfx1100_register_descriptor


def test_gfx1100_descriptor_snapshot_is_stable():
  descriptor = gfx1100_register_descriptor()
  assert descriptor == GFX1100_REGISTER_DESCRIPTOR
  assert descriptor.target == "AMD:gfx1100" and descriptor.wave_size == 32
  assert (descriptor.vgpr_count, descriptor.sgpr_count) == (256, 104)
  assert descriptor.snapshot() == {
    "target": "AMD:gfx1100", "wave_size": 32, "vgpr_count": 256, "sgpr_count": 104,
    "leases": [
      {"role": "kernarg", "bank": "sgpr", "start": 0, "end": 2, "width": 2, "mode": "abi", "alignment": 2, "reserved": True},
      {"role": "workgroup_id", "bank": "sgpr", "start": 2, "end": 6, "width": 4, "mode": "abi", "alignment": 1, "reserved": True},
      {"role": "pointer", "bank": "sgpr", "start": 6, "end": 40, "width": 34, "mode": "abi", "alignment": 2, "reserved": True},
      {"role": "scalar_counter", "bank": "sgpr", "start": 40, "end": 64, "width": 24, "mode": "default", "alignment": 1, "reserved": True},
      {"role": "scalar_temp", "bank": "sgpr", "start": 64, "end": 104, "width": 40, "mode": "default", "alignment": 1, "reserved": True},
      {"role": "workitem_id", "bank": "vgpr", "start": 0, "end": 1, "width": 1, "mode": "abi", "alignment": 1, "reserved": True},
      {"role": "virtual", "bank": "vgpr", "start": 1, "end": 256, "width": 255, "mode": "no_wmma", "alignment": 1, "reserved": True},
      {"role": "virtual", "bank": "vgpr", "start": 1, "end": 200, "width": 199, "mode": "wmma_single_tile", "alignment": 1, "reserved": True},
      {"role": "accumulator", "bank": "vgpr", "start": 1, "end": 17, "width": 16, "mode": "legacy_accum", "alignment": 1, "reserved": True},
      {"role": "accumulator", "bank": "vgpr", "start": 8, "end": 16, "width": 8, "mode": "wmma_multi_tile", "alignment": 8, "reserved": True},
      {"role": "fragment", "bank": "vgpr", "start": 200, "end": 238, "width": 38, "mode": "wmma_single_tile", "alignment": 2, "reserved": True},
      {"role": "lds_pack", "bank": "vgpr", "start": 232, "end": 236, "width": 4, "mode": "lds_pack", "alignment": 2, "reserved": True},
    ]}


def test_descriptor_allows_alternative_mode_overlap_but_rejects_same_mode_overlap():
  Lease(RegisterRole.WORKITEM_ID, RegisterBank.VGPR, 0, 1)
  with pytest.raises(ValueError, match="overlapping"):
    RegisterDescriptor("test", 32, 16, 16, (Lease(RegisterRole.VIRTUAL, RegisterBank.VGPR, 0, 4),
      Lease(RegisterRole.FRAGMENT, RegisterBank.VGPR, 2, 4)))
  descriptor = RegisterDescriptor("test", 32, 16, 16, (Lease(RegisterRole.VIRTUAL, RegisterBank.VGPR, 0, 4, mode="a"),
    Lease(RegisterRole.FRAGMENT, RegisterBank.VGPR, 2, 4, mode="b")))
  assert descriptor.lease(RegisterRole.VIRTUAL, mode="a").end == 4


@pytest.mark.parametrize("kwargs", ({"start": 1, "width": 2, "alignment": 2}, {"start": 0, "width": 0}, {"start": -1, "width": 1}))
def test_lease_rejects_invalid_ranges(kwargs):
  with pytest.raises(ValueError): Lease(RegisterRole.VIRTUAL, RegisterBank.VGPR, **kwargs)
