import pytest

from tinygrad.dtype import dtypes
from tinygrad.renderer.amd import elf
from tinygrad.uop.ops import Ops, UOp


class _Descriptor:
  compute_pgm_rsrc1 = 0
  group_segment_fixed_size = 0
  kernel_code_properties = 1 << elf.amdgpu_kd.KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32_SHIFT


def _program(): return UOp(Ops.PROGRAM, src=(UOp(Ops.SINK),))
def _linear(): return UOp(Ops.LINEAR, src=(UOp(Ops.INS, dtypes.void, arg="s_endpgm"),))


def test_final_elf_capture_uses_exact_binary_without_reassembly(monkeypatch):
  monkeypatch.setattr(elf, "assemble_linear", lambda *_: (_ for _ in ()).throw(AssertionError("must not reassemble")))
  monkeypatch.setattr(elf, "kernel_descriptor_from_elf", lambda binary: _Descriptor())
  capture = elf.final_elf_capture(_program(), _linear(), "gfx1100", binary=b"exact-final-elf", target="gfx1100")
  assert capture["binary"] == b"exact-final-elf"
  assert capture["descriptor"]["authority"] == "final_code_object_descriptor"
  assert capture["descriptor"]["resources"]["wavefront_size"] == 32


def test_final_elf_capture_rejects_missing_exact_binary():
  with pytest.raises(ValueError, match="exact final ELF"):
    elf.final_elf_capture(_program(), _linear(), "gfx1100", binary=b"")


def test_descriptor_register_extraction_masks_neighboring_control_fields():
  desc = _Descriptor()
  desc.compute_pgm_rsrc1 = (26 << elf.amdgpu_kd.COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT) | \
    (9 << elf.amdgpu_kd.COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT_SHIFT) | (1 << 30)
  assert elf.descriptor_register_counts(desc, is_cdna=False) == (216, None)
  assert elf.descriptor_register_counts(desc, is_cdna=True) == (216, 80)
