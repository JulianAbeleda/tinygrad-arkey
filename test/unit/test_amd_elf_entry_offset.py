from tinygrad.renderer.amd.elf import assemble_linear, kernel_descriptor_from_elf
from tinygrad.runtime.autogen import hsa
from tinygrad.runtime.autogen.amd.cdna.ins import s_endpgm as s_endpgm_cdna, s_nop as s_nop_cdna
from tinygrad.runtime.autogen.amd.rdna3.ins import s_code_end, s_endpgm
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.uop.ops import Ops, UOp


def _assemble(instructions, arch="gfx1100"):
  program = UOp(Ops.PROGRAM, src=(UOp(Ops.SINK),))
  linear = UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=instruction) for instruction in instructions))
  return assemble_linear(program, linear, arch)


def _text(binary):
  _, sections, _ = elf_loader(binary)
  return next(section for section in sections if section.name == ".text")


def test_native_elf_descriptor_entry_points_to_text_section_start():
  binary = _assemble((s_endpgm(),))
  _, sections, _ = elf_loader(binary)
  text = next(section for section in sections if section.name == ".text")
  rodata = next(section for section in sections if section.name == ".rodata")
  descriptor = kernel_descriptor_from_elf(binary)

  assert int(rodata.header.sh_addr) + descriptor.kernel_code_entry_byte_offset == int(text.header.sh_addr)


def test_native_elf_gfx11_text_has_three_cache_lines_after_alignment():
  text = _text(_assemble((s_endpgm(),))).content

  assert len(text) == 4 * 128
  assert text[:4] == s_endpgm().to_bytes()
  assert text[4:] == s_code_end().to_bytes() * ((4 * 128 - 4) // 4)


def test_native_elf_gfx11_exact_cache_line_gets_three_guard_lines():
  instructions = (s_endpgm(),) * (128 // 4)
  text = _text(_assemble(instructions)).content

  assert len(text) == 4 * 128
  assert text[:128] == s_endpgm().to_bytes() * (128 // 4)
  assert text[128:] == s_code_end().to_bytes() * (3 * 128 // 4)


def test_native_elf_gfx11_near_cache_line_still_gets_three_full_guard_lines():
  instructions = (s_endpgm(),) * (124 // 4)
  text = _text(_assemble(instructions)).content

  assert len(text) == 4 * 128
  assert text[124:] == s_code_end().to_bytes() * ((3 * 128 + 4) // 4)


def test_native_elf_gfx10_uses_three_64_byte_guard_lines():
  instructions = (s_endpgm(),) * (64 // 4)
  text = _text(_assemble(instructions, "gfx1030")).content

  assert len(text) == 4 * 64
  assert text[64:] == s_code_end().to_bytes() * (3 * 64 // 4)


def test_native_elf_generic_cdna_preserves_alignment_only_policy():
  text = _text(_assemble((s_endpgm_cdna(),), "gfx908")).content

  assert len(text) == hsa.AMD_ISA_ALIGN_BYTES
  assert text[:4] == s_endpgm_cdna().to_bytes()
  assert text[4:] == s_nop_cdna(0).to_bytes() * ((hsa.AMD_ISA_ALIGN_BYTES - 4) // 4)


def test_native_elf_gfx90a_uses_sixteen_64_byte_guard_lines():
  instructions = (s_endpgm_cdna(),) * (64 // 4)
  text = _text(_assemble(instructions, "gfx90a")).content

  assert len(text) == 17 * 64
  assert text[64:] == s_nop_cdna(0).to_bytes() * (16 * 64 // 4)
