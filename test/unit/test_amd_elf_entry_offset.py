from tinygrad.renderer.amd.elf import assemble_linear, kernel_descriptor_from_elf
from tinygrad.runtime.autogen.amd.rdna3.ins import s_endpgm
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.uop.ops import Ops, UOp


def test_native_elf_descriptor_entry_points_to_text_section_start():
  program = UOp(Ops.PROGRAM, src=(UOp(Ops.SINK),))
  linear = UOp(Ops.LINEAR, src=(UOp(Ops.INS, arg=s_endpgm()),))
  binary = assemble_linear(program, linear, "gfx1100")
  _, sections, _ = elf_loader(binary)
  text = next(section for section in sections if section.name == ".text")
  rodata = next(section for section in sections if section.name == ".rodata")
  descriptor = kernel_descriptor_from_elf(binary)

  assert int(rodata.header.sh_addr) + descriptor.kernel_code_entry_byte_offset == int(text.header.sh_addr)
