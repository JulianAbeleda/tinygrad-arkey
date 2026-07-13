"""The AMD compile boundary must not promote register roles from raw operands."""
from tinygrad.renderer.isa.amd import AMDISARenderer


def test_compile_capture_requires_compiler_owned_final_role_proof():
  # No proof attachment means no authority, regardless of the binary bytes.
  assert AMDISARenderer.__new__(AMDISARenderer).compile_capture(object(), object(), b"elf") is None
