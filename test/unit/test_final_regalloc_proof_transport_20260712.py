"""CPU-only regression tests for the final-regalloc capture transport."""

from tinygrad.codegen import do_assemble
from tinygrad.renderer.isa import ISARenderer
from tinygrad.uop.ops import Ops, UOp


def test_compile_capture_receives_hashable_final_regalloc_proof():
  proof = ("final_regalloc", ("status", "post_regalloc"),
           ("scratch_spills", 0), ("vgpr_spills", 0), ("sgpr_spills", 0))
  seen = []

  class CaptureRenderer(ISARenderer):
    def asm(self, _prg, _lin): return b"binary"
    def compile_capture(self, prg, lin, binary, final_regalloc):
      seen.append((prg, lin, binary, final_regalloc))
      return None

  renderer = CaptureRenderer.__new__(CaptureRenderer)
  renderer.compile_capture = CaptureRenderer.compile_capture.__get__(renderer, CaptureRenderer)
  prg = UOp(Ops.PROGRAM, src=())
  lin = UOp(Ops.LINEAR, arg=proof, src=(UOp(Ops.INS, arg="nop"),))

  do_assemble(renderer, prg, lin)

  assert seen and seen[0][2] == b"binary"
  assert seen[0][3] == proof
  assert hash(seen[0][3])
