from tinygrad.codegen import do_assemble
from tinygrad.dtype import dtypes
from tinygrad.renderer.isa import ISARenderer
from tinygrad.uop.ops import Ops, ProgramInfo, UOp


class _Renderer(ISARenderer):
  def asm(self, prg, lin): return b"final-elf"


class _CapturingRenderer(_Renderer):
  def compile_capture(self, prg, lin, binary):
    return {"opaque": True, "binary": binary, "uop_count": len(lin.src)}


def _program():
  return UOp(Ops.PROGRAM, src=(UOp(Ops.SINK), UOp(Ops.DEVICE, arg="CPU")), arg=ProgramInfo())


def _linear():
  return UOp(Ops.LINEAR, src=(UOp(Ops.INS, dtype=dtypes.void, arg="s_endpgm"),))


def test_do_assemble_attaches_only_opt_in_opaque_compile_record():
  plain = do_assemble(_Renderer(None), _program(), _linear())
  captured = do_assemble(_CapturingRenderer(None), _program(), _linear())
  assert plain.arg.aux == ()
  attachment = captured.arg.aux[0]
  assert attachment.record == {"opaque": True, "binary": b"final-elf", "uop_count": 1}
  assert captured.src[-1].arg == b"final-elf"
