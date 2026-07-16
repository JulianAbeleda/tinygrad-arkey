from tinygrad.dtype import dtypes
from tinygrad.renderer.isa.x86 import X86Ops, X86Renderer, encodings, RAX, RBX, GPR, XMM
from tinygrad.uop.ops import Ops, UOp

def reg(dtype, register): return UOp(Ops.DEFINE_REG, dtype, tag=register)
def ins(op, dtype, register, *src): return UOp(Ops.INS, dtype, src, arg=op, tag=register)

def test_x86_encoding_byte_fixtures():
  cases = {
    "legacy_rex": ins(X86Ops.MOV, dtypes.int64, RAX, reg(dtypes.int64, RBX)),
    "extended_registers": ins(X86Ops.MOV, dtypes.int32, GPR[9], reg(dtypes.int32, GPR[10])),
    "sib_disp8": ins(X86Ops.MOV, dtypes.int64, GPR[9], reg(dtypes.uint64, GPR[12]), reg(dtypes.int64, GPR[10]), UOp.const(dtypes.int8, 24)),
    "immediate": ins(X86Ops.MOVi, dtypes.int64, GPR[13], UOp.const(dtypes.int32, -7)),
    "memory_write_disp32": ins(X86Ops.MOVm, dtypes.void, None, reg(dtypes.uint64, GPR[12]), reg(dtypes.int64, GPR[10]),
                                      UOp.const(dtypes.int32, 1024), reg(dtypes.int64, GPR[9])),
    "vex2": ins(X86Ops.VADDPS, dtypes.float32.vec(4), XMM[2], reg(dtypes.float32.vec(4), XMM[1]), reg(dtypes.float32.vec(4), XMM[3])),
    "vex3": ins(X86Ops.VADDPS, dtypes.float32.vec(8), XMM[8], reg(dtypes.float32.vec(8), XMM[9]), reg(dtypes.float32.vec(8), XMM[10])),
    "vex3_imm": ins(X86Ops.VPINSRQ, dtypes.int64.vec(2), XMM[8], reg(dtypes.int64.vec(2), XMM[9]), reg(dtypes.int64, GPR[10]),
                       UOp.const(dtypes.uint8, 1)),
  }
  expected = {"legacy_rex":"488bc3", "extended_registers":"458bca", "sib_disp8":"4f8b4cd418", "immediate":"49c7c5f9ffffff",
              "memory_write_disp32":"4f898cd400040000", "vex2":"c5f058d3", "vex3":"c4413458c2", "vex3_imm":"c443b122c201"}
  assert {name: encodings[u.arg](u).hex() for name,u in cases.items()} == expected

def test_x86_jump_fixup_byte_fixture():
  jump = UOp(Ops.INS, arg=X86Ops.JNE, tag="done")
  ret, label = UOp(Ops.INS, arg=X86Ops.RET), UOp(Ops.INS, arg=X86Ops.LABEL, tag="done")
  assert object.__new__(X86Renderer).render([jump, ret, label, ret]) == "0f8501000000c3c3"
