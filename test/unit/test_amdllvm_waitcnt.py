import pytest

from tinygrad.codegen.opt.compiler_policies import WaitCount
from tinygrad.dtype import dtypes
from tinygrad.helpers import Target
from tinygrad.renderer.llvmir import AMDLLVMRenderer
from tinygrad.uop.ops import Ops, UOp


def _wait(count):
  return UOp(Ops.WAIT, dtypes.void, (), count)


def test_wait_count_is_typed_and_packs_amd_sopp_fields():
  assert WaitCount(vmcnt=8, lgkmcnt=63, expcnt=7).simm16 == (8 << 10) | (63 << 4) | 7
  with pytest.raises(ValueError): WaitCount(vmcnt=64)
  with pytest.raises(ValueError): WaitCount(lgkmcnt=-1)
  with pytest.raises(ValueError): WaitCount(expcnt=8)


def test_amdllvm_wait_lowers_to_intrinsic_without_raw_instruction_uop():
  renderer = AMDLLVMRenderer(Target.parse("AMD:LLVM:gfx1100"))
  source = renderer.render([_wait(WaitCount(vmcnt=8))])
  assert "call void @llvm.amdgcn.s.waitcnt(i32 9207)" in source
  assert "Ops.INS" not in source


def test_amdllvm_wait_intrinsic_compiles_to_object():
  renderer = AMDLLVMRenderer(Target.parse("AMD:LLVM:gfx1100"))
  source = renderer.render([_wait(WaitCount(vmcnt=8))])
  assert renderer.compiler.compile_to_obj(source)


def test_amdllvm_wait_rejects_untyped_payload():
  renderer = AMDLLVMRenderer(Target.parse("AMD:LLVM:gfx1100"))
  with pytest.raises(ValueError, match="typed WaitCount"):
    renderer.render([_wait(None)])
