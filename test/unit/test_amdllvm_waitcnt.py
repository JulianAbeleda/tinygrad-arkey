import pytest

from tinygrad.codegen import full_rewrite_to_sink
from tinygrad.codegen.opt.compiler_policies import WaitCount
from tinygrad.dtype import dtypes
from tinygrad.helpers import Context, Target
from tinygrad.renderer.llvmir import AMDLLVMRenderer
from tinygrad.runtime.ops_amd import _amd_renderers
from tinygrad.uop.ops import Ops, UOp


def _wait(count):
  return UOp(Ops.WAIT, dtypes.void, (), count)


def test_native_isa_renderer_is_registered_only_for_explicit_target():
  with Context(DEV="AMD"):
    assert [x.__name__ for x in _amd_renderers("gfx1100")] == ["HIPRenderer", "AMDLLVMRenderer", "HIPCCRenderer"]
  with Context(DEV="AMD:ISA"):
    assert [x.__name__ for x in _amd_renderers("gfx1100")][-1] == "AMDISARenderer"


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


def test_wait_count_node_survives_full_rewrite_spec_boundary():
  renderer = AMDLLVMRenderer(Target.parse("AMD:LLVM:gfx1100"))
  sink = UOp(Ops.SINK, dtypes.void, (_wait(WaitCount(vmcnt=8)),))
  rewritten = full_rewrite_to_sink(sink, renderer, optimize=False)
  assert rewritten.src[0].op is Ops.WAIT and rewritten.src[0].arg == WaitCount(vmcnt=8)


def test_wait_node_without_payload_fails_spec_boundary():
  renderer = AMDLLVMRenderer(Target.parse("AMD:LLVM:gfx1100"))
  sink = UOp(Ops.SINK, dtypes.void, (_wait(None),))
  with pytest.raises(RuntimeError, match="UOp verification failed"):
    full_rewrite_to_sink(sink, renderer, optimize=False)
