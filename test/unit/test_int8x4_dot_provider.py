import pytest

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.codegen.late.int8_dot import int8x4_dot, pm_lower_int8x4_dot
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import ClangRenderer, HIPRenderer
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import KernelInfo, Ops, UOp, graph_rewrite

def _dot_ast(a:int=0x7f80ff01, b:int=0x02ff0180, acc:int=17):
  out = UOp.placeholder((1,), dtypes.int32, 0)
  val = int8x4_dot(UOp.const(dtypes.int32, acc), UOp.const(dtypes.uint32, a), UOp.const(dtypes.uint32, b))
  return out[0].store(val).sink(arg=KernelInfo(name="int8x4_dot_probe", opts_to_apply=()))

def _source(ast, ren):
  prog = to_program(ast, ren)
  return next(u.arg for u in prog.src if u.op is Ops.SOURCE)

def test_int8x4_dot_type_and_signed_lane_contract():
  # 1*-128 + -1*1 + -128*-1 + 127*2 + 17 = 270
  a, b, acc = 0x7f80ff01, 0x02ff0180, 17
  expected = acc + sum(int.from_bytes(bytes([(a >> (8*i)) & 255]), "little", signed=True) *
                       int.from_bytes(bytes([(b >> (8*i)) & 255]), "little", signed=True) for i in range(4))
  assert expected == 270
  tagged = int8x4_dot(UOp.const(dtypes.int32, acc), UOp.const(dtypes.uint32, a), UOp.const(dtypes.uint32, b))
  assert tagged.dtype == dtypes.int32 and tagged.arg == ("int8x4_dot",)
  with pytest.raises(TypeError): int8x4_dot(tagged.src[0], tagged.src[0], tagged.src[2])

def test_cuda_and_amd_providers_pin_signed_dot_semantics():
  tagged = next(u for u in _dot_ast().toposort() if u.op is Ops.CUSTOMI)
  cuda = CUDARenderer.__new__(CUDARenderer); cuda.target = Target.parse("NV:CUDA:sm_120")
  cu = graph_rewrite(tagged, pm_lower_int8x4_dot, ctx=cuda)
  assert cu.arg == "__dp4a((int){1}, (int){2}, {0})"
  amd = graph_rewrite(tagged, pm_lower_int8x4_dot, ctx=HIPRenderer(Target.parse("AMD:HIP:gfx1100")))
  assert amd.arg == "__builtin_amdgcn_sdot4({1}, {2}, {0}, false)"

def test_unprovided_target_fails_loudly():
  tagged = next(u for u in _dot_ast().toposort() if u.op is Ops.CUSTOMI)
  with pytest.raises(NotImplementedError, match="int8x4_dot.*ClangRenderer"):
    graph_rewrite(tagged, pm_lower_int8x4_dot, ctx=ClangRenderer(Target.parse("CPU:CLANG:x86_64,znver2")))

def test_sm120_source_compiles_to_dp4a_class_instruction():
  ren = CUDARenderer(Target.parse("NV:CUDA:sm_120"))
  src = _source(_dot_ast(), ren)
  assert "__dp4a(" in src and "amdgcn" not in src
  ptx = NVRTCCompiler("sm_120", ptx=True, cache_key="int8x4_dot_test").compile(src).decode()
  assert "dp4a.s32.s32" in ptx
