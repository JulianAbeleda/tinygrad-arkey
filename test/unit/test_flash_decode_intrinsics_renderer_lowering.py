"""TG7: flash decode's two AMD-only intrinsics become renderer-lowered operations (docs/task_workflow/input/
target-capability-policy-decoupling-scope-20260730.md). Mirrors test_warp_shfl_xor_renderer_lowering.py's
(TG1) shape: pin that AMD's rendered source is byte-identical to the pre-TG7 literal, that Metal gets a real
native equivalent (and, since this machine has real Metal hardware, that it actually compiles -- a strictly
stronger check than TG1 could offer for its own AMD/Metal pair), and that an unprovided renderer fails loudly
at lowering rather than silently emitting the wrong target's text.
"""
import pytest

from tinygrad import dtypes
from tinygrad.helpers import Target
from tinygrad.codegen import to_program
from tinygrad.renderer.cstyle import ClangRenderer, HIPRenderer, MetalRenderer
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops, UOp, graph_rewrite
from tinygrad.codegen.late.flash_decode_intrinsics import fdot2, exp2f, pm_lower_flash_decode_intrinsics
from tinygrad.llm.flash_decode_attention import describe_flash_decode_attention

# Pre-TG7 literals (tinygrad/llm/flash_decode_attention.py:25 and :166 before this change). AMD must render
# byte-identical text through the new provider seam -- these strings are the pin, not a re-derivation of them.
_ORIGINAL_FDOT2_TEMPLATE = "__builtin_amdgcn_fdot2({1}, {2}, {0}, false)"
_ORIGINAL_EXP2F_TEMPLATE = "__builtin_amdgcn_exp2f({0})"


def _tile_ast(hq=32, hd=128, hkv=8, maxc=8192, split_count=48):
  spec = describe_flash_decode_attention(hq, hd, hkv, maxc, split_count)
  tc = UOp.variable("Tc", 0, maxc)
  pout = UOp.placeholder((hq * split_count * (hd + 2),), dtypes.float32, 0)
  q = UOp.placeholder((hq * hd,), dtypes.float16, 1)
  cache = UOp.placeholder((2, 1, hkv, maxc, hd), dtypes.float16, 2)
  return spec.emit_tile(tc)(pout, q, cache)


def _rendered_source(ast, renderer) -> str:
  prog = to_program(ast, renderer)
  return next(u.arg for u in prog.src if u.op is Ops.SOURCE)


# ---- Provider-level: exact byte-for-byte template reproduction --------------------------------------------

def test_amd_fdot2_provider_reproduces_the_pre_tg7_literal_byte_for_byte():
  acc = UOp.const(dtypes.float32, 0.0)
  a = UOp(Ops.STACK, dtypes.half.vec(2), (UOp.const(dtypes.half, 1.0), UOp.const(dtypes.half, 2.0)))
  b = UOp(Ops.STACK, dtypes.half.vec(2), (UOp.const(dtypes.half, 3.0), UOp.const(dtypes.half, 4.0)))
  tagged = fdot2(acc, a, b)
  ren = HIPRenderer(Target.parse("AMD:HIP:gfx1100"))
  lowered = graph_rewrite(tagged, pm_lower_flash_decode_intrinsics, ctx=ren)
  assert lowered.op is Ops.CUSTOMI
  assert lowered.arg == _ORIGINAL_FDOT2_TEMPLATE
  assert lowered.src == (acc, a, b)


def test_amd_exp2f_provider_reproduces_the_pre_tg7_literal_byte_for_byte():
  x = UOp.const(dtypes.float32, 1.5)
  tagged = exp2f(x)
  ren = HIPRenderer(Target.parse("AMD:HIP:gfx1100"))
  lowered = graph_rewrite(tagged, pm_lower_flash_decode_intrinsics, ctx=ren)
  assert lowered.op is Ops.CUSTOMI
  assert lowered.arg == _ORIGINAL_EXP2F_TEMPLATE
  assert lowered.src == (x,)


# ---- Full-kernel rendered-source checks --------------------------------------------------------------------

def test_amd_kernel_source_is_byte_identical_to_pre_tg7():
  """scope section 8: no AMD hardware here, so full-kernel AMD verification can only ever be rendered-source
  equality, never an execution result. The production tile kernel (flash_decode_attention.py) always emits
  exactly one fdot2 call site (the QK dot-product reduction loop is a real RANGE, not unrolled)."""
  ren = HIPRenderer(Target.parse("AMD:HIP:gfx1100"))
  src = _rendered_source(_tile_ast(), ren)
  assert src.count("__builtin_amdgcn_fdot2(") == 1
  assert _ORIGINAL_FDOT2_TEMPLATE.format("{0}", "{1}", "{2}") not in src  # sanity: filled in, not verbatim
  assert "__builtin_amdgcn_exp2f" not in src  # DECODE_FAST_EXP2 is off by default


def test_amd_kernel_source_has_fast_exp2_when_opted_in(monkeypatch):
  from tinygrad.helpers import getenv
  monkeypatch.setenv("DECODE_FAST_EXP2", "1")
  getenv.cache_clear()
  try:
    ren = HIPRenderer(Target.parse("AMD:HIP:gfx1100"))
    src = _rendered_source(_tile_ast(), ren)
    assert src.count("__builtin_amdgcn_exp2f(") == 2  # correction + probability, one call site each
  finally:
    getenv.cache_clear()


def test_metal_kernel_source_has_no_amd_builtin_and_compiles_on_real_metal_hardware():
  """This machine's Metal backend is real hardware (scope section 8 note: unlike AMD, this is not a
  structural proxy) -- compiling is a strictly stronger check than rendered-source equality alone."""
  ren = MetalRenderer(Target.parse("METAL:METAL:Apple9"))
  src = _rendered_source(_tile_ast(), ren)
  assert "amdgcn" not in src and "fdot2" not in src
  assert src.count("float(") >= 4  # the fp32-accumulate fdot2 substitute: 2 casts on a, 2 on b
  assert "exp2(" in src
  lib = ren.compiler.compile(src)
  assert len(lib) > 0


def test_metal_fdot2_semantics_are_fp32_accumulate_not_half2_dot():
  """Pin the exact substitute expression so a future edit can't silently swap in `dot(half2,half2)` (which is
  not guaranteed to accumulate in fp32 the way AMD's fdot2 does)."""
  acc = UOp.const(dtypes.float32, 0.0)
  a = UOp(Ops.STACK, dtypes.half.vec(2), (UOp.const(dtypes.half, 1.0), UOp.const(dtypes.half, 2.0)))
  b = UOp(Ops.STACK, dtypes.half.vec(2), (UOp.const(dtypes.half, 3.0), UOp.const(dtypes.half, 4.0)))
  ren = MetalRenderer(Target.parse("METAL:METAL:Apple9"))
  lowered = graph_rewrite(fdot2(acc, a, b), pm_lower_flash_decode_intrinsics, ctx=ren)
  assert lowered.arg == "({0}) + float({1}.x) * float({2}.x) + float({1}.y) * float({2}.y)"


# ---- Fail loud, never a silent fallback --------------------------------------------------------------------

def test_cuda_has_exp2f_but_no_fdot2_one_liner():
  """TG1's rule for CUDA providers: a one-liner is fine, otherwise leave unprovided. exp2f is native; fdot2
  has no native packed-fp16x2 dot-accumulate CUDA builtin, so it stays unprovided and must raise."""
  assert CUDARenderer.exp2f is not None
  assert CUDARenderer.fdot2 is None
  x = UOp.const(dtypes.float32, 1.5)
  lowered = CUDARenderer.exp2f(x)
  assert lowered.op is Ops.CUSTOMI and lowered.arg == "exp2f({0})"
  acc = UOp.const(dtypes.float32, 0.0)
  a = UOp(Ops.STACK, dtypes.half.vec(2), (UOp.const(dtypes.half, 1.0), UOp.const(dtypes.half, 2.0)))
  b = UOp(Ops.STACK, dtypes.half.vec(2), (UOp.const(dtypes.half, 3.0), UOp.const(dtypes.half, 4.0)))
  ren = CUDARenderer.__new__(CUDARenderer)  # no NVRTC on this machine (scope section 8); provider needs no hardware
  ren.target = Target.parse("NV:CUDA:sm_80")  # __new__ skips __init__, so set the field the error message reads
  with pytest.raises(NotImplementedError, match="fdot2"):
    graph_rewrite(fdot2(acc, a, b), pm_lower_flash_decode_intrinsics, ctx=ren)


def test_unprovided_renderer_raises_naming_the_op_and_the_target():
  ren = ClangRenderer(Target.parse("CPU:CLANG:x86_64,znver2"))
  assert ClangRenderer.fdot2 is None and ClangRenderer.exp2f is None
  acc = UOp.const(dtypes.float32, 0.0)
  a = UOp(Ops.STACK, dtypes.half.vec(2), (UOp.const(dtypes.half, 1.0), UOp.const(dtypes.half, 2.0)))
  b = UOp(Ops.STACK, dtypes.half.vec(2), (UOp.const(dtypes.half, 3.0), UOp.const(dtypes.half, 4.0)))
  with pytest.raises(NotImplementedError, match="fdot2") as excinfo:
    graph_rewrite(fdot2(acc, a, b), pm_lower_flash_decode_intrinsics, ctx=ren)
  assert "ClangRenderer" in str(excinfo.value) and "CPU" in str(excinfo.value)
  with pytest.raises(NotImplementedError, match="exp2f") as excinfo2:
    graph_rewrite(exp2f(UOp.const(dtypes.float32, 1.0)), pm_lower_flash_decode_intrinsics, ctx=ren)
  assert "ClangRenderer" in str(excinfo2.value) and "CPU" in str(excinfo2.value)


def test_unprovided_renderer_raises_does_not_dispatch_on_device_default():
  """Regression guard for scope requirement 3.2/TG1, extended to TG7's own intrinsics: resolution must key
  off the renderer instance passed as `ctx`, never Device.DEFAULT."""
  from tinygrad.helpers import Context
  ren = ClangRenderer(Target.parse("CPU:CLANG:x86_64,znver2"))
  tagged = exp2f(UOp.const(dtypes.float32, 1.0))
  with Context(DEV="METAL"):
    from tinygrad import Device
    assert Device.DEFAULT == "METAL"
    with pytest.raises(NotImplementedError, match="exp2f"):
      graph_rewrite(tagged, pm_lower_flash_decode_intrinsics, ctx=ren)


# ---- Capability facts are derived from the provider, never restated (TG2 shape) ----------------------------

def test_capability_properties_are_derived_from_the_provider_not_restated():
  assert HIPRenderer(Target.parse("AMD:HIP:gfx1100")).supports_flash_decode_fdot2 is True
  assert HIPRenderer(Target.parse("AMD:HIP:gfx1100")).supports_flash_decode_exp2f is True
  assert MetalRenderer(Target.parse("METAL:METAL:Apple9")).supports_flash_decode_fdot2 is True
  assert MetalRenderer(Target.parse("METAL:METAL:Apple9")).supports_flash_decode_exp2f is True
  assert ClangRenderer(Target.parse("CPU:CLANG:x86_64,znver2")).supports_flash_decode_fdot2 is False
  assert ClangRenderer(Target.parse("CPU:CLANG:x86_64,znver2")).supports_flash_decode_exp2f is False
