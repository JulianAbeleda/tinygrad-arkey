"""TG1: cross-lane XOR shuffle is a renderer-lowered operation (docs/task_workflow/input/
target-capability-policy-decoupling-scope-20260730.md). These pin the three acceptance facts: AMD's rendered
source is unchanged, Metal gets its own native intrinsic with no leftover AMD/lane arithmetic, and a renderer
that declares no provider fails loudly at lowering instead of silently emitting AMD text on the wrong target.
"""
import pytest

from tinygrad import Tensor, dtypes
from tinygrad.helpers import Target
from tinygrad.codegen import to_program
from tinygrad.renderer.cstyle import ClangRenderer, HIPRenderer, MetalRenderer
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops, UOp, graph_rewrite
from tinygrad.codegen.late.warp_reduce import WARP, warp_shfl_xor, pm_lower_warp_shfl_xor, warp_bpermute, pm_lower_warp_bpermute
from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_kernel

# Pre-TG1 literal (tinygrad/codegen/late/warp_reduce.py:22-27 before this change). AMD must render byte-identical
# text through the new provider seam -- this string is the pin, not a re-derivation of it.
_ORIGINAL_AMD_TEMPLATE = "__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute({1}, __builtin_bit_cast(int, {0})))"


def _kernel_ast(device:str):
  rows, k = 32, 1024
  out = Tensor.empty(rows, dtype=dtypes.float32, device=device)
  words = Tensor.empty(rows * (k // 256) * 36, dtype=dtypes.uint32, device=device)
  x = Tensor.empty(k, dtype=dtypes.float32, device=device)
  res = out.uop_program(words, x, fxn=q4k_g3_lanemap_gemv_kernel(rows, k))
  calls = res[0].schedule_linear().src
  assert len(calls) == 1
  return calls[0].src[0]


def _rendered_source(device:str, renderer) -> str:
  prog = to_program(_kernel_ast(device), renderer)
  return next(u.arg for u in prog.src if u.op is Ops.SOURCE)


def test_amd_provider_reproduces_the_pre_tg1_literal_byte_for_byte():
  """The AMD provider (HIPRenderer.warp_shfl_xor) must build the exact same UOp shape the old inline
  string did: same template text, same (val, idx-cast-int) sources -- not just equivalent-looking text."""
  lane = UOp.special(WARP, "lidx0")
  val = lane.cast(dtypes.float32)
  tagged = warp_shfl_xor(val, 16, lane)
  ren = HIPRenderer(Target.parse("AMD:HIP:gfx1100"))
  lowered = graph_rewrite(tagged, pm_lower_warp_shfl_xor, ctx=ren)
  assert lowered.op is Ops.CUSTOMI
  assert lowered.arg == _ORIGINAL_AMD_TEMPLATE
  assert lowered.src[0] is val
  assert lowered.src[1].op is Ops.CAST and lowered.src[1].dtype == dtypes.int


def test_amd_kernel_source_is_byte_identical_to_pre_tg1():
  """Full-kernel rendered-source equality (scope section 8: this is the only AMD verification possible
  without hardware -- there is no AMD GPU on this machine, so this can never be an execution check)."""
  ren = HIPRenderer(Target.parse("AMD:HIP:gfx1100"))
  src = _rendered_source("AMD", ren)
  assert src.count("__builtin_amdgcn_ds_bpermute") == 5
  assert _ORIGINAL_AMD_TEMPLATE.format("{0}", "{1}") not in src  # sanity: template must be filled in, not verbatim
  # the exact literal call form the pre-TG1 code produced for this ladder step
  assert "__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute(" in src


def test_metal_kernel_source_has_simd_shuffle_xor_and_no_amd_builtin_or_dead_lane_arithmetic():
  ren = MetalRenderer(Target.parse("METAL:METAL:Apple9"))
  src = _rendered_source("METAL", ren)
  assert src.count("simd_shuffle_xor(") == 5
  assert "amdgcn" not in src and "ds_bpermute" not in src
  # Metal's simd_shuffle_xor takes a mask, not a computed source-lane address -- no AMD-style
  # "(lane ^ offset) * 4" byte-address arithmetic should appear anywhere in the emitted source.
  assert "*4" not in src.replace(" ", "") and "* 4" not in src


def test_cuda_provider_uses_a_lane_mask_like_metal_not_an_amd_address():
  """CUDA's __shfl_xor_sync also takes a lane mask directly (like Metal), so it needs no per-lane address
  either. Call the provider directly -- CUDARenderer.__init__ requires a real NVRTC library, unavailable on
  this machine, but the provider callable itself needs no hardware."""
  lane = UOp.special(WARP, "lidx0")
  val = lane.cast(dtypes.float32)
  lowered = CUDARenderer.warp_shfl_xor(val, 16, lane)
  assert lowered.op is Ops.CUSTOMI and lowered.src == (val,)
  assert lowered.arg == "__shfl_xor_sync(0xffffffffu, {0}, 16)"


def test_unprovided_renderer_raises_naming_the_op_and_the_target():
  """CPU provides an identity warp_shfl_xor (LOCAL ranges serialize to loops), but the
  byte-address warp_bpermute stays unprovided and must fail loudly at lowering, naming
  both the operation and the target -- never render another target's text silently."""
  ren = ClangRenderer(Target.parse("CPU:CLANG:x86_64,znver2"))
  lane = UOp.special(WARP, "lidx0")
  val = lane.cast(dtypes.float32)
  assert graph_rewrite(warp_shfl_xor(val, 16, lane), pm_lower_warp_shfl_xor, ctx=ren) is val
  tagged = _attention_style_bpermute()
  with pytest.raises(NotImplementedError, match="warp_bpermute"):
    graph_rewrite(tagged, pm_lower_warp_bpermute, ctx=ren)
  # the message must also name the target, not just the operation, so a failure is actionable
  try:
    graph_rewrite(tagged, pm_lower_warp_bpermute, ctx=ren)
    assert False, "expected NotImplementedError"
  except NotImplementedError as e:
    assert "ClangRenderer" in str(e) and "CPU" in str(e)


def test_unprovided_renderer_raises_does_not_dispatch_on_device_default():
  """Regression guard for scope requirement 3.2/TG1: the resolution must key off the renderer instance
  passed as `ctx`, never `Device.DEFAULT` (a prior diagnostic did `Device.DEFAULT == 'METAL'`, which breaks
  under multi-device). Flip Device.DEFAULT to METAL while resolving against a CPU renderer -- it must still
  raise for CPU, proving the decision reads `ctx`, not the process-global default device."""
  from tinygrad.helpers import Context
  ren = ClangRenderer(Target.parse("CPU:CLANG:x86_64,znver2"))
  tagged = _attention_style_bpermute()
  with Context(DEV="METAL"):
    from tinygrad import Device
    assert Device.DEFAULT == "METAL"
    with pytest.raises(NotImplementedError, match="warp_bpermute"):
      graph_rewrite(tagged, pm_lower_warp_bpermute, ctx=ren)


# --- warp_bpermute (fused-attention row-softmax byte-address permute, P1 of the NV port scope) ---

_HIP_BPERMUTE_F32 = "__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute({0}, __builtin_bit_cast(unsigned int, {1})))"


def _attention_style_bpermute():
  """The exact shape the row-softmax lowering builds: byte address (lane^offset)*4 first, fp32 value second."""
  lane = UOp.special(WARP, "lidx0")
  addr = lane.cast(dtypes.int).alu(Ops.XOR, UOp.const(dtypes.int, 1)).alu(Ops.MUL, UOp.const(dtypes.int, 4))
  value = lane.cast(dtypes.float32)
  return warp_bpermute(addr, value)


def test_hip_warp_bpermute_preserves_the_pinned_attention_literal():
  """The attention bpermute is a tagged CUSTOMI at authoring time; HIP resolves it to the exact pre-tag
  literal -- `unsigned int` value cast, (addr, value) sources -- which is what the four-hash AMD control
  pins. A spelling drift here is a byte-identity regression, not a style point."""
  ren = HIPRenderer(Target.parse("AMD:HIP:gfx1100"))
  lowered = graph_rewrite(_attention_style_bpermute(), pm_lower_warp_bpermute, ctx=ren)
  assert lowered.op is Ops.CUSTOMI
  assert lowered.arg == _HIP_BPERMUTE_F32
  assert lowered.src[0].op is Ops.MUL and lowered.src[0].dtype == dtypes.int   # the byte address
  assert lowered.src[1].op is Ops.CAST and lowered.src[1].dtype == dtypes.float32  # the value


def test_cuda_warp_bpermute_renders_shfl_sync_from_the_byte_address():
  """CUDA has no register-by-address read; the byte address IS the source lane index times four, so
  `__shfl_sync(0xffffffffu, value, addr >> 2)` is the exact, generic translation -- correct for both the
  XOR butterfly and the half-wave broadcast the attention ABI builds."""
  addr, value = _attention_style_bpermute().src
  lowered = CUDARenderer.warp_bpermute(addr, value)
  assert lowered.op is Ops.CUSTOMI and lowered.src == (addr, value)
  assert lowered.arg == "__shfl_sync(0xffffffffu, {1}, (({0}) >> 2))"


def test_isa_warp_bpermute_resolves_to_the_f2_marker():
  """The AMD ISA renderer resolves the tag back to the Phase F.2 marker its isel consumes
  (isel_customi arg == "bpermute", src=(addr, data)) -- the encoder path is untouched byte-for-byte."""
  from tinygrad.renderer.isa.amd import AMDISARenderer
  addr, value = _attention_style_bpermute().src
  lowered = AMDISARenderer.warp_bpermute(addr, value)
  assert lowered.op is Ops.CUSTOMI and lowered.arg == "bpermute" and lowered.src == (addr, value)


def test_unprovided_warp_bpermute_renderer_raises_naming_the_op_and_the_target():
  """Same fail-loud contract as warp_shfl_xor: a renderer with no warp_bpermute provider raises at
  lowering, naming both the operation and the target, instead of rendering another target's text."""
  ren = ClangRenderer(Target.parse("CPU:CLANG:x86_64,znver2"))
  assert getattr(ClangRenderer, "warp_bpermute", None) is None
  with pytest.raises(NotImplementedError, match="warp_bpermute"):
    graph_rewrite(_attention_style_bpermute(), pm_lower_warp_bpermute, ctx=ren)
