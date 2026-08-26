from tinygrad import dtypes
from tinygrad.codegen import full_rewrite_to_sink, line_rewrite, pm_linearize_cleanups
from tinygrad.codegen.late.linearizer import linearize
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Target
from tinygrad.llm.flash_decode_attention import flash_single_stage_d512_kernel, flash_vec_llama_score_pv_kernel
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops, UOp


def _ast(output_fp16=True):
  out = UOp.placeholder((32*128,), dtypes.float16 if output_fp16 else dtypes.float32, 0)
  q = UOp.placeholder((32*128,), dtypes.float16, 1)
  cache = UOp.placeholder((2,1,8,4608,128), dtypes.float16, 2)
  return flash_single_stage_d512_kernel(128, 32, 8, 128, UOp.const(dtypes.int, 513), output_fp16=output_fp16)(out, q, cache)


def _render(ast, renderer):
  sink = full_rewrite_to_sink(ast, renderer)
  return renderer.render(line_rewrite(linearize(sink), pm_linearize_cleanups))


def test_single_stage_ast_has_no_global_partial_abi_and_fixed_local_resources():
  ast = _ast()
  assert ast.arg.name == "flash_single_stage_d512_f16_32_128"
  params = [u for u in ast.toposort() if u.op is Ops.PARAM]
  assert len(params) == 3  # out, q, cache: no pout argument
  locals_ = [u for u in ast.toposort() if u.op is Ops.DEFINE_LOCAL]
  assert all(u.dtype.addrspace is AddrSpace.LOCAL for u in locals_)
  assert sorted(u.dtype.size for u in locals_) == [2080, 8192, 8192]


def test_single_stage_cuda_and_hip_render_topology():
  cuda = _render(_ast(), CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=True))
  hip = _render(_ast(), HIPRenderer(Target.parse("AMD:HIP:gfx1100")))
  for src in (cuda, hip):
    assert "flash_single_stage_d512_f16_32_128" in src
    assert src.count("syncthreads") >= 3 or src.count("barrier") >= 3
    assert "buf2[8192]" in src and "buf3[8192]" in src and "buf7[2080]" in src
  assert "__launch_bounds__(512)" in cuda
  assert "threadIdx.z; /* 4 */" in cuda and "threadIdx.y; /* 4 */" in cuda and "threadIdx.x; /* 32 */" in cuda


def test_single_stage_shape_is_closed_to_exact_candidate():
  try:
    flash_single_stage_d512_kernel(128, 40, 8, 128, UOp.const(dtypes.int, 513))
  except ValueError as e:
    assert "fixed" in str(e)
  else:
    raise AssertionError("non-candidate shape must fail closed")


def _vec_ast():
  pout = UOp.placeholder((32*4*130,), dtypes.float32, 0)
  q = UOp.placeholder((32*128,), dtypes.float16, 1)
  cache = UOp.placeholder((2,1,8,4608,128), dtypes.float16, 2)
  return flash_vec_llama_score_pv_kernel(128, 32, 8, 4608, 4, UOp.const(dtypes.int, 513))(pout, q, cache)


def test_vec_llama_score_pv_is_closed_and_keeps_partial_abi():
  ast = _vec_ast()
  assert ast.arg.name == "flash_vec_llama_score_pv_32_128_4"
  params = [u for u in ast.toposort() if u.op is Ops.PARAM]
  assert len(params) == 3  # pout, q, cache: the legacy partial ABI, not a direct out
  locals_ = [u for u in ast.toposort() if u.op is Ops.DEFINE_LOCAL]
  assert sorted(u.dtype.size for u in locals_) == [4, 4, 512]


def test_vec_llama_score_pv_cuda_and_hip_render_topology():
  cuda = _render(_vec_ast(), CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=True))
  hip = _render(_vec_ast(), HIPRenderer(Target.parse("AMD:HIP:gfx1100")))
  for src in (cuda, hip):
    assert "flash_vec_llama_score_pv_32_128_4" in src
  # Single-pass substrate: the only barrier is the final cross-warp combine, not per-tile K/V staging.
  assert cuda.count("__syncthreads") == 1
  assert hip.count("__builtin_amdgcn_s_barrier") == 1
  assert "__launch_bounds__(128)" in cuda
  # Lane stays along threadIdx.x (32) so __shfl_xor_sync addresses the 8-lane groups correctly.
  assert "threadIdx.x; /* 32 */" in cuda and "threadIdx.y; /* 4 */" in cuda
  assert cuda.count("__shfl_xor_sync") >= 9
  assert hip.count("__builtin_amdgcn_ds_bpermute") >= 9


def test_vec_llama_score_pv_shape_is_closed_to_exact_candidate():
  try:
    flash_vec_llama_score_pv_kernel(128, 40, 8, 4608, 4, UOp.const(dtypes.int, 513))
  except ValueError as e:
    assert "fixed" in str(e)
  else:
    raise AssertionError("non-candidate shape must fail closed")


def test_vec_llama_wide_kv_renders_llama_16_byte_copy_grammar():
  pout = UOp.placeholder((32*6*130,), dtypes.float32, 0)
  # The research boundary supplies zero-copy uint32 views: every uint4 load is
  # one aligned 16-byte cooperative copy covering eight adjacent fp16 values.
  q = UOp.placeholder((32*64,), dtypes.uint32, 1)
  cache = UOp.placeholder((2,1,8,768,64), dtypes.uint32, 2)
  ast = flash_vec_llama_score_pv_kernel(128, 32, 8, 768, 6, UOp.const(dtypes.int, 513), wide_kv=True)(pout, q, cache)
  cuda = _render(ast, CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=True))
  assert ast.arg.name == "flash_vec_llama_score_pv_32_128_6_widekv16"
  assert cuda.count("uint4") >= 6
  assert "half4" not in cuda and "half8" not in cuda
  assert cuda.count("__syncthreads") == 1
