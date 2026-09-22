import ast, inspect, textwrap

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import Ops, UOp
from tinygrad.llm.decode_routes import _Q6KDecodeCandidate
from tinygrad.llm.model_route_plan import (decode_q6k_v_four_warp_fp16_geometry_promoted,
  decode_q6k_vocab_four_warp_fp16_promoted)
from tinygrad.llm.q6k_v_mmvq import (K, ROWS, Q6KVFourWarpAdmission,
  emit_q6k_v_four_warp_fp16_direct, q6k_v_four_warp_call)


def test_production_call_is_closed_without_explicit_admission():
  assert q6k_v_four_warp_call(None, None, None, None) is None
  try:
    Q6KVFourWarpAdmission(-1)
  except ValueError:
    pass
  else:
    raise AssertionError("negative block admission must fail closed")


def test_default_decode_route_import_is_strictly_behind_explicit_lease_guard():
  fn = ast.parse(textwrap.dedent(inspect.getsource(_Q6KDecodeCandidate.execute))).body[0]
  guarded = [node for node in fn.body if isinstance(node, ast.If)
             and "_q6k_v_four_warp_admission" in ast.unparse(node.test)]
  assert len(guarded) == 1
  assert any(isinstance(node, ast.ImportFrom) and node.module == "tinygrad.llm.q6k_v_mmvq"
             for node in ast.walk(guarded[0]))
  assert not any(isinstance(node, ast.ImportFrom) and node.module == "tinygrad.llm.q6k_v_mmvq"
                 for node in fn.body)


def test_four_warp_emitter_renders_one_row_per_block_and_128_threads():
  from tinygrad.llm.qk_layout import Q6K_HALFWORDS_PER_BLOCK
  halfs_words = ROWS * (K // 256) * Q6K_HALFWORDS_PER_BLOCK
  ast = emit_q6k_v_four_warp_fp16_direct()(
    UOp.placeholder((ROWS,), dtypes.float32, 0),
    UOp.placeholder((halfs_words,), dtypes.uint16, 1),
    UOp.placeholder((K,), dtypes.float16, 2))
  program = to_program(ast, CUDARenderer(Target.parse("NV:CUDA:sm_120")))
  source = next(u.arg for u in program.src if u.op is Ops.SOURCE)
  ptx = NVRTCCompiler("sm_120", ptx=True, cache_key="q6k_v_fp16_direct_v1").compile(source).decode()
  assert program.arg.global_size == (ROWS, 1, 1)
  assert program.arg.local_size == (128, 1, 1)
  assert "q6k_v_four_warp_fp16_direct_1024_4096" in source
  assert "__shfl_xor_sync" in source and "__syncthreads" in source
  assert "st.global" in ptx and "shfl.sync" in ptx


def test_route_policy_promotes_nv_sm120_only():
  # Promoted on NV sm_120 after the wall bracket measured -147.35 us/token
  # (+3.07%), token-exact across control A/candidate/control C.
  assert decode_q6k_v_four_warp_fp16_geometry_promoted(("NV", "sm_120"))
  assert not decode_q6k_v_four_warp_fp16_geometry_promoted(("AMD", "gfx1100"))
  assert not decode_q6k_v_four_warp_fp16_geometry_promoted(("CUDA", "sm_120"))

def test_vocab_route_policy_and_exact_geometry():
  rows = 151936
  assert decode_q6k_vocab_four_warp_fp16_promoted(("NV", "sm_120"))
  assert not decode_q6k_vocab_four_warp_fp16_promoted(("AMD", "gfx1100"))
  ast = emit_q6k_v_four_warp_fp16_direct(rows=rows,
    kernel_name=f"q6k_vocab_four_warp_fp16_direct_{rows}_{K}")(
      UOp.placeholder((rows,), dtypes.float32, 0),
      UOp.placeholder((rows * (K // 256) * 105,), dtypes.uint16, 1),
      UOp.placeholder((K,), dtypes.float16, 2))
  program = to_program(ast, CUDARenderer(Target.parse("NV:CUDA:sm_120")))
  source = next(u.arg for u in program.src if u.op is Ops.SOURCE)
  assert program.arg.global_size == (rows, 1, 1)
  assert program.arg.local_size == (128, 1, 1)
  assert f"q6k_vocab_four_warp_fp16_direct_{rows}_{K}" in source
