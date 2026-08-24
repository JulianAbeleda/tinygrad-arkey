import ast, inspect, textwrap

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import Ops, UOp
from tinygrad.llm.decode_routes import _Q6KDecodeCandidate
from tinygrad.llm.model_route_plan import decode_q6k_ffn_down_fp16_geometry_promoted
from tinygrad.llm.q6k_ffn_down_mmvq import (K, ROWS, Q6KFFNDownMMVQAdmission,
  emit_q6k_four_warp_fp16_direct, q6k_ffn_down_mmvq_call)
from tinygrad.llm.qk_layout import Q6K_HALFWORDS_PER_BLOCK


def test_production_call_is_closed_without_explicit_admission():
  assert q6k_ffn_down_mmvq_call(None, None, None, None, {}) is None
  try:
    Q6KFFNDownMMVQAdmission(-1)
  except ValueError:
    pass
  else:
    raise AssertionError("negative block admission must fail closed")
  try:
    Q6KFFNDownMMVQAdmission(0, fp16_fma=1)
  except ValueError:
    pass
  else:
    raise AssertionError("fp16_fma admission must require an explicit bool")
  try:
    Q6KFFNDownMMVQAdmission(0, rows_per_block=3)
  except ValueError:
    pass
  else:
    raise AssertionError("unsupported row packing must fail closed")


def test_default_decode_route_import_is_strictly_behind_explicit_lease_guard():
  fn = ast.parse(textwrap.dedent(inspect.getsource(_Q6KDecodeCandidate.execute))).body[0]
  guarded = [node for node in fn.body if isinstance(node, ast.If)
             and "_q6k_ffn_down_mmvq_admission" in ast.unparse(node.test)]
  assert len(guarded) == 1
  assert any(isinstance(node, ast.ImportFrom) and node.module == "tinygrad.llm.q6k_ffn_down_mmvq"
             for node in ast.walk(guarded[0]))
  assert not any(isinstance(node, ast.ImportFrom) and node.module == "tinygrad.llm.q6k_ffn_down_mmvq"
                 for node in fn.body)


def test_four_warp_emitter_renders_requested_rows_per_block():
  halfs_words = ROWS * (K // 256) * Q6K_HALFWORDS_PER_BLOCK
  for rows_per_block in (1, 2, 4):
    ast = emit_q6k_four_warp_fp16_direct(rows_per_block=rows_per_block)(
      UOp.placeholder((ROWS,), dtypes.float32, 0),
      UOp.placeholder((halfs_words,), dtypes.uint16, 1),
      UOp.placeholder((K,), dtypes.float16, 2),
      UOp.placeholder((ROWS,), dtypes.float32, 3))
    program = to_program(ast, CUDARenderer(Target.parse("NV:CUDA:sm_120")))
    source = next(u.arg for u in program.src if u.op is Ops.SOURCE)
    ptx = NVRTCCompiler("sm_120", ptx=True, cache_key=f"q6k_ffn_down_fp16_direct_rpb{rows_per_block}_v1").compile(source).decode()
    assert program.arg.global_size == (ROWS // rows_per_block, 1, 1)
    assert program.arg.local_size == (128 * rows_per_block, 1, 1)
    name = ("q6k_fp16_mmvq_direct_4096_12288_epi_ffnresadd" if rows_per_block == 1 else
            f"q6k_fp16_mmvq_direct_rpb{rows_per_block}_4096_12288_epi_ffnresadd")
    assert name in source
    assert "__shfl_xor_sync" in source and "__syncthreads" in source
    assert "st.global" in ptx and "shfl.sync" in ptx

  try: emit_q6k_four_warp_fp16_direct(rows_per_block=3)
  except ValueError: pass
  else: raise AssertionError("unsupported rows-per-block geometry must fail closed")


def test_route_policy_promotes_nv_sm120_only():
  # Promoted on NV sm_120 after the capture-safe prune fix restored the decode
  # wall and the re-bracket measured -39.0 us/token (+0.79%), token-exact.
  assert decode_q6k_ffn_down_fp16_geometry_promoted(("NV", "sm_120"))
  assert not decode_q6k_ffn_down_fp16_geometry_promoted(("AMD", "gfx1100"))
  assert not decode_q6k_ffn_down_fp16_geometry_promoted(("CUDA", "sm_120"))
