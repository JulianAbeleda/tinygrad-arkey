from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import Ops, UOp
from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_kernel
from extra.llm_research.decode.q4k_warp_ownership_static import (
  K, ROWS, cooperative_ownership_coordinates, emit_q4k_warp_cooperative_partial,
  flat_cooperative_ownership_coordinates, installed_ownership_coordinates)

def _elements(rows):
  return {(block*256 + group*32 + word*4 + i) for _, _, block, group, word in rows for i in range(4)}

def test_q4_installed_vs_cooperative_exact_coverage_and_work_per_lane():
  old, new = installed_ownership_coordinates(), cooperative_ownership_coordinates()
  assert len(old) == 32*4*8 and len(new) == 4*32*4*2
  assert _elements(old) == set(range(K)) == _elements(new)
  assert {len([x for x in old if x[1] == lane]) for lane in range(32)} == {32}
  assert {len([x for x in new if x[0] == warp and x[1] == lane]) for warp in range(4) for lane in range(32)} == {8}

def test_q4_flat_local_spelling_and_warp_stripes_are_exact():
  rows = cooperative_ownership_coordinates()
  assert rows == flat_cooperative_ownership_coordinates()
  for warp in range(4): assert {x[2] for x in rows if x[0] == warp} == set(range(warp*4, warp*4+4))

def test_q4_coop_static_sm120_render_has_one_128_local_axis_and_local_rendezvous():
  out = UOp.placeholder((ROWS, 4), dtypes.float32, 0)
  words = UOp.placeholder((ROWS*(K//256)*36,), dtypes.uint32, 1)
  x = UOp.placeholder((K,), dtypes.float16, 2)
  ast = emit_q4k_warp_cooperative_partial()(out, words, x)
  prog = to_program(ast, CUDARenderer(Target.parse("NV:CUDA:sm_120")))
  src = next(u.arg for u in prog.src if u.op is Ops.SOURCE)
  assert prog.arg.local_size == (128, 1, 1)
  assert "lidx0" in src and "__shfl_xor_sync" in src
  assert "q4k_warp_coop_partial_4096_4096" in src

def test_q4_coop_static_ptx_exposes_the_remaining_lowering_problem():
  """The ownership is different, but the naïve static group select is not a GPU candidate.

  It renders the same 40 static global-load instructions *per thread* as the
  installed 32-thread body.  At 128 threads/output that would multiply dynamic
  issue traffic; a real follow-up needs dynamic packed-address/lane ownership,
  not this control-masked helper expansion.
  """
  ren = CUDARenderer(Target.parse("NV:CUDA:sm_120"))
  def ptx(fn, out_shape, key):
    out = UOp.placeholder(out_shape, dtypes.float32, 0)
    words = UOp.placeholder((ROWS*(K//256)*36,), dtypes.uint32, 1)
    x = UOp.placeholder((K,), dtypes.float16, 2)
    prog = to_program(fn(out, words, x), ren)
    src = next(u.arg for u in prog.src if u.op is Ops.SOURCE)
    return prog.arg.local_size, NVRTCCompiler("sm_120", ptx=True, cache_key=key).compile(src).decode()
  old_shape, old = ptx(q4k_g3_lanemap_gemv_kernel(ROWS, K), (ROWS,), "q4k_installed_ownership_static_v1")
  new_shape, new = ptx(emit_q4k_warp_cooperative_partial(), (ROWS, 4), "q4k_coop_ownership_static_v1")
  assert old_shape == (32, 1, 1) and new_shape == (128, 1, 1)
  assert old.count("ld.global") == new.count("ld.global") == 40
  assert old.count("shfl.sync") == new.count("shfl.sync") == 5
