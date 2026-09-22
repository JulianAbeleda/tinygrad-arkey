from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import Ops, UOp
from extra.llm_research.decode.q4k_warp_cooperative_dynamic import (
  K, ROWS, dynamic_ownership_coordinates, emit_q4k_warp_cooperative_q8_partial,
  emit_q4k_warp_cooperative_q8_partial_runtime_blocks)
from extra.llm_research.decode.route_class_numerics import _make_q4k_words


def test_dynamic_q4_ownership_is_complete_without_duplicate_logical_words():
  got = dynamic_ownership_coordinates()
  assert len(got) == 4*32*4*2 == len(set(got))
  for warp in range(4):
    for block in range(warp*4, warp*4+4):
      words = {(group, word) for w, _, b, group, word in got if w == warp and b == block}
      assert words == {(group, word) for group in range(8) for word in range(8)}
  assert {len([x for x in got if x[0] == w and x[1] == lane and x[2] == block])
          for w in range(4) for lane in range(32) for block in range(w*4,w*4+4)} == {2}


def test_dynamic_q4_header_and_nibbles_match_packed_oracle_all_16_blocks():
  """Independent byte-layout oracle for every group/word/value in one K row."""
  words, raw = _make_q4k_words(1, K, 20260805)
  blocks = raw.reshape(K//256, 144)
  for block in range(K//256):
    base=block*36; rb=blocks[block]
    w1,w2,w3=(int(words[base+i]) for i in (1,2,3))
    for group in range(8):
      g4=group%4; b1=(w1>>(g4*8))&0xff; b2=(w2>>(g4*8))&0xff; hb=(w3>>(g4*8))&0xff
      dynamic_sc=(b1&63) if group<4 else (hb&0xf)|((b1>>6)<<4)
      dynamic_mn=(b2&63) if group<4 else (hb>>4)|((b2>>6)<<4)
      oracle_sc=(int(rb[4+group])&63) if group<4 else (int(rb[12+g4])&0xf)|((int(rb[4+g4])>>6)<<4)
      oracle_mn=(int(rb[8+group])&63) if group<4 else (int(rb[12+g4])>>4)|((int(rb[8+g4])>>6)<<4)
      assert (dynamic_sc,dynamic_mn)==(oracle_sc,oracle_mn)
      for word in range(8):
        qword=int(words[base+4+(group//2)*8+word])
        packed=(qword>>((group%2)*4))&0x0F0F0F0F
        for nib in range(4):
          dynamic_q=(packed>>(nib*8))&0xf
          oracle_q=(int(rb[16+(group//2)*32+word*4+nib])>>((group%2)*4))&0xf
          assert dynamic_q==oracle_q


def _ptx(fn, out_shape, key):
  out = UOp.placeholder(out_shape, dtypes.float32, 0)
  words = UOp.placeholder((ROWS*(K//256)*36,), dtypes.uint32, 1)
  xp = UOp.placeholder((K//4+K//32,), dtypes.uint32, 2)
  prog = to_program(fn(out, words, xp), CUDARenderer(Target.parse("NV:CUDA:sm_120")))
  src = next(u.arg for u in prog.src if u.op is Ops.SOURCE)
  return prog, NVRTCCompiler("sm_120", ptx=True, cache_key=key).compile(src).decode()


def test_dynamic_q4_static_render_exposes_flat_local_dynamic_loads_and_dp4a():
  prog, ptx = _ptx(emit_q4k_warp_cooperative_q8_partial(), (ROWS,4), "q4k_coop_dynamic_q8_v1")
  assert prog.arg.local_size == (128, 1, 1)
  # Dynamic word/group addressing must survive source lowering; unlike v1 it
  # has no eight static group bodies. CUDA lowers CUSTOMI to the DP4A contract.
  assert "dp4a" in ptx
  # NVRTC currently unrolls the four fixed blocks, so this is an auditable
  # *failure* of the pre-GPU body-wide load cap (36, versus required <=10).
  # Keep the exact number pinned: changing it requires rerunning the gate.
  assert ptx.count("ld.global") == 36
  assert ptx.count("dp4a") == 16
  assert ".local " not in ptx


def test_runtime_scalar_loop_bound_survives_nvrtc_without_source_pragma():
  """An unbound DEFINE_VAR becomes a scalar parameter and prevents cloning.

  This proves the generic IR capability separately from the stricter Q4
  register-budget gate. Runtime execution would bind this extent to four.
  """
  prog, ptx = _ptx(emit_q4k_warp_cooperative_q8_partial_runtime_blocks(), (ROWS,4), "q4k_coop_dynamic_runtime_v1")
  assert prog.arg.local_size == (128, 1, 1)
  assert len(prog.arg.vars) == 1 and prog.arg.vars[0].arg == ("q4k_coop_blocks", 1, 4)
  assert ptx.count("ld.global") == 9 and ptx.count("dp4a") == 12 and ".local " not in ptx
  # The body is one runtime loop (rather than four cloned block bodies), but
  # The corrected byte-stride header body uses 101 virtual b32 names. Physical
  # allocation is separately established by ptxas, not this namespace size.
  assert ".reg .b32 \t%r<101>;" in ptx
