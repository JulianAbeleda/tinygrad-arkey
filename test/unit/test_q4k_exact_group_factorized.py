import numpy as np

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import Ops,UOp
from extra.llm_research.decode.q4k_exact_group_factorized import (
  emit_q4k_exact_four_warp,emit_q4k_exact_four_warp_runtime_blocks,emit_q4k_exact_group_factorized,oracle_gemv,unpack_q4k_row)
from extra.llm_research.decode.route_class_numerics import _make_q4k_words


def _render(fn,rows=4096,k=4096):
  out=UOp.placeholder((rows,),dtypes.float32,0)
  words=UOp.placeholder((rows*(k//256)*36,),dtypes.uint32,1)
  x=UOp.placeholder((k,),dtypes.float16,2)
  prg=to_program(fn(out,words,x),CUDARenderer(Target.parse("NV:CUDA:sm_120")))
  src=next(u.arg for u in prg.src if u.op is Ops.SOURCE)
  return prg,src,NVRTCCompiler("sm_120",ptx=True,cache_key="q4k_exact_group_factorized_v1").compile(src).decode()


def test_independent_unpack_oracle_covers_every_group_and_nibble():
  words,raw=_make_q4k_words(1,4096,20260806)
  got=unpack_q4k_row(words,4096)
  blocks=raw.reshape(16,144)
  for b,rb in enumerate(blocks):
    d=np.frombuffer(rb[0:2].tobytes(),dtype="<f2")[0].astype(np.float32)
    dm=np.frombuffer(rb[2:4].tobytes(),dtype="<f2")[0].astype(np.float32)
    for g in range(8):
      sc=(int(rb[4+g])&63) if g<4 else (int(rb[12+g-4])&15)|((int(rb[4+g-4])>>6)<<4)
      mn=(int(rb[8+g])&63) if g<4 else (int(rb[12+g-4])>>4)|((int(rb[8+g-4])>>6)<<4)
      for p in range(32):
        q=(int(rb[16+(g//2)*32+p])>>((g%2)*4))&15
        assert got[b*256+g*32+p] == np.float32(d*np.float32(sc*q)-dm*np.float32(mn))


def test_oracle_is_finite_on_production_k():
  words,_=_make_q4k_words(3,4096,20260807)
  x=np.random.default_rng(20260807).normal(0,0.25,4096).astype(np.float16)
  got=oracle_gemv(words,x,3,4096)
  assert got.shape==(3,) and np.isfinite(got).all()


def test_factorized_emitter_is_one_warp_direct_fp16_and_no_dp4a():
  prg,src,ptx=_render(emit_q4k_exact_group_factorized(4096,4096))
  assert prg.arg.local_size==(32,1,1)
  assert "dp4a" not in ptx and "mma.sync" not in ptx
  assert "ld.global.u16" in ptx or "ld.global.b16" in ptx
  assert "q4k_exact_group_factorized_4096_4096" in src
  assert ".local " not in ptx


def test_four_warp_exact_emitter_is_single_kernel_direct_fp16():
  prg,src,ptx=_render(emit_q4k_exact_four_warp(4096,4096))
  assert prg.arg.local_size==(128,1,1)
  assert "dp4a" not in ptx and "mma.sync" not in ptx
  assert "bar.sync" in ptx and ".shared" in ptx
  assert "q4k_exact_four_warp_4096_4096" in src
  assert ".local " not in ptx


def test_four_warp_runtime_extent_keeps_one_compact_loop_body():
  prg,_,ptx=_render(emit_q4k_exact_four_warp_runtime_blocks(4096,4096))
  assert prg.arg.local_size==(128,1,1)
  assert len(prg.arg.vars)==1 and prg.arg.vars[0].arg==("q4k_exact_blocks",1,4)
  assert "dp4a" not in ptx and "mma.sync" not in ptx and ".local " not in ptx
