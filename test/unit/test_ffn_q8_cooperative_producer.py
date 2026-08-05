import pathlib
import numpy as np
import pytest

from tinygrad import dtypes
from tinygrad.uop.ops import AxisType, Ops, UOp
from extra.llm_research.decode.ffn_q8_cooperative_producer import (
  FFNQ8CooperativePlan, K, ROWS, Q8_METADATA_BASE, emit_ffn_w1w3_q8_cooperative, fp16_glu, pack_q8_1_private)
from extra.llm_research.decode.ffn_q8_cooperative_producer import (llama_q8_1_oracle_private,
  emit_q4_q8_ffn_down_consumer, q4_q8_ffn_down_row_reference)
from extra.llm_research.decode.route_class_numerics import _make_q4k_words


def _unpack(packet):
  payload=packet[:Q8_METADATA_BASE].view(np.uint8).reshape(-1,4).reshape(-1)
  meta=packet[Q8_METADATA_BASE:]
  d=(meta & 0xffff).astype(np.uint16).view(np.float16)
  s=(meta >> 16).astype(np.uint16).view(np.float16)
  return payload.view(np.int8),d,s


def test_ffn_plan_is_one_packet_per_cta_and_strictly_smaller_chain():
  p=FFNQ8CooperativePlan(); p.validate()
  assert p.grid_x == 384 and p.output_words == 3456
  assert p.topology()["strictly_smaller"]
  assert p.topology()["candidate_programs"] < p.topology()["baseline_programs_min"]


def test_cpu_oracle_round_point_and_private_q8_abi():
  oracle_lib=pathlib.Path("/home/ubuntu/env/llama.cpp/build-cuda/bin/libggml-base.so.0.14.0")
  if not oracle_lib.is_file(): pytest.skip("independent llama Q8 CPU oracle library is not installed")
  rng=np.random.default_rng(20260805)
  gate=rng.normal(size=ROWS).astype(np.float32); up=rng.normal(size=ROWS).astype(np.float32)
  rounded=fp16_glu(gate,up)
  expected=(gate/(1+np.exp(-gate))*up).astype(np.float16)
  assert np.array_equal(rounded,expected)
  packed=pack_q8_1_private(rounded)
  cpu_ref=llama_q8_1_oracle_private(rounded,str(oracle_lib))
  # The independent ggml CPU reference agrees on payload and d. Its ``s`` is
  # model-file sum(q)*d semantics; CUDA's live provider deliberately records
  # the raw fp16 input sum, which is the P3b consumer ABI used here.
  assert np.array_equal(packed[:Q8_METADATA_BASE],cpu_ref[:Q8_METADATA_BASE])
  assert np.array_equal(packed[Q8_METADATA_BASE:] & 0xffff,cpu_ref[Q8_METADATA_BASE:] & 0xffff)
  q,d,s=_unpack(packed)
  assert q.shape == (ROWS,) and d.shape == s.shape == (384,)
  # Metadata is d and the fp16 raw input sum, rather than sum(q)*d.
  def cuda_sum(x):
    x=x.astype(np.float32).copy()
    for off in (16,8,4,2,1): x[:off]=x[:off]+x[off:off+off]
    return np.float16(x[0])
  assert np.array_equal(s,np.array([cuda_sum(x) for x in rounded.reshape(-1,32)]))
  assert np.all(d >= 0)


def test_cooperative_emitter_uses_runtime_loops_shared_packet_and_no_activation_output():
  fn=emit_ffn_w1w3_q8_cooperative()
  out=UOp.placeholder((3456,),dtypes.uint32,0)
  gw=UOp.placeholder((ROWS*16*36,),dtypes.uint32,1)
  uw=UOp.placeholder((ROWS*16*36,),dtypes.uint32,2)
  x=UOp.placeholder((K,),dtypes.float16,3)
  ast=fn(out,gw,uw,x)
  assert ast.arg.name == "ffn_w1w3_q8_cooperative_12288_4096"
  # The only output store target is the packed Q8 buffer; the 32 FP16 values
  # live in LOCAL storage and cannot become a materialized global activation.
  nodes=ast.toposort()
  assert any(u.op is Ops.BARRIER for u in nodes)
  assert sum(u.op is Ops.RANGE and u.arg[-1] is AxisType.LOOP for u in nodes) >= 2
  assert any(u.op is Ops.DEFINE_LOCAL and u.dtype.base is dtypes.half for u in nodes)


def test_q4_ffn_down_consumer_is_exactly_compatible_with_provider_private_abi():
  rng=np.random.default_rng(20260805)
  packet=pack_q8_1_private(fp16_glu(rng.normal(size=ROWS).astype(np.float32),rng.normal(size=ROWS).astype(np.float32)))
  words,_=_make_q4k_words(1,ROWS,20260805)
  got=q4_q8_ffn_down_row_reference(words,packet)
  assert np.isfinite(got)
  fn=emit_q4_q8_ffn_down_consumer()
  ast=fn(UOp.placeholder((4096,),dtypes.float32,0),UOp.placeholder((4096*48*36,),dtypes.uint32,1),UOp.placeholder((3456,),dtypes.uint32,2))
  assert ast.arg.name == "q4k_q8_ffn_down_4096_12288"
