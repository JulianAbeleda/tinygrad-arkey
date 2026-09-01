from tinygrad import dtypes
from tinygrad.uop.ops import UOp
from tinygrad.llm.boltbeam_authority import lower_authorized_candidate

def _buf(size, dtype): return UOp.new_buffer("NV", size, dtype)

def test_q4_gate_up_provider_matches_direct_uop_artifact():
  from tinygrad.llm.q4k_gate_up_four_warp_mmvq import emit_q4k_gate_up_four_warp_fp16
  provider,_=lower_authorized_candidate({"family":"q4_gate_up.v1","vector_loads":True},
    (("decode_q4k_gate_up_four_warp_vector","q4_gate_up_four_warp"),))
  args=(_buf(12288,dtypes.float16),_buf(12288*16*36,dtypes.uint32),_buf(12288*16*36,dtypes.uint32),_buf(4096,dtypes.float16))
  assert provider(*args).key == emit_q4k_gate_up_four_warp_fp16(True)(*args).key


def test_q4_g3_generated_provider_matches_direct_uop_artifact():
  from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_kernel
  candidate={"family":"q4_g3_route.v1","rows":4096,"k":4096,"load_style":"vector",
    "epilogue_kind":"","epilogue_binding":None}
  provider,_=lower_authorized_candidate(candidate,(("decode_q4k_g3_generated","q4_g3_gemv"),))
  args=(_buf(4096,dtypes.float32),_buf(4096*16*36,dtypes.uint32),_buf(4096,dtypes.float16))
  assert provider(*args).key == q4k_g3_lanemap_gemv_kernel(4096,4096,load_style="vector")(*args).key

def test_q4_g3_epilogue_providers_match_direct_uop_artifacts():
  from tinygrad.llm.decode_kernels import Q4KGEMVEpilogue, q4k_g3_lanemap_gemv_kernel
  cases=((1024,4096,"fp16_cast",dtypes.float16,()),
         (4096,4096,"residual_add",dtypes.float32,(_buf(4096,dtypes.float32),)))
  for rows,k,kind,out_dtype,extra in cases:
    epi=Q4KGEMVEpilogue(kind)
    candidate={"family":"q4_g3_route.v1","rows":rows,"k":k,"load_style":"vector",
      "epilogue_kind":kind,"epilogue_binding":"epilogue_spec"}
    authorities=(("decode_q4k_g3_generated","q4_g3_gemv"),
      ("decode_q4k_epilogue_fusion","q4_gemv_epilogue"))
    if kind == "residual_add": authorities += (("decode_q4k_epilogue_resadd","q4_gemv_residual_epilogue"),)
    provider,_=lower_authorized_candidate(candidate,authorities,lowering_bindings={"epilogue_spec":epi})
    args=(_buf(rows,out_dtype),_buf(rows*(k//256)*36,dtypes.uint32),_buf(k,dtypes.float16),*extra)
    assert provider(*args).key == q4k_g3_lanemap_gemv_kernel(rows,k,epilogue=epi,load_style="vector")(*args).key

def test_q6_v_provider_matches_direct_uop_artifact():
  from tinygrad.llm.q6k_v_mmvq import emit_q6k_v_four_warp_fp16_direct
  provider,_=lower_authorized_candidate({"family":"q6_v.v1"},
    (("decode_q6k_v_four_warp_fp16_geometry","q6_v_four_warp_fp16"),))
  args=(_buf(1024,dtypes.float32),_buf(1024*16*105,dtypes.uint16),_buf(4096,dtypes.float16))
  assert provider(*args).key == emit_q6k_v_four_warp_fp16_direct()(*args).key


def test_q6_generated_gemv_and_vocab_reduce_match_direct_uop_artifacts():
  from tinygrad.llm.decode_kernels import (emit_q6k_gemv_kernel,emit_q6k_vocab_scalar_reduce_kernel,q6k_spec_for_role)
  gemv=q6k_spec_for_role(1024,4096,role="attn_kv",parts=1,row_tile=2,use_coop=True,reduction="in_kernel")
  gemv_candidate={"family":"q6_gemv_route.v1","rows":gemv.rows,"k":gemv.k,"row_tile":gemv.row_tile,
    "reduction":gemv.reduction,"epilogue":gemv.epilogue,"spec_binding":"q6_spec"}
  provider,_=lower_authorized_candidate(gemv_candidate,(("decode_q6k_coop_generated","q6_gemv"),),
    lowering_bindings={"q6_spec":gemv})
  gemv_args=(_buf(1024,dtypes.float32),_buf(1024*16*105,dtypes.uint16),_buf(4096,dtypes.float16))
  assert provider(*gemv_args).key == emit_q6k_gemv_kernel(gemv)(*gemv_args).key
  vocab=q6k_spec_for_role(131072,256)
  vocab_candidate={"family":"q6_vocab_reduce_route.v1","rows":vocab.rows,"k":vocab.k,"row_tile":vocab.row_tile,
    "reduction":vocab.reduction,"epilogue":vocab.epilogue,"spec_binding":"q6_spec"}
  reducer,_=lower_authorized_candidate(vocab_candidate,(("decode_q6k_coop_generated","q6_vocab_reduce"),),
    lowering_bindings={"q6_spec":vocab})
  reduce_args=(_buf(vocab.rows,dtypes.float32),
    _buf(vocab.rows*vocab.partial_axis_extent,dtypes.float32).reshape(vocab.rows,vocab.partial_axis_extent))
  assert reducer(*reduce_args).key == emit_q6k_vocab_scalar_reduce_kernel(vocab)(*reduce_args).key

def test_q6_epilogue_provider_matches_direct_uop_artifact():
  from tinygrad.llm.decode_kernels import emit_q6k_gemv_kernel, q6k_spec_for_role
  spec=q6k_spec_for_role(4096,12288,role="ffn_down",row_tile=2,reduction="in_kernel",
    target="NV:sm_120",epilogue="ffn_down_resadd")
  candidate={"family":"q6_gemv_route.v1","rows":spec.rows,"k":spec.k,"row_tile":spec.row_tile,
    "reduction":spec.reduction,"epilogue":spec.epilogue,"spec_binding":"q6_spec"}
  authorities=(("decode_q6k_coop_generated","q6_gemv"),
    ("decode_epilogue_fusion","q6_gemv_epilogue"),("decode_ffn_down_resadd","q6_ffn_down_resadd"))
  provider,_=lower_authorized_candidate(candidate,authorities,lowering_bindings={"q6_spec":spec})
  args=(_buf(4096,dtypes.float32),_buf(4096*48*105,dtypes.uint16),_buf(12288,dtypes.float16),
    _buf(4096,dtypes.float32))
  assert provider(*args).key == emit_q6k_gemv_kernel(spec)(*args).key


def test_q4_kv_pair_provider_matches_direct_uop_artifact():
  from tinygrad.llm.q4k_kv_pair import emit_q4k_kv_pair_vector
  provider,_=lower_authorized_candidate({"family":"q4_kv_pair.v1","rows":1024,"k":4096},
    (("decode_q4k_kv_pair","q4_kv_pair"),))
  args=(_buf(1024,dtypes.float32),_buf(1024,dtypes.float32),_buf(1024*16*36,dtypes.uint32),
    _buf(1024*16*36,dtypes.uint32),_buf(4096,dtypes.float16))
  assert provider(*args).key == emit_q4k_kv_pair_vector(1024,4096)(*args).key


def test_q4q4_qkv_full_provider_matches_direct_uop_artifact():
  from tinygrad.llm.q4k_kv_pair import emit_q4k_qkv_full
  provider,_=lower_authorized_candidate({"family":"q4q4_qkv_full.v1","mixed_q6_v":False},
    (("decode_q4k_q4q4_qkv_full","q4q4_qkv_full"),))
  args=(_buf(4096,dtypes.float32),_buf(1024,dtypes.float32),_buf(1024,dtypes.float32),
    _buf(4096*16*36,dtypes.uint32),_buf(2*1024*16*36,dtypes.uint32),_buf(4096,dtypes.float16))
  assert provider(*args).key == emit_q4k_qkv_full()(*args).key


def test_shared_q8_pair_providers_match_direct_uop_artifacts():
  from tinygrad.llm.shared_q8_attention import _emit_q4_cooperative_pair, _emit_q4_q6_cooperative_pair
  blocks=UOp.const(dtypes.weakint,4); packed=4096//4+4096//32
  cases=(
    ("q4kv_pair","decode_shared_q8_q4kv_pair","shared_q8_q4_kv_pair",_emit_q4_cooperative_pair(1024,blocks),
      (_buf(1024,dtypes.float32),_buf(1024,dtypes.float32),_buf(1024*16*36,dtypes.uint32),
       _buf(1024*16*36,dtypes.uint32),_buf(packed,dtypes.uint32))),
    ("q4q6_pair","decode_shared_q8_q4q6_kv_pair","shared_q8_q4q6_kv_pair",_emit_q4_q6_cooperative_pair(1024,blocks),
      (_buf(1024,dtypes.float32),_buf(1024,dtypes.float32),_buf(1024*16*36,dtypes.uint32),
       _buf(1024*16*110,dtypes.uint16),_buf(packed,dtypes.uint32))))
  for variant,route,component,direct,args in cases:
    candidate={"family":"shared_q8_multi_output.v1","variant":variant,"rows":1024,
      "block_count_binding":"cooperative_blocks"}
    provider,_=lower_authorized_candidate(candidate,((route,component),),lowering_bindings={"cooperative_blocks":blocks})
    assert provider(*args).key == direct(*args).key


def test_shared_q8_q4q4_qkv_full_provider_matches_direct_uop_artifact():
  from tinygrad.llm.shared_q8_attention import _emit_q4_cooperative_qkv_full
  blocks=UOp.const(dtypes.weakint,4);packed=4096//4+4096//32
  candidate={"family":"shared_q8_multi_output.v1","variant":"q4q4_qkv","rows":1024,
    "block_count_binding":"cooperative_blocks"}
  provider,_=lower_authorized_candidate(candidate,
    (("decode_shared_q8_q4q4_qkv_full","shared_q8_q4q4_qkv_full"),),lowering_bindings={"cooperative_blocks":blocks})
  args=(_buf(4096,dtypes.float32),_buf(1024,dtypes.float32),_buf(1024,dtypes.float32),
    _buf(4096*16*36,dtypes.uint32),_buf(2*1024*16*36,dtypes.uint32),_buf(packed,dtypes.uint32))
  assert provider(*args).key == _emit_q4_cooperative_qkv_full(blocks)(*args).key


def test_shared_q8_provider_and_direct_consumers_match_direct_uop_artifacts():
  from tinygrad.llm.shared_q8_attention import _emit_q8_provider, _emit_q4_cooperative, _emit_q6_warp_direct
  blocks=UOp.const(dtypes.weakint,4); packed=4096//4+4096//32
  provider,_=lower_authorized_candidate({"family":"shared_q8_provider.v1","source_dtype":"fp16","k":4096},
    (("decode_shared_q8_attention","shared_q8_provider"),))
  provider_args=(_buf(packed,dtypes.uint32),_buf(4096,dtypes.float16))
  assert provider(*provider_args).key == _emit_q8_provider()(*provider_args).key
  q4_candidate={"family":"shared_q8_attention_consumer.v1","rows":1024,"variant":"q4_cooperative",
    "direct_output":True,"block_count_binding":"cooperative_blocks"}
  q4,_=lower_authorized_candidate(q4_candidate,(("decode_shared_q8_attention","shared_q8_q4_consumer"),
    ("decode_q4_direct_shared_q8_attention","shared_q8_q4_direct")),lowering_bindings={"cooperative_blocks":blocks})
  q4_args=(_buf(1024,dtypes.float32),_buf(1024*16*36,dtypes.uint32),_buf(packed,dtypes.uint32))
  assert q4(*q4_args).key == _emit_q4_cooperative(1024,blocks,direct_output=True)(*q4_args).key
  q6_candidate={"family":"shared_q8_attention_consumer.v1","rows":1024,"variant":"q6_warp_direct",
    "direct_output":False,"block_count_binding":None}
  q6,_=lower_authorized_candidate(q6_candidate,(("decode_shared_q8_attention","shared_q8_q6_consumer"),
    ("decode_q6_direct_shared_q8_attention","shared_q8_q6_direct")))
  q6_args=(_buf(1024,dtypes.float32),_buf(1024*16*110,dtypes.uint16),_buf(packed,dtypes.uint32))
  assert q6(*q6_args).key == _emit_q6_warp_direct(1024)(*q6_args).key

def test_q4_ffn_down_provider_matches_direct_uop_artifact():
  from tinygrad.llm.q4k_ffn_down_mmvq import emit_four_warp_fp16_direct
  blocks=UOp.const(dtypes.weakint,3)
  provider,_=lower_authorized_candidate({"family":"q4_ffn_down.v1","block_count":3,"resadd":True,"load_style":"vector"},
    (("decode_q4k_ffn_down_fp16_geometry","q4_ffn_down_fp16"),("decode_ffn_down_resadd","q4_ffn_down_resadd")))
  args=(_buf(4096,dtypes.float32),_buf(4096*48*36,dtypes.uint32),_buf(12288,dtypes.float16),_buf(4096,dtypes.float32))
  assert provider(*args).key == emit_four_warp_fp16_direct(blocks,resadd=True,load_style="vector")(*args).key


def test_q4_w1w3_providers_match_direct_uop_artifacts():
  from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_w1w3_kernel
  args32=(_buf(12288,dtypes.float32),_buf(12288*16*36,dtypes.uint32),
    _buf(12288*16*36,dtypes.uint32),_buf(4096,dtypes.float16))
  args16=(_buf(12288,dtypes.float16),)+args32[1:]
  for store_fp16,args in ((False,args32),(True,args16)):
    authorities=(("decode_q4k_w1w3_fusion","q4_w1w3_fused"),)+(
      (("decode_q4k_w1w3_fp16_store","q4_w1w3_fused_fp16"),) if store_fp16 else ())
    candidate={"family":"q4_w1w3.v1","rows":12288,"k":4096,"load_style":"vector","store_fp16":store_fp16}
    provider,_=lower_authorized_candidate(candidate,authorities)
    assert provider(*args).key == q4k_g3_lanemap_gemv_w1w3_kernel(12288,4096,load_style="vector",store_fp16=store_fp16)(*args).key

def test_argmax_provider_matches_direct_uop_artifact():
  from tinygrad.llm.packed_argmax import emit_native_finite_fp32_argmax
  provider,_=lower_authorized_candidate({"family":"finite_argmax.v1","n":151936,"threads":1024,"host_mirror":False},
    (("decode_native_argmax","finite_fp32_argmax"),))
  args=(_buf(1,dtypes.int32),_buf(151936,dtypes.float32))
  assert provider(*args).key == emit_native_finite_fp32_argmax(151936,1024)(*args).key

def test_q4_attention_k_provider_matches_direct_uop_artifact():
  from extra.llm_research.decode.q4k_exact_group_factorized import emit_q4k_exact_four_warp
  provider,_=lower_authorized_candidate({"family":"q4_k_four_warp.v1","rows":1024,"k":4096},
    (("decode_q4k_k_four_warp","q4_k_four_warp"),))
  args=(_buf(1024,dtypes.float32),_buf(1024*16*36,dtypes.uint32),_buf(4096,dtypes.float16))
  assert provider(*args).key == emit_q4k_exact_four_warp(1024,4096)(*args).key

def test_q6_ffn_down_provider_matches_direct_uop_artifact():
  from tinygrad.llm.q6k_ffn_down_mmvq import emit_q6k_four_warp_fp16_direct
  candidate={"family":"q6_ffn_down.v1","rows_per_block":8,"packed_lanemap":False,
             "unroll_blocks":None,"split_weight_stream":False}
  provider,_=lower_authorized_candidate(candidate,(("decode_q6k_ffn_down_fp16_geometry","q6_ffn_down_fp16"),))
  args=(_buf(4096,dtypes.float32),_buf(4096*48*105,dtypes.uint16),_buf(12288,dtypes.float16),_buf(4096,dtypes.float32))
  direct=emit_q6k_four_warp_fp16_direct(rows_per_block=8,packed_lanemap=False,unroll_blocks=None,split_weight_stream=False)
  assert provider(*args).key == direct(*args).key


def test_q6_ffn_down_variant_providers_match_direct_uop_artifacts():
  from tinygrad.llm.q6k_ffn_down_mmvq import emit_q6k_four_warp_fp16_direct
  args=(_buf(4096,dtypes.float32),_buf(4096*48*105,dtypes.uint16),_buf(12288,dtypes.float16),_buf(4096,dtypes.float32))
  base=(("decode_q6k_ffn_down_fp16_geometry","q6_ffn_down_fp16"),("decode_ffn_down_resadd","q6_ffn_down_resadd"))
  cases=((True,None,base+(("decode_q6k_ffn_down_packed_lanemap","q6_ffn_down_packed_lanemap"),)),
         (True,12,base+(("decode_q6k_ffn_down_packed_lanemap","q6_ffn_down_packed_lanemap"),
                        ("decode_q6k_ffn_down_unroll","q6_ffn_down_packed_unroll"))))
  for packed_lanemap,unroll_blocks,authorities in cases:
    candidate={"family":"q6_ffn_down.v1","rows_per_block":1,"packed_lanemap":packed_lanemap,
      "unroll_blocks":unroll_blocks,"split_weight_stream":False}
    provider,_=lower_authorized_candidate(candidate,authorities)
    direct=emit_q6k_four_warp_fp16_direct(rows_per_block=1,packed_lanemap=packed_lanemap,
      unroll_blocks=unroll_blocks,split_weight_stream=False)
    assert provider(*args).key == direct(*args).key

def test_decode_rmsnorm_provider_matches_direct_uop_artifact():
  from tinygrad.llm.decode_kernels import DecodeRMSNormSpec, emit_decode_rmsnorm_kernel
  spec=DecodeRMSNormSpec(rows=1,dim=4096,eps=1e-6,warps_per_row=16,x_dtype=dtypes.float32,
    weight_dtype=dtypes.float16,out_dtype=dtypes.float32,x_rank=1)
  candidate={"family":"decode_rmsnorm.v1","rows":1,"dim":4096,"eps":1e-6,"warps_per_row":16,
    "x_dtype":"dtypes.float","weight_dtype":"dtypes.half","out_dtype":"dtypes.float","x_rank":1}
  provider,_=lower_authorized_candidate(candidate,(("decode_rmsnorm_native_lowering","decode_rmsnorm"),
    ("decode_norm_fusion","decode_norm_fusion")))
  args=(_buf(4096,dtypes.float32),_buf(4096,dtypes.float32),_buf(4096,dtypes.float16))
  assert provider(*args).key == emit_decode_rmsnorm_kernel(spec)(*args).key

def test_cache_sink_provider_matches_direct_uop_artifact():
  from tinygrad.uop.ops import Ops, ReduceOutputSpec
  from tinygrad.llm.producer_kv_cache_sink import emit_reduce_output_rope_kv_cache
  spec=ReduceOutputSpec(rows=8,dim=128,eps=1e-6,out_dtype=dtypes.float32,affine=True,
    recipe="sumsq_rsqrt_affine",reduce_op=Ops.ADD,warps=8,lanes=32,per_lane=4,epilogue="rope")
  candidate={"family":"qk_norm_rope_cache_sink.v1","spec_repr":repr(spec),"producer_dtype":"dtypes.float",
    "weight_dtype":"dtypes.half","cache_dtype":"dtypes.half","max_context":4096,"spec_binding":"reduce_output_spec"}
  provider,_=lower_authorized_candidate(candidate,(("decode_producer_kv_cache_sink","qk_norm_rope_cache_sink"),),
    lowering_bindings={"reduce_output_spec":spec})
  direct=emit_reduce_output_rope_kv_cache(spec,dtypes.float32,dtypes.float16,dtypes.float16,4096)
  args=(_buf(2*1*8*4096*128,dtypes.float16).reshape(2,1,8,4096,128),_buf(1024,dtypes.float32),
        _buf(128,dtypes.float16),_buf(1024,dtypes.float32),_buf(4096*128,dtypes.float32).reshape(4096,128))
  assert provider(*args).key == direct(*args).key
