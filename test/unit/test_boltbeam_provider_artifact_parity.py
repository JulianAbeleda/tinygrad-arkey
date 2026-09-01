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

def test_q6_v_provider_matches_direct_uop_artifact():
  from tinygrad.llm.q6k_v_mmvq import emit_q6k_v_four_warp_fp16_direct
  provider,_=lower_authorized_candidate({"family":"q6_v.v1"},
    (("decode_q6k_v_four_warp_fp16_geometry","q6_v_four_warp_fp16"),))
  args=(_buf(1024,dtypes.float32),_buf(1024*16*105,dtypes.uint16),_buf(4096,dtypes.float16))
  assert provider(*args).key == emit_q6k_v_four_warp_fp16_direct()(*args).key

def test_q4_ffn_down_provider_matches_direct_uop_artifact():
  from tinygrad.llm.q4k_ffn_down_mmvq import emit_four_warp_fp16_direct
  blocks=UOp.const(dtypes.weakint,3)
  provider,_=lower_authorized_candidate({"family":"q4_ffn_down.v1","block_count":3,"resadd":True,"load_style":"vector"},
    (("decode_q4k_ffn_down_fp16_geometry","q4_ffn_down_fp16"),))
  args=(_buf(4096,dtypes.float32),_buf(4096*48*36,dtypes.uint32),_buf(12288,dtypes.float16),_buf(4096,dtypes.float32))
  assert provider(*args).key == emit_four_warp_fp16_direct(blocks,resadd=True,load_style="vector")(*args).key

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
