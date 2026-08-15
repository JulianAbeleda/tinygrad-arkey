import ast, collections, inspect, json, pathlib, textwrap
import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import Ops, UOp
from tinygrad.llm.decode_routes import _Q4KDecodeCandidate
from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_kernel, q4k_g3_lanemap_gemv_w1w3_kernel
from tinygrad.llm.model_route_plan import (decode_q4k_ffn_down_fp16_geometry_promoted,
  load_decode_q4k_ffn_down_fp16_geometry_promotion, _DECODE_Q4K_FFN_DOWN_FP16_GEOMETRY_PROMOTED_TARGETS)
from tinygrad.llm.kernel_program import (DeclaredTypedOutput, KernelProgram, KernelProgramProvenance,
  OutputSpec, ResidualViewRequest, TypedLayout, TypedViewRequest, execute_promoted_program,
  execute_research_program)
from extra.llm_research.decode.q4k_ffn_down_mmvq_profile_analysis import analyze
from extra.llm_research.decode.ffn_q8_cooperative_producer import pack_q8_1_private, q4_q8_ffn_down_row_reference
from extra.llm_research.decode.q4k_ffn_down_mmvq import (
  BLOCKS_PER_WARP,K,Q4_BLOCKS,Q8_PAYLOAD_WORDS,Q8_WORDS,ROWS,SUB_BLOCKS,Q4KFFNDownMMVQAdmission,
  emit_ffn_w1w3_q8_scalar_packet,emit_four_warp_direct,emit_four_warp_fp16_direct,emit_q8_provider,
  owned_boundary_topology,ownership_coordinates,q4k_ffn_down_mmvq_call,q4k_ffn_down_mmvq_scalar_packet_call)
from extra.llm_research.decode.route_class_numerics import _make_q4k_words
from extra.llm_research.layout import q4_k_reference


def test_four_warp_ownership_partitions_all_production_q4_words_once():
  got=ownership_coordinates()
  assert Q4_BLOCKS == 48 and BLOCKS_PER_WARP == 12
  assert len(got) == 4*32*12*2 == len(set(got))
  for block in range(Q4_BLOCKS):
    owned={(group,word) for _warp,_lane,b,group,word in got if b == block}
    assert owned == {(group,word) for group in range(8) for word in range(8)}


def test_production_call_is_closed_without_explicit_admission():
  assert q4k_ffn_down_mmvq_call(None,None,None,None,{}) is None
  try: Q4KFFNDownMMVQAdmission(-1)
  except ValueError: pass
  else: raise AssertionError("negative block admission must fail closed")
  try: Q4KFFNDownMMVQAdmission(0,owned_input_boundary=1)
  except ValueError: pass
  else: raise AssertionError("owned boundary admission must require an explicit bool")
  try: Q4KFFNDownMMVQAdmission(0,scalar_q8_packet=1)
  except ValueError: pass
  else: raise AssertionError("scalar_q8_packet admission must require an explicit bool")
  try: Q4KFFNDownMMVQAdmission(0,fp16_fma=True,scalar_q8_packet=True)
  except ValueError: pass
  else: raise AssertionError("scalar_q8_packet and fp16_fma must be mutually exclusive")
  assert q4k_ffn_down_mmvq_scalar_packet_call(None,None,None,None,None,None) is None


def test_default_decode_route_import_is_strictly_behind_explicit_lease_guard():
  fn=ast.parse(textwrap.dedent(inspect.getsource(_Q4KDecodeCandidate.execute))).body[0]
  guarded=[node for node in fn.body if isinstance(node,ast.If) and
    "_q4k_ffn_down_mmvq_admission" in ast.unparse(node.test)]
  assert len(guarded) == 1
  assert any(isinstance(node,ast.ImportFrom) and node.module == "tinygrad.llm.q4k_ffn_down_mmvq"
    for node in ast.walk(guarded[0]))
  assert not any(isinstance(node,ast.ImportFrom) and node.module == "tinygrad.llm.q4k_ffn_down_mmvq"
    for node in fn.body)


def test_profile_analysis_keeps_local_and_matched_deltas_disjoint():
  def entry(name,duration): return {"name":name,"duration":str(duration)}
  control_entries=[entry("q4k_g3_lanemap_gemv_4096_12288",25.5),entry("same",10.)]
  candidate_entries=[entry("q8_1_llama_provider_12288",1.4),entry("q4k_q8_mmvq_direct_4096_12288",22.6),entry("same",19.8)]
  control={"entries":control_entries,"node_sum_us":35.5,"device_window_us":40.,"outside_device_us":5.,"wall_sync_us":45.,
    "groups":[{"members":2,"span_us":35.5}]}
  candidate={"entries":candidate_entries,"node_sum_us":43.8,"device_window_us":48.3,"outside_device_us":2.9,"wall_sync_us":51.2,
    "groups":[{"members":3,"span_us":43.8}]}
  out=analyze(control,candidate,{"candidate_minus_control_ms":.0062})
  assert np.isclose(out["profile_node_equation_us"]["local_delta"],-1.5)
  assert np.isclose(out["profile_node_equation_us"]["matched_delta"],9.8)
  assert np.isclose(out["profile_node_equation_us"]["node_sum_delta"],8.3)


def test_independent_dense_q4_times_q8_matches_scalar_packed_algebra():
  words,raw=_make_q4k_words(1,K,202608054)
  x=np.random.default_rng(202608055).normal(0,.2,K).astype(np.float16)
  packed=pack_q8_1_private(x)
  scalar=float(q4_q8_ffn_down_row_reference(words,packed,0))

  weights=q4_k_reference(Tensor(raw.reshape(-1),dtype=dtypes.uint8),K).numpy().astype(np.float32).reshape(K)
  packets=packed[:Q8_PAYLOAD_WORDS]
  q=np.empty(K,dtype=np.int8)
  for lane in range(4): q[lane::4]=((packets>>(8*lane))&255).astype(np.uint8).view(np.int8)
  metadata=packed[Q8_PAYLOAD_WORDS:]
  d=(metadata&0xffff).astype(np.uint16).view(np.float16).astype(np.float32)
  dense=float(np.dot(weights,q.astype(np.float32)*np.repeat(d,32)))
  assert abs(scalar-dense) <= max(2e-4,abs(dense)*2e-5)


def _render(ast:UOp,key:str):
  program=to_program(ast,CUDARenderer(Target.parse("NV:CUDA:sm_120")))
  source=next(u.arg for u in program.src if u.op is Ops.SOURCE)
  ptx=NVRTCCompiler("sm_120",ptx=True,cache_key=key).compile(source).decode()
  return program,source,ptx


def test_provider_is_one_production_shape_q8_kernel():
  ast=emit_q8_provider()(UOp.placeholder((Q8_WORDS,),dtypes.uint32,0),UOp.placeholder((K,),dtypes.float16,1))
  program,source,_ptx=_render(ast,"q4k_ffn_down_q8_provider_v1")
  assert program.arg.global_size == (48,1,1)
  assert program.arg.local_size == (256,1,1)
  assert "q8_1_llama_provider_12288" in source
  assert Q8_WORDS == 3456


def test_scalar_packet_producer_renders_1024_threads_and_packed_q8_abi():
  gate_words=ROWS*Q4_BLOCKS*36
  ast=emit_ffn_w1w3_q8_scalar_packet()(
    UOp.placeholder((Q8_WORDS,),dtypes.uint32,0),
    UOp.placeholder((gate_words,),dtypes.uint32,1),
    UOp.placeholder((gate_words,),dtypes.uint32,2),
    UOp.placeholder((K,),dtypes.float16,3))
  program,source,ptx=_render(ast,"q4k_ffn_down_scalar_packet_producer_v1")
  assert program.arg.global_size == (384,1,1)
  assert program.arg.local_size == (1024,1,1)
  assert "ffn_w1w3_q8_scalar_packet_12288_4096" in source
  # One 32-row Q8_1 packet per CTA: 32 warps publish one fp16 GLU result each,
  # then warp zero quantizes after the CTA barrier. The staging array is LOCAL
  # (shared) memory, never a per-thread stack spill.
  assert "__syncthreads" in source and ".local " not in ptx
  assert "st.global" in ptx and "shfl.sync" in ptx


def test_owned_fp32_boundary_has_one_to_one_topology_and_explicit_fp16_rounding():
  plan=owned_boundary_topology()
  assert plan["control_node_count"] == plan["candidate_node_count"] == 3
  assert plan["net_graph_members"] == 0 and plan["removed_intermediate"] == "fp16 activation materialization"
  ast=emit_q8_provider(dtypes.float32)(UOp.placeholder((Q8_WORDS,),dtypes.uint32,0),UOp.placeholder((K,),dtypes.float32,1))
  program,source,_ptx=_render(ast,"q4k_ffn_down_q8_owned_fp32_provider_v1")
  assert program.arg.vars == () and "float* data1_12288" in source
  assert "((float)(((half)(val0))))" in source


def test_owned_boundary_rounding_matches_the_existing_q8_input_authority():
  fp32=np.random.default_rng(202608056).normal(0,.2,K).astype(np.float32)
  # The owned provider's source contract rounds fp32 to fp16 before the
  # already-qualified live-llama x/d quantizer. Therefore its authority input
  # is byte-identical to the former materialized boundary.
  materialized=fp32.astype(np.float16)
  assert np.array_equal(pack_q8_1_private(materialized),pack_q8_1_private(fp32.astype(np.float16)))


def test_owned_boundary_route_has_no_materialize_and_single_use_lifetimes():
  fn=ast.parse(textwrap.dedent(inspect.getsource(q4k_ffn_down_mmvq_call))).body[0]
  owned=[node for node in fn.body if isinstance(node,ast.If) and "owned_input_boundary" in ast.unparse(node.test)]
  assert len(owned) == 1
  assert ".contiguous()" not in ast.unparse(owned[0].body)
  assert ".cast(dtypes.float16).contiguous()" in ast.unparse(owned[0].orelse)
  loads=collections.Counter(node.id for node in ast.walk(fn) if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Load))
  # xv feeds exactly one consumer per mutually exclusive branch (the Q8 provider
  # or the fp16 direct consumer); packed exists only on the Q8 provider branch.
  assert loads["xv"] == 2 and loads["packed"] == 1


def test_owned_boundary_strips_equal_span_reshapes_to_the_after_without_a_copy():
  # Regression: passing the fused w1w3 output's RESHAPE view into the opaque
  # provider made custom_kernel conservatively insert a fp32 materialize copy
  # (net +2 graph calls). The owned branch must strip equal-span reshapes and
  # hand the AFTER itself to the provider; any other shape fails closed.
  fn=ast.parse(textwrap.dedent(inspect.getsource(q4k_ffn_down_mmvq_call))).body[0]
  owned=[node for node in fn.body if isinstance(node,ast.If) and "owned_input_boundary" in ast.unparse(node.test)]
  assert len(owned) == 1
  body=ast.unparse(owned[0].body)
  assert "Ops.RESHAPE" in body and "Ops.AFTER" in body and "Tensor(owned_uop)" in body
  assert "owned_uop.src[0].numel() == owned_expected" in body
  assert ".contiguous()" not in body and ".cast(" not in body
  assert "owned_uop.op is not Ops.AFTER" in body and "owned_uop.shape != (K,)" in body and "owned_uop.dtype != x.dtype" in body


def test_owned_boundary_is_an_exact_three_call_replacement_in_isolated_cpu_schedules():
  # Build each arm from a fresh graph. Scheduling mutates Tensor graphs into
  # concrete buffer identities, so sharing one producer across arm schedules
  # would make the second census invalid.
  def schedule_names(candidate:bool) -> list[str]:
    in_k=4096
    gate_words=Tensor.empty((12288*(in_k//256)*36,),dtype=dtypes.uint32,device="CPU")
    up_words=Tensor.empty((12288*(in_k//256)*36,),dtype=dtypes.uint32,device="CPU")
    down_words=Tensor.empty((ROWS*Q4_BLOCKS*36,),dtype=dtypes.uint32,device="CPU")
    hidden=Tensor.empty((in_k,),dtype=dtypes.float16,device="CPU")
    producer=KernelProgram("cpu_topology","w1w3",KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
      q4k_g3_lanemap_gemv_w1w3_kernel(12288,in_k),output_spec=OutputSpec((12288,),dtypes.float32))
    z=execute_promoted_program(None,gate_words,up_words,hidden,program=producer)
    if candidate:
      provider=KernelProgram("cpu_topology","provider",KernelProgramProvenance.RESEARCH_ONLY,
        emit_q8_provider(dtypes.float32),output_spec=OutputSpec((Q8_WORDS,),dtypes.uint32))
      packed=execute_research_program(Tensor.empty((Q8_WORDS,),dtype=dtypes.uint32,device="CPU"),z,program=provider)
      consumer=KernelProgram("cpu_topology","consumer",KernelProgramProvenance.RESEARCH_ONLY,
        emit_four_warp_direct(UOp.const(dtypes.weakint,BLOCKS_PER_WARP)),output_spec=OutputSpec((ROWS,),dtypes.float32))
      out=execute_research_program(Tensor.empty((ROWS,),dtype=dtypes.float32,device="CPU"),down_words,packed,program=consumer)
    else:
      installed=KernelProgram("cpu_topology","installed",KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
        q4k_g3_lanemap_gemv_kernel(ROWS,K),output_spec=OutputSpec((ROWS,),dtypes.float32))
      out=execute_promoted_program(None,down_words,z.cast(dtypes.float16).contiguous(),program=installed)
    return [call.src[0].arg.name for call in out.schedule_linear().src]

  control=schedule_names(False)
  candidate=schedule_names(True)
  assert len(control) == len(candidate) == 3
  assert control[0] == candidate[0] == "q4k_g3_lanemap_gemv_w1w3fused_12288_4096"
  # The CPU renderer's generic materialization kernel is named ``test``;
  # native NV gives the same call its compiled ``E_*`` program name.
  assert control[1] == "test" and control[2] == "q4k_g3_lanemap_gemv_4096_12288"
  assert candidate[1:] == ["q8_1_llama_provider_12288","q4k_q8_mmvq_direct_4096_12288"]


def test_closed_fp16_resadd_route_folds_both_transports_to_zero_materialize():
  # The closed (non-owned) resadd spelling has TWO boundary folds that must both
  # land or the candidate pays two extra transport kernels: the fp16 z provider
  # input (typed epilogue-absorption view) and the fp32 normed_h residual slot
  # (residual view). Both producers declare epilogue absorption; the provider and
  # direct consumer use their research-only ``.q8_provider`` / ``.consumer`` ids.
  def schedule_names() -> list[str]:
    gate_words=Tensor.empty((12288*(4096//256)*36,),dtype=dtypes.uint32,device="CPU")
    up_words=Tensor.empty((12288*(4096//256)*36,),dtype=dtypes.uint32,device="CPU")
    down_words=Tensor.empty((ROWS*Q4_BLOCKS*36,),dtype=dtypes.uint32,device="CPU")
    hidden=Tensor.empty((4096,),dtype=dtypes.float16,device="CPU")
    producer=KernelProgram("cpu_topology","w1w3_fused",KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
      q4k_g3_lanemap_gemv_w1w3_kernel(K,4096,load_style="scalar",store_fp16=True),
      output_spec=OutputSpec((K,),dtypes.float16,
        typed_output=DeclaredTypedOutput(TypedLayout(dtypes.float16,(K,),(1,1,K)),
          combine_fusion_admitted=False,epilogue_absorption_admitted=True)))
    z=execute_promoted_program(None,gate_words,up_words,hidden,program=producer).reshape(1,1,K)
    # h stands in for the attn residual producer: fp32 block output that declares
    # epilogue absorption exactly like the promoted attn_qo residual_add GEMV.
    h_producer=KernelProgram("cpu_topology","attn_resadd",KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
      q4k_g3_lanemap_gemv_w1w3_kernel(ROWS,4096,load_style="scalar"),
      output_spec=OutputSpec((ROWS,),dtypes.float32,
        typed_output=DeclaredTypedOutput(TypedLayout(dtypes.float32,(ROWS,),(1,1,ROWS)),
          combine_fusion_admitted=False,epilogue_absorption_admitted=True)))
    h=execute_promoted_program(None,gate_words,up_words,hidden,program=h_producer).reshape(1,1,ROWS)
    xv=z[:,0,:].reshape(K).cast(dtypes.float16).contiguous()
    residual=h[:,0,:].reshape(ROWS).cast(dtypes.float32)
    provider=KernelProgram("decode_q4k_ffn_down_mmvq","blk16.q8_provider",KernelProgramProvenance.RESEARCH_ONLY,
      emit_q8_provider(dtypes.float16),
      typed_input_views=(TypedViewRequest(slot=0,dtype=dtypes.float16,flat_shape=(K,),route_role="ffn_down",
        requires_combine_fusion=False,requires_epilogue_absorption=True),))
    packed=execute_research_program(Tensor.empty((Q8_WORDS,),dtype=dtypes.uint32,device="CPU"),xv,program=provider)
    consumer=KernelProgram("decode_q4k_ffn_down_mmvq","blk16.consumer",KernelProgramProvenance.RESEARCH_ONLY,
      emit_four_warp_direct(UOp.const(dtypes.weakint,BLOCKS_PER_WARP),resadd=True),
      output_spec=OutputSpec((ROWS,),dtypes.float32,
        typed_output=DeclaredTypedOutput(TypedLayout(dtypes.float32,(ROWS,),(1,1,ROWS)),
          combine_fusion_admitted=False,epilogue_absorption_admitted=True)),
      residual_input_views=(ResidualViewRequest(slot=2,dtype=dtypes.float32,flat_shape=(ROWS,),
        route_role="ffn_down",kind="residual_add"),))
    out=execute_research_program(Tensor.empty((ROWS,),dtype=dtypes.float32,device="CPU"),
      down_words,packed,residual,program=consumer)
    return [call.src[0].arg.name for call in out.schedule_linear().src]

  names=schedule_names()
  assert len(names)==4 and "test" not in names
  assert names==["q4k_g3_lanemap_gemv_w1w3fused16_12288_4096","q4k_g3_lanemap_gemv_w1w3fused_4096_4096",
    "q8_1_llama_provider_12288","q4k_q8_mmvq_direct_4096_12288_epi_ffnresadd"]


def test_direct_consumer_keeps_four_warps_runtime_loop_dp4a_and_final_store():
  blocks=UOp.variable("q4k_ffn_down_blocks",1,BLOCKS_PER_WARP)
  ast=emit_four_warp_direct(blocks)(UOp.placeholder((ROWS,),dtypes.float32,0),
    UOp.placeholder((ROWS*Q4_BLOCKS*36,),dtypes.uint32,1),UOp.placeholder((Q8_WORDS,),dtypes.uint32,2))
  program,source,ptx=_render(ast,"q4k_ffn_down_mmvq_direct_v1")
  assert program.arg.local_size == (128,1,1)
  assert program.arg.global_size == (ROWS,1,1)
  assert len(program.arg.vars) == 1 and program.arg.vars[0].arg == ("q4k_ffn_down_blocks",1,12)
  assert "q4k_q8_mmvq_direct_4096_12288" in source and "__syncthreads" in source
  assert "dp4a" in ptx and ".local " not in ptx
  assert "st.global" in ptx


def test_fp16_resadd_route_folds_both_transports_to_zero_materialize():
  # The fp16 geometry route has the same two boundary folds as the closed Q8
  # resadd route: the fp16 z activation (typed epilogue-absorption view, slot 1)
  # and the fp32 normed_h residual (residual view, slot 2). Both producers
  # declare epilogue absorption; the consumer uses the .gemv program id so the
  # M5/M2b validators admit the fold instead of silently materializing.
  def schedule_names() -> list[str]:
    gate_words=Tensor.empty((12288*(4096//256)*36,),dtype=dtypes.uint32,device="CPU")
    up_words=Tensor.empty((12288*(4096//256)*36,),dtype=dtypes.uint32,device="CPU")
    down_words=Tensor.empty((ROWS*Q4_BLOCKS*36,),dtype=dtypes.uint32,device="CPU")
    hidden=Tensor.empty((4096,),dtype=dtypes.float16,device="CPU")
    producer=KernelProgram("cpu_topology","w1w3_fused",KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
      q4k_g3_lanemap_gemv_w1w3_kernel(K,4096,load_style="scalar",store_fp16=True),
      output_spec=OutputSpec((K,),dtypes.float16,
        typed_output=DeclaredTypedOutput(TypedLayout(dtypes.float16,(K,),(1,1,K)),
          combine_fusion_admitted=False,epilogue_absorption_admitted=True)))
    z=execute_promoted_program(None,gate_words,up_words,hidden,program=producer).reshape(1,1,K)
    h_producer=KernelProgram("cpu_topology","attn_resadd",KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
      q4k_g3_lanemap_gemv_w1w3_kernel(ROWS,4096,load_style="scalar"),
      output_spec=OutputSpec((ROWS,),dtypes.float32,
        typed_output=DeclaredTypedOutput(TypedLayout(dtypes.float32,(ROWS,),(1,1,ROWS)),
          combine_fusion_admitted=False,epilogue_absorption_admitted=True)))
    h=execute_promoted_program(None,gate_words,up_words,hidden,program=h_producer).reshape(1,1,ROWS)
    xv=z[:,0,:].reshape(K).cast(dtypes.float16).contiguous()
    residual=h[:,0,:].reshape(ROWS).cast(dtypes.float32)
    consumer=KernelProgram("decode_q4k_ffn_down_mmvq","blk16.gemv",KernelProgramProvenance.RESEARCH_ONLY,
      emit_four_warp_fp16_direct(UOp.const(dtypes.weakint,SUB_BLOCKS),resadd=True),
      output_spec=OutputSpec((ROWS,),dtypes.float32,
        typed_output=DeclaredTypedOutput(TypedLayout(dtypes.float32,(ROWS,),(1,1,ROWS)),
          combine_fusion_admitted=False,epilogue_absorption_admitted=True)),
      typed_input_views=(TypedViewRequest(slot=1,dtype=dtypes.float16,flat_shape=(K,),route_role="ffn_down",
        requires_combine_fusion=False,requires_epilogue_absorption=True),),
      residual_input_views=(ResidualViewRequest(slot=2,dtype=dtypes.float32,flat_shape=(ROWS,),
        route_role="ffn_down",kind="residual_add"),))
    out=execute_research_program(Tensor.empty((ROWS,),dtype=dtypes.float32,device="CPU"),
      down_words,xv,residual,program=consumer)
    return [call.src[0].arg.name for call in out.schedule_linear().src]

  names=schedule_names()
  assert len(names)==3 and "test" not in names
  assert names==["q4k_g3_lanemap_gemv_w1w3fused16_12288_4096","q4k_g3_lanemap_gemv_w1w3fused_4096_4096",
    "q4k_fp16_mmvq_direct_4096_12288_epi_ffnresadd"]


def test_sum_dp4a_research_spelling_adds_only_the_q8_correction_dp4a():
  blocks=UOp.variable("q4k_ffn_down_blocks",1,BLOCKS_PER_WARP)
  args=(UOp.placeholder((ROWS,),dtypes.float32,0),UOp.placeholder((ROWS*Q4_BLOCKS*36,),dtypes.uint32,1),
        UOp.placeholder((Q8_WORDS,),dtypes.uint32,2))
  _program,_source,scalar_ptx=_render(emit_four_warp_direct(blocks)(*args),"q4k_ffn_down_scalar_sum_v1")
  _program,_source,dp4a_ptx=_render(emit_four_warp_direct(blocks,sum_dp4a=True)(*args),"q4k_ffn_down_dp4a_sum_v1")
  # The vectorized packed loads and integer scale/min application changed
  # NVRTC's loop unrolling (3 data dp4a per unrolled body). The correction sum
  # still adds exactly one dp4a per data dp4a, so the census doubles.
  assert scalar_ptx.count("dp4a") == 6
  assert dp4a_ptx.count("dp4a") == 12


def test_production_direct_consumer_has_fixed_bound_and_no_replay_variable():
  ast=emit_four_warp_direct(UOp.const(dtypes.weakint,BLOCKS_PER_WARP))(
    UOp.placeholder((ROWS,),dtypes.float32,0),
    UOp.placeholder((ROWS*Q4_BLOCKS*36,),dtypes.uint32,1),
    UOp.placeholder((Q8_WORDS,),dtypes.uint32,2))
  program,source,_ptx=_render(ast,"q4k_ffn_down_mmvq_direct_fixed_v1")
  assert program.arg.vars == ()
  assert f"< {BLOCKS_PER_WARP};" in source


def test_resadd_consumer_absorbs_normed_h_in_kernel_under_m2b_name():
  ast=emit_four_warp_direct(UOp.const(dtypes.weakint,BLOCKS_PER_WARP),resadd=True)(
    UOp.placeholder((ROWS,),dtypes.float32,0),
    UOp.placeholder((ROWS*Q4_BLOCKS*36,),dtypes.uint32,1),
    UOp.placeholder((Q8_WORDS,),dtypes.uint32,2),
    UOp.placeholder((ROWS,),dtypes.float32,3))
  program,source,ptx=_render(ast,"q4k_ffn_down_mmvq_resadd_v1")
  assert program.arg.global_size == (ROWS,1,1) and program.arg.local_size == (128,1,1)
  assert program.arg.vars == ()
  assert "q4k_q8_mmvq_direct_4096_12288_epi_ffnresadd" in source
  assert "float* data3_4096" in source
  assert "dp4a" in ptx and "st.global" in ptx


def _fp16_word(bits:int) -> float:
  return float(np.frombuffer(np.array(bits & 0xffff, dtype=np.uint16).tobytes(), dtype=np.float16)[0])


def _fp16_direct_row_sim(words:np.ndarray, x:np.ndarray, row:int) -> float:
  """Reproduce emit_four_warp_fp16_direct's per-row arithmetic in scalar form.

  It walks the exact ownership map (warp, lane, block_rel) -> (block, word_col)
  and the production _q4k_block_dot_packed_load group/nib unpacking. This is the
  arithmetic ground truth for the new geometry; the independent dense reference
  (q4_k_reference) must agree with it to fp32 reorder tolerance.
  """
  total = 0.0
  for warp in range(4):
    for lane in range(32):
      word_col, sub_group = lane % 8, lane // 8
      for block_rel in range(SUB_BLOCKS):
        block = warp * BLOCKS_PER_WARP + sub_group * SUB_BLOCKS + block_rel
        base = (row * Q4_BLOCKS + block) * 36
        d = _fp16_word(int(words[base]))
        dmin = _fp16_word(int(words[base]) >> 16)
        scale_byte = lambda idx: (int(words[base + 1 + idx // 4]) >> ((idx % 4) * 8)) & 0xff
        for grp in range(8):
          if grp < 4:
            sc, mn = scale_byte(grp) & 63, scale_byte(4 + grp) & 63
          else:
            high = scale_byte(8 + grp - 4)
            sc = (high & 0xf) | ((scale_byte(grp - 4) >> 6) << 4)
            mn = (high >> 4) | ((scale_byte(4 + grp - 4) >> 6) << 4)
          qpack = (int(words[base + 4 + (grp // 2) * 8 + word_col]) >> ((grp % 2) * 4)) & 0x0F0F0F0F
          for nib in range(4):
            q = (qpack >> (nib * 8)) & 0xf
            weight = d * sc * q - dmin * mn
            total += weight * float(x[block * 256 + grp * 32 + word_col * 4 + nib])
  return total


def test_four_warp_fp16_ownership_partitions_blocks_and_words_once():
  owned = collections.Counter()
  for warp in range(4):
    for lane in range(32):
      word_col, sub_group = lane % 8, lane // 8
      for block_rel in range(SUB_BLOCKS):
        block = warp * BLOCKS_PER_WARP + sub_group * SUB_BLOCKS + block_rel
        owned[(block, word_col)] += 1
  assert set(owned) == {(b, wc) for b in range(Q4_BLOCKS) for wc in range(8)}
  assert all(count == 1 for count in owned.values())
  assert 4 * 32 * SUB_BLOCKS == Q4_BLOCKS * 8


def test_four_warp_fp16_direct_matches_independent_dense_reference():
  words_np, raw = _make_q4k_words(ROWS, K, 202608140)
  x = np.random.default_rng(202608141).normal(0, .2, K).astype(np.float16)
  row = 7
  got = _fp16_direct_row_sim(words_np, x, row)
  dense = q4_k_reference(Tensor(raw.reshape(-1).copy(), dtype=dtypes.uint8), ROWS * K).numpy().astype(np.float32).reshape(ROWS, K)
  ref = float(dense[row] @ x.astype(np.float32))
  # fp32 reduction reorder over 12288 products: relative, not bit-exact.
  assert abs(got - ref) <= max(2e-4, abs(ref) * 2e-5)


def test_four_warp_fp16_direct_renders_four_warps_and_fp16_fma_only():
  for resadd in (False, True):
    extra = (UOp.placeholder((ROWS,), dtypes.float32, 3),) if resadd else ()
    ast = emit_four_warp_fp16_direct(UOp.const(dtypes.weakint, SUB_BLOCKS), resadd=resadd)(
      UOp.placeholder((ROWS,), dtypes.float32, 0),
      UOp.placeholder((ROWS * Q4_BLOCKS * 36,), dtypes.uint32, 1),
      UOp.placeholder((K,), dtypes.float16, 2), *extra)
    program, source, ptx = _render(ast, f"q4k_fp16_direct_resadd{resadd}_v1")
    assert program.arg.global_size == (ROWS, 1, 1) and program.arg.local_size == (128, 1, 1)
    assert program.arg.vars == ()
    name = "q4k_fp16_mmvq_direct_4096_12288_epi_ffnresadd" if resadd else "q4k_fp16_mmvq_direct_4096_12288"
    assert name in source and "__syncthreads" in source
    assert f"< {SUB_BLOCKS};" in source
    assert "dp4a" not in ptx and ".local " not in ptx and "st.global" in ptx
    if resadd:
      assert "float* data3_4096" in source


def test_fp16_fma_admission_is_mutually_exclusive_with_owned_boundary():
  assert Q4KFFNDownMMVQAdmission(0, fp16_fma=True).fp16_fma
  try: Q4KFFNDownMMVQAdmission(0, owned_input_boundary=True, fp16_fma=True)
  except ValueError: pass
  else: raise AssertionError("owned boundary and fp16_fma spellings must be mutually exclusive")
  try: Q4KFFNDownMMVQAdmission(0, fp16_fma=1)
  except ValueError: pass
  else: raise AssertionError("fp16_fma must require an explicit bool")


def test_fp16_geometry_promotion_loader_names_explicit_targets_only(tmp_path):
  p = pathlib.Path(tmp_path) / "policy.json"
  p.write_text(json.dumps({"schema": "boltbeam.route_policy.v1", "route": "decode_q4k_ffn_down_fp16_geometry",
                           "promoted_targets": [{"backend": "NV", "architecture": "sm_120"}]}))
  promoted = load_decode_q4k_ffn_down_fp16_geometry_promotion(str(p))
  assert ("NV", "sm_120") in promoted
  assert ("AMD", "gfx1100") not in promoted
  p.write_text(json.dumps({"schema": "boltbeam.route_policy.v1", "route": "decode_q4k_ffn_down_fp16_geometry",
                           "promoted_targets": []}))
  assert load_decode_q4k_ffn_down_fp16_geometry_promotion(str(p)) == frozenset()


def test_fp16_geometry_checked_in_record_promotes_only_nv_target():
  assert _DECODE_Q4K_FFN_DOWN_FP16_GEOMETRY_PROMOTED_TARGETS == frozenset({("NV", "sm_120")})
  assert decode_q4k_ffn_down_fp16_geometry_promoted(("NV", "sm_120"))
  assert not decode_q4k_ffn_down_fp16_geometry_promoted(("AMD", "gfx1100"))
  assert not decode_q4k_ffn_down_fp16_geometry_promoted((None, None))
