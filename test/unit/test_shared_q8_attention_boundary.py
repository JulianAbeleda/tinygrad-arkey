from types import SimpleNamespace

from tinygrad import Tensor, dtypes
from tinygrad.llm.qk_primitives import Q4KPrimitiveLinear, Q6KPrimitiveLinear
from tinygrad.llm.shared_q8_attention import SharedQ8AttentionAdmission, shared_q8_attention_call
from tinygrad.dtype import Invalid
from tinygrad.uop.ops import Ops, UOp


def _q4(rows):
  out = object.__new__(Q4KPrimitiveLinear)
  out.in_features, out.out_features, out.decode_enabled = 4096, rows, True
  out.route_admission = SimpleNamespace(admitted=True)
  out.q4k_storage = SimpleNamespace(words=Tensor.empty(rows * 9, dtype=dtypes.uint32))
  return out


def _q6(rows):
  out = object.__new__(Q6KPrimitiveLinear)
  out.in_features, out.out_features, out.decode_enabled = 4096, rows, True
  out.route_admission = SimpleNamespace(admitted=True)
  out.q6k_storage = SimpleNamespace(halfs=Tensor.empty(rows * 9, dtype=dtypes.uint16))
  return out


def test_shared_q8_is_closed_without_an_explicit_lease():
  assert shared_q8_attention_call(None, _q4(4096), _q4(1024), _q6(1024), Tensor.empty(1, 1, 4096)) is None


def test_shared_q8_accepts_both_real_qwen_v_formats_and_rejects_shape_mismatch():
  lease = SharedQ8AttentionAdmission(0)
  assert shared_q8_attention_call(lease, _q4(4096), _q4(1024), _q4(1024), Tensor.empty(1, 1, 4096)) is not None
  assert shared_q8_attention_call(lease, _q4(4096), _q4(1024), _q6(1024), Tensor.empty(1, 2, 4096)) is None


def test_shared_q8_builds_one_provider_and_exactly_three_consumers(monkeypatch):
  import tinygrad.llm.shared_q8_attention as route
  seen = []
  def fake_execute(_output, *inputs, program):
    seen.append((inputs, program))
    return Tensor.empty(*program.output_spec.shape, dtype=program.output_spec.dtype)
  monkeypatch.setattr(route, "execute_promoted_program", fake_execute)
  out = shared_q8_attention_call(SharedQ8AttentionAdmission(7), _q4(4096), _q4(1024), _q6(1024), Tensor.empty(1, 1, 4096))
  assert tuple(x.shape for x in out) == ((1, 1, 4096), (1, 1, 1024), (1, 1, 1024))
  assert len(seen) == 4
  assert {program.route_id for _inputs, program in seen} == {"decode_shared_q8_attention"}
  # All Q/K/V consumers receive the same packed activation and scale UOps;
  # this guards against accidentally reverting to three hidden providers.
  consumers = seen[1:]
  assert len({id(inputs[1].uop) for inputs, _program in consumers}) == 1
  assert {program.provenance.value for _inputs, program in seen} == {"machine_search_generated"}


def test_shared_q8_q4_v_reuses_same_provider_and_q4_consumer_abi(monkeypatch):
  import tinygrad.llm.shared_q8_attention as route
  seen = []
  def fake_execute(_output, *inputs, program):
    seen.append((inputs, program))
    return Tensor.empty(*program.output_spec.shape, dtype=program.output_spec.dtype)
  monkeypatch.setattr(route, "execute_promoted_program", fake_execute)
  out = shared_q8_attention_call(SharedQ8AttentionAdmission(4), _q4(4096), _q4(1024), _q4(1024), Tensor.empty(1, 1, 4096))
  assert tuple(x.shape for x in out) == ((1, 1, 4096), (1, 1, 1024), (1, 1, 1024))
  assert len(seen) == 4 and all("q4q4q4.blk4" in program.program_id for _inputs,program in seen)
  assert len({id(inputs[1].uop) for inputs,_program in seen[1:]}) == 1
  assert seen[-1][1].emitter(UOp.placeholder((1024,),dtypes.float32,0),
    UOp.placeholder((1024*16*144,),dtypes.uint32,1),UOp.placeholder((1152,),dtypes.uint32,2)).arg.name == "q4k_q8_dp4a_1024_4096"


def test_cooperative_q4_lease_reuses_one_provider_and_substitutes_every_q4(monkeypatch):
  import tinygrad.llm.shared_q8_attention as route
  seen=[]
  def fake_execute(_output,*inputs,program):
    seen.append((inputs,program)); return Tensor.empty(*program.output_spec.shape,dtype=program.output_spec.dtype)
  monkeypatch.setattr(route,"execute_promoted_program",fake_execute)
  out=shared_q8_attention_call(SharedQ8AttentionAdmission(3,cooperative_q4=True),_q4(4096),_q4(1024),_q6(1024),
    Tensor.empty(1,1,4096),UOp.variable("start_pos",0,1023).bind(513))
  assert tuple(x.shape for x in out)==((1,1,4096),(1,1,1024),(1,1,1024))
  assert len(seen)==4 and seen[0][1].program_id.endswith(".provider")
  q4_programs=[p for _i,p in seen if p.program_id.endswith(".coop")]
  assert len(q4_programs)==2 and {p.output_spec.shape for p in q4_programs}=={(4096,4),(1024,4)}
  assert len({id(inputs[1].uop) for inputs,_p in seen[1:]})==1
  emitted=[p.emitter(UOp.placeholder(p.output_spec.shape,dtypes.float32,0),
    UOp.placeholder((p.output_spec.shape[0]*16*36,),dtypes.uint32,1),
    UOp.placeholder((1152,),dtypes.uint32,2)) for p in q4_programs]
  assert {ast.arg.name for ast in emitted}=={
    "q4k_warp_coop_q8_dp4a_partial_4096_4096","q4k_warp_coop_q8_dp4a_partial_1024_4096"}
  assert all(not any(u.op is Ops.BIND for u in ast.toposort()) for ast in emitted)
  assert all([u.arg for u in ast.toposort() if u.op is Ops.DEFINE_VAR]==[("start_pos",0,1023)] for ast in emitted)

def test_q6_direct_lease_changes_only_q6_v_consumer_and_keeps_the_provider_shared(monkeypatch):
  import tinygrad.llm.shared_q8_attention as route
  seen=[]
  def fake_execute(_output,*inputs,program):
    seen.append((inputs,program)); return Tensor.empty(*program.output_spec.shape,dtype=program.output_spec.dtype)
  monkeypatch.setattr(route,"execute_promoted_program",fake_execute)
  out=shared_q8_attention_call(SharedQ8AttentionAdmission(3,q6_direct_output=True),_q4(4096),_q4(1024),_q6(1024),
    Tensor.empty(1,1,4096))
  assert tuple(x.shape for x in out)==((1,1,4096),(1,1,1024),(1,1,1024))
  assert len(seen)==4 and sum("q6_direct" in p.program_id for _i,p in seen)==1
  direct=seen[-1][1]
  assert direct.output_spec.shape == (1024,)
  ast=direct.emitter(UOp.placeholder((1024,),dtypes.float32,0),UOp.placeholder((1024*16*210,),dtypes.uint16,1),
                     UOp.placeholder((1152,),dtypes.uint32,2))
  assert ast.arg.name == "q6k_q8_warp_direct_1024_4096"
  assert len({id(inputs[1].uop) for inputs,_p in seen[1:]})==1


def test_shared_q8_consumes_only_explicit_reduce_output_marker_sources(monkeypatch):
  import tinygrad.llm.shared_q8_attention as route
  seen = []
  def fake_execute(_output, *inputs, program):
    seen.append((inputs, program))
    return Tensor.empty(*program.output_spec.shape, dtype=program.output_spec.dtype)
  monkeypatch.setattr(route, "execute_promoted_program", fake_execute)
  raw, weight = Tensor.empty(1, 1, 4096), Tensor.empty(4096, dtype=dtypes.float16)
  ordinary = Tensor.empty(1, 1, 4096)
  marked = ordinary._semantic_reduce_output_rmsnorm(raw, ordinary, weight, 1e-6)
  shared_q8_attention_call(SharedQ8AttentionAdmission(0), _q4(4096), _q4(1024), _q6(1024), marked, norm_weight=weight)
  provider_inputs, provider = seen[0]
  assert provider.program_id.endswith(".rmsnorm_provider")
  assert provider.emitter(UOp.placeholder((1152,), dtypes.uint32, 0), UOp.placeholder((4096,), dtypes.float32, 1),
                          UOp.placeholder((4096,), dtypes.float16, 2)).arg.name == "rmsnorm_q8_1_llama_provider_4096"
  assert len(provider_inputs) == 2 and provider_inputs[0].uop.base is raw.uop.base and provider_inputs[1] is weight


def test_shared_q8_does_not_infer_rmsnorm_from_an_ordinary_value(monkeypatch):
  import tinygrad.llm.shared_q8_attention as route
  seen = []
  def fake_execute(_output, *inputs, program):
    seen.append((inputs, program))
    return Tensor.empty(*program.output_spec.shape, dtype=program.output_spec.dtype)
  monkeypatch.setattr(route, "execute_promoted_program", fake_execute)
  x, weight = Tensor.empty(1, 1, 4096), Tensor.empty(4096, dtype=dtypes.float16)
  shared_q8_attention_call(SharedQ8AttentionAdmission(0), _q4(4096), _q4(1024), _q6(1024), x, norm_weight=weight)
  assert seen[0][1].program_id.endswith(".provider") and not seen[0][1].program_id.endswith(".rmsnorm_provider")


def test_shared_q8_lease_cannot_widen_target_or_block_scope():
  import pytest
  with pytest.raises(ValueError): SharedQ8AttentionAdmission(-1)
  with pytest.raises(ValueError): SharedQ8AttentionAdmission(0, ("AMD", "gfx1100"))
  with pytest.raises(ValueError): SharedQ8AttentionAdmission(0, cooperative_q4=1)
  with pytest.raises(ValueError): SharedQ8AttentionAdmission(0, q6_direct_output=1)


def test_shared_q8_emitters_keep_the_authoritative_source_identity():
  # KernelInfo feeds generated NV source.  These names are the source-hash
  # identity from the isolated real-payload PASS, so this is a hermetic guard
  # against source-only drift at the promoted boundary.
  from tinygrad.llm.shared_q8_attention import _emit_q4, _emit_q6, _emit_q6_warp_direct
  out = UOp.placeholder((4096,), dtypes.float32, 0)
  q4 = _emit_q4(4096)(out, UOp.placeholder((4096*16*144,), dtypes.uint32, 1),
                      UOp.placeholder((1152,), dtypes.uint32, 2))
  q6 = _emit_q6(1024)(UOp.placeholder((1024,), dtypes.float32, 4), UOp.placeholder((1024*16*210,), dtypes.uint16, 5),
                      UOp.placeholder((1152,), dtypes.uint32, 6))
  assert q4.arg.name == "q4k_q8_dp4a_4096_4096"
  assert q6.arg.name == "q6k_q8_dp4a_1024_4096"
  direct=_emit_q6_warp_direct(1024)(UOp.placeholder((1024,),dtypes.float32,7),UOp.placeholder((1024*16*210,),dtypes.uint16,8),
                                  UOp.placeholder((1152,),dtypes.uint32,9))
  assert direct.arg.name == "q6k_q8_warp_direct_1024_4096"

def test_fused_rmsnorm_q8_provider_passes_final_nv_program_spec():
  from tinygrad.codegen import full_rewrite_to_sink
  from tinygrad.helpers import Context, Target
  from tinygrad.renderer.cuda import CUDARenderer
  from tinygrad.llm.shared_q8_attention import _emit_rmsnorm_q8_provider
  from tinygrad.uop.ops import ReduceOutputSpec
  for x_dtype,out_dtype in ((dtypes.float32,dtypes.float32),(dtypes.float16,dtypes.float16)):
    spec=ReduceOutputSpec(1,4096,1e-6,out_dtype)
    sink=_emit_rmsnorm_q8_provider(spec,x_dtype,dtypes.float16)(
      UOp.placeholder((1152,),dtypes.uint32,0),UOp.placeholder((4096,),x_dtype,1),
      UOp.placeholder((4096,),dtypes.float16,2))
    with Context(SPEC=1):
      lowered=full_rewrite_to_sink(sink,CUDARenderer(Target("NV",arch="sm_120"),use_nvcc=True))
    assert lowered.op is Ops.SINK and all(not (u.op is Ops.CONST and u.arg is Invalid) for u in lowered.toposort())

def test_q6_direct_consumer_passes_final_nv_program_spec():
  from tinygrad.codegen import full_rewrite_to_sink
  from tinygrad.helpers import Context, Target
  from tinygrad.renderer.cuda import CUDARenderer
  from tinygrad.llm.shared_q8_attention import _emit_q6_warp_direct
  sink=_emit_q6_warp_direct(1024)(UOp.placeholder((1024,),dtypes.float32,0),
    UOp.placeholder((1024*16*210,),dtypes.uint16,1),UOp.placeholder((1152,),dtypes.uint32,2))
  with Context(SPEC=1): lowered=full_rewrite_to_sink(sink,CUDARenderer(Target("NV",arch="sm_120"),use_nvcc=True))
  assert lowered.op is Ops.SINK and all(not (u.op is Ops.CONST and u.arg is Invalid) for u in lowered.toposort())


def test_q6_direct_consumer_uses_both_llama_mmvq_scale_slots():
  # vec_dot_q6_K_q8_1_impl_mmvq reads scales[scale_offset] and
  # scales[scale_offset + 4] for the two packed int8x4 terms. The old flat
  # consumer reused the first scale for both terms; this pins the two
  # halfword loads so that regression cannot silently re-enter.
  from tinygrad.codegen import to_program
  from tinygrad.helpers import Target
  from tinygrad.renderer.cuda import CUDARenderer
  from tinygrad.llm.shared_q8_attention import _emit_q6_warp_direct
  from tinygrad.uop.ops import Ops
  ast=_emit_q6_warp_direct(1024)(UOp.placeholder((1024,),dtypes.float32,0),
    UOp.placeholder((1024*16*210,),dtypes.uint16,1),UOp.placeholder((1152,),dtypes.uint32,2))
  prog=to_program(ast,CUDARenderer(Target.parse("NV:CUDA:sm_120")))
  src=next(u.arg for u in prog.src if u.op is Ops.SOURCE)
  assert "(alu8+96)" in src and "(alu8+98)" in src
