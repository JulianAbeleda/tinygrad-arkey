import numpy as np

from tinygrad import Tensor, dtypes, nn
from tinygrad.uop.ops import Ops

def _norm(weight, enabled=True):
  n = nn.RMSNorm(4096, eps=1e-6)
  n.weight = weight
  return n

def _apply(norm, x, enabled=True):
  out = norm(x)
  return out._semantic_reduce_output_rmsnorm(x, out, norm.weight, norm.eps) if enabled and norm.weight is not None else out

def _names(out):
  linear, _ = out.linear_with_vars()
  return [x.src[0].arg.name for x in linear.src]

def test_marker_is_default_off_and_source_visible():
  x = Tensor.empty(1, 4096, dtype=dtypes.float16)
  w = Tensor.ones(4096, dtype=dtypes.float16)
  n = _norm(w)
  assert _apply(n, x, False).uop.op is not Ops.REDUCE_OUTPUT
  marked = _apply(n, x)
  assert marked.uop.op is Ops.REDUCE_OUTPUT and marked.uop.src[0].op is not Ops.REDUCE_OUTPUT

def test_marker_cannot_be_enabled_through_rmsnorm_attribute():
  x = Tensor.empty(1, 4096, dtype=dtypes.float16)
  n = _norm(Tensor.ones(4096, dtype=dtypes.float16))
  n._reduce_output_rmsnorm_promoted = True
  assert n(x).uop.op is not Ops.REDUCE_OUTPUT
  assert _apply(n, x).uop.op is Ops.REDUCE_OUTPUT

def test_realized_identity_views_lower_to_one_ordinary_call():
  x = Tensor.randn(1, 4096, dtype=dtypes.float16).realize()
  w = Tensor.ones(4096, dtype=dtypes.float16).realize()
  out = _apply(_norm(w), x)
  assert _names(out) == ["reduce_output_rmsnorm_1_4096"]

def test_lazy_input_fails_closed_without_materialization():
  x = Tensor.randn(1, 4096, dtype=dtypes.float16).realize()
  w = Tensor.ones(4096, dtype=dtypes.float16).realize()
  names = _names(_apply(_norm(w), x+x))
  assert names and "reduce_output_rmsnorm_1_4096" not in names

def test_marker_records_pre_callify_identity_instead_of_inferring_param_safety():
  x = Tensor.randn(1,4096,dtype=dtypes.float16).realize()
  w = Tensor.ones(4096,dtype=dtypes.float16).realize()
  assert _apply(_norm(w), x).uop.arg.input_identity_at_marker is True
  assert _apply(_norm(w), x+x).uop.arg.input_identity_at_marker is False

def test_precompiled_function_output_is_marker_identity_but_views_and_nonprecompiled_are_not():
  from tinygrad.function import function
  x = Tensor.empty(1,4096,dtype=dtypes.float16)
  w = Tensor.ones(4096,dtype=dtypes.float16)
  @function(precompile=True)
  def precompiled(v): return v + 1
  @function
  def ordinary(v): return v + 1
  exact = precompiled(x)
  assert exact.uop.has_precompiled_output_identity()
  assert _apply(_norm(w), exact).uop.arg.input_identity_at_marker is True
  assert _apply(_norm(w), exact.contiguous()).uop.arg.input_identity_at_marker is True
  assert not ordinary(x).uop.has_precompiled_output_identity()
  assert _apply(_norm(w), ordinary(x).contiguous()).uop.arg.input_identity_at_marker is False
  assert _apply(_norm(w), (x+1).contiguous()).uop.arg.input_identity_at_marker is False
  assert not exact.permute(1,0).uop.has_precompiled_output_identity()
  assert not exact.shrink(((0,1),(1,4096))).uop.has_precompiled_output_identity()

def test_identity_view_rejects_offsets_movements_and_dependencies():
  from tinygrad.schedule.rangeify import _identity_buffer_view
  from tinygrad.uop.ops import UOp
  p=UOp.param(0,dtypes.float16,shape=(4096,),device="NV")
  assert _identity_buffer_view(p.reshape(1,4096)) is p
  assert _identity_buffer_view(p.reshape(64,64).permute((1,0))) is None
  assert _identity_buffer_view(p.shrink(((1,4096),))) is None
  assert _identity_buffer_view(p.after(UOp(Ops.NOOP))) is None

def test_callified_precompiled_output_retains_exact_after_dependency():
  from tinygrad.callify import pm_early_transform_tensor_graph
  from tinygrad.function import function
  from tinygrad.schedule.rangeify import _identity_buffer_view, lower_reduce_output_store
  from tinygrad.uop.ops import UOp, graph_rewrite
  # Device-tagged UOps only: this is a CPU structural test and performs no NV execution.
  x = Tensor.empty(1,4096,dtype=dtypes.float16,device="NV")
  w = Tensor.empty(4096,dtype=dtypes.float16,device="NV")
  @function(precompile=True)
  def producer(v): return v + 1
  marked = _apply(_norm(w), producer(x).contiguous())
  callified = graph_rewrite(marked.uop, pm_early_transform_tensor_graph, name="test precompiled output callify")
  call_input = callified.src[1]
  assert call_input.op is Ops.AFTER and _identity_buffer_view(call_input) is call_input
  target = UOp.new_buffer("NV", 4096, dtypes.float16).reshape(1,4096)
  lowered = lower_reduce_output_store(target.store(callified))
  assert lowered is not None and lowered.op is Ops.CALL
  assert lowered.src[2] is call_input  # body, output, dependency-bearing input, weight
  # Same bytes with the producer dependency removed must fail closed.
  assert _identity_buffer_view(call_input.src[0].after(UOp(Ops.NOOP))) is None

def test_owner_preserving_direct_reduce_output_carrier_lowers_and_retains_owner():
  from tinygrad.function import function
  from tinygrad.llm.memory_semantics import runtime_activation
  from tinygrad.schedule.rangeify import lower_reduce_output_store
  from tinygrad.callify import pm_early_transform_tensor_graph
  from tinygrad.uop.ops import UOp, graph_rewrite
  x = Tensor.empty(1,4096,dtype=dtypes.float16,device="NV")
  w = Tensor.empty(4096,dtype=dtypes.float16,device="NV")
  @function(precompile=True)
  def producer(v): return v + 1
  marked = graph_rewrite(_apply(_norm(w), producer(x).contiguous()).uop,
                         pm_early_transform_tensor_graph, name="owner carrier callify")
  target = UOp.new_buffer("NV",4096,dtypes.float16).reshape(1,4096)
  carrier = runtime_activation(marked)
  lowered = lower_reduce_output_store(target.store(carrier), carrier)
  assert lowered is not None and lowered.op is Ops.CALL
  # The exact original owner is attached to this emitters's fixed output slot;
  # the input retains its dependency-bearing AFTER contract.
  assert lowered.src[0].arg.memory_semantic_slots == ((0, carrier.arg),)
  assert lowered.src[1] is target.base
  assert lowered.src[2].op is Ops.AFTER

def test_owner_preserving_selector_rejects_nested_and_movement_carriers():
  from tinygrad.llm.memory_semantics import runtime_activation
  from tinygrad.schedule.rangeify import lower_reduce_output_store
  from tinygrad.uop.ops import ReduceOutputSpec, UOp
  target = UOp.new_buffer("NV",4096,dtypes.float16)
  x, w = UOp.new_buffer("NV",4096,dtypes.float16), UOp.new_buffer("NV",4096,dtypes.float16)
  marker = UOp(Ops.REDUCE_OUTPUT,dtypes.float16,(target,x,w),ReduceOutputSpec(1,4096,1e-6,dtypes.float16,input_identity_at_marker=True))
  owned = runtime_activation(marker)
  nested = UOp(Ops.MEMORY_SEMANTIC, owned.dtype, (owned,), owned.arg)
  assert lower_reduce_output_store(target.store(nested)) is None
  # A movement between the exact carrier and the marker is not the admitted spelling.
  assert lower_reduce_output_store(target.store(runtime_activation(marker.reshape(64,64).permute(1,0).reshape(4096)))) is None

def test_owner_preserving_carrier_selects_in_full_recursive_schedule():
  from tinygrad.function import function
  from tinygrad.helpers import Context
  from tinygrad.llm.memory_semantics import runtime_activation
  x = Tensor.empty(1,4096,dtype=dtypes.float16,device="NV")
  w = Tensor.empty(4096,dtype=dtypes.float16,device="NV")
  @function(precompile=True)
  def producer(v): return v + 1
  value = runtime_activation(_apply(_norm(w), producer(x).contiguous()))
  with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1): linear,_ = value.linear_with_vars()
  assert [call.src[0].arg.name for call in linear.src] == ["test", "reduce_output_rmsnorm_1_4096"]

def test_reduce_output_trace_records_bounded_parent_chain_without_uops():
  from tinygrad.function import function
  from tinygrad.helpers import Context
  from tinygrad.llm.memory_semantics import runtime_activation
  from tinygrad.llm.reduce_output_trace import REDUCE_OUTPUT_TRACE, reset_reduce_output_trace, reduce_output_trace_snapshot
  x = Tensor.empty(1,4096,dtype=dtypes.float16,device="NV")
  w = Tensor.empty(4096,dtype=dtypes.float16,device="NV")
  @function(precompile=True)
  def producer(v): return v + 1
  reset_reduce_output_trace()
  with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, REDUCE_OUTPUT_TRACE=1):
    runtime_activation(_apply(_norm(w), producer(x).contiguous())).linear_with_vars()
  chains = reduce_output_trace_snapshot()["_details"]["before_rangeify_parent_chain"]
  assert chains and all("REDUCE_OUTPUT" in chain and "shape=" in chain and "dtype=" in chain for chain in chains)

def test_precompiled_output_redirect_survives_full_recursive_schedule():
  from tinygrad.function import function
  from tinygrad.helpers import Context
  x = Tensor.empty(1,4096,dtype=dtypes.float16,device="NV")
  w = Tensor.empty(4096,dtype=dtypes.float16,device="NV")
  @function(precompile=True)
  def producer(v): return v + 1
  value = producer(x).contiguous()
  marked = _apply(_norm(w), value)
  with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1): linear,_ = marked.linear_with_vars()
  names = [call.src[0].arg.name for call in linear.src]
  assert names == ["test", "reduce_output_rmsnorm_1_4096"]

def test_owned_contiguous_candidate_survives_full_recursive_schedule():
  from tinygrad.function import function
  from tinygrad.helpers import Context
  from tinygrad.llm.memory_semantics import runtime_activation
  x = Tensor.empty(1,1,4096,dtype=dtypes.float32,device="NV")
  w = Tensor.empty(4096,dtype=dtypes.float16,device="NV")
  @function(precompile=True,allow_implicit=True)
  def producer(v): return runtime_activation((runtime_activation(v)+1).contiguous())
  value = runtime_activation(producer(x).contiguous())
  marked = _apply(_norm(w), value)
  assert marked.uop.arg.input_identity_at_marker is False
  assert marked.uop.arg.owned_contiguous_candidate is True
  with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1): linear,_ = marked.linear_with_vars()
  assert [call.src[0].arg.name for call in linear.src] == ["test","reduce_output_rmsnorm_1_4096"]

def test_owned_candidate_inside_precompiled_consumer_binds_exact_invocation_input():
  from tinygrad.function import function
  from tinygrad.helpers import Context
  from tinygrad.llm.memory_semantics import runtime_activation
  x = Tensor.empty(1,1,4096,dtype=dtypes.float32,device="NV")
  w = Tensor.empty(4096,dtype=dtypes.float16,device="NV")
  norm = _norm(w)
  @function(precompile=True,allow_implicit=True)
  def producer(v): return runtime_activation((runtime_activation(v)+1).contiguous())
  @function(precompile=True,allow_implicit=True)
  def consumer(v): return runtime_activation(_apply(norm,v).contiguous())
  value = consumer(runtime_activation(producer(x).contiguous()))
  with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1): linear,_ = value.linear_with_vars()
  assert [call.src[0].arg.name for call in linear.src] == ["test","reduce_output_rmsnorm_1_4096"]

def test_nested_invocation_input_proof_rejects_default_off_noncall_and_movement():
  from tinygrad.function import function
  from tinygrad.helpers import Context
  from tinygrad.llm.memory_semantics import runtime_activation
  x = Tensor.empty(1,1,4096,dtype=dtypes.float32,device="NV")
  norm = _norm(Tensor.empty(4096,dtype=dtypes.float16,device="NV"))
  @function(precompile=True,allow_implicit=True)
  def producer(v): return runtime_activation((runtime_activation(v)+1).contiguous())
  @function(precompile=True,allow_implicit=True)
  def consumer(v): return runtime_activation(_apply(norm,v).contiguous())
  exact = runtime_activation(producer(x).contiguous())
  moved = runtime_activation(producer(x).reshape(64,64).permute(1,0).reshape(1,1,4096).contiguous())
  noncall = runtime_activation((x+1).contiguous())
  with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=0):
    assert "reduce_output_rmsnorm_1_4096" not in _names(consumer(exact))
  with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1):
    assert "reduce_output_rmsnorm_1_4096" not in _names(consumer(moved))
    assert "reduce_output_rmsnorm_1_4096" not in _names(consumer(noncall))


def test_callify_flags_leave_non_reduce_output_precompiled_functions_untouched():
  """The candidate callify flags must only alter reduce-output-bearing
  functions.  A precompiled residual-family function (no REDUCE_OUTPUT marker)
  must transform to the exact same graph under the candidate flags as under
  the closed control graph; otherwise the census's E_32_32_4 residual side
  effects (-36/+36/+54/-18/-71) return and the wall bracket regresses.  Each
  callify transform allocates its output buffer, so the two arms are compared
  structurally with allocation identity (UNIQUE ids) elided."""
  from tinygrad.callify import pm_early_transform_tensor_graph
  from tinygrad.function import function
  from tinygrad.helpers import Context
  from tinygrad.uop.ops import Ops, UOp, graph_rewrite
  x = Tensor.empty(1,4096,dtype=dtypes.float16,device="NV")
  @function(precompile=True)
  def residual(v): return v + 1
  value = residual(x).contiguous()
  control = graph_rewrite(value.uop, pm_early_transform_tensor_graph, name="control residual callify")
  with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
    candidate = graph_rewrite(value.uop, pm_early_transform_tensor_graph, name="candidate residual callify")
  assert control is not None and candidate is not None
  def structural(u:UOp):
    if u.op is Ops.UNIQUE: return ("UNIQUE",)
    return (u.op, tuple(structural(s) for s in u.src), u.arg)
  assert structural(control) == structural(candidate)
  # No owned-redirect contract leaks onto a non-route function: the opaque
  # CALL carries no output slots in either arm, so rangeify cannot admit the
  # owned precompiled-output proof for it.
  for root in (control, candidate):
    call = next(u for u in root.toposort() if u.op is Ops.CALL and bool(getattr(u.arg, "precompile", False)))
    assert call.arg.precompiled_output_slots == ()

def test_precompiled_output_slot_contract_is_closed_with_redirect_default():
  from tinygrad.function import function
  from tinygrad.helpers import Context
  x = Tensor.empty(1,4096,dtype=dtypes.float16,device="NV")
  w = Tensor.empty(4096,dtype=dtypes.float16,device="NV")
  @function(precompile=True)
  def producer(v): return v + 1
  value = producer(x).contiguous()
  marked = _apply(_norm(w), value)
  with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=0): linear,_ = marked.linear_with_vars()
  assert "reduce_output_rmsnorm_1_4096" not in [call.src[0].arg.name for call in linear.src]

def test_precompiled_output_redirect_does_not_admit_movement_view():
  from tinygrad.function import function
  from tinygrad.helpers import Context
  x = Tensor.empty(1,4096,dtype=dtypes.float16,device="NV")
  w = Tensor.empty(4096,dtype=dtypes.float16,device="NV")
  @function(precompile=True)
  def producer(v): return v + 1
  moved = producer(x).reshape(64,64).permute(1,0).reshape(1,4096).contiguous()
  marked = _apply(_norm(w), moved)
  with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1): linear,_ = marked.linear_with_vars()
  assert "reduce_output_rmsnorm_1_4096" not in [call.src[0].arg.name for call in linear.src]

def test_owned_contiguous_hint_alone_rejects_movement_and_non_call_outputs():
  from tinygrad.function import function
  from tinygrad.helpers import Context
  from tinygrad.llm.memory_semantics import runtime_activation
  x = Tensor.empty(1,4096,dtype=dtypes.float16,device="NV")
  w = Tensor.empty(4096,dtype=dtypes.float16,device="NV")
  @function(precompile=True)
  def producer(v): return v + 1
  values = (
    runtime_activation(producer(x).reshape(64,64).permute(1,0).reshape(1,4096).contiguous()),
    runtime_activation((x+1).contiguous()),
  )
  for value in values:
    marked = _apply(_norm(w),value)
    assert marked.uop.arg.input_identity_at_marker is False and marked.uop.arg.owned_contiguous_candidate is True
    with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1): linear,_ = marked.linear_with_vars()
    assert "reduce_output_rmsnorm_1_4096" not in [call.src[0].arg.name for call in linear.src]

def _synthetic_precompiled_multioutput_after(output_slots=(1,2), alias=False, extra_dependency=False):
  from tinygrad.uop.ops import CallInfo, UOp
  payload = UOp.new_buffer("NV",4,dtypes.float16)
  out0,out1 = UOp.new_buffer("NV",4,dtypes.float16),UOp.new_buffer("NV",4,dtypes.float16)
  p0,p1 = UOp.param(1,dtypes.float16,(4,),"NV"),UOp.param(2,dtypes.float16,(4,),"NV")
  body_sink = UOp.sink(p0.store(UOp.param(0,dtypes.float16,(4,),"NV")),
                       p1.store(UOp.param(0,dtypes.float16,(4,),"NV")))
  body = UOp(Ops.LINEAR,src=(body_sink.call(payload,out0,out1),))
  args = (out0 if alias else payload,out0,out1)
  call = UOp(Ops.CALL,dtypes.void,(body,*args),CallInfo(name="multi",precompile=True,precompiled_output_slots=output_slots))
  return out0.after(call,UOp(Ops.NOOP)) if extra_dependency else out0.after(call)

def test_precompiled_output_slot_contract_rejects_unlisted_multioutput_slot_and_alias():
  from tinygrad.schedule.rangeify import _identity_buffer_view
  assert _identity_buffer_view(_synthetic_precompiled_multioutput_after()) is not None
  assert _identity_buffer_view(_synthetic_precompiled_multioutput_after(output_slots=(2,))) is None
  assert _identity_buffer_view(_synthetic_precompiled_multioutput_after(alias=True)) is None

def test_precompiled_output_slot_contract_requires_exact_invocation_lifetime():
  from tinygrad.schedule.rangeify import _identity_buffer_view
  assert _identity_buffer_view(_synthetic_precompiled_multioutput_after(extra_dependency=True)) is None

def test_owned_contiguous_hint_cannot_admit_aliased_invocation_output():
  from tinygrad.schedule.rangeify import lower_reduce_output_store
  from tinygrad.uop.ops import ReduceOutputSpec, UOp
  x = _synthetic_precompiled_multioutput_after(alias=True)
  target = UOp.new_buffer("NV",4,dtypes.float16)
  weight = UOp.new_buffer("NV",4,dtypes.float16)
  marker = UOp(Ops.REDUCE_OUTPUT,dtypes.float16,(target,x,weight),
               ReduceOutputSpec(1,4096,1e-6,dtypes.float16,owned_contiguous_candidate=True))
  assert lower_reduce_output_store(target.store(marker)) is None

def _typed_semantic_call_input_chain(*, moved=False, scratch=True):
  """Exact CPU-only reproduction of the production CALL-input boundary."""
  from tinygrad.llm.memory_semantics import runtime_scratch, runtime_activation
  from tinygrad.uop.ops import ReduceOutputSpec, UOp
  x,weight = UOp.param(3,dtypes.float16,(1,1,4096),"NV"),UOp.param(9,dtypes.float16,(4096,),"NV")
  ordinary = x  # The marker fallback is irrelevant if its dedicated producer lowers.
  marker = UOp(Ops.REDUCE_OUTPUT,dtypes.float16,(ordinary,x,weight),ReduceOutputSpec(1,4096,1e-6,dtypes.float16,input_identity_at_marker=True))
  carrier = (runtime_scratch if scratch else runtime_activation)(marker)
  flat = carrier.reshape((4096,))
  if moved: flat = flat.reshape((64,64)).permute((1,0)).reshape((4096,))
  return flat.contiguous()

def test_typed_semantic_call_input_lowers_to_precompiled_producer_with_after_dependency():
  from tinygrad.callify import CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER, callify_typed_semantic_call_inputs
  from tinygrad.helpers import Context
  from tinygrad.uop.ops import UOp
  typed = _typed_semantic_call_input_chain()
  out = UOp.new_buffer("NV",4096,dtypes.float16)
  # An opaque consumer is enough to test the generic boundary; it does not
  # know that its input was originally a REDUCE_OUTPUT semantic producer.
  consumer = UOp.sink(UOp.param(0,dtypes.float16,(4096,),"NV").store(UOp.param(1,dtypes.float16,(4096,),"NV"))).call(out,typed)
  assert callify_typed_semantic_call_inputs(consumer) is None
  with Context(CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1): lowered = callify_typed_semantic_call_inputs(consumer)
  assert lowered is not None
  arg = lowered.src[2]
  assert arg.dtype is dtypes.float16 and arg.shape == (4096,) and arg.numel() == 4096
  assert arg.op is Ops.RESHAPE and arg.src[0].op is Ops.AFTER
  producer = arg.src[0].src[1]
  assert producer.op is Ops.CALL and producer.arg.precompile
  assert producer.arg.precompiled_output_slots == (2,)
  stores = [u for u in producer.src[0].toposort() if u.op is Ops.STORE]
  assert len(stores) == 1 and stores[0].src[1].op is Ops.REDUCE_OUTPUT

def test_typed_semantic_call_input_prepass_reaches_late_store_selector_without_adapter():
  from tinygrad.callify import (CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER, pm_early_transform_tensor_graph,
                                pm_typed_semantic_call_input)
  from tinygrad.helpers import Context
  from tinygrad.schedule import lower_sink_to_linear
  from tinygrad.uop.ops import UOp, graph_rewrite
  typed = _typed_semantic_call_input_chain()
  out = UOp.new_buffer("NV",4096,dtypes.float16)
  consumer = UOp.sink(UOp.param(0,dtypes.float16,(4096,),"NV").store(UOp.param(1,dtypes.float16,(4096,),"NV"))).call(out,typed)
  with Context(CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
    rewritten = graph_rewrite(consumer,pm_typed_semantic_call_input,bottom_up=False,name="typed producer prepass")
    rewritten = graph_rewrite(rewritten,pm_early_transform_tensor_graph,name="typed producer early callify")
    producer = next(u for u in rewritten.toposort()
                    if u.op is Ops.CALL and u.arg.name == "typed_semantic_reduce_output_producer")
    linear = lower_sink_to_linear(producer.src[0])
  assert [call.src[0].arg.name for call in linear.src] == ["reduce_output_rmsnorm_1_4096"]
  # The producer result is the consumer's dependency-bearing value, not a
  # second transport CONTIGUOUS/COPY adapter.
  consumer_arg = rewritten.src[2]
  assert consumer_arg.op is Ops.RESHAPE and consumer_arg.src[0].op is Ops.AFTER
  assert not any(u.op in {Ops.COPY, Ops.CONTIGUOUS} for u in consumer_arg.toposort())

def test_typed_semantic_call_input_rejects_movement_wrong_owner_and_alias():
  from tinygrad.callify import _typed_semantic_reduce_output_input
  assert _typed_semantic_reduce_output_input(_typed_semantic_call_input_chain()) is not None
  assert _typed_semantic_reduce_output_input(_typed_semantic_call_input_chain(moved=True)) is None
  assert _typed_semantic_reduce_output_input(_typed_semantic_call_input_chain(scratch=False)) is None
  from tinygrad.uop.ops import ReduceOutputSpec, UOp
  x=UOp.param(3,dtypes.float16,(1,1,4096),"NV")
  marker=UOp(Ops.REDUCE_OUTPUT,dtypes.float16,(x,x,x),ReduceOutputSpec(1,4096,1e-6,dtypes.float16,input_identity_at_marker=True))
  from tinygrad.llm.memory_semantics import runtime_scratch
  assert _typed_semantic_reduce_output_input(runtime_scratch(marker).reshape((4096,)).contiguous()) is None

def test_precompiled_contiguous_output_replay_has_invocation_lifetime():
  from tinygrad import TinyJit
  from tinygrad.function import function
  @function(precompile=True)
  def producer(v): return v + 1
  @TinyJit
  def consume(v): return producer(v).contiguous() * 2
  # Ignore, capture, and replay with different physical inputs. A stale alias
  # or missing producer dependency returns a previous invocation's value.
  for value in (1.0, 3.0, 7.0):
    got = consume(Tensor.full((8,), value).contiguous()).numpy()
    np.testing.assert_allclose(got, np.full((8,), (value+1)*2, dtype=np.float32), rtol=0, atol=0)

def test_decode_helper_marks_only_identity_qualified_input():
  from tinygrad.function import function
  from tinygrad.llm.model import _decode_reduce_output_rmsnorm
  x = Tensor.empty(1,4096,dtype=dtypes.float16)
  w = Tensor.ones(4096,dtype=dtypes.float16)
  @function(precompile=True)
  def producer(v): return v + 1
  admitted = _decode_reduce_output_rmsnorm(_norm(w), producer(x).contiguous(), True).uop
  declined = _decode_reduce_output_rmsnorm(_norm(w), x+1, True).uop
  assert admitted.op is Ops.REDUCE_OUTPUT and admitted.arg.input_identity_at_marker is True
  assert declined.op is Ops.REDUCE_OUTPUT and declined.arg.input_identity_at_marker is False

def test_fp16_consumer_marker_owns_existing_cast_and_is_closed_default():
  from tinygrad.llm.model import _decode_reduce_output_rmsnorm_fp16_consumer
  x = Tensor.randn(1,4096,dtype=dtypes.float).realize()
  n = _norm(Tensor.ones(4096,dtype=dtypes.float).realize())
  ordinary = _decode_reduce_output_rmsnorm_fp16_consumer(n,x,False)
  typed = _decode_reduce_output_rmsnorm_fp16_consumer(n,x,True)
  assert ordinary.dtype is dtypes.float
  assert typed.uop.op is Ops.REDUCE_OUTPUT and typed.dtype is dtypes.float16
  assert typed.uop.src[0].op is Ops.CAST and typed.uop.arg.out_dtype is dtypes.float16
  # The typed marker owns the established cast; it does not need a selector
  # to pierce CAST in the later scheduler graph.
  assert _names(typed) == ["reduce_output_rmsnorm_1_4096"]

def test_fp16_cooperative_body_stays_within_model_numeric_envelope():
  from tinygrad.llm.model import _decode_reduce_output_rmsnorm_fp16_consumer
  x = Tensor(np.random.default_rng(20260805).normal(0,.2,(1,4096)).astype(np.float32),dtype=dtypes.float).realize()
  n = _norm(Tensor.ones(4096,dtype=dtypes.float).realize())
  got = _decode_reduce_output_rmsnorm_fp16_consumer(n,x,True).numpy().astype(np.float32)
  ref = _decode_reduce_output_rmsnorm_fp16_consumer(n,x,False).cast(dtypes.float16).numpy().astype(np.float32)
  assert np.isfinite(got).all()
  assert np.max(np.abs(got-ref)) <= .01
  assert np.linalg.norm(got-ref) / np.linalg.norm(ref) <= 1e-5

def test_shared_q8_fused_lease_marks_attention_only_and_never_prefill():
  from types import SimpleNamespace
  from tinygrad.llm.model import _decode_reduce_output_norm_flags
  from tinygrad.llm.shared_q8_attention import SharedQ8AttentionAdmission
  block=SimpleNamespace(_shared_q8_attention_admission=SharedQ8AttentionAdmission(1),
                        _decode_reduce_output_attn_rmsnorm_promoted=True)
  assert _decode_reduce_output_norm_flags(block,False) == (True,False)
  assert _decode_reduce_output_norm_flags(block,True) == (False,False)
  del block._decode_reduce_output_attn_rmsnorm_promoted
  assert _decode_reduce_output_norm_flags(block,False) == (False,False)
  block._decode_reduce_output_rmsnorm_promoted=True
  assert _decode_reduce_output_norm_flags(block,False) == (True,True)

def test_native_value_matches_ordinary():
  rng = np.random.default_rng(20260805)
  # The model's real dtype mix is fp32 x + fp16 w; the fused body must be
  # bitwise equal to the ordinary graph for every mix, not just all-fp16
  # (the 08-05 ladder association differed in the fp32 last ulp for fp32 x).
  for xdt, wdt in [(dtypes.float, dtypes.float16), (dtypes.float16, dtypes.float16), (dtypes.float, dtypes.float)]:
    x = Tensor(rng.normal(0, .2, (1,4096)).astype(np.float32), dtype=xdt).realize()
    w = Tensor(rng.normal(1, .05, (4096,)).astype(np.float32), dtype=wdt).realize()
    got, ref = _apply(_norm(w), x), _apply(_norm(w), x, False)
    np.testing.assert_allclose(got.numpy(), ref.numpy(), rtol=0, atol=0)

def test_body_has_restored_lane_phase_and_no_custom_op():
  from tinygrad.codegen.late.reduce_output import emit_reduce_output_rmsnorm
  from tinygrad.uop.ops import ReduceOutputSpec, UOp, AxisType
  spec = ReduceOutputSpec(1, 4096, 1e-6, dtypes.float16)
  out, x, w = (UOp.placeholder((4096,), dtypes.float16, i) for i in range(3))
  body = emit_reduce_output_rmsnorm(spec, dtypes.float16, dtypes.float16)(out, x, w)
  topo = body.toposort()
  assert sum(u.op is Ops.BARRIER for u in topo) == 1
  assert any(u.op is Ops.RANGE and u.arg == (2, AxisType.REDUCE) for u in topo)
  assert any(u.op is Ops.RANGE and u.arg == (2, AxisType.LOOP) for u in topo)
  assert not any(u.op is Ops.CUSTOM for u in topo)
