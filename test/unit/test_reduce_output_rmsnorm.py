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


def test_non_terminal_marker_producer_keeps_closed_graph_spelling():
  """The production per-block ``_run`` chain feeds one block's output into the
  next block's marker INPUT, but the marker is consumed INSIDE that body (the
  result is the residual stream).  Such a producer is on the route (ro_route)
  but NOT on the marker's output boundary, so its output redirect and direct
  input view must not fire: under the candidate flags it transforms
  byte-identically to the closed control graph, or the census's residual
  E_32_32_4 kernel identities shift.  The route sets are computed exactly as
  transform_to_call does; only precompiled_output_slots (route metadata, not a
  graph spelling change) is normalized between the arms."""
  from dataclasses import replace
  from tinygrad.callify import (_ACTIVE_REDUCE_OUTPUT_ROUTE_FUNCTIONS, _ACTIVE_REDUCE_OUTPUT_OUT_ROUTE_FUNCTIONS,
                                CallInfo, _reduce_output_route_function_ids, pm_early_transform_tensor_graph)
  from tinygrad.function import function
  from tinygrad.helpers import Context
  from tinygrad.llm.memory_semantics import runtime_activation
  from tinygrad.uop.ops import Ops, UOp, graph_rewrite
  x = Tensor.empty(1,1,4096,dtype=dtypes.float32,device="NV")
  w = Tensor.empty(4096,dtype=dtypes.float16,device="NV")
  norm = _norm(w)
  @function(precompile=True,allow_implicit=True)
  def prev_block(v): return runtime_activation((v + 1).contiguous())
  @function(precompile=True,allow_implicit=True)
  def block_run(v):
    marked = _apply(norm, v)
    return runtime_activation((marked + v).contiguous())
  value = block_run(runtime_activation(prev_block(x).contiguous()))
  route_ids, out_route_ids = _reduce_output_route_function_ids(value.uop)
  assert route_ids and not out_route_ids, "route scan must see the producer without a terminal marker"
  control = graph_rewrite(value.uop, pm_early_transform_tensor_graph, name="control nonterminal callify")
  with Context(_ACTIVE_REDUCE_OUTPUT_ROUTE_FUNCTIONS=route_ids,
               _ACTIVE_REDUCE_OUTPUT_OUT_ROUTE_FUNCTIONS=out_route_ids,
               CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
    candidate = graph_rewrite(value.uop, pm_early_transform_tensor_graph, name="candidate nonterminal callify")
  assert control is not None and candidate is not None
  def structural(u:UOp):
    if u.op is Ops.UNIQUE: return ("UNIQUE",)
    arg = u.arg if not isinstance(u.arg, CallInfo) else replace(u.arg, precompiled_output_slots=())
    return (u.op, tuple(structural(s) for s in u.src), arg)
  assert structural(control) == structural(candidate)


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

def test_fp32_end_to_end_marker_owns_no_cast_and_body_has_no_fp16_round_points():
  """llama-parity fp32 norm contract (nv-reduce-output-phase6 analysis,
  2026-08-10): llama's rms_norm_f32 reads the fp32 residual, computes sumsq
  and the affine epilogue in fp32, and only the q8_1 quantize rounds to half.
  The typed fp16 consumer route owns its cast by design (the test above); the
  fp32 route must own NONE, so the consumer can round in-kernel and the graph
  carries no cast or materialization between residual and consumer."""
  rng = np.random.default_rng(20260810)
  x = Tensor(rng.normal(0, .2, (1, 4096)).astype(np.float32)).realize()
  w = Tensor(rng.normal(1, .05, (4096,)).astype(np.float32)).realize()
  marked = _apply(_norm(w), x)
  assert marked.dtype is dtypes.float32
  assert marked.uop.op is Ops.REDUCE_OUTPUT
  assert marked.uop.src[0].op is not Ops.CAST
  assert _names(marked) == ["reduce_output_rmsnorm_1_4096"]
  # The bitwise NV tripwire for the fp32 x / fp16 w and fp32 x / fp32 w mixes
  # is test_native_value_matches_ordinary; this CPU test is structural only.
  from tinygrad.codegen.late.reduce_output import emit_reduce_output
  from tinygrad.uop.ops import ReduceOutputSpec, UOp
  spec = ReduceOutputSpec(1, 4096, 1e-6, dtypes.float32)
  out, xx, ww = (UOp.placeholder((4096,), dtypes.float32, i) for i in range(3))
  body = emit_reduce_output(spec, dtypes.float32, dtypes.float32)(out, xx, ww)
  topo = body.toposort()
  assert not any(u.op is Ops.CAST and u.dtype is dtypes.float16 for u in topo)
  assert not any(u.dtype is dtypes.float16 for u in topo)

def test_fp16_cooperative_body_stays_within_model_numeric_envelope():
  from tinygrad.llm.model import _decode_reduce_output_rmsnorm_fp16_consumer
  x = Tensor(np.random.default_rng(20260805).normal(0,.2,(1,4096)).astype(np.float32),dtype=dtypes.float).realize()
  n = _norm(Tensor.ones(4096,dtype=dtypes.float).realize())
  got = _decode_reduce_output_rmsnorm_fp16_consumer(n,x,True).numpy().astype(np.float32)
  ref = _decode_reduce_output_rmsnorm_fp16_consumer(n,x,False).cast(dtypes.float16).numpy().astype(np.float32)
  assert np.isfinite(got).all()
  assert np.max(np.abs(got-ref)) <= .01
  assert np.linalg.norm(got-ref) / np.linalg.norm(ref) <= 1e-5

def test_reduce_output_helpers_bind_materialized_identity_weight():
  """Production norm weights are lazy fp16 casts over quantized storage; a
  marker on a lazy weight makes the selector emit one fresh weight
  materialization per fused body (the phase-6 8eeb0be1 overhead).  The
  load-time ``_decode_reduce_output_weight`` gives the marker an identity
  buffer, so the selector binds it directly and no fresh store is scheduled."""
  from tinygrad.llm.model import _decode_reduce_output_rmsnorm, _decode_reduce_output_rmsnorm_fp16_consumer
  from tinygrad.schedule.rangeify import _identity_buffer_view
  x = Tensor.randn(1, 4096, dtype=dtypes.float32).realize()
  lazy = Tensor.randn(4096, dtype=dtypes.float32).realize().cast(dtypes.float16)  # lazy cast, no identity
  assert not lazy.uop.has_buffer_identity()
  n = _norm(lazy)
  n._decode_reduce_output_weight = lazy.contiguous().realize()
  plain = _decode_reduce_output_rmsnorm(n, x, True)
  typed = _decode_reduce_output_rmsnorm_fp16_consumer(n, x, True)
  assert plain.uop.op is Ops.REDUCE_OUTPUT and plain.uop.src[2].has_buffer_identity()
  assert typed.uop.op is Ops.REDUCE_OUTPUT and typed.uop.src[2].has_buffer_identity()
  assert plain.uop.src[2] is n._decode_reduce_output_weight.uop
  assert typed.uop.src[2] is n._decode_reduce_output_weight.uop
  for marked in (plain, typed):
    linear, _ = marked.linear_with_vars()
    fused = [call for call in linear.src if call.src[0].arg.name == "reduce_output_rmsnorm_1_4096"]
    assert len(fused) == 1
    assert _identity_buffer_view(fused[0].src[3]) is n._decode_reduce_output_weight.uop
  # Without the load-time materialization the marker keeps the lazy weight
  # (the pre-fix spelling: the selector then emits the fresh-store weight
  # materialization, which is the phase-6 8eeb0be1 overhead).
  n._decode_reduce_output_weight = None
  declined = _decode_reduce_output_rmsnorm(n, x, True)
  assert declined.uop.op is Ops.REDUCE_OUTPUT
  assert declined.uop.src[2] is lazy.uop

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


def _fma(a: float, b: float, c: float) -> np.float32:
  """Correctly-rounded fp32 a*b+c (clang contracts ``acc + v*v`` into fma on
  both the ordinary CPU kernel and the fused body; numpy has no fma)."""
  return np.float32(np.longdouble(a) * np.longdouble(b) + np.longdouble(c))


def _multi_row_body_association(rows, dim, eps, x_dtype, w_dtype):
  """Emit the cooperative body and extract the association it actually encodes:
  the per-warp reduce chain extent and base, the epilogue weight index, and the
  cross-warp combine shape.  The reconstruction below is driven by these, so a
  drift in the emitter changes the extracted chain and breaks the bitwise gate.
  """
  from tinygrad.codegen.late.reduce_output import emit_reduce_output
  from tinygrad.uop.ops import ReduceOutputSpec, UOp, AxisType, Ops
  spec = ReduceOutputSpec(rows, dim, eps, x_dtype, warps=rows, lanes=32, per_lane=dim // 32)
  out = UOp.placeholder((rows * dim,), x_dtype, 0)
  x = UOp.placeholder((rows * dim,), x_dtype, 1)
  w = UOp.placeholder((dim,), w_dtype, 2)
  body = emit_reduce_output(spec, x_dtype, w_dtype)(out, x, w)
  topo = body.toposort()
  # Per-warp serial reduce chain: the x read whose index is warp*red_base + red.
  x_idx = [u for u in topo if u.op is Ops.INDEX and u.src[0] is x]
  red_chain = [u for u in x_idx if len(u.src[1].src) == 2 and u.src[1].src[1].op is Ops.RANGE and
               u.src[1].src[1].arg == (2, AxisType.REDUCE)]
  assert len(red_chain) == 1, "reduce body must contain exactly one per-warp serial x chain"
  idx = red_chain[0].src[1]
  assert idx.op is Ops.ADD and idx.src[0].op is Ops.MUL and idx.src[1].op is Ops.RANGE, f"unexpected chain index {idx.op}"
  red = idx.src[1]
  red_base = red.src[0].arg
  red_extent = idx.src[0].src[1].arg  # warp * CONST(red_base)
  # Epilogue weight index must be the row-local element (multi-row shares a (dim,) weight).
  w_idx = [u for u in topo if u.op is Ops.INDEX and u.src[0] is w]
  assert len(w_idx) == 1
  widx = w_idx[0].src[1]
  assert (widx.op is Ops.ADD and widx.src[0].op is Ops.RANGE and widx.src[1].op is Ops.MUL and
          widx.src[1].src[0].op is Ops.RANGE), "multi-row weight index must be row-local laneid + epi*lane"
  # No cross-warp serial combine: exactly one smem-after read, no ADD chain over it.
  smem_after = [u for u in topo if u.op is Ops.AFTER and len(u.src) == 2 and u.src[0].op is Ops.DEFINE_LOCAL]
  assert len(smem_after) == 1
  reads = [u for u in topo if u.op is Ops.INDEX and len(u.src) == 2 and u.src[0] is smem_after[0]]
  assert len(reads) == 1 and not any(u.op is Ops.ADD and reads[0] in u.src for u in topo)
  assert body.arg.name == f"reduce_output_rmsnorm_{rows}_{dim}"
  return dict(red_extent=red_extent, red_base=red_base, dim=dim, eps=eps, x_dtype=x_dtype, w_dtype=w_dtype)


def _multi_row_fused_output(x_np, w_np, assoc):
  """Reconstruct the fused body output bitwise: per-row plain ``dim``-contiguous
  serial FMA sumsq chain (the row total), per-row scale, ``(x*scale)*w``
  epilogue.  Rows are independent in row-mode (no cross-row combine)."""
  rows, dim, eps = x_np.shape[0], assoc["dim"], assoc["eps"]
  assert assoc["red_extent"] == assoc["red_base"] == dim
  scale = np.zeros((rows,), dtype=np.float32)
  for r in range(rows):
    acc = np.float32(0.0)
    for v in x_np[r]: acc = _fma(np.float32(v), np.float32(v), acc)
    scale[r] = np.float32(1.0 / np.sqrt(np.float32(np.float32(acc / dim) + eps)))
  out = np.empty_like(x_np, dtype=np.float32)
  for r in range(rows):
    out[r] = (x_np[r] * scale[r]).astype(np.float32) * w_np
  return out


def test_multi_row_fused_body_matches_ordinary_bitwise():
  """The row-mode q/k gate: for (rows, dim) in {(32,128),(8,128)} the emitted
  cooperative body is bitwise-equal to the ordinary CPU RMSNorm for the fp32 x /
  fp32 w and fp32 x / fp16 w mixes.  This is the exact-logits tripwire: a 1-ulp
  drift in the reduce association (or a cross-row combine) flips it red."""
  rng = np.random.default_rng(20260810)
  for rows in (32, 8):
    dim = 128
    x_np = rng.normal(0, 0.2, (rows, dim)).astype(np.float32)
    w_np = rng.normal(1, 0.05, (dim,)).astype(np.float32)
    x = Tensor(x_np).realize()
    for wdt in (dtypes.float, dtypes.float16):
      w_eff = w_np.astype(np.float16).astype(np.float32) if wdt is dtypes.float16 else w_np
      w = Tensor(w_eff, dtype=wdt).realize()
      norm = nn.RMSNorm(dim, eps=1e-6)
      norm.weight = w
      ordinary = norm(x).numpy()
      assoc = _multi_row_body_association(rows, dim, 1e-6, dtypes.float32, wdt)
      fused = _multi_row_fused_output(x_np, w_eff, assoc)
      np.testing.assert_array_equal(fused, ordinary)


def test_multi_row_scale_isolation_no_cross_row_combine():
  """Row-mode must never combine rows: each warp's scale depends only on its own
  row.  Rows span wildly different magnitudes (10**row), so a serial cross-row
  partial chain (the single-row combine pattern) cannot reproduce the ordinary
  per-row result while the isolated per-row body must, bitwise."""
  rows, dim = 32, 128
  rng = np.random.default_rng(20260811)
  x_np = (rng.normal(0, 0.2, (rows, dim)) * (10.0 ** np.arange(rows)[:, None])).astype(np.float32)
  w_np = rng.normal(1, 0.05, (dim,)).astype(np.float32)
  x = Tensor(x_np).realize()
  w = Tensor(w_np).realize()
  norm = nn.RMSNorm(dim, eps=1e-6)
  norm.weight = w
  ordinary = norm(x).numpy()
  assoc = _multi_row_body_association(rows, dim, 1e-6, dtypes.float32, dtypes.float32)
  fused = _multi_row_fused_output(x_np, w_np, assoc)
  np.testing.assert_array_equal(fused, ordinary)
  # Countercheck: a cross-row serial combine must NOT match ordinary.  Chain the
  # row sumsq partials across rows (the single-row combine pattern applied to
  # rows), which corrupts every row after the first by the row-magnitude spread.
  corrupted = np.empty_like(x_np, dtype=np.float32)
  running = np.float32(0.0)
  for r in range(rows):
    acc = np.float32(0.0)
    for v in x_np[r]: acc = _fma(np.float32(v), np.float32(v), acc)
    running = np.float32(running + acc)
    scale = np.float32(1.0 / np.sqrt(np.float32(np.float32(running / dim) + 1e-6)))
    corrupted[r] = (x_np[r] * scale).astype(np.float32) * w_np
  assert not np.array_equal(corrupted, ordinary), "cross-row combine must not match ordinary"
  # The structural isolation contract the body must keep: one smem-after read,
  # no serial ADD chain over the published partials.
  from tinygrad.codegen.late.reduce_output import emit_reduce_output
  from tinygrad.uop.ops import ReduceOutputSpec, UOp, AxisType
  spec = ReduceOutputSpec(rows, dim, 1e-6, dtypes.float32, warps=rows, lanes=32, per_lane=4)
  out, xp, wp = (UOp.placeholder((rows * dim,), dtypes.float32, i) for i in range(3))
  body = emit_reduce_output(spec, dtypes.float32, dtypes.float32)(out, xp, wp)
  topo = body.toposort()
  smem_after = [u for u in topo if u.op is Ops.AFTER and len(u.src) == 2 and u.src[0].op is Ops.DEFINE_LOCAL]
  assert len(smem_after) == 1
  reads = [u for u in topo if u.op is Ops.INDEX and len(u.src) == 2 and u.src[0] is smem_after[0]]
  assert len(reads) == 1 and not any(u.op is Ops.ADD and reads[0] in u.src for u in topo)


def test_multi_row_marker_lowers_to_fused_call_name():
  """The full CPU schedule must lower the fp32 q/k markers to the row-aware fused
  body names (the emitter admits them through the rangeify selector)."""
  rng = np.random.default_rng(20260812)
  for rows in (32, 8):
    x = Tensor(rng.normal(0, 0.2, (1, rows, 1, 128)).astype(np.float32)).realize()
    w = Tensor(rng.normal(1, 0.05, (128,)).astype(np.float32)).realize()
    n = nn.RMSNorm(128, eps=1e-6)
    n.weight = w
    marked = _apply(n, x)
    assert marked.uop.op is Ops.REDUCE_OUTPUT
    assert _names(marked) == [f"reduce_output_rmsnorm_{rows}_128"]


def test_multi_row_fail_closed_combinations():
  from tinygrad.codegen.late.reduce_output import emit_reduce_output
  from tinygrad.uop.ops import ReduceOutputSpec
  for spec in (
    ReduceOutputSpec(4, 128, 1e-6, dtypes.float32, warps=4, lanes=32, per_lane=4),
    ReduceOutputSpec(32, 64, 1e-6, dtypes.float32, warps=32, lanes=32, per_lane=2),
    ReduceOutputSpec(8, 128, 1e-6, dtypes.float32, warps=4, lanes=32, per_lane=8),
    ReduceOutputSpec(16, 128, 1e-6, dtypes.float32, warps=16, lanes=32, per_lane=4),
    ReduceOutputSpec(32, 256, 1e-6, dtypes.float32, warps=32, lanes=32, per_lane=8),
    ReduceOutputSpec(1, 128, 1e-6, dtypes.float32, warps=1, lanes=32, per_lane=4),
  ):
    try:
      emit_reduce_output(spec, dtypes.float32, dtypes.float32)
      raise AssertionError(f"emit_reduce_output accepted {spec}")
    except ValueError:
      pass


def test_multi_row_4096_dim_body_structure():
  """The marker admits multi-row dim 4096 (one warp per row, per_lane=128); the
  emitter must cover it with the same row-mode body (full-row chain, one smem
  readback, no cross-warp combine)."""
  from tinygrad.codegen.late.reduce_output import emit_reduce_output
  from tinygrad.uop.ops import ReduceOutputSpec, UOp, AxisType
  rows, dim = 32, 4096
  spec = ReduceOutputSpec(rows, dim, 1e-6, dtypes.float32, warps=rows, lanes=32, per_lane=128)
  out, x, w = (UOp.placeholder((rows * dim,), dtypes.float32, i) for i in range(3))
  body = emit_reduce_output(spec, dtypes.float32, dtypes.float32)(out, x, w)
  topo = body.toposort()
  assert body.arg.name == "reduce_output_rmsnorm_32_4096"
  assert sum(u.op is Ops.BARRIER for u in topo) == 1
  smem_after = [u for u in topo if u.op is Ops.AFTER and len(u.src) == 2 and u.src[0].op is Ops.DEFINE_LOCAL]
  reads = [u for u in topo if u.op is Ops.INDEX and len(u.src) == 2 and u.src[0] is smem_after[0]]
  assert len(reads) == 1 and not any(u.op is Ops.ADD and reads[0] in u.src for u in topo)
  assert any(u.op is Ops.RANGE and u.src[0].op is Ops.CONST and u.src[0].arg == 4096 and u.arg == (2, AxisType.REDUCE) for u in topo)
