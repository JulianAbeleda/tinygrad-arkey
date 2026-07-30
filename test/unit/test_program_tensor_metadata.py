import gc, weakref
from dataclasses import replace
from typing import NamedTuple

from tinygrad import Tensor, dtypes
from tinygrad.engine import jit
from tinygrad.engine.jit import GraphAdmission, GraphAdmissionDecision, GraphAdmissionReason
from tinygrad.engine.metadata import buffer_metadata
from tinygrad.engine.realize import compile_linear
from tinygrad.helpers import Context, Metadata
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.model_facts import (ProgramTensorFact, TensorFact, bind_program_tensor_fact,
  bind_gguf_program_tensor_facts, program_identities_from_call, program_tensor_facts)
from tinygrad.schedule import schedule_cache
from tinygrad.tensor import role_metadata
from tinygrad.uop.ops import Ops, UOp


class _ProgramInfo(NamedTuple):
  name: str
  outs: tuple[int, ...]
  ins: tuple[int, ...]


def _program_call(*args, outs=(0,), ins=None):
  if ins is None: ins = tuple(range(1, len(args)))
  return UOp(Ops.PROGRAM, arg=_ProgramInfo("opaque", outs, ins)).call(*args)


def _fact(name, *, rows=32, cols=1024, quant="Q4_K", role="ffn_down"):
  return TensorFact(f"blk.0.{name}.weight", f"blk.0.{name}", 12 if quant == "Q4_K" else 14, rows, cols, quant, role)


def test_exact_shared_backing_intervals_do_not_smear_tensor_facts():
  backing = UOp.new_buffer("CPU", 4096, dtypes.uint8)
  left = UOp(Ops.SLICE, dtypes.uint8, (backing, UOp.const(dtypes.weakint, 256)), 512)
  right = UOp(Ops.SLICE, dtypes.uint8, (backing, UOp.const(dtypes.weakint, 1024)), 512)
  left_subview = UOp(Ops.SLICE, dtypes.uint8, (backing, UOp.const(dtypes.weakint, 320)), 64)
  left_fact, right_fact = _fact("ffn_down"), _fact("attn_q", role="attn_qo")
  bind_program_tensor_fact(left, left_fact, alias="weight")
  bind_program_tensor_fact(right, right_fact, alias="weight")
  assert program_tensor_facts(left_subview) == (ProgramTensorFact(left_fact, "weight"),)
  assert program_tensor_facts(right) == (ProgramTensorFact(right_fact, "weight"),)
  assert buffer_metadata(backing) == ()


def test_distinct_model_backings_with_identical_intervals_do_not_cross_contaminate():
  first_backing, second_backing = (UOp.new_buffer("CPU", 4096, dtypes.uint8) for _ in range(2))
  first_view = UOp(Ops.SLICE, dtypes.uint8, (first_backing, UOp.const(dtypes.weakint, 256)), 512)
  second_view = UOp(Ops.SLICE, dtypes.uint8, (second_backing, UOp.const(dtypes.weakint, 256)), 512)
  first_fact, second_fact = _fact("ffn_down"), _fact("attn_q", role="attn_qo")
  bind_program_tensor_fact(first_view, first_fact, alias="weight")
  bind_program_tensor_fact(second_view, second_fact, alias="weight")
  assert program_tensor_facts(first_view) == (ProgramTensorFact(first_fact, "weight"),)
  assert program_tensor_facts(second_view) == (ProgramTensorFact(second_fact, "weight"),)


def test_side_registry_does_not_retain_dead_model_backings():
  backing = UOp.new_buffer("CPU", 4096, dtypes.uint8)
  view = UOp(Ops.SLICE, dtypes.uint8, (backing, UOp.const(dtypes.weakint, 256)), 512)
  backing_ref = weakref.ref(backing)
  bind_program_tensor_fact(view, _fact("ffn_down"), alias="weight")
  del backing, view
  gc.collect()
  assert backing_ref() is None


def test_nested_slice_preserves_outer_length_and_accumulates_offsets():
  backing = UOp.new_buffer("CPU", 4096, dtypes.uint8)
  parent = UOp(Ops.SLICE, dtypes.uint8, (backing, UOp.const(dtypes.weakint, 256)), 512)
  child = UOp(Ops.SLICE, dtypes.uint8, (parent, UOp.const(dtypes.weakint, 64)), 32)
  fact = _fact("ffn_down")
  bind_program_tensor_fact(child, fact, alias="weight")
  exact = UOp(Ops.SLICE, dtypes.uint8, (backing, UOp.const(dtypes.weakint, 320)), 32)
  adjacent = UOp(Ops.SLICE, dtypes.uint8, (backing, UOp.const(dtypes.weakint, 352)), 32)
  assert program_tensor_facts(exact) == (ProgramTensorFact(fact, "weight"),)
  assert program_tensor_facts(parent) == () and program_tensor_facts(adjacent) == ()


def test_gguf_payload_authority_binds_raw_backing_before_dequant_fusion():
  raw = Tensor.empty(1024, dtype=dtypes.uint8, device="CPU")
  fact = TensorFact("blk.0.ffn_down.weight", "blk.0.ffn_down", 12, 2, 256, "Q4_K", "ffn_down")
  meta = {"data_start":128, "tensor_infos":[(fact.name, (256, 2), 12, 0)], "raw_tensor":raw}
  assert bind_gguf_program_tensor_facts(meta, (fact,)) == (fact.name,)
  payload = UOp(Ops.SLICE, dtypes.uint8, (raw.uop.buf_uop, UOp.const(dtypes.weakint, 128)), 288)
  padding = UOp(Ops.SLICE, dtypes.uint8, (raw.uop.buf_uop, UOp.const(dtypes.weakint, 416)), 32)
  assert program_tensor_facts(payload) == (ProgramTensorFact(fact, "weight"),)
  assert program_tensor_facts(padding) == () and program_tensor_facts(raw) == ()


def test_ordinary_quant_dequant_call_retains_exact_raw_fact_without_program_drift():
  raw = Tensor.empty(8192, dtype=dtypes.uint8, device="CPU")
  fact = TensorFact("blk.0.ffn_down.weight", "blk.0.ffn_down", 12, 32, 256, "Q4_K", "ffn_down")
  weight = ggml_data_to_tensor(raw[128:], 8192, 12).reshape(32, 256).cast(dtypes.float16)
  def schedule():
    activation = Tensor.empty(1, 256, dtype=dtypes.float16, device="CPU")
    return compile_linear((activation @ weight.transpose()).schedule_linear())
  control = schedule()
  meta = {"data_start":128, "tensor_infos":[(fact.name, (256, 32), 12, 0)], "raw_tensor":raw}
  bind_gguf_program_tensor_facts(meta, (fact,))
  observed = schedule()
  assert len(observed.src) == len(control.src) == 1
  assert observed.src[0].src[0].key == control.src[0].src[0].key
  assert [item for arg in observed.src[0].src[1:] for item in program_tensor_facts(arg)] == [ProgramTensorFact(fact, "weight")]
  assert observed.src[0].src[0].arg.outs == (0,)
  identity, = program_identities_from_call(observed.src[0])
  assert (identity.tensor_name, identity.logical_m, identity.output_dtype) == (fact.name, 1, str(dtypes.float16))


def test_concrete_program_uses_declared_output_slot_and_truthful_fp32_accumulator():
  fact = _fact("ffn_down", quant="Q6_K")
  packed = UOp.new_buffer("CPU", 840, dtypes.uint32)
  activation = UOp.new_buffer("CPU", fact.cols, dtypes.float16)
  output = UOp.new_buffer("CPU", fact.rows, dtypes.float16)
  bind_program_tensor_fact(packed, fact, alias="packed")
  # Output is deliberately slot 2, proving the resolver follows ProgramInfo.outs.
  identity, = program_identities_from_call(_program_call(packed, activation, output, outs=(2,), ins=(0, 1)))
  assert (identity.phase, identity.logical_m, identity.logical_n, identity.logical_k) == ("decode", 1, fact.rows, fact.cols)
  assert identity.module_representation == "qk_primitive_adapter"
  assert identity.input_dtype == str(dtypes.float16) and identity.output_dtype == str(dtypes.float16)
  assert identity.accumulator_dtype == str(dtypes.float32)


def test_activation_arena_slice_uses_its_exact_extent_for_logical_m():
  fact = _fact("ffn_down", rows=32, cols=256)
  packed = UOp.new_buffer("CPU", 288, dtypes.uint8)
  arena = UOp.new_buffer("CPU", 4096, dtypes.float16)
  activation = UOp(Ops.SLICE, dtypes.float16, (arena, UOp.const(dtypes.weakint, 1024)), fact.cols)
  output = UOp.new_buffer("CPU", fact.rows, dtypes.float16)
  bind_program_tensor_fact(packed, fact, alias="packed")
  identity, = program_identities_from_call(_program_call(output, packed, activation, outs=(0,), ins=(1, 2)))
  assert identity.logical_m == 1 and identity.phase == "decode"


def test_admission_observation_merges_generic_and_resolved_side_metadata():
  fact = _fact("ffn_down", quant="Q6_K")
  packed = UOp.new_buffer("CPU", 840, dtypes.uint32)
  activation = UOp.new_buffer("CPU", fact.cols, dtypes.float16)
  output = UOp.new_buffer("CPU", fact.rows, dtypes.float16)
  bind_program_tensor_fact(packed, fact, alias="packed")
  generic = Metadata("generic", "fixture")
  call = _program_call(packed, activation, output, outs=(2,), ins=(0, 1))
  call = call.replace(arg=replace(call.arg, metadata=(generic,)))
  observation = jit._admission_observation(0, call, GraphAdmissionDecision.ADMITTED,
    GraphAdmission(True, GraphAdmissionReason.ADMITTED))
  assert observation.metadata[0] == generic
  identity, = observation.metadata[1:]
  assert identity.tensor_name == fact.name and identity.accumulator_dtype == str(dtypes.float32)


def test_fused_program_reports_each_exact_weight_once_in_argument_order():
  first, second = _fact("ffn_down", cols=8, quant="Q6_K"), _fact("attn_q", rows=16, cols=12, role="attn_qo")
  output = UOp.new_buffer("CPU", 32, dtypes.float32)
  packed_a, activation_a = UOp.new_buffer("CPU", 64, dtypes.uint8), UOp.new_buffer("CPU", 8, dtypes.float16)
  packed_b, activation_b = UOp.new_buffer("CPU", 96, dtypes.uint8), UOp.new_buffer("CPU", 12, dtypes.float16)
  bind_program_tensor_fact(packed_a, first, alias="packed")
  bind_program_tensor_fact(packed_b, second, alias="packed")
  identities = program_identities_from_call(_program_call(output, packed_a, activation_a, packed_b, activation_b))
  assert [identity.tensor_name for identity in identities] == [first.name, second.name]
  assert all(identity.logical_m == 1 and identity.accumulator_dtype == str(dtypes.float32) for identity in identities)


def test_missing_or_nonquant_facts_fail_closed():
  output = UOp.new_buffer("CPU", 4, dtypes.float32)
  activation = UOp.new_buffer("CPU", 8, dtypes.float16)
  ordinary = UOp.new_buffer("CPU", 32, dtypes.float16)
  bind_program_tensor_fact(ordinary, TensorFact("x.weight", "x", 0, 4, 8, "F32", None), alias="weight")
  assert program_identities_from_call(_program_call(output, ordinary, activation)) == ()
  assert program_identities_from_call(_program_call(output, UOp.new_buffer("CPU", 32, dtypes.uint8), activation)) == ()


def test_registered_fact_without_declared_output_fails_closed():
  fact = _fact("ffn_down", rows=32, cols=256)
  packed = UOp.new_buffer("CPU", 288, dtypes.uint8)
  activation = UOp.new_buffer("CPU", fact.cols, dtypes.float16)
  bind_program_tensor_fact(packed, fact, alias="packed")
  assert program_identities_from_call(_program_call(packed, activation, outs=(), ins=(0, 1))) == ()


def _shared_decode_graph(trace:int, bind_facts:bool):
  with Context(TRACEMETA=trace):
    x = Tensor.empty(1, 16, device="CPU")
    for block in range(8):
      left, right = Tensor.empty(16, 16, device="CPU"), Tensor.empty(16, 16, device="CPU")
      if bind_facts:
        bind_program_tensor_fact(left, _fact(f"blk{block}_left", rows=16, cols=16), alias="weight")
        bind_program_tensor_fact(right, _fact(f"blk{block}_right", rows=16, cols=16), alias="weight")
      with role_metadata(Metadata(f"block_{block}", "fixture")):
        x = (x + (x @ left.transpose()).silu() * (x @ right.transpose())).contiguous()
  return x


def test_shared_repeated_graph_has_zero_call_function_or_program_identity_drift():
  schedule_cache.clear()
  control = _shared_decode_graph(0, False).schedule_linear()
  control_functions = [call.src[0].key for call in control.src]
  control_programs = [call.src[0].key for call in compile_linear(control).src]
  schedule_cache.clear()
  observed = _shared_decode_graph(1, True).schedule_linear()
  assert len(observed.src) == len(control.src) == 8
  assert [call.src[0].key for call in observed.src] == control_functions
  assert [call.src[0].key for call in compile_linear(observed).src] == control_programs


def test_side_bindings_do_not_enter_normalized_schedule_cache():
  schedule_cache.clear()
  first = _shared_decode_graph(0, False).schedule_linear()
  cached_keys = set(schedule_cache)
  second = _shared_decode_graph(1, True).schedule_linear()
  assert set(schedule_cache) == cached_keys
  assert [call.src[0].key for call in first.src] == [call.src[0].key for call in second.src]
