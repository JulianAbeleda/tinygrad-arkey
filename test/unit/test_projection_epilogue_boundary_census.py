"""CPU-only P2b boundary census.

These are deliberately structural tests: a projection epilogue may only be
timed if its extra inputs do not introduce an opaque-boundary materialization.
"""
import numpy as np
import pytest

from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT
from tinygrad.device import Device
from tinygrad.function import function
from tinygrad.helpers import Context
from tinygrad.llm.decode_kernels import Q4KGEMVEpilogue, q4k_g3_lanemap_gemv_kernel
from tinygrad.llm.kernel_program import (DeclaredTypedOutput, KernelProgram, KernelProgramProvenance, OutputSpec,
  TypedLayout, TypedViewRequest, execute_promoted_program)
from tinygrad.llm.flash_decode_attention import describe_flash_decode_attention
from tinygrad.llm.memory_semantics import runtime_activation, runtime_scratch
from tinygrad.uop.ops import Ops, UOp
from extra.llm_research.decode.nv_projection_epilogue_qualification import post_callify_copy_trace


def _cuda_available():
  try:
    return str(Device.DEFAULT).startswith(("CUDA", "NV"))
  except Exception:
    return False


@pytest.fixture(autouse=True)
def _candidate_callify_redirect():
  # The implementation is closed by default. This file qualifies its opt-in
  # behavior; the rollback test below temporarily restores the default arm.
  with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1): yield


def _program(n:int, k:int, epilogue:Q4KGEMVEpilogue|None=None, load_style:str="scalar",
             typed_output:DeclaredTypedOutput|None=None) -> KernelProgram:
  return KernelProgram("p2b_boundary_census", "p2b_boundary_census.gemv", KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
    q4k_g3_lanemap_gemv_kernel(n, k, epilogue=epilogue, load_style=load_style),
    output_spec=OutputSpec((n,), dtypes.float32, typed_output=typed_output))


@pytest.mark.parametrize("load_style", ("scalar", "vector"))
def test_declared_output_survives_precompiled_function_substitution(load_style):
  from tinygrad.callify import _declared_epilogue_absorption_after
  n = k = 4096
  declared = DeclaredTypedOutput(TypedLayout(dtypes.float32, (n,), (1, 1, n)), False, True)
  epi = _program(n, k, Q4KGEMVEpilogue("residual_add"), load_style=load_style, typed_output=declared)
  words, activation = _words(n, k), Tensor.empty(k, dtype=dtypes.float16)
  @function(precompile=True, allow_implicit=True)
  def consumer(residual):
    out = execute_promoted_program(None, words, activation, residual.reshape(n), program=epi)
    return runtime_activation(out.reshape(1, 1, n).contiguous())
  result = consumer(Tensor.empty(1, 1, n, dtype=dtypes.float32))
  fn_uop = result.uop.src[0]
  assert fn_uop.op is Ops.FUNCTION and fn_uop.src[0].op is Ops.TUPLE
  assert _declared_epilogue_absorption_after(fn_uop.src[0].src[0]) is not None


def _calls(t:Tensor) -> list[str]:
  linear, _ = t.linear_with_vars()
  return [getattr(getattr(u.src[0], "arg", None), "name", "") for u in linear.toposort() if u.op is Ops.CALL]


def _words(n:int, k:int) -> Tensor: return Tensor.empty(n * (k // 256) * 36, dtype=dtypes.uint32)


def _raw_residual_input(t:Tensor) -> Ops:
  # Before callify, the custom kernel body has the exact arguments selected by
  # UOp.custom_kernel. The last one is the residual input to the real emitter.
  calls = [u for u in t.uop.toposort() if u.op is Ops.CALL and u.src[0].op is Ops.SINK]
  assert len(calls) == 1
  return calls[0].src[-1].op


def test_non_precompiled_function_output_needs_materialization_but_after_does_not():
  # This is the actual distinction behind custom_kernel's AFTER-preservation rule:
  # a block/call result is GETTUPLE and has no physical identity, while an opaque
  # producer result carries the exact buffer identity the epilogue can consume.
  n = k = 4096
  x = Tensor.empty(k, dtype=dtypes.float16)
  @function(precompile=False)
  def block_output(v): return v + 1
  from_call = block_output(Tensor.empty(n, dtype=dtypes.float32))
  assert from_call.uop.op is Ops.GETTUPLE and not from_call.uop.has_buffer_identity()
  epi = _program(n, k, Q4KGEMVEpilogue("residual_add"))
  materialized = execute_promoted_program(None, _words(n, k), x, from_call, program=epi)
  assert _calls(materialized) == ["test", "q4k_g3_lanemap_gemv_epi_resadd_4096_4096"]

  producer = execute_promoted_program(None, _words(n, k), x, program=_program(n, k))
  direct = execute_promoted_program(None, _words(n, k), x, producer, program=epi)
  assert _calls(direct) == ["q4k_g3_lanemap_gemv_4096_4096", "q4k_g3_lanemap_gemv_epi_resadd_4096_4096"]


def test_post_callify_trace_links_a_materialization_writer_to_epilogue_slot():
  # The tracer is intentionally graph-only. This synthetic non-precompiled
  # boundary creates the same writer -> physical-buffer -> opaque-consumer
  # relationship that the real 86a2 census must resolve, without a GPU.
  n = k = 4096
  @function(precompile=False)
  def block(v): return v + 1
  epi = _program(n, k, Q4KGEMVEpilogue("residual_add"))
  out = execute_promoted_program(None, _words(n, k), Tensor.empty(k, dtype=dtypes.float16),
                                 block(Tensor.empty(n, dtype=dtypes.float32)), program=epi)
  trace = post_callify_copy_trace(out.linear_with_vars()[0], "test")
  assert trace
  # At least one test writer reaches the residual-add kernel. We assert the
  # consumer slot and ABI, not a brittle call ordering/object identity.
  endpoints = [edge for record in trace for write in record["edges"] for edge in write["chain"]
               if edge["program"] == "q4k_g3_lanemap_gemv_epi_resadd_4096_4096"]
  assert endpoints and all(edge["shape"] == ["4096"] and edge["dtype"] == "dtypes.float" for edge in endpoints)


@pytest.mark.skipif(not _cuda_available(), reason="requires CUDA/NV grid-parallel kernels")
def test_post_callify_trace_uses_compiled_program_output_slots():
  from tinygrad.engine.realize import compile_linear
  n = k = 4096
  @function(precompile=False)
  def block(v): return v + 1
  epi = _program(n, k, Q4KGEMVEpilogue("residual_add"))
  out = execute_promoted_program(None, _words(n, k), Tensor.empty(k, dtype=dtypes.float16),
                                 block(Tensor.empty(n, dtype=dtypes.float32)), program=epi)
  trace = post_callify_copy_trace(compile_linear(out.linear_with_vars()[0]), "E_")
  endpoints = [edge for record in trace for write in record["edges"] for edge in write["chain"]
               if edge["program"] == "q4k_g3_lanemap_gemv_epi_resadd_4096_4096"]
  assert endpoints and {edge["slot"] for edge in endpoints} == {3}


def test_precompiled_function_result_is_a_promised_direct_epilogue_input():
  # callify already gives each precompiled invocation a fresh output buffer.
  # Preserve that promise through custom_kernel; callify then turns it into the
  # dependency-bearing AFTER rather than making a pre-call contiguous adapter.
  n = k = 4096
  @function(precompile=True)
  def producer(v): return v + 1
  @function(precompile=False)
  def not_a_producer(v): return v + 1
  x, residual = Tensor.empty(k, dtype=dtypes.float16), Tensor.empty(n, dtype=dtypes.float32)
  epi = _program(n, k, Q4KGEMVEpilogue("residual_add"))
  direct = execute_promoted_program(None, _words(n, k), x, producer(residual), program=epi)
  rejected = execute_promoted_program(None, _words(n, k), x, not_a_producer(residual), program=epi)
  assert _raw_residual_input(direct) is Ops.GETTUPLE
  assert _raw_residual_input(rejected) is Ops.CONTIGUOUS
  # The promised producer's concrete output is also the exact residual buffer
  # read by the epilogue after callify (no second materialization call).
  calls = [u for u in direct.linear_with_vars()[0].toposort() if u.op is Ops.CALL]
  assert [getattr(getattr(u.src[0], "arg", None), "name", "") for u in calls] == ["test", "q4k_g3_lanemap_gemv_epi_resadd_4096_4096"]
  assert calls[0].src[1].buf_uop is calls[1].src[-1].buf_uop


def test_precompiled_output_contract_rejects_movement_views():
  n = k = 4096
  @function(precompile=True)
  def producer(v): return v + 1
  x, residual = Tensor.empty(k, dtype=dtypes.float16), Tensor.empty(n, dtype=dtypes.float32)
  epi = _program(n, k, Q4KGEMVEpilogue("residual_add"))
  # A reshape is an exact zero-offset equal-span view and stays eligible; a
  # permute has no such contract and must retain the legacy contiguous path.
  reshaped = producer(residual).reshape(1, n).reshape(n)
  moved = producer(residual).reshape(64, 64).permute(1, 0).reshape(n)
  assert _raw_residual_input(execute_promoted_program(None, _words(n, k), x, reshaped, program=epi)) is Ops.RESHAPE
  assert _raw_residual_input(execute_promoted_program(None, _words(n, k), x, moved, program=epi)) is Ops.CONTIGUOUS


def test_precompiled_output_predicate_defers_explicit_contiguous_to_callify():
  n = k = 4096
  @function(precompile=True)
  def producer(v): return v + 1
  x, residual = Tensor.empty(k, dtype=dtypes.float16), Tensor.empty(n, dtype=dtypes.float32)
  epi = _program(n, k, Q4KGEMVEpilogue("residual_add"))
  wrapped = producer(residual).contiguous()
  # CONTIGUOUS over the exact precompiled output is the promised materialization:
  # the output-buffer contract survives the explicit materialization request.
  assert wrapped.uop.has_precompiled_output_identity()
  out = execute_promoted_program(None, _words(n, k), x, wrapped, program=epi)
  assert _raw_residual_input(out) is Ops.CONTIGUOUS


def test_contiguous_contract_does_not_cross_movement_or_offset_views():
  n = k = 4096
  @function(precompile=True)
  def producer(v): return v + 1
  x, residual = Tensor.empty(k, dtype=dtypes.float16), Tensor.empty(n, dtype=dtypes.float32)
  epi = _program(n, k, Q4KGEMVEpilogue("residual_add"))
  moved = producer(residual).reshape(64, 64).permute(1, 0).contiguous().reshape(n)
  assert not moved.uop.has_precompiled_output_identity()
  out = execute_promoted_program(None, _words(n, k), x, moved, program=epi)
  assert _raw_residual_input(out) is Ops.CONTIGUOUS
  # The producer and its movement materialization remain two distinct calls;
  # the owned callify redirect must not erase the latter.
  assert _calls(out) == ["test", "test", "q4k_g3_lanemap_gemv_epi_resadd_4096_4096"]


def test_synthetic_composed_fp16_combine_and_plain_call_output_has_no_adapter():
  hq, hd, split, max_context = 32, 128, 48, 4608
  combine_spec = describe_flash_decode_attention(hq, hd, 8, max_context, split, fused_combine=True, combine_fp16=True)
  combine = KernelProgram("decode_flash_live_split_g4_kvboth", "attention_decode.flash_live_split.combine",
    KernelProgramProvenance.MACHINE_SEARCH_GENERATED, combine_spec.emit_combine(),
    output_spec=OutputSpec((hq*hd,), dtypes.float16,
      typed_output=DeclaredTypedOutput(TypedLayout(dtypes.float16, (hq*hd,), (hq, hd)), True)))
  attn = execute_promoted_program(None, Tensor.empty(hq*split*(hd+2), dtype=dtypes.float32), program=combine)
  # Exact real activation chain: lossless fp16->fp32->fp16 around singleton-axis
  # movement; M5 proves it is a view of the combine AFTER.
  activation = attn.reshape(1, hq, 1, hd).cast(dtypes.float32).transpose(1, 2).reshape(1, 1, -1)
  activation = activation[:, 0, :].reshape(hq*hd).cast(dtypes.float16).contiguous()
  @function(precompile=True)
  def block(v): return (v + 1).contiguous()
  residual = block(Tensor.empty(hq*hd, dtype=dtypes.float32)).contiguous()
  epi = KernelProgram("decode_q4k_g3_generated", "quant_linear_decode.q4k_generated_g3.gemv",
    KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
    q4k_g3_lanemap_gemv_kernel(hq*hd, hq*hd, epilogue=Q4KGEMVEpilogue("residual_add")),
    output_spec=OutputSpec((hq*hd,), dtypes.float32),
    typed_input_views=(TypedViewRequest(1, dtypes.float16, (hq*hd,), "attn_qo"),))
  out = execute_promoted_program(None, _words(hq*hd, hq*hd), activation, residual, program=epi)
  names = _calls(out)
  assert names == ["flash_fused_gmax_combine_f16_32_128", "test", "q4k_g3_lanemap_gemv_epi_resadd_4096_4096"]


@pytest.mark.parametrize("load_style", ("scalar", "vector"))
@pytest.mark.skipif(not _cuda_available(), reason="requires CUDA/NV grid-parallel kernels")
def test_owned_output_stays_direct_through_precompiled_consumer_opaque_call(load_style):
  """Reproduce the real block-output -> next block -> attention-O topology.

  The consumer FUNCTION's invocation normalization and its nested opaque CALL
  used to emit two identical fp32 copies per block.  The prior synthetic case
  connected the producer directly to the epilogue and therefore missed this
  cross-FUNCTION input boundary.
  """
  from tinygrad.engine.realize import compile_linear
  n = k = 4096
  words, activation = _words(n, k), Tensor.empty(k, dtype=dtypes.float16)
  epi = _program(n, k, Q4KGEMVEpilogue("residual_add"), load_style=load_style)
  @function(precompile=True, allow_implicit=True)
  def producer(v): return runtime_activation((runtime_activation(v) + 1).contiguous())
  @function(precompile=True, allow_implicit=True)
  def consumer(residual):
    out = execute_promoted_program(None, words, activation, residual.reshape(n), program=epi)
    return runtime_activation(out.reshape(1, 1, n).contiguous())
  residual = runtime_activation(producer(Tensor.empty(1, 1, n)).contiguous()).reshape(1, 1, n)
  out = consumer(residual)
  linear = compile_linear(out.linear_with_vars()[0])
  names = [getattr(u.src[0].arg, "name", "") for u in linear.toposort() if u.op is Ops.CALL]
  trace = post_callify_copy_trace(linear)
  assert "E_32_32_4_86a23e1a5cd1cbd6101066fd85449138b653e9ecbb53d1d704f32aa470cd6f2b" not in names, (names, trace)
  expected = ("q4k_g3_lanemap_gemv_epi_resadd_4096_4096" if load_style == "scalar" else
              "q4k_g3_lanemap_gemv_vec_epi_resadd_4096_4096")
  assert names[2] == expected


@pytest.mark.skipif(not _cuda_available(), reason="requires CUDA/NV grid-parallel kernels")
def test_owned_output_consumer_input_contract_has_exact_default_rollback():
  from tinygrad.engine.realize import compile_linear
  n = k = 4096
  with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=0):
    words, activation = _words(n, k), Tensor.empty(k, dtype=dtypes.float16)
    epi = _program(n, k, Q4KGEMVEpilogue("residual_add"))
    @function(precompile=True, allow_implicit=True)
    def producer(v): return runtime_activation((runtime_activation(v) + 1).contiguous())
    @function(precompile=True, allow_implicit=True)
    def consumer(residual):
      out = execute_promoted_program(None, words, activation, residual.reshape(n), program=epi)
      return runtime_activation(out.reshape(1, 1, n).contiguous())
    residual = runtime_activation(producer(Tensor.empty(1, 1, n)).contiguous()).reshape(1, 1, n)
    names = [getattr(u.src[0].arg, "name", "") for u in compile_linear(consumer(residual).linear_with_vars()[0]).toposort()
             if u.op is Ops.CALL]
  legacy = "E_32_32_4_86a23e1a5cd1cbd6101066fd85449138b653e9ecbb53d1d704f32aa470cd6f2b"
  assert legacy not in names
  assert names == ["E_32_32_4_c5fc6b8256e282fa6093fda0f2ce652461e310197b08976bd8149373392b366c",
                   "E_32_32_4_fab82d40f922cf5f6decf5a3a82d6b4c2c4b20acb8161ea4de44b3da581fa65b",
                   "q4k_g3_lanemap_gemv_epi_resadd_4096_4096",
                   "E_32_32_4_fab82d40f922cf5f6decf5a3a82d6b4c2c4b20acb8161ea4de44b3da581fa65b"]


def test_owned_invocation_input_matcher_rejects_movement_and_offset():
  from tinygrad.callify import _exact_invocation_param_contiguous
  param = UOp.param(3, dtypes.float32, (4096,), "NV")
  def requested(x): return UOp(Ops.CONTIGUOUS, x.dtype, (x,))
  assert _exact_invocation_param_contiguous(requested(param.reshape(64, 64).reshape(4096))) == 3
  assert _exact_invocation_param_contiguous(requested(param.reshape(64, 64).permute((1, 0)))) is None
  assert _exact_invocation_param_contiguous(requested(param.shrink(((1, 4096),)))) is None


def test_real_nested_memory_semantic_call_output_is_direct_and_shape_preserved():
  n = k = 4096
  @function(precompile=True, allow_implicit=True)
  def block(v):
    return runtime_activation((runtime_activation(v) + 1).contiguous())
  residual = runtime_activation(block(Tensor.empty(1, 1, n, dtype=dtypes.float32)).contiguous()).reshape(1, 1, n).reshape(n)
  assert residual.uop.shape == (n,) and residual.uop.has_precompiled_output_identity()
  epi = _program(n, k, Q4KGEMVEpilogue("residual_add"))
  out = execute_promoted_program(None, _words(n, k), Tensor.empty(k, dtype=dtypes.float16), residual, program=epi)
  # This exercises callify/rangeify, not just the pre-boundary graph: exactly
  # one producer and one epilogue, with no materialization between them.
  assert _calls(out) == ["test", "q4k_g3_lanemap_gemv_epi_resadd_4096_4096"]


def test_real_nested_redirect_has_exact_closed_default_rollback():
  n = k = 4096
  with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=0):
    @function(precompile=True, allow_implicit=True)
    def block(v): return runtime_activation((runtime_activation(v) + 1).contiguous())
    residual = runtime_activation(block(Tensor.empty(1, 1, n, dtype=dtypes.float32)).contiguous()).reshape(1, 1, n).reshape(n)
    epi = _program(n, k, Q4KGEMVEpilogue("residual_add"))
    out = execute_promoted_program(None, _words(n, k), Tensor.empty(k, dtype=dtypes.float16), residual, program=epi)
    # Producer + epilogue. The nested precompiled output keeps its direct slot, so
    # the closed-default switch leaves the same direct ABI as the opt-in path.
    assert _calls(out) == ["test", "q4k_g3_lanemap_gemv_epi_resadd_4096_4096"]


def test_owned_precompiled_redirect_preserves_multioutput_slots_and_owners():
  from tinygrad.callify import transform_precompiled_call
  from tinygrad.uop.ops import memory_semantic_owner
  x = Tensor.empty(8, dtype=dtypes.float32)
  @function(precompile=True)
  def pair(v): return runtime_activation((v+1).contiguous()), runtime_scratch((v+2).contiguous())
  first, second = pair(x)
  call = first.uop.src[0]
  assert call is second.uop.src[0] and call.op is Ops.FUNCTION
  transformed = transform_precompiled_call(call)
  assert transformed is not None and transformed.op is Ops.TUPLE and len(transformed.src) == 2
  a, b = transformed.src
  assert a.op is b.op is Ops.AFTER and a.src[1] is b.src[1]
  assert a.src[0].buf_uop is not b.src[0].buf_uop
  assert sum(x.buf_uop is a.src[0].buf_uop for x in a.src[1].src[1:]) == 1
  assert sum(x.buf_uop is b.src[0].buf_uop for x in b.src[1].src[1:]) == 1
  assert memory_semantic_owner(a) != memory_semantic_owner(b)


def test_owned_precompiled_redirect_rejects_wrong_span_and_intervening_movement():
  from tinygrad.callify import _precompiled_output_redirect
  from tinygrad.uop import RUNTIME_ACTIVATION
  src = UOp.new_buffer("NV", 8, dtypes.float32)
  owned = UOp(Ops.MEMORY_SEMANTIC, dtypes.float32, (src.contiguous(),), RUNTIME_ACTIVATION)
  assert _precompiled_output_redirect(owned, UOp.param(0, dtypes.float32, (4,), "NV"), True) is None
  moved = UOp(Ops.MEMORY_SEMANTIC, dtypes.float32, (src.contiguous().reshape(2,4).permute((1,0)),), RUNTIME_ACTIVATION)
  assert _precompiled_output_redirect(moved, UOp.param(0, dtypes.float32, (4,2), "NV"), True) is None


def test_nested_owned_precompiled_output_replay_has_invocation_lifetime():
  @function(precompile=True, allow_implicit=True)
  def producer(v): return runtime_activation((runtime_activation(v) + 1).contiguous())
  @TinyJit
  def consume(v):
    nested = runtime_activation(producer(v).contiguous()).reshape(1, 8).reshape(8)
    return (nested * 2).contiguous()
  # Warmup, capture, and multiple replays with distinct physical inputs. A
  # stale output-slot alias or dropped CALL dependency repeats an old value.
  for value in (1.0, 3.0, 7.0, 11.0):
    got = consume(Tensor.full((8,), value).contiguous()).numpy()
    np.testing.assert_allclose(got, np.full((8,), (value+1)*2, dtype=np.float32), rtol=0, atol=0)


def test_ffn_gate_up_after_inputs_are_direct_but_lazy_h_residual_materializes():
  # gate/up already have concrete producer identities. h is a live arithmetic
  # value, so the old fused down route still needs exactly one adapter; it also
  # retains the independently closed per-output-row SiLU recomputation defect.
  input_k, hidden, intermediate = 4096, 4096, 12288
  x = Tensor.empty(input_k, dtype=dtypes.float16)
  gate_up = _program(intermediate, input_k)
  gate = execute_promoted_program(None, _words(intermediate, input_k), x, program=gate_up)
  up = execute_promoted_program(None, _words(intermediate, input_k), x, program=gate_up)
  h = Tensor.empty(hidden, dtype=dtypes.float32) + 1
  down = _program(hidden, intermediate, Q4KGEMVEpilogue("ffn_down_fused"))
  fused = execute_promoted_program(None, _words(hidden, intermediate), gate, up, h, program=down)
  assert _calls(fused) == ["q4k_g3_lanemap_gemv_12288_4096", "q4k_g3_lanemap_gemv_12288_4096", "test",
                           "q4k_g3_lanemap_gemv_epi_ffndown_4096_12288"]
