"""M5 typed-boundary P0 unit tests
(docs/task_workflow/input/m5-variant-reopen-boundary-p0-scope-20260803.md sections 3 and 6):
the fp16 combine output declares a typed layout, the o-proj attn_qo Q4K GEMV opts in to the
typed input ABI, and the consumer's activation prelude
``x[:, 0, :].reshape(K).cast(fp16).contiguous()`` folds to a zero-copy view of the combine
AFTER. The ABI is closed-default; every validator failure rejects back to the generic
flat-buffer ABI, which keeps its materializing copy (byte-identical)."""
import hashlib
from types import SimpleNamespace

import pytest

from tinygrad import dtypes, Tensor
from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_kernel
from tinygrad.llm.decode_routes import Q4K_DECODE_CANDIDATE, _LinearDecodeBinding
from tinygrad.llm.flash_decode_attention import describe_flash_decode_attention
from tinygrad.llm.kernel_program import (DeclaredTypedOutput, KernelProgram, KernelProgramProvenance, OutputSpec,
                                         TypedLayout, TypedViewRequest, execute_promoted_program)
from tinygrad.llm.qk_layout import Q4_K
from tinygrad.uop.ops import Ops, UOp


Hq, Hd, S, MAXC = 32, 128, 48, 4608
COMBINE_F16 = "flash_fused_gmax_combine_f16_32_128"
COMBINE_F32 = "flash_fused_gmax_combine_32_128"
GEMV = "q4k_g3_lanemap_gemv_4096_4096"


def _combine_program(output_fp16: bool = True, declared: bool = True, admitted: bool = True) -> KernelProgram:
  spec = describe_flash_decode_attention(Hq, Hd, 8, MAXC, S, fused_combine=True, combine_fp16=output_fp16)
  typed_output = None
  if declared and output_fp16:
    typed_output = DeclaredTypedOutput(TypedLayout(dtypes.float16, (Hq * Hd,), (Hq, Hd)), admitted)
  return KernelProgram("decode_flash_live_split_g4_kvboth", "attention_decode.flash_live_split.combine",
    KernelProgramProvenance.MACHINE_SEARCH_GENERATED, spec.emit_combine(),
    output_spec=OutputSpec((Hq * Hd,), dtypes.float16 if output_fp16 else dtypes.float32, typed_output=typed_output))


def _gemv_program(opt_in: bool = True, route_role: str = "attn_qo", dtype=dtypes.float16,
                  flat_shape=(Hq * Hd,), requires: bool = True, route_id: str = "decode_q4k_g3_generated",
                  program_id: str = "quant_linear_decode.q4k_generated_g3.gemv") -> KernelProgram:
  typed_views = ()
  if opt_in:
    typed_views = (TypedViewRequest(slot=1, dtype=dtype, flat_shape=flat_shape, route_role=route_role,
                                    requires_combine_fusion=requires),)
  return KernelProgram(route_id, program_id, KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
    q4k_g3_lanemap_gemv_kernel(Hq * Hd, Hq * Hd), output_spec=OutputSpec((Hq * Hd,), dtypes.float32),
    typed_input_views=typed_views)


def _schedule_names(combine: KernelProgram, gemv: KernelProgram) -> list[str]:
  """Build fp16 combine AFTER -> attn output (with the model's transpose/reshape) -> attn_qo
  prelude -> GEMV, and return the linearized CALL function names in order. A scheduler-owned
  elementwise copy among them is the E_32_32_4_3b0fcfbc materialization class."""
  partial = Tensor.empty(Hq * S * (Hd + 2), dtype=dtypes.float32)
  out = execute_promoted_program(None, partial, program=combine)
  x = out.reshape(1, Hq, 1, Hd).cast(dtypes.float16)
  x = x.transpose(1, 2).reshape(1, 1, -1)
  v = x[:, 0, :].reshape(Hq * Hd).cast(dtypes.float16).contiguous()
  words = Tensor.empty(Hq * Hd * Hq * Hd // 16, dtype=dtypes.uint32)
  gemv_out = execute_promoted_program(None, words, v, program=gemv)
  linear, _ = gemv_out.linear_with_vars()
  names = []
  for u in linear.toposort():
    if u.op is Ops.CALL:
      name = getattr(getattr(u.src[0], "arg", None), "name", None)
      if name: names.append(name)
  return names


def _copy_calls(linear: UOp) -> list[UOp]:
  """The E_32_32_4_3b0fcfbc-shaped copy class: a CALL with exactly two buffers of identical
  shape and dtype (out and in), unlike the combine (two differently-typed buffers) or the
  GEMV (three buffers)."""
  copies = []
  for u in linear.toposort():
    if u.op is not Ops.CALL: continue
    bufs = [s.buf_uop for s in u.src[1:]]
    if len(bufs) == 2 and bufs[0].shape == bufs[1].shape and bufs[0].dtype == bufs[1].dtype:
      copies.append(u)
  return copies


# ── producer declaration ────────────────────────────────────────────────────

def test_fp16_combine_declares_typed_output_layout():
  program = _combine_program()
  declared = program.output_spec.typed_output
  assert declared is not None
  assert declared.layout.dtype is dtypes.float16
  assert declared.layout.flat_shape == (Hq * Hd,)
  assert declared.layout.logical_shape == (Hq, Hd)
  assert declared.layout.row_major is True
  assert declared.combine_fusion_admitted is True


def test_legacy_fp32_combine_declares_no_typed_output():
  program = _combine_program(output_fp16=False)
  assert program.output_spec.typed_output is None
  assert program.output_spec.dtype is dtypes.float32


def test_typed_layout_declaration_is_fail_closed_by_construction():
  with pytest.raises(ValueError, match="equal numel"):
    TypedLayout(dtypes.float16, (4096,), (32, 64))
  with pytest.raises(ValueError, match="positive"):
    TypedLayout(dtypes.float16, (0,), None)
  with pytest.raises(ValueError, match="row-major"):
    TypedLayout(dtypes.float16, (4096,), (32, 128), row_major=False)
  with pytest.raises(ValueError, match="must match"):
    OutputSpec((4096,), dtypes.float16,
      typed_output=DeclaredTypedOutput(TypedLayout(dtypes.float32, (4096,), (32, 128)), True))


def test_typed_view_request_and_program_validation():
  with pytest.raises(ValueError, match="slot"):
    TypedViewRequest(slot=-1, dtype=dtypes.float16, flat_shape=(4096,), route_role="attn_qo")
  with pytest.raises(ValueError, match="route_role"):
    TypedViewRequest(slot=1, dtype=dtypes.float16, flat_shape=(4096,), route_role="")
  with pytest.raises(ValueError, match="slots must be unique"):
    KernelProgram("r", "p", KernelProgramProvenance.MACHINE_SEARCH_GENERATED, lambda: None,
      typed_input_views=(TypedViewRequest(1, dtypes.float16, (4096,), "attn_qo"),
                         TypedViewRequest(1, dtypes.float16, (4096,), "attn_qo")))


# ── graph probe: fold vs fail-closed contrast (scope section 6.1) ──────────

def test_typed_abi_folds_contiguous_request_to_view_with_zero_materialization():
  assert _schedule_names(_combine_program(), _gemv_program()) == [COMBINE_F16, GEMV]


def test_real_model_fp32_pipeline_chain_folds_to_view_with_zero_materialization():
  # The real o-proj chain (model.py:704 with fp32 decode q + the prelude): the fp16
  # combine AFTER is upcast to fp32, permuted/reshaped, then the consumer prelude casts
  # back to fp16 and asks for contiguous. The fp16->fp32->fp16 pair is exact, so under
  # the typed ABI the contiguous request folds to a view of the AFTER with no copy.
  assert _real_schedule_names(_combine_program(), _gemv_program()) == [COMBINE_F16, GEMV]


def test_without_typed_abi_the_copy_is_materialized():
  names = _schedule_names(_combine_program(), _gemv_program(opt_in=False))
  assert names == [COMBINE_F16, "test", GEMV]
  linear, _ = _schedule_linear(_combine_program(), _gemv_program(opt_in=False))
  copies = _copy_calls(linear)
  assert len(copies) == 1
  bufs = [s.buf_uop for s in copies[0].src[1:]]
  assert bufs[0].shape == bufs[1].shape == (Hq * Hd,)
  assert bufs[0].dtype == bufs[1].dtype == dtypes.float16


def test_real_model_chain_without_abi_still_materializes_the_copy():
  # Fail-closed contrast on the real chain: without the typed ABI the fp32 pipeline
  # materializes the fp16 activation (the E_32_32_4_3b0fcfbc-shaped copy class).
  assert _real_schedule_names(_combine_program(), _gemv_program(opt_in=False)) == \
    [COMBINE_F16, "test", GEMV]
  linear, _ = _real_schedule_linear(_combine_program(), _gemv_program(opt_in=False))
  copies = _copy_calls(linear)
  assert len(copies) == 1
  bufs = [s.buf_uop for s in copies[0].src[1:]]
  assert bufs[0].shape == bufs[1].shape == (Hq * Hd,)
  assert bufs[0].dtype == bufs[1].dtype == dtypes.float16


def test_typed_abi_binds_the_gemv_to_the_combine_after_buffer():
  partial = Tensor.empty(Hq * S * (Hd + 2), dtype=dtypes.float32)
  out = execute_promoted_program(None, partial, program=_combine_program())
  after = out.uop
  x = out.reshape(1, Hq, 1, Hd).cast(dtypes.float16)
  x = x.transpose(1, 2).reshape(1, 1, -1)
  v = x[:, 0, :].reshape(Hq * Hd).cast(dtypes.float16).contiguous()
  words = Tensor.empty(Hq * Hd * Hq * Hd // 16, dtype=dtypes.uint32)
  gemv_out = execute_promoted_program(None, words, v, program=_gemv_program())
  linear, _ = gemv_out.linear_with_vars()
  gemv_calls = [u for u in linear.toposort() if u.op is Ops.CALL
                and getattr(getattr(u.src[0], "arg", None), "name", None) == GEMV]
  assert len(gemv_calls) == 1
  input_bufs = [s.buf_uop for s in gemv_calls[0].src[1:]]
  assert any(b is after.src[0] for b in input_bufs)


def test_lossy_bf16_roundtrip_rejects_to_generic_abi():
  # A bf16 intermediate between the casts is not the exact fp16->fp32->fp16 pair (bf16
  # round trips are lossy); the validator must reject and the copy stays materialized.
  partial = Tensor.empty(Hq * S * (Hd + 2), dtype=dtypes.float32)
  out = execute_promoted_program(None, partial, program=_combine_program())
  attn = out.reshape(1, Hq, 1, Hd).cast(dtypes.bfloat16).transpose(1, 2).reshape(1, 1, -1)
  v = attn[:, 0, :].reshape(Hq * Hd).cast(dtypes.float16).contiguous()
  words = Tensor.empty(Hq * Hd * Hq * Hd // 16, dtype=dtypes.uint32)
  gemv_out = execute_promoted_program(None, words, v, program=_gemv_program())
  linear, _ = gemv_out.linear_with_vars()
  names = [getattr(getattr(u.src[0], "arg", None), "name", None) for u in linear.toposort()
           if u.op is Ops.CALL]
  assert names == [COMBINE_F16, "test", GEMV]


def test_arithmetic_between_cast_pair_rejects_to_generic_abi():
  # fp32 arithmetic between the upcast and the downcast changes values; the pair is not
  # lossless, so the fold must reject and the copy stays materialized.
  partial = Tensor.empty(Hq * S * (Hd + 2), dtype=dtypes.float32)
  out = execute_promoted_program(None, partial, program=_combine_program())
  attn = (out.reshape(1, Hq, 1, Hd).cast(dtypes.float32) + 1.0)
  attn = attn.transpose(1, 2).reshape(1, 1, -1)
  v = attn[:, 0, :].reshape(Hq * Hd).cast(dtypes.float16).contiguous()
  words = Tensor.empty(Hq * Hd * Hq * Hd // 16, dtype=dtypes.uint32)
  gemv_out = execute_promoted_program(None, words, v, program=_gemv_program())
  linear, _ = gemv_out.linear_with_vars()
  names = [getattr(getattr(u.src[0], "arg", None), "name", None) for u in linear.toposort()
           if u.op is Ops.CALL]
  assert names == [COMBINE_F16, "test", GEMV]


def test_data_moving_permute_in_cast_pair_rejects_to_generic_abi():
  # A non-identity movement between the two casts: the composed fp16 view is not an
  # offset-0 reshape, so the fold must reject even though the cast pair is fp16->fp32->fp16.
  partial = Tensor.empty(Hq * S * (Hd + 2), dtype=dtypes.float32)
  out = execute_promoted_program(None, partial, program=_combine_program())
  attn = out.reshape(1, Hq, 1, Hd).cast(dtypes.float32).permute(0, 3, 2, 1)
  attn = attn.reshape(1, 1, -1)
  v = attn[:, 0, :].reshape(Hq * Hd).cast(dtypes.float16).contiguous()
  words = Tensor.empty(Hq * Hd * Hq * Hd // 16, dtype=dtypes.uint32)
  gemv_out = execute_promoted_program(None, words, v, program=_gemv_program())
  linear, _ = gemv_out.linear_with_vars()
  names = [getattr(getattr(u.src[0], "arg", None), "name", None) for u in linear.toposort()
           if u.op is Ops.CALL]
  assert names == [COMBINE_F16, "test", GEMV]


# ── fail-closed validator (scope section 4) ─────────────────────────────────

def test_missing_producer_declaration_rejects_to_generic_abi():
  assert _schedule_names(_combine_program(declared=False), _gemv_program()) == [COMBINE_F16, "test", GEMV]


def test_combine_fusion_gate_closed_rejects():
  assert _schedule_names(_combine_program(admitted=False), _gemv_program()) == [COMBINE_F16, "test", GEMV]


def test_wrong_consumer_route_role_rejects():
  assert _schedule_names(_combine_program(), _gemv_program(route_role="ffn_down")) == [COMBINE_F16, "test", GEMV]


def test_dtype_mismatch_rejects():
  assert _schedule_names(_combine_program(), _gemv_program(dtype=dtypes.float32)) == [COMBINE_F16, "test", GEMV]


def test_flat_shape_mismatch_rejects():
  assert _schedule_names(_combine_program(), _gemv_program(flat_shape=(Hq * Hd // 2,))) == [COMBINE_F16, "test", GEMV]


def test_typed_abi_gate_closed_rejects():
  assert _schedule_names(_combine_program(), _gemv_program(requires=False)) == [COMBINE_F16, "test", GEMV]


def test_wrong_program_identity_rejects():
  assert _schedule_names(_combine_program(),
                         _gemv_program(route_id="decode_q6k_coop_generated",
                                       program_id="quant_linear_decode.q6k_generated_coop.gemv")) == \
    [COMBINE_F16, "test", GEMV]


def test_legacy_fp32_combine_route_stays_byte_identical():
  # gate closed: fp32 combine + generic cast prelude; the typed request is rejected and the
  # legacy route (with its fp32->fp16 materializing cast) is unchanged.
  assert _schedule_names(_combine_program(output_fp16=False), _gemv_program()) == [COMBINE_F32, "test", GEMV]


def test_non_contiguous_request_is_rejected():
  import tinygrad.llm.kernel_program as kernel_program
  from tinygrad.llm.kernel_program import _validated_typed_view
  partial = Tensor.empty(Hq * S * (Hd + 2), dtype=dtypes.float32)
  out = execute_promoted_program(None, partial, program=_combine_program())
  x = out.reshape(1, Hq, 1, Hd).cast(dtypes.float16).transpose(1, 2).reshape(1, 1, -1)
  bare = x[:, 0, :].reshape(Hq * Hd).cast(dtypes.float16)  # no .contiguous()
  view, reason = _validated_typed_view(bare.uop, _gemv_program().typed_input_views[0], _gemv_program())
  assert view is None and "contiguous" in reason


def test_view_over_non_after_base_is_rejected():
  from tinygrad.llm.kernel_program import _validated_typed_view
  t = Tensor.empty(2 * 4 * 8, dtype=dtypes.float16).reshape(2, 4, 8)
  chain = t.permute(0, 2, 1).reshape(64).contiguous()
  assert chain.uop.op is Ops.CONTIGUOUS
  view, reason = _validated_typed_view(chain.uop, _gemv_program().typed_input_views[0], _gemv_program())
  assert view is None and "AFTER" in reason


# ── emitted kernel unchanged under the view ─────────────────────────────────

def test_gemv_function_ast_unchanged_under_typed_view():
  def gemv_function(opt_in: bool):
    partial = Tensor.empty(Hq * S * (Hd + 2), dtype=dtypes.float32)
    out = execute_promoted_program(None, partial, program=_combine_program())
    x = out.reshape(1, Hq, 1, Hd).cast(dtypes.float16).transpose(1, 2).reshape(1, 1, -1)
    v = x[:, 0, :].reshape(Hq * Hd).cast(dtypes.float16).contiguous()
    words = Tensor.empty(Hq * Hd * Hq * Hd // 16, dtype=dtypes.uint32)
    gemv_out = execute_promoted_program(None, words, v, program=_gemv_program(opt_in=opt_in))
    linear, _ = gemv_out.linear_with_vars()
    for u in linear.toposort():
      if u.op is Ops.CALL and getattr(getattr(u.src[0], "arg", None), "name", None) == GEMV:
        return u.src[0]
    raise AssertionError("GEMV call missing")
  plain, folded = gemv_function(False), gemv_function(True)
  assert plain.key == folded.key
  assert hashlib.sha256(repr(plain.key).encode()).hexdigest() == \
    hashlib.sha256(repr(folded.key).encode()).hexdigest()


def test_real_chain_gemv_function_ast_unchanged_under_typed_view():
  # The fp32 pipeline chain must also leave the emitted o-proj GEMV byte-identical: the
  # fold only changes the buffer binding, never the index math or buffer roles (scope 7).
  def gemv_function(opt_in: bool):
    partial = Tensor.empty(Hq * S * (Hd + 2), dtype=dtypes.float32)
    out = execute_promoted_program(None, partial, program=_combine_program())
    attn = out.reshape(1, Hq, 1, Hd).cast(dtypes.float32).transpose(1, 2).reshape(1, 1, -1)
    v = attn[:, 0, :].reshape(Hq * Hd).cast(dtypes.float16).contiguous()
    words = Tensor.empty(Hq * Hd * Hq * Hd // 16, dtype=dtypes.uint32)
    gemv_out = execute_promoted_program(None, words, v, program=_gemv_program(opt_in=opt_in))
    linear, _ = gemv_out.linear_with_vars()
    for u in linear.toposort():
      if u.op is Ops.CALL and getattr(getattr(u.src[0], "arg", None), "name", None) == GEMV:
        return u.src[0]
    raise AssertionError("GEMV call missing")
  plain, folded = gemv_function(False), gemv_function(True)
  assert plain.key == folded.key


# ── decode_routes consumer wiring ───────────────────────────────────────────

def _decode_binding() -> _LinearDecodeBinding:
  return _LinearDecodeBinding("quant_linear_decode.q4k_generated_g3", "decode_q4k_g3_generated", Q4_K,
                              "amd_gfx1100", 1, 1, Hq * Hd, Hq * Hd)


class _FakeQ4KLinear:
  def __init__(self, route_role: str, q4k_epi: bool = False):
    self.route_role = route_role
    self.q4k_storage = SimpleNamespace(mode="sidecar",
                                       words=Tensor.empty(Hq * Hd * Hq * Hd // 16, dtype=dtypes.uint32))
    self.route_admission = SimpleNamespace(q4k_epilogue_fusion_admitted=q4k_epi)


def _attn_output_chain():
  partial = Tensor.empty(Hq * S * (Hd + 2), dtype=dtypes.float32)
  out = execute_promoted_program(None, partial, program=_combine_program())
  return out.reshape(1, Hq, 1, Hd).cast(dtypes.float16).transpose(1, 2).reshape(1, 1, -1)


def test_decode_routes_o_proj_opts_in_and_folds():
  x = _attn_output_chain()
  out = Q4K_DECODE_CANDIDATE.execute(_FakeQ4KLinear("attn_qo"), x, _decode_binding())
  linear, _ = out.linear_with_vars()
  names = [getattr(getattr(u.src[0], "arg", None), "name", None) for u in linear.toposort()
           if u.op is Ops.CALL]
  assert COMBINE_F16 in names and GEMV in names
  assert _copy_calls(linear) == []


def test_decode_routes_o_proj_folds_real_model_fp32_pipeline_chain():
  # The decode_routes consumer wiring must fold the full real-model chain: the fp16
  # combine AFTER upcast to fp32 (model.py:704), permuted/reshaped, then the prelude's
  # cast back to fp16 + contiguous. No E_32_32_4_3b0fcfbc-shaped copy may be scheduled.
  partial = Tensor.empty(Hq * S * (Hd + 2), dtype=dtypes.float32)
  out = execute_promoted_program(None, partial, program=_combine_program())
  attn = out.reshape(1, Hq, 1, Hd).cast(dtypes.float32).transpose(1, 2).reshape(1, 1, -1)
  gemv_out = Q4K_DECODE_CANDIDATE.execute(_FakeQ4KLinear("attn_qo"), attn, _decode_binding())
  linear, _ = gemv_out.linear_with_vars()
  names = [getattr(getattr(u.src[0], "arg", None), "name", None) for u in linear.toposort()
           if u.op is Ops.CALL]
  assert COMBINE_F16 in names and GEMV in names
  assert _copy_calls(linear) == []


def test_decode_routes_attn_kv_keeps_generic_flat_buffer_abi():
  x = _attn_output_chain()
  out = Q4K_DECODE_CANDIDATE.execute(_FakeQ4KLinear("attn_kv", q4k_epi=True), x, _decode_binding())
  linear, _ = out.linear_with_vars()
  # attn_kv keeps the generic flat-buffer ABI: no typed input views are issued, so the
  # activation materialization copy is still scheduled (the fail-closed contrast).
  copies = _copy_calls(linear)
  assert len(copies) == 1
  bufs = [s.buf_uop for s in copies[0].src[1:]]
  assert bufs[0].shape == bufs[1].shape == (Hq * Hd,) and bufs[0].dtype == bufs[1].dtype == dtypes.float16


def _schedule_linear(combine: KernelProgram, gemv: KernelProgram):
  partial = Tensor.empty(Hq * S * (Hd + 2), dtype=dtypes.float32)
  out = execute_promoted_program(None, partial, program=combine)
  x = out.reshape(1, Hq, 1, Hd).cast(dtypes.float16)
  x = x.transpose(1, 2).reshape(1, 1, -1)
  v = x[:, 0, :].reshape(Hq * Hd).cast(dtypes.float16).contiguous()
  words = Tensor.empty(Hq * Hd * Hq * Hd // 16, dtype=dtypes.uint32)
  gemv_out = execute_promoted_program(None, words, v, program=gemv)
  return gemv_out.linear_with_vars()


def _real_schedule_names(combine: KernelProgram, gemv: KernelProgram) -> list[str]:
  partial = Tensor.empty(Hq * S * (Hd + 2), dtype=dtypes.float32)
  out = execute_promoted_program(None, partial, program=combine)
  attn = out.reshape(1, Hq, 1, Hd).cast(dtypes.float32).transpose(1, 2).reshape(1, 1, -1)
  v = attn[:, 0, :].reshape(Hq * Hd).cast(dtypes.float16).contiguous()
  words = Tensor.empty(Hq * Hd * Hq * Hd // 16, dtype=dtypes.uint32)
  gemv_out = execute_promoted_program(None, words, v, program=gemv)
  linear, _ = gemv_out.linear_with_vars()
  names = []
  for u in linear.toposort():
    if u.op is Ops.CALL:
      name = getattr(getattr(u.src[0], "arg", None), "name", None)
      if name: names.append(name)
  return names


def _real_schedule_linear(combine: KernelProgram, gemv: KernelProgram):
  partial = Tensor.empty(Hq * S * (Hd + 2), dtype=dtypes.float32)
  out = execute_promoted_program(None, partial, program=combine)
  attn = out.reshape(1, Hq, 1, Hd).cast(dtypes.float32).transpose(1, 2).reshape(1, 1, -1)
  v = attn[:, 0, :].reshape(Hq * Hd).cast(dtypes.float16).contiguous()
  words = Tensor.empty(Hq * Hd * Hq * Hd // 16, dtype=dtypes.uint32)
  gemv_out = execute_promoted_program(None, words, v, program=gemv)
  return gemv_out.linear_with_vars()
