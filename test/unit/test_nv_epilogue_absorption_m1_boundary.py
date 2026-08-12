"""Hermetic CPU tests for the M1 raw-x activation-slot boundary fold.

The M1 fused w1+w3 rms-affine gate/up GEMV absorbs the ffn-norm epilogue
in-kernel, so its raw-x input is the fp32 block hidden state h: the SAME AFTER
the scale reduce (r_16_256) and the ffn_down residual read.  Without the
activation-slot typed-input fold the generic flat-buffer ABI materializes an
identity transport copy per block (E_32_32_4_86a23e1a), exactly cancelling the
M1 -36 program fold.  The fold must bind the raw-x slot to the epi_resadd
AFTER (no copy) and stay closed-default on any mismatch (copy stays).
"""
import pytest
from tinygrad import Tensor, dtypes
from tinygrad.engine.realize import compile_linear
from tinygrad.llm.decode_kernels import Q4KGEMVEpilogue, q4k_g3_lanemap_gemv_kernel, q4k_g3_lanemap_gemv_w1w3_rms_affine_kernel
from tinygrad.llm.kernel_program import (ActivationViewRequest, DeclaredTypedOutput, KernelProgram,
                                         KernelProgramProvenance, OutputSpec, TypedLayout,
                                         _validated_activation_view, execute_promoted_program,
                                         execute_research_program)
from tinygrad.llm.memory_semantics import runtime_activation
from tinygrad.uop.ops import Ops
from extra.llm_research.decode.nv_epilogue_absorption_m1_ab import validate_cost_prediction


TRANSPORT_COPY = "E_32_32_4_86a23e1a5cd1cbd6101066fd85449138b653e9ecbb53d1d704f32aa470cd6f2b"
N = K = 4096


def _cost_bracket(candidate_ms: float, control_ms: float) -> dict:
  # Mirrors the harness bracket field: (control - candidate) * 1000, positive = candidate FASTER.
  return {"candidate_minus_control_bracket_us": (control_ms - candidate_ms) * 1000.0}


def _cost_census(norm_median_us: float) -> dict:
  return {"histogram": [["E_32_32_4_f14a5cc0", 37, norm_median_us]]}


def test_m1_cost_gate_contradicts_measured_loss():
  # The M1 campaign measured candidate 84.4us SLOWER (5.251 vs 5.167 ms).  The
  # bracket field is (control - candidate) = -84.4; the cost gate must negate it
  # so the reconcile sees a +84.4 loss, which CONTRADICTS the predicted -25.2 win.
  bracket = _cost_bracket(5.25143134375, 5.1669955000000005)
  census = _cost_census(2.30)
  out = validate_cost_prediction(bracket, census, census)
  assert out["result"] == "FAIL"
  assert out["reconciliation"]["result"] == "CONTRADICTED"
  assert out["measured_delta_us"] == pytest.approx(84.43584374999969)
  assert out["bracket_field_us"] == pytest.approx(-84.43584374999969)


def test_m1_cost_gate_confirms_measured_win():
  # Candidate 20us faster than control: bracket field +20 -> measured -20, inside
  # the tolerance band around the -25.2 point prediction -> CONFIRMED / PASS.
  bracket = _cost_bracket(5.147, 5.167)
  census = _cost_census(2.30)
  out = validate_cost_prediction(bracket, census, census)
  assert out["result"] == "PASS"
  assert out["reconciliation"]["result"] == "CONFIRMED"


def _words(n: int, k: int) -> Tensor:
  return Tensor.empty(n * (k // 256) * 36, dtype=dtypes.uint32)


def _resadd_program(n: int, k: int) -> KernelProgram:
  return KernelProgram("decode_q4k_g3_generated", "quant_linear_decode.q4k_generated_g3.gemv",
    KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
    q4k_g3_lanemap_gemv_kernel(n, k, epilogue=Q4KGEMVEpilogue("residual_add")),
    output_spec=OutputSpec((n,), dtypes.float32,
      typed_output=DeclaredTypedOutput(TypedLayout(dtypes.float32, (n,), (1, 1, n)),
                                       combine_fusion_admitted=False, epilogue_absorption_admitted=True)))


def _m1_program() -> KernelProgram:
  return KernelProgram("decode_q4k_g3_generated", "quant_linear_decode.q4k_generated_g3.rms_affine_qualification",
  KernelProgramProvenance.RESEARCH_ONLY,
  q4k_g3_lanemap_gemv_w1w3_rms_affine_kernel(12288, K, store_fp16=True),
  output_spec=OutputSpec((12288,), dtypes.float16),
  activation_input_views=(ActivationViewRequest(slot=2, dtype=dtypes.float32, flat_shape=(K,), route_role="ffn_norm"),))


def _hidden_state() -> tuple[Tensor, Tensor]:
  h = execute_promoted_program(None, _words(N, K), Tensor.empty(K, dtype=dtypes.float16),
                               Tensor.empty(N, dtype=dtypes.float32), program=_resadd_program(N, K))
  # The model wraps the block hidden state in the RUNTIME_ACTIVATION role mark
  # (model.py _prefill_semantic); the fold proof walks it as a transparent leg.
  return runtime_activation(h.reshape(1, 1, N)), h


def _compiled_names(out: Tensor) -> list[str]:
  linear = compile_linear(out.linear_with_vars()[0])
  return [getattr(u.src[0].arg, "name", "") for u in linear.toposort() if u.op is Ops.CALL]


def _m1_calls(out: Tensor) -> list:
  return [u for u in out.uop.toposort()
          if u.op is Ops.CALL and u.src[0].op is Ops.SINK
          and getattr(u.src[0].arg, "name", "").startswith("q4k_g3_lanemap_w1w3_rms_affine16_")]


def test_m1_raw_x_folds_to_epi_resadd_after_no_transport_copy():
  h, _ = _hidden_state()
  xv = h[:, 0, :].reshape(K).contiguous()
  scale = (h.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt().reshape(1)
  out = execute_research_program(None, _words(12288, K), _words(12288, K), xv,
                                 Tensor.empty(K, dtype=dtypes.float16), scale, program=_m1_program())
  calls = _m1_calls(out)
  assert len(calls) == 1
  # raw-x is input slot 3 (CALL src index 4): the fold bound the producer AFTER directly.
  assert calls[0].src[4].op is Ops.AFTER
  names = _compiled_names(out)
  assert TRANSPORT_COPY not in names
  assert any(name.startswith("q4k_g3_lanemap_w1w3_rms_affine16_") for name in names)


def test_m1_raw_x_closed_default_rollback_keeps_flat_buffer_abi():
  h, _ = _hidden_state()
  xv = h[:, 0, :].reshape(K).contiguous()
  scale = (h.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt().reshape(1)
  program = KernelProgram("decode_q4k_g3_generated", "quant_linear_decode.q4k_generated_g3.rms_affine_qualification",
    KernelProgramProvenance.RESEARCH_ONLY,
    q4k_g3_lanemap_gemv_w1w3_rms_affine_kernel(12288, K, store_fp16=True),
    output_spec=OutputSpec((12288,), dtypes.float16))
  out = execute_research_program(None, _words(12288, K), _words(12288, K), xv,
                                 Tensor.empty(K, dtype=dtypes.float16), scale, program=program)
  calls = _m1_calls(out)
  assert len(calls) == 1
  # Without the opt-in the raw-x input stays the generic flat-buffer ABI (a
  # view chain, not the producer AFTER); the transport materialization itself is
  # scheduler/arena-dependent and is gated by the harness census on the GPU arm.
  assert calls[0].src[4].op is not Ops.AFTER


def test_activation_view_validator_is_fail_closed():
  h, _ = _hidden_state()
  xv = h[:, 0, :].reshape(K).contiguous()
  request = ActivationViewRequest(slot=2, dtype=dtypes.float32, flat_shape=(K,), route_role="ffn_norm")
  ok_program = _m1_program()
  view, reason = _validated_activation_view(xv.uop, request, ok_program)
  assert view is not None and reason == "ok"
  # A consumer that is not the M1 qualification program is rejected.
  wrong = KernelProgram("decode_q4k_g3_generated", "quant_linear_decode.q4k_generated_g3.gemv",
    KernelProgramProvenance.RESEARCH_ONLY,
    q4k_g3_lanemap_gemv_w1w3_rms_affine_kernel(12288, K, store_fp16=True),
    output_spec=OutputSpec((12288,), dtypes.float16),
    activation_input_views=(request,))
  assert _validated_activation_view(xv.uop, request, wrong)[0] is None
  # A producer WITHOUT a declared typed output is rejected.
  h_plain = execute_promoted_program(None, _words(N, K), Tensor.empty(K, dtype=dtypes.float16),
                                     Tensor.empty(N, dtype=dtypes.float32),
                                     program=KernelProgram("decode_q4k_g3_generated",
                                       "quant_linear_decode.q4k_generated_g3.gemv",
                                       KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
                                       q4k_g3_lanemap_gemv_kernel(N, K, epilogue=Q4KGEMVEpilogue("residual_add")),
                                       output_spec=OutputSpec((N,), dtypes.float32))).reshape(1, 1, N)
  xv_plain = h_plain[:, 0, :].reshape(K).contiguous()
  assert _validated_activation_view(xv_plain.uop, request, ok_program)[0] is None
