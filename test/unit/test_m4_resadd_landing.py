"""M4 residual_add landing unit tests
(docs/task_workflow/input/m4-resadd-landing-scope-20260806.md section 2):
the o-proj attn_qo residual_add Q4K GEMV opts in to the residual-slot typed input ABI, and the
residual chain ``epi_inputs["residual"][:, 0, :].reshape(N).cast(fp32)`` folds to a zero-copy
view of the ordinary block-output producer. The ABI is closed-default; every validator failure
rejects back to the generic flat-buffer ABI, which keeps its materializing copy. The combined
M4 record and the M5 combine ABI are untouched."""
from extra.llm_research.decode.m4_residual_boundary_fold_probe import _block_output_producer, residual_chain
from tinygrad import Tensor, dtypes
from tinygrad.llm.decode_kernels import Q4KGEMVEpilogue, q4k_g3_lanemap_gemv_kernel
from tinygrad.llm.kernel_program import (KernelProgram, KernelProgramProvenance, OutputSpec,
                                         ResidualViewRequest, execute_promoted_program)
from tinygrad.uop.ops import Ops


N = 4096
EPI_RESADD = f"q4k_g3_lanemap_gemv_epi_resadd_{N}_{N}"
LEGACY = f"q4k_g3_lanemap_gemv_{N}_{N}"


def _gemv_program(opt_in: bool = True, route_role: str = "attn_qo", kind: str = "residual_add",
                  slot: int = 2, dtype=dtypes.float32, flat_shape=(N,),
                  route_id: str = "decode_q4k_g3_generated",
                  program_id: str = "quant_linear_decode.q4k_generated_g3.gemv") -> KernelProgram:
  residual_views = ()
  if opt_in:
    residual_views = (ResidualViewRequest(slot=slot, dtype=dtype, flat_shape=flat_shape,
                                          route_role=route_role, kind=kind),)
  return KernelProgram(route_id, program_id, KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
    q4k_g3_lanemap_gemv_kernel(N, N, epilogue=Q4KGEMVEpilogue("residual_add")),
    output_spec=OutputSpec((N,), dtypes.float32),
    residual_input_views=residual_views)


def _copy_calls(linear):
  """The E_32_32_4_86a2-shaped copy class: a CALL with exactly two same-shape/same-dtype buffers."""
  out = []
  for u in linear.toposort():
    if u.op is not Ops.CALL: continue
    bufs = [s.buf_uop for s in u.src[1:]]
    if len(bufs) == 2 and bufs[0].shape == bufs[1].shape and bufs[0].dtype == bufs[1].dtype:
      out.append(u)
  return out


def _schedule(opt_in: bool, producer=None, request: ResidualViewRequest | None = None) -> dict:
  """One schedule through the PRODUCTION path: with the residual opt-in issued, the fold fires
  inside execute_promoted_program (_fold_residual_input_views); without it, the generic
  flat-buffer ABI materializes the boundary copy."""
  from tinygrad.llm.kernel_program import _validated_residual_view
  producer = producer or _block_output_producer()
  resid = residual_chain(producer)
  program = _gemv_program(opt_in)
  verdict = {"fold": False, "reason": "no residual request supplied"}
  if opt_in and request is not None:
    view, reason = _validated_residual_view(resid.uop, request, program)
    verdict = {"fold": view is not None, "reason": reason, "base_op": None if view is None else str(view.op)}
  words = Tensor.empty(N * N // 16, dtype=dtypes.uint32)
  xv = Tensor.empty(N, dtype=dtypes.float16)
  out = execute_promoted_program(Tensor.empty(N, dtype=dtypes.float32), words, xv, resid, program=program)
  linear, _ = out.linear_with_vars()
  gemv = [u for u in linear.toposort() if u.op is Ops.CALL
          and getattr(getattr(u.src[0], "arg", None), "name", None) == EPI_RESADD]
  residual_buf = None
  if gemv:
    bufs = [s.buf_uop for s in gemv[0].src[1:]]
    if len(bufs) >= 4: residual_buf = {"shape": list(bufs[3].shape), "dtype": str(bufs[3].dtype)}
  return {"verdict": verdict, "copies": _copy_calls(linear), "gemv_residual_buf": residual_buf,
          "names": [getattr(getattr(u.src[0], "arg", None), "name", None)
                    for u in linear.toposort() if u.op is Ops.CALL]}


# ── request + program validation ──────────────────────────────────────────

def test_residual_view_request_validation():
  import pytest
  with pytest.raises(ValueError, match="slot"):
    ResidualViewRequest(slot=-1, dtype=dtypes.float32, flat_shape=(N,), route_role="attn_qo")
  with pytest.raises(ValueError, match="route_role"):
    ResidualViewRequest(slot=2, dtype=dtypes.float32, flat_shape=(N,), route_role="")
  with pytest.raises(ValueError, match="flat_shape"):
    ResidualViewRequest(slot=2, dtype=dtypes.float32, flat_shape=(), route_role="attn_qo")
  with pytest.raises(ValueError, match="slots must be unique"):
    KernelProgram("r", "p", KernelProgramProvenance.MACHINE_SEARCH_GENERATED, lambda: None,
      residual_input_views=(ResidualViewRequest(2, dtypes.float32, (N,), "attn_qo"),
                            ResidualViewRequest(2, dtypes.float32, (N,), "attn_qo")))
  with pytest.raises(ValueError, match="must not overlap"):
    from tinygrad.llm.kernel_program import TypedViewRequest
    KernelProgram("r", "p", KernelProgramProvenance.MACHINE_SEARCH_GENERATED, lambda: None,
      typed_input_views=(TypedViewRequest(1, dtypes.float16, (N,), "attn_qo"),),
      residual_input_views=(ResidualViewRequest(1, dtypes.float32, (N,), "attn_qo"),))


# ── validator contract (probe-1 extended contract, ported) ────────────────

def test_validator_fires_on_real_block_output_chain():
  from tinygrad.llm.kernel_program import _validated_residual_view
  resid = residual_chain(_block_output_producer())
  view, reason = _validated_residual_view(resid.uop, ResidualViewRequest(
    slot=2, dtype=dtypes.float32, flat_shape=(N,), route_role="attn_qo"), _gemv_program())
  assert view is not None and reason == "ok"
  assert view.op is Ops.CONTIGUOUS


def test_validator_fails_closed_on_wrong_opt_in():
  from tinygrad.llm.kernel_program import _validated_residual_view
  resid = residual_chain(_block_output_producer()).uop
  base = ResidualViewRequest(slot=2, dtype=dtypes.float32, flat_shape=(N,), route_role="attn_qo")
  assert _validated_residual_view(resid, ResidualViewRequest(slot=1, dtype=dtypes.float32, flat_shape=(N,), route_role="attn_qo"), _gemv_program())[1] == "not the residual slot"
  assert _validated_residual_view(resid, ResidualViewRequest(slot=2, dtype=dtypes.float32, flat_shape=(N,), route_role="ffn_down"), _gemv_program())[1].startswith("wrong consumer route_role")
  assert _validated_residual_view(resid, ResidualViewRequest(slot=2, dtype=dtypes.float32, flat_shape=(N,), route_role="attn_qo", kind="ffn_down_fused"), _gemv_program())[1].startswith("wrong epilogue kind")
  assert _validated_residual_view(resid, ResidualViewRequest(slot=2, dtype=dtypes.float16, flat_shape=(N,), route_role="attn_qo"), _gemv_program())[1] == "request dtype mismatch"
  assert _validated_residual_view(resid, ResidualViewRequest(slot=2, dtype=dtypes.float32, flat_shape=(N // 2,), route_role="attn_qo"), _gemv_program())[1] == "request numel mismatch"


def test_validator_rejects_non_q4k_program_and_impure_chain():
  from tinygrad.llm.kernel_program import _validated_residual_view
  producer = _block_output_producer()
  resid = residual_chain(producer).uop
  wrong_program = _gemv_program(route_id="decode_other", program_id="other.gemv")
  assert _validated_residual_view(resid, ResidualViewRequest(slot=2, dtype=dtypes.float32, flat_shape=(N,), route_role="attn_qo"), wrong_program)[1] == "program is not a q4k GEMV consumer"
  impure = producer.transpose(1, 2).reshape(1, 1, N)
  chain = impure[:, 0, :].reshape(N).cast(dtypes.float32).uop
  assert _validated_residual_view(chain, ResidualViewRequest(slot=2, dtype=dtypes.float32, flat_shape=(N,), route_role="attn_qo"), _gemv_program())[1] == "view is not a contiguous offset-0 reshape"


# ── graph probe: fold vs fail-closed contrast ─────────────────────────────

def test_fold_removes_the_boundary_copy_with_zero_materialization():
  request = ResidualViewRequest(slot=2, dtype=dtypes.float32, flat_shape=(N,), route_role="attn_qo")
  with_fold = _schedule(True, request=request)
  assert with_fold["verdict"]["fold"] is True
  assert len(with_fold["copies"]) == 0
  assert with_fold["gemv_residual_buf"] == {"shape": [N], "dtype": "dtypes.float"}


def test_without_residual_request_the_copy_stays():
  without = _schedule(False)
  assert len(without["copies"]) == 1
  assert EPI_RESADD in without["names"]


def test_fold_substitutes_the_producer_buffer_into_the_gemv():
  request = ResidualViewRequest(slot=2, dtype=dtypes.float32, flat_shape=(N,), route_role="attn_qo")
  row = _schedule(True, request=request)
  assert EPI_RESADD in row["names"]
  assert LEGACY not in row["names"]


# ── record + admission gating ─────────────────────────────────────────────

def test_resadd_record_is_closed_default():
  import tinygrad.llm.model_route_plan as mrp
  assert mrp.decode_q4k_epilogue_resadd_promoted(("NV", "sm_120")) is False
  assert mrp.decode_q4k_epilogue_resadd_promoted(("NV", "gfx1100")) is False


def test_combined_m4_record_stays_closed():
  import tinygrad.llm.model_route_plan as mrp
  assert mrp.decode_q4k_epilogue_fusion_promoted(("NV", "sm_120")) is False
