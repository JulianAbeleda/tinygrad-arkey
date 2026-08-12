"""Hermetic CPU tests for the M2b ffn_down in-kernel residual add.

The ffn_down Q4K/Q6K decode GEMVs store fp32; the graph then renders the
ordinary E_32_32_4_02a9738c fp32 add (h + ffn_out) after every block.  Under
the harness-installed ``_ffn_down_resadd_lease`` the GEMVs render distinct
``*_epi_ffnresadd`` variants that add the hidden-state residual h in-kernel
(``total + h[row]``, fp32 store).  The in-kernel add is the same fp32
expression the separate add kernel lowers, so the stored bytes are
bitwise-identical.  The fp16-store spelling of the original M2b premise is NOT
bitwise-safe: the next block's attention residual consumes the fp32 block
output, so the block output dtype must not change (scope doc section 3 M2b).
The legacy kernel names and routes are unchanged when the lease is absent.
"""
import numpy as np
import pytest

from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.llm.decode_kernels import (Q4KGEMVEpilogue, Q6KGEMVRouteSpec, emit_q6k_gemv_kernel,
  q4k_g3_lanemap_gemv_kernel, q6k_spec_for_role)
from tinygrad.llm.decode_routes import q4k_primitive_linear_call, q6k_primitive_linear_call
from tinygrad.llm.kernel_program import (DeclaredTypedOutput, KernelProgram, KernelProgramProvenance,
  ResidualViewRequest, TypedLayout, _DECLARED_TYPED_OUTPUTS, _validated_residual_view)
from tinygrad.llm.memory_semantics import RUNTIME_ACTIVATION, mark_memory_semantic
from tinygrad.llm.model import TransformerBlock, TransformerConfig
from tinygrad.llm.qk_primitives import Q4KPrimitiveLinear
from tinygrad.llm.qk_primitives import QKPrimitiveCapability, QKPrimitiveRouteAdmission
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops, UOp


_N = 4096


def _render_cuda(ast: UOp) -> str:
  src = next(u.arg for u in to_program(ast, CUDARenderer(Target("NV", arch="sm_120"))).src
             if u.op is Ops.SOURCE)
  return src


def test_q4k_ffn_down_resadd_kernel_name_and_validate():
  epi = Q4KGEMVEpilogue("ffn_down_resadd")
  assert epi.kernel_suffix == "_epi_ffnresadd"
  out = UOp.placeholder((4096,), dtypes.float32, 0)
  words = UOp.placeholder((4096 * (12288 // 256) * 36,), dtypes.uint32, 1)
  x = UOp.placeholder((12288,), dtypes.float16, 2)
  h = UOp.placeholder((4096,), dtypes.float32, 3)
  ast = q4k_g3_lanemap_gemv_kernel(4096, 12288, epilogue=epi)(out, words, x, h)
  assert ast.arg.name == "q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288"
  with pytest.raises(ValueError, match="requires rows=4096"):
    Q4KGEMVEpilogue("ffn_down_resadd").validate(32, 1024)
  with pytest.raises(ValueError, match="unsupported"):
    Q4KGEMVEpilogue("not_a_kind").validate(4096, 12288)


def test_q4k_ffn_down_resadd_renders_the_in_kernel_add():
  epi = Q4KGEMVEpilogue("ffn_down_resadd")
  out = UOp.placeholder((4096,), dtypes.float32, 0)
  words = UOp.placeholder((4096 * (12288 // 256) * 36,), dtypes.uint32, 1)
  x = UOp.placeholder((12288,), dtypes.float16, 2)
  h = UOp.placeholder((4096,), dtypes.float32, 3)
  src = _render_cuda(q4k_g3_lanemap_gemv_kernel(4096, 12288, epilogue=epi)(out, words, x, h))
  # The residual h arrives as the first extra input (data3) and is added to the
  # per-row total inside the GEMV body; the store stays fp32.
  assert "float* data3_4096" in src
  assert "(*(data3_4096+gidx0))" in src
  assert "((half)" not in src
  # The plain variant has no extra input and no residual add.
  plain = q4k_g3_lanemap_gemv_kernel(4096, 12288)
  plain_src = _render_cuda(plain(out, words, x))
  assert "data3" not in plain_src
  assert plain(out, words, x).arg.name == "q4k_g3_lanemap_gemv_4096_12288"


def test_q6k_ffn_down_resadd_spec_name_and_validate():
  spec = q6k_spec_for_role(4096, 12288, row_tile=2, reduction="in_kernel",
                           target="NV:sm_120", epilogue="ffn_down_resadd")
  assert spec.kernel_name == "q6k_gen_coop_4096_12288_inkernel_epi_ffnresadd"
  assert spec.to_json()["epilogue"] == "ffn_down_resadd"
  with pytest.raises(ValueError, match="requires in_kernel"):
    Q6KGEMVRouteSpec(rows=4096, k=12288, row_tile=2, reduction="external_sum", epilogue="ffn_down_resadd").validate()
  with pytest.raises(ValueError, match="requires rows=4096"):
    Q6KGEMVRouteSpec(rows=32, k=1024, row_tile=2, reduction="in_kernel", epilogue="ffn_down_resadd").validate()
  with pytest.raises(ValueError, match="unsupported epilogue"):
    Q6KGEMVRouteSpec(rows=4096, k=12288, reduction="in_kernel", epilogue="bogus").validate()


def test_q6k_ffn_down_resadd_renders_h_input_through_cuda():
  spec = q6k_spec_for_role(4096, 12288, row_tile=2, reduction="in_kernel",
                           target="NV:sm_120", epilogue="ffn_down_resadd")
  kernel = emit_q6k_gemv_kernel(spec)
  partials = UOp.placeholder((4096,), dtypes.float32, 0)
  halfs = UOp.placeholder((12288 // 16 * 16,), dtypes.uint16, 1)
  x = UOp.placeholder((12288,), dtypes.float16, 2)
  h = UOp.placeholder((4096,), dtypes.float32, 3)
  ast = kernel(partials, halfs, x, h)
  assert ast.arg.name == spec.kernel_name
  src = _render_cuda(ast)
  assert "half* data2_12288" in src and "float* data3_4096" in src
  # The residual h is consumed in-kernel (loaded from data3) and the store is
  # fp32; the plain variant never sees a data3 input.
  assert "data3_4096" in src
  plain = emit_q6k_gemv_kernel(q6k_spec_for_role(4096, 12288, row_tile=2, reduction="in_kernel",
                                                 target="NV:sm_120"))
  plain_src = _render_cuda(plain(partials, halfs, x))
  assert "data3" not in plain_src


class _FakeQ4KFFNDown:
  def __init__(self, leased: bool):
    self.route_admission = QKPrimitiveRouteAdmission(QKPrimitiveCapability("NV", "sm_120", 32, True), True)
    self.bias, self.decode_enabled = None, True
    self.out_features, self.in_features = 4096, 12288
    self.route_role = "ffn_down"
    self.q4k_storage = type("S", (), {"mode": "sidecar",
                                      "words": Tensor.zeros(4096 * (12288 // 256) * 36, dtype=dtypes.uint32, device="CPU")} )()
    self._ffn_down_resadd_lease = leased


def _call_names(t: Tensor) -> list[str]:
  names = []
  for call in t.schedule_linear().src:
    arg = call.src[0].arg
    if arg is not None and getattr(arg, "name", None) is not None:
      names.append(arg.name)
  return names


def test_q4k_route_picks_ffn_down_resadd_only_under_the_lease():
  x = Tensor.zeros((1, 1, 12288), dtype=dtypes.float16, device="CPU")
  h = Tensor.zeros((1, 1, 4096), dtype=dtypes.float32, device="CPU")
  leased = q4k_primitive_linear_call(_FakeQ4KFFNDown(True), x, fallback=lambda _: x, arch_ok=True,
                                     epilogue_inputs={"normed_h": h})
  names = _call_names(leased)
  assert "q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288" in names
  assert not any("gemv_4096_12288" in name and "ffnresadd" not in name for name in names)
  assert leased.dtype == dtypes.float32

  closed = q4k_primitive_linear_call(_FakeQ4KFFNDown(False), x, fallback=lambda _: x, arch_ok=True,
                                     epilogue_inputs={"normed_h": h})
  closed_names = _call_names(closed)
  assert "q4k_g3_lanemap_gemv_4096_12288" in closed_names
  assert not any("ffnresadd" in name for name in closed_names)


def test_q4k_route_fallback_reproduces_the_add_when_leased_but_unbound():
  # Lease on, normed_h threaded, but arch_ok=False -> binding misses: the route
  # must reproduce h + ffn_out so the block graph is unchanged (fail-closed).
  x = Tensor.zeros((1, 1, 12288), dtype=dtypes.float16, device="CPU")
  h = Tensor.ones((1, 1, 4096), dtype=dtypes.float32, device="CPU")
  linear = _FakeQ4KFFNDown(True)
  linear.route_admission = QKPrimitiveRouteAdmission(QKPrimitiveCapability("METAL", "m3", 32, True), True)
  out = q4k_primitive_linear_call(linear, x, fallback=lambda _: Tensor.ones((1, 1, 4096), dtype=dtypes.float32, device="CPU"),
                                  arch_ok=False, epilogue_inputs={"normed_h": h})
  # The route reproduced h + ffn_out (ones + ones), so the block graph is unchanged.
  assert np.asarray(out.numpy()).min() == 2.0


def test_q4k_ffn_down_resadd_declares_typed_fp32_output(monkeypatch):
  # The absorbing ffn_down GEMV's fp32 AFTER is the concrete block output; the route must
  # declare the typed layout (mirroring the w1w3fused16 producer) so the next block's
  # attention residual fold can bind it in place instead of rendering a boundary copy.
  from tinygrad.llm import decode_routes
  captured = []

  def fake_execute(output, *inputs, program):
    captured.append(program)
    return Tensor.zeros(*program.output_spec.shape, dtype=program.output_spec.dtype, device="CPU")

  monkeypatch.setattr(decode_routes, "execute_promoted_program", fake_execute)
  x = Tensor.zeros((1, 1, 12288), dtype=dtypes.float16, device="CPU")
  h = Tensor.zeros((1, 1, 4096), dtype=dtypes.float32, device="CPU")
  decode_routes.q4k_primitive_linear_call(_FakeQ4KFFNDown(True), x, fallback=lambda _: x, arch_ok=True,
                                          epilogue_inputs={"normed_h": h})
  assert len(captured) == 1
  typed = captured[0].output_spec.typed_output
  assert typed is not None
  assert typed.layout.dtype is dtypes.float32
  assert typed.layout.flat_shape == (4096,)
  assert typed.layout.logical_shape == (1, 1, 4096)
  assert typed.combine_fusion_admitted is False
  assert typed.epilogue_absorption_admitted is True
  assert len(captured[0].residual_input_views) == 1
  rv = captured[0].residual_input_views[0]
  assert rv.slot == 2 and rv.route_role == "ffn_down" and rv.kind == "residual_add"
  assert rv.dtype is dtypes.float32 and rv.flat_shape == (4096,)


def test_q4k_ffn_down_closed_lease_declares_no_typed_output(monkeypatch):
  from tinygrad.llm import decode_routes
  captured = []

  def fake_execute(output, *inputs, program):
    captured.append(program)
    return Tensor.zeros(*program.output_spec.shape, dtype=program.output_spec.dtype, device="CPU")

  monkeypatch.setattr(decode_routes, "execute_promoted_program", fake_execute)
  x = Tensor.zeros((1, 1, 12288), dtype=dtypes.float16, device="CPU")
  h = Tensor.zeros((1, 1, 4096), dtype=dtypes.float32, device="CPU")
  decode_routes.q4k_primitive_linear_call(_FakeQ4KFFNDown(False), x, fallback=lambda _: x, arch_ok=True,
                                          epilogue_inputs={"normed_h": h})
  assert len(captured) == 1
  assert captured[0].output_spec.typed_output is None
  assert captured[0].residual_input_views == ()


def test_m2b_ffn_down_after_accepts_next_block_residual_view():
  # Next-block attn_qo residual_add contract: with the declaration, the ffn_down AFTER (viewed
  # through the block's reshape) validates as a zero-copy residual producer; without it the
  # validator rejects and the generic flat-buffer boundary copy stays (fail-closed).
  x = Tensor.zeros((1, 1, 12288), dtype=dtypes.float16, device="CPU")
  h = Tensor.zeros((1, 1, 4096), dtype=dtypes.float32, device="CPU")
  out = q4k_primitive_linear_call(_FakeQ4KFFNDown(True), x, fallback=lambda _: x, arch_ok=True,
                                  epilogue_inputs={"normed_h": h})
  consumer = KernelProgram("decode_q4k_attn_qo", "candidate.gemv",
                           KernelProgramProvenance.MACHINE_SEARCH_GENERATED, emitter=lambda *a, **k: None)
  request = ResidualViewRequest(slot=2, dtype=dtypes.float32, flat_shape=(4096,), route_role="attn_qo")
  view, reason = _validated_residual_view(out.uop, request, consumer)
  assert reason == "ok"
  assert view is not None and view.base.op is Ops.AFTER

  # M2b ffn_down consumer spelling: the same zero-copy residual contract, with the
  # epilogue-absorption gate on the producer's declared typed output.
  ff_consumer = KernelProgram("decode_q6k_ffn_down", "candidate.gemv",
                              KernelProgramProvenance.MACHINE_SEARCH_GENERATED, emitter=lambda *a, **k: None)
  ff_request = ResidualViewRequest(slot=2, dtype=dtypes.float32, flat_shape=(4096,), route_role="ffn_down")
  view, reason = _validated_residual_view(out.uop, ff_request, ff_consumer)
  assert reason == "ok"
  assert view is not None and view.base.op is Ops.AFTER

  closed = q4k_primitive_linear_call(_FakeQ4KFFNDown(False), x, fallback=lambda _: x, arch_ok=True,
                                     epilogue_inputs={"normed_h": h})
  view, reason = _validated_residual_view(closed.uop, request, consumer)
  assert view is None and "no buffer/precompiled-output identity" in reason
  view, reason = _validated_residual_view(closed.uop, ff_request, ff_consumer)
  assert view is None


def _declared_after_chain(admitted: bool = True, key: str = "after", sink_owner: bool = True) -> tuple[UOp, UOp]:
  """M2c structural spelling: MS(RESHAPE(AFTER(param, CALL))) over a declared
  epilogue-absorbing fp32 typed output, plus the caller output slot t. The
  declaration is keyed on the AFTER itself (``key="after"``) or on the opaque
  CALL's SINK body (``key="sink"``), mirroring the two lookups callify performs
  after an enclosing @function substitutes its inputs."""
  param = UOp.param(0, dtypes.float32, (_N,))
  body = UOp.sink(param)
  call = body.call(param, name=f"q4k_g3_lanemap_gemv_epi_ffnresadd_{_N}_{12288}", precompile=True)
  after = param.after(call)
  declared = DeclaredTypedOutput(TypedLayout(dtypes.float32, (_N,), (1, 1, _N)),
                                 combine_fusion_admitted=False, epilogue_absorption_admitted=admitted)
  if key == "after":
    _DECLARED_TYPED_OUTPUTS[after] = declared
  else:
    _DECLARED_TYPED_OUTPUTS[call.src[0]] = declared
  s = mark_memory_semantic(after.reshape(1, 1, _N), RUNTIME_ACTIVATION)
  t = UOp.param(4, dtypes.float32, (_N,))
  return s, t


def test_m2c_declared_after_rebind_proves_the_output_slot_contract():
  from tinygrad.callify import _body_output_is_declared_after, _declared_after_output_slot_rebind, \
    _declared_epilogue_absorption_after
  s, t = _declared_after_chain(key="after")
  assert _declared_epilogue_absorption_after(s) is not None
  assert _body_output_is_declared_after((s,)) is True
  rebind = _declared_after_output_slot_rebind(s, t)
  assert rebind is not None
  param, view = rebind
  assert param.op is Ops.PARAM and param.dtype is dtypes.float32 and param.numel() == _N
  assert view.dtype is dtypes.float32 and view.numel() == _N


def test_m2c_declared_after_sink_key_survives_after_substitution():
  # The AFTER node is rebuilt when an enclosing @function substitutes its inputs, so the
  # declaration is also keyed by the opaque CALL's SINK body; both spellings must fire.
  from tinygrad.callify import _body_output_is_declared_after, _declared_after_output_slot_rebind, \
    _declared_epilogue_absorption_after
  s, t = _declared_after_chain(key="sink")
  assert _declared_epilogue_absorption_after(s) is not None
  assert _body_output_is_declared_after((s,)) is True
  assert _declared_after_output_slot_rebind(s, t) is not None


def test_m2c_declared_after_rebind_is_fail_closed():
  from tinygrad.callify import _body_output_is_declared_after, _declared_after_output_slot_rebind, \
    _declared_epilogue_absorption_after
  # No declaration at all: every helper fails closed.
  s, t = _declared_after_chain(admitted=False, key="after")
  del _DECLARED_TYPED_OUTPUTS[s.src[0].src[0]]  # drop the AFTER-keyed declaration
  s2, t2 = _declared_after_chain(admitted=False, key="sink")
  for chain in ((s, t), (s2, t2)):
    assert _declared_epilogue_absorption_after(chain[0]) is None
    assert _body_output_is_declared_after((chain[0],)) is False
    assert _declared_after_output_slot_rebind(*chain) is None
  # A non-absorbing declaration also fails closed.
  s3, t3 = _declared_after_chain(admitted=False, key="after")
  assert _declared_epilogue_absorption_after(s3) is None
  assert _declared_after_output_slot_rebind(s3, t3) is None


def test_m2c_declared_after_rebind_rejects_wrong_output_slot():
  from tinygrad.callify import _declared_after_output_slot_rebind
  s, t = _declared_after_chain(key="after")
  # Wrong dtype or span on the caller output slot rejects back to the generic spelling.
  assert _declared_after_output_slot_rebind(s, UOp.param(4, dtypes.float16, (_N,))) is None
  assert _declared_after_output_slot_rebind(s, UOp.param(4, dtypes.float32, (_N // 2,))) is None
  # A body whose AFTER wraps one param while the CALL writes a different slot rejects
  # (the rebind requires the AFTER's own output param to be the CALL's slot-1 writer).
  param = UOp.param(0, dtypes.float32, (_N,))
  other = UOp.param(1, dtypes.float32, (_N,))
  body = UOp.sink(other)
  call = body.call(other, name=f"q4k_g3_lanemap_gemv_epi_ffnresadd_{_N}_{12288}", precompile=True)
  after = param.after(call)
  _DECLARED_TYPED_OUTPUTS[after] = DeclaredTypedOutput(TypedLayout(dtypes.float32, (_N,), (1, 1, _N)),
                                                       combine_fusion_admitted=False,
                                                       epilogue_absorption_admitted=True)
  s_bad = mark_memory_semantic(after.reshape(1, 1, _N), RUNTIME_ACTIVATION)
  assert _declared_after_output_slot_rebind(s_bad, t) is None


class _FakeQ6KFFNDown:
  def __init__(self, leased: bool, fused: bool):
    self.q6k_storage = type("S", (), {"halfs": Tensor.zeros(16, dtype=dtypes.uint16, device="CPU")})()
    self.decode_enabled, self.bias = True, None
    self.in_features, self.out_features = 12288, 4096
    self.parts, self.opts = 1, ()
    self.route_admission = QKPrimitiveRouteAdmission(QKPrimitiveCapability("NV", "sm_120", 32, True), True,
                                                     epilogue_fusion_promoted=fused)
    self.route_role = "ffn_down"
    self._ffn_down_resadd_lease = leased


def test_q6k_route_picks_ffn_down_resadd_under_the_lease(monkeypatch):
  from tinygrad.llm import decode_routes
  linear = _FakeQ6KFFNDown(leased=True, fused=True)
  x = Tensor.zeros((1, 1, 12288), dtype=dtypes.float16, device="CPU")
  h = Tensor.zeros((1, 1, 4096), dtype=dtypes.float32, device="CPU")
  captured = []
  captured_specs = []

  real_emit = decode_routes.emit_q6k_gemv_kernel

  def spy_emit(spec):
    captured_specs.append(spec)
    return real_emit(spec)

  def fake_execute(output, *inputs, program):
    captured.append(program)
    return Tensor.zeros(*program.output_spec.shape, dtype=program.output_spec.dtype, device="CPU")

  monkeypatch.setattr(decode_routes, "emit_q6k_gemv_kernel", spy_emit)
  monkeypatch.setattr(decode_routes, "execute_promoted_program", fake_execute)
  out = decode_routes.q6k_primitive_linear_call(linear, x, fallback=lambda _: x, arch_ok=True,
                                                epilogue_inputs={"normed_h": h})
  assert len(captured) == 1
  assert len(captured_specs) == 1
  assert captured_specs[0].epilogue == "ffn_down_resadd"
  assert captured_specs[0].kernel_name == "q6k_gen_coop_4096_12288_inkernel_epi_ffnresadd"
  assert captured[0].output_spec.dtype is dtypes.float32
  assert tuple(captured[0].output_spec.shape) == (4096,)
  assert out.dtype is dtypes.float32
  typed = captured[0].output_spec.typed_output
  assert typed is not None
  assert typed.layout.dtype is dtypes.float32
  assert typed.layout.flat_shape == (4096,)
  assert typed.layout.logical_shape == (1, 1, 4096)
  assert typed.combine_fusion_admitted is False
  assert typed.epilogue_absorption_admitted is True
  assert len(captured[0].residual_input_views) == 1
  rv = captured[0].residual_input_views[0]
  assert rv.slot == 2 and rv.route_role == "ffn_down" and rv.kind == "residual_add"
  assert rv.dtype is dtypes.float32 and rv.flat_shape == (4096,)


def test_q6k_route_keeps_legacy_without_lease_or_fusion(monkeypatch):
  from tinygrad.llm import decode_routes
  captured_specs = []
  captured_programs = []
  real_emit = decode_routes.emit_q6k_gemv_kernel

  def spy_emit(spec):
    captured_specs.append(spec)
    return real_emit(spec)

  def fake_execute(output, *inputs, program):
    captured_programs.append(program)
    return Tensor.zeros(*program.output_spec.shape, dtype=program.output_spec.dtype, device="CPU")

  monkeypatch.setattr(decode_routes, "emit_q6k_gemv_kernel", spy_emit)
  monkeypatch.setattr(decode_routes, "execute_promoted_program", fake_execute)
  for leased, fused in ((False, True), (True, False), (False, False)):
    linear = _FakeQ6KFFNDown(leased=leased, fused=fused)
    x = Tensor.zeros((1, 1, 12288), dtype=dtypes.float16, device="CPU")
    h = Tensor.zeros((1, 1, 4096), dtype=dtypes.float32, device="CPU")
    captured_specs.clear()
    decode_routes.q6k_primitive_linear_call(linear, x, fallback=lambda _: x, arch_ok=True,
                                            epilogue_inputs={"normed_h": h})
    assert len(captured_specs) == 1
    assert captured_specs[0].epilogue == ""
    assert len(captured_programs) == 1
    assert captured_programs[0].output_spec.typed_output is None
    assert captured_programs[0].residual_input_views == ()
    captured_programs.clear()


def test_model_absorbed_block_return_does_not_force_contiguous(monkeypatch):
  # The model's M2b branch returns the ffn_down GEMV output in place (same pattern as the M4
  # attn_out residual): no .contiguous() is forced on the absorbed block output, so no
  # E_32_32_4 copy materializes between blocks. The spy fails if the absorbed return touches
  # the ffn_down tensor's .contiguous().
  class _IdentityAttnBlock(TransformerBlock):
    def _attention(self, x, start_pos, ring_freqs=None, residual_for_output=None):
      return x

  block = _IdentityAttnBlock(TransformerConfig(num_blocks=1, dim=8, hidden_dim=16, n_heads=2, n_kv_heads=1,
                                               norm_eps=1e-5, vocab_size=32, head_dim=4, rope_theta=10001.0,
                                               rope_dim=4, v_head_dim=4, max_context=8))
  block._ffn_down_resadd_lease = True
  linear = Q4KPrimitiveLinear(None, None, Tensor.empty(8, dtype=dtypes.uint32, device="CPU"), 1, 1, 1, (),
                              "q4", 32, 0, "shared", route_role="ffn_down",
                              route_admission=QKPrimitiveRouteAdmission(QKPrimitiveCapability("NV", "sm_120", 32, True), True))
  linear._ffn_down_resadd_lease = True
  block.ffn_down = linear

  marker = Tensor.zeros((1, 1, 8), dtype=dtypes.float32, device="CPU")
  calls = []
  real_contiguous = Tensor.contiguous

  def spy_contiguous(self):
    if self is marker:
      calls.append("contiguous")
    return real_contiguous(self)

  monkeypatch.setattr(Tensor, "contiguous", spy_contiguous)
  monkeypatch.setattr(block, "_feed_forward", lambda x, residual=None: marker)
  out = block(Tensor.zeros((1, 1, 8), dtype=dtypes.float32, device="CPU"), 0)
  assert calls == []
  assert out is not None
