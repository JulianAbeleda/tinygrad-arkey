"""Decode route shape gate (model_route_plan.DecodeRouteShapeGate): a promotion record names the target, the
kernel's own validate names the shape. A record-promoted route whose kernel refuses this model's (rows, k) is not
promoted for the model, the route plan says why (one line per route, "plain kernels used"), and the plain tinygrad
kernels run. A harness lease forcing such a route raises the kernel's refusal with "lease forced".

Seen on the RTX 5090 with Qwen3.5-27B (dim 5120, ffn 17408): the promoted ffn_down residual-add epilogue reached
`Q4KGEMVEpilogue.validate` and raised "requires rows=4096, got rows=5120" instead of falling back."""
from types import SimpleNamespace

import numpy as np
import pytest

from tinygrad import Tensor, dtypes
from tinygrad.llm import decode_routes
from tinygrad.llm.decode_kernels import Q4KGEMVEpilogue
from tinygrad.llm.decode_routes import DECODE_ROUTE_SHAPE_PROBES, q4k_primitive_linear_call
from tinygrad.llm.model import _decode_route_promotions
from tinygrad.llm.model_route_plan import DecodeRouteShapeGate, DecodeRouteShapes, build_model_route_plan, lease_forced
from tinygrad.llm.qk_primitives import QKPrimitiveCapability, QKPrimitiveRouteAdmission

NV = ("NV", "sm_120")
_NO_ENV = lambda *_a, **_k: 0

def _config(dim, hidden_dim, n_heads, n_kv_heads, head_dim, vocab):
  return SimpleNamespace(dim=dim, hidden_dim=hidden_dim, n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim,
                         vocab_size=vocab, norm_eps=1e-6)

QWEN3_8B = _config(4096, 12288, 32, 8, 128, 151936)      # the qualified production shape
QWEN35_27B = _config(5120, 17408, 24, 4, 256, 248320)    # the shape that raised on the 5090

FFN_DOWN_ROUTES = ("decode_ffn_down_resadd", "decode_q4k_ffn_down_fp16_geometry", "decode_q6k_ffn_down_fp16_geometry")
W1W3_ROUTES = ("decode_q4k_w1w3_fusion", "decode_q4k_w1w3_fp16_store", "decode_q4k_gate_up_four_warp_vector")
ATTENTION_ROUTES = ("decode_shared_q8_attention", "decode_q4_direct_shared_q8_attention", "decode_reduce_output_rmsnorm",
                    "decode_qk_norm_rope", "decode_q6k_v_four_warp_fp16_geometry", "decode_q4k_kv_pair")


def test_gate_rule_record_first_then_the_kernels_validate():
  shapes = DecodeRouteShapes(5120, 17408, 24, 4, 256, 248320, 1e-6, NV)
  asked = []
  def refuses(s): asked.append(s); raise ValueError("needs rows=4096")
  gate = DecodeRouteShapeGate(shapes, {"refused": refuses, "accepted": lambda s: None})
  assert gate.admit("refused", False) is False and asked == []            # closed record: the kernel is never asked
  assert gate.admit("refused", True) is False and asked == [shapes]        # open record, kernel refuses: not promoted
  assert gate.admit("accepted", True) is True
  assert gate.admit("no_probe", True) is True                              # no probe: the record alone decides
  assert gate.admitted("refused") is False and gate.admitted("accepted") and gate.admitted("never_asked") is False
  assert gate.report_lines() == ["refused: needs rows=4096; plain kernels used"]
  assert shapes.kv_dim == 1024


def test_every_probe_builds_the_kernel_spec_for_the_qualified_shape():
  shapes = DecodeRouteShapes(4096, 12288, 32, 8, 128, 151936, 1e-6, NV)
  for route, probe in DECODE_ROUTE_SHAPE_PROBES.items():
    probe(shapes)  # the qualified production shape must raise nowhere


def test_dim_4096_keeps_every_promoted_route():
  gate = _decode_route_promotions(QWEN3_8B, NV, getenv_fn=_NO_ENV)
  assert gate.skipped == [] and gate.report_lines() == []
  for route in FFN_DOWN_ROUTES + W1W3_ROUTES + ATTENTION_ROUTES: assert gate.admitted(route), route


def test_dim_5120_gets_no_promoted_ffn_down_route_and_the_plan_says_why():
  gate = _decode_route_promotions(QWEN35_27B, NV, getenv_fn=_NO_ENV)
  for route in FFN_DOWN_ROUTES: assert not gate.admitted(route), route
  reasons = {s.route: s.reason for s in gate.skipped}
  assert reasons["decode_ffn_down_resadd"] == "ffn_down_resadd epilogue requires rows=4096, got rows=5120"
  # the geometry routes are ANDed with the residual-add promotion, so they were never promoted here (nothing to refuse)
  assert "decode_q4k_ffn_down_fp16_geometry" not in reasons and "decode_q6k_ffn_down_fp16_geometry" not in reasons
  # the fused w1+w3 GEMV lane map accepts k=5120 (20 Q4_K blocks over 4 lane groups): that route keeps its promotion;
  # the four-warp gate/up kernel is fixed to 12288x4096 and says so
  assert gate.admitted("decode_q4k_w1w3_fusion") and gate.admitted("decode_q4k_w1w3_fp16_store")
  assert reasons["decode_q4k_gate_up_four_warp_vector"] == "Q4_K gate/up four-warp route requires (rows,k)=(12288,4096), got (17408,5120)"
  # the attention routes are fixed to the 32x128 / 8x128 head geometry
  for route in ATTENTION_ROUTES: assert not gate.admitted(route), route
  assert reasons["decode_shared_q8_attention"].startswith("shared-Q8 attention requires (dim,q_rows,kv_rows)=(4096,4096,1024)")
  assert reasons["decode_reduce_output_rmsnorm"] == "reduce-output requires rows in (1, 8, 32), got 24"
  plan = build_model_route_plan(None, None, shape_gate=gate)
  assert plan.report_lines() == gate.report_lines()
  assert "decode_ffn_down_resadd: ffn_down_resadd epilogue requires rows=4096, got rows=5120; plain kernels used" in plan.report_lines()
  assert all(line.endswith("; plain kernels used") for line in plan.report_lines())
  assert build_model_route_plan(None, None).report_lines() == []


def test_closed_records_are_never_probed():
  # No record promotes anything on this target, so every route is off and nothing is reported as refused.
  gate = _decode_route_promotions(QWEN35_27B, ("AMD", "gfx1100"), getenv_fn=_NO_ENV)
  assert gate.skipped == [] and not any(gate.decisions.values())


def test_lease_forced_carries_the_validate_message_and_names_the_lease():
  with pytest.raises(ValueError, match=r"decode_ffn_down_resadd: ffn_down_resadd epilogue requires rows=4096, got rows=5120 \(lease forced\)"):
    lease_forced("decode_ffn_down_resadd", lambda: Q4KGEMVEpilogue("ffn_down_resadd").validate(5120, 17408))
  lease_forced("decode_ffn_down_resadd", lambda: Q4KGEMVEpilogue("ffn_down_resadd").validate(4096, 12288))


class _FakeQ4KFFNDown:
  def __init__(self, rows:int, k:int, *, leased:bool=False, promoted:bool=False, arch=("NV", "sm_120")):
    self.route_admission = QKPrimitiveRouteAdmission(QKPrimitiveCapability(*arch, 32, True), True)
    self.bias, self.decode_enabled = None, True
    self.out_features, self.in_features = rows, k
    self.route_role = "ffn_down"
    self.q4k_storage = type("S", (), {"mode": "sidecar", "words": Tensor.zeros(rows * (k // 256) * 36, dtype=dtypes.uint32, device="CPU")})()
    self._ffn_down_resadd_lease = leased
    self._decode_ffn_down_resadd_promoted = promoted


def _call_names(t:Tensor) -> list[str]:
  names = []
  for call in t.schedule_linear().src:
    arg = call.src[0].arg
    if arg is not None and getattr(arg, "name", None) is not None: names.append(arg.name)
  return names


def test_route_not_admitted_returns_the_plain_fallback_unchanged():
  x = Tensor.zeros((1, 1, 17408), dtype=dtypes.float16, device="CPU")
  h = Tensor.ones((1, 1, 5120), dtype=dtypes.float32, device="CPU")
  marker = Tensor.full((1, 1, 5120), 3.0, dtype=dtypes.float32, device="CPU")
  # Binding miss, no lease, no promoted flag: fallback(x) is returned as-is, no epilogue, no add.
  out = q4k_primitive_linear_call(_FakeQ4KFFNDown(5120, 17408), x, fallback=lambda _: marker, arch_ok=False,
                                  epilogue_inputs={"normed_h": h})
  assert out is marker
  # Binding miss under the promoted flag (the block threaded normed_h): fallback(x) + normed_h, no epilogue.
  out = q4k_primitive_linear_call(_FakeQ4KFFNDown(5120, 17408, promoted=True), x, fallback=lambda _: marker, arch_ok=False,
                                  epilogue_inputs={"normed_h": h})
  assert np.asarray(out.numpy()).min() == 4.0 and np.asarray(out.numpy()).max() == 4.0
  # No flags at all, the binding holds: the plain GEMV lowers, no residual-add epilogue anywhere.
  plain = q4k_primitive_linear_call(_FakeQ4KFFNDown(5120, 17408), x, fallback=lambda _: marker, arch_ok=True)
  names = _call_names(plain)
  assert "q4k_g3_lanemap_gemv_vec_5120_17408" in names and not any("ffnresadd" in n for n in names)


def _zeros_execute(captured):
  def fake_execute(output, *inputs, program):
    captured.append(program)
    return Tensor.zeros(*program.output_spec.shape, dtype=program.output_spec.dtype, device="CPU")
  return fake_execute


def test_promoted_flag_on_a_refused_shape_lowers_the_plain_gemv_and_adds_h(monkeypatch):
  # Belt and braces behind the loader gate: if the promoted flag ever reaches a shape the epilogue refuses,
  # the route lowers the plain GEMV (no epilogue contract declared) and issues the ordinary h+ffn_out add.
  captured = []
  monkeypatch.setattr(decode_routes, "execute_promoted_program", _zeros_execute(captured))
  x = Tensor.zeros((1, 1, 17408), dtype=dtypes.float16, device="CPU")
  h = Tensor.ones((1, 1, 5120), dtype=dtypes.float32, device="CPU")
  out = q4k_primitive_linear_call(_FakeQ4KFFNDown(5120, 17408, promoted=True), x, fallback=lambda _: x, arch_ok=True,
                                  epilogue_inputs={"normed_h": h})
  assert len(captured) == 1 and len(captured[0].residual_input_views) == 0 and captured[0].output_spec.typed_output is None
  assert tuple(captured[0].output_spec.shape) == (5120,) and len(captured[0].typed_input_views) == 1  # plain ffn_down GEMV ABI
  assert out.shape == (1, 1, 5120) and out.dtype == dtypes.float32
  assert np.asarray(out.numpy()).min() == 1.0 and np.asarray(out.numpy()).max() == 1.0  # zeros + h


def test_lease_on_a_refused_shape_raises_lease_forced():
  x = Tensor.zeros((1, 1, 17408), dtype=dtypes.float16, device="CPU")
  h = Tensor.ones((1, 1, 5120), dtype=dtypes.float32, device="CPU")
  with pytest.raises(ValueError, match=r"requires rows=4096, got rows=5120 \(lease forced\)"):
    q4k_primitive_linear_call(_FakeQ4KFFNDown(5120, 17408, leased=True), x, fallback=lambda _: x, arch_ok=True,
                              epilogue_inputs={"normed_h": h})


def test_qualified_shape_still_lowers_the_fused_epilogue_under_the_promoted_flag():
  x = Tensor.zeros((1, 1, 12288), dtype=dtypes.float16, device="CPU")
  h = Tensor.zeros((1, 1, 4096), dtype=dtypes.float32, device="CPU")
  out = q4k_primitive_linear_call(_FakeQ4KFFNDown(4096, 12288, promoted=True), x, fallback=lambda _: x, arch_ok=True,
                                  epilogue_inputs={"normed_h": h})
  assert "q4k_g3_lanemap_gemv_vec_epi_ffnresadd_4096_12288" in _call_names(out)


class _FakeQ6KFFNDown:
  def __init__(self, rows:int, k:int, *, leased:bool=False, promoted:bool=False):
    self.route_admission = QKPrimitiveRouteAdmission(QKPrimitiveCapability("NV", "sm_120", 32, True), True,
                                                     epilogue_fusion_promoted=True)
    self.bias, self.decode_enabled, self.parts, self.opts = None, True, 1, ()
    self.out_features, self.in_features = rows, k
    self.route_role = "ffn_down"
    self.q6k_storage = type("S", (), {"mode": "sidecar", "halfs": Tensor.zeros(rows * (k // 256) * 105, dtype=dtypes.uint16, device="CPU")})()
    self._ffn_down_resadd_lease = leased
    self._decode_ffn_down_resadd_promoted = promoted


def test_q6k_promoted_flag_on_a_refused_shape_lowers_the_plain_coop_gemv_and_adds_h(monkeypatch):
  captured = []
  monkeypatch.setattr(decode_routes, "execute_promoted_program", _zeros_execute(captured))
  x = Tensor.zeros((1, 1, 17408), dtype=dtypes.float16, device="CPU")
  h = Tensor.ones((1, 1, 5120), dtype=dtypes.float32, device="CPU")
  out = decode_routes.q6k_primitive_linear_call(_FakeQ6KFFNDown(5120, 17408, promoted=True), x, fallback=lambda _: x, arch_ok=True,
                                                epilogue_inputs={"normed_h": h})
  assert len(captured) == 1 and len(captured[0].residual_input_views) == 0 and captured[0].output_spec.typed_output is None
  assert tuple(captured[0].output_spec.shape) == (5120,)  # the in-kernel coop reduce, no residual epilogue
  assert out.shape == (1, 1, 5120) and np.asarray(out.numpy()).min() == 1.0 and np.asarray(out.numpy()).max() == 1.0
  with pytest.raises(ValueError, match=r"requires rows=4096, got rows=5120 \(lease forced\)"):
    decode_routes.q6k_primitive_linear_call(_FakeQ6KFFNDown(5120, 17408, leased=True), x, fallback=lambda _: x, arch_ok=True,
                                            epilogue_inputs={"normed_h": h})
