from pathlib import Path

from tinygrad.llm import prefill_routes
from tinygrad.llm.prefill_attachments import PrefillDirectPackedBinding, PrefillRouteAttachment


class _Weight:
  def __init__(self, calls): self.calls = calls
  def cast(self, dtype): self.calls.append(("weight_cast", dtype)); return self
  def transpose(self): self.calls.append(("weight_transpose",)); return self


class _Activation:
  def __init__(self, shape, calls): self.shape, self.calls = shape, calls
  def cast(self, dtype): self.calls.append(("activation_cast", dtype)); return self
  def linear(self, weight, bias):
    self.calls.append(("linear", weight, bias))
    return "generic-result"


class _Linear:
  bias = None
  def __init__(self, calls, *, quant="q4", shape=(512, 5120, 5120), role="attn_qo"):
    self.weight = _Weight(calls)
    self.out_features, self.in_features = shape[1], shape[2]
    self._prefill_graph_role = role
    self.prefill_packed_weight = lambda: None
    setattr(self, f"{quant}k_storage", object())


def _legacy_attachment(lin, shape, role="attn_qo", tensor="blk.0.attn_q.weight"):
  candidate = "direct-packed-baseline"
  lin._prefill_route_attachment = PrefillRouteAttachment(
    "invocation", candidate, tensor, {"candidate_id": candidate}, {"backend": "AMD"})
  lin._prefill_direct_packed_binding = PrefillDirectPackedBinding("invocation", "prefill", role, shape)


def test_exact_selected_packed_wmma_result_wins(monkeypatch):
  sentinel = object()
  monkeypatch.setattr(prefill_routes, "_attached_production_route", lambda *_: "packed_wmma")
  monkeypatch.setattr(prefill_routes, "prefill_route_mode", lambda: "auto")
  monkeypatch.setattr(prefill_routes, "route_packed_wmma_prefill", lambda *_: sentinel)
  assert prefill_routes.route_prefill_linear(object(), object()) is sentinel


def test_selector_decline_falls_through_to_ordinary_tinygrad_graph(monkeypatch):
  calls = []
  lin, x = _Linear(calls), _Activation((1, 512, 5120), calls)
  monkeypatch.setattr(prefill_routes, "_attached_production_route", lambda *_: "packed_wmma")
  monkeypatch.setattr(prefill_routes, "prefill_route_mode", lambda: "auto")
  monkeypatch.setattr(prefill_routes, "route_packed_wmma_prefill", lambda *_: None)
  assert prefill_routes.route_prefill_linear(lin, x) == "generic-result"
  assert any(call[0] == "linear" for call in calls)


def test_disabled_packed_wmma_never_enters_selector(monkeypatch):
  calls = []
  lin, x = _Linear(calls), _Activation((1, 512, 5120), calls)
  monkeypatch.setattr(prefill_routes, "_attached_production_route", lambda *_: "packed_wmma")
  monkeypatch.setattr(prefill_routes, "prefill_route_mode", lambda: "fp16")
  monkeypatch.setattr(prefill_routes, "route_packed_wmma_prefill",
                      lambda *_: (_ for _ in ()).throw(AssertionError("disabled selector ran")))
  assert prefill_routes.route_prefill_linear(lin, x) == "generic-result"


def test_unsupported_packed_shape_declines_to_generic(monkeypatch):
  calls = []
  shape = (256, 5120, 5120)  # not one of the six exact M=512 production rows
  lin, x = _Linear(calls, shape=shape), _Activation((1, shape[0], shape[2]), calls)
  _legacy_attachment(lin, shape)
  monkeypatch.setattr(prefill_routes, "prefill_route_mode", lambda: "auto")
  monkeypatch.setattr(prefill_routes, "select_packed_wmma_prefill_candidate", lambda *_: None)
  from tinygrad.llm.prefill_route_observer import prefill_route_scope
  with prefill_route_scope(): result = prefill_routes.route_prefill_linear(lin, x)
  assert result == "generic-result"


def test_exact_q6_vocab_uses_generic_graph_when_packed_selector_declines(monkeypatch):
  calls = []
  shape = (512, 151936, 4096)
  lin, x = _Linear(calls, quant="q6", shape=shape, role="lm_head"), _Activation((1, shape[0], shape[2]), calls)
  _legacy_attachment(lin, shape, role="lm_head", tensor="output.weight")
  monkeypatch.setattr(prefill_routes, "prefill_route_mode", lambda: "auto")
  selected = []
  monkeypatch.setattr(prefill_routes, "select_packed_wmma_prefill_candidate",
                      lambda *_: selected.append(True))
  from tinygrad.llm.prefill_route_observer import prefill_route_scope
  with prefill_route_scope(): result = prefill_routes.route_prefill_linear(lin, x)
  assert result == "generic-result" and selected == [True]


def test_unattached_general_linear_uses_generic_graph(monkeypatch):
  calls = []
  lin, x = _Linear(calls), _Activation((1, 17, 5120), calls)
  monkeypatch.setattr(prefill_routes, "prefill_route_mode", lambda: "auto")
  assert prefill_routes.route_prefill_linear(lin, x) == "generic-result"


def test_runtime_source_has_no_handwritten_direct_packed_emitters():
  source = Path(prefill_routes.__file__).read_text()
  for forbidden in ("emit_q4k_packed_prefill_kernel", "emit_q6k_packed_prefill_kernel",
                    "DirectPackedPrefillCandidate", "route_direct_packed_prefill", "custom_kernel"):
    assert forbidden not in source
