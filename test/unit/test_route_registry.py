from __future__ import annotations

from dataclasses import dataclass

from tinygrad.llm.route_registry import RouteCandidateRegistry
from tinygrad.llm.route_request import RouteBinding, RouteContext, RouteRequest, RouteSelection


def _ctx():
  return RouteContext(linear="linear", x="x", fallback=lambda x: x, arch_ok=True, getenv_fn=lambda _name, default=None: default)


@dataclass
class _FakeCandidate:
  route_id: str
  can_use: bool = True
  bind_ok: bool = True

  def available(self, request, ctx):
    assert request.request_id
    assert ctx.arch_ok
    return RouteSelection(status="available", candidate=self) if self.can_use else RouteSelection(status="unavailable", candidate=self, reason="gate closed")

  def bind(self, request, _ctx):
    if not self.bind_ok:
      return RouteSelection(status="unavailable", candidate=self, reason="shape rejected")
    return RouteBinding(request=request, candidate=self, concrete_spec={"route": self.route_id}, output_shape=(1, 2), resources={"lds": 0})


def test_registry_preserves_registration_order_by_default():
  first = _FakeCandidate("first")
  second = _FakeCandidate("second")
  registry = RouteCandidateRegistry()
  registry.register(first)
  registry.register(second)

  assert registry.candidates() == (first, second)


def test_registry_selects_first_available_binding():
  request = RouteRequest(op="linear", request_id="req-1", attrs={"kind": "decode"})
  rejected = _FakeCandidate("closed", can_use=False)
  selected = _FakeCandidate("open")
  registry = RouteCandidateRegistry([rejected, selected])

  selection = registry.select(request, _ctx())

  assert selection.selected
  assert selection.candidate is selected
  assert selection.binding is not None
  assert selection.binding.request is request
  assert selection.binding.concrete_spec == {"route": "open"}
  assert selection.binding.output_shape == (1, 2)
  assert selection.binding.resources == {"lds": 0}


def test_registry_honors_preferred_candidate_order():
  first = _FakeCandidate("first")
  second = _FakeCandidate("second")
  registry = RouteCandidateRegistry([first, second])

  selection = registry.select(RouteRequest(op="linear", request_id="req-2"), _ctx(), preferred=("second",))

  assert selection.selected
  assert selection.candidate is second


def test_registry_reports_rejection_reasons_when_none_bind():
  registry = RouteCandidateRegistry([_FakeCandidate("closed", can_use=False), _FakeCandidate("bad-shape", bind_ok=False)])

  selection = registry.select(RouteRequest(op="linear", request_id="req-3"), _ctx())

  assert selection.status == "unavailable"
  assert "closed: gate closed" in selection.reason
  assert "bad-shape: shape rejected" in selection.reason
