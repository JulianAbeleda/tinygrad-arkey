"""Host-side (no GPU) proof of the submit-ahead (launch-hiding) eligibility
gate for Transformer.generate steady decode.

The reorder itself (submit token N+1's graph before item() on token N) is a
GPU-qualified measurement on the native NV route; this test pins only the
closed-default admission contract: the route must never engage unless every
promotion flag is set AND the greedy pingpong flash pair is captured with an
admitted alias contract.  Any cold/unadmitted state falls back to the ordinary
single-sync pingpong route.
"""
from types import SimpleNamespace

from tinygrad.llm.model import Transformer
import tinygrad.llm.model as model_mod
import tinygrad.llm.feedback_pingpong as fp_mod


def _pair(captured: bool) -> tuple:
  jit = SimpleNamespace(captured=object() if captured else None)
  return (jit, jit)


def _model(*, submit_ahead: bool = False, direct_greedy: bool = True,
           pingpong: bool = True, captured: bool = True, admitted: bool = True):
  return SimpleNamespace(
    _decode_submit_ahead_promoted=submit_ahead,
    _decode_direct_greedy_promoted=direct_greedy,
    _decode_feedback_pingpong_promoted=pingpong,
    rollout_greedy_pingpong_jits_flash=_pair(captured),
  )


def test_submit_ahead_closed_default(monkeypatch):
  """The flag is unset by default, so generate never engages the reorder."""
  model = _model(submit_ahead=False)
  monkeypatch.setattr(fp_mod, "pingpong_capture_contract",
                      lambda pair: {"admitted": True})
  assert Transformer._decode_submit_ahead_eligible(model) is False


def test_submit_ahead_requires_all_promotions(monkeypatch):
  monkeypatch.setattr(fp_mod, "pingpong_capture_contract",
                      lambda pair: {"admitted": True})
  assert Transformer._decode_submit_ahead_eligible(_model(direct_greedy=False)) is False
  assert Transformer._decode_submit_ahead_eligible(_model(pingpong=False)) is False


def test_submit_ahead_requires_warmed_admitted_pair(monkeypatch):
  captured_reason: list[str] = []

  def contract(pair):
    captured_reason.append("called")
    return {"admitted": False}

  monkeypatch.setattr(fp_mod, "pingpong_capture_contract", contract)
  # Cold pair (never captured) -> fall back without even consulting the contract.
  assert Transformer._decode_submit_ahead_eligible(_model(captured=False)) is False
  assert captured_reason == []
  # Captured but contract not admitted -> fall back (contract still consulted).
  assert Transformer._decode_submit_ahead_eligible(_model(submit_ahead=True, captured=True)) is False
  assert captured_reason == ["called"]


def test_submit_ahead_engages_only_when_fully_qualified(monkeypatch):
  monkeypatch.setattr(fp_mod, "pingpong_capture_contract",
                      lambda pair: {"admitted": True})
  model = _model(submit_ahead=True)
  assert Transformer._decode_submit_ahead_eligible(model) is True


def test_module_level_import_alias_is_consistent():
  # The gate imports the contract from the same module object this test patches,
  # so monkeypatching fp_mod.pingpong_capture_contract reaches the gate.
  import inspect
  src = inspect.getsource(Transformer._decode_submit_ahead_eligible)
  assert "from tinygrad.llm.feedback_pingpong import pingpong_capture_contract" in src
