from __future__ import annotations

import pytest

from tinygrad import UOp
from tinygrad.llm import prefill_routes
from tinygrad.llm.prefill_route_observer import PrefillDirectPackedBinding
from tinygrad.llm.prefill_routes import direct_packed_prefill_policy, prefill_route_mode, validate_prefill_route_mode
from tinygrad.llm.route_policy import decode_route_mode, should_use_flash_decode
from tinygrad.llm.route_selection import RouteLifecycle


def _env(values): return lambda name, default: values.get(name, default)


def test_prefill_canonical_modes_and_legacy_boolean_are_normalized():
  assert prefill_route_mode(_env({"TINYGRAD_PREFILL_ROUTE": "fp16"})) == "fp16"
  assert prefill_route_mode(_env({"TINYGRAD_PREFILL_PACKED_WMMA": "1"})) == "auto"
  assert prefill_route_mode(_env({"TINYGRAD_PREFILL_PACKED_WMMA": "0"})) == "direct_packed"


def test_prefill_direct_packed_quarantine_is_structural_not_model_named():
  assert direct_packed_prefill_policy(32, 8).lifecycle is RouteLifecycle.PROMOTED
  assert direct_packed_prefill_policy(40, 8).lifecycle is RouteLifecycle.QUARANTINED


def test_prefill_forced_quarantined_candidate_fails_loudly(monkeypatch):
  monkeypatch.setenv("TINYGRAD_PREFILL_ROUTE", "direct_packed")
  with pytest.raises(RuntimeError, match="quarantined"): validate_prefill_route_mode(40, 8)


def test_quarantine_blocks_direct_implementation_not_shared_packed_spec(monkeypatch):
  lin = type("Linear", (), {})()
  policy = direct_packed_prefill_policy(40, 8)
  lin._prefill_direct_packed_binding = PrefillDirectPackedBinding(
    "invocation", "prefill", "attn_qo", (512, 5120, 5120), policy.lifecycle, policy.reason)
  monkeypatch.setattr(prefill_routes, "_attached_direct_packed_spec", lambda *_: pytest.fail("quarantined direct route built a spec"))
  assert prefill_routes.route_direct_packed_prefill(lin, object()) is None


def test_decode_uses_same_auto_forced_candidate_fallback_contract():
  start = UOp.variable("start", 0, 4096).bind(512)
  assert decode_route_mode(_env({"TINYGRAD_DECODE_ROUTE": "fp16"})) == "fp16"
  assert not should_use_flash_decode(start, 1, getenv_fn=_env({"TINYGRAD_DECODE_ROUTE": "fp16"}))
  assert should_use_flash_decode(start, 1, getenv_fn=_env({"TINYGRAD_DECODE_ROUTE": "flash"}))
  assert should_use_flash_decode(start, 1, getenv_fn=_env({"TINYGRAD_DECODE_ROUTE": "auto"}))


def test_invalid_route_modes_fail_loudly():
  with pytest.raises(ValueError): prefill_route_mode(_env({"TINYGRAD_PREFILL_ROUTE": "mystery"}))
  with pytest.raises(ValueError): decode_route_mode(_env({"TINYGRAD_DECODE_ROUTE": "mystery"}))
