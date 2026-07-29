from __future__ import annotations

import pytest

from tinygrad import UOp
from tinygrad.llm import prefill_routes
from tinygrad.llm import decode_routes
from tinygrad.llm.prefill_routes import direct_packed_prefill_policy, prefill_route_mode, validate_prefill_route_mode
from tinygrad.llm.route_policy import decode_route_mode, should_use_flash_decode
from tinygrad.llm.route_selection import RouteLifecycle


def _env(values): return lambda name, default: values.get(name, default)


def test_prefill_canonical_modes_and_legacy_boolean_are_normalized():
  assert prefill_route_mode(_env({"TINYGRAD_PREFILL_ROUTE": "fp16"})) == "fp16"
  assert prefill_route_mode(_env({"TINYGRAD_PREFILL_PACKED_WMMA": "1"})) == "auto"
  assert prefill_route_mode(_env({"TINYGRAD_PREFILL_PACKED_WMMA": "0"})) == "fp16"
  assert prefill_route_mode(_env({"TINYGRAD_PREFILL_ROUTE": "direct_packed"})) == "fp16"


def test_prefill_direct_packed_compatibility_policy_is_generic_fallback():
  for heads in ((32, 8), (40, 8)):
    policy = direct_packed_prefill_policy(*heads)
    assert policy.candidate_id == "generic-tinygrad-prefill"
    assert policy.lifecycle is RouteLifecycle.FALLBACK


def test_prefill_forced_legacy_direct_mode_is_generic(monkeypatch):
  monkeypatch.setenv("TINYGRAD_PREFILL_ROUTE", "direct_packed")
  validate_prefill_route_mode(40, 8)
  assert prefill_route_mode() == "fp16"


def test_decode_uses_same_auto_forced_candidate_fallback_contract():
  start = UOp.variable("start", 0, 4096).bind(512)
  assert decode_route_mode(_env({"TINYGRAD_DECODE_ROUTE": "fp16"})) == "fp16"
  assert not should_use_flash_decode(start, 1, getenv_fn=_env({"TINYGRAD_DECODE_ROUTE": "fp16"}))
  assert should_use_flash_decode(start, 1, getenv_fn=_env({"TINYGRAD_DECODE_ROUTE": "flash"}))
  assert should_use_flash_decode(start, 1, getenv_fn=_env({"TINYGRAD_DECODE_ROUTE": "auto"}))


def test_decode_candidates_own_separate_g4_and_g5_split_geometry():
  assert decode_routes.FLASH_DECODE_CANDIDATE.query_heads == 32
  assert decode_routes.FLASH_DECODE_CANDIDATE.split_size == 48
  assert decode_routes.FLASH_DECODE_CANDIDATE.stage_width == 1
  assert decode_routes.FLASH_DECODE_G5_CANDIDATE.query_heads == 40
  assert decode_routes.FLASH_DECODE_G5_CANDIDATE.split_size == 32
  assert decode_routes.FLASH_DECODE_G5_CANDIDATE.query_group_size == 2
  assert decode_routes.FLASH_DECODE_G5_CANDIDATE.stage_width == 4


def test_decode_flash_selector_only_binds_promoted_g4_g5_shapes():
  """Characterize the production selector itself, rather than a parallel route table."""
  g4 = decode_routes.FLASH_DECODE_CANDIDATE.bind(1, 32, 8, 128, "AMD:0")
  g5 = decode_routes.FLASH_DECODE_G5_CANDIDATE.bind(1, 40, 8, 128, "AMD:0")
  assert g4 is not None and g4.route_id == "decode_flash_live_split_g4_kvboth"
  assert g5 is not None and g5.route_id == "decode_flash_live_split_g5_kvboth"
  assert decode_routes.FLASH_DECODE_CANDIDATE.bind(1, 40, 8, 128, "AMD:0") is None
  assert decode_routes.FLASH_DECODE_G5_CANDIDATE.bind(1, 32, 8, 128, "AMD:0") is None
  assert decode_routes.FLASH_DECODE_CANDIDATE.bind(1, 32, 8, 128, "CPU") is None


def test_invalid_route_modes_fail_loudly():
  with pytest.raises(ValueError): prefill_route_mode(_env({"TINYGRAD_PREFILL_ROUTE": "mystery"}))
  with pytest.raises(ValueError): decode_route_mode(_env({"TINYGRAD_DECODE_ROUTE": "mystery"}))
