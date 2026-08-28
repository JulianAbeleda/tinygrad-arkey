from types import SimpleNamespace

from tinygrad.llm.model import prefill_v2_target_admitted


def test_prefill_v2_admits_qualified_nv_sm120_target():
  assert prefill_v2_target_admitted(SimpleNamespace(backend="NV", architecture="sm_120"))


def test_prefill_v2_keeps_other_targets_and_missing_facts_unchanged():
  assert prefill_v2_target_admitted(SimpleNamespace(backend="AMD", architecture="gfx1100"))
  assert prefill_v2_target_admitted(SimpleNamespace(backend="NV", architecture="sm_90"))
  assert prefill_v2_target_admitted(None)
