from types import SimpleNamespace

from extra.qk.prefill_whole_synced import shared_attention_attribution


def test_shared_attention_attribution_reports_request_not_fusion():
  model = SimpleNamespace(config=SimpleNamespace(prefill_tc_attn=True, prefill_v2=True))
  got = shared_attention_attribution(model)
  assert got == {
    "schema": "shared-prefill-attention-route.v1",
    "requested": True,
    "boundary": "shared_prefill_attention",
    "selected_lowering": "bounded_online_primitive",
    "fallback_contract": "ordinary_sdpa",
    "fusion_proven": False,
  }


def test_shared_attention_attribution_is_fail_closed_when_not_admitted():
  model = SimpleNamespace(config=SimpleNamespace(prefill_tc_attn=False, prefill_v2=True))
  got = shared_attention_attribution(model)
  assert not got["requested"]
  assert got["boundary"] == "scaled_dot_product_attention"
  assert got["selected_lowering"] == "ordinary_sdpa"
