from types import SimpleNamespace

from extra.qk.prefill_whole_synced import shared_attention_attribution


def test_shared_attention_attribution_reports_request_not_fusion():
  model = SimpleNamespace(config=SimpleNamespace(prefill_tc_attn=True, prefill_v2=True))
  got = shared_attention_attribution(model)
  assert got == {
    "schema": "shared-prefill-attention-route.v2",
    "requested": True,
    "boundary": "shared_prefill_attention",
    "semantic_candidate": "bounded_online_primitive",
    "selected_lowering": "ordinary_sdpa",
    "fallback_contract": "ordinary_sdpa",
    "fusion_proven": False,
    "dual_wmma_proven": False,
    "performance_proven": False,
    "promotion_eligible": False,
    "blocker": "generic tiled fused attention lowering is not implemented",
  }


def test_shared_attention_attribution_is_fail_closed_when_not_admitted():
  model = SimpleNamespace(config=SimpleNamespace(prefill_tc_attn=False, prefill_v2=True))
  got = shared_attention_attribution(model)
  assert not got["requested"]
  assert got["boundary"] == "scaled_dot_product_attention"
  assert got["semantic_candidate"] is None
  assert got["selected_lowering"] == "ordinary_sdpa"
  assert not got["promotion_eligible"]
