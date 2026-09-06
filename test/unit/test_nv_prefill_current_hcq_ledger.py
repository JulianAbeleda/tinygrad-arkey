import sys

sys.path.insert(0, "extra/llm_research/prefill")
from nv_prefill_current_hcq_ledger import CURRENT_QO_ID, _specialize_current, _lifecycle_kind


def test_current_streamk_physical_launches_keep_semantic_pairing():
  rows=[]
  for _ in range(36):
    rows += [
      {"name":"qo", "metadata":{"canonical_identity":CURRENT_QO_ID}},
      {"name":"qo", "metadata":{"canonical_identity":CURRENT_QO_ID}},
      {"name":"q4_qo_streamk"}, {"name":"q4_qo_streamk"},
      {"name":"q4k_imma_fixup_active"}, {"name":"q4k_imma_fixup_active"},
    ]
  _specialize_current(rows)
  assert [x["primary"] for x in rows[:6]] == ["q","o","gate","up","gate","up"]


def test_current_streamk_census_fails_closed_on_missing_fixup():
  rows=[]
  for _ in range(36):
    rows += [
      {"name":"qo", "metadata":{"canonical_identity":CURRENT_QO_ID}},
      {"name":"qo", "metadata":{"canonical_identity":CURRENT_QO_ID}},
      {"name":"q4_qo_streamk"}, {"name":"q4_qo_streamk"},
      {"name":"q4k_imma_fixup_active"}, {"name":"q4k_imma_fixup_active"},
    ]
  rows.pop()
  try: _specialize_current(rows)
  except ValueError as exc: assert "incomplete current dense role census" in str(exc)
  else: raise AssertionError("incomplete physical route must fail")


def test_current_q4_down_fixup_is_not_charged_to_gate():
  rows=[]
  for _ in range(36):
    rows += [
      {"name":"qo", "metadata":{"canonical_identity":CURRENT_QO_ID}},
      {"name":"qo", "metadata":{"canonical_identity":CURRENT_QO_ID}},
      {"name":"q4_qo_streamk"}, {"name":"q4_qo_streamk"},
      {"name":"q4k_imma_fixup_active"}, {"name":"q4k_imma_fixup_active"},
    ]
  for _ in range(18): rows += [{"name":"q4_down_streamk"},{"name":"q4k_imma_fixup_active"}]
  _specialize_current(rows)
  assert [x["primary"] for x in rows[-2:]] == ["down","down"]
  assert sum(x["primary"]=="gate" for x in rows)==72
  assert sum(x["primary"]=="up" for x in rows)==72


def test_generated_vocabulary_is_not_hidden_in_support():
  rows=[]
  for _ in range(36):
    rows += [{"name":"qo", "metadata":{"canonical_identity":CURRENT_QO_ID}} for _ in range(2)]
    rows += [{"name":"q4_qo_streamk"} for _ in range(2)]
    rows += [{"name":"q4k_imma_fixup_active"} for _ in range(2)]
  rows += [{"name":"q6k_v_four_warp_fp16_direct_151936_4096"},{"name":"unrecognized_program"}]
  _specialize_current(rows)
  assert rows[-2]["primary"] == "vocabulary"
  assert rows[-2]["role"] == "generated_vocabulary_main"
  assert rows[-1]["role"] == "support"


def test_lifecycle_boundary_does_not_charge_fixup_or_producer_as_main():
  assert _lifecycle_kind({"name":"q4k_imma_fixup_active","primary":"gate","role":"gate"}) == "projection_fixup"
  assert _lifecycle_kind({"name":"q8_streamk_record_fp16_q6_ffn_down","primary":"residual_rope_kv_support",
    "role":"support"}) == "projection_producer"
  assert _lifecycle_kind({"name":"q4_down_streamk","primary":"down","role":"q4_down_main"}) == "projection_main"
  assert _lifecycle_kind({"name":"unknown","primary":"residual_rope_kv_support","role":"support"}) == "unresolved_support"
