import sys

sys.path.insert(0, "extra/llm_research/prefill")
from nv_prefill_current_hcq_ledger import CURRENT_QO_ID, _specialize_current


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
