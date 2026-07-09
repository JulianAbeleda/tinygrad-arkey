from extra.qk.prefill.lds2_s9_layout_search import candidate_proposals
from extra.qk.prefill.wmma import default_lds2_reg_layout


def test_layout_search_candidates_include_baseline_and_positive_shifts():
  candidates = candidate_proposals(wm=2, wn=4, loads_a=2, loads_b=2)
  by_name = {c["name"]: c for c in candidates}
  baseline = default_lds2_reg_layout(2, 4, 2, 2)

  assert by_name["baseline"]["valid"] is True
  assert by_name["baseline"]["layout"] == baseline.__dict__
  assert by_name["block_shift_plus_1"]["valid"] is True
  assert by_name["block_shift_plus_1"]["layout"]["FA"] == baseline.FA + 1
  assert by_name["block_shift_plus_8"]["layout"]["SCR"] == baseline.SCR + 8


def test_layout_search_reports_invalid_candidates():
  candidates = candidate_proposals(wm=4, wn=4, loads_a=8, loads_b=8, plrab=1)

  assert any(c["valid"] is False and "invalid_reason" in c for c in candidates)
