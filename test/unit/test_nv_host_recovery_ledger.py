import pytest

from extra.llm_research.decode.nv_host_recovery_ledger import (
  BANKED_AT_COMPOSED_TOTAL_US, CACHE_LOOKUP_US, COMPOSED_BASELINE_MS,
  DEFENSIVE_COPY_REBIND_US, HOST_ITEMS, HOST_PROMOTION_GATE_US, INPUT_DISCOVERY_US,
  ITEMS_BY_ID, JIT_SIGNATURE_RECONSTRUCTION_US, LLAMA_OUTSIDE_US,
  LedgerClaimError, NATIVE_OUTSIDE_US, OUTSIDE_DELTA_US, P1_BOOKED_US,
  P5_BOOKED_US, PARTITION_SUM_US, PRE_FIRST_GRAPH_TOTAL_US, PRE_FIRST_SUB_SUM_US,
  PRE_TINYJIT_US, PYTHON_YIELD_TAIL_US, REDUNDANT_SYNC_US, SCALAR_COPYOUT_ITEM_US,
  STRUCTURAL_REWRITE_US, VARIABLE_METADATA_US, evaluate, marginal_tok_s,
  max_claimable, parse_assume_recovered, tok_s,
)


def test_baseline_split_and_partition_arithmetic():
  # 239.805 delta = 321.784 native outside - 81.979 llama outside.
  assert NATIVE_OUTSIDE_US - LLAMA_OUTSIDE_US == 239.805
  assert OUTSIDE_DELTA_US == pytest.approx(239.804933, abs=1e-9)  # audit exact
  assert abs(OUTSIDE_DELTA_US - (NATIVE_OUTSIDE_US - LLAMA_OUTSIDE_US)) == pytest.approx(0.000067, abs=1e-9)
  # Record-stated disjoint sum and its closure against the canonical native term.
  assert PARTITION_SUM_US == 338.458
  assert PARTITION_SUM_US - NATIVE_OUTSIDE_US == pytest.approx(16.674, abs=1e-9)
  row_sum = PRE_FIRST_GRAPH_TOTAL_US + SCALAR_COPYOUT_ITEM_US + PYTHON_YIELD_TAIL_US + REDUNDANT_SYNC_US
  assert row_sum == pytest.approx(338.258, abs=1e-9)  # record rows; stated sum carries 0.2 rounding


def test_sub_item_sums():
  # KEY FACTS pre-first sub-components.
  assert DEFENSIVE_COPY_REBIND_US + JIT_SIGNATURE_RECONSTRUCTION_US == pytest.approx(199.307, abs=1e-9)
  assert PRE_FIRST_SUB_SUM_US == pytest.approx(236.056, abs=1e-9)
  # Breakdown record sub-components of _prepare_jit_inputs and of the copy path.
  assert STRUCTURAL_REWRITE_US + INPUT_DISCOVERY_US + VARIABLE_METADATA_US == pytest.approx(71.075, abs=1e-9)
  assert 16.892 + 99.088 == pytest.approx(115.980, abs=1e-9)


def test_banked_arithmetic_at_composed_baseline():
  assert P1_BOOKED_US == pytest.approx(66.6620940, abs=1e-9)
  assert P5_BOOKED_US == pytest.approx(91.6365625, abs=1e-9)
  assert BANKED_AT_COMPOSED_TOTAL_US == pytest.approx(158.2986565, abs=1e-9)
  assert OUTSIDE_DELTA_US - BANKED_AT_COMPOSED_TOTAL_US == pytest.approx(81.5062765, abs=1e-6)
  # Caps exclude the already-banked attribution.
  assert max_claimable(ITEMS_BY_ID["jit_signature_reconstruction"]) == pytest.approx(27.924, abs=1e-6)
  assert max_claimable(ITEMS_BY_ID["defensive_copy_rebind"]) == pytest.approx(1.3714375, abs=1e-6)
  assert max_claimable(ITEMS_BY_ID["predispatch_oracle_ab"]) == 0.0
  assert max_claimable(ITEMS_BY_ID["redundant_sync"]) == 0.0


def test_tok_s_conversion():
  assert tok_s(COMPOSED_BASELINE_MS) == pytest.approx(187.82, abs=0.01)
  assert marginal_tok_s(OUTSIDE_DELTA_US) == pytest.approx(196.68, abs=0.01)
  for partition_us, expected in {
    69.166: 190.29, 77.927: 190.61, 121.380: 192.20, 83.247: 190.80,
    35.637: 189.09, 3.096: 187.93, 4.358: 187.97, 1.112: 187.86,
  }.items():
    assert marginal_tok_s(partition_us) == pytest.approx(expected, abs=0.01)


def test_no_go_fail_closed():
  with pytest.raises(LedgerClaimError, match="NO_GO"): evaluate({"packed_argmax": 0.0})
  with pytest.raises(LedgerClaimError, match="NO_GO"): evaluate({"p2b_owned_invocation_input": 1.0})
  with pytest.raises(LedgerClaimError, match="NO_GO"): parse_assume_recovered("packed_argmax")
  with pytest.raises(LedgerClaimError, match="unknown"): evaluate({"not_a_thing": 1.0})


def test_banked_and_over_cap_claims_fail_closed():
  with pytest.raises(LedgerClaimError, match="banked"): evaluate({"predispatch_oracle_ab": 69.166})
  with pytest.raises(LedgerClaimError, match="absent"): evaluate({"redundant_sync": 4.358})
  # Claiming the full pre-P1 partition double-counts the banked P1 portion.
  with pytest.raises(LedgerClaimError, match="exceeds max"): evaluate({"jit_signature_reconstruction": 77.927})
  with pytest.raises(LedgerClaimError, match="exceeds max"): evaluate({"scalar_copyout_item": 100.0})
  with pytest.raises(LedgerClaimError, match="negative"): evaluate({"pre_tinyjit": -1.0})
  # "redundant_sync" alone has no remaining credit.
  with pytest.raises(LedgerClaimError, match="no claimable"): parse_assume_recovered("redundant_sync")


def test_additive_semantics_and_marginal_frame():
  none = evaluate({})
  assert none["recovered_total_us"] == 0.0
  assert none["remaining_host_delta_us"] == pytest.approx(OUTSIDE_DELTA_US)
  assert none["resulting_tok_s"] == pytest.approx(187.82, abs=0.01)
  assert none["remaining_host_delta_at_composed_baseline_us"] == pytest.approx(81.506, abs=0.001)
  assert none["warnings"] == []

  item = evaluate({"scalar_copyout_item": 83.247})
  assert item["recovered_total_us"] == pytest.approx(83.247)
  assert item["remaining_host_delta_us"] == pytest.approx(OUTSIDE_DELTA_US - 83.247)
  assert item["resulting_tok_s"] == pytest.approx(190.80, abs=0.01)
  assert item["tok_s_delta_vs_composed"] == pytest.approx(2.98, abs=0.01)

  two = evaluate({"jit_signature_reconstruction": 27.924, "pre_tinyjit": 35.637})
  assert two["recovered_total_us"] == pytest.approx(63.561, abs=1e-3)
  assert two["remaining_host_delta_us"] == pytest.approx(OUTSIDE_DELTA_US - 63.561, abs=1e-3)
  reversed_claims = evaluate({"pre_tinyjit": 35.637, "jit_signature_reconstruction": 27.924})
  assert two["recovered_total_us"] == reversed_claims["recovered_total_us"]
  assert two["remaining_host_delta_us"] == reversed_claims["remaining_host_delta_us"]
  assert two["resulting_tok_s"] == reversed_claims["resulting_tok_s"]


def test_partition_rows_are_not_additive_recoveries():
  # Summing every remaining cap plus the already-banked P1+P5 over-reaches the
  # outside delta; the tool warns instead of blessing the sum as a booking.
  all_claims = {item["id"]: max_claimable(item) for item in HOST_ITEMS if max_claimable(item) > 0}
  payload = evaluate(all_claims)
  assert any("not additive" in warning for warning in payload["warnings"])
  assert payload["remaining_host_delta_at_composed_baseline_us"] < 0
  # A single bucket claim matches the parent scope marginal table.
  assert payload["items"][0]["marginal_tok_s_if_only_this_partition_recovered"] == pytest.approx(190.29, abs=0.01)


def test_per_item_promotion_gate_break_even():
  assert HOST_PROMOTION_GATE_US == 25.0
  above = {i["id"] for i in evaluate({})["items"] if i["promotion_gate"] == "ABOVE_HOST_PROMOTION_GATE"}
  assert above == {"predispatch_oracle_ab", "jit_signature_reconstruction", "defensive_copy_rebind",
                   "pre_tinyjit", "scalar_copyout_item"}
  below = {i["id"] for i in evaluate({})["items"] if i["promotion_gate"] == "BELOW_GATE_NOISE"}
  assert below == {"cache_lookup", "python_yield_tail", "redundant_sync"}


def test_parse_assume_recovered_cli():
  assert parse_assume_recovered(None) == {}
  assert parse_assume_recovered("scalar_copyout_item") == {"scalar_copyout_item": 83.247}
  assert parse_assume_recovered("jit_signature_reconstruction=20.0, pre_tinyjit=10") == {
    "jit_signature_reconstruction": 20.0, "pre_tinyjit": 10.0}
  with pytest.raises(LedgerClaimError, match="invalid amount"): parse_assume_recovered("pre_tinyjit=abc")
  with pytest.raises(LedgerClaimError, match="unknown"): parse_assume_recovered("bogus")
