"""Hermetic CPU tests for the NV residual fusion exact-output A/B harness."""
import pytest

from extra.llm_research.decode.nv_residual_fusion_ab import (
  POP_RESIDUAL, ConstructionGapError, _configure, boundary_free_gate, candidate_topology_probe,
  no_go_record, validate_census, validate_cost_prediction, validate_logits_gate, validate_timing_bracket,
)


def test_boundary_free_gate_returns_construction_gap():
  gate = boundary_free_gate()
  assert gate["verdict"] == "CONSTRUCTION_GAP"
  assert not any(gate["conditions"].values())
  phase0 = gate["phase0_baseline"]
  assert phase0["verdict"] == "OPAQUE_PRODUCER_GAP"
  for row in phase0["baseline"].values():
    assert row["program_count"] == 1
    assert row["contains_custom_kernel"] is False
    assert row["contains_contiguous"] is False
  probe = gate["candidate_probe"]
  assert probe["consumer_is_ordinary"] is False
  assert probe["residual_absorbable"] is False
  assert probe["candidate_removes_programs_in_graph"] is False


def test_candidate_probe_residual_add_ordinary_but_gemv_boundary_opaque():
  probe = candidate_topology_probe()
  # The residual add is one ordinary elementwise program for realized and lazy
  # inputs, with no CUSTOM and no CONTIGUOUS.
  assert probe["residual_add_program_count"] == 1
  assert probe["residual_add_lazy_program_count"] == 1
  assert probe["residual_add_contains_custom"] is False
  assert probe["residual_add_contains_contiguous"] is False
  # The real producer is the opaque q4k GEMV custom program: the fused graph
  # keeps the GEMV CALL plus the separate ordinary add (two programs).
  assert probe["gemv_plus_residual_program_count"] == 2
  assert probe["gemv_plus_residual_programs"] == ["q4k_g3_lanemap_gemv_4096_4096", "test"]
  assert probe["gemv_plus_residual_contains_call"] is True
  assert probe["consumer_is_ordinary"] is False
  assert probe["residual_absorbable"] is False
  assert probe["candidate_removes_programs_in_graph"] is False


def test_candidate_arm_raises_construction_gap_error():
  with pytest.raises(ConstructionGapError):
    _configure("candidate")
  _configure("control")  # must be a no-op
  with pytest.raises(ValueError):
    _configure("not-an-arm")


def _logits(tokens, digest, shape=(2, 1, 151936)):
  return {"tokens": tokens, "logits_sha256": digest, "shape": list(shape)}


def test_validate_logits_gate_exact_output():
  control = _logits([1, 2], "a" * 64)
  assert validate_logits_gate(control, _logits([1, 2], "a" * 64))["gate_pass"] is True
  bad_sha = validate_logits_gate(control, _logits([1, 2], "b" * 64))
  assert bad_sha["gate_pass"] is False and bad_sha["logits_sha256_equal"] is False
  bad_tokens = validate_logits_gate(control, _logits([1, 3], "a" * 64))
  assert bad_tokens["gate_pass"] is False and bad_tokens["tokens_equal"] is False
  bad_shape = validate_logits_gate(control, _logits([1, 2], "a" * 64, shape=(2, 1, 128)))
  assert bad_shape["gate_pass"] is False and bad_shape["shape_equal"] is False


def _census(residual, roles, pops):
  return {"residual_kernels": residual, "residual_roles": roles, "population_counts": pops}


def test_validate_census_confinement():
  control = _census(145, {"ffn_residual_add": 36, "attention_cast": 36},
                    {POP_RESIDUAL: 145, "quant_core": 217, "flash": 72})
  good = _census(0, {}, {POP_RESIDUAL: 0, "quant_core": 217, "flash": 72})
  result = validate_census(control, good)
  assert result["gate_pass"] is True
  assert result["residual_epilogues_removed"] == 145
  assert result["residual_reduce_unchanged"] is True
  assert result["confined_to_residual"] is True
  # A quant-core count change breaks confinement.
  leaky = _census(0, {}, {POP_RESIDUAL: 0, "quant_core": 218, "flash": 72})
  assert validate_census(control, leaky)["gate_pass"] is False
  # No epilogue removal (identical residual census) does not promote.
  unchanged = _census(145, {"ffn_residual_add": 36, "attention_cast": 36},
                      {POP_RESIDUAL: 145, "quant_core": 217, "flash": 72})
  assert validate_census(control, unchanged)["gate_pass"] is False
  # A reduce-role appearing in the residual roles breaks the contract even when
  # the count is confined (defensive: the ledger residual row has zero reduces).
  reduce_changed = _census(145, {"rmsnorm_reduce": 1, "ffn_residual_add": 35},
                           {POP_RESIDUAL: 145, "quant_core": 217, "flash": 72})
  assert validate_census(control, reduce_changed)["gate_pass"] is False


def _timing(median, stream="s" * 64, token="t" * 64):
  return {"median_ms_per_token": median, "token_stream_hash": stream, "token_hashes": [token]}


def test_validate_timing_bracket_promotion_requires_both_controls():
  # control 5.324 ms vs candidate 5.274 ms: exactly +50 us on arm A, but only
  # +46 us vs arm B, so the bracket is not promoted.
  below = validate_timing_bracket([_timing(5.324), _timing(5.274), _timing(5.320)])
  assert below["all_token_hashes_equal"] is True
  assert below["candidate_minus_control_a_us"] == pytest.approx(50.0)
  assert below["candidate_minus_control_b_us"] == pytest.approx(46.0)
  assert below["promoted"] is False
  # candidate 5.270 ms clears +50 us vs both arms.
  above = validate_timing_bracket([_timing(5.324), _timing(5.270), _timing(5.320)])
  assert above["candidate_minus_control_a_us"] == pytest.approx(54.0)
  assert above["candidate_minus_control_b_us"] == pytest.approx(50.0)
  assert above["promoted"] is True
  # A divergent token stream voids the bracket even with a large win.
  divergent = validate_timing_bracket([_timing(5.324), _timing(5.260, stream="x" * 64), _timing(5.320)])
  assert divergent["all_token_hashes_equal"] is False
  assert divergent["promoted"] is False
  with pytest.raises(ValueError):
    validate_timing_bracket([_timing(5.0), _timing(4.9)])


def _residual_histogram(median=1.5):
  # attention_cast classifies to POP_RESIDUAL.
  return [(f"E_32_32_4_0a5eb0ac56c097a0{'0' * 48}", 36, median)]


def test_validate_cost_prediction_confirms_launch_shaped_win():
  # R=1, M=1.5us, 36 blocks -> point prediction -54us (candidate faster).  The
  # bracket field is (control - candidate), so +54us is the matching win.
  result = validate_cost_prediction({"candidate_minus_control_bracket_us": 54.0},
                                    {"histogram": _residual_histogram()}, {})
  assert result["run"] is True
  assert result["result"] == "PASS"
  assert result["reconciliation"]["result"] == "CONFIRMED"


def test_validate_cost_prediction_contradiction_fails_closed():
  # A candidate that is slower (+10us measured) contradicts the -54us point
  # prediction on the opposite side of zero and must fail the campaign.
  result = validate_cost_prediction({"candidate_minus_control_bracket_us": -10.0},
                                    {"histogram": _residual_histogram()}, {})
  assert result["result"] == "FAIL"
  assert result["reconciliation"]["result"] == "CONTRADICTED"


def test_validate_cost_prediction_no_go_without_family():
  result = validate_cost_prediction({"candidate_minus_control_bracket_us": 54.0}, {"histogram": []}, {})
  assert result["result"] == "NO-GO"


def test_no_go_record_shape_and_evidence():
  gate = boundary_free_gate()
  record = no_go_record(gate, {"run": False, "reference": {}}, model="/m", depth=512)
  assert record["verdict"] == "NO-GO"
  assert record["boundary_free_gate"]["verdict"] == "CONSTRUCTION_GAP"
  assert record["logits_gate"]["run"] is False
  assert record["logits_gate"]["result"] == "NOT_AUTHORIZED"
  assert record["wall_bracket"]["run"] is False
  assert record["census"]["run"] is False
  assert record["hard_stop_notes"]
  assert record["citations"]
  assert record["construction"]["population"] == "residual_cast_contiguous"
  assert record["target"]["depth"] == 512


def test_no_go_record_ledger_fallback():
  record = no_go_record(boundary_free_gate())
  census = record["census"]
  assert census["run"] is False
  assert census["reference"]["node_count"] == 145
  assert census["reference"]["anchor_child_epilogue_count"] == 145
  assert census["reference"]["boundary_free_eligible"] is True
