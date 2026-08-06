"""Hermetic CPU tests for the NV norms fusion exact-output A/B harness."""
import pytest

from extra.llm_research.decode.nv_norms_fusion_ab import (
  POP_NORMS, ConstructionGapError, _configure, boundary_free_gate, candidate_topology_probe,
  no_go_record, validate_census, validate_logits_gate, validate_timing_bracket,
)


def test_boundary_free_gate_returns_construction_gap():
  gate = boundary_free_gate()
  assert gate["verdict"] == "CONSTRUCTION_GAP"
  assert not any(gate["conditions"].values())
  phase0 = gate["phase0_baseline"]
  assert phase0["verdict"] == "CONSTRUCTION_GAP"
  for row in phase0["baseline"].values():
    assert row["program_count"] == 2
    assert row["contains_custom_kernel"] is False
    assert row["contains_contiguous"] is False
  probe = gate["candidate_probe"]
  assert probe["consumer_is_ordinary"] is False
  assert probe["reduce_absorbable"] is False
  assert probe["candidate_removes_programs_in_graph"] is False


def test_candidate_probe_fused_epilogue_is_bitwise_exact_and_ordinary():
  probe = candidate_topology_probe()
  assert probe["fused_epilogue_bitwise_equal"] is True
  assert probe["fused_epilogue_max_abs"] == 0.0
  assert probe["fused_epilogue_contains_custom"] is False
  assert probe["fused_epilogue_contains_contiguous"] is False
  # The reduce half stays a separate program; the affine epilogue alone is one
  # ordinary program when the fp32 reduce scalar is shared as an input.
  assert probe["fused_pair_program_count"] == 2
  assert probe["affine_epilogue_program_count"] == 1
  assert probe["reduce_scalar_dtype"] == "float32"


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


def _census(norms, roles, pops):
  return {"norms_kernels": norms, "norms_roles": roles, "population_counts": pops}


def test_validate_census_confinement():
  control = _census(362, {"rmsnorm_reduce": 73, "rmsnorm_epilogue": 72},
                    {POP_NORMS: 362, "quant_core": 217, "flash": 72})
  good = _census(147, {"rmsnorm_reduce": 73},
                 {POP_NORMS: 147, "quant_core": 217, "flash": 72})
  result = validate_census(control, good)
  assert result["gate_pass"] is True
  assert result["norms_epilogues_removed"] == 215
  assert result["norms_reduce_unchanged"] is True
  assert result["confined_to_norms"] is True
  # A quant-core count change breaks confinement.
  leaky = _census(147, {"rmsnorm_reduce": 73}, {POP_NORMS: 147, "quant_core": 218, "flash": 72})
  assert validate_census(control, leaky)["gate_pass"] is False
  # A reduce-role change breaks the contract even when confined.
  reduce_changed = _census(148, {"rmsnorm_reduce": 74}, {POP_NORMS: 148, "quant_core": 217, "flash": 72})
  assert validate_census(control, reduce_changed)["gate_pass"] is False
  # No epilogue removal (identical norms census) does not promote.
  unchanged = _census(362, {"rmsnorm_reduce": 73, "rmsnorm_epilogue": 72},
                      {POP_NORMS: 362, "quant_core": 217, "flash": 72})
  assert validate_census(control, unchanged)["gate_pass"] is False


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
  assert record["construction"]["population"] == "norms"
  assert record["target"]["depth"] == 512


def test_no_go_record_ledger_fallback():
  record = no_go_record(boundary_free_gate())
  census = record["census"]
  assert census["run"] is False
  assert census["reference"]["node_count"] == 362
  assert census["reference"]["anchor_child_epilogue_count"] == 215
