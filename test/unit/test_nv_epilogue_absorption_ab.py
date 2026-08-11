"""Hermetic CPU tests for the residual-family epilogue absorption AB harness."""
import pytest

from extra.llm_research.decode.nv_epilogue_absorption_ab import (
  CENSUS_SCHEMA, TIMING_SCHEMA, _assert_candidate_configured, _assert_control_closed,
  _configure, _gates, validate_census, validate_timing_bracket,
)


class _FakeBlock:
  pass


class _FakeModel:
  def __init__(self, blocks=3):
    self.blk = [_FakeBlock() for _ in range(blocks)]


def test_candidate_arm_installs_lease_on_model_and_every_block():
  from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
  from tinygrad.helpers import Context
  model = _FakeModel()
  with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
    _configure(model, "candidate")
  gates = _gates(model)
  assert gates["w1w3_fp16_store_lease"] is True
  assert gates["block_w1w3_fp16_store_lease"] == [True, True, True]
  _assert_candidate_configured(gates)


def test_control_arm_keeps_lease_closed():
  from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
  from tinygrad.helpers import Context
  model = _FakeModel()
  with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
    _configure(model, "control")
  gates = _gates(model)
  assert gates["w1w3_fp16_store_lease"] is False
  assert gates["block_w1w3_fp16_store_lease"] == [False, False, False]
  _assert_control_closed(gates)


def test_candidate_gate_fails_closed_on_missing_lease():
  _assert_candidate_configured({"w1w3_fp16_store_lease": True,
                                "block_w1w3_fp16_store_lease": [True, True, True]})
  with pytest.raises(RuntimeError, match=r"block\[1\]"):
    _assert_candidate_configured({"w1w3_fp16_store_lease": True,
                                  "block_w1w3_fp16_store_lease": [True, False, True]})
  with pytest.raises(RuntimeError, match="model._q4k_w1w3_fp16_store_lease"):
    _assert_candidate_configured({"w1w3_fp16_store_lease": False,
                                  "block_w1w3_fp16_store_lease": [True, True, True]})


def test_control_gate_fails_closed_if_lease_observed():
  _assert_control_closed({"w1w3_fp16_store_lease": False,
                          "block_w1w3_fp16_store_lease": [False, False, False]})
  model = _FakeModel()
  model._q4k_w1w3_fp16_store_lease = True
  with pytest.raises(RuntimeError, match="model._q4k_w1w3_fp16_store_lease"):
    _assert_control_closed(_gates(model))


def _control_census(cast=36, fused=36):
  return {"schema": CENSUS_SCHEMA, "kernels": 751, "ffn_activation_cast_count": cast,
          "w1w3_fused16_count": 0, "w1w3_fused_count": fused,
          "program_counts": {f"q4k_g3_lanemap_gemv_w1w3fused_12288_4096": fused,
                             f"E_128_32_3": cast, "r_16_256_r": 37,
                             "q4k_g3_lanemap_gemv_x": 217},
          "population_counts": {"norms": 74, "quant_core": 217 + fused,
                                "residual_cast_contiguous": cast, "flash": 20, "other": 6}}


def _candidate_census(cast=0, fused16=36, fused=0):
  return {"schema": CENSUS_SCHEMA, "kernels": 751 - 36, "ffn_activation_cast_count": cast,
          "w1w3_fused16_count": fused16, "w1w3_fused_count": fused,
          "program_counts": {f"q4k_g3_lanemap_gemv_w1w3fused16_12288_4096": fused16,
                             f"E_128_32_3": cast, "r_16_256_r": 37,
                             "q4k_g3_lanemap_gemv_x": 217},
          "population_counts": {"norms": 74, "quant_core": 217 + fused16,
                                "residual_cast_contiguous": cast, "flash": 20, "other": 6}}


def test_census_gate_passes_on_expected_absorption():
  result = validate_census(_control_census(), _candidate_census())
  assert result["gate_pass"] is True
  assert result["cast_control"] == 36 and result["cast_candidate"] == 0
  assert result["fused16_candidate"] == 36
  assert result["honest_net_program_delta"] == -36


def test_census_gate_fails_closed_when_cast_remains():
  result = validate_census(_control_census(), _candidate_census(cast=36, fused16=0, fused=36))
  assert result["gate_pass"] is False
  assert any("still renders" in reason for reason in result["fail_closed"])


def test_census_gate_fails_closed_on_unrelated_shift():
  candidate = _candidate_census()
  candidate["program_counts"]["E_other"] = 1
  candidate["kernels"] = 716
  result = validate_census(_control_census(), candidate)
  assert result["gate_pass"] is False
  assert any("unrelated" in reason for reason in result["fail_closed"])


def test_census_gate_fails_closed_when_fp32_fused_remains():
  result = validate_census(_control_census(), _candidate_census(fused16=36, fused=18))
  assert result["gate_pass"] is False
  assert any("fp32 w1w3" in reason for reason in result["fail_closed"])


def _timing_row(ms):
  return {"schema": TIMING_SCHEMA, "median_ms_per_token": ms,
          "token_hashes": ["abc"], "tokens_identical_within_arm": True}


def test_timing_bracket_promotes_when_faster_than_both_controls():
  result = validate_timing_bracket([_timing_row(5.33), _timing_row(5.26), _timing_row(5.32)], settled_continuous=False)
  assert result["promoted"] is True
  assert result["candidate_minus_control_bracket_us"] > 50


def test_timing_bracket_rejects_slow_candidate():
  result = validate_timing_bracket([_timing_row(5.26), _timing_row(5.32), _timing_row(5.27)], settled_continuous=False)
  assert result["promoted"] is False
