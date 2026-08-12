"""Hermetic CPU tests for the M2b+M2c+M2d epilogue-absorption AB harness.

M2d extends the BOOKED M2c candidate conditions.  The control arm is the M2c
candidate (callify flags + reduce-output promotion + the w1w3 fp16-store lease
+ the ffn_down residual-add lease) WITHOUT the flash-combine fp16-store lease;
the candidate arm additionally installs ``_flash_combine_fp16_lease`` on the
model and every block.  The census gate requires every ``E_32_32_4_02a9738c``
residual add to fold into a ``*_epi_ffnresadd`` GEMV body 1:1, every
``E_32_32_4_fab82d40`` block-output copy to fold, and every
``E_32_32_4_0a5eb0ac`` attention cast to fold into the fp16 combine store
(flash_fused_gmax_combine_* -> flash_fused_gmax_combine_f16_* 1:1) with no
other program-count shift and the M2a fused16/cast families byte-identical
between arms.
"""
import pytest

from extra.llm_research.decode.nv_epilogue_absorption_ab import (
  CENSUS_SCHEMA, TIMING_SCHEMA, _assert_candidate_configured, _assert_control_closed,
  _configure, _gates, validate_census, validate_timing_bracket,
)


class _FakeFFNDown:
  pass


class _FakeBlock:
  def __init__(self):
    self.ffn_down = _FakeFFNDown()


class _FakeModel:
  def __init__(self, blocks=3):
    self.blk = [_FakeBlock() for _ in range(blocks)]


def test_candidate_arm_installs_m2a_m2b_and_m2d_leases():
  from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
  from tinygrad.helpers import Context
  model = _FakeModel()
  with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
    _configure(model, "candidate")
  gates = _gates(model)
  # Both arms carry the booked M2c leases (M2a fp16-store + M2b residual-add);
  # the candidate adds the M2d flash-combine fp16-store lease.
  assert gates["w1w3_fp16_store_lease"] is True
  assert gates["block_w1w3_fp16_store_lease"] == [True, True, True]
  assert gates["ffn_down_resadd_lease"] is True
  assert gates["block_ffn_down_resadd_lease"] == [True, True, True]
  assert gates["ffn_down_linear_resadd_lease"] == [True, True, True]
  assert gates["flash_combine_fp16_lease"] is True
  assert gates["block_flash_combine_fp16_lease"] == [True, True, True]
  _assert_candidate_configured(gates)


def test_control_arm_keeps_m2c_leases_and_closes_m2d():
  from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
  from tinygrad.helpers import Context
  model = _FakeModel()
  with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
    _configure(model, "control")
  gates = _gates(model)
  assert gates["w1w3_fp16_store_lease"] is True
  assert gates["block_w1w3_fp16_store_lease"] == [True, True, True]
  assert gates["ffn_down_resadd_lease"] is True
  assert gates["block_ffn_down_resadd_lease"] == [True, True, True]
  assert gates["ffn_down_linear_resadd_lease"] == [True, True, True]
  assert gates["flash_combine_fp16_lease"] is False
  assert gates["block_flash_combine_fp16_lease"] == [False, False, False]
  _assert_control_closed(gates)


def _m2_gates(ffn_down_lease=True, block_lease=True, linear_lease=True, m2a=True,
              combine_lease=False, block_combine_lease=False):
  return {
    "w1w3_fp16_store_lease": m2a, "block_w1w3_fp16_store_lease": [m2a, m2a, m2a],
    "ffn_down_resadd_lease": ffn_down_lease,
    "block_ffn_down_resadd_lease": [block_lease, block_lease, block_lease],
    "ffn_down_linear_resadd_lease": [linear_lease, linear_lease, linear_lease],
    "flash_combine_fp16_lease": combine_lease,
    "block_flash_combine_fp16_lease": [block_combine_lease, block_combine_lease, block_combine_lease],
  }


def test_candidate_gate_fails_closed_on_missing_lease():
  _assert_candidate_configured(_m2_gates(combine_lease=True, block_combine_lease=True))
  with pytest.raises(RuntimeError, match=r"block\[1\]"):
    _assert_candidate_configured(_m2_gates(block_lease=False))
  with pytest.raises(RuntimeError, match="model._ffn_down_resadd_lease"):
    _assert_candidate_configured(_m2_gates(ffn_down_lease=False))
  with pytest.raises(RuntimeError, match=r"block\[1\]\.ffn_down"):
    _assert_candidate_configured(_m2_gates(linear_lease=False))
  with pytest.raises(RuntimeError, match="model._flash_combine_fp16_lease"):
    _assert_candidate_configured(_m2_gates(block_combine_lease=True))
  with pytest.raises(RuntimeError, match=r"block\[1\]\._flash_combine_fp16_lease"):
    _assert_candidate_configured(_m2_gates(combine_lease=True))


def test_control_gate_fails_closed_if_m2d_lease_observed():
  _assert_control_closed(_m2_gates())
  model = _FakeModel()
  model._flash_combine_fp16_lease = True
  with pytest.raises(RuntimeError, match="model._flash_combine_fp16_lease"):
    _assert_control_closed(_gates(model))
  model2 = _FakeModel()
  model2.blk[1]._flash_combine_fp16_lease = True
  with pytest.raises(RuntimeError, match=r"block\[1\]\._flash_combine_fp16_lease"):
    _assert_control_closed(_gates(model2))
  with pytest.raises(RuntimeError, match="model._ffn_down_resadd_lease"):
    _assert_control_closed(_m2_gates(ffn_down_lease=False, block_lease=False, linear_lease=False))


def _control_census(fused16=36, resadd=0, ffnresadd=36, plain_ffn_down=0, block_output_copies=0,
                    attention_casts=36, combine_f32=36, combine_f16=0, combine_copy=0):
  return {"schema": CENSUS_SCHEMA, "kernels": 630, "ffn_activation_cast_count": 0,
          "w1w3_fused16_count": fused16, "w1w3_fused_count": 0,
          "ffn_residual_add_count": resadd, "ffn_down_resadd_count": ffnresadd,
          "block_output_copy_count": block_output_copies, "attention_cast_count": attention_casts,
          "flash_combine_f16_count": combine_f16, "flash_combine_f32_count": combine_f32,
          "combine_copy_count": combine_copy,
          "program_counts": {f"q4k_g3_lanemap_gemv_w1w3fused16_12288_4096": fused16,
                             f"E_32_32_4_02a9738c": resadd, "r_16_256_r": 37,
                             "q4k_g3_lanemap_gemv_x": 217,
                             "q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288": ffnresadd // 2,
                             "q6k_gen_coop_4096_12288_inkernel_epi_ffnresadd": ffnresadd // 2,
                             "E_32_32_4_fab82d40x": block_output_copies,
                             "E_32_32_4_0a5eb0acx": attention_casts,
                             "flash_fused_gmax_combine_32_128": combine_f32,
                             "flash_fused_gmax_combine_f16_32_128": combine_f16,
                             "E_32_32_4_3b0fcfbcx": combine_copy},
          "population_counts": {"norms": 74, "quant_core": 217 + fused16,
                                "residual_cast_contiguous": resadd, "flash": 20, "other": 6}}


def _candidate_census(fused16=36, resadd=0, ffnresadd=36, fused=0, block_output_copies=0, attention_casts=0,
                      combine_f32=0, combine_f16=36, combine_copy=0):
  # M2d control is the booked M2c candidate (630 kernels: 0 adds, 0 copies, 36 casts,
  # 36 fp32 combines).  The candidate drops the 36 casts and swaps the combines 1:1, so
  # the fixture reconstructs the candidate kernel count from the control baseline.
  control_kernels = 630
  candidate_kernels = control_kernels - 36 + block_output_copies + attention_casts + (combine_f16 + combine_f32 - 36)
  return {"schema": CENSUS_SCHEMA, "kernels": candidate_kernels, "ffn_activation_cast_count": 0,
          "w1w3_fused16_count": fused16, "w1w3_fused_count": fused,
          "ffn_residual_add_count": resadd, "ffn_down_resadd_count": ffnresadd,
          "block_output_copy_count": block_output_copies, "attention_cast_count": attention_casts,
          "flash_combine_f16_count": combine_f16, "flash_combine_f32_count": combine_f32,
          "combine_copy_count": combine_copy,
          "program_counts": {f"q4k_g3_lanemap_gemv_w1w3fused16_12288_4096": fused16,
                             "q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288": ffnresadd // 2,
                             "q6k_gen_coop_4096_12288_inkernel_epi_ffnresadd": ffnresadd // 2,
                             "r_16_256_r": 37, "q4k_g3_lanemap_gemv_x": 217,
                             "E_32_32_4_fab82d40x": block_output_copies,
                             "E_32_32_4_0a5eb0acx": attention_casts,
                             "flash_fused_gmax_combine_32_128": combine_f32,
                             "flash_fused_gmax_combine_f16_32_128": combine_f16,
                             "E_32_32_4_3b0fcfbcx": combine_copy},
          "population_counts": {"norms": 74, "quant_core": 217 + fused16,
                                "residual_cast_contiguous": resadd, "flash": 20, "other": 6}}


def test_census_gate_passes_on_expected_absorption():
  result = validate_census(_control_census(), _candidate_census())
  assert result["gate_pass"] is True
  assert result["ffn_residual_add_control"] == 0 and result["ffn_residual_add_candidate"] == 0
  assert result["ffn_down_resadd_candidate"] == 36
  assert result["block_output_copy_control"] == 0 and result["block_output_copy_candidate"] == 0
  assert result["attention_cast_control"] == 36 and result["attention_cast_candidate"] == 0
  assert result["flash_combine_f32_control"] == 36 and result["flash_combine_f32_candidate"] == 0
  assert result["flash_combine_f16_control"] == 0 and result["flash_combine_f16_candidate"] == 36
  assert result["combine_copy_control"] == result["combine_copy_candidate"] == 0
  assert result["honest_net_program_delta"] == -36


def test_census_gate_fails_closed_when_block_output_copy_remains():
  result = validate_census(_control_census(block_output_copies=49), _candidate_census(block_output_copies=49))
  assert result["gate_pass"] is False
  assert any("fab82d40" in reason for reason in result["fail_closed"])


def test_census_gate_fails_closed_when_attention_cast_remains():
  result = validate_census(_control_census(), _candidate_census(attention_casts=36, combine_f16=0, combine_f32=36))
  assert result["gate_pass"] is False
  assert any("attention cast" in reason for reason in result["fail_closed"])


def test_census_gate_fails_closed_when_combine_swap_is_not_one_for_one():
  result = validate_census(_control_census(), _candidate_census(combine_f16=18))
  assert result["gate_pass"] is False
  assert any("combine" in reason for reason in result["fail_closed"])


def test_census_gate_fails_closed_when_fp32_combine_remains():
  result = validate_census(_control_census(), _candidate_census(combine_f32=36, combine_f16=0))
  assert result["gate_pass"] is False
  assert any("combine" in reason for reason in result["fail_closed"])


def test_census_gate_fails_closed_when_opaque_combine_copy_appears():
  result = validate_census(_control_census(), _candidate_census(combine_copy=36))
  assert result["gate_pass"] is False
  assert any("3b0fcfbc" in reason for reason in result["fail_closed"])


def test_census_gate_fails_closed_on_unrelated_shift():
  candidate = _candidate_census()
  candidate["program_counts"]["E_other"] = 1
  candidate["kernels"] = 716
  result = validate_census(_control_census(), candidate)
  assert result["gate_pass"] is False
  assert any("unrelated" in reason for reason in result["fail_closed"])


def test_census_gate_fails_closed_when_ffnresadd_not_one_for_one():
  result = validate_census(_control_census(ffnresadd=36), _candidate_census(ffnresadd=18))
  assert result["gate_pass"] is False
  assert any("ffnresadd" in reason for reason in result["fail_closed"])


def test_census_gate_fails_closed_when_m2a_families_shift():
  result = validate_census(_control_census(), _candidate_census(fused16=18))
  assert result["gate_pass"] is False
  assert any("fused16" in reason for reason in result["fail_closed"])
  result = validate_census(_control_census(), _candidate_census(fused=18))
  assert result["gate_pass"] is False
  assert any("fp32 fused" in reason for reason in result["fail_closed"])


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
