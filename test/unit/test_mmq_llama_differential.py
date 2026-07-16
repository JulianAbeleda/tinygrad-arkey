from copy import deepcopy

import pytest

from extra.qk.mmq_llama_differential import (
  REQUIRED_DIMENSIONS, compare_current_direct_uop, compare_structures, current_direct_uop_descriptor,
  llama_q4k_q8_structural_descriptor,
)
from extra.qk.q4k_q8_mmq_uop import describe_q4k_q8_mmq_wmma


def test_current_emitter_is_blocked_with_exact_structural_gaps():
  spec = describe_q4k_q8_mmq_wmma(m=16, n=16, k=256)
  result = compare_current_direct_uop(spec)

  assert result.status == "BLOCKED" and result.equivalent is False
  assert [(gap["dimension"], gap["kind"]) for gap in result.gaps] == [
    ("tile_m", "mismatch"), ("tile_n", "mismatch"), ("waves_workgroup", "missing"),
    ("lds_q4_panel", "missing"), ("lds_q8_panel", "missing"), ("barriers", "missing"),
    ("q8_ds_semantics", "mismatch"), ("accumulator_ownership", "missing"),
    ("k_lifecycle", "missing"), ("dot_primitive", "missing"), ("writeback", "missing"),
    ("stream_k", "missing"), ("resource_fields", "missing"),
  ]
  payload = result.to_json()
  assert payload["numeric_correctness_considered"] is False
  assert set(REQUIRED_DIMENSIONS) == set(llama_q4k_q8_structural_descriptor()["dimensions"])


def test_current_descriptor_does_not_invent_structure_from_correctness_evidence():
  spec = describe_q4k_q8_mmq_wmma()
  candidate = current_direct_uop_descriptor(spec, {"correctness": {"passed": True, "max_abs_error": 0.0}})
  result = compare_structures(candidate)
  assert result.status == "BLOCKED"
  assert "barriers" not in candidate["dimensions"] and "writeback" not in candidate["dimensions"]


def test_gfx1100_runtime_ds4_barriers_and_exact_lds_contract():
  descriptor = llama_q4k_q8_structural_descriptor()
  dims = descriptor["dimensions"]
  assert descriptor["source_commit"] == "ac4cddeb0dbd778f650bf568f6f08344a06abe3a"
  assert dims["waves_workgroup"] == {"block": (32, 8, 1), "waves": 8, "wave_size": 32,
                                      "workgroup_threads": 256}
  assert dims["stream_k"] == {
    "enabled": False, "target": "gfx1100_rdna3", "runtime_path": "conventional_tiling",
    "grid": {"x": "ceil(nrows_x/128)", "y": "ceil(ncols_max/128)", "z": "channels*samples"},
    "generic_source_support": True, "enable_condition": "nvidia_volta_plus_or_cdna"}
  assert dims["q8_ds_semantics"]["sum_semantic"] == "sum_original_fp"
  assert dims["q8_ds_semantics"]["sum_timing"] == "before_quantization"
  assert dims["barriers"]["per_k_iteration"] == 4
  assert dims["barriers"]["sequence"] == (
    "stage_q4_and_q8_half0", "barrier", "dot_half0", "barrier",
    "stage_q8_half1", "barrier", "dot_half1", "barrier")
  assert dims["lds_q4_panel"]["row_stride_ints"] == 76
  assert dims["lds_q4_panel"]["bytes"] == 128 * 76 * 4 == 38912
  assert dims["lds_q4_panel"]["row_layout"] == "aos_interleaved"
  assert dims["lds_q4_panel"]["row_components"] == (("qs", 0, 256), ("dm", 256, 32), ("padding", 288, 16))
  assert dims["lds_q8_panel"]["bytes"] == 128 * 144 == 18432
  assert dims["lds_q8_panel"]["half_k_elements"] == 128
  assert dims["lds_q8_panel"]["row_components"] == (("ds", 0, 16), ("qs", 16, 128))
  assert dims["k_lifecycle"]["scale_groups_per_step"] == 8
  assert dims["k_lifecycle"]["integer_accumulator_scope"] == "one_scale_group"
  assert dims["k_lifecycle"]["float_correction_timing"] == "immediately_after_each_scale_group"
  assert dims["dot_primitive"]["intrinsic_k"] == 16
  assert dims["dot_primitive"]["wmma_per_scale_group"] == 2
  assert dims["resource_fields"]["lds_bytes"] == 512 + 18432 + 38912 == 57856
  assert dims["writeback"]["role_tails"] is False


def test_wmma_mnemonic_alone_does_not_claim_the_oracle_group_lifecycle():
  oracle_dot = llama_q4k_q8_structural_descriptor()["dimensions"]["dot_primitive"]
  assert oracle_dot["signed_A"] is oracle_dot["signed_B"] is True
  spec = describe_q4k_q8_mmq_wmma()
  evidence = {"final_isa": {"wmma_mnemonic": "v_wmma_i32_16x16x16_iu8"}}
  candidate = current_direct_uop_descriptor(spec, evidence)
  assert candidate["dimensions"]["dot_primitive"] == {"isa": "v_wmma_i32_16x16x16_iu8"}
  assert candidate["dimensions"]["dot_primitive"] != oracle_dot
  assert any(gap["dimension"] == "dot_primitive" for gap in compare_structures(candidate).gaps)


def test_missing_candidate_fact_and_missing_oracle_vocabulary_fail_closed():
  oracle = llama_q4k_q8_structural_descriptor()
  candidate = deepcopy(oracle)
  del candidate["dimensions"]["barriers"]
  result = compare_structures(candidate, oracle)
  assert [(gap["dimension"], gap["kind"]) for gap in result.gaps] == [("barriers", "missing")]

  broken_oracle = deepcopy(oracle)
  del broken_oracle["dimensions"]["stream_k"]
  with pytest.raises(ValueError, match="oracle missing required dimension stream_k"):
    compare_structures(oracle, broken_oracle)


def test_synthetic_oracle_equivalent_descriptor_passes():
  oracle = llama_q4k_q8_structural_descriptor()
  synthetic = {"schema": oracle["schema"], "descriptor_id": "synthetic-equivalent",
               "dimensions": deepcopy(oracle["dimensions"])}
  result = compare_structures(synthetic, oracle)
  assert result.status == "EQUIVALENT" and result.equivalent is True and result.gaps == ()
