import argparse
import numpy as np
import pytest

from extra.qk.q4k_q8_mmq_uop_bench import (CANONICAL_14B_ROLE_SHAPES, DEFAULT_SHAPES, Shape, deterministic_fixture,
  REL_RMSE_THRESHOLD, bounded_original_fp_ds4_oracle, original_fp_subset_authority, parse_shape, prepare_fixture,
  relative_performance, relative_rmse, summarize_samples, validate_original_fp_sum_payload, wide_execution_contract)


def test_default_shape_metrics_and_alignment():
  shape = Shape(16, 32, 256); shape.validate()
  assert shape.logical_ops == 262144
  assert shape.text() == "16x32x256"
  assert DEFAULT_SHAPES == ((16,32,256),(16,32,512),(16,32,5120))


@pytest.mark.parametrize("text", ("15x16x256", "16x16x128", "16x0x256", "broken"))
def test_parse_shape_fails_closed(text):
  with pytest.raises(argparse.ArgumentTypeError): parse_shape(text)


def test_fixture_is_deterministic_packed_and_finite():
  shape = Shape(32, 32, 512)
  a = deterministic_fixture(shape); b = deterministic_fixture(shape)
  assert all(np.array_equal(x, y) for x, y in zip(a, b))
  words, xq, scales = a
  assert (words.dtype, words.shape) == (np.dtype(np.uint32), (32*2*36,))
  assert (xq.dtype, xq.shape) == (np.dtype(np.int8), (32, 512))
  assert (scales.dtype, scales.shape) == (np.dtype(np.float32), (32, 16))
  assert np.isfinite(words.view(np.uint8)).all() and np.isfinite(scales).all()


def test_fixture_original_fp_payload_uses_canonical_llama_producer():
  shape = Shape(16, 32, 256)
  words, xq, scales, sums, prep = prepare_fixture(shape)
  assert np.array_equal((words, xq, scales)[0], deterministic_fixture(shape)[0])
  dequantized_q8 = scales * xq.reshape(shape.m, shape.k//32, 32).astype(np.float32).sum(axis=2)
  assert np.max(np.abs(sums-dequantized_q8)) > 1e-4
  assert prep == {**prep, "producer":"extra.qk.mmq_q4k_q8_reference.q8_1_mmq_ds4_quantize_reference",
                  "sum_semantics":"llama_ds4_y_original_fp32_group_sum", "device_ms":None,
                  "program_count":0, "kernel_count":0}


def test_sum_payload_fails_closed_if_replaced_by_dequantized_q8(monkeypatch):
  from extra.qk.mmq_q4k_q8_reference import q8_1_mmq_ds4_quantize_reference as canonical
  def bad(source):
    values, scales, _ = canonical(source)
    sums = values.reshape(values.shape[0], values.shape[1], 4, 32).astype(np.float32).sum(3) * scales
    return values, scales, sums
  monkeypatch.setattr("extra.qk.mmq_q4k_q8_reference.q8_1_mmq_ds4_quantize_reference", bad)
  with pytest.raises(RuntimeError, match="dequantized-Q8 semantics"):
    prepare_fixture(Shape(16, 16, 256))


def test_all_four_canonical_14b_roles_are_exposed():
  assert CANONICAL_14B_ROLE_SHAPES == {"attn_kv":(512,1024,5120), "attn_qo":(512,5120,5120),
    "ffn_down":(512,5120,17408), "ffn_gate_up":(512,17408,5120)}


def test_full_role_k_original_fp_authority_accepts_semantic_split_and_rejects_dequant_sum():
  # Bounded M/N keeps the independent oracle cheap while K is the real attn_kv role K.
  shape = Shape(16, 16, CANONICAL_14B_ROLE_SHAPES["attn_kv"][2])
  words, xq, scales, sums, _ = prepare_fixture(shape)
  original = bounded_original_fp_ds4_oracle(shape, words, xq, scales, sums)
  dequant_sums = scales * xq.reshape(shape.m, shape.k//32, 32).astype(np.float32).sum(2)
  wrong = bounded_original_fp_ds4_oracle(shape, words, xq, scales, dequant_sums)
  accepted = original_fp_subset_authority(original, original, full_k=shape.k)
  rejected = original_fp_subset_authority(wrong, original, full_k=shape.k)
  assert accepted["rel_rmse_pass"] is True and accepted["full_k"] == shape.k
  assert rejected["rel_rmse"] > 0  # Output tolerance alone cannot establish sum provenance.
  assert not np.allclose(original, wrong, rtol=3e-4, atol=3e-4)
  assert validate_original_fp_sum_payload(xq, scales, sums)["not_dequantized_q8_sum"] is True
  with pytest.raises(RuntimeError, match="dequantized-Q8 semantics"):
    validate_original_fp_sum_payload(xq, scales, dequant_sums)


def test_relative_rmse_is_full_role_authority_while_strict_delta_can_be_diagnostic():
  ref = np.linspace(-100, 100, 1024, dtype=np.float32)
  got = ref + np.float32(0.01)
  assert not np.allclose(got, ref, rtol=3e-4, atol=3e-4)
  assert relative_rmse(got, ref) < REL_RMSE_THRESHOLD


def test_sample_summary_reports_sync_accounting_and_logical_rates():
  row = summarize_samples([.002,.001,.003], [.0015,.001,.002], [1,1,1], 2_000_000_000)
  assert row["wall_median_ms"] == pytest.approx(2)
  assert row["device_median_ms"] == pytest.approx(1.5)
  assert row["device_time_trustworthy"] is True
  assert row["logical_wall_tops"] == pytest.approx(1.0)
  assert row["logical_device_tflops"] == pytest.approx(4/3)


def test_device_time_fails_closed_on_missing_sample_or_kernel_drift():
  missing = summarize_samples([.001,.001], [.0008,None], [1,1], 1000)
  drift = summarize_samples([.001,.001], [.0008,.0008], [1,2], 1000)
  assert missing["device_time_trustworthy"] is False and missing["logical_device_tops"] is None
  assert drift["device_time_trustworthy"] is False and drift["logical_device_tflops"] is None


def test_device_time_requires_exactly_one_launch():
  row = summarize_samples([.001,.001], [.0008,.0008], [2,2], 1000)
  assert row["device_time_trustworthy"] is False


def test_relative_performance_reports_speedup_and_slowdown():
  assert relative_performance({"device_median_ms":1}, {"device_median_ms":2}) == {
    "device_speedup_x":2, "device_change_percent":100, "classification":"speedup"}
  slow = relative_performance({"device_median_ms":4}, {"device_median_ms":2})
  assert slow["device_speedup_x"] == pytest.approx(.5)
  assert slow["device_change_percent"] == pytest.approx(-50)
  assert slow["classification"] == "slowdown"


def test_wide_execution_contract_requires_exact_geometry_and_signed_wmma():
  def row(global_size, signed): return {"compile":{"program_count":1, "signed_i8_wmma_programs":signed,
    "global_sizes":[global_size]}, "kernel_counts":[1,1,1]}
  contract = wide_execution_contract({"wide_wmma_uop":row([1,1,1],1), "wmma_uop":row([2,1,1],1),
                                      "scalar_direct_uop":row([1,16,1],0)})
  assert all(all(checks.values()) for checks in contract.values())
  bad = wide_execution_contract({"wide_wmma_uop":row([2,1,1],1), "wmma_uop":row([2,1,1],0),
                                 "scalar_direct_uop":row([1,16,1],0)})
  assert bad["wide_wmma_uop"]["exact_one_workgroup"] is False
  assert bad["wmma_uop"]["signed_i8_wmma"] is False


def test_execution_contract_allows_full_role_without_exact_shape_wide_comparator():
  row = {"compile":{"program_count":1, "signed_i8_wmma_programs":1, "global_sizes":[[64,32,1]]}, "kernel_counts":[1]}
  scalar = {"compile":{"program_count":1, "signed_i8_wmma_programs":0, "global_sizes":[[1,512,1]]}, "kernel_counts":[1]}
  contract = wide_execution_contract({"wmma_uop":row, "sum_original_fp_wmma":row, "scalar_direct_uop":scalar})
  assert "wide_wmma_uop" not in contract
  assert all(all(checks.values()) for checks in contract.values())
