import numpy as np
import pytest
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import Ops
from extra.qk.q4k_q8_activation_producer import (Q4KQ8ActivationProducer, Q4KQ8ActivationSumSemantics,
  LlamaDS4Q8ActivationSumOriginalFPProducer, benchmark_llama_ds4_q8_activation_sum_original_fp,
  produce_llama_ds4_q8_activation_sum_original_fp)


def test_q8_producer_materializes_once_and_reuses_tiles(monkeypatch):
  monkeypatch.setenv("DEV", "PYTHON")
  x = Tensor(np.arange(2 * 64, dtype=np.float32).reshape(2, 64) - 31)
  p = Q4KQ8ActivationProducer(x)
  values, scales, sums = p.operands
  a, b = p.tile(0, 1, 0, 32), p.tile(1, 1, 32, 32)
  assert a.values.numpy().tolist() == values.numpy().reshape(2, 64)[0:1, :32].tolist()
  assert b.scales.numpy().tolist() == scales.numpy().reshape(2, 2)[1:2, 1:2].tolist()
  expected = (values.numpy().reshape(2, 2, 32).astype(np.float32) * scales.numpy().reshape(2, 2, 1)).sum(axis=2)
  np.testing.assert_allclose(sums.numpy().reshape(2, 2), expected, rtol=1e-6, atol=1e-4)


def _sum_original_fp_reference(x):
  blocks = np.asarray(x, dtype=np.float32).reshape(-1, 32)
  scales = np.max(np.abs(blocks), axis=1) / np.float32(127)
  scales = np.where(scales == 0, np.float32(1), scales).astype(np.float32)
  values = np.clip(np.rint(blocks / scales[:, None]), -128, 127).astype(np.int8)
  return values.reshape(-1), scales, blocks.sum(axis=1, dtype=np.float32)


def test_sum_original_fp_semantic_split_on_adversarial_values():
  # 0.1 is deliberately not represented exactly by the scale selected by the 1.0 outlier.
  x_np = np.array([[0.1] * 31 + [1.0]], dtype=np.float32)
  x = Tensor(x_np, device="AMD").realize()
  original = produce_llama_ds4_q8_activation_sum_original_fp(x)
  values, scales, sums_original_fp = (t.numpy() for t in original.operands_sum_original_fp)
  dequant_sum = (values.reshape(1, 32).astype(np.float32) * scales.reshape(1, 1)).sum(axis=1)
  np.testing.assert_allclose(sums_original_fp, x_np.sum(axis=1), rtol=0, atol=2e-6)
  assert not np.array_equal(sums_original_fp, dequant_sum)


def test_sum_original_fp_matches_python_reference_and_ds4_layout():
  rng = np.random.default_rng(20260715)
  x_np = rng.normal(size=(3, 256)).astype(np.float32)
  x = Tensor(x_np, device="AMD").realize()
  producer = LlamaDS4Q8ActivationSumOriginalFPProducer(x)
  got = tuple(t.numpy() for t in producer.operands_sum_original_fp)
  ref = _sum_original_fp_reference(x_np)
  np.testing.assert_array_equal(got[0], ref[0])
  np.testing.assert_allclose(got[1], ref[1], rtol=2e-6, atol=1e-7)
  np.testing.assert_allclose(got[2], ref[2], rtol=2e-6, atol=2e-5)
  ds4_values, ds4_scales, ds4_sums = producer.source_anchored_ds4_sum_original_fp_operands()
  assert ds4_values.shape == (2, 3, 128)
  assert ds4_scales.shape == ds4_sums.shape == (2, 3, 4)
  assert producer.sum_semantics is Q4KQ8ActivationSumSemantics.SUM_ORIGINAL_FP
  assert any("quantize_mmq_q8_1" in anchor for anchor in producer.source_anchors)


def test_sum_original_fp_exactly_one_program_and_one_kernel():
  from tinygrad.engine.realize import compile_linear, run_linear
  from tinygrad.helpers import GlobalCounters
  x = Tensor.rand(2, 64, device="AMD", dtype=dtypes.float32).realize()
  producer = produce_llama_ds4_q8_activation_sum_original_fp(x)
  linear = compile_linear(producer.values.schedule_linear())
  assert len([u for u in linear.toposort() if u.op is Ops.PROGRAM]) == 1
  before = GlobalCounters.kernel_count
  run_linear(linear, wait=True)
  assert GlobalCounters.kernel_count - before == 1


def test_one_sum_original_fp_materialization_is_reused_across_tiles():
  from tinygrad.engine.realize import compile_linear, run_linear
  from tinygrad.helpers import GlobalCounters
  x = Tensor.rand(4, 128, device="AMD", dtype=dtypes.float32).realize()
  producer = produce_llama_ds4_q8_activation_sum_original_fp(x)
  tile_a = producer.tile_sum_original_fp(0, 2, 0, 64)
  tile_b = producer.tile_sum_original_fp(2, 2, 64, 64)
  assert tile_a.values.uop.base is producer.values.uop.base and tile_b.values.uop.base is producer.values.uop.base
  assert tile_a.sums.uop.base is producer.sums_original_fp.uop.base and tile_b.sums.uop.base is producer.sums_original_fp.uop.base
  linear = compile_linear(producer.values.schedule_linear())
  before = GlobalCounters.kernel_count
  run_linear(linear, wait=True)
  assert GlobalCounters.kernel_count - before == 1


@pytest.mark.parametrize("ambiguous", ["sum_original_fp", "sum_dequant_q8", None])
def test_sum_original_fp_rejects_ambiguous_or_dequant_sum_request(ambiguous):
  with pytest.raises(ValueError, match="distinct sum_original_fp semantic enum"):
    LlamaDS4Q8ActivationSumOriginalFPProducer(Tensor.zeros(1, 32), sum_semantics=ambiguous)


def test_sum_original_fp_small_shape_benchmark_exposes_role_accounting_helper():
  x = Tensor.rand(2, 64, device="AMD", dtype=dtypes.float32).realize()
  report = benchmark_llama_ds4_q8_activation_sum_original_fp(x, warmups=1, rounds=2)
  assert report["producer"] == "llama_ds4_q8_activation_sum_original_fp"
  assert report["sum_semantics"] == "sum_original_fp"
  assert report["program_count"] == 1 and report["kernel_counts"] == [1, 1]
  assert report["wall_median_ms"] > 0 and report["device_median_ms"] > 0
