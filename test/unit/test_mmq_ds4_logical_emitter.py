from dataclasses import replace
import numpy as np
import pytest
from tinygrad import Tensor, dtypes

from extra.qk.layout import q8_1_quantize
from extra.qk.mmq_ds4_logical_emitter import pack_q8_1_mmq_ds4, packed_row_major_candidate
from extra.qk.q4k_q8_mmq_prefill_spec import Q4KQ8MMQPrefillSpec


def _spec(**kw):
  fields = dict(workload="prefill", profile="qwen3-14b", role="attn_kv", quant_format="Q4_K",
                activation_format="Q8_1", weight_layout="q4k", output_layout="tokens_rows",
                m=16, n=16, k=256)
  fields.update(kw)
  return Q4KQ8MMQPrefillSpec(**fields)


def test_packed_ds4_candidate_declares_supplied_sums_and_physical_mapping():
  candidate = _spec().packed_ds4_logical_candidate()
  assert candidate.mapping.lifecycle == "packed_ds4"
  assert candidate.mapping.workgroup_size == candidate.mapping.wave_size == 32
  assert candidate.mapping.wmma_shape == (4, 16, 16)
  assert candidate.descriptor.q8.sum_policy == "supplied"
  assert candidate.descriptor.q8.sum_operand


def test_q8_ds4_packer_uses_candidate_axes_and_flat_abi():
  candidate = _spec().packed_ds4_logical_candidate()
  x = Tensor(np.arange(16 * 256, dtype=np.float32).reshape(16, 256))
  values, scales, sums = pack_q8_1_mmq_ds4(x, candidate)
  assert values.shape == (16 * 256,)
  assert scales.shape == sums.shape == (16 * 8,)
  assert values.dtype == dtypes.int8
  assert np.isfinite(scales.numpy()).all() and np.isfinite(sums.numpy()).all()


def test_q8_ds4_packer_supplies_weighted_dequantized_sums():
  candidate = _spec().packed_ds4_logical_candidate()
  x = Tensor(np.random.default_rng(77).standard_normal((16, 256)).astype(np.float32))
  values, scales, sums = pack_q8_1_mmq_ds4(x, candidate)
  qvalues, qscales = q8_1_quantize(x.cast(dtypes.float32))
  q8 = candidate.descriptor.q8
  expected = (qvalues.reshape(16, 256 // q8.packed_block_elements, q8.groups_per_packed_block, q8.block_elements).cast(dtypes.float32) *
              qscales.reshape(16, 256 // q8.packed_block_elements, q8.groups_per_packed_block, 1).expand(
                16, 256 // q8.packed_block_elements, q8.groups_per_packed_block, q8.block_elements)).sum(axis=3).permute(1, 0, 2).reshape(-1)
  np.testing.assert_allclose(sums.numpy(), expected.numpy(), rtol=0, atol=0)
  np.testing.assert_array_equal(values.numpy(), qvalues.reshape(16, 256 // q8.packed_block_elements, q8.groups_per_packed_block,
                                                               q8.block_elements).permute(1, 0, 2, 3).reshape(-1).numpy())
  np.testing.assert_allclose(scales.numpy(), qscales.reshape(16, 256 // q8.packed_block_elements,
                                                             q8.groups_per_packed_block).permute(1, 0, 2).reshape(-1).numpy(), rtol=0, atol=0)


def test_packed_ds4_packer_rejects_unsupported_descriptor_geometry():
  candidate = _spec().packed_ds4_logical_candidate()
  bad_q4 = replace(candidate.descriptor.q4k, metadata_words=3)
  bad = replace(candidate, descriptor=replace(candidate.descriptor, q4k=bad_q4))
  with pytest.raises(ValueError, match="canonical Q4_K"):
    pack_q8_1_mmq_ds4(Tensor.zeros((16, 256)), bad)


def test_row_major_candidate_declares_storage_and_preserves_flat_shapes():
  candidate = packed_row_major_candidate(16, 16, 256, role="attn_kv")
  assert candidate.descriptor.abi["activation_storage"] == "row_major"
  values, scales, sums = pack_q8_1_mmq_ds4(Tensor.zeros((16, 256)), candidate)
  assert values.shape == (16 * 256,)
  assert scales.shape == sums.shape == (16 * 8,)


def test_packed_ds4_packer_rejects_scheduler_candidate():
  with pytest.raises(ValueError, match="packed_ds4"):
    pack_q8_1_mmq_ds4(Tensor.zeros((16, 256)), _spec().logical_candidate())
