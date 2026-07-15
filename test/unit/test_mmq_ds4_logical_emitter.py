import numpy as np
import pytest
from tinygrad import Tensor, dtypes

from extra.qk.mmq_ds4_logical_emitter import pack_q8_1_mmq_ds4
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


def test_packed_ds4_packer_rejects_scheduler_candidate():
  with pytest.raises(ValueError, match="packed_ds4"):
    pack_q8_1_mmq_ds4(Tensor.zeros((16, 256)), _spec().logical_candidate())
