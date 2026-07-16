import numpy as np
import pytest

from tinygrad import Tensor, dtypes
from tinygrad.engine.realize import compile_linear
from tinygrad.uop.ops import Ops

from extra.qk.mmq_llama_ds4_record_producer import (
  LLAMA_DS4_RECORD_SOURCE_ANCHORS, LlamaDS4RecordProducer, produce_llama_ds4_records)


def _numpy_records(x: np.ndarray):
  m, k = x.shape
  groups = x.reshape(m, k//128, 4, 32).transpose(1, 0, 2, 3)
  amax = np.max(np.abs(groups), axis=-1).astype(np.float32)
  d = (amax / np.float32(127.0)).astype(np.float32)
  divided = np.divide(groups, d[..., None], out=np.zeros_like(groups), where=d[..., None] != 0)
  qs = np.clip(np.rint(divided), -128, 127).astype(np.int8).reshape(k//128, m, 128)
  sums = np.sum(groups, axis=-1, dtype=np.float32)
  ds = np.stack((d, sums), axis=-1).astype(np.float16)
  raw = np.empty((k//128, m, 144), dtype=np.uint8)
  raw[..., :16] = ds.view(np.uint8).reshape(k//128, m, 16)
  raw[..., 16:] = qs.view(np.uint8)
  return raw, ds, qs


def _produce(x): return produce_llama_ds4_records(Tensor(x, dtype=dtypes.float32).realize())


@pytest.mark.parametrize("x", [
  np.zeros((2, 128), dtype=np.float32),
  np.array([np.linspace(-1, 1, 128, dtype=np.float32)], dtype=np.float32),
  np.random.default_rng(12).normal(size=(3, 256)).astype(np.float32),
])
def test_physical_record_matches_independent_numpy_bytes_metadata_and_signed_q(x):
  producer = _produce(x)
  raw, ds, qs = _numpy_records(x)
  np.testing.assert_array_equal(producer.records.numpy(), raw)
  np.testing.assert_array_equal(producer.ds.numpy(), ds)
  np.testing.assert_array_equal(producer.qs.numpy(), qs)
  assert producer.records.shape == (x.shape[1]//128, x.shape[0], 144)
  assert producer.ds.dtype is dtypes.half and producer.qs.dtype is dtypes.int8


def test_quantization_uses_float_scale_before_half_rounding():
  # Values straddle q rounding boundaries for the fp32 d, while d.astype(f16)
  # would move several boundaries.  The latter is explicitly not this ABI.
  x = np.zeros((1, 128), np.float32)
  rng = np.random.default_rng(1)
  for _ in range(2): x[0, :32] = rng.normal(size=32).astype(np.float32) * np.float32(rng.uniform(.01, 100))
  raw, ds, qs = _numpy_records(x)
  producer = _produce(x)
  np.testing.assert_array_equal(producer.records.numpy(), raw)
  half_d_qs = np.clip(np.rint(x[0, :32] / np.float32(ds[0, 0, 0, 0])), -128, 127).astype(np.int8)
  assert np.any(half_d_qs != qs[0, 0, :32])


def test_record_order_is_kblock_then_row_and_views_are_reused():
  x = np.arange(4*256, dtype=np.float32).reshape(4, 256) - np.float32(377)
  producer = _produce(x)
  raw, _, _ = _numpy_records(x)
  np.testing.assert_array_equal(producer.records.numpy(), raw)
  assert producer.views is producer.views
  assert producer.records is producer.views.records
  assert producer.ds is producer.views.ds and producer.qs is producer.views.qs


def test_one_generated_program_and_one_launch_materializes_all_views():
  producer = _produce(np.random.default_rng(4).normal(size=(2, 256)).astype(np.float32))
  linear = compile_linear(producer.records.schedule_linear())
  programs = [u for u in linear.toposort() if u.op is Ops.PROGRAM]
  assert len(programs) == 1
  producer.records.realize()
  assert producer.views is producer.views


def test_rejects_wrong_shape_dtype_alignment_and_non_original_sum_semantics():
  with pytest.raises(ValueError, match="rank 2"): LlamaDS4RecordProducer(Tensor.zeros(128, dtype=dtypes.float32))
  with pytest.raises(TypeError, match="float32"): LlamaDS4RecordProducer(Tensor.zeros(1, 128, dtype=dtypes.float16))
  with pytest.raises(ValueError, match="multiple of 128"): LlamaDS4RecordProducer(Tensor.zeros(1, 160, dtype=dtypes.float32))
  for rejected in ("dequantized_q8_sum", "split", None):
    with pytest.raises(ValueError, match="sum_original_fp"):
      LlamaDS4RecordProducer(Tensor.zeros(1, 128, dtype=dtypes.float32), sum_semantics=rejected)
  assert any("quantize.cu" in anchor for anchor in LLAMA_DS4_RECORD_SOURCE_ANCHORS)
