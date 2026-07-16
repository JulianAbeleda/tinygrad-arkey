import pytest
from tinygrad import Tensor, dtypes
from extra.qk.layout import Q4K_WORDS_PER_BLOCK

from extra.qk.prefill_int8_wmma_spec import describe_q4k_int8_wmma_tiled_prefill, emit_q4k_int8_wmma_tiled_scheduler_tensor


def test_large_scheduler_shape_is_fail_closed_without_output_staging():
  spec = describe_q4k_int8_wmma_tiled_prefill(128, 256, 128, role="compile_gate", n_tile=128)
  with pytest.raises(NotImplementedError, match="compile-backed output subtiles"):
    emit_q4k_int8_wmma_tiled_scheduler_tensor(None, None, None, spec)


def test_large_scheduler_shape_requires_exact_wmma_aligned_stages():
  spec = describe_q4k_int8_wmma_tiled_prefill(128, 256, 128, role="stage_contract", n_tile=32)
  assert spec.n_tile == 32
  assert spec.n % spec.n_tile == 0
  assert spec.live_raw_elems == 16 * 32 * 1

  bad = describe_q4k_int8_wmma_tiled_prefill(128, 256, 128, role="bad_stage", n_tile=16)
  assert bad.n // bad.n_tile == 8


def test_large_scheduler_shape_decomposes_into_n_subtiles(monkeypatch):
  monkeypatch.setenv("DEV", "PYTHON")
  n, k, m = 128, 256, 16
  spec = describe_q4k_int8_wmma_tiled_prefill(n, k, m, role="n_subtile", m_tile=16, n_tile=32, group_tile=8)
  words = Tensor.zeros(n * (k // 256) * Q4K_WORDS_PER_BLOCK, dtype=dtypes.uint32)
  xq = Tensor.zeros(m, k, dtype=dtypes.int8)
  xscales = Tensor.ones(m, k // 32, dtype=dtypes.float32)

  got = emit_q4k_int8_wmma_tiled_scheduler_tensor(words, xq, xscales, spec).realize().numpy()

  assert got.shape == (m, n)
  assert (got == 0).all()
