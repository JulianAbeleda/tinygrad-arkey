import numpy as np
import pytest

from tinygrad import Tensor, dtypes

from extra.qk.layout import Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, Q8_1_BLOCK_ELEMS, q4_k_reference, q8_1_quantize
from extra.qk.mmq_q4k_q8_reference import describe_q4k_q8_1_mmq_tile, q4k_q8_1_mmq_tile_reference


def _finite_q4k_bytes(n:int, k:int, seed:int) -> np.ndarray:
  rng = np.random.default_rng(seed)
  assert k % Q4_K_BLOCK_ELEMS == 0
  nblocks = n * k // Q4_K_BLOCK_ELEMS
  raw = rng.integers(0, 256, size=(nblocks, Q4_K_BLOCK_BYTES), dtype=np.uint8)
  raw[:, 0:2] = (rng.standard_normal(nblocks).astype(np.float32) * 0.05).astype(np.float16).view(np.uint8).reshape(nblocks, 2)
  raw[:, 2:4] = (rng.standard_normal(nblocks).astype(np.float32) * 0.05).astype(np.float16).view(np.uint8).reshape(nblocks, 2)
  return raw.reshape(n, k // Q4_K_BLOCK_ELEMS, Q4_K_BLOCK_BYTES)


def _q8_inputs(m:int, k:int, seed:int):
  x = Tensor(np.random.default_rng(seed).standard_normal((m, k)).astype(np.float32)).realize()
  xq, xscales = q8_1_quantize(x.cast(dtypes.float32))
  return xq.numpy().reshape(m, k), xscales.numpy().reshape(m, k // Q8_1_BLOCK_ELEMS)


def _dequant_reference(raw:np.ndarray, xq:np.ndarray, xscales:np.ndarray) -> np.ndarray:
  n, k_blocks, _ = raw.shape
  m, k = xq.shape
  w = q4_k_reference(Tensor(raw.reshape(-1).copy()), n * k_blocks * Q4_K_BLOCK_ELEMS).reshape(n, -1).numpy().astype(np.float32)
  x = (xq.reshape(m, -1, Q8_1_BLOCK_ELEMS).astype(np.float32) * xscales.reshape(m, -1, 1)).reshape(m, k)
  return (x @ w.T).astype(np.float32)


def test_q4k_q8_1_mmq_tile_reference_matches_existing_q4k_reference():
  m, n, k = 8, 12, 256
  raw = _finite_q4k_bytes(n, k, seed=20260710)
  xq, xscales = _q8_inputs(m, k, seed=20260711)
  spec = describe_q4k_q8_1_mmq_tile(role="unit", m=m, n=n, k=k, m0=2, n0=3, m_tile=4, n_tile=5)

  got = q4k_q8_1_mmq_tile_reference(raw, xq, xscales, spec)
  ref = _dequant_reference(raw, xq, xscales)[2:6, 3:8]

  np.testing.assert_allclose(got, ref, rtol=2e-6, atol=2e-5)
  assert spec.to_json()["quant_format"] == "Q4_K"
  assert spec.to_json()["activation_format"] == "Q8_1"


def test_q4k_q8_1_mmq_k_split_tiles_sum_to_full_tile():
  m, n, k = 6, 7, 512
  raw = _finite_q4k_bytes(n, k, seed=1234)
  xq, xscales = _q8_inputs(m, k, seed=1235)
  full = describe_q4k_q8_1_mmq_tile(role="split", m=m, n=n, k=k, m_tile=m, n_tile=n)
  first = describe_q4k_q8_1_mmq_tile(role="split", m=m, n=n, k=k, m_tile=m, n_tile=n, k0=0, k_groups=8)
  second = describe_q4k_q8_1_mmq_tile(role="split", m=m, n=n, k=k, m_tile=m, n_tile=n, k0=256, k_groups=8)

  got = q4k_q8_1_mmq_tile_reference(raw, xq, xscales, first) + q4k_q8_1_mmq_tile_reference(raw, xq, xscales, second)
  ref = q4k_q8_1_mmq_tile_reference(raw, xq, xscales, full)

  np.testing.assert_allclose(got, ref, rtol=2e-6, atol=2e-5)


def test_q4k_q8_1_mmq_tile_contract_rejects_unaligned_k0():
  with pytest.raises(ValueError, match="Q8_1 block aligned"):
    describe_q4k_q8_1_mmq_tile(role="bad", m=1, n=1, k=256, k0=16)


def test_q4k_q8_1_mmq_tile_reference_rejects_bad_weight_size():
  spec = describe_q4k_q8_1_mmq_tile(role="bad_size", m=1, n=1, k=256)
  with pytest.raises(ValueError, match="expected 144 Q4_K bytes"):
    q4k_q8_1_mmq_tile_reference(np.zeros(143, dtype=np.uint8), np.zeros((1, 256), dtype=np.int8),
                                np.ones((1, 8), dtype=np.float32), spec)
