import numpy as np

from tinygrad import Tensor, dtypes

from extra.qk.layout import Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, Q8_1_BLOCK_ELEMS, q8_1_quantize
from extra.qk.mmq_lifecycle import COUNTER_NAMES
from extra.qk.mmq_q4k_q8_atom import (
  BACKEND_ATOM_ID, run_q4k_q8_1_mmq_tile, run_q4k_q8_1_mmq_tile_with_lifecycle,
)
from extra.qk.mmq_q4k_q8_reference import describe_q4k_q8_1_mmq_tile, q4k_q8_1_mmq_tile_reference


def _finite_q4k_bytes(n:int, k:int, seed:int) -> np.ndarray:
  rng = np.random.default_rng(seed)
  nblocks = n * k // Q4_K_BLOCK_ELEMS
  raw = rng.integers(0, 256, size=(nblocks, Q4_K_BLOCK_BYTES), dtype=np.uint8)
  raw[:, 0:2] = (rng.standard_normal(nblocks).astype(np.float32) * 0.05).astype(np.float16).view(np.uint8).reshape(nblocks, 2)
  raw[:, 2:4] = (rng.standard_normal(nblocks).astype(np.float32) * 0.05).astype(np.float16).view(np.uint8).reshape(nblocks, 2)
  return raw.reshape(n, k // Q4_K_BLOCK_ELEMS, Q4_K_BLOCK_BYTES)


def _q8_inputs(m:int, k:int, seed:int):
  x = Tensor(np.random.default_rng(seed).standard_normal((m, k)).astype(np.float32)).realize()
  xq, xscales = q8_1_quantize(x.cast(dtypes.float32))
  return xq.numpy().reshape(m, k), xscales.numpy().reshape(m, k // Q8_1_BLOCK_ELEMS)


def test_q4k_q8_1_mmq_atom_matches_reference_tile():
  m, n, k = 7, 9, 256
  raw = _finite_q4k_bytes(n, k, seed=20260712)
  xq, xscales = _q8_inputs(m, k, seed=20260713)
  spec = describe_q4k_q8_1_mmq_tile(role="ffn_gate_up", m=m, n=n, k=k, m0=2, n0=3, m_tile=4, n_tile=5)

  got = run_q4k_q8_1_mmq_tile(raw, xq, xscales, spec)
  ref = q4k_q8_1_mmq_tile_reference(raw, xq, xscales, spec)

  np.testing.assert_allclose(got, ref, rtol=0, atol=0)


def test_q4k_q8_1_mmq_atom_reports_lifecycle_contract():
  m, n, k = 8, 8, 256
  raw = _finite_q4k_bytes(n, k, seed=101)
  xq, xscales = _q8_inputs(m, k, seed=102)
  spec = describe_q4k_q8_1_mmq_tile(role="ffn_gate_up", m=m, n=n, k=k, m_tile=4, n_tile=4)

  result = run_q4k_q8_1_mmq_tile_with_lifecycle(raw, xq, xscales, spec)
  row = result.to_json()

  assert row["backend_atom_id"] == BACKEND_ATOM_ID
  assert row["output_shape"] == [4, 4]
  assert set(row["lifecycle"]["counters"]) == set(COUNTER_NAMES)
  assert row["lifecycle"]["counters"]["activation_quant_epochs"] == 1
  assert row["lifecycle"]["counters"]["output_stores"] == 16
