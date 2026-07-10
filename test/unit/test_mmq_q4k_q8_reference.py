import numpy as np
import pytest

from tinygrad import Tensor, dtypes

from extra.qk.layout import Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, Q8_1_BLOCK_ELEMS, q4_k_reference, q8_1_quantize
from extra.qk.mmq_q4k_q8_reference import (
  MMQOutputTileSpec, Q81ActivationTileSpec, describe_q4k_q8_1_mmq_tile, q4k_q8_1_mmq_tile_reference,
)


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
  assert spec.to_json()["weight_block_elems"] == Q4_K_BLOCK_ELEMS
  assert spec.to_json()["activation_group_elems"] == Q8_1_BLOCK_ELEMS
  assert spec.activation_spec.to_json()["scale_dtype"] == "float32"
  assert spec.output_spec.to_json()["accumulator_dtype"] == "float32"


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


@pytest.mark.parametrize("value", [0, -7, 127, -128])
def test_q4k_q8_1_mmq_tile_reference_handles_zero_negative_and_edge_q8_values(value:int):
  m, n, k = 3, 4, 256
  raw = _finite_q4k_bytes(n, k, seed=5678)
  xq = np.full((m, k), value, dtype=np.int8)
  xscales = np.full((m, k // Q8_1_BLOCK_ELEMS), 0.25, dtype=np.float32)
  spec = describe_q4k_q8_1_mmq_tile(role="edge_q8", m=m, n=n, k=k, m_tile=m, n_tile=n)

  got = q4k_q8_1_mmq_tile_reference(raw, xq, xscales, spec)
  ref = _dequant_reference(raw, xq, xscales)

  np.testing.assert_allclose(got, ref, rtol=2e-6, atol=2e-5)


def test_q4k_q8_1_mmq_tile_reference_handles_partial_m_and_n_edges():
  m, n, k = 5, 6, 256
  raw = _finite_q4k_bytes(n, k, seed=91011)
  xq, xscales = _q8_inputs(m, k, seed=91012)
  spec = describe_q4k_q8_1_mmq_tile(role="partial", m=m, n=n, k=k, m0=3, n0=4, m_tile=8, n_tile=8)

  got = q4k_q8_1_mmq_tile_reference(raw, xq, xscales, spec)
  ref = _dequant_reference(raw, xq, xscales)[3:5, 4:6]

  assert got.shape == (2, 2)
  assert spec.to_json()["tile_m"] == 2
  assert spec.to_json()["tile_n"] == 2
  np.testing.assert_allclose(got, ref, rtol=2e-6, atol=2e-5)


def test_q4k_q8_1_mmq_tile_reference_handles_multiple_groups_middle_k_slice():
  m, n, k = 4, 5, 768
  raw = _finite_q4k_bytes(n, k, seed=1213)
  xq, xscales = _q8_inputs(m, k, seed=1214)
  spec = describe_q4k_q8_1_mmq_tile(role="middle_k", m=m, n=n, k=k, m_tile=m, n_tile=n, k0=256, k_groups=8)

  got = q4k_q8_1_mmq_tile_reference(raw, xq, xscales, spec)
  w = q4_k_reference(Tensor(raw.reshape(-1).copy()), n * k).reshape(n, k).numpy().astype(np.float32)
  x = (xq[:, 256:512].reshape(m, 8, Q8_1_BLOCK_ELEMS).astype(np.float32) *
       xscales[:, 8:16].reshape(m, 8, 1)).reshape(m, 256)
  ref = x @ w[:, 256:512].T

  assert spec.to_json()["k1"] == 512
  np.testing.assert_allclose(got, ref, rtol=2e-6, atol=2e-5)


def test_q4k_q8_1_mmq_tile_reference_tracks_q4k_scale_min_metadata():
  m, n, k = 2, 3, 256
  raw = _finite_q4k_bytes(n, k, seed=1415)
  changed = raw.copy()
  changed[:, :, 0:2] = np.array([0.125], dtype=np.float16).view(np.uint8)
  changed[:, :, 2:4] = np.array([-0.0625], dtype=np.float16).view(np.uint8)
  xq, xscales = _q8_inputs(m, k, seed=1416)
  spec = describe_q4k_q8_1_mmq_tile(role="scale_min", m=m, n=n, k=k, m_tile=m, n_tile=n)

  got = q4k_q8_1_mmq_tile_reference(changed, xq, xscales, spec)
  ref = _dequant_reference(changed, xq, xscales)
  original = q4k_q8_1_mmq_tile_reference(raw, xq, xscales, spec)

  np.testing.assert_allclose(got, ref, rtol=2e-6, atol=2e-5)
  assert not np.allclose(got, original, rtol=2e-6, atol=2e-5)


def test_q8_1_activation_and_output_tile_specs_validate_and_serialize_edges():
  act = Q81ActivationTileSpec(m=5, k=256, m0=4, m_tile=16, k0=224, k_groups=1)
  out = MMQOutputTileSpec(m=5, n=7, m0=4, n0=6, m_tile=16, n_tile=16)

  act.validate()
  out.validate()

  assert act.to_json()["tile_m"] == 1
  assert act.to_json()["k1"] == 256
  assert out.to_json()["tile_m"] == 1
  assert out.to_json()["tile_n"] == 1


def test_q4k_q8_1_mmq_tile_contract_rejects_unaligned_k0():
  with pytest.raises(ValueError, match="Q8_1 block aligned"):
    describe_q4k_q8_1_mmq_tile(role="bad", m=1, n=1, k=256, k0=16)


@pytest.mark.parametrize(
  "kwargs, message",
  [
    ({"quant_format": "Q6_K"}, "quant_format must be Q4_K"),
    ({"activation_format": "fp16"}, "activation_format must be Q8_1"),
    ({"packed_weight_layout": "blocked"}, "unsupported packed_weight_layout"),
    ({"activation_layout": "blocked"}, "unsupported activation_layout"),
    ({"output_layout": "strided"}, "unsupported output_layout"),
    ({"split_policy": "reduce_k"}, "unsupported split_policy"),
    ({"accumulator_dtype": "int32"}, "accumulator_dtype must be float32"),
    ({"output_dtype": "float16"}, "output_dtype must be float32"),
    ({"weight_block_elems": 128}, "weight_block_elems must be"),
    ({"activation_group_elems": 64}, "activation_group_elems must be"),
  ],
)
def test_q4k_q8_1_mmq_tile_contract_rejects_unsupported_spec_fields(kwargs, message):
  with pytest.raises(ValueError, match=message):
    describe_q4k_q8_1_mmq_tile(role="bad_fields", m=1, n=1, k=256, **kwargs)


def test_q4k_q8_1_mmq_tile_reference_rejects_bad_activation_shapes():
  spec = describe_q4k_q8_1_mmq_tile(role="bad_activation", m=2, n=1, k=256)
  raw = _finite_q4k_bytes(1, 256, seed=1516)
  with pytest.raises(ValueError, match="xq shape"):
    q4k_q8_1_mmq_tile_reference(raw, np.zeros((1, 256), dtype=np.int8), np.ones((2, 8), dtype=np.float32), spec)
  with pytest.raises(ValueError, match="xscales shape"):
    q4k_q8_1_mmq_tile_reference(raw, np.zeros((2, 256), dtype=np.int8), np.ones((2, 7), dtype=np.float32), spec)


def test_q4k_q8_1_mmq_tile_reference_rejects_bad_weight_size():
  spec = describe_q4k_q8_1_mmq_tile(role="bad_size", m=1, n=1, k=256)
  with pytest.raises(ValueError, match="expected 144 Q4_K bytes"):
    q4k_q8_1_mmq_tile_reference(np.zeros(143, dtype=np.uint8), np.zeros((1, 256), dtype=np.int8),
                                np.ones((1, 8), dtype=np.float32), spec)
