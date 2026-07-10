import numpy as np
import pytest

from tinygrad import Tensor, dtypes

from extra.qk.layout import Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, Q8_1_BLOCK_ELEMS, q4_k_reference, q8_1_quantize
from extra.qk.mmq_q4k_q8_reference import (
  MMQOutputTileSpec, Q81ActivationTileSpec, Q81MMQDS4ActivationSpec, Q8_1_MMQ_DS4_BLOCK_ELEMS,
  Q8_1_MMQ_DS4_GROUPS_PER_BLOCK, Q8_1_MMQ_DS4_LAYOUT, Q8_1_MMQ_DS4_VALUES_PER_GROUP,
  describe_q4k_q8_1_mmq_tile, q4k_q8_1_mmq_ds4_tile_reference, q4k_q8_1_mmq_tile_reference,
  q8_1_mmq_ds4_dequantize_reference, q8_1_mmq_ds4_from_row_major_reference, q8_1_mmq_ds4_quantize_reference,
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


def _row_major_ds4_values(values:np.ndarray) -> np.ndarray:
  k_blocks, m, _ = values.shape
  return values.reshape(k_blocks, m, Q8_1_MMQ_DS4_GROUPS_PER_BLOCK, Q8_1_MMQ_DS4_VALUES_PER_GROUP).transpose(1, 0, 2, 3).reshape(m, -1)


def _row_major_ds4_scales(scales:np.ndarray) -> np.ndarray:
  return scales.transpose(1, 0, 2).reshape(scales.shape[1], -1)


def _dequant_reference(raw:np.ndarray, xq:np.ndarray, xscales:np.ndarray) -> np.ndarray:
  n, k_blocks, _ = raw.shape
  m, k = xq.shape
  w = q4_k_reference(Tensor(raw.reshape(-1).copy()), n * k_blocks * Q4_K_BLOCK_ELEMS).reshape(n, -1).numpy().astype(np.float32)
  x = (xq.reshape(m, -1, Q8_1_BLOCK_ELEMS).astype(np.float32) * xscales.reshape(m, -1, 1)).reshape(m, k)
  return (x @ w.T).astype(np.float32)


def test_q8_1_mmq_ds4_quantize_reference_matches_existing_q8_1_dequant():
  m, k = 5, 256
  x = np.random.default_rng(1617).standard_normal((m, k)).astype(np.float32) * 3.0
  values, scales, sums = q8_1_mmq_ds4_quantize_reference(x)
  xq, xscales = q8_1_quantize(Tensor(x).cast(dtypes.float32))
  ref = (xq.numpy().reshape(m, k // Q8_1_BLOCK_ELEMS, Q8_1_BLOCK_ELEMS).astype(np.float32) *
         xscales.numpy().reshape(m, k // Q8_1_BLOCK_ELEMS, 1)).reshape(m, k)

  assert values.shape == (k // Q8_1_MMQ_DS4_BLOCK_ELEMS, m, Q8_1_MMQ_DS4_BLOCK_ELEMS)
  assert scales.shape == sums.shape == (k // Q8_1_MMQ_DS4_BLOCK_ELEMS, m, Q8_1_MMQ_DS4_GROUPS_PER_BLOCK)
  np.testing.assert_array_equal(_row_major_ds4_values(values), xq.numpy().reshape(m, k))
  np.testing.assert_allclose(_row_major_ds4_scales(scales), xscales.numpy().reshape(m, k // Q8_1_BLOCK_ELEMS), rtol=0, atol=0)
  np.testing.assert_allclose(q8_1_mmq_ds4_dequantize_reference(values, scales), ref, rtol=0, atol=0)


def test_q8_1_mmq_ds4_zero_inputs_do_not_produce_nan_and_store_real_sums():
  x = np.zeros((3, 128), dtype=np.float32)
  values, scales, sums = q8_1_mmq_ds4_quantize_reference(x)
  deq = q8_1_mmq_ds4_dequantize_reference(values, scales)

  assert np.all(values == 0)
  assert np.all(scales == 1.0)
  assert np.all(sums == 0.0)
  assert np.isfinite(deq).all()
  assert not np.isnan(scales).any()


def test_q8_1_mmq_ds4_signed_negatives_preserve_int8_semantics():
  x = -np.tile(np.arange(1, 129, dtype=np.float32), (2, 1))
  values, scales, sums = q8_1_mmq_ds4_quantize_reference(x)
  row_values = _row_major_ds4_values(values)

  assert values.dtype == np.int8
  assert np.all(row_values <= 0)
  assert np.any(row_values < 0)
  np.testing.assert_allclose(sums, x.reshape(2, 1, 4, 32).sum(axis=3).transpose(1, 0, 2), rtol=0, atol=0)


def test_q8_1_mmq_ds4_edge_magnitudes_match_existing_q8_1_reference():
  x = np.array([
    np.linspace(-1000.0, 1000.0, 128, dtype=np.float32),
    np.concatenate([np.array([-1e20, -129.0, -128.0, -127.5, 0.0, 127.5, 128.0, 129.0, 1e20], dtype=np.float32),
                    np.linspace(-2.0, 2.0, 119, dtype=np.float32)]),
  ], dtype=np.float32)
  values, scales, _ = q8_1_mmq_ds4_quantize_reference(x)
  xq, xscales = q8_1_quantize(Tensor(x).cast(dtypes.float32))

  np.testing.assert_array_equal(_row_major_ds4_values(values), xq.numpy().reshape(x.shape))
  np.testing.assert_allclose(_row_major_ds4_scales(scales), xscales.numpy().reshape(2, 4), rtol=0, atol=0)
  assert np.isfinite(q8_1_mmq_ds4_dequantize_reference(values, scales)).all()


def test_q8_1_mmq_ds4_sums_match_original_fp32_per_32_groups():
  x = (np.random.default_rng(1819).standard_normal((4, 384)).astype(np.float32) * 0.5).astype(np.float32)
  _, _, sums = q8_1_mmq_ds4_quantize_reference(x)
  ref = x.reshape(4, 3, 4, 32).sum(axis=3, dtype=np.float32).transpose(1, 0, 2)

  np.testing.assert_allclose(sums, ref, rtol=0, atol=0)
  assert np.any(sums != 0.0)


def test_q8_1_mmq_ds4_spec_requires_128_aligned_k_and_whole_blocks():
  spec = Q81MMQDS4ActivationSpec(m=5, k=256, m0=4, m_tile=16, k0=128, k_groups=4)
  spec.validate()

  assert spec.to_json()["tile_m"] == 1
  assert spec.to_json()["layout"] == Q8_1_MMQ_DS4_LAYOUT
  assert spec.to_json()["block_elems"] == 128
  with pytest.raises(ValueError, match="128-aligned"):
    q8_1_mmq_ds4_quantize_reference(np.zeros((1, 160), dtype=np.float32))
  with pytest.raises(ValueError, match="whole 128-value"):
    Q81MMQDS4ActivationSpec(m=1, k=256, k_groups=1).validate()


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


def test_q4k_q8_1_mmq_ds4_tile_reference_matches_row_major_reference():
  m, n, k = 8, 12, 256
  raw = _finite_q4k_bytes(n, k, seed=20261710)
  xq, xscales = _q8_inputs(m, k, seed=20261711)
  q8_ds4 = q8_1_mmq_ds4_from_row_major_reference(xq, xscales)
  spec = describe_q4k_q8_1_mmq_tile(role="ds4_unit", m=m, n=n, k=k, m0=2, n0=3, m_tile=4, n_tile=5,
                                    activation_layout=Q8_1_MMQ_DS4_LAYOUT)
  row_spec = describe_q4k_q8_1_mmq_tile(role="row_unit", m=m, n=n, k=k, m0=2, n0=3, m_tile=4, n_tile=5)

  got = q4k_q8_1_mmq_ds4_tile_reference(raw, q8_ds4, spec)
  ref = q4k_q8_1_mmq_tile_reference(raw, xq, xscales, row_spec)

  np.testing.assert_allclose(got, ref, rtol=2e-6, atol=2e-5)
  assert spec.ds4_activation_spec.to_json()["layout"] == Q8_1_MMQ_DS4_LAYOUT


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


def test_q4k_q8_1_mmq_ds4_k_split_tiles_sum_to_full_tile():
  m, n, k = 6, 7, 512
  raw = _finite_q4k_bytes(n, k, seed=2234)
  xq, xscales = _q8_inputs(m, k, seed=2235)
  q8_ds4 = q8_1_mmq_ds4_from_row_major_reference(xq, xscales)
  full = describe_q4k_q8_1_mmq_tile(role="ds4_split", m=m, n=n, k=k, m_tile=m, n_tile=n,
                                    activation_layout=Q8_1_MMQ_DS4_LAYOUT)
  first = describe_q4k_q8_1_mmq_tile(role="ds4_split", m=m, n=n, k=k, m_tile=m, n_tile=n, k0=0, k_groups=8,
                                     activation_layout=Q8_1_MMQ_DS4_LAYOUT)
  second = describe_q4k_q8_1_mmq_tile(role="ds4_split", m=m, n=n, k=k, m_tile=m, n_tile=n, k0=256, k_groups=8,
                                      activation_layout=Q8_1_MMQ_DS4_LAYOUT)

  got = q4k_q8_1_mmq_ds4_tile_reference(raw, q8_ds4, first) + q4k_q8_1_mmq_ds4_tile_reference(raw, q8_ds4, second)
  ref = q4k_q8_1_mmq_ds4_tile_reference(raw, q8_ds4, full)

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


def test_q4k_q8_1_mmq_ds4_tile_reference_tracks_q4k_scale_min_metadata():
  m, n, k = 2, 3, 256
  raw = _finite_q4k_bytes(n, k, seed=2415)
  changed = raw.copy()
  changed[:, :, 0:2] = np.array([0.125], dtype=np.float16).view(np.uint8)
  changed[:, :, 2:4] = np.array([-0.0625], dtype=np.float16).view(np.uint8)
  xq, xscales = _q8_inputs(m, k, seed=2416)
  q8_ds4 = q8_1_mmq_ds4_from_row_major_reference(xq, xscales)
  spec = describe_q4k_q8_1_mmq_tile(role="ds4_scale_min", m=m, n=n, k=k, m_tile=m, n_tile=n,
                                    activation_layout=Q8_1_MMQ_DS4_LAYOUT)
  row_spec = describe_q4k_q8_1_mmq_tile(role="row_scale_min", m=m, n=n, k=k, m_tile=m, n_tile=n)

  got = q4k_q8_1_mmq_ds4_tile_reference(changed, q8_ds4, spec)
  ref = q4k_q8_1_mmq_tile_reference(changed, xq, xscales, row_spec)
  original = q4k_q8_1_mmq_ds4_tile_reference(raw, q8_ds4, spec)

  np.testing.assert_allclose(got, ref, rtol=2e-6, atol=2e-5)
  assert not np.allclose(got, original, rtol=2e-6, atol=2e-5)


def test_q4k_q8_1_mmq_ds4_tile_reference_uses_precomputed_sums_for_min_correction():
  m, n, k = 3, 4, 256
  raw = _finite_q4k_bytes(n, k, seed=2515)
  xq, xscales = _q8_inputs(m, k, seed=2516)
  q8_ds4 = q8_1_mmq_ds4_from_row_major_reference(xq, xscales)
  bad_ds4 = type(q8_ds4)(values=q8_ds4.values, scales=q8_ds4.scales, sums=q8_ds4.sums + np.float32(7.0), spec=q8_ds4.spec)
  spec = describe_q4k_q8_1_mmq_tile(role="ds4_sum_sensitive", m=m, n=n, k=k, m_tile=m, n_tile=n,
                                    activation_layout=Q8_1_MMQ_DS4_LAYOUT)

  got = q4k_q8_1_mmq_ds4_tile_reference(raw, bad_ds4, spec)
  ref = q4k_q8_1_mmq_ds4_tile_reference(raw, q8_ds4, spec)

  assert not np.allclose(got, ref, rtol=2e-6, atol=2e-5)


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
