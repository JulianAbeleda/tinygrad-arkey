import numpy as np
import pytest

from tinygrad import Tensor, dtypes

from extra.qk.layout import Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, Q8_1_BLOCK_ELEMS, q8_1_quantize
from extra.qk.mmq_lifecycle import COUNTER_NAMES
from extra.qk.mmq_q4k_q8_atom import (
  AMD_BACKEND_ATOM_ID, AMD_STAGED_DS4_BACKEND_ATOM_ID, AMD_WARP_BACKEND_ATOM_ID, BACKEND_ATOM_ID,
  q8_1_mmq_ds4_from_row_major, run_q4k_q8_1_mmq_staged_ds4_atom, amd_atom_source_hash,
  amd_warp_atom_source_hash, staged_ds4_atom_source_hash, run_q4k_q8_1_mmq_tile,
  run_q4k_q8_1_mmq_tile_amd, run_q4k_q8_1_mmq_tile_amd_warp,
  run_q4k_q8_1_mmq_tile_with_lifecycle,
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


def test_q4k_q8_1_mmq_staged_ds4_atom_matches_reference_and_uses_sums():
  m, n, k = 4, 5, 256
  raw = _finite_q4k_bytes(n, k, seed=503)
  xq, xscales = _q8_inputs(m, k, seed=504)
  ds4 = q8_1_mmq_ds4_from_row_major(xq, xscales)
  spec = describe_q4k_q8_1_mmq_tile(role="ffn_gate_up", m=m, n=n, k=k, m_tile=m, n_tile=n, k_groups=8)

  result = run_q4k_q8_1_mmq_staged_ds4_atom(raw, ds4, spec)
  ref = q4k_q8_1_mmq_tile_reference(raw, xq, xscales, spec)

  assert result.backend_atom_id == AMD_STAGED_DS4_BACKEND_ATOM_ID
  assert result.lifecycle_detail["backend_stage"] == "reference_backed_staged_ds4_probe"
  assert result.lifecycle_detail["promotion_claim"] is False
  assert result.lifecycle_detail["uses_precomputed_activation_sums"] is True
  assert staged_ds4_atom_source_hash(spec)
  np.testing.assert_allclose(result.output, ref, rtol=0, atol=8e-4)

  changed_sums = q8_1_mmq_ds4_from_row_major(xq, xscales)
  changed_sums = type(changed_sums)(values=changed_sums.values, scales=changed_sums.scales,
                                    sums=changed_sums.sums + 1.0, spec=changed_sums.spec)
  changed = run_q4k_q8_1_mmq_staged_ds4_atom(raw, changed_sums, spec).output
  assert not np.allclose(changed, result.output, rtol=0, atol=1e-6)


def test_q4k_q8_1_mmq_staged_ds4_lifecycle_distinguishes_stages():
  m, n, k = 4, 4, 256
  raw = _finite_q4k_bytes(n, k, seed=603)
  xq, xscales = _q8_inputs(m, k, seed=604)
  spec = describe_q4k_q8_1_mmq_tile(role="ffn_gate_up", m=m, n=n, k=k, m_tile=m, n_tile=n, k_groups=8)

  row = run_q4k_q8_1_mmq_staged_ds4_atom(raw, q8_1_mmq_ds4_from_row_major(xq, xscales), spec).to_json()

  counters = row["lifecycle"]["counters"]
  detail = row["lifecycle_detail"]
  assert counters["activation_q8_1_reads"] > 0
  assert counters["packed_weight_global_loads"] > 0
  assert counters["scale_min_metadata_loads"] > 0
  assert counters["barriers"] > 0
  assert counters["dot_accumulation_epochs"] > 0
  assert counters["output_stores"] == m * n
  assert detail["global_activation_ds4_loads"] == detail["staged_activation_tile_loads"]
  assert detail["global_q4k_tile_loads"] == detail["staged_q4k_tile_loads"]


def _has_amd() -> bool:
  try:
    Tensor([1.0], device="AMD").realize().numpy()
    return True
  except Exception:
    return False


def test_q4k_q8_1_mmq_amd_atom_matches_reference_when_amd_available():
  if not _has_amd():
    pytest.skip("AMD device is not available")
  m, n, k = 4, 4, 256
  raw = _finite_q4k_bytes(n, k, seed=303)
  xq, xscales = _q8_inputs(m, k, seed=304)
  spec = describe_q4k_q8_1_mmq_tile(role="ffn_gate_up", m=m, n=n, k=k, m_tile=m, n_tile=n, k_groups=8)

  result = run_q4k_q8_1_mmq_tile_amd(raw, xq, xscales, spec)
  ref = q4k_q8_1_mmq_tile_reference(raw, xq, xscales, spec)

  assert result.backend_atom_id == AMD_BACKEND_ATOM_ID
  assert amd_atom_source_hash(spec)
  np.testing.assert_allclose(result.output, ref, rtol=1e-6, atol=5e-4)


def test_q4k_q8_1_mmq_amd_warp_atom_matches_reference_when_amd_available():
  if not _has_amd():
    pytest.skip("AMD device is not available")
  m, n, k = 4, 4, 256
  raw = _finite_q4k_bytes(n, k, seed=403)
  xq, xscales = _q8_inputs(m, k, seed=404)
  spec = describe_q4k_q8_1_mmq_tile(role="ffn_gate_up", m=m, n=n, k=k, m_tile=m, n_tile=n, k_groups=8)

  result = run_q4k_q8_1_mmq_tile_amd_warp(raw, xq, xscales, spec)
  ref = q4k_q8_1_mmq_tile_reference(raw, xq, xscales, spec)

  assert result.backend_atom_id == AMD_WARP_BACKEND_ATOM_ID
  assert amd_warp_atom_source_hash(spec)
  np.testing.assert_allclose(result.output, ref, rtol=1e-6, atol=5e-4)
