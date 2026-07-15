import numpy as np
import pytest

from tinygrad import Tensor, dtypes

from extra.qk.layout import Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, Q8_1_BLOCK_ELEMS, q8_1_quantize
from extra.qk.mmq_lifecycle import COUNTER_NAMES
from extra.qk.mmq_q4k_q8_atom import (
  AMD_BACKEND_ATOM_ID, AMD_DS4_DOT4X4_BACKEND_ATOM_ID, AMD_DS4_WARP_BACKEND_ATOM_ID, AMD_STAGED_DS4_BACKEND_ATOM_ID,
  AMD_DS4_COOP_TILE_BACKEND_ATOM_ID, AMD_DS4_COOP_TILE_BLOCKER, AMD_DS4_LDS_SKELETON_BACKEND_ATOM_ID,
  AMD_WARP_BACKEND_ATOM_ID, BACKEND_ATOM_ID, q8_1_mmq_ds4_from_row_major,
  run_q4k_q8_1_mmq_bounded_amd_ds4_dot4x4,
  run_q4k_q8_1_mmq_bounded_amd_ds4_warp, run_q4k_q8_1_mmq_staged_ds4_atom, amd_atom_source_hash,
  amd_ds4_coop_tile_atom_source_hash, amd_ds4_dot4x4_atom_source_hash, amd_ds4_lds_skeleton_atom_source_hash,
  amd_ds4_warp_atom_source_hash,
  amd_warp_atom_source_hash, staged_ds4_atom_source_hash, run_q4k_q8_1_mmq_tile, run_q4k_q8_1_mmq_tile_amd,
  run_q4k_q8_1_mmq_bounded_amd_ds4_lds_skeleton, run_q4k_q8_1_mmq_tile_amd_warp,
  run_q4k_q8_1_mmq_bounded_amd_ds4_coop_tile, run_q4k_q8_1_mmq_tile_with_lifecycle,
)
from extra.qk.mmq_q4k_q8_reference import (
  Q8_1_MMQ_DS4_LAYOUT, describe_q4k_q8_1_mmq_tile, q4k_q8_1_mmq_ds4_tile_reference,
  q4k_q8_1_mmq_tile_reference,
)


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


def test_q4k_q8_1_mmq_amd_ds4_dot4x4_atom_matches_reference_when_amd_available():
  if not _has_amd():
    pytest.skip("AMD device is not available")
  m, n, k = 8, 7, 256
  raw = _finite_q4k_bytes(n, k, seed=703)
  xq, xscales = _q8_inputs(m, k, seed=704)
  ds4 = q8_1_mmq_ds4_from_row_major(xq, xscales)
  result = run_q4k_q8_1_mmq_bounded_amd_ds4_dot4x4(raw, ds4, role="ffn_gate_up")
  spec = describe_q4k_q8_1_mmq_tile(role="ffn_gate_up", m=m, n=n, k=k, m_tile=m, n_tile=n, k_groups=8)
  ref = q4k_q8_1_mmq_tile_reference(raw, xq, xscales, spec)

  assert result.backend_atom_id == AMD_DS4_DOT4X4_BACKEND_ATOM_ID
  assert result.lifecycle_detail["backend_stage"] == "amd_ds4_dot4x4_direct_gpu"
  assert result.lifecycle_detail["uses_precomputed_activation_sums"] is True
  assert result.lifecycle_detail["shared_memory_staging"] is False
  assert amd_ds4_dot4x4_atom_source_hash(m, n, k, "ffn_gate_up")
  np.testing.assert_allclose(result.output, ref, rtol=1e-6, atol=8e-4)


def test_q4k_q8_1_mmq_amd_ds4_warp_atom_matches_reference_when_amd_available():
  if not _has_amd():
    pytest.skip("AMD device is not available")
  m, n, k = 4, 5, 256
  raw = _finite_q4k_bytes(n, k, seed=803)
  xq, xscales = _q8_inputs(m, k, seed=804)
  ds4 = q8_1_mmq_ds4_from_row_major(xq, xscales)
  result = run_q4k_q8_1_mmq_bounded_amd_ds4_warp(raw, ds4, role="ffn_gate_up")
  spec = describe_q4k_q8_1_mmq_tile(role="ffn_gate_up", m=m, n=n, k=k, m_tile=m, n_tile=n, k_groups=8)
  ref = q4k_q8_1_mmq_tile_reference(raw, xq, xscales, spec)

  assert result.backend_atom_id == AMD_DS4_WARP_BACKEND_ATOM_ID
  assert result.lifecycle_detail["backend_stage"] == "amd_ds4_warp_direct_gpu"
  assert result.lifecycle_detail["uses_precomputed_activation_sums"] is True
  assert result.lifecycle_detail["shared_memory_staging"] is False
  assert amd_ds4_warp_atom_source_hash(m, n, k, "ffn_gate_up")
  np.testing.assert_allclose(result.output, ref, rtol=1e-6, atol=1e-3)


def test_q4k_q8_1_mmq_amd_ds4_lds_skeleton_hash_is_real_local_barrier_uop():
  h = amd_ds4_lds_skeleton_atom_source_hash(4, 5, 256, "ffn_gate_up")

  assert h


def test_q4k_q8_1_mmq_amd_ds4_coop_tile_hash_exists():
  h = amd_ds4_coop_tile_atom_source_hash(16, 16, 256, "ffn_gate_up")

  assert h
  assert AMD_DS4_COOP_TILE_BACKEND_ATOM_ID == "q4k_q8_1_mmq_amd_ds4_coop_tile_atom_v0"


def test_q4k_q8_1_mmq_coop_writeback_is_explicitly_owner_only():
  from extra.qk.mmq_q4k_q8_atom import _q4k_q8_1_bounded_ds4_coop_tile_kernel
  from tinygrad.uop.ops import UOp

  fn = _q4k_q8_1_bounded_ds4_coop_tile_kernel(16, 16, 256, "test", "gated_matrix_v0")
  body = repr(fn(UOp.placeholder((16, 16), dtypes.float32, 0),
                 UOp.placeholder((16 * 1 * 36, ), dtypes.uint32, 1),
                 UOp.placeholder((16 * 256, ), dtypes.int8, 2),
                 UOp.placeholder((16 * 2, ), dtypes.float32, 3),
                 UOp.placeholder((16 * 2, ), dtypes.float32, 4)))
  assert "gate" in body
  assert "lidx0" in body
  assert "gidx0" in body and "gidx1" in body


def test_q4k_q8_1_mmq_direct_owner_stages_reduction_before_lane_gate():
  from extra.qk.mmq_q4k_q8_atom import _q4k_q8_1_bounded_ds4_coop_tile_kernel
  from tinygrad.uop.ops import UOp

  fn = _q4k_q8_1_bounded_ds4_coop_tile_kernel(16, 16, 256, "test", "direct_owner_v0")
  body = repr(fn(UOp.placeholder((16, 16), dtypes.float32, 0),
                 UOp.placeholder((16 * 1 * 36,), dtypes.uint32, 1),
                 UOp.placeholder((16 * 256,), dtypes.int8, 2),
                 UOp.placeholder((16 * 2,), dtypes.float32, 3),
                 UOp.placeholder((16 * 2,), dtypes.float32, 4)))
  assert "lidx0" in body
  assert "ds_bpermute" in body
  assert "90" in body  # staged REG slot used before the divergent owner gate


def test_q4k_q8_1_mmq_coop_direct_owner_emits_one_dynamic_global_store():
  from extra.qk.mmq_q4k_q8_atom import _q4k_q8_1_bounded_ds4_coop_tile_kernel
  from tinygrad.uop.ops import Ops, UOp

  fn = _q4k_q8_1_bounded_ds4_coop_tile_kernel(16, 16, 256, "test", "direct_owner_v0")
  body = fn(UOp.placeholder((16, 16), dtypes.float32, 0),
            UOp.placeholder((16 * 1 * 36,), dtypes.uint32, 1),
            UOp.placeholder((16 * 256,), dtypes.int8, 2),
            UOp.placeholder((16 * 2,), dtypes.float32, 3),
            UOp.placeholder((16 * 2,), dtypes.float32, 4))
  stores = []
  seen = set()
  def visit(u):
    if id(u) in seen: return
    seen.add(id(u))
    if u.op is Ops.STORE: stores.append(u)
    for child in u.src: visit(child)
  visit(body)
  # The direct owner path must not carry the gated matrix's 256 candidate
  # global-store instructions.
  assert len(stores) < 16


@pytest.mark.parametrize("writeback_mode", ("gated_matrix_v0", "direct_owner_v0"))
def test_q4k_q8_1_mmq_amd_ds4_coop_tile_matches_reference_when_amd_available(writeback_mode):
  if not _has_amd():
    pytest.skip("AMD device is not available")
  m = n = 16
  k = 256
  raw = _finite_q4k_bytes(n, k, seed=1003)
  xq, xscales = _q8_inputs(m, k, seed=1004)
  ds4 = q8_1_mmq_ds4_from_row_major(xq, xscales)
  spec = describe_q4k_q8_1_mmq_tile(
    role="ffn_gate_up", m=m, n=n, k=k, m_tile=m, n_tile=n, k_groups=8, activation_layout=Q8_1_MMQ_DS4_LAYOUT)
  ref = q4k_q8_1_mmq_ds4_tile_reference(raw, ds4, spec)

  result = run_q4k_q8_1_mmq_bounded_amd_ds4_coop_tile(
    raw, ds4, role="ffn_gate_up", writeback_mode=writeback_mode)

  assert result.backend_atom_id == AMD_DS4_COOP_TILE_BACKEND_ATOM_ID
  assert result.lifecycle_detail["shared_memory_staging"] is True
  assert result.lifecycle_detail["bounded_only"] is True
  assert result.lifecycle_detail["production_dispatch_changed"] is False
  assert result.lifecycle_detail["store_owner_metadata"] is False
  assert result.lifecycle_detail["store_owner_count"] == 0
  assert result.lifecycle_detail["writeback_mode"] == writeback_mode
  assert result.lifecycle_detail["store_owner_proof"] == "separate_r4_lowered_isa_trace"
  assert result.lifecycle_detail["default_route"] == "direct_packed"
  assert "R4 owner proof remains separate" in AMD_DS4_COOP_TILE_BLOCKER
  np.testing.assert_allclose(result.output, ref, rtol=1e-6, atol=1e-3)


def test_q4k_q8_1_mmq_amd_ds4_lds_skeleton_matches_reference_when_amd_available():
  if not _has_amd():
    pytest.skip("AMD device is not available")
  m, n, k = 4, 5, 256
  raw = _finite_q4k_bytes(n, k, seed=903)
  xq, xscales = _q8_inputs(m, k, seed=904)
  ds4 = q8_1_mmq_ds4_from_row_major(xq, xscales)
  result = run_q4k_q8_1_mmq_bounded_amd_ds4_lds_skeleton(raw, ds4, role="ffn_gate_up")
  spec = describe_q4k_q8_1_mmq_tile(role="ffn_gate_up", m=m, n=n, k=k, m_tile=m, n_tile=n, k_groups=8)
  ref = q4k_q8_1_mmq_tile_reference(raw, xq, xscales, spec)

  assert result.backend_atom_id == AMD_DS4_LDS_SKELETON_BACKEND_ATOM_ID
  assert result.lifecycle_detail["backend_stage"] == "amd_ds4_lds_skeleton_gpu"
  assert result.lifecycle_detail["shared_memory_staging"] is True
  assert result.lifecycle_detail["bounded_only"] is True
  assert result.lifecycle_detail["promotion_eligible"] is False
  assert result.lifecycle_detail["production_dispatch_changed"] is False
  assert result.lifecycle_detail["default_route"] == "direct_packed"
  assert result.lifecycle_detail["global_activation_ds4_loads"] == m * k
  assert result.lifecycle_detail["local_activation_q8_stores"] == m * k
  assert result.lifecycle_detail["local_activation_q8_loads"] == m * n * k
  assert result.lifecycle_detail["global_q4k_tile_loads"] == m * n * (k // Q4_K_BLOCK_ELEMS)
  assert result.lifecycle.counters["barriers"] > 0
  assert result.lifecycle.counters["output_stores"] == m * n
  np.testing.assert_allclose(result.output, ref, rtol=1e-6, atol=1e-3)


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
