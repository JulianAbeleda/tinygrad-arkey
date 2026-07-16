import pytest

from extra.qk.prefill_int8_wmma_spec import (
  Q4KInt8WMMATiledPrefillSpec,
  _prove_integrated_loop_dynamic_owner,
  emit_q4k_int8_wmma_tiled_scheduler_tensor,
)


def test_integrated_loop_rejects_full_role_fail_closed_before_operands():
  spec = Q4KInt8WMMATiledPrefillSpec(
    n=5120, k=17408, m=512, role="ffn_down",
    m_tile=16, n_tile=16, group_tile=8, implementation="integrated_loop")
  with pytest.raises(NotImplementedError, match="bounded packed-Q4 owner tile"):
    emit_q4k_int8_wmma_tiled_scheduler_tensor(None, None, None, spec)


def test_integrated_loop_dynamic_owner_proof_uses_packed_q4_decode_and_writeback():
  spec = Q4KInt8WMMATiledPrefillSpec(n=16, k=256, m=16, m_tile=16, n_tile=16,
                                     group_tile=1, implementation="integrated_loop")
  _prove_integrated_loop_dynamic_owner(spec)


def test_integrated_loop_rejects_unproven_group_tile():
  spec = Q4KInt8WMMATiledPrefillSpec(n=16, k=256, m=16, m_tile=16, n_tile=16,
                                     group_tile=8, implementation="integrated_loop")
  with pytest.raises(NotImplementedError, match="bounded packed-Q4 owner tile"):
    _prove_integrated_loop_dynamic_owner(spec)


def test_direct_tiled_implementation_remains_distinct_from_integrated_loop():
  direct = Q4KInt8WMMATiledPrefillSpec(n=16, k=256, m=16, implementation="direct_tiled_wmma_v0")
  integrated = Q4KInt8WMMATiledPrefillSpec(n=16, k=256, m=16, implementation="integrated_loop")
  assert direct.implementation != integrated.implementation
