import pytest

from extra.qk.mmq_q4_lifecycle import Q4MultiWaveLifecycleSpec, activation_reuse, describe, edge_predicate, staging_layout


def test_layout_offsets_are_aligned_nonoverlapping_and_lifetimes_are_explicit():
  spec = Q4MultiWaveLifecycleSpec(64, 32, 512, n_panels=2)
  regions = staging_layout(spec)
  assert [r.name for r in regions] == ["q4_weights", "q8_values", "q8_scales", "q8_sums"]
  assert all(r.offset % 256 == 0 for r in regions)
  assert all(a.offset + a.size <= b.offset for a, b in zip(regions, regions[1:]))
  assert all(r.lifetime == ("load_q4", "compute") or r.lifetime == ("load_activation", "compute") for r in regions)


def test_edges_are_tile_local_and_activation_is_reused_across_panels():
  spec = Q4MultiWaveLifecycleSpec(48, 20, 256, m0=32, n0=16, n_panels=3)
  assert edge_predicate(spec, 15, 3, 255)
  assert not edge_predicate(spec, 16, 0, 0)
  assert not edge_predicate(spec, 0, 4, 0)
  reuse = activation_reuse(spec)
  assert reuse["staged_once"] and reuse["reuse_across_n_panels"] == 3
  assert reuse["activation_load_epochs"] == 1


def test_description_is_mechanically_serializable_and_fail_closed_integration():
  payload = describe(Q4MultiWaveLifecycleSpec(32, 16, 256))
  assert payload["schema"] == "tinygrad.q4_mmq_multiwave_lifecycle.v1"
  assert payload["waves"] == {"count": 2, "width": 32, "m_rows_per_wave": 16}
  assert payload["barriers"][0]["uniform"] is True
  assert payload["integration"]["connected"] is False
  assert "no lifecycle-consumer interface" in payload["integration"]["exact_blocker"]


def test_rejects_unaligned_k_and_invalid_origin():
  with pytest.raises(ValueError, match="block aligned"):
    staging_layout(Q4MultiWaveLifecycleSpec(32, 16, 260))
  with pytest.raises(ValueError, match="outside"):
    staging_layout(Q4MultiWaveLifecycleSpec(32, 16, 256, m0=32))
