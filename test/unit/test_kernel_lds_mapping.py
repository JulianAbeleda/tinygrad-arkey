import pytest

from tinygrad.codegen.opt.kernel_lds import cooperative_lds_padding_offsets, cooperative_lds_stores, semantic_wave_coords
from tinygrad.uop.ops import KernelLDSWindow, KernelTileGeometry


def _geometry():
  return KernelTileGeometry((128, 128, 32), (4, 2), 256, 32,
    (KernelLDSWindow("A", 0, 10240, 80), KernelLDSWindow("B", 10240, 20480, 80)))


@pytest.mark.parametrize("role,base,end", (("A", 0, 10240), ("B", 10240, 20480)))
def test_exact_cooperative_store_coverage_election_and_bounds(role, base, end):
  stores = cooperative_lds_stores(_geometry(), role)
  assert len(stores) == 512
  assert len({s.byte_offset for s in stores}) == 512
  assert all(base <= s.byte_offset and s.byte_offset + 16 <= end for s in stores)
  assert {s.thread for s in stores} == set(range(256))
  assert all(sum(s.thread == tid for s in stores) == 2 for tid in range(256))
  assert all(s.iteration in (0, 1) for s in stores)
  assert {(s.row, s.vector) for s in stores} == {(row, vec) for row in range(128) for vec in range(4)}


def test_a_b_windows_do_not_overlap_and_padding_has_no_store_owner():
  geometry = _geometry()
  a, b = cooperative_lds_stores(geometry, "A"), cooperative_lds_stores(geometry, "B")
  a_bytes = {s.byte_offset for s in a}
  b_bytes = {s.byte_offset for s in b}
  assert a_bytes.isdisjoint(b_bytes)
  for role, stores in (("A", a), ("B", b)):
    padding = cooperative_lds_padding_offsets(geometry, role)
    assert len(padding) == 128
    assert set(padding).isdisjoint(s.byte_offset for s in stores)
    window = next(w for w in geometry.lds_windows if w.role == role)
    assert all(window.base <= offset and offset + 16 <= window.end for offset in padding)
    assert {s.byte_offset for s in stores} | set(padding) == set(range(window.base, window.end, 16))


def test_store_election_matches_two_rows_per_thread():
  for role in ("A", "B"):
    by_thread = {tid: [] for tid in range(256)}
    for store in cooperative_lds_stores(_geometry(), role): by_thread[store.thread].append(store)
    for tid, stores in by_thread.items():
      assert [(s.row, s.vector) for s in stores] == [(tid // 4, tid % 4), (tid // 4 + 64, tid % 4)]


def test_semantic_wave_coordinates_cover_exact_4x2_topology():
  geometry = _geometry()
  coords = [semantic_wave_coords(geometry, tid) for tid in range(256)]
  assert {(m, n) for m, n, _lane in coords} == {(m, n) for m in range(4) for n in range(2)}
  assert all(sum((m, n) == (cm, cn) for m, n, _lane in coords) == 32 for cm in range(4) for cn in range(2))
  assert [semantic_wave_coords(geometry, tid) for tid in (0, 31, 32, 63, 255)] == [
    (0, 0, 0), (0, 0, 31), (0, 1, 0), (0, 1, 31), (3, 1, 31)]


@pytest.mark.parametrize("thread", (-1, 256, True, 1.5))
def test_semantic_wave_coordinates_reject_bad_threads(thread):
  with pytest.raises(ValueError, match="thread must be"): semantic_wave_coords(_geometry(), thread)


@pytest.mark.parametrize("role", ("a", "C", ""))
def test_mapping_rejects_unknown_roles(role):
  with pytest.raises(ValueError, match="role must be A or B"): cooperative_lds_stores(_geometry(), role)


def test_mapping_rejects_window_shape_vector_and_divisibility_errors():
  geometry = _geometry()
  bad_size = KernelTileGeometry((128, 128, 32), (4, 2), 256, 32,
    (KernelLDSWindow("A", 0, 10224, 80), KernelLDSWindow("B", 10224, 20464, 80)))
  with pytest.raises(ValueError, match="exactly equal"): cooperative_lds_stores(bad_size, "A")
  with pytest.raises(ValueError, match="divisible"): cooperative_lds_stores(geometry, "A", vector_bytes=24)
  with pytest.raises(ValueError, match="positive int"): cooperative_lds_stores(geometry, "A", element_bytes=0)
