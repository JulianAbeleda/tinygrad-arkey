import pytest

from tinygrad.codegen.opt.kernel_lds import (cooperative_lds_padding_offsets, cooperative_lds_stores, rdna3_wmma_output_coord,
                                             derive_precontract_factors, semantic_wave_coords, validate_rdna3_wmma_descriptor, wmma_fragment_loads,
                                             wmma_output_owners)
from tinygrad.codegen.opt.tc import amd_rdna3
from tinygrad import dtypes
from tinygrad.uop.ops import KernelLDSWindow, KernelTileGeometry


def _geometry():
  return KernelTileGeometry((128, 128, 32), (4, 2), 256, 32,
    (KernelLDSWindow("A", 0, 10240, 80), KernelLDSWindow("B", 10240, 20480, 80)))

def _tc(): return next(tc for tc in amd_rdna3 if tc.dtype_in == dtypes.half and tc.dtype_out == dtypes.float)


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


@pytest.mark.parametrize("role", ("A", "B"))
def test_every_fragment_load_is_staged_in_bounds_and_obeys_stride(role):
  geometry = _geometry()
  staged = {offset + byte for store in cooperative_lds_stores(geometry, role)
            for offset in (store.byte_offset,) for byte in range(store.vector_bytes)}
  window = next(w for w in geometry.lds_windows if w.role == role)
  loads = wmma_fragment_loads(geometry, role, tc=_tc())
  assert {load.k_substep for load in loads} == {0, 1}
  assert all(window.base <= load.byte_offset and load.byte_offset + 2 <= window.end for load in loads)
  assert all(load.byte_offset == window.base + load.logical_row * 80 + load.logical_k * 2 for load in loads)
  assert all(load.byte_offset in staged and load.byte_offset + 1 in staged for load in loads)


@pytest.mark.parametrize("role", ("A", "B"))
def test_upper_fragment_lanes_duplicate_lower_lane_addresses(role):
  loads = wmma_fragment_loads(_geometry(), role, tc=_tc())
  key = lambda x: (x.wave_m, x.wave_n, x.subtile, x.k_substep, x.element)
  by_thread = {(load.thread, key(load)): load.byte_offset for load in loads}
  for wave in range(8):
    for lane in range(16):
      lower, upper = wave * 32 + lane, wave * 32 + lane + 16
      lower_rows = {k: off for (thread, k), off in by_thread.items() if thread == lower}
      upper_rows = {k: off for (thread, k), off in by_thread.items() if thread == upper}
      assert lower_rows == upper_rows


def test_output_ownership_covers_128_square_exactly_once():
  owners = wmma_output_owners(_geometry(), tc=_tc())
  assert len(owners) == 128 * 128
  coords = [(owner.row, owner.col) for owner in owners]
  assert len(set(coords)) == len(coords)
  assert set(coords) == {(row, col) for row in range(128) for col in range(128)}
  assert {(o.subtile_m, o.subtile_n) for o in owners} == {(m, n) for m in range(2) for n in range(4)}


def test_logical_rdna3_formulas_match_core_tensor_descriptor_and_interpreter_map():
  tc = _tc()
  validate_rdna3_wmma_descriptor(tc)
  assert tc.dims == (16, 16, 16) and tc.threads == 32 and tc.elements_per_thread == (16, 16, 8)
  # This is the c_map formula in tinygrad.runtime.ops_python's AMD WMMA model.
  assert [rdna3_wmma_output_coord(lane, elem, tc=tc) for lane in range(32) for elem in range(8)] == [
    (lane % 16, lane // 16 + elem * 2) for lane in range(32) for elem in range(8)]
  assert len(set(rdna3_wmma_output_coord(lane, elem, tc=tc) for lane in range(32) for elem in range(8))) == 256


@pytest.mark.parametrize("lane,element", ((-1, 0), (32, 0), (0, -1), (0, 8), (True, 0)))
def test_rdna3_output_formula_rejects_bad_coordinates(lane, element):
  with pytest.raises(ValueError): rdna3_wmma_output_coord(lane, element, tc=_tc())


def test_fragment_and_output_mapping_fail_closed_on_unsupported_geometry():
  k24 = KernelTileGeometry((128, 128, 24), (4, 2), 256, 32,
    (KernelLDSWindow("A", 0, 10240, 80), KernelLDSWindow("B", 10240, 20480, 80)))
  with pytest.raises(ValueError, match="K divisible by 16"): wmma_fragment_loads(k24, "A", tc=_tc())
  wave64 = KernelTileGeometry((128, 128, 32), (2, 2), 256, 64,
    (KernelLDSWindow("A", 0, 10240, 80), KernelLDSWindow("B", 10240, 20480, 80)))
  with pytest.raises(ValueError, match="wave32"): wmma_fragment_loads(wave64, "B", tc=_tc())
  with pytest.raises(ValueError, match="wave32"): wmma_output_owners(wave64, tc=_tc())


class _DescriptorDrift:
  def __init__(self, base, field, value): self.base, self.field, self.value = base, field, value
  def __getattr__(self, name): return self.value if name == self.field else getattr(self.base, name)

class _RemapDrift:
  def __init__(self, base): self.base = base
  def __getattr__(self, name): return self if name == "lane_map" else getattr(self.base, name)
  def remaps(self): return [{"drift": "true"}, {"drift": "true"}]


@pytest.mark.parametrize("field,value", (
  ("dims", (16, 16, 8)), ("threads", 64), ("elements_per_thread", (16, 8, 8)),
  ("dtype_in", dtypes.bfloat16), ("dtype_out", dtypes.half),
  ("opts", ("l0",)),
  ("swizzle", (((), (), ()), ((), (), ()))),
))
def test_descriptor_fingerprint_drift_fails_closed(field, value):
  drift = _DescriptorDrift(_tc(), field, value)
  with pytest.raises(ValueError, match=field): validate_rdna3_wmma_descriptor(drift)
  with pytest.raises(ValueError, match=field): wmma_fragment_loads(_geometry(), "A", tc=drift)
  with pytest.raises(ValueError, match=field): wmma_output_owners(_geometry(), tc=drift)


def test_descriptor_remap_drift_and_missing_descriptor_fail_closed():
  with pytest.raises(ValueError, match="remaps drifted"): validate_rdna3_wmma_descriptor(_RemapDrift(_tc()))
  with pytest.raises(ValueError, match="dims drifted"): wmma_fragment_loads(_geometry(), "A", tc=object())
  with pytest.raises(ValueError, match="dims drifted"): rdna3_wmma_output_coord(0, 0, tc=None)


def test_precontract_factor_derivation_exact_anchor_and_legal_smaller_family():
  exact = derive_precontract_factors(_geometry(), _tc())
  assert (exact.subtiles_m, exact.subtiles_n, exact.waves_m, exact.waves_n, exact.k_substeps,
          exact.vectors_per_row, exact.loads_a, exact.loads_b) == (2, 4, 4, 2, 2, 4, 2, 2)
  smaller = KernelTileGeometry((64, 64, 32), (2, 2), 128, 32,
    (KernelLDSWindow("A", 0, 5120, 80), KernelLDSWindow("B", 5120, 10240, 80)))
  factors = derive_precontract_factors(smaller, _tc())
  assert (factors.subtiles_m, factors.subtiles_n, factors.k_substeps, factors.vectors_per_row,
          factors.loads_a, factors.loads_b) == (2, 2, 2, 4, 2, 2)


def test_precontract_factor_derivation_rejects_nondivisible_and_bad_windows():
  nondivisible = KernelTileGeometry((80, 64, 16), (2, 2), 128, 32,
    (KernelLDSWindow("A", 0, 3840, 48), KernelLDSWindow("B", 3840, 6912, 48)))
  with pytest.raises(ValueError, match="whole per-wave"): derive_precontract_factors(nondivisible, _tc())
  uneven = KernelTileGeometry((64, 64, 32), (4, 4), 512, 32,
    (KernelLDSWindow("A", 0, 5120, 80), KernelLDSWindow("B", 5120, 10240, 80)))
  with pytest.raises(ValueError, match="divide evenly"): derive_precontract_factors(uneven, _tc())
