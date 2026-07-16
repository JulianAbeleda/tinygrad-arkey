import pytest

from tinygrad import dtypes
from tinygrad.codegen.opt.kernel_writeback import (WMMAIDsReady, WMMAWritebackDescriptor, WMMAWritebackLayout,
  WMMAWritebackProof, WMMAWritebackTileMapping, build_wmma_writeback)
from tinygrad.codegen.opt.tc import amd_rdna3
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import (AxisType, KernelLDSArenaRegion, KernelLDSRecordComponent, KernelLDSRecordLayout,
  KernelLDSWindow, KernelTileGeometry, Ops, UOp)


def _tc(): return next(x for x in amd_rdna3 if x.dtype_in == dtypes.char and x.dtype_out == dtypes.int)


def _geometry(ids_dtype=dtypes.int, ids_rows=128, waves=(8, 1)):
  ids = KernelLDSArenaRegion("ids", 0, ids_rows*4, 16, KernelLDSRecordLayout(ids_rows, 4,
    (KernelLDSRecordComponent("value", ids_dtype, 0, 4, 4),)))
  a0 = ids.end
  return KernelTileGeometry((128, 128, 32), waves, waves[0]*waves[1]*32, 32,
    (KernelLDSWindow("A", a0, a0+8192, 64), KernelLDSWindow("B", a0+8192, a0+16384, 64)), (),
    (ids, KernelLDSArenaRegion("A", a0, a0+8192), KernelLDSArenaRegion("B", a0+8192, a0+16384)))


def _fixture(geometry=None, *, dtype=dtypes.float, count=8):
  geometry = geometry or _geometry()
  proof = WMMAWritebackProof.prove(WMMAWritebackDescriptor(geometry, _tc(), dtype, count,
    WMMAWritebackLayout("col", "row", 128), "ids"))
  arena = UOp.placeholder((geometry.lds_bytes,), dtypes.uint8, 700, addrspace=AddrSpace.LOCAL)
  init = arena.index(UOp.const(dtypes.weakint, 0), dtype=dtypes.int).store(UOp.const(dtypes.int, 0))
  ids = WMMAIDsReady(arena, UOp.barrier(init), "ids")
  dst = UOp.placeholder((16384,), dtypes.float, 701)
  acc = tuple(UOp.const(dtypes.float.vec(8), float(i)) for i in range(count))
  axes = (UOp.range(8, 710, AxisType.LOCAL), UOp.range(1, 711, AxisType.LOCAL), UOp.range(32, 712, AxisType.LOCAL))
  return proof, ids, dst, acc, axes


def test_tile128x128_ids_writeback_has_exact_owner_proof_and_ready_loads():
  proof, ids, dst, acc, axes = _fixture()
  assert proof.descriptor.tc.dtype_out == dtypes.int and proof.descriptor.accumulator_dtype == dtypes.float
  assert proof.owner_count == 16_384 and len(proof.coordinates) == 16_384
  out = build_wmma_writeback(proof, destination=dst, accumulators=acc,
                             wave_m=axes[0], wave_n=axes[1], lane=axes[2], ids=ids)
  assert len(out.stores) == 64
  loads = [u for u in out.sink.toposort() if u.op is Ops.LOAD]
  assert len(loads) == 64
  assert all(ids.ready in u.backward_slice_with_self for u in loads)
  assert all(len(load.src) == 1 for load in loads)
  assert all(len(store.src) == 2 for store in out.stores)
  assert all(s.src[1].dtype == dtypes.float for s in out.stores)


def test_layout_selects_identified_axis_without_q4_vocabulary():
  geometry = _geometry()
  desc = WMMAWritebackDescriptor(geometry, _tc(), dtypes.float, 8, WMMAWritebackLayout("row", "col", 256), "ids")
  base = _fixture()[1:]
  out = build_wmma_writeback(WMMAWritebackProof.prove(desc), destination=base[1], accumulators=base[2],
                             wave_m=base[3][0], wave_n=base[3][1], lane=base[3][2], ids=base[0])
  assert all(s.tag[-1] == "row" for s in out.stores)


def test_fail_closed_region_readiness_and_accumulator_drift():
  proof, ids, dst, acc, axes = _fixture()
  kwargs = dict(destination=dst, accumulators=acc, wave_m=axes[0], wave_n=axes[1], lane=axes[2], ids=ids)
  detached = WMMAIDsReady(ids.allocation, UOp.barrier(UOp(Ops.NOOP, dtypes.void)), "ids")
  with pytest.raises(ValueError, match="detached"): build_wmma_writeback(proof, **{**kwargs, "ids":detached})
  with pytest.raises(ValueError, match="missing"): build_wmma_writeback(proof, **{**kwargs, "ids":None})
  with pytest.raises(ValueError, match="dtype/count"): build_wmma_writeback(proof, **{**kwargs, "accumulators":acc[:-1]})
  with pytest.raises(ValueError, match="dtype/count"):
    build_wmma_writeback(proof, **{**kwargs, "accumulators":tuple(UOp.const(dtypes.int.vec(8), 0) for _ in range(8))})


def test_fail_closed_wrong_region_dtype_size_and_wave_geometry():
  for geometry in (_geometry(dtypes.uint, 128), _geometry(dtypes.int, 64)):
    with pytest.raises(ValueError, match="dtype or size"):
      proof, ids, dst, acc, axes = _fixture(geometry)
      build_wmma_writeback(proof, destination=dst, accumulators=acc, wave_m=axes[0], wave_n=axes[1], lane=axes[2], ids=ids)
  with pytest.raises(ValueError, match="accumulator count"):
    WMMAWritebackDescriptor(_geometry(), _tc(), dtypes.float, 7, WMMAWritebackLayout("col", "row", 128), "ids")
  proof, ids, dst, acc, axes = _fixture()
  with pytest.raises(ValueError, match="wrong wave geometry"):
    build_wmma_writeback(proof, destination=dst, accumulators=acc, wave_m=UOp.range(4, 713, AxisType.LOCAL),
                         wave_n=axes[1], lane=axes[2], ids=ids)


def test_runtime_edge_mapping_gates_ids_loads_and_stores_and_maps_destination():
  geometry = _geometry()
  desc = WMMAWritebackDescriptor(geometry, _tc(), dtypes.float, 8,
    WMMAWritebackLayout("col", "row", 257), "ids", False)
  proof, ids, dst, acc, axes = _fixture()
  proof = WMMAWritebackProof.prove(desc)
  mapping = WMMAWritebackTileMapping(UOp.variable("M", 1, 4096), UOp.variable("N", 1, 4096),
    UOp.variable("m_base", 0, 4095), UOp.variable("n_base", 0, 4095))
  out = build_wmma_writeback(proof, destination=dst, accumulators=acc,
    wave_m=axes[0], wave_n=axes[1], lane=axes[2], ids=ids, mapping=mapping)
  assert len(out.stores) == 64 and all(len(store.src) == 3 and store.src[2].dtype == dtypes.bool for store in out.stores)
  loads = [u for u in out.sink.toposort() if u.op is Ops.LOAD]
  assert len(loads) == 64 and all(len(load.src) == 3 and load.src[2].dtype == dtypes.bool for load in loads)
  # Both destination origins participate in every address, including when the N/col coordinate is IDs-remapped.
  for store in out.stores:
    address_slice = store.src[0].src[1].backward_slice_with_self
    assert mapping.m_offset in address_slice and mapping.n_offset in address_slice


def test_edge_predicates_select_exactly_once_subset_of_owner_proof():
  desc = WMMAWritebackDescriptor(_geometry(), _tc(), dtypes.float, 8,
    WMMAWritebackLayout("col", "row", 257), "ids", False)
  proof = WMMAWritebackProof.prove(desc)
  m_extent, n_extent, m_offset, n_offset = 257, 259, 256, 256
  valid = [(r, c) for r, c in proof.coordinates if m_offset+r < m_extent and n_offset+c < n_extent]
  assert len(valid) == 3 and len(valid) == len(set(valid))
  assert set(valid) == {(0, 0), (0, 1), (0, 2)}


def test_edge_mapping_is_required_only_for_non_exact_tiles():
  edge = WMMAWritebackDescriptor(_geometry(), _tc(), dtypes.float, 8,
    WMMAWritebackLayout("col", "row", 128), "ids", False)
  proof, ids, dst, acc, axes = _fixture()
  kwargs = dict(destination=dst, accumulators=acc, wave_m=axes[0], wave_n=axes[1], lane=axes[2], ids=ids)
  with pytest.raises(ValueError, match="requires runtime M/N"):
    build_wmma_writeback(WMMAWritebackProof.prove(edge), **kwargs)
  mapping = WMMAWritebackTileMapping(*(UOp.const(dtypes.int, x) for x in (128, 128, 0, 0)))
  with pytest.raises(ValueError, match="exact-tile"):
    build_wmma_writeback(proof, **kwargs, mapping=mapping)


@pytest.mark.parametrize("dtype", (dtypes.float.vec(2), dtypes.void, dtypes.float.ptr()))
def test_accumulator_output_dtype_must_be_explicit_numeric_scalar(dtype):
  with pytest.raises(ValueError, match="scalar numeric output"):
    WMMAWritebackDescriptor(_geometry(), _tc(), dtype, 8, WMMAWritebackLayout("col", "row", 128), "ids")
