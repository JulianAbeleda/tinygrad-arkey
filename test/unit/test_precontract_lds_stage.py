import pytest

from tinygrad import dtypes
from tinygrad.codegen.opt.kernel_lds import PrecontractOperandTemplate, build_precontract_lds_stage
from tinygrad.codegen.opt.tc import amd_rdna3
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelLDSWindow, KernelTileGeometry, Ops, UOp


def _geometry(): return KernelTileGeometry((128, 128, 32), (4, 2), 256, 32,
  (KernelLDSWindow("A", 0, 10240, 80), KernelLDSWindow("B", 10240, 20480, 80)))
def _tc(): return next(tc for tc in amd_rdna3 if tc.dtype_in == dtypes.half and tc.dtype_out == dtypes.float)

def _inputs():
  row_a, row_b = UOp.range(128, 20, AxisType.LOOP), UOp.range(128, 21, AxisType.LOOP)
  ka, kb = UOp.range(4096, 22, AxisType.REDUCE), UOp.range(4096, 23, AxisType.REDUCE)
  a, b = UOp.param(0, dtypes.half.ptr(128*4096)), UOp.param(1, dtypes.half.ptr(128*4096))
  sa, sb = a.index(row_a*4096+ka).load(), b.index(row_b*4096+kb).load()
  return (PrecontractOperandTemplate("A", sa, row_a, ka), PrecontractOperandTemplate("B", sb, row_b, kb))

def _stage(operands=None, end_ranges=()):
  return build_precontract_lds_stage(_geometry(), tc=_tc(), operands=_inputs() if operands is None else operands,
    thread=UOp.special(256, "lidx0"), k_tile_base=UOp.const(dtypes.weakint, 0), k_substep=UOp.range(2, 24, AxisType.REDUCE),
    subtile_m=UOp.range(2, 25, AxisType.UPCAST), subtile_n=UOp.range(4, 26, AxisType.UPCAST),
    fragment_axis=UOp.range(16, 27, AxisType.UPCAST), end_ranges=end_ranges)

def test_precontract_stage_has_one_allocation_one_barrier_and_paired_contracts():
  stage = _stage()
  assert stage.allocation.op is Ops.DEFINE_LOCAL and stage.allocation.ptrdtype.addrspace is AddrSpace.LOCAL
  assert stage.allocation.dtype.size * stage.allocation.dtype.base.itemsize == 20480
  assert stage.barrier.op is Ops.BARRIER and stage.barrier.src == (stage.producer,)
  assert stage.fragment_a.op is stage.fragment_b.op is Ops.CONTRACT
  assert stage.fragment_a.tag == ("kernel_tile_fragment", "A")
  assert stage.fragment_b.tag == ("kernel_tile_fragment", "B")
  assert {u for u in UOp.sink(stage.fragment_a, stage.fragment_b).backward_slice if u is stage.allocation} == {stage.allocation}

def test_scalar_producers_cover_only_data_intervals_and_share_barrier_dependency():
  stage = _stage()
  stores = [u for u in stage.producer.backward_slice_with_self if u.op is Ops.STORE]
  assert len(stores) == 32
  assert all(stage.barrier in load.backward_slice for load in
             UOp.sink(stage.fragment_a, stage.fragment_b).backward_slice if load.op is Ops.LOAD and stage.allocation in load.backward_slice)
  indices = [store.src[0].src[1] for store in stores]
  thread = next(u for u in stage.producer.backward_slice if u.op is Ops.SPECIAL and u.arg == "lidx0")
  for tid in range(256):
    actual = {index.substitute({thread: UOp.const(dtypes.weakint, tid)}).simplify().arg for index in indices}
    row, vector = tid // 4, tid % 4
    expected = {(base//2) + r*40 + vector*8 + elem for base in (0, 10240)
                for r in (row, row+64) for elem in range(8)}
    assert actual == expected
    assert all((index-(0 if index < 5120 else 5120)) % 40 < 32 for index in actual)

def test_fragment_consumers_cover_all_sixteen_elements_for_both_roles():
  stage = _stage()
  assert stage.fragment_a.dtype == stage.fragment_b.dtype == dtypes.half.vec(16)
  assert stage.fragment_a.arg == stage.fragment_b.arg == ((27, 16),)
  for fragment in (stage.fragment_a, stage.fragment_b):
    fragment_axis = next(r for r in fragment.backward_slice if r.op is Ops.RANGE and r.arg[0] == 27)
    assert fragment_axis.vmax + 1 == 16

def test_requested_end_ranges_are_preserved_on_producer():
  outer = UOp.range(3, 30, AxisType.LOOP)
  stage = _stage(end_ranges=(outer,))
  assert stage.producer.op is Ops.END and stage.producer.src[1:] == (outer,)

def test_precontract_stage_fails_closed_on_role_and_axis_mutation():
  a, b = _inputs()
  with pytest.raises(ValueError, match="ordered A and B"): _stage((b, a))
  bad = PrecontractOperandTemplate("A", a.source, UOp.const(dtypes.int, 0), a.k_axis)
  with pytest.raises(ValueError, match="axes must be RANGE"): _stage((bad, b))
  detached = PrecontractOperandTemplate("A", UOp.const(dtypes.half, 0), a.row_axis, a.k_axis)
  with pytest.raises(ValueError, match="does not retain"): _stage((detached, b))
  with pytest.raises(ValueError, match="K/subtile axes"):
    build_precontract_lds_stage(_geometry(), tc=_tc(), operands=(a, b), thread=UOp.special(256, "lidx0"),
      k_tile_base=UOp.const(dtypes.weakint, 0), k_substep=UOp.range(4, 24, AxisType.REDUCE), subtile_m=UOp.range(2, 25, AxisType.UPCAST),
      subtile_n=UOp.range(4, 26, AxisType.UPCAST), fragment_axis=UOp.range(16, 27, AxisType.UPCAST))
  mutated_geometry = KernelTileGeometry((64, 128, 32), (2, 2), 128, 32,
    (KernelLDSWindow("A", 0, 5120, 80), KernelLDSWindow("B", 5120, 15360, 80)))
  with pytest.raises(ValueError, match="exact validated anchor"):
    build_precontract_lds_stage(mutated_geometry, tc=_tc(), operands=(a, b), thread=UOp.special(128, "lidx0"),
      k_tile_base=UOp.const(dtypes.weakint, 0), k_substep=UOp.range(2, 24, AxisType.REDUCE), subtile_m=UOp.range(2, 25, AxisType.UPCAST),
      subtile_n=UOp.range(4, 26, AxisType.UPCAST), fragment_axis=UOp.range(16, 27, AxisType.UPCAST))
