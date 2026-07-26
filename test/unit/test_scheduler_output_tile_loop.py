import pytest

from tinygrad import dtypes
from tinygrad.codegen.opt.kernel_pipeline import (PINNED_WMMA_VGPR_BUDGET, SchedulerOutputTileLoop,
  build_scheduler_output_tile_loop, build_scheduler_output_tile_owner)
from tinygrad.codegen.opt.heuristic import bounded_reduction_unroll
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.uop.ops import AxisType, Ops, UOp


def test_output_tiles_are_symbolic_and_owner_is_called_once():
  calls = []

  def owner(tile):
    calls.append(tile)
    return UOp(Ops.NOOP, dtypes.float, (tile,))

  graph = build_scheduler_output_tile_loop(SchedulerOutputTileLoop(64), owner)
  assert len(calls) == 1
  assert calls[0].op is Ops.RANGE
  assert calls[0].arg == (9300, AxisType.LOOP)
  assert calls[0].src[0].arg == 64
  assert len([u for u in graph.toposort() if u.op is Ops.RANGE]) == 1
  ends = [u for u in graph.toposort() if u.op is Ops.END]
  assert len(ends) == 1 and ends[0].src[-1] is calls[0]


def test_output_loop_closes_the_owner_body_without_dropping_q4_readiness():
  ready = UOp.barrier(UOp.group(UOp(Ops.NOOP, dtypes.int, ())) )

  def owner(tile):
    store = UOp(Ops.NOOP, dtypes.float, (ready, tile))
    return store

  graph = build_scheduler_output_tile_loop(SchedulerOutputTileLoop(8), owner)
  assert ready in graph.backward_slice
  assert any(u.op is Ops.END and u.src[-1].op is Ops.RANGE for u in graph.toposort())


def test_output_loop_reuses_the_pinned_wmma_footprint():
  plan = SchedulerOutputTileLoop(256, resident_accumulator_vgprs=128, resident_fragment_vgprs=64)
  assert plan.tile_count == 256
  assert plan.resident_accumulator_vgprs + plan.resident_fragment_vgprs == PINNED_WMMA_VGPR_BUDGET


def test_reduction_unroll_is_bounded_by_resident_output_pressure():
  # output upcast 4 * reduction unroll 32 was the symbolic attention-score
  # failure.  Admission retains a reduction loop instead of expanding 128
  # accumulator lanes into one thread.
  assert bounded_reduction_unroll(4, 32, (32, 16, 8, 4, 2)) == 8
  assert bounded_reduction_unroll(1, 32, (32, 16, 8, 4, 2)) == 32


def test_postrange_bounds_preexpanded_flat_reduction_body():
  output = UOp.range(4, 9600, AxisType.UPCAST)
  reduction = UOp.range(32, 9601, AxisType.REDUCE)
  expanded = UOp.range(4, 9602, AxisType.UNROLL)
  streams = [UOp.param(i+1, dtypes.float.ptr(4096)).index(reduction*4+expanded+output*128+i).load() for i in range(8)]
  body = streams[0]
  for stream in streams[1:]: body = body + stream
  reduced = UOp(Ops.REDUCE, dtypes.float, (body, reduction, expanded), (Ops.ADD, ()))
  ast = UOp.sink(UOp.param(0, dtypes.float.ptr(4)).index(output).store(reduced))

  scheduler = Scheduler(ast, None)
  scheduler.bound_expanded_reduction_pressure()

  assert [int(x.vmax+1) for x in scheduler.ranges_of(AxisType.LOOP)] == [2, 2]
  assert [int(x.vmax+1) for x in scheduler.ranges_of(AxisType.UPCAST)] == []
  assert [int(x.vmax+1) for x in scheduler.ranges_of(AxisType.REDUCE)] == [32]
  assert [int(x.vmax+1) for x in scheduler.ranges_of(AxisType.UNROLL)] == [4]


@pytest.mark.parametrize("kwargs", [
  {"tile_count": 0},
  {"tile_count": 2, "resident_accumulator_vgprs": 129, "resident_fragment_vgprs": 64},
  {"tile_count": 2, "resident_accumulator_vgprs": 128, "resident_fragment_vgprs": 65},
])
def test_output_loop_fails_closed_before_graph_construction(kwargs):
  with pytest.raises(ValueError): SchedulerOutputTileLoop(**kwargs)


def test_owner_contract_preserves_barrier_dependency():
  ready = UOp.barrier(UOp.group(UOp(Ops.NOOP, dtypes.int, ())))

  def owner(tile):
    return UOp(Ops.NOOP, dtypes.float, (ready, tile))

  graph = build_scheduler_output_tile_loop(SchedulerOutputTileLoop(8), owner)
  assert ready in graph.backward_slice

def test_real_owner_carries_m_n_group_into_load_wmma_and_store():
  buf = UOp.param(0, dtypes.float.ptr(4096))
  def owner(ix):
    address = ix.m * 256 + ix.n * 16 + ix.group
    loaded = buf.index(address).load()
    wmma = UOp(Ops.WMMA, dtypes.float, (loaded, loaded, loaded), ("owner",))
    return buf.index(address).store(wmma)
  graph = build_scheduler_output_tile_owner(SchedulerOutputTileLoop(2, loop_id=9400),
    SchedulerOutputTileLoop(3, loop_id=9401), SchedulerOutputTileLoop(4, loop_id=9402), owner)
  nodes = graph.toposort()
  axes = [x for x in nodes if x.op is Ops.RANGE]
  assert {x.arg[0] for x in axes} == {9400, 9401, 9402}
  assert any(x.op is Ops.LOAD for x in nodes) and any(x.op is Ops.WMMA for x in nodes)
  assert any(x.op is Ops.STORE for x in nodes)
  assert len([x for x in nodes if x.op is Ops.END]) == 3

def test_real_owner_rejects_detached_dynamic_indexing():
  foreign = UOp.range(7, 9499, AxisType.LOOP)
  buf = UOp.param(0, dtypes.float.ptr(64))
  def owner(ix):
    address = ix.m + ix.n + ix.group + foreign
    return buf.index(address).load()
  with pytest.raises(ValueError, match="outside scheduler tile ownership"):
    build_scheduler_output_tile_owner(SchedulerOutputTileLoop(2, loop_id=9500),
      SchedulerOutputTileLoop(1, loop_id=9501), SchedulerOutputTileLoop(1, loop_id=9502), owner)
