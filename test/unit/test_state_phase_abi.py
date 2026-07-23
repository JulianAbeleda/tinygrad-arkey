import pytest

from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop import Ops
from tinygrad.uop.ops import UOp, StateRegionSpec, PhaseBoundarySpec, StateHandle
from tinygrad.uop.spec import spec_full, type_verify


def _handle(lanes=1):
  return StateHandle(StateRegionSpec("running_state", dtypes.float, lanes), PhaseBoundarySpec("produce", "consume", 3))


def test_scalar_state_publish_reload_preserves_typed_identity():
  handle = _handle()
  value = UOp.const(dtypes.float, 1.0)
  published, reloaded = handle.publish(value), handle.reload(handle.publish(value))
  assert published.dtype == reloaded.dtype == dtypes.float
  assert reloaded.src == (published,)
  type_verify(UOp.sink(reloaded), spec_full)


def test_vector_state_publish_reload_preserves_lanes_and_phase_boundary():
  handle = _handle(8)
  value = UOp.const(dtypes.float.vec(8), 1.0)
  published = handle.publish(value)
  reloaded = handle.reload(published)
  assert handle.dtype == reloaded.dtype == dtypes.float.vec(8)
  assert reloaded.arg[1].boundary == PhaseBoundarySpec("produce", "consume", 3)
  type_verify(UOp.sink(reloaded), spec_full)


def test_vector_state_reload_is_a_typed_scalar_lane_carrier():
  handle = _handle(8)
  reloaded = handle.reload(handle.publish(UOp.const(dtypes.float.vec(8), 1.0)))
  lane = reloaded.gep(3)
  assert lane.dtype == dtypes.float
  type_verify(UOp.sink(lane), spec_full)


def test_scalar_state_reload_remains_a_raw_scalar_value():
  handle = _handle()
  reloaded = handle.reload(handle.publish(UOp.const(dtypes.float, 1.0)))
  assert reloaded.dtype == dtypes.float
  type_verify(UOp.sink(reloaded), spec_full)


def test_state_handle_rejects_invalid_phase_lifetime_and_foreign_publication():
  with pytest.raises(ValueError): PhaseBoundarySpec("same", "same").validate()
  handle, foreign = _handle(), StateHandle(StateRegionSpec("other", dtypes.float), PhaseBoundarySpec("produce", "consume"))
  with pytest.raises(ValueError): handle.reload(foreign.publish(UOp.const(dtypes.float, 1.0)))
  malformed = UOp(Ops.CUSTOMI, dtypes.float, (UOp.const(dtypes.float, 1.0),), ("state_reload_v1", handle))
  with pytest.raises(RuntimeError): type_verify(UOp.sink(malformed), spec_full)


def test_lane_major_local_state_publish_reload_tracks_storage_and_wait_order():
  storage = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(128, AddrSpace.LOCAL), arg=91)
  lane = UOp.special(8, "state_lane")
  handle = StateHandle(StateRegionSpec("vector_state", dtypes.float, 8), PhaseBoundarySpec("publish", "reload"),
                       storage=storage, lane=lane, lane_stride=16, element_offset=4)
  published = handle.publish(UOp.const(dtypes.float.vec(8), 1.0))
  wait = UOp(Ops.WAIT, dtypes.void, (published,), arg=("state_handle_wait_v1", handle))
  reloaded = handle.reload(published, wait)
  assert published.src[1:] == (storage, lane)
  assert reloaded.op is Ops.CUSTOMI and reloaded.arg == ("state_reload_v1", handle)
  assert reloaded.src[0].op is Ops.STACK and reloaded.src[0].tag == ("state_reload_lanes_v1", handle)
  assert len(reloaded.src[0].src) == 8 and all(source.op is Ops.LOAD for source in reloaded.src[0].src)
  type_verify(UOp.sink(reloaded), spec_full)


def test_storage_backed_vector_reload_stack_tag_allows_scalar_gep():
  storage = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(128, AddrSpace.LOCAL), arg=94)
  handle = StateHandle(StateRegionSpec("vector_state", dtypes.float, 8), PhaseBoundarySpec("publish", "reload"),
                       storage=storage, lane=UOp.special(8, "state_lane"), lane_stride=8)
  reloaded = handle.reload(handle.publish(UOp.const(dtypes.float.vec(8), 1.0)))
  assert reloaded.op is Ops.CUSTOMI and reloaded.arg == ("state_reload_v1", handle)
  lane = reloaded.gep(5)
  assert lane.src == (reloaded,) and lane.dtype == dtypes.float
  type_verify(UOp.sink(lane), spec_full)


def test_one_source_reload_lowers_at_generic_vector_lane_boundary():
  from tinygrad.renderer.isa.amd import lower_state_phase_transfer, native_repack_matcher
  from tinygrad.uop.ops import graph_rewrite
  storage = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(128, AddrSpace.LOCAL), arg=95)
  handle = StateHandle(StateRegionSpec("vector_state", dtypes.float, 8), PhaseBoundarySpec("publish", "reload"),
                       storage=storage, lane=UOp.special(8, "state_lane"), lane_stride=8)
  reloaded = handle.reload(handle.publish(UOp.const(dtypes.float.vec(8), 1.0)))
  assert len(reloaded.src) == 1 and lower_state_phase_transfer(reloaded) is None
  lowered = graph_rewrite(UOp.sink(reloaded.gep(2)), native_repack_matcher, bottom_up=True)
  assert not any(u.op is Ops.CUSTOMI and u.arg == ("state_reload_v1", handle) for u in lowered.toposort())
  assert any(u.op is Ops.LOAD for u in lowered.toposort())


def test_handle_owned_loop_state_lane_lowers_without_register_placeholder():
  from tinygrad.renderer.isa.amd import lower_state_phase_transfer
  storage = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(128, AddrSpace.LOCAL), arg=96)
  handle = StateHandle(StateRegionSpec("loop_state", dtypes.float, 8), PhaseBoundarySpec("loop_write", "loop_read"),
                       storage=storage, lane=UOp.special(8, "state_lane"), lane_stride=8)
  lane = handle.loop_read(3)
  type_verify(UOp.sink(lane), spec_full)
  lowered = lower_state_phase_transfer(lane)
  assert lowered is not None and lowered.op is Ops.LOAD
  assert not any(u.op is Ops.DEFINE_REG for u in lowered.toposort())


def test_handle_owned_loop_state_init_iteration_final_write_read_ownership():
  from tinygrad.renderer.isa.amd import lower_state_phase_transfer
  storage = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(128, AddrSpace.LOCAL), arg=97)
  handle = StateHandle(StateRegionSpec("loop_state", dtypes.float, 8), PhaseBoundarySpec("init", "final"),
                       storage=storage, lane=UOp.special(8, "state_lane"), lane_stride=8)
  init = handle.loop_write(UOp.const(dtypes.float, 0.0), 0)
  iteration = handle.loop_write(UOp.const(dtypes.float, 1.0), 0, after=handle.loop_read(0))
  final = handle.loop_read(0)
  type_verify(UOp.sink(init, iteration, final), spec_full)
  assert lower_state_phase_transfer(init).op is Ops.STORE
  assert lower_state_phase_transfer(iteration).op is Ops.STORE
  assert lower_state_phase_transfer(final).op is Ops.LOAD


def test_storage_backed_state_rejects_invalid_storage_lane_and_offset():
  storage = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(64, AddrSpace.LOCAL), arg=92)
  lane = UOp.special(8, "state_lane")
  region = StateRegionSpec("vector_state", dtypes.float, 8)
  boundary = PhaseBoundarySpec("publish", "reload")
  with pytest.raises(TypeError): StateHandle(region, boundary, storage=UOp(Ops.DEFINE_REG, dtypes.float.ptr(64, AddrSpace.REG), arg=93), lane=lane, lane_stride=8).validate()
  with pytest.raises(TypeError): StateHandle(region, boundary, storage=storage, lane=UOp.const(dtypes.float, 0.0), lane_stride=8).validate()
  with pytest.raises(ValueError): StateHandle(region, boundary, storage=storage, lane=lane, lane_stride=8, element_offset=1).validate()
