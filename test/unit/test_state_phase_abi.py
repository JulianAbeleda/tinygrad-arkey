import pytest

from tinygrad.dtype import dtypes
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


def test_state_handle_rejects_invalid_phase_lifetime_and_foreign_publication():
  with pytest.raises(ValueError): PhaseBoundarySpec("same", "same").validate()
  handle, foreign = _handle(), StateHandle(StateRegionSpec("other", dtypes.float), PhaseBoundarySpec("produce", "consume"))
  with pytest.raises(ValueError): handle.reload(foreign.publish(UOp.const(dtypes.float, 1.0)))
  malformed = UOp(Ops.CUSTOMI, dtypes.float, (UOp.const(dtypes.float, 1.0),), ("state_reload_v1", handle))
  with pytest.raises(RuntimeError): type_verify(UOp.sink(malformed), spec_full)
