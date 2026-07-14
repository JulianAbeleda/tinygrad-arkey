import pytest

from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp
from tinygrad.uop.spec import spec_tensor, type_verify


def vector_load(dtype):
  # The tensor LOAD rule only requires an INDEX-shaped memory source here; index
  # graph validity is orthogonal to the LOAD -> GEP invariant under test.
  index = UOp(Ops.INDEX, dtype.ptr())
  return UOp(Ops.LOAD, dtype, (index,))


@pytest.mark.parametrize("dtype", [dtypes.uint32.vec(2), dtypes.uint32.vec(4),
                                    dtypes.uint16.vec(4), dtypes.uint16.vec(8)])
def test_tensor_vector_integer_load_scalar_gep(dtype):
  load = vector_load(dtype)
  for lane in range(dtype.vcount): type_verify([load, load.gep(lane)], spec_tensor)


@pytest.mark.parametrize("load_dtype,gep_dtype", [
  (dtypes.float32.vec(4), dtypes.float32),
  (dtypes.uint32, dtypes.uint32),
  (dtypes.uint32.vec(4), dtypes.uint16),
  (dtypes.uint32.vec(4), dtypes.uint32.vec(2)),
])
def test_tensor_vector_load_gep_rejects_invalid_dtype(load_dtype, gep_dtype):
  gep = UOp(Ops.GEP, gep_dtype, (vector_load(load_dtype),), (0,))
  with pytest.raises(RuntimeError, match="UOp verification failed"): type_verify([gep], spec_tensor)


@pytest.mark.parametrize("arg", [(), (0, 1), ("0",), (True,), None])
def test_tensor_vector_load_gep_rejects_invalid_lane(arg):
  gep = UOp(Ops.GEP, dtypes.uint32, (vector_load(dtypes.uint32.vec(4)),), arg)
  with pytest.raises(RuntimeError, match="UOp verification failed"): type_verify([gep], spec_tensor)


@pytest.mark.parametrize("lane", [-1, 4])
def test_tensor_vector_load_gep_rejects_out_of_range_lane(lane):
  gep = UOp(Ops.GEP, dtypes.uint32, (vector_load(dtypes.uint32.vec(4)),), (lane,))
  with pytest.raises(RuntimeError, match="UOp verification failed"): type_verify([gep], spec_tensor)
