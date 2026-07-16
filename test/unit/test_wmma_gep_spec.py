import pytest

from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp
from tinygrad.uop.spec import spec_tensor, type_verify


def wmma(dtype):
  operand = UOp(Ops.NOOP, dtypes.half.vec(16))
  return UOp(Ops.WMMA, dtype, (operand, operand, UOp.const(dtype, 0)), (None,) * 8)


@pytest.mark.parametrize("dtype", (dtypes.float.vec(8), dtypes.int.vec(8)))
def test_tensor_wmma_scalar_gep(dtype):
  value = wmma(dtype)
  for lane in range(dtype.vcount): type_verify([value, value.gep(lane)], spec_tensor)


@pytest.mark.parametrize("dtype", (dtypes.float.vec(8), dtypes.int.vec(8)))
@pytest.mark.parametrize("lane", (-1, 8))
def test_tensor_wmma_scalar_gep_rejects_out_of_range_lane(dtype, lane):
  value = wmma(dtype)
  gep = UOp(Ops.GEP, dtype.scalar(), (value,), (lane,))
  with pytest.raises(RuntimeError, match="UOp verification failed"): type_verify([gep], spec_tensor)


@pytest.mark.parametrize("dtype", (dtypes.float.vec(8), dtypes.int.vec(8)))
def test_tensor_wmma_scalar_gep_rejects_wrong_result_dtype(dtype):
  value = wmma(dtype)
  wrong = dtypes.int if dtype.scalar() == dtypes.float else dtypes.float
  gep = UOp(Ops.GEP, wrong, (value,), (0,))
  with pytest.raises(RuntimeError, match="UOp verification failed"): type_verify([gep], spec_tensor)
