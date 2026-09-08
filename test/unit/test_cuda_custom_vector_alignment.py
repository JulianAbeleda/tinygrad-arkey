from tinygrad import dtypes
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer


def test_non_native_vector_alignment_is_power_of_two_and_size_preserving():
  prefix = CUDARenderer(Target.parse("NV:CUDA:sm_120")).render_vector_prefix(dtypes.float.vec(6))
  assert "struct __align__(8) float6" in prefix
  assert "__align__(24)" not in prefix
