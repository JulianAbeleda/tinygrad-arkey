from tinygrad import dtypes
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer


def test_non_native_vector_alignment_is_power_of_two_and_size_preserving():
  prefix = CUDARenderer(Target.parse("NV:CUDA:sm_120")).render_vector_prefix(dtypes.float.vec(6))
  assert "struct __align__(8) float6" in prefix
  assert "__align__(24)" not in prefix


def test_bfloat16_pair_does_not_redefine_the_cuda_bf16_typedef():
  # cuda_bf16.h (included for any bf16 kernel) already defines nv_bfloat162; redefining it fails NVRTC
  prefix = CUDARenderer(Target.parse("NV:CUDA:sm_120")).render_vector_prefix(dtypes.bfloat16.vec(2))
  assert " nv_bfloat162 " not in prefix and "make_nv_bfloat162" not in prefix
  assert "struct __align__(4) tg_bfloat162 { nv_bfloat16 x, y; }" in prefix
