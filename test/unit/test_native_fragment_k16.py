from tinygrad.codegen.late.native_fragment import native_fragment_x2
from tinygrad.codegen.opt.tc import cuda_81616_i8
def test_native_k16_descriptor():
  assert cuda_81616_i8[0].dims == (8,16,16)
  assert cuda_81616_i8[0].elements_per_thread == (8,4,4)
  assert native_fragment_x2
