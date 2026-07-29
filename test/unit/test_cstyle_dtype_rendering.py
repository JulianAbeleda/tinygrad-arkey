import pytest

from tinygrad.dtype import dtypes
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.uop.ops import UOp


class CompactTypeRenderer(CStyleLanguage):
  type_map = {dtypes.uint16: "ushort", dtypes.uint32: "uint"}


def test_scalar_dtype_is_orthogonal_to_vector_lanes():
  renderer = CompactTypeRenderer(Target("TEST"))
  vector = UOp.const(dtypes.uint32.vec(2), (1, 2))

  assert renderer.render_dtype(vector.dtype) == "uint"
  assert renderer.render_scalar_dtype(vector.dtype) == "uint"
  assert renderer.render_vector_dtype(vector.dtype, 2) == "uint2"
  assert renderer.render_type(vector) == "uint2"


def test_target_owns_scalar_and_vector_spelling():
  renderer = CompactTypeRenderer(Target("TEST"))

  assert renderer.render_vector_dtype(dtypes.uint16, 4) == "ushort4"
  assert renderer.render_vector_dtype(dtypes.uint32, 1) == "uint"
  with pytest.raises(AssertionError, match="lane count must be positive"):
    renderer.render_vector_dtype(dtypes.uint32, 0)
