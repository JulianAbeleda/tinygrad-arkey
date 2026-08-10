"""Marker-side admission for the fp32 q/k reduce-output route (Wave 1, piece B).

Structural CPU-only tests: the multi-row marker must exist with the ordinary
reduce association in the spec, and the identity walk must admit pure PERMUTE
views over precompiled outputs while staying closed to ADD/CAST/SHRINK chains.
No emitter/rangeify change is required here, so these tests never linearize or
lower a multi-row graph.
"""

from tinygrad import Tensor, dtypes, nn
from tinygrad.uop.ops import Ops


def _norm(dim, dtype=dtypes.float32):
  n = nn.RMSNorm(dim, eps=1e-6)
  n.weight = Tensor.ones(dim, dtype=dtype)
  return n


def _apply(norm, x):
  out = norm(x)
  return out._semantic_reduce_output_rmsnorm(x, out, norm.weight, norm.eps)


def _precompiled_producer():
  from tinygrad.function import function
  @function(precompile=True)
  def producer(v): return v + 1
  return producer


def test_marker_admits_fp32_q_shape():
  x = Tensor.empty(1, 32, 1, 128, dtype=dtypes.float32)
  marked = _apply(_norm(128), x)
  assert marked.uop.op is Ops.REDUCE_OUTPUT
  assert marked.uop.src[0].op is not Ops.REDUCE_OUTPUT
  spec = marked.uop.arg
  assert spec.rows == 32 and spec.dim == 128
  assert spec.warps == 32 and spec.lanes == 32 and spec.per_lane == 4
  assert spec.out_dtype is dtypes.float32 and spec.eps == 1e-6


def test_marker_admits_fp32_k_shape():
  x = Tensor.empty(1, 8, 1, 128, dtype=dtypes.float32)
  marked = _apply(_norm(128), x)
  assert marked.uop.op is Ops.REDUCE_OUTPUT
  spec = marked.uop.arg
  assert spec.rows == 8 and spec.dim == 128
  assert spec.warps == 8 and spec.lanes == 32 and spec.per_lane == 4
  assert spec.out_dtype is dtypes.float32


def test_marker_admits_multi_row_4096_dim():
  x = Tensor.empty(1, 32, 1, 4096, dtype=dtypes.float32)
  marked = _apply(_norm(4096), x)
  assert marked.uop.op is Ops.REDUCE_OUTPUT
  spec = marked.uop.arg
  assert spec.rows == 32 and spec.warps == 32 and spec.lanes == 32 and spec.per_lane == 128


def test_single_row_marker_fields_unchanged():
  for dim, warps, per_lane in ((4096, 16, 8), (128, 1, 4)):
    x = Tensor.empty(1, dim, dtype=dtypes.float32)
    marked = _apply(_norm(dim), x)
    assert marked.uop.op is Ops.REDUCE_OUTPUT
    spec = marked.uop.arg
    assert spec.rows == 1 and spec.dim == dim
    assert spec.warps == warps and spec.lanes == 32 and spec.per_lane == per_lane


def test_fp16_multi_row_uses_same_association():
  x = Tensor.empty(1, 8, 1, 128, dtype=dtypes.float16)
  marked = _apply(_norm(128, dtype=dtypes.float16), x)
  assert marked.uop.op is Ops.REDUCE_OUTPUT
  spec = marked.uop.arg
  assert spec.rows == 8 and spec.warps == 8 and spec.per_lane == 4


def test_rows_outside_admit_set_returns_ordinary_fallback():
  for shape in ((1, 4, 1, 256), (4, 256), (1, 16, 1, 128), (1, 8, 1, 256)):
    x = Tensor.empty(*shape, dtype=dtypes.float32)
    marked = _apply(_norm(shape[-1]), x)
    assert marked.uop.op is not Ops.REDUCE_OUTPUT


def test_permute_view_over_precompiled_output_is_identity():
  producer = _precompiled_producer()
  x = Tensor.empty(1, 32, 1, 128, dtype=dtypes.float32)
  for value in (
    producer(x).reshape(32, 1, 128).permute((1, 0, 2)),
    producer(x).permute((0, 2, 1, 3)),
  ):
    assert value.uop.op is Ops.PERMUTE and value.shape[-1] == 128
    marked = _apply(_norm(128), value)
    assert marked.uop.op is Ops.REDUCE_OUTPUT
    assert marked.uop.arg.input_identity_at_marker is True


def test_permute_over_lazy_add_stays_false():
  x = Tensor.empty(1, 32, 1, 128, dtype=dtypes.float32)
  value = (x + 1).reshape(32, 1, 128).permute((1, 0, 2))
  marked = _apply(_norm(128), value)
  assert marked.uop.op is Ops.REDUCE_OUTPUT
  assert marked.uop.arg.input_identity_at_marker is False


def test_permute_over_lazy_cast_stays_false():
  x = Tensor.empty(1, 32, 1, 128, dtype=dtypes.float32)
  value = x.cast(dtypes.float16).reshape(32, 1, 128).permute((1, 0, 2))
  marked = _apply(_norm(128), value)
  assert marked.uop.op is Ops.REDUCE_OUTPUT
  assert marked.uop.arg.input_identity_at_marker is False


def test_permute_chain_with_shrink_offset_stays_false():
  producer = _precompiled_producer()
  x = Tensor.empty(1, 32, 1, 256, dtype=dtypes.float32)
  value = producer(x).shrink(((0, 1), (0, 32), (0, 1), (0, 128))).reshape(32, 1, 128).permute((1, 0, 2))
  assert value.shape == (1, 32, 128)
  marked = _apply(_norm(128), value)
  assert marked.uop.op is Ops.REDUCE_OUTPUT
  assert marked.uop.arg.input_identity_at_marker is False


def test_multi_row_marker_records_realized_buffer_identity():
  x = Tensor.randn(1, 32, 1, 128, dtype=dtypes.float32).realize()
  marked = _apply(_norm(128), x)
  assert marked.uop.op is Ops.REDUCE_OUTPUT
  assert marked.uop.arg.input_identity_at_marker is True
