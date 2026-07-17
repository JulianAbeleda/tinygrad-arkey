from types import SimpleNamespace

from tinygrad import dtypes
from tinygrad.codegen.late.devectorizer import split_load_store
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import Ops, UOp


def _context(widths=(), *, requires_alignment=True):
  return SimpleNamespace(target=SimpleNamespace(device="TEST"), supports_float4=True,
    local_store_vector_widths={dtypes.half: widths} if widths else {},
    local_store_requires_static_alignment=requires_alignment)


def _half8_store():
  buf = UOp.placeholder((1024,), dtypes.half, 7, addrspace=AddrSpace.LOCAL)
  idx = UOp.range(32, 0) * 3  # deliberately not statically divisible by eight
  value = UOp(Ops.STACK, dtypes.half.vec(8), tuple(UOp.const(dtypes.half, x) for x in range(8)))
  return buf.index(idx, dtype=dtypes.half.vec(8)).store(value)


def test_local_store_width_capability_is_opt_in_and_generic():
  store = _half8_store()
  default = split_load_store(_context(), store, store.src[0])
  assert default is not None
  assert len([x for x in default.toposort() if x.op is Ops.STORE]) == 8

  # A backend that accepts the transaction without a syntactic divisibility proof preserves the core memory op.
  assert split_load_store(_context((8, 4, 2), requires_alignment=False), store, store.src[0]) is None


def test_local_store_width_can_still_require_static_alignment():
  store = _half8_store()
  rewritten = split_load_store(_context((8, 4, 2)), store, store.src[0])
  assert rewritten is not None
  assert len([x for x in rewritten.toposort() if x.op is Ops.STORE]) == 8


def _global_load(dtype, size, offset, width):
  buf = UOp.placeholder((size,), dtype, 8, addrspace=AddrSpace.GLOBAL)
  idx = buf.index(UOp.const(dtypes.weakint, offset), dtype=dtype.vec(width))
  return idx.load()


def test_generic_packed_integer_global_widths_preserve_b128_and_b64():
  for dtype, widths in ((dtypes.uint32, (4, 2)), (dtypes.uint16, (8, 4))):
    for width in widths:
      load = _global_load(dtype, 32, 0, width)
      assert split_load_store(_context(), load, load.src[0]) is None


def test_generic_packed_integer_global_widths_require_static_alignment():
  for dtype, width in ((dtypes.uint32, 4), (dtypes.uint16, 8)):
    load = _global_load(dtype, 32, 1, width)
    rewritten = split_load_store(_context(), load, load.src[0])
    assert rewritten is not None
    assert [u.dtype.count for u in rewritten.toposort() if u.op is Ops.LOAD] == [1] * width


def test_generic_vector_chunks_never_cross_buffer_bounds():
  # The source operation has a common validity gate. Even if its full logical
  # width is larger, the final in-bounds suffix must not become a vector read.
  load = _global_load(dtypes.uint16, 6, 4, 4)
  rewritten = split_load_store(_context(), load, load.src[0])
  assert rewritten is not None
  assert all(u.dtype.count == 1 for u in rewritten.toposort() if u.op is Ops.LOAD)


def test_byte_backed_local_half_vector_uses_byte_stride_per_lane():
  # Packed LDS records are allocated as a uchar arena but their metadata
  # fields are half2.  Splitting the vector must advance by two bytes for the
  # second lane (the arena index is already a byte offset).
  buf = UOp.placeholder((1024,), dtypes.uint8, 9, addrspace=AddrSpace.LOCAL)
  idx = buf.index(UOp.const(dtypes.weakint, 512), dtype=dtypes.half.vec(2))
  value = UOp(Ops.STACK, dtypes.half.vec(2), (UOp.const(dtypes.half, 1), UOp.const(dtypes.half, 2)))
  store = idx.store(value)
  rewritten = split_load_store(_context(), store, store.src[0])
  stores = [u for u in rewritten.toposort() if u.op is Ops.STORE]
  assert len(stores) == 2
  offsets = [s.src[0].src[1].get_idx().src[1].arg for s in stores]
  assert offsets == [0, 2]
