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
