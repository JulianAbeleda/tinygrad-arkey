"""CPU-only boundary tests for output-store ownership rewrites."""
from tinygrad import dtypes
from tinygrad.codegen.late.devectorizer import (devectorize_bare_output_store, devectorize_output_projection_store,
  gep_on_store, load_store_folding, reduce_duplicate_output_store)
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import Ops, UOp, graph_rewrite


def _duplicate_gep(*, output_space=AddrSpace.GLOBAL, value_space=AddrSpace.REG, arg=(0, 0, 1, 1), count=4):
  out = UOp.placeholder((64,), dtypes.float, 0, addrspace=output_space)
  base = UOp(Ops.INDEX, dtypes.float.ptr(64, addrspace=output_space).vec(2),
             (out, UOp.vectorize(UOp.const(dtypes.weakint, 2), UOp.const(dtypes.weakint, 9))))
  reg = UOp(Ops.DEFINE_REG, dtypes.float.ptr(count, addrspace=value_space), arg=0)
  value = UOp(Ops.INDEX, dtypes.float.vec(count), (reg, UOp.const(dtypes.weakint, 0)))
  if value_space is AddrSpace.REG:
    contribution = UOp.vectorize(*(UOp.const(dtypes.float, i+1) for i in range(count)))
    update = value.store(value + contribution)
    value = UOp(Ops.INDEX, dtypes.float.vec(count), (reg.after(update), UOp.const(dtypes.weakint, 0)))
  return base.gep(arg), value


def test_duplicate_reducer_is_global_output_reg_partial_and_uniform_only():
  assert reduce_duplicate_output_store(*_duplicate_gep()) is not None
  # A broadcast/single value is not an additive set of distinct partial lanes.
  target, value = _duplicate_gep(count=1)
  assert reduce_duplicate_output_store(target, value) is None
  assert reduce_duplicate_output_store(*_duplicate_gep(arg=(0, 0, 1, 2))) is None
  assert reduce_duplicate_output_store(*_duplicate_gep(output_space=AddrSpace.LOCAL)) is None
  assert reduce_duplicate_output_store(*_duplicate_gep(value_space=AddrSpace.LOCAL)) is None
  assert reduce_duplicate_output_store(*_duplicate_gep(value_space=AddrSpace.GLOBAL)) is None


def test_gated_duplicate_store_is_not_claimed_by_additive_projection_reducer():
  target, value = _duplicate_gep()
  gate = UOp.const(dtypes.bool.vec(4), True)
  store = target.store(value, gate)
  lowered = graph_rewrite(store, load_store_folding)
  assert lowered.op is Ops.STORE and lowered.src[0].op is Ops.GEP
  assert len(set(lowered.src[0].arg)) < len(lowered.src[0].arg)


def test_duplicate_reducer_uses_destination_keys_and_legacy_inversion_fails_closed():
  # Destination grouping is semantic, not positional: both a permuted map and
  # interleaved duplicate groups retain all additive partial lanes.
  for arg in ((1, 1, 0, 0), (0, 1, 0, 1)):
    target, value = _duplicate_gep(arg=arg)
    lowered = reduce_duplicate_output_store(target, value)
    assert lowered is not None and lowered.src[0] is target.src[0]
    assert lowered.src[1].op is Ops.GEP and lowered.src[1].src[0].op is Ops.STACK
    assert len({u for u in lowered.src[1].src[0].src if u.op is Ops.ADD}) == 2

  # An ordinary REG read has no proven ADD accumulator provenance. Neither the
  # specialized reducer nor generic GEP inversion may silently choose a lane.
  target, value = _duplicate_gep()
  plain_reg = UOp(Ops.DEFINE_REG, dtypes.float.ptr(4, addrspace=AddrSpace.REG), arg=1)
  plain = UOp(Ops.INDEX, dtypes.float.vec(4), (plain_reg, UOp.const(dtypes.weakint, 0)))
  assert reduce_duplicate_output_store(target, plain) is None
  assert gep_on_store(target, plain) is None

  # A REG reduction with a different combine op is not an output projection.
  max_reg = UOp(Ops.DEFINE_REG, dtypes.float.ptr(4, addrspace=AddrSpace.REG), arg=2)
  max_read = UOp(Ops.INDEX, dtypes.float.vec(4), (max_reg, UOp.const(dtypes.weakint, 0)))
  max_update = max_read.store(max_read.alu(Ops.MAX, UOp.const(dtypes.float.vec(4), 1)))
  max_value = UOp(Ops.INDEX, dtypes.float.vec(4), (max_reg.after(max_update), UOp.const(dtypes.weakint, 0)))
  assert reduce_duplicate_output_store(target, max_value) is None


def test_sparse_duplicate_destination_keys_remain_visible_for_index_and_ptrcat():
  out = UOp.placeholder((64,), dtypes.float, 0, addrspace=AddrSpace.GLOBAL)
  offsets = UOp.vectorize(*(UOp.const(dtypes.weakint, x) for x in (2, 4, 6, 8)))
  index_base = UOp(Ops.INDEX, dtypes.float.ptr(64, addrspace=AddrSpace.GLOBAL).vec(4), (out, offsets))
  p0 = out.index(UOp.const(dtypes.weakint, 2)).cast(dtypes.float.vec(2).ptr(64, addrspace=AddrSpace.GLOBAL))
  p1 = out.index(UOp.const(dtypes.weakint, 6)).cast(dtypes.float.vec(2).ptr(64, addrspace=AddrSpace.GLOBAL))
  ptrcat_base = UOp(Ops.PTRCAT, dtypes.float.ptr(64, addrspace=AddrSpace.GLOBAL).vec(4), (p0, p1))
  reg = UOp(Ops.DEFINE_REG, dtypes.float.ptr(4, addrspace=AddrSpace.REG), arg=8)
  read = UOp(Ops.INDEX, dtypes.float.vec(4), (reg, UOp.const(dtypes.weakint, 0)))
  update = read.store(read + UOp.const(dtypes.float.vec(4), 1))
  value = UOp(Ops.INDEX, dtypes.float.vec(4), (reg.after(update), UOp.const(dtypes.weakint, 0)))
  for base in (index_base, ptrcat_base):
    target = base.gep((0, 0, 2, 2))
    assert reduce_duplicate_output_store(target, value) is None
    lowered = graph_rewrite(target.store(value), load_store_folding)
    assert lowered.op is Ops.STORE and lowered.src[0].op is Ops.GEP
    assert lowered.src[0].src[0] is base and lowered.src[0].arg == (0, 0, 2, 2)


def test_full_permuted_interleaved_ptrcat_projection_reduces_then_splits_exactly():
  out = UOp.placeholder((64,), dtypes.float, 0, addrspace=AddrSpace.GLOBAL)
  p0 = out.index(UOp.const(dtypes.weakint, 2)).cast(dtypes.float.vec(2).ptr(64, addrspace=AddrSpace.GLOBAL))
  p1 = out.index(UOp.const(dtypes.weakint, 6)).cast(dtypes.float.vec(2).ptr(64, addrspace=AddrSpace.GLOBAL))
  base = UOp(Ops.PTRCAT, dtypes.float.ptr(64, addrspace=AddrSpace.GLOBAL).vec(4), (p0, p1))
  arg = (3, 0, 3, 0, 2, 1, 2, 1)
  reg = UOp(Ops.DEFINE_REG, dtypes.float.ptr(8, addrspace=AddrSpace.REG), arg=9)
  read = UOp(Ops.INDEX, dtypes.float.vec(8), (reg, UOp.const(dtypes.weakint, 0)))
  update = read.store(read + UOp.const(dtypes.float.vec(8), 1))
  value = UOp(Ops.INDEX, dtypes.float.vec(8), (reg.after(update), UOp.const(dtypes.weakint, 0)))
  reduced = reduce_duplicate_output_store(base.gep(arg), value)
  assert reduced is not None and reduced.src[0] is base and reduced.src[1].dtype.count == 4
  lowered = graph_rewrite(reduced, load_store_folding)
  stores = [u for u in lowered.toposort() if u.op is Ops.STORE]
  assert len(stores) == 3  # accumulator update plus the two exact PTRCAT segment stores
  output_stores = [store for store in stores if store.src[0] in (p0, p1)]
  assert len(output_stores) == 2 and all(store.src[1].dtype.count == 2 for store in output_stores)


def test_scalarized_nested_accumulator_expression_is_normalized_once_per_destination():
  target, value = _duplicate_gep()
  leaves = [value.gep(i) for i in range(4)]
  nested = [leaf * (leaf + UOp.const(dtypes.float, 1)).reciprocal() for leaf in leaves]
  stacked = UOp(Ops.STACK, dtypes.float.vec(4), tuple(nested))
  lowered = reduce_duplicate_output_store(target, stacked)
  assert lowered is not None and lowered.src[1].dtype.count == 2
  # Two destination expressions survive; each contains a sum of its two
  # partial accumulator leaves, reused by the nonlinear subtree.
  assert lowered.src[1].op is Ops.STACK
  assert all(any(u.op is Ops.ADD for u in lane.backward_slice) for lane in lowered.src[1].src)


def test_distinct_gep_store_keeps_ordinary_lane_store_semantics():
  target, value = _duplicate_gep(arg=(0, 1, 2, 3))
  store = target.store(value)
  lowered = graph_rewrite(store, load_store_folding)
  assert lowered.op is Ops.STORE
  # The established GEP-store inversion moves the vector lane map to the
  # value; it must retain the original concrete output INDEX untouched.
  assert lowered.src[0] is target.src[0] and lowered.src[1].op is Ops.GEP


def _loaded_targets(space=AddrSpace.GLOBAL, repeats=False):
  buf = UOp.placeholder((64,), dtypes.float, 0, addrspace=space)
  lanes = []
  for i in range(4):
    index = buf.index(UOp.const(dtypes.weakint, 3 if repeats and i < 2 else i * 4))
    lanes.append(index.load().gep((0,)))
  return UOp(Ops.STACK, dtypes.float.vec(4), tuple(lanes)), UOp.vectorize(*(UOp.const(dtypes.float, i) for i in range(4)))


def test_output_projection_declines_nonuniform_broadcast_and_non_global_forms():
  target, value = _loaded_targets()
  # Distinct loaded addresses are an ordinary store, not a repeated projection.
  assert devectorize_output_projection_store(target, value) is None
  repeated, value = _loaded_targets(repeats=True)
  assert devectorize_output_projection_store(repeated, value) is None  # nonuniform 2/1/1 groups
  local, value = _loaded_targets(AddrSpace.LOCAL, repeats=True)
  assert devectorize_output_projection_store(local, value) is None
  # Bare recovery remains able to preserve a broadcast value and gated lanes.
  target, _ = _loaded_targets()
  gate = UOp.const(dtypes.bool.vec(4), True)
  assert devectorize_bare_output_store(target, UOp.const(dtypes.float, 1), gate) is None
