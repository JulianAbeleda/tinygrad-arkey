"""Regression coverage for nested runtime RANGE lifetime and range simplification."""

from tinygrad import dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.codegen.late.linearizer import pm_split_ends
from tinygrad.codegen.simplify import flatten_range, pm_flatten_range, pm_simplify_ranges, simplify_merge_adjacent
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp, graph_rewrite


def _range(root:UOp, axis_id:int) -> UOp:
  matches=[x for x in root.toposort() if x.op is Ops.RANGE and x.arg[0] == axis_id]
  assert len(matches) == 1
  return matches[0]


def _nested_segment_epoch_lifecycle() -> tuple[UOp, UOp, UOp, UOp]:
  """One owner executes one or two segments, each with a runtime epoch loop.

  The register store before the inner loop is the per-segment reset, the store
  inside the epoch RANGE is the loop-carried accumulator update, and the final
  global store consumes the accumulator only after the inner END.
  """
  owner=UOp.special(170, "gidx0")
  owner_start=owner*6144//170; owner_stop=(owner+1)*6144//170
  tile0=owner_start//48; boundary=(tile0+1)*48; first_stop=owner_stop.minimum(boundary)
  segment_count=1+(owner_stop>boundary).cast(dtypes.int32)
  segment=UOp.range(segment_count, 70, axis_type=AxisType.LOOP)
  segment_depth=(segment>0).where(owner_stop-first_stop, first_stop-owner_start)
  epoch=UOp.range(segment_depth, 71, axis_type=AxisType.LOOP)

  acc=UOp.placeholder((1,), dtypes.float32, 90, addrspace=AddrSpace.REG)
  partials=UOp.placeholder((2,), dtypes.float32, 0)
  reset=acc.after(segment)[0].store(0.0)
  carrier=acc.after(reset).after(epoch)
  carry=carrier[0].store(carrier[0]+(epoch+1).cast(dtypes.float32))
  phase_barrier=UOp.barrier(UOp.group(carry))
  inner_end=UOp.group(phase_barrier).end(epoch)
  partial=partials[segment].store(acc.after(inner_end)[0])
  segment_barrier=UOp.barrier(UOp.group(partial))
  outer_end=UOp.group(segment_barrier).end(segment)
  return UOp.sink(outer_end), inner_end, segment, epoch


def test_nested_runtime_range_keeps_lexical_ends_and_lifecycle_order():
  sink, inner_end, segment, epoch=_nested_segment_epoch_lifecycle()

  # A bare inner RANGE is a lexical leaf.  Its runtime extent references the
  # outer RANGE, but flattening must not claim that the inner END closes both.
  assert segment in epoch.src[0].ranges
  assert flatten_range(inner_end) is None

  rewritten=graph_rewrite(sink, pm_flatten_range+pm_simplify_ranges, ctx={}, name="nested runtime range lifecycle")
  segment,epoch=_range(rewritten,70),_range(rewritten,71)
  ends=[x for x in rewritten.toposort() if x.op is Ops.END]
  inner=next(x for x in ends if epoch in x.ended_ranges)
  outer=next(x for x in ends if segment in x.ended_ranges and x is not inner)

  assert tuple(x for x in inner.ended_ranges if x.op is Ops.RANGE) == (epoch,)
  assert tuple(x for x in outer.ended_ranges if x.op is Ops.RANGE) == (segment,)
  assert segment in inner.ranges and epoch not in inner.ranges
  assert inner in outer.src[0].backward_slice_with_self

  stores=[x for x in rewritten.toposort() if x.op is Ops.STORE]
  assert any(segment in x.ranges and epoch not in x.ranges for x in stores)  # reset
  assert any(segment in x.ranges and epoch in x.ranges for x in stores)      # carry
  assert any(inner in x.backward_slice_with_self for x in stores)            # partial publication
  barriers=[x for x in rewritten.toposort() if x.op is Ops.BARRIER]
  assert len(barriers) == 2 and all(any(s.dtype == dtypes.void for s in x.src) for x in barriers)
  groups=[x for x in rewritten.toposort() if x.op is Ops.GROUP]
  assert not any({Ops.BARRIER, Ops.GROUP} <= {s.op for s in x.src} for x in groups)
  phase_barrier=next(x for x in barriers if epoch in x.ranges)
  segment_barrier=next(x for x in barriers if epoch not in x.ranges)
  assert phase_barrier in inner.backward_slice_with_self
  assert inner in segment_barrier.backward_slice_with_self and segment_barrier in outer.backward_slice_with_self


def test_adjacent_merge_rejects_a_range_dependent_extent():
  outer=UOp.range(2, 80, axis_type=AxisType.LOOP)
  inner=UOp.range(outer+1, 81, axis_type=AxisType.LOOP)
  out=UOp.placeholder((1,), dtypes.int32, 0)
  ended=out[0].store(outer+inner).end(outer,inner)
  assert simplify_merge_adjacent(ended) is ended


def test_adjacent_merge_preserves_independent_rectangular_ranges():
  r0=UOp.range(2, 82, axis_type=AxisType.LOOP)
  r1=UOp.range(3, 83, axis_type=AxisType.LOOP)
  out=UOp.placeholder((6,), dtypes.int32, 0)
  linear=r0*3+r1
  idx=(linear//3)*3+linear%3
  ended=out[idx].store(linear).end(r0,r1)
  merged=simplify_merge_adjacent(ended)
  bare=tuple(x for x in merged.ended_ranges if x.op is Ops.RANGE)
  assert len(bare) == 1 and bare[0].src[0].op is Ops.CONST and bare[0].src[0].arg == 6


def test_final_end_split_keeps_nested_runtime_range_lexical():
  owner=UOp.special(170, "gidx0")
  segment=UOp.range((owner<1).where(2,1), 1498, axis_type=AxisType.LOOP)
  epoch=UOp.range((segment>0).where(1,2), 1499, axis_type=AxisType.LOOP)
  acc=UOp.placeholder((1,), dtypes.float32, 90, addrspace=AddrSpace.REG)
  out=UOp.placeholder((340,), dtypes.float32, 0)

  reset=acc.after(segment)[0].store(0.0)
  carrier=acc.after(reset).after(epoch)
  update=carrier[0].store(carrier[0]+1.0)
  inner_end=UOp.group(UOp.barrier(update)).end(epoch)
  partial=out[segment*170+owner].store(acc.after(inner_end)[0])
  root=UOp.sink(UOp.group(partial).end(segment), arg=KernelInfo(name="nested_runtime_range_split"))

  rewritten=graph_rewrite(root, pm_split_ends)
  ends=[x for x in rewritten.toposort() if x.op is Ops.END]
  inner=next(x for x in ends if epoch in x.ended_ranges)
  outer=next(x for x in ends if segment in x.ended_ranges)
  assert tuple(x for x in inner.ended_ranges if x.op is Ops.RANGE) == (epoch,)
  assert tuple(x for x in outer.ended_ranges if x.op is Ops.RANGE) == (segment,)
  assert inner in outer.src[0].backward_slice_with_self
