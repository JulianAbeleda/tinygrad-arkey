import functools
from dataclasses import dataclass
from tinygrad.dtype import dtypes, DType, AddrSpace
from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher, GroupOp, RegisterResidentAccumulator, identity_element, AxisType
from tinygrad.helpers import flatten, prod

# *** Ops.REDUCE -> Ops.DEFINE_ACC ***

@dataclass
class ReduceContext:
  acc_num: int = 0

def physical_composite_slot_dtype(composite, slot_idx:int) -> DType:
  lane_shape = composite.lane_shapes[slot_idx] if getattr(composite, "lane_shapes", ()) else ()
  slot_dtype = composite.slots[slot_idx].dtype
  return slot_dtype.scalar().vec(prod(lane_shape)) if lane_shape else slot_dtype

def horizontal_reduce(inp:UOp, out_dtype:DType) -> list[UOp]:
  # if this has a horizontal reduction component, do that first
  if inp.dtype != out_dtype:
    # NOTE: [0 1 2 3 4 5 6 7] -> [0+4, 1+5, 2+6, 3+7]
    horizontal_amount = inp.dtype.count//out_dtype.count
    return [inp.gep(tuple(range(i, inp.dtype.count, horizontal_amount))) for i in range(0, horizontal_amount)]
  return [inp]

def _select_vector_lane_at(inp:UOp, lane:UOp) -> UOp:
  """Select one scalar lane while retaining the live loop dependency."""
  if inp.dtype.count == 1: return inp
  if inp.op in GroupOp.ALU or inp.op in (Ops.CAST, Ops.BITCAST):
    return UOp(inp.op, inp.dtype.scalar(), tuple(_select_vector_lane_at(s, lane) if s.dtype.count > 1 else s for s in inp.src), inp.arg)
  ret = inp.gep(0)
  for i in range(1, inp.dtype.count):
    lane_ne_i = lane.alu(Ops.CMPNE, UOp.const(lane.dtype, i))
    ret = lane_ne_i.alu(Ops.WHERE, ret, inp.gep(i))
  return ret

def _vectorize_live_v_index(v_src:UOp, reduce_range, lane_group:int, dtype:DType) -> UOp|None:
  """Rebuild V[kv, hd] from the optimized carrier without preserving its wrong-axis vector lanes."""
  if lane_group <= 1: return None
  outer = v_src
  carrier = outer.src[0] if outer.op is Ops.CAST and outer.src else outer
  if carrier.op is not Ops.INDEX or len(carrier.src) < 2: return None
  # A post-expander owned V vector already has the requested physical lane
  # ABI.  It is indexed by the live KV range, so scalarizing it would both
  # duplicate a load and incorrectly reinterpret its lanes as Hd positions.
  if v_src.dtype.count == lane_group and any(r in carrier.backward_slice for r in reduce_range): return v_src
  if v_src.dtype.count == lane_group and reduce_range:
    # Expander's vector lanes are KV-strided values for one Hd position. Use
    # lane zero only for its batch/head base, then express the required row as
    # base + live_kv*Hd + hd so no V value can be hoisted out of the KV loop.
    pointer = carrier.src[0].src[0] if carrier.src[0].op is Ops.STACK else carrier.src[0]
    base = carrier.src[1].src[0] if carrier.src[1].op is Ops.STACK else carrier.src[1]
    kv = reduce_range[-1]
    lanes = []
    for lane in range(lane_group):
      offset = kv * UOp.const(kv.dtype, lane_group) + UOp.const(kv.dtype, lane)
      lanes.append(pointer.index(base + offset).load(dtype=dtype.scalar()))
    return UOp.vectorize(*lanes)
  reduce_set = set(reduce_range)
  index_exprs = carrier.src[1:]
  candidates = tuple(dict.fromkeys(r for idx in index_exprs for r in idx.backward_slice
    if r.op is Ops.RANGE and r not in reduce_set and r.vmin == 0 and r.vmax == lane_group-1))
  if len(candidates) != 1: return None
  hd_range = candidates[0]
  lanes = []
  for lane in range(lane_group):
    indexed = carrier.substitute({hd_range: UOp.const(hd_range.dtype, lane)})
    lanes.append(indexed.cast(outer.dtype) if outer.op is Ops.CAST else indexed)
  if any(r in lane.backward_slice for lane in lanes for r in (hd_range,)): raise RuntimeError("Hd RANGE survived V lane substitution")
  return UOp.vectorize(*lanes)

def _load_v_at_reduce_pos(v_src:UOp, composite, input_ranges, reduce_range, score_shape=None, axis_map=None, lane_group=1):
  """Create a LOAD from V at the current reduce position.
  Uses RANGE UOps from input_ranges and reduce_range to build indices.
  """
  # Rangeify may already have indexed and contracted the logical value input
  # into its authoritative Hd lane group. Re-indexing that value as a pointer
  # would invent ownership and produce an invalid INDEX over an ALU node.
  if lane_group > 1 and v_src.dtype.count == lane_group: return v_src
  # A declared map owns this load. It avoids inferring logical inputs from
  # expander-created STACK/weakint range carriers.
  if axis_map is not None:
    if len(axis_map) == 0: return v_src.load(dtype=composite.slots[-1].dtype)
    range_by_axis = {r.arg[0]: r for r in input_ranges + reduce_range}
    source = v_src.src[0] if v_src.op is Ops.SCOPED_VALUE else v_src
    # None is a broadcast-zero source axis. -1 is value-local and deliberately
    # omitted, preserving the trailing contiguous lane for a vector load.
    idxs = tuple(range_by_axis[axis] if axis is not None else UOp.const(dtypes.weakint, 0)
                 for axis in axis_map if axis != -1)
    # A grouped lane carrier is lowered as an explicit vector of scalar loads.
    # This preserves source indexing (and therefore aliasing/layout semantics)
    # while making the logical Hd lane visible to the combine.  Backends may
    # later replace this STACK/vectorize with a fragment load; no backend
    # assumptions are made here.
    if lane_group > 1:
      scalar_dtype = composite.slots[-1].dtype.scalar()
      return UOp.vectorize(*(source.index(*(idxs[:-1] + (idxs[-1] + UOp.const(dtypes.weakint, lane),))).load(dtype=scalar_dtype)
                            for lane in range(lane_group)))
    return source.index(*idxs).load(dtype=composite.slots[-1].dtype)
  # Build axis -> RANGE mapping from all visible ranges
  range_by_axis = {}
  for r in input_ranges + reduce_range:
    range_by_axis[r.arg[0]] = r
  # Determine rank from V shape; fall back to max axis + 1
  try:
    v_shape = v_src._shape
    v_rank = len(v_shape) if v_shape is not None else (max(range_by_axis.keys()) + 2 if range_by_axis else 1)
  except Exception:
    v_rank = max(range_by_axis.keys()) + 2 if range_by_axis else 1
  # V is [..., KV, Hd], while score is [..., Q, KV]. The query axis is
  # absent from V, so score-axis numbers cannot be copied directly.
  v_indices = []
  score_rank = len(score_shape) if score_shape is not None else 0
  reduce_axis = reduce_range[-1].arg[0] if reduce_range else None
  query_axis = reduce_axis - 1 if reduce_axis is not None and reduce_axis > 0 else None
  for v_axis in range(v_rank - 1):
    if score_rank == v_rank and query_axis is not None and v_axis == query_axis:
      v_indices.append(reduce_range[-1])
    elif score_rank == v_rank and query_axis is not None and v_axis > query_axis:
      v_indices.append(range_by_axis.get(v_axis + 1, UOp.const(dtypes.weakint, 0)))
    else:
      v_indices.append(range_by_axis.get(v_axis, UOp.const(dtypes.weakint, 0)))
  v_indices = tuple(v_indices)
  if not v_indices:
    v_indices = (UOp.const(dtypes.weakint, 0),)
  v_index = v_src.index(*v_indices)
  if lane_group > 1:
    scalar_dtype = composite.slots[-1].dtype.scalar()
    return UOp.vectorize(*(v_src.index(*(v_indices + (UOp.const(dtypes.weakint, lane),))).load(dtype=scalar_dtype)
                          for lane in range(lane_group)))
  # The auxiliary value is a logical element for the final state slot. A
  # generic composite may have one or many slots; never assume online-softmax
  # has already supplied a third slot here.
  return v_index.load(dtype=composite.slots[-1].dtype)

def _partition_composite_sources(srcs, composite):
  """Separate range context from the explicitly declared logical inputs.

  Rangeify may append RANGE UOps to a REDUCE source list.  They are loop
  context, never auxiliary tensors.  Composite input ownership is therefore
  determined only from non-RANGE sources and the declared input-spec count.
  """
  ranges = tuple(x for x in srcs if x.op is Ops.RANGE)
  candidates = tuple(x for x in srcs if x.op is not Ops.RANGE)
  ninputs = len(getattr(composite, "input_specs", ()))
  return ranges, (candidates[-ninputs:] if ninputs else ())

def reduce_to_acc(ctx:ReduceContext, red:UOp):
  from tinygrad.uop.ops import CompositeReduce
  composite_arg = red.arg[0] if isinstance(red.arg, tuple) and len(red.arg) > 0 else None
  # CompositeReduce is immutable compiler data. Recognize its structural
  # contract too so graph reconstruction/module identity cannot silently route
  # a stateful reduction through ordinary ALU lowering.
  composite = composite_arg if isinstance(composite_arg, CompositeReduce) or \
    (hasattr(composite_arg, "slots") and hasattr(composite_arg, "combine_fn")) else None
  inp = red.src[0]
  raw_rest = red.src[1:]
  range_srcs = tuple(x for x in raw_rest if x.op is Ops.RANGE)
  # Ordinary REDUCE lowering owns every range supplied by rangeify: output
  # loop ranges are part of the loop-carried accumulator context too. Only a
  # composite with explicit auxiliary sources needs to split REDUCE ranges
  # from those sources.
  if composite is not None and getattr(composite, "reduce_range_axes", ()):
    reduce_range = tuple(x for x in range_srcs if x.arg[0] in composite.reduce_range_axes)
  else:
    reduce_range = tuple(x for x in range_srcs if x.arg[1] is AxisType.REDUCE) if composite is not None else raw_rest
  extra_srcs = tuple(x for x in raw_rest if x.op is not Ops.RANGE)
  if composite is not None:
    # Keep the source partition explicit; non-range carriers not declared by
    # CompositeInputSpec must not become logical auxiliary inputs.
    _, extra_srcs = _partition_composite_sources(raw_rest, composite)

  # Composite reduce with no ranges yet: rangeify inline
  if composite is not None and len(reduce_range) == 0:
    axis = red.arg[1]
    if not axis:
      # Source roles are part of CompositeReduce metadata. Do not infer them
      # from expander-created weakint/STACK carriers.
      input_specs = getattr(composite, "input_specs", ())
      auxiliary_inputs = extra_srcs[-len(input_specs):] if input_specs else ()
      from tinygrad.codegen.late.composite_combines import _handle_no_range_generic
      # UOp src is always a tuple; a one-slot combine may return a scalar UOp.
      result = _handle_no_range_generic(inp, composite, red, auxiliary_inputs)
      return UOp(Ops.TUPLE, dtypes.void, result if isinstance(result, tuple) else (result,)).replace(tag=("composite_reduce", composite))
    rngs = tuple(UOp.range(UOp.const(dtypes.weakint, red.src[0].shape[i]), i, AxisType.REDUCE) for i in axis)
    red = UOp(Ops.REDUCE, red.dtype, src=(red.src[0],) + rngs + extra_srcs, arg=(red.arg[0], ()))
    inp, reduce_range = red.src[0], rngs

  lst = horizontal_reduce(inp, red.dtype)
  assert all(x.dtype == red.dtype for x in lst), f"horizontal reduction mismatch {lst[0].dtype} != {red.dtype}"
  # if we have a range
  if len(reduce_range) != 0:
    topo = inp.toposort()
    ended_ranges = flatten([x.ended_ranges for x in topo if x.op is Ops.END])
    input_ranges = tuple(dict.fromkeys(x for x in (*range_srcs, *topo)
      if x.op is Ops.RANGE and x not in reduce_range and x not in ended_ranges))

    # Check for composite reduce (multi-accumulator)
    if composite is not None:

      input_specs = getattr(composite, "input_specs", ())
      auxiliary_inputs = extra_srcs[-len(input_specs):] if input_specs else ()
      # Score-expanded-Hd is semantically repeated across the output lane.
      # Select one representative score lane; the declared logical V input
      # still supplies the output-Hd value for the accumulator update.
      if input_specs and input_specs[0].primary_repeated and inp.dtype.count > 1:
        inp = _select_vector_lane_at(inp, reduce_range[-1])
      # Auxiliary V loads are admitted only after a real shaped-fragment
      # lowering exists.  Until then the generic scalar reducer must remain
      # authoritative; passing a lane-shaped LOAD here creates invalid ALU
      # shape pairs and can corrupt unrelated CPU/AMD attention kernels.
      v_inp = None
      if composite.combine_fn == "online_softmax_state":
        if len(auxiliary_inputs) != 1 or len(input_specs) != 1:
          raise RuntimeError("online_softmax_state requires exactly one declared V input")
        spec = input_specs[0]
        lane_group = prod(getattr(composite, "lane_shapes", ((), (), ()))[2] or (1,))
        v_inp = _vectorize_live_v_index(auxiliary_inputs[0], reduce_range, lane_group, composite.slots[-1].dtype)
        if v_inp is None:
          axis_map = tuple(spec.axis_map[:-1]) + (-1,) if lane_group > 1 and spec.axis_map else spec.axis_map
          v_inp = _load_v_at_reduce_pos(auxiliary_inputs[0], composite, input_ranges, reduce_range, inp._shape,
                                        axis_map=axis_map, lane_group=lane_group)

      # Create accumulators (common to all combines)
      accs = []
      acc_reads = []
      for i, slot in enumerate(composite.slots):
        physical_dtype = physical_composite_slot_dtype(composite, i)
        ident = red.const(physical_dtype, slot.identity if slot.identity is not None else identity_element(slot.op, physical_dtype.scalar()))
        acc = UOp.placeholder((1,), physical_dtype, ctx.acc_num, AddrSpace.REG)
        ctx.acc_num += 1
        acc_init = acc.after(*input_ranges).index(UOp.const(dtypes.weakint, 0)).store(ident)
        acc_read = acc.after(acc_init, *reduce_range).index(UOp.const(dtypes.weakint, 0))
        accs.append(acc)
        acc_reads.append(acc_read)

      from tinygrad.codegen.late.composite_combines import COMBINE_REGISTRY, _independent_slots, validate_composite_state
      combine_fn = COMBINE_REGISTRY.get(composite.combine_fn)
      if combine_fn is None and composite.combine_fn is not None:
        raise RuntimeError(f"unknown composite combine {composite.combine_fn!r}")
      combine_fn = combine_fn or _independent_slots
      result = validate_composite_state(combine_fn(ctx, accs, acc_reads, inp, composite, input_ranges, reduce_range, red, v_inp=v_inp), composite)
      return UOp(Ops.TUPLE, dtypes.void, result if isinstance(result, tuple) else (result,)).replace(tag=("composite_reduce", composite))

    if not isinstance(red.arg[0], Ops):
      raise RuntimeError(f"non-ALU reduction reached ordinary lowering: arg={red.arg!r}, composite={composite!r}, src_ops={[x.op for x in red.src]}")
    identity = red.const(red.dtype, identity_element(red.arg[0], red.dtype.scalar()))
    acc = UOp.placeholder((1,), red.dtype, ctx.acc_num, AddrSpace.REG).replace(tag=red.tag if isinstance(red.tag, RegisterResidentAccumulator) else None)
    acc_init = acc.after(*input_ranges).index(UOp.const(dtypes.weakint, 0)).store(identity)
    lst = [acc.after(acc_init, *reduce_range).index(UOp.const(dtypes.weakint, 0))] + lst  # put acc as the first element
    ctx.acc_num += 1
  if not isinstance(red.arg[0], Ops):
    raise RuntimeError(f"non-ALU no-range reduction: arg={red.arg!r}, composite={composite!r}, src_ops={[x.op for x in red.src]}")
  ret = functools.reduce(lambda x,y: x.alu(red.arg[0], y), lst)
  if len(reduce_range) == 0: return ret
  end = acc.index(UOp.const(dtypes.weakint, 0)).store(ret).end(*reduce_range).rtag("mergeable")
  return acc.after(end).index(UOp.const(dtypes.weakint, 0))

def merge_reduce_ends(ctx:ReduceContext, sink:UOp):
  # merge ENDs that share the same range and nesting context (only those created by reduce_to_acc)
  # ENDs at different nesting depths get cloned RANGEs so each RANGE maps to one END
  range_to_ends: dict[tuple[UOp, ...], list[UOp]] = {}
  for u in sink.backward_slice:
    if u.op is Ops.END and u.tag == "mergeable": range_to_ends.setdefault(u.src[1:], []).append(u)
  subs: dict[UOp, UOp] = {}
  next_axis = max((u.arg[0] for u in sink.backward_slice if u.op is Ops.RANGE), default=-1) + 1
  for r, ends in range_to_ends.items():
    if len(ends) <= 1: continue
    by_ctx: dict[frozenset[UOp], list[UOp]] = {}
    for e in ends: by_ctx.setdefault(frozenset(e.ranges), []).append(e)
    for i, group in enumerate(by_ctx.values()):
      tr = r if i == 0 else tuple(rr.replace(arg=(next_axis + j, *rr.arg[1:])) for j, rr in enumerate(r))
      if i > 0: next_axis += len(r)
      mapped = [e.substitute(dict(zip(r, tr))) if i > 0 else e for e in group]
      merged = mapped[0] if len(mapped) == 1 else UOp.group(*(e.src[0] for e in mapped)).end(*tr)
      for e in group: subs[e] = merged
  return sink.substitute(subs) if subs else None

def _resolve_reduce_slot_pm(slot):
    from tinygrad.codegen.late.composite_combines import resolve_reduce_slot_tensor, resolve_composite_reduce_slot_prebufferize
    # Expander can leave a validated composite_view INDEX around the tuple.
    # Resolve that form before applying the stricter direct-tuple resolver.
    # UOp does not have boolean truth semantics; using ``or`` here evaluates
    # a successfully resolved value as a scalar expression and can crash (or
    # silently mis-route) before the provenance-aware fallback runs.
    resolved = resolve_reduce_slot_tensor(slot)
    return resolved if resolved is not None else resolve_composite_reduce_slot_prebufferize(slot)

def _project_deferred_carrier(carrier:UOp, slot:int) -> UOp|None:
  """Project a physical slot through optimizer-only unary state wrappers."""
  if carrier.op is Ops.TUPLE:
    if not isinstance(slot, int) or not 0 <= slot < len(carrier.src):
      raise RuntimeError(f"invalid composite reduction slot {slot}")
    return carrier.src[slot]
  if carrier.op is Ops.UNROLL and len(carrier.src) == 1:
    inner = _project_deferred_carrier(carrier.src[0], slot)
    if inner is None: return None
    return inner if inner.dtype.count == 1 else UOp(Ops.UNROLL, inner.dtype.scalar(), (inner,), carrier.arg)
  return None

def validate_deferred_state_liveness(state:UOp) -> bool:
  """Validate physical acc/l update ownership before devectorization erases it."""
  carrier = state.src[0]
  while carrier.op is Ops.UNROLL and len(carrier.src) == 1: carrier = carrier.src[0]
  if carrier.op is not Ops.TUPLE or state.arg.normalize_by is None: return False
  if not (0 <= state.arg.slot < len(carrier.src) and 0 <= state.arg.normalize_by < len(carrier.src)): return False
  acc, den = carrier.src[state.arg.slot], carrier.src[state.arg.normalize_by]
  acc_ends, den_ends = ([u for u in x.backward_slice if u.op is Ops.END] for x in (acc, den))
  if not acc_ends or not den_ends: return False
  update_stores = [u for end in acc_ends for u in (end.src[0], *end.src[0].backward_slice) if u.op is Ops.STORE]
  if acc.op is not Ops.INDEX: return False
  read_base = acc.src[0]
  while read_base.op is Ops.AFTER: read_base = read_base.src[0]
  for store in update_stores:
    write_idx = store.src[0]
    if write_idx.op is not Ops.INDEX: continue
    write_base = write_idx.src[0]
    while write_base.op is Ops.AFTER: write_base = write_base.src[0]
    rhs = store.src[-1]
    rhs_ops = {u.op for u in (rhs, *rhs.backward_slice)}
    vector_inputs = []
    for u in (rhs, *rhs.backward_slice):
      if u.op is not Ops.INDEX or u.dtype.count != rhs.dtype.count: continue
      base = u.src[0]
      while base.op is Ops.AFTER: base = base.src[0]
      if base is not read_base: vector_inputs.append(u)
    rhs_is_lane_update = Ops.ADD in rhs_ops and Ops.MUL in rhs_ops and Ops.EXP2 in rhs_ops and Ops.STACK in rhs_ops and bool(vector_inputs)
    if read_base is write_base and acc.src[1:] == write_idx.src[1:] and \
       acc.dtype.count == rhs.dtype.count and acc.dtype.count > 1 and rhs_is_lane_update: return True
  return False

def lower_deferred_reduce_slot(state:UOp):
  """Resolve once after REDUCE lowering, consuming the carrier rather than materializing it."""
  if state.op is not Ops.DEFERRED_REDUCE_SLOT: return None
  result = _project_deferred_carrier(state.src[0], state.arg.slot)
  if result is None:
    return None
  if state.arg.normalize_by is not None and any(u.op is Ops.TUPLE for u in state.src[0].backward_slice) and \
     not validate_deferred_state_liveness(state):
    raise RuntimeError("physical deferred acc projection lost its KV update END")
  if not isinstance(state.arg.slot, int):
    raise RuntimeError(f"invalid deferred composite slot {state.arg.slot}")
  cursor = 1
  if state.arg.normalize_by is not None:
    den = _project_deferred_carrier(state.src[0], state.arg.normalize_by)
    if den is None: raise RuntimeError("invalid deferred normalization slot")
    den = den if den.dtype.count == result.dtype.count else den.broadcast(result.dtype.count)
    result = result.alu(Ops.MUL, den.alu(Ops.RECIPROCAL))
  for op, arg, count in state.arg.views:
    extra = state.src[cursor:cursor+count]
    if len(extra) != count: raise RuntimeError("truncated deferred composite view sources")
    result, cursor = UOp(op, state.dtype, (result, *extra), arg), cursor+count
  if cursor != len(state.src): raise RuntimeError("unused deferred composite view sources")
  return result

def lower_deferred_reduce_owner(owner:UOp):
  if owner.op is not Ops.DEFERRED_REDUCE_OWNER or owner.src[0].op is not Ops.TUPLE: return None
  return owner.src[0]

def lower_composite_accumulator(state:UOp):
  """Lower a heterogeneous composite state carrier to an explicit tuple.

  COMPOSITE_ACCUMULATOR is intentionally backend-neutral: its ``arg`` records
  the logical shape of each state slot while sources carry the actual scalar or
  vector values.  Do not flatten slots into one vector (that loses the scalar
  m/l versus vector acc ABI).  The tuple is the scheduler-visible primitive;
  later register lowering can allocate each member independently.
  """
  if state.op is not Ops.COMPOSITE_ACCUMULATOR: return None
  shapes = state.arg if isinstance(state.arg, tuple) else ()
  if len(shapes) != len(state.src):
    raise RuntimeError(f"composite accumulator slot/source mismatch: {len(shapes)} != {len(state.src)}")
  for i, (src, shape) in enumerate(zip(state.src, shapes)):
    if shape is not None and tuple(src.shape) != tuple(shape):
      raise RuntimeError(f"composite accumulator slot {i} shape mismatch: {src.shape} != {shape}")
  return UOp(Ops.TUPLE, dtypes.void, state.src).replace(tag=("composite_accumulator", shapes))

def composite_reduce_state_adapter(values:tuple[UOp, ...], shapes:tuple[tuple|None, ...]):
  """Opt-in adapter for synthetic composite-reduce state experiments.

  This deliberately does not alter bounded attention.  It provides a small
  backend-neutral bridge from already-computed slot values to the explicit
  heterogeneous carrier used by primitive ABI tests.
  """
  if len(values) != len(shapes):
    raise ValueError(f"composite state arity mismatch: {len(values)} != {len(shapes)}")
  return UOp(Ops.COMPOSITE_ACCUMULATOR, values[0].dtype if values else dtypes.void, values, shapes)

pm_reduce = PatternMatcher([
  (UPat(Ops.COMPOSITE_ACCUMULATOR, name="state"), lower_composite_accumulator),
  # REDUCE -> DEFINE_ACC+ASSIGN, then merge ENDs with same range
  (UPat(Ops.REDUCE, name="red"), reduce_to_acc),
  (UPat(Ops.DEFERRED_REDUCE_OWNER, name="owner"), lower_deferred_reduce_owner),
  (UPat(Ops.DEFERRED_REDUCE_SLOT, name="state"), lower_deferred_reduce_slot),
  # REDUCE_SLOT is only a projection from the graph-local TUPLE result.
  (UPat(Ops.REDUCE_SLOT, src=(UPat(),), name="slot"), _resolve_reduce_slot_pm),
  (UPat(Ops.SINK, name="sink"), merge_reduce_ends),
  # tensor core built in accumulate
  (UPat(Ops.WMMA, name="wmma") + UPat.var("add"),
    lambda add, wmma: UOp(wmma.op, wmma.dtype, (wmma.src[0], wmma.src[1], wmma.src[2]+add), wmma.arg)),
])
