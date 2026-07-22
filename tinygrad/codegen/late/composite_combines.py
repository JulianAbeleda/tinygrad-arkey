"""Combine functions for composite REDUCE. Each is a callable that takes
(ctx, accs, acc_reads, inp, composite, input_ranges, reduce_range, red)
and returns a UOp that replaces the REDUCE in the graph.

The combine is responsible for:
- Computing new accumulator values from the current reads and input
- Storing the new values back to accumulators  
- Returning a replacement UOp anchored on all accumulator ends

This keeps reduce_to_acc completely combine-agnostic.
"""
import functools
from tinygrad.uop.ops import UOp, Ops, dtypes, AxisType, AddrSpace
from tinygrad.uop.ops import identity_element, CompositeReduce, AccumulatorSlot

def _independent_slots(ctx, accs, acc_reads, inp, composite, input_ranges, reduce_range, red, v_inp=None):
    """Default combine: each slot independently reduces the input using its op."""
    results = []
    for i, (slot, acc, acc_read) in enumerate(zip(composite.slots, accs, acc_reads)):
        inp_lst = _horizontal_reduce(inp, slot.dtype)
        lst = [acc_read] + inp_lst
        ret = functools.reduce(lambda x, y: x.alu(slot.op, y), lst)
        end = acc.index(UOp.const(dtypes.weakint, 0)).store(ret).end(*reduce_range).rtag("mergeable")
        results.append(acc.after(end).index(UOp.const(dtypes.weakint, 0)))
    return tuple(result.after(*[r.src[0] for r in results]) for result in results)

def online_softmax_l(ctx, accs, acc_reads, inp, composite, input_ranges, reduce_range, red, v_inp=None):
    """Online-softmax: (m, l) state with correction-based combine.
    
    Decomposes the input into scalar elements (horizontal reduce) and iterates,
    applying the per-element online softmax combine step for each element.
    """
    LOG2E = UOp.const(dtypes.float32, 1.4426950408889634)
    NEG1 = UOp.const(dtypes.float32, -1.0)
    
    # Decompose input into scalar elements (like _independent_slots does)
    inp_lst = _horizontal_reduce(inp, composite.slots[0].dtype)
    m_old, l_old = acc_reads
    
    m_new, l_new = m_old, l_old
    for inp_score in inp_lst:
        m_new = m_new.alu(Ops.MAX, inp_score)
        diff = m_old.alu(Ops.ADD, m_new.alu(Ops.MUL, NEG1))
        corr = diff.alu(Ops.MUL, LOG2E).alu(Ops.EXP2)
        score_shifted = inp_score.alu(Ops.ADD, m_new.alu(Ops.MUL, NEG1))
        exp_score = score_shifted.alu(Ops.MUL, LOG2E).alu(Ops.EXP2)
        l_new = l_new.alu(Ops.MUL, corr).alu(Ops.ADD, exp_score)
        m_old, l_old = m_new, l_new
    
    ends = [acc.index(UOp.const(dtypes.weakint, 0)).store(new_val).end(*reduce_range).rtag("mergeable")
            for acc, new_val in zip(accs, [m_new, l_new])]
    return tuple(acc.after(*ends).index(UOp.const(dtypes.weakint, 0)) for acc in accs)

def online_softmax(ctx, accs, acc_reads, inp, composite, input_ranges, reduce_range, red, v_inp=None):
    """Online-softmax: (m, l, acc) state with correction + acc/l output."""
    LOG2E = UOp.const(dtypes.float32, 1.4426950408889634)
    NEG1 = UOp.const(dtypes.float32, -1.0)
    
    # Score: inp is the reduction input (score value at current KV position)
    inp_score = inp
    # V: provided by devectorizer from composite.v_uop, else fall back to gep
    if v_inp is None:
        inp_v = inp if inp.dtype.count == 1 else inp.gep(1)
    else:
        inp_v = v_inp

    m_old, l_old, acc_old = acc_reads
    
    m_new = m_old.alu(Ops.MAX, inp_score)
    diff = m_old.alu(Ops.ADD, m_new.alu(Ops.MUL, NEG1))
    corr = diff.alu(Ops.MUL, LOG2E).alu(Ops.EXP2)
    score_shifted = inp_score.alu(Ops.ADD, m_new.alu(Ops.MUL, NEG1))
    exp_score = score_shifted.alu(Ops.MUL, LOG2E).alu(Ops.EXP2)
    l_new = l_old.alu(Ops.MUL, corr).alu(Ops.ADD, exp_score)
    acc_new = acc_old.alu(Ops.MUL, corr).alu(Ops.ADD, exp_score.alu(Ops.MUL, inp_v))
    
    ends = [acc.index(UOp.const(dtypes.weakint, 0)).store(new_val).end(*reduce_range).rtag("mergeable")
            for acc, new_val in zip(accs, [m_new, l_new, acc_new])]
    return tuple(acc.after(*ends).index(UOp.const(dtypes.weakint, 0)) for acc in accs)

# Registry: combine_fn string -> callable
COMBINE_REGISTRY = {
    None: _independent_slots,
    "online_softmax_l": online_softmax_l,
    "online_softmax": online_softmax,
}

def _horizontal_reduce(inp: UOp, out_dtype):
    """Split vector input into scalar components."""
    if inp.dtype != out_dtype and inp.dtype.count > out_dtype.count:
        horizontal_amount = inp.dtype.count // out_dtype.count
        return [inp.gep(tuple(range(i, inp.dtype.count, horizontal_amount))) for i in range(0, horizontal_amount)]
    return [inp]

def _combine_step_online_softmax_l(m_old, l_old, score):
    """Per-element step for online-softmax (m,l) combine."""
    from tinygrad.uop.ops import UOp, Ops, dtypes
    LOG2E = UOp.const(dtypes.float32, 1.4426950408889634)
    NEG1 = UOp.const(dtypes.float32, -1.0)
    m_new = m_old.alu(Ops.MAX, score)
    diff = m_old.alu(Ops.ADD, m_new.alu(Ops.MUL, NEG1))
    corr = diff.alu(Ops.MUL, LOG2E).alu(Ops.EXP2)
    score_shifted = score.alu(Ops.ADD, m_new.alu(Ops.MUL, NEG1))
    exp_score = score_shifted.alu(Ops.MUL, LOG2E).alu(Ops.EXP2)
    l_new = l_old.alu(Ops.MUL, corr).alu(Ops.ADD, exp_score)
    return m_new, l_new

def _combine_step_independent(op, acc, elem):
    """Per-element step for independent-slot combine."""
    return acc.alu(op, elem)

def _combine_step_online_softmax(m_old, l_old, acc_old, score, v_val):
    """Per-element step for online-softmax (m,l,acc) combine."""
    from tinygrad.uop.ops import UOp, Ops, dtypes
    LOG2E = UOp.const(dtypes.float32, 1.4426950408889634)
    NEG1 = UOp.const(dtypes.float32, -1.0)
    m_new = m_old.alu(Ops.MAX, score)
    diff = m_old.alu(Ops.ADD, m_new.alu(Ops.MUL, NEG1))
    corr = diff.alu(Ops.MUL, LOG2E).alu(Ops.EXP2)
    score_shifted = score.alu(Ops.ADD, m_new.alu(Ops.MUL, NEG1))
    exp_score = score_shifted.alu(Ops.MUL, LOG2E).alu(Ops.EXP2)
    l_new = l_old.alu(Ops.MUL, corr).alu(Ops.ADD, exp_score)
    acc_new = acc_old.alu(Ops.MUL, corr).alu(Ops.ADD, exp_score.alu(Ops.MUL, v_val))
    return m_new, l_new, acc_new

# Registry: combine_fn -> (step_fn, num_slots, identity_getter, elements_per_step)
COMBINE_STEP_REGISTRY = {
    None: (lambda *args: args[0].alu(args[1].op, args[2]) if len(args) == 3 else None, None, None, 1),
    "online_softmax_l": (_combine_step_online_softmax_l, 2, lambda slot: slot.identity, 1),
    "online_softmax": (_combine_step_online_softmax, 3, lambda slot: slot.identity, 2),
}

def _handle_no_range_generic(inp, composite, red, auxiliary_inputs=()):
    """Generic no-range handler: iterate over elements using combine step function."""
    from tinygrad.uop.ops import Ops, dtypes, identity_element
    inp_lst = _horizontal_reduce(inp, composite.slots[0].dtype)
    
    if composite.combine_fn is None:
        # Independent slots: reduce each independently
        results = []
        for slot in composite.slots:
            slot_lst = _horizontal_reduce(inp, slot.dtype)
            results.append(functools.reduce(lambda x,y: x.alu(slot.op, y), slot_lst))
        return tuple(results)
    
    entry = COMBINE_STEP_REGISTRY.get(composite.combine_fn)
    if entry is None or entry[0] is None:
        raise RuntimeError(f"Unknown composite combine: {composite.combine_fn}")
    step_fn, num_slots, _, elems_per_step = entry
    
    # Initialize state from slot identities
    state = []
    for slot in composite.slots:
        ident_val = slot.identity if slot.identity is not None else identity_element(slot.op, slot.dtype.scalar())
        state.append(red.const(slot.dtype, ident_val))
    
    # Separate auxiliary inputs are lane-aligned logical elements. The packed
    # representation remains supported for composites that place every input
    # in the primary vector.
    if auxiliary_inputs:
        if len(auxiliary_inputs) + 1 != elems_per_step:
            raise RuntimeError(f"composite {composite.combine_fn!r} expects {elems_per_step} logical inputs, "
                               f"got one primary and {len(auxiliary_inputs)} auxiliary inputs")
        auxiliary_lanes = [_horizontal_reduce(x, composite.slots[-1].dtype) for x in auxiliary_inputs]
        if any(len(x) != len(inp_lst) for x in auxiliary_lanes):
            raise RuntimeError("composite auxiliary inputs must have the same horizontal lane count as the primary input")
        for i, primary in enumerate(inp_lst):
            state = list(step_fn(*state, primary, *(x[i] for x in auxiliary_lanes)))
    else:
        for i in range(0, len(inp_lst), elems_per_step):
            group = inp_lst[i:i + elems_per_step]
            state = list(step_fn(*state, *group))
    
    return tuple(state)

def _lower_composite_no_range_pm(red):
    """PatternMatcher callback: lower composite REDUCE with no ranges.

    When the REDUCE has no ranges and a scalar input (pre-rangeify), creates
    synthetic RANGE sources so reduce_to_acc can handle them.  When the input
    is already a vector (post-expander STACK form), uses the no-range generic
    handler.
    """
    from tinygrad.uop.ops import CompositeReduce, AxisType
    composite = red.arg[0]
    if not isinstance(composite, CompositeReduce) and not (hasattr(composite, "slots") and hasattr(composite, "combine_fn")): return None
    if any(x.op is Ops.RANGE for x in red.src[1:]): return None

    # Pre-rangeify: the REDUCE has an axis but no ranges yet.  Create synthetic
    # RANGEs so the range path in reduce_to_acc can lower it correctly.
    axis = red.arg[1]
    if axis and red.src[0].dtype.count == 1:
        try:
            shape = red.src[0].shape
            rngs = tuple(UOp.range(UOp.const(dtypes.weakint, shape[i]), i, AxisType.REDUCE) for i in axis)
            return UOp(Ops.REDUCE, red.dtype, src=(red.src[0],) + rngs + red.src[1:], arg=(composite, ()))
        except Exception:
            pass

    # Expander/rangeify may leave auxiliary carriers and RANGE context after
    # the primary input.  Only sources declared by CompositeInputSpec are
    # logical combine inputs; range carriers are never V tensors.
    candidates = tuple(x for x in red.src[1:] if x.op is not Ops.RANGE)
    ninputs = len(getattr(composite, "input_specs", ()))
    auxiliary_inputs = candidates[-ninputs:] if ninputs else ()
    result = _handle_no_range_generic(red.src[0], composite, red, auxiliary_inputs)
    return UOp(Ops.TUPLE, dtypes.void, result)

def resolve_reduce_slot_tensor(slot):
    """Graph-local projection from the structured composite reduction result."""
    src = slot.src[0]
    # Horizontal expansion can leave the consumed reduction-axis carrier
    # around the structured result. It is not an output expansion: every
    # TUPLE member is already the fully reduced scalar state.
    if src.op is Ops.UNROLL and len(src.src) == 1 and src.src[0].op is Ops.TUPLE:
      src = src.src[0]
    if src.op is not Ops.TUPLE: return None
    if not isinstance(slot.arg, int) or not 0 <= slot.arg < len(src.src):
      raise RuntimeError(f"invalid composite reduction slot {slot.arg}")
    # Project directly while the structured result is still in compiler IR.
    # This leaves no TUPLE/GETTUPLE operation for the renderer and preserves
    # the one reduction's shared END dependencies.
    return src.src[slot.arg]
