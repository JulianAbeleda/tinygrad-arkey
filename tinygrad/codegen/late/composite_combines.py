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

# Cache: maps (composite, axis) tuple -> list of slot-result UOps, and each slot result -> list
# Used by _resolve_reduce_slot to resolve REDUCE_SLOT ops after REDUCE lowering.
_composite_result_cache: dict = {}

def _independent_slots(ctx, accs, acc_reads, inp, composite, input_ranges, reduce_range, red):
    """Default combine: each slot independently reduces the input using its op."""
    results = []
    for i, (slot, acc, acc_read) in enumerate(zip(composite.slots, accs, acc_reads)):
        inp_lst = _horizontal_reduce(inp, slot.dtype)
        lst = [acc_read] + inp_lst
        ret = functools.reduce(lambda x, y: x.alu(slot.op, y), lst)
        end = acc.index(UOp.const(dtypes.weakint, 0)).store(ret).end(*reduce_range).rtag("mergeable")
        results.append(acc.after(end).index(UOp.const(dtypes.weakint, 0)))
    return results[-1]

def online_softmax_l(ctx, accs, acc_reads, inp, composite, input_ranges, reduce_range, red):
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
    return accs[-1].after(*ends).index(UOp.const(dtypes.weakint, 0))

def online_softmax(ctx, accs, acc_reads, inp, composite, input_ranges, reduce_range, red):
    """Online-softmax: (m, l, acc) state with correction + acc/l output."""
    LOG2E = UOp.const(dtypes.float32, 1.4426950408889634)
    NEG1 = UOp.const(dtypes.float32, -1.0)
    
    inp_score = inp if inp.dtype.count == 1 else inp.gep(0)
    inp_v = inp if inp.dtype.count == 1 else inp.gep(1)
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
    rcp_l = accs[1].after(ends[1]).index(UOp.const(dtypes.weakint, 0)).alu(Ops.RECIPROCAL)
    ret_acc = accs[2].after(ends[2]).index(UOp.const(dtypes.weakint, 0))
    anchored = ret_acc.after(ends[0]).after(ends[1])
    return anchored.alu(Ops.MUL, rcp_l)

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

# Registry: combine_fn -> (step_fn, num_slots, identity_getter)
# step_fn takes (state_values..., element_parts...) and returns new state
COMBINE_STEP_REGISTRY = {
    None: (lambda *args: args[0].alu(args[1].op, args[2]) if len(args) == 3 else None, None, None),
    "online_softmax_l": (_combine_step_online_softmax_l, 2, lambda slot: slot.identity),
}

def _handle_no_range_generic(inp, composite, red):
    """Generic no-range handler: iterate over elements using combine step function."""
    from tinygrad.uop.ops import Ops, dtypes, identity_element
    inp_lst = _horizontal_reduce(inp, composite.slots[0].dtype)
    
    if composite.combine_fn is None:
        # Independent slots: reduce each independently
        results = []
        for slot in composite.slots:
            slot_lst = _horizontal_reduce(inp, slot.dtype)
            results.append(functools.reduce(lambda x,y: x.alu(slot.op, y), slot_lst))
        # Populate cache for REDUCE_SLOT resolution
        _composite_result_cache[red.arg] = results
        for r in results:
            _composite_result_cache[r] = results
        return results[-1]
    
    step_fn = COMBINE_STEP_REGISTRY.get(composite.combine_fn, (None,))[0]
    if step_fn is None:
        raise RuntimeError(f"Unknown composite combine: {composite.combine_fn}")
    
    # Initialize state from slot identities
    state = []
    for slot in composite.slots:
        ident_val = slot.identity if slot.identity is not None else identity_element(slot.op, slot.dtype.scalar())
        state.append(red.const(slot.dtype, ident_val))
    
    # Iterate over elements
    for elem in inp_lst:
        state = list(step_fn(*state, elem))
    
    # Populate cache with all final state values
    _composite_result_cache[red.arg] = state
    for s in state:
        _composite_result_cache[s] = state
    return state[-1]

def _handle_no_range(inp, composite, red):
    """Handle composite REDUCE with no ranges (STACK-based, post-expander).
    Iterates over input elements using the combine logic."""
    from tinygrad.uop.ops import CompositeReduce, AccumulatorSlot
    inp_lst = _horizontal_reduce(inp, composite.slots[0].dtype)
    
    if composite.combine_fn is None:
        results = []
        for slot in composite.slots:
            slot_lst = _horizontal_reduce(inp, slot.dtype)
            results.append(functools.reduce(lambda x,y: x.alu(slot.op, y), slot_lst))
        # Populate cache for REDUCE_SLOT resolution
        _composite_result_cache[red.arg] = results
        for r in results:
            _composite_result_cache[r] = results
        return results[-1]
    
    if composite.combine_fn == "online_softmax_l":
        LOG2E = UOp.const(dtypes.float32, 1.4426950408889634)
        NEG1 = UOp.const(dtypes.float32, -1.0)
        m = red.const(composite.slots[0].dtype, composite.slots[0].identity if composite.slots[0].identity is not None else identity_element(composite.slots[0].op, composite.slots[0].dtype.scalar()))
        l = red.const(composite.slots[1].dtype, composite.slots[1].identity if composite.slots[1].identity is not None else identity_element(composite.slots[1].op, composite.slots[1].dtype.scalar()))
        for score in inp_lst:
            m_new = m.alu(Ops.MAX, score)
            diff = m.alu(Ops.ADD, m_new.alu(Ops.MUL, NEG1))
            corr = diff.alu(Ops.MUL, LOG2E).alu(Ops.EXP2)
            score_shifted = score.alu(Ops.ADD, m_new.alu(Ops.MUL, NEG1))
            exp_score = score_shifted.alu(Ops.MUL, LOG2E).alu(Ops.EXP2)
            l = l.alu(Ops.MUL, corr).alu(Ops.ADD, exp_score)
            m = m_new
        return l
    
    if composite.combine_fn == "online_softmax":
        LOG2E = UOp.const(dtypes.float32, 1.4426950408889634)
        NEG1 = UOp.const(dtypes.float32, -1.0)
        m = red.const(composite.slots[0].dtype, composite.slots[0].identity if composite.slots[0].identity is not None else identity_element(composite.slots[0].op, composite.slots[0].dtype.scalar()))
        l = red.const(composite.slots[1].dtype, composite.slots[1].identity if composite.slots[1].identity is not None else identity_element(composite.slots[1].op, composite.slots[1].dtype.scalar()))
        acc = red.const(composite.slots[2].dtype, composite.slots[2].identity if composite.slots[2].identity is not None else identity_element(composite.slots[2].op, composite.slots[2].dtype.scalar()))
        for elem in inp_lst:
            score = elem.gep(0) if elem.dtype.count > 1 else elem
            v_val = elem.gep(1) if elem.dtype.count > 1 else elem
            m_new = m.alu(Ops.MAX, score)
            diff = m.alu(Ops.ADD, m_new.alu(Ops.MUL, NEG1))
            corr = diff.alu(Ops.MUL, LOG2E).alu(Ops.EXP2)
            score_shifted = score.alu(Ops.ADD, m_new.alu(Ops.MUL, NEG1))
            exp_score = score_shifted.alu(Ops.MUL, LOG2E).alu(Ops.EXP2)
            l = l.alu(Ops.MUL, corr).alu(Ops.ADD, exp_score)
            acc = acc.alu(Ops.MUL, corr).alu(Ops.ADD, exp_score.alu(Ops.MUL, v_val))
            m = m_new
        rcp_l = l.alu(Ops.RECIPROCAL)
        return acc.alu(Ops.MUL, rcp_l)
    
    return inp_lst[-1]  # fallback: return last element

def _lower_composite_no_range_pm(red):
    """PatternMatcher callback: lower composite REDUCE with no ranges."""
    from tinygrad.uop.ops import CompositeReduce
    if not isinstance(red.arg[0], CompositeReduce): return None
    if len(red.src) >= 2: return None
    return _handle_no_range_generic(red.src[0], red.arg[0], red)

def _resolve_reduce_slot(slot):
    """PatternMatcher callback: resolve REDUCE_SLOT to the cached slot result.
    The cache is populated by REDUCE lowering (both no-range and range paths).
    After the REDUCE is lowered, REDUCE_SLOT.src[0] points to the lowering result
    which is also stored as a cache key.
    """
    cached = _composite_result_cache.get(slot.src[0])
    if cached is None: return None
    return cached[slot.arg]
