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
    """Online-softmax: (m, l) state with correction-based combine."""
    LOG2E = UOp.const(dtypes.float32, 1.4426950408889634)
    NEG1 = UOp.const(dtypes.float32, -1.0)
    
    inp_score = inp
    m_old, l_old = acc_reads
    
    m_new = m_old.alu(Ops.MAX, inp_score)
    diff = m_old.alu(Ops.ADD, m_new.alu(Ops.MUL, NEG1))
    corr = diff.alu(Ops.MUL, LOG2E).alu(Ops.EXP2)
    score_shifted = inp_score.alu(Ops.ADD, m_new.alu(Ops.MUL, NEG1))
    exp_score = score_shifted.alu(Ops.MUL, LOG2E).alu(Ops.EXP2)
    l_new = l_old.alu(Ops.MUL, corr).alu(Ops.ADD, exp_score)
    
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
