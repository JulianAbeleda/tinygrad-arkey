# Phase 0 Design: Composite-Accumulator REDUCE

Date: 2026-07-22
Author: Julian Abeleda & Claude Opus 4.8

## 1. Problem restatement

Today `Ops.REDUCE` lowers to a single scalar accumulator via `reduce_to_acc`
(devectorizer.py:369). The accumulator is init'd to `identity_element(op, dtype)`,
updated with `op(acc, contrib)`, and closed with `end(reduce_range)`.

Online-softmax requires carrying `(m, l, acc)` state across KV blocks where:
- `m` = running max (MAX reduce)
- `l` = running sum of exp(score - m) (ADD reduce with correction)
- `acc` = running weighted sum (ADD reduce with correction)

Carrying three values with a correction term cannot be expressed as a single
scalar accumulator with a simple binary op. The REDUCE primitive must be
extended to support a **composite accumulator** with a custom combine function.

## 2. Composite accumulator representation

### 2.1 REDUCE arg extension

Current: `arg = (op: Ops, axes: tuple[int, ...])`

Extended: `arg = (op_or_composite, axes)` where `op_or_composite` is either:
- An `Ops` value (backward compatible — existing single-op REDUCE)
- A `CompositeReduce` namedtuple for multi-accumulator reduces

```python
class CompositeReduce(NamedTuple):
    slots: tuple[AccumulatorSlot, ...]  # one per carried value
    combine: UOp                        # UOp sub-graph: (state, element) -> new_state

class AccumulatorSlot(NamedTuple):
    op: Ops          # underlying reduce op (ADD, MAX, MUL)
    dtype: DType     # dtype of this slot
    identity: PyConst # identity element
    name: str        # "m", "l", "acc" for debugging
```

### 2.2 Online-softmax composite spec

```python
COMPOSITE_ONLINE_SOFTMAX = CompositeReduce(
    slots=(
        AccumulatorSlot(op=Ops.MAX, dtype=dtypes.float, identity=float('-inf'), name='m'),
        AccumulatorSlot(op=Ops.ADD, dtype=dtypes.float, identity=0.0, name='l'),
        AccumulatorSlot(op=Ops.ADD, dtype=dtypes.float.vec(Hd), identity=0.0, name='acc'),
    ),
    combine=... # UOp sub-graph (see §3)
)
```

### 2.3 Why this representation

- **Backward compatible:** a bare `Ops` in arg[0] is the existing single-op REDUCE.
  All existing code paths (TC opt, rangeify, devectorizer) are unchanged.
- **UOp-native:** the combine is a UOp sub-graph, so it participates in the
  same rewrite/substitute/graph_rewrite pipeline as everything else.
- **Explicit slots:** each accumulator has a named slot with its own op, dtype,
  and identity. The devectorizer can lower each slot to its own DEFINE_ACC.

## 3. Combine function as a UOp sub-graph

The `combine` field is a UOp tree that takes two inputs:
- `state`: the current composite state (struct of m, l, acc)
- `element`: the new element (score scalar + v vector for one KV position)

And produces a new state. The UOp tree uses placeholder nodes for state and
element, which are substituted with the actual accumulators during lowering.

For online-softmax:

```
combine(state, element):
    m_old, l_old, acc_old = state.m, state.l, state.acc
    score, v = element.score, element.v

    m_new = max(m_old, score)
    correction = exp(m_old - m_new)
    l_new = l_old * correction + exp(score - m_new)
    acc_new = acc_old * correction + exp(score - m_new) * v

    return (m_new, l_new, acc_new)
```

In UOp form, this is a tree of ALU operations with two placeholder inputs
referencing the state slots and element slots.

### 3.1 Placeholder convention

Two placeholder UOps are created during lowering:
- `state_placeholders[i]` — reads slot i of the accumulator state
- `element_placeholders[i]` — reads slot i of the incoming element

The combine tree references these placeholders. During `reduce_to_acc`, the
placeholders are substituted with the actual accumulator reads and element
computations.

## 4. Lowering in reduce_to_acc (devectorizer.py)

Current flow:
```
1. Create acc = UOp.placeholder((1,), dtype, ...)  # single scalar
2. Init: acc.store(identity_element(op, dtype))
3. Loop: for each element, acc.store(op(acc.read(), element))
4. End: acc.end(reduce_range)
```

Extended flow for composite:
```
1. Create acc[i] = UOp.placeholder((1,), slot.dtype, ...) for each slot
2. Init: for each slot, acc[i].store(slot.identity)
3. Loop: for each element:
   a. Read current state: state[i] = acc[i].read()
   b. Compute new state = substitute(combine_tree, {state_ph: state, elem_ph: element})
   c. For each slot, acc[i].store(new_state[i])
4. End: all acc[i].end(reduce_range)
```

The key change: instead of `op(acc, contrib)`, the update is the combine
sub-graph evaluated with the current state and new element.

## 5. WMMA attachment (TC opt, postrange.py)

The TC opt at postrange.py:305-307 already has a hook for "epilogue reduction
around the dot-product": it selects the inner ADD+MUL (the dot product) as the
TC candidate while keeping an outer epilogue reduction.

For the composite reduce:
- The QKᵀ contraction (ADD over Hd, body=MUL) is an inner reduce inside the
  KV composite reduce.
- The PV contraction (ADD over Hd), similarly.
- The TC opt `self.reduceops` must find these inner ADD+MUL reduces inside
  the composite reduce body.

### 5.1 TC finding strategy

Currently `get_single_element([... tag=="TC"])` at postrange.py:391 assumes
ONE TC-tagged reduce. For composite, the body may contain TWO TC-eligible
reduces (QKᵀ and PV).

Proposed: tag BOTH inner contractions with distinct tags (`"TC_QK"`, `"TC_PV"`),
apply TC to each independently. The outer composite reduce kernel contains
both WMMA'd contractions.

### 5.2 Geometry

The composite reduce has:
- Outer KV range (REDUCE) — the composite accumulator carries state across this
- Inner Hd range (REDUCE) — the QKᵀ dot product, WMMA'd
- Inner KV_block range (REDUCE) — the PV dot product, WMMA'd

TC attaches to both inner reduces. The outer composite reduce stays as a
regular REDUCE (with composite accumulator, not WMMA).

## 6. Attention expression → composite reduce

The user writes standard attention:
```python
scores = (q @ k.transpose(-1, -2)) * scale
probs = scores.softmax(-1)
out = probs @ v
```

Rangeify (or a pre-rangeify graph_rewrite) recognizes this pattern and
replaces it with a composite REDUCE over KV that:
1. Inner: computes `score = (q @ k_block.T) * scale` (WMMA on QKᵀ)
2. Inner: updates composite state via online-softmax combine
3. Inner: computes `acc += softmax(score) @ v_block` (WMMA on PV)

The composite REDUCE is then lowered by the scheduler as a single kernel.

### 6.1 Pattern match

The graph_rewrite matches:
```
matmul(q, k.T) → softmax → matmul(?, v)
```

And restructures into:
```
composite_reduce_over_KV(q, k, v) → out
```

Where `composite_reduce_over_KV` is a REDUCE UOp with `CompositeReduce` arg.

## 7. Implementation phasing

### Phase 1 (toy): 2-accumulator reduce without online-softmax

Implement `CompositeReduce` with two slots, e.g. `(ADD, MAX)` on the same
input. Prove the lowering works: kernel shows two DEFINE_ACCs, two ENDs,
correct result.

### Phase 2 (residency): online-softmax composite reduce, no WMMA

Implement the full online-softmax combine as a UOp sub-graph. Express
attention via the composite reduce. Prove score buffer is gone and
correctness holds.

### Phase 3 (WMMA): TC attachment inside composite reduce

Make the TC opt find and WMMA the QKᵀ and PV contractions inside the
composite reduce body. Per-kernel WMMA dump shows two `__WMMA` call sites.

### Phase 4 (gate): benchmark vs SDPA, wire into 14B

## 8. Risk assessment

- **Highest risk:** the combine UOp sub-graph may not be expressible in
  the current UOp algebra (it needs exp, max, multiply-add, and
  conditional/broadcast logic all within a single kernel body).
- **Medium risk:** TC opt may not handle two WMMA'd reduces in one kernel.
- **Low risk:** the devectorizer lowering is straightforward (multiple
  DEFINE_ACCs with a shared combine).

If Phase 1 (toy 2-accumulator) cannot be completed, the project is blocked
and the primitive is not expressible in the current architecture.
