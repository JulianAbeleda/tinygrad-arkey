# M0 Design: Multi-Output REDUCE via Ops.REDUCE_SLOT

## Representation

- Composite REDUCE: `UOp(Ops.REDUCE, composite_dtype, src=(input, ranges...), arg=(CompositeReduce(slots=(...)), axes))`
- Consumer reads slot i: `UOp(Ops.REDUCE_SLOT, slot_dtype, src=(reduce_uop,), arg=i)`
  - `slot_dtype = slots[i].dtype`
  - `arg = i` (integer index into slots tuple)
- The REDUCE itself carries `dtypes.void` or the last slot's dtype (TBD by what survives the size formula at rangeify.py:423)

## Lowering (reduce_to_acc)

`reduce_to_acc` currently:
1. Creates one DEFINE_ACC per slot
2. Returns `accs[-1].after(end).index(0)` — only exposes the last slot

With REDUCE_SLOT:
- `reduce_to_acc` returns None for the REDUCE (doesn't replace it with a single value)
- `REDUCE_SLOT(reduce, i)` resolves to `accs[i].after(end_i).index(0)` — the final value of accumulator i
- The accumulator end-chain carries all slots; each REDUCE_SLOT reads its specific one

## Dispatch audit

Every code site that switches on `Ops` must handle REDUCE_SLOT:

| Site | What it does | REDUCE_SLOT action |
|------|-------------|-------------------|
| `ops.py` GroupOp buckets | Categorizes ops | Add to appropriate bucket (passthrough like GEP/CAST) |
| `ops.py` identity_element | Returns identity for reduce ops | N/A (REDUCE_SLOT is a read, not a reduce) |
| `ops.py` Ops enum | Op definitions | Add REDUCE_SLOT |
| `spec.py:190` REDUCE rule | Validates REDUCE args | Add REDUCE_SLOT spec |
| `devectorizer.py` reduce_to_acc | Lowers REDUCE→DEFINE_ACC | Resolve REDUCE_SLOT→acc[i] |
| `symbolic.py` | Constant folding, GEP identity | Pass through (or fold if reduce is const) |
| `rangeify.py:423` | Size formula: `prod(shape)//dtype.count` | REDUCE_SLOT dtype is per-slot, no issue |
| `cstyle.py` renderer | Renders ops to C | Should never see REDUCE_SLOT (lowered before render) |
| `isa/` renderers | ISA rendering | Should never see REDUCE_SLOT |

## Composite REDUCE dtype handling

The REDUCE node's dtype must survive rangeify's size formula at line 423:
```python
size = prod(x.shape) // x.dtype.count
```
For a composite REDUCE with shape `(1,)` and dtype say `dtypes.float`, `size = 1 // 1 = 1` — fine.
If we use `dtypes.void`, `size = 1 // 1 = 1` — also fine (void.count = 1).

Recommendation: use the last slot's dtype for the REDUCE dtype, and each REDUCE_SLOT uses its own slot's dtype.
