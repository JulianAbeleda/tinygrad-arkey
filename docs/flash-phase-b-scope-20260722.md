# Phase B Scope: 4D composite REDUCE through attention graph

## Root cause
The 1D composite REDUCE tests pass (arange inputs). The 4D attention graph
fails because the scheduler creates different range/axis structures for
multi-dimensional inputs.

## What differs between 1D and 4D

**1D (arange(1..N)):** Flat tensor, all axes reduced. Input to reduce_to_acc
is scalar per-element. No non-reduced axes to UPCAST. Works with NOOPT=1.

**4D ((q@k.T)*scale, shape (B,H,T,KV)):** Multi-dimensional. Non-reduced
axes (B,H,T) remain. The scheduler may UPCAST the T axis, producing vector
inputs to the REDUCE body. Fails with shape mismatch at Ops.MAX: [(), (8,)].

## Required changes (3 files)

### 1. expander.py: fix_reduce_unroll skip ✓ (COMMITTED: 433da4696)
`if isinstance(x.arg[0], CompositeReduce): return None`
Preserves loop structure through the expander.

### 2. devectorizer.py: vector-aware online_softmax_l combine
Detect vector input via inp_val.shape[0]. If vector, iterate lanes with
gep(lane) and accumulate state per lane. Use scalar accumulator registers.

### 3. devectorizer.py: _devec_broadcast_reg_store
Handle STORE(STACK(reg_slot,...,reg_slot), val) where all slots are the
same register. Emit a scalar store instead of failing with "expression
is not assignable" at render time.

### 4. devectorizer.py: pm_reduce caching
The PatternMatcher compiles reduce_to_acc at import time. File edits to
reduce_to_acc don't take effect unless pm_reduce is recompiled. Fix:
either use compiled=False or recompile at use time.

## Test plan
1. Existing 1D tests (9 passed) — must not regress
2. 4D attention test: rel_err <= 1e-2 without NOOPT
3. Full WMMA suite unregressed
