# A layout/mapping IR for tinygrad — design + first increment (2026-06-25)

## Why
The owned quantized kernels (Q4_K GEMV, attention tile) beat the scheduler entirely in the **physical realization**
layer — data layout + thread→address coalescing + tile/fragment layouts — which tinygrad does **not** represent or
search. Today the search ranges over the *schedule* (which `Ops.RANGE` gets tagged `GLOBAL/LOCAL/UPCAST/GROUP`);
**coalescing is emergent**: it falls out of which RANGE happens to land stride-1 in a buffer's `INDEX` sum
(`gpudims.py`) + a float4 peephole (`devectorizer.py fold_expanded_index`). There is no `OptOps.COALESCE`, no
thread→address map object; the one explicit thread→fragment layout (`TensorCore.swizzle`, `tc.py`) is a hand-authored
per-arch table, not searched. **The wall is the absence of a first-class, searchable layout/mapping representation.**

## Empirical grounding — and the central caveat (Phase B)
We tested the **tractable sidestep** (Marlin-style: arrange the data/access so coalescing is "already correct") by
restructuring the scheduler GEMV three ways, reading the *same already-contiguous* packed words:
| arm | tok/s | vs owned |
|---|---|---|
| owned (hand kernel) | 103 | — |
| mode 1 (fp16-logical, `x.linear`) | 50 | 2× |
| mode 2 (word-structured 256-order) | 22 | 4.6× |
| **mode 3 (word axis on the lane, forced `GROUP`)** | **14** | **7×** |

And forcing the coalescing opt directly (`MV_DEQUANT` → `GROUP(32)` fires on the K-reduce) gave **no speedup** (49.9 ≈
50.9). **Conclusion: coalescing is necessary but NOT sufficient.** No expression restructuring + forced-coalescing
reaches owned, because the owned win is the whole *kernel structure* — one packed word per lane + in-register
8-nibble dequant + 4-way block-group-K + a single REG-accumulate/cross-lane-reduce/store — not merely a coalesced
load. **This is the layout-IR design's own #1 risk, now confirmed: a layout choice can make loads coalesced and still
not match owned.** The IR makes the structure *expressible and searchable*; it does not by itself guarantee speed
(the M4 codegen pieces below are still required).

## The design (additive, on the existing RANGE/INDEX algebra)
This fork already deleted the monolithic `ShapeTracker`/`View`; layout now lives in three layers of the UOp graph:
movement ops (`Ops.RESHAPE/PERMUTE/...`), the **RANGE/INDEX address algebra** (`apply_movement_op`,
`schedule/indexing.py:134` — a small *total* layout algebra: SHRINK=+off, PERMUTE=reorder, FLIP=(s-1)-a,
EXPAND=stride-0, PAD=where-mask, RESHAPE=mixed-radix), and `RANGE.arg[-1]=AxisType` (the hardware-role tag — already
Triton-LinearLayout-style *named* dims: GLOBAL/THREAD/WARP/LOCAL/UPCAST/...). So the substrate is **better** than the
old stride-tuple. Add:

1. **`LayoutFn`** — a queryable `(Shape,Stride)` view over the fact that `Ops.INDEX(buf, idx)` *already* encodes a
   layout (an axis's "stride" = the coefficient of its RANGE in the index sum, recoverable via `split_uop(Ops.ADD)`,
   `heuristic.py:142-145`). Methods: `coeff(range)`, `is_unit_stride(range)`, `compose(other)`. The one genuinely
   missing CuTe operator is **composition** `R(c)=A(B(c))` ("apply a thread-map to a data-layout") — a graph_rewrite
   substituting one RANGE's index-expr into another's. (CuTe = `(Shape,Stride)` as a function; the algebra =
   coalesce/divide/product/complement, most already present as movement ops.)
2. **`LaneMap` (Thread-Value layout)** — generalize `TensorCore.swizzle` (the in-tree PoC: a per-operand permutation
   of named axes `l0/u1/r2` = thread→fragment) from WMMA-only into a first-class, validated object *any* kernel
   carries: `f:(thread,value)→coord` (CuTe TV layout / Hexcute "layout is part of the tensor's type").
3. **`Ops.LAYOUT_TRANSFORM`** — a declared movement-style op recording a *storage permutation* of a buffer (the
   Marlin reshuffle) as data, lowered through a new `apply_movement_op` case so it composes with reshape/permute and
   survives the simplifier; its arg names the `LaneMap` the consumer expects, so reshuffle + dequant-index are
   derived from one object (closes "the lowerer won't synthesize a coalesced load from the gather" by construction).
4. **`OptOps.COALESCE` + a static cost** — make coalescing a searchable choice scored by a *static predicate*
   (`is_unit_stride(thread_range)` on the composed layout) + vector-width = longest unit-stride run, instead of a
   benchmark. Search strategy = **Hexcute anchored propagation**: anchor on the dominant op's required layout (WMMA
   fragment, or the GEMV's per-row-workgroup `LaneMap`), propagate by composition so most layouts are *forced*, with
   a tiny bounded DFS only where real instruction choice exists — cheaper than today's whole-kernel beam timing.

Hook points are clean: a layout pass slots into `full_rewrite_to_sink` (`codegen/__init__.py`) between range-simplify
and `apply_opts`, exactly where the `WARP_REDUCE_LOWERING` (M5) milestone already sits.

## Milestone path (honest: full IR is multi-month)
- **M0 (~1-2wk, the seed):** a static **coalescing predicate** (✓ built this session, `extra/qk_layout_coalesce_check.py`)
  + a declared offline Q4_K reshuffle + manifest entry + W==D gate. *Caveat from Phase B: the reshuffle alone is
  expected to be insufficient — its value is proving the predicate + the declared-layout mechanism, not closing the
  GEMV.*
- **M1 (~3-4wk):** `LayoutFn` + the **composition** operator as a graph_rewrite.
- **M2 (~4-6wk):** generalize `TensorCore.swizzle` → first-class `LaneMap`; re-express WMMA through it (regression-proves).
- **M3 (~6-8wk):** `OptOps.COALESCE` + static cost wired into beam; anchored propagation.
- **M4 (later, the deep codegen):** a coalesced packed-uint32-word load the lowerer can *exploit* (one word/lane,
  in-register multi-nibble dequant) + the `v_dot2`/cross-lane renderer lowerings + the schedule (waitcnt/clause) gap.
  **This is where owned-kernel *speed* actually lives** — the IR makes it expressible/searchable; M4 makes it fast.

## Risks (the live ones)
- **Representation ≠ speed** (confirmed by Phase B): coalesced loads ≠ owned bandwidth; the win is the kernel
  structure. The predicate is necessary, not sufficient.
- Codegen wall behind the IR (M4): even a perfect layout choice needs the lowerer to emit the coalesced packed-word
  load + in-register dequant — which it doesn't today (packed scheduler arm = 22 tok/s).
- Composition over masked/padded (`PAD`→WHERE/Invalid) or mixed-radix RESHAPE index exprs can defeat a naive
  stride-read → need CuTe-style divisibility/admissibility invariants.
- Power-of-two / non-pow2 shapes (LinearLayout's F2 limit); the real K=4096 mixed serial+group reduce.
- Scope creep: M0 is self-contained and *kill-able* — don't build the general IR before the lever shows a W==D gain.

## Bottom line
"Do both" lands as: **(b) the data-reshuffle sidestep does not close the GEMV for a pure-scheduler kernel
(coalescing is necessary-not-sufficient — proven), so (a) the layout/mapping IR is genuinely required** — and its
*speed* ultimately depends on the M4 codegen layer, not just the representation. The representation is the right,
field-aligned (CuTe/LinearLayout/Hexcute) frame; the honest expectation is it makes the owned structure *searchable*,
with a real codegen lift still behind it.
