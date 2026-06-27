# Coalesced vector-load lowering — a durable codegen PRIMITIVE (scope, 2026-06-26)

## Principle (read first): primitive over quick win

The decode-attention tile is the **last** default hand kernel blocking pure machine search
(`docs/pure-machine-search-roadmap.md`; GEMV is already pure/generated via BubbleBeam G3). The generated tile
is correct + route-clean + token-matched but **W==D-refuted ~99×** (6.5/3.6/0.9 vs 103/102/95 tok/s), and the
gap is **pinned to codegen strategy, not layout, not search-navigation** (ISA diff: owned `global_load_d16`=22,
generated=**0** — scalar loads, ~28 GB/s of a 960 GB/s card).

There are two ways to close it:

- **Quick win (rejected):** hand-edit `flash_block_tiled_…_kernel` to declare its contiguous load axes `UPCAST`
  / build `STACK`s of contiguous ptr-INDEXes until the existing coalescer happens to fire. This makes ONE
  kernel fast but adds hand-owned authoring — it is *more* of the thing pure-search is trying to retire, and it
  generalizes to nothing.
- **Primitive (this scope):** build the **decision** that is actually missing — "which contiguous load axis to
  widen, and to what width" — as a first-class, predicate-driven, env-gated codegen pass that any generated
  bandwidth-bound kernel triggers. This *is* the layout-IR's `OptOps.COALESCE` realized in codegen (M3/M4 of
  `docs/layout-mapping-ir-design-20260625.md`), built on the already-shipped coalescing predicate
  (`extra/qk_layout_coalesce_check.py`) and `LayoutFn` (`extra/qk_layout_fn.py`). It is foundation-first: the
  **bandwidth gate that sits UNDER** block-tile structure and instruction scheduling — and it is exactly the
  capability the owner directive ([[value-long-term-solution-not-cheap]], foundation-first) asks for.

Acceptance is therefore deliberately *not* "the tile got faster." It is "coalescing is now a general,
predicate-driven, proving-ground-validated codegen primitive that fires correctly on representative idioms
independent of the attention tile — and, applied to the tile, produces vectorized loads."

## Grounding (proven Phase 0, 2026-06-26)

- `flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel` renders scalar loads:
  `float val0 = *(data2_524288+alu8)` (K cache), `val1 = *(…+262144)` (V cache), `val4/val5 = *(data1+…)` (q
  pair) — no `float4`/`*((float4*)…)`. ~5–28 GB/s. (Baseline microgate still PASS, max_abs 1.526e-05.)
- The coalescer machinery (`tinygrad/codegen/late/devectorizer.py`):
  - `split_load_store` (`:153-200`) vectorizes a LOAD/STORE **only when its INDEX already has vector count > 1**
    (`sz==1 → None`, `:157-158`); fold lengths `[4,2]` float/half, `[8,4,2]` if `ALLOW_HALF8`, `[4]` for uint32
    GLOBAL. **It already skips `AddrSpace.REG` (`:171-172`)** → REG accumulator stores stay scalar for free.
  - `fold_expanded_index` (`:81-117`) folds a `STACK` of contiguous **ptr-INDEXes** into a `PTRCAT`→`VCAT`.
- The generated tile presents **neither**: the staging load is a scalar `INDEX→LOAD` (sz==1), and the dot's
  `STACK` (`qk_flash_decode.py:970-972`) is over `CAST(LOAD(INDEX))`, not over ptr-INDEXes. So nothing widens.
- ⇒ The missing piece is the **decision to present a contiguous load axis as a vector** (then the existing
  machinery + comgr/LLVM emit the wide load — no renderer change, per `decode-generated-tile-codegen-scope.md`
  §B). The V inner load `vsh[tt*Hd + lane*R + dd]` is **unit-stride in `dd`** (R=4 consecutive) — the canonical
  coalescable axis; today `dd` is a plain `LOOP` range so it stays scalar.

## The primitive

`extra/qk_coalesced_load_lowering.py` — env-gated `COALESCED_LOAD_LOWERING`, AMD-only, default-off, added to the
`to_program` cache key (mirror the `V_DOT2_LOWERING`/`WARP_REDUCE_LOWERING` recipe, `codegen/__init__.py`).

A `graph_rewrite` pass that:

1. **Detects** a load whose address is **unit-stride affine** in a contiguous loop/upcast axis, using the layout
   predicate as the steering function — `LayoutFn(idx).coeff(axis)==1` / `is_coalesced` — i.e. coalescing is a
   *queried* property, not an emergent accident. Width = longest unit-stride run capped by the dtype fold table
   (`vector_width`, the `[4,2]`/`[8,4,2]` policy).
2. **Promotes** that axis to a vectorized load by presenting the INDEX in the vector form `split_load_store`
   consumes (the contiguous-axis offsets as a single vec INDEX / a `STACK` of contiguous ptr-INDEXes), letting
   the EXISTING devectorizer fold it. Reuse, not reimplementation.
3. **Never touches `AddrSpace.REG`** (accumulator stores stay scalar — both because `split_load_store` skips REG
   and because the pass filters to GLOBAL/LOCAL contiguous loads). This is the explicit guard against the
   `make_float4(...) = make_float4(...)` invalid-C failure mode (the REG-accumulator-vectorization bug).

Decision fork (record which, with evidence, in the result doc — both are *primitives*, not hacks):
- **(A) authoring primitive** — if a small reusable `coalesced_load(buf, base, axis, width)` builder that emits
  the vector form is sufficient for the existing coalescer to fire, that is the lower-risk primitive (the
  kernel *calls* it; it is still general and proving-ground-tested). Prefer if it suffices.
- **(B) lowering pass** — if no authoring form triggers the coalescer on the custom (`opts_to_apply=()`) path,
  the genuine codegen gap is real → build the predicate-driven promotion pass (above). This is the milestone.

## Proving ground (the part that makes it a primitive)

`test/external/test_coalesced_load_lowering.py` — **independent of the attention tile**, constructs
representative idioms and asserts the primitive behaves:

- **P-contiguous:** N unit-stride scalar loads from one GLOBAL buf → exactly one vectorized load
  (`*((float4*)…)` / VCAT), numerically equal to the scalar version (exhaustive over the index space).
- **P-stack-cast:** a `STACK` over `CAST(LOAD(contiguous INDEX))` → folded vec load (the tile's dot idiom).
- **P-decline:** non-unit-stride / data-dependent / masked access → **declines** (no false-positive widening;
  reuse `LayoutFn`'s admissibility refusal).
- **P-reg-scalar:** a REG-addrspace accumulator store remains scalar after the pass (no `make_floatN = …`).
- **P-default-off:** flag unset ⇒ byte-identical UOp graph (a quick AMD sanity: matmul+reduce+elementwise
  unchanged).

This is the foundation-first deliverable: the predicate + promotion proven on idioms, not on one kernel.

## Acceptance ladder (in order; do not skip)

1. Proving-ground test green on gfx1100 (structural + numeric + decline + reg-scalar + default-off).
2. Block-tile microgate still PASS (`extra/qk_decode_attention_block_tile_microgate.py`), max_abs unchanged.
3. ISA shows vectorized loads: `global_load_d16 > 0` or `global_load_dwordx4 > 0` on the generated tile
   (`extra/qk_decode_attention_isa_diff_gate.py` markers, or objdump of the captured `.co`).
4. **Only then** W==D vs owned baseline (`extra/qk_decode_runtime_overhead.py`) — report honestly; the primitive
   is the deliverable even if W==D needs the block-tile/scheduling levers stacked on top.

## Guardrails

- **Default-off; cache-keyed.** Flag unset ⇒ byte-identical default path (owned AMDGCN tile + generated q4k
  GEMVs unchanged). Add the flag to the `to_program` cache key.
- **Correctness is authority.** Never regress microgate/route token-match. Cross-lane/LDS/fdot2 staging rules
  (`extra/amd_warp_reduce.py`) and the `CUSTOMI`-carries-`src[0]`-shape rule still hold.
- **No new attention layout** (layout is proven) — strictly the coalescing decision + vectorization.
- **No per-kernel hack masquerading as the deliverable.** If a change only helps this one tile and is not
  predicate-driven/proving-ground-validated, it is out of scope.
- **No claim without evidence.** "Coalesced" requires ISA `d16/dwordx4 > 0`; "faster" requires W==D movement;
  isolated timing is not authority (hard-won repo lesson).
- Don't commit non-deterministic bench timing artifacts.

## Why this is the right foundation, restated

Coalescing is *necessary-but-not-sufficient* (proven for the GEMV). This primitive supplies the necessary
bandwidth layer as a **general, searchable, predicate-driven capability** — the codegen realization of the
layout IR's `OptOps.COALESCE` — under the block-tile structure (exists) and the shipped `SCHED_UNROLL`/
`SCHED_LIST` scheduler. It is the durable primitive, not the one-kernel win, and it is the gate on retiring the
last default hand kernel.
