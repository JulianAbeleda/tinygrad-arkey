# Coalesced vector-load lowering primitive — built + proven, boundary named (2026-06-26)

Scope: `docs/decode-coalesced-load-primitive-scope-20260626.md`

## Verdict

`COALESCED_LOAD_PRIMITIVE_BUILT__GLOBAL_STAGING_NEEDS_LANEMAP`

The durable, general coalesced-load lowering **primitive** is built, wired (default-off, cache-keyed), and
proven on representative idioms independent of any kernel. Applied to the generated block-tile it correctly
vectorizes the contiguous-loop-axis load (V from LDS) with numerics unchanged — but does **not** move W==D,
because the bandwidth bottleneck is the GLOBAL cache-staging load, whose contiguous dimension is mapped across
*threads*, not a loop axis. That gap is the next layer (LaneMap), precisely named — not a failure of this
primitive. This is the foundation/primitive deliverable, honestly bounded; **no speed is claimed** (timing flat).

## Built

- `extra/qk_coalesced_load_lowering.py` — `coalesce_loads(sink)`, env-gated `COALESCED_LOAD_LOWERING` (AMD),
  default-off, added to the `to_program` cache key. The codegen realization of the layout-IR `OptOps.COALESCE`:
  a **predicate-driven** pass that finds a small loop/reduce RANGE which is **unit-stride in a GLOBAL/LOCAL load
  index** (steered by the shipped coalescing predicate `axis_stride`, `extra/qk_layout_coalesce_check.py`) and
  promotes its `AxisType` to `UPCAST`, so the EXISTING expander + devectorizer fold the load into a vector load
  (`*((float4*)..)`/`half4`). Pairs with `REG_STORE_DEVEC` — the codegen hook now fires reg-store-devec whenever
  `COALESCED_LOAD_LOWERING` is on — keeping accumulator stores scalar (avoids the
  `make_float4(...)=make_float4(...)` invalid-C). Hooked in `tinygrad/codegen/__init__.py` after postopt-symbolic,
  before the expander (mirrors the `WARP_REDUCE_LOWERING`/`V_DOT2_LOWERING` recipe).

## Proven (proving ground, tile-independent)

`test/external/test_coalesced_load_lowering.py` — 5/5 green on gfx1100:
- promotes a unit-stride load axis to UPCAST; leaves the GLOBAL grid axis untouched;
- **declines** strided, REG-buffer, and oversized (> fold width) axes — no false-positive widening;
- end-to-end AMD: the promoted kernel is numerically exact and renders `float4 val0 = *((float4*)..)` + a
  **scalar** accumulator.

Mechanism proof (DEBUG=4): a plain scalar-loop kernel + `COALESCED_LOAD_LOWERING=1` →
`[COALESCE] promote axis=1 REDUCE size=4 -> UPCAST` → `float4` load + `(*(buf0+0)) = (*(buf0+0))+val0.x` (scalar
accumulate), numeric OK.

## Applied to the generated block-tile

`extra/qk_decode_attention_block_tile_microgate.py` with `COALESCED_LOAD_LOWERING=1`:
- promotes **axis 7** (the PV `dd` loop, R=4 contiguous in V) → `half4 val6 = *((half4*)(buf6+..))` (vectorized
  V LDS load);
- **microgate PASS** all 4 cases, `max_abs` unchanged (1.526e-05); **default-off byte-identical**;
- isolated tile timing **flat**: ctx512 1.025→1.037 ms, ctx4096 7.307→7.375 ms (within noise / marginally
  worse). The V LDS load is not the bottleneck.

## The boundary (the honest, useful finding)

The primitive vectorizes contiguous **loop/reduce** load axes. The block-tile's bandwidth-dominant load is the
GLOBAL K/V **cache staging** (`cache[0,0,kvh,t_safe_stage,e_stage]`, `qk_flash_decode.py:959-960`): each thread
loads ONE element per stage iteration, with the contiguous (Hd) dimension spread across `tid = warp*32+lane`
(a hardware thread special), **not** a promotable loop axis. So no axis-type promotion can widen it —
vectorizing it requires each thread to own a **contiguous chunk** of the row, i.e. a thread→element **mapping**
(LaneMap / CuTe TV-layout), which is layout-IR M2, not this pass. `global_load_d16` therefore stays 0 here; the
owned tile's 22 comes from its hand-authored per-thread-contiguous staging map.

So the levers compose as predicted: **coalesced loads (this primitive, loop-axis case) → LaneMap staging map
(next) → block-tile/multi-warp (exists) → scheduling (`SCHED_UNROLL`/`SCHED_LIST`, shipped)**. This primitive
is the loop-axis half of the bandwidth gate and a clean, validated foundation; the staging half is the LaneMap.

## Guardrails honored

Default-off; cache-keyed; byte-identical when unset; correctness-first (microgate unchanged); no new attention
layout; no per-kernel hack (predicate-driven + proving-ground-validated); **no speed claimed** (timing flat,
reported honestly); no bench timing artifacts committed.

## Next

Layout-IR **M2 LaneMap** (`docs/layout-codegen-full-scope-20260625.md` P1.2): a first-class thread→element map
so the cache staging can declare a per-thread contiguous chunk that this primitive then vectorizes — the half
needed to move `global_load_d16 > 0` and W==D on the generated attention tile.
