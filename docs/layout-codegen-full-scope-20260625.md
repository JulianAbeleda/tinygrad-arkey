# Layout/codegen wall — Codex execution plan

This doc is written to be **executed by Codex**. It closes the "layout wall": the decode Q4_K GEMV runs at ~22 tok/s
through the scheduler vs ~100 tok/s for the hand kernel, because the owned kernel's *physical structure*
(coalesced packed-word load + in-register dequant + per-row-workgroup thread-map) is **not expressible** by
tinygrad's scheduler. The fix is a searchable layout/mapping representation + the renderer-codegen lift behind it.

## How to use this doc
- Execute the **TASK**s in order. Each TASK is self-contained: **Goal / Read-first / Build / Gate / Kill / Depends**.
- A TASK is "done" only when its **Gate** passes. If a **Kill** condition triggers, STOP that branch and record why.
- `P2.3 (M-E)` is the **project decision gate** — the whole plan pivots on its result. Do not start P3 unless M-E wins.
- Honor the strategic context below; it tells you *why* and *when to stop*.

## Global setup & conventions (apply to every TASK)
- Repo: `/home/ubuntu/tinygrad-arkey`. GPU: RX 7900 XTX / gfx1100, wave32. Python: `.venv/bin/python`.
- Run pattern: `DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python ...`. Inspect kernels with `DEBUG=4`. Pin clocks for
  timing with `rocm-smi --setperflevel high`.
- **Everything is gated and DEFAULT-OFF. Never change a default.** With the new flag unset, behavior must be
  byte-identical to today (run a quick AMD sanity: matmul + reduce + elementwise correct).
- **Correctness is authority.** Decode token-correctness = byte-identical greedy. Non-byte-exact paths use
  `rel_rmse ≤ 1e-3` vs the reference.
- **W==D is the only speed authority.** Use the in-process, clock-pinned, interleaved harness
  `extra/qk_q4k_packed_gemv_wd.py` (or `extra/qk_decode_runtime_overhead.py`) — never isolated/standalone timings
  (they don't transfer in-model; this is a hard-won lesson in this repo).
- Each TASK ships its own test under `test/external/` and a result note. Commit per TASK with a `[nn]` (code) or
  `[docs]` prefix and a message stating what shipped + the gate result. **Do NOT add any `Co-Authored-By` / Claude
  attribution line.**
- Already-built pieces you will reuse (read them, don't rebuild): `extra/qk_warp_reduce_lowering.py` (M5 cross-lane,
  the additive gated-`PatternMatcher` pattern), `extra/amd_warp_reduce.py` (the `Ops.CUSTOMI` AMD-builtin emit idiom),
  `extra/qk_layout_coalesce_check.py` (the static coalescing predicate), `extra/qk_amdgpu_isa_primitive_audit.py` (ISA
  flags: `has_v_dot2`, `has_cross_lane`, `has_lds`), `tinygrad/codegen/__init__.py` (the `WARP_REDUCE_LOWERING` hook +
  the `to_program` cache key), `bench/qk-search-spaces/decode_ffn_gemv_gfx1100_v1.json` (the search-space manifest).

## Strategic context (read once)
- **One wall, two sides.** SIDE 1 = decode Q4_K GEMV (live gap, ~22 vs ~100 — this is the target). SIDE 2 = prefill
  GEMM (already ~96% of Tensile / ≥ llama; settled — do NOT work on it).
- **Coalescing is necessary but NOT sufficient; representation ≠ speed.** Forcing `GROUP(32)` gave no speedup
  (49.9≈50.9); three scheduler GEMV restructures got 50/22/14 vs owned 103. The owned win is the whole *kernel
  structure*, not a coalesced load. So the layout IR makes the structure *searchable*; the codegen lift makes it
  *fast*; and even both together may land at ~80–95% of owned (Tier 2) with CUSTOM the last-mile ceiling.
- **HEAD facts (verified):** AMD does NOT use the `ISARenderer`/regalloc path (x86-only) — it hands C-style to
  LLVM/HIP which owns instruction scheduling; `fdot2` and `OptOps.COALESCE` are absent; the
  `extra/qk_asm_scheduler_inc*_test.py` files are NOT on HEAD; load-dedup is already done (UOps are interned, so one
  packed word is loaded once and 8 nibbles extracted in-register — the 22→100 gap is **100% thread-map**).
- **Dependency graph:** `M1 (LayoutFn+compose) → M2 (LaneMap) → M-B (LaneMap-aware add_gpudims, KEYSTONE) → M-C
  (reuse float4 peephole + cross-lane) → M-E (W==D gate) → [if win] M3 (searchable)`. `v_dot2` is an **independent**
  lever (feeds the attention tile, a different kernel). Prefill asm-scheduler + AMD ISARenderer are **deprioritized**.

---

## TASK P0.1 — `v_dot2` (AMD `fdot2`) renderer lowering  [independent, cheapest real lever]
**Goal.** Make a generic fp16 fused multiply-accumulate auto-lower to `__builtin_amdgcn_fdot2` (the fused fp16 dot),
gated `V_DOT2_LOWERING`, default-off. This closes the second instruction gap (cross-lane was the first) and feeds the
owned attention tile's q·k dot. **Instruction-expose only — it does not by itself make the tile fast.**
**Read-first.** `extra/qk_warp_reduce_lowering.py` + `test/external/test_warp_reduce_lowering.py` (mirror exactly);
`extra/amd_warp_reduce.py` (`Ops.CUSTOMI` emit idiom, shape-carrying, shaped value first); `tinygrad/renderer/
cstyle.py` (CUSTOM/CUSTOMI render via `arg.format(*srcs)`); `docs/archive/native-codegen-microprimitive-search-
result-20260623.md` (confirms `v_dot2` is a renderer GAP — fp16 MAC lowers to `v_pk_*`+reduce today).
**Build.**
1. Probe first: render a small fp16 dot/matmul on AMD with `DEBUG=4`, identify the exact post-devectorize idiom (two
   scalar `MUL`s over `gep(half2,0/1).cast(f32)` summed into an f32 acc). Pin the UOp form before writing the matcher.
2. `extra/qk_fdot2_lowering.py`: `pm_fdot2` `PatternMatcher` matching that idiom (f32 sum of two f32-cast half×half
   products + optional f32 acc, the two halves being lanes of `half.vec(2)` values) → one `Ops.CUSTOMI` with
   `"__builtin_amdgcn_fdot2({0}, {1}, {2}, false)"` (a, b half2; acc f32). Conservative: float-only, decline anything
   not the exact pair idiom (no false positives).
3. Gate `getenv("V_DOT2_LOWERING")` + `ren.target.device=="AMD"`; inject as a `graph_rewrite` at the correct stage
   (post-devectorize — verify by DEBUG; the WARP_REDUCE hook is in the expander, this is later). Add
   `getenv("V_DOT2_LOWERING")` to the `to_program` cache key next to `WARP_REDUCE_LOWERING`.
4. `test/external/test_fdot2_lowering.py`: (a) structural — idiom → exactly one `fdot2` CUSTOMI, declines negatives;
   (b) AMD end-to-end — fp16 dot/matmul flag-on is `rel_rmse ≤ 1e-2` vs flag-off and `qk_amdgpu_isa_primitive_audit.py`
   `has_v_dot2` is true (objdump shows `v_dot2`).
**Gate.** Both tests green on gfx1100; ISA shows `v_dot2`; default path (flag off) unchanged.
**Kill.** If the rule never fires on a real post-devectorize fp16 sum graph (idiom in an unmatched normalized form)
and the canonical form can't be pinned in ~3 days → leave `v_dot2` unexposed (the `v_pk_*`+reduce path is still
correct) and note it; the GEMV critical path is unaffected.
**Depends.** None.

## TASK P0.2 — Coalesced-dequant attribution memo (M-A)  [kills a non-problem before P2]
**Goal.** Prove the 22 tok/s scheduler-GEMV loss is **100% thread-map**, not redundant loads — so P2 doesn't waste
budget on "load-dedup."
**Read-first.** `extra/qk_q4k_scheduler_gemv.py` (`Q4K_GEMV_SCHEDULER=2` path), `extra/q4_k_gemv_primitive.py`
(`_q4k_group_dot_packed_load`, the owned structure), `tinygrad/codegen/late/devectorizer.py` (`pm_add_loads` —
idempotent load dedup).
**Build.** Run the packed arm (`Q4K_GEMV_SCHEDULER=2`, `DEBUG=4`); count `LOAD` UOps per packed word in the rendered
gate/up kernel. Write `docs/coalesced-dequant-attribution-20260625.md`: confirm 1 load/word (CSE'd) and conclude the
loss is the strided output-parallel thread-map (not loads).
**Gate.** Rendered kernel shows no duplicate `words[base+...]` loads; memo written.
**Kill.** If instrumentation shows duplicate loads (contradicts CSE), STOP and re-scope — the whole coalesced-dequant
attribution is wrong.
**Depends.** None.

## TASK P0.3 — Schedule-asm park decision (M0)  [retire the lowest-EV tracks]
**Goal.** Formally park the prefill asm-scheduler and the AMD ISARenderer decode-backend before they consume budget.
**Read-first.** `structure/Development/session-handoff.md` (prefill asm-scheduler / DNR entries), the
`extra/qk_asm_scheduler.py` arc, `tinygrad/codegen/__init__.py` (`do_linearize` — note the `ISARenderer` path is
x86-only).
**Build.** Restore `extra/qk_asm_scheduler_inc{0,1,2,3}_test.py` to HEAD (they are NOT currently present), confirm
Inc0-3 PASS on gfx1100, then write `docs/schedule-asm-park-decision-20260625.md`: prefill asm-scheduler is
measured-complete (no transferable whole-prefill win); AMD ISARenderer decode-backend is out-of-scope unless P2's M-E
forces the CUSTOM conclusion and portability is explicitly valued over speed.
**Gate.** Tests PASS + decision memo written.
**Kill.** n/a (decision task).
**Depends.** None.

## TASK P1.1 — Layout IR: `LayoutFn` + CuTe composition (M1)
**Goal.** A queryable layout object over `Ops.INDEX(buf, idx)` + the one missing CuTe operator (composition), so
"is this access coalesced" and "compose a thread-map onto a data-layout" are library calls.
**Read-first.** `extra/qk_layout_coalesce_check.py` (the M0 stride predicate — extend it); `tinygrad/codegen/opt/
heuristic.py:142-145` (coefficient-of-RANGE stride recovery via `split_uop(Ops.ADD)`); `tinygrad/schedule/
indexing.py` (`apply_movement_op` — the layout algebra: SHRINK/PERMUTE/FLIP/EXPAND/PAD/RESHAPE on RANGEs);
`tinygrad/uop/ops.py` (`get_idx`/`get_valid`, `UOp.substitute`, `symbolic`).
**Build.** `extra/qk_layout_fn.py`: a `LayoutFn` wrapping an `Ops.INDEX` with `coeff(range)` / `is_unit_stride(range)`
/ `compose(other)`. Implement **composition** `R(c)=A(B(c))` as a `graph_rewrite` that substitutes one RANGE's
index-expr into another and re-simplifies through `symbolic`. Add CuTe-style divisibility/admissibility invariants so
composition over masked (`PAD`→WHERE/Invalid) or mixed-radix (`RESHAPE`) index exprs **raises** rather than silently
returns a wrong stride.
**Gate.** Property tests (`test/external/test_layout_fn.py`): `coeff` matches hand-derived strides on
RESHAPE/PERMUTE/SHRINK/EXPAND outputs of `apply_movement_op`; `compose(A,B)` is byte-equal (after `symbolic`) to
manually substituting B into A on a matmul + a GEMV AST; masked/mixed-radix cases raise (not mis-stride). No model
wiring.
**Kill / pause.** If composition cannot reject masked-PAD / mixed-radix without false strides → PAUSE (silent wrong
strides are worse than no IR; fix the invariants first).
**Depends.** M0 (shipped).

## TASK P1.2 — Layout IR: first-class `LaneMap` from `TensorCore.swizzle` (M2)
**Goal.** Generalize the WMMA-only thread→fragment swizzle into a first-class, validated `LaneMap` any kernel can
carry (the object M-B needs).
**Read-first.** `tinygrad/codegen/opt/tc.py` (`TensorCore.swizzle`, `permutes_for_shape_str`, `__post_init__`
asserts); `tinygrad/codegen/opt/postrange.py` (`_apply_tc_opt`, where the swizzle is consumed).
**Build.** `LaneMap` object `f:(thread,value)→coord` (the CuTe TV-layout), with its own validation subsuming the
`tc.py` asserts. Re-express the WMMA path to consume a `LaneMap` instead of the raw swizzle tuple. Any non-WMMA kernel
can now carry one.
**Gate.** REGRESSION: every existing WMMA shape (cuda/amd_rdna/amd_cdna/metal in `tc.py`) produces **byte-identical**
kernels through the LaneMap re-expression; existing TC tests pass unchanged; `LaneMap.validate` rejects malformed maps
the swizzle asserts caught.
**Kill.** Any existing WMMA shape not byte-identical → STOP (non-negotiable; it breaks the one working
thread→fragment path).
**Depends.** P1.1 (compose, to validate a fragment load against a data LayoutFn).

## TASK P2.1 — KEYSTONE: LaneMap-aware `add_gpudims` (M-B)
**Goal.** Make the owned kernel's composite per-row-workgroup thread-map **expressible from the scheduler** —
`lane = block_group*8 + word_col`, splitting a `REDUCE` range across one wave. This is the single change that targets
the live in-model gap.
**Read-first.** `tinygrad/codegen/gpudims.py` (`add_gpudims`/`get_grouped_dims`; note `:103` SKIPS REDUCE axes today —
this is the core limitation); `extra/q4_k_gemv_primitive.py:466-495` (`q4k_gemv_warp_kernel` = the target structure:
`lane=bg*8+lane4`, `bg` splits k_blocks 4-way, `lane4` = within-block word col); the M2 `LaneMap`.
**Build.** Teach `add_gpudims`/`get_grouped_dims` to map a RANGE to `lane%8` (word col) and split a REDUCE range
`bg=lane//8` across one wave, driven by a `LaneMap`. Re-express the owned q4k `lane=bg*8+lane4` decode GEMV as a
pure-**scheduler** kernel (keep it q4k-specific for now), reading the packed words via the existing
`extra/qk_q4k_scheduler_gemv.py` dequant.
**Gate.** `DEBUG=4` shows `special(lidx0)` with `lane4=lidx0%8` making 8 adjacent lanes' word offsets **consecutive
(stride-1)** — verify with `extra/qk_layout_coalesce_check.py:is_coalesced` on the weight INDEX.
**Kill / re-scope.** If splitting a REDUCE range across hardware lanes within one wave needs a structural rewrite of
`get_grouped_dims` larger than the rest of the roadmap → re-scope the question to "is the owned thread-map
fundamentally outside the RANGE/AxisType model?" before continuing.
**Depends.** P1.2 (LaneMap) + P1.1 (compose).

## TASK P2.2 — Wire the rest of the owned structure (M-C)  [reuse, no new renderer work]
**Goal.** Make the M-B scheduler kernel structurally isomorphic to the owned warp GEMV.
**Read-first.** `tinygrad/codegen/late/devectorizer.py` (`fold_expanded_index`/`load_store_folding` — the float4
peephole); `extra/qk_warp_reduce_lowering.py` (cross-lane reduce, already shipped).
**Build.** Confirm the float4 `uint32` vec-load fires for free once the load is stride-1; wire `WARP_REDUCE_LOWERING`
for the cross-lane reduce + single store. No new renderer code.
**Gate.** Rendered kernel = coalesced uint32 vec-load + warp reduce + single store; tokens match owned.
**Kill.** n/a (integration).
**Depends.** P2.1.

## TASK P2.3 — THE DECISION GATE: in-model W==D (M-E)
**Goal.** Measure whether the structurally-correct scheduler GEMV actually closes the gap.
**Build.** `extra/qk_q4k_packed_gemv_wd.py` (extend it): clock-pinned interleaved W==D, FFN gate/up, ctx
512/1024/2048/4096, arms = owned vs the M-C scheduler kernel. Write `docs/coalesced-dequant-mE-result-<date>.md`.
**Gate (DECISION).** scheduler arm **≥ ~90% of owned** (≥ ~90 tok/s) with `tokens_match` → pure-search effectively
wins; proceed to P3.
**Kill (PROJECT KILL GATE).** If it plateaus near the 49.9–50 `MV_DEQUANT` ceiling instead of ≥90 → **declare CUSTOM
fundamentally needed for the GEMV, STOP, do NOT fund P3**, and re-frame the layout-IR deliverable as
searchability/portability only. Record this in the manifest.
**Depends.** P2.2.

## TASK P3.x — Generalize to searchable  [ONLY if P2.3/M-E wins]
**Goal.** Convert the hand-pinned keystone into a beam-discoverable capability.
**Build.** (1) `OptOps.COALESCE` in the enum (`opt/__init__.py`) + an `apply_opt` case (`postrange.py`, `shift_to` a
data axis onto the thread/lane RANGE) + register in the beam `actions` (`search.py`); static cost =
`is_unit_stride(thread_range)` on the composed layout + `vector_width`, used to **prune in beam before timing**;
Hexcute anchored propagation (anchor on the dominant op's LaneMap, bounded DFS only at real instruction choice).
(2) `Ops.LAYOUT_TRANSFORM` declared storage-permutation op in `GroupOp.Movement` + an `apply_movement_op` case whose
arg names the consumer's LaneMap. (3) Make the q4k LaneMap **discovered** by beam (no kernel-specific flag).
**Gate.** Beam reproduces the hand-found coalesced schedule **without timing**; static cost ranks coalesced above
uncoalesced (predicate vs measured agree on a probe set); `LAYOUT_TRANSFORM` survives the simplifier; anchored
propagation forces >90% of layouts.
**Kill.** If the static COALESCE predicate disagrees with measured ordering on the probe set → STOP (it would prune
the fast candidate); if anchored propagation doesn't force >90% of layouts → it's not cheaper than whole-kernel beam,
drop it.
**Depends.** P2.3 (M-E win).

## P4 — DEPRIORITIZED (do NOT execute unless a gate forces it)
- Prefill asm-scheduler (M1 WGM8 / M2 register-pool / M5 transfer): measured-complete; every lever must beat the
  ~0.7% whole-prefill noise floor on clock-pinned synced A/B (prior art predicts all die there). Do not fund.
- AMD `ISARenderer` decode backend + ATT/SQTT attribution: a 4–8 wk compiler-backend project (AMD has no in-tree ISA
  path today), gated behind locally-infeasible attribution tooling; DNR3/4 found every lever sub-material. Fund ONLY
  if P2.3 proved CUSTOM is needed AND portability is explicitly valued over speed.

## Honest end-state & total effort
- **Tier 1 (achievable):** M-B/M-C make the owned *structure* emittable from the scheduler for the first time → moves
  the GEMV from 22 to well past the 49.9 ceiling.
- **Tier 2 (most likely, ~80–95% of owned):** searchable + real, but the last 5–20% is instruction scheduling
  LLVM/HIP owns (AMD has no in-tree ISARenderer) — the same wall the prefill LDS-GEMM arc hit.
- **Tier 3 (NOT reachable by pure-search):** byte-for-byte owned/Marlin parity → CUSTOM is the documented last-mile
  ceiling.
- **Effort:** critical path to the M-E decision (P0+P1+P2) ≈ **16–22 weeks**; if M-E wins, P3 adds ~8–12 wk. Largest
  risk: P2.1/M-B (splitting a REDUCE range across lanes in `add_gpudims`, which the code doesn't do today). Treat
  ~16–22 wk-to-M-E as the real commitment; everything past M-E is conditional on its result.
