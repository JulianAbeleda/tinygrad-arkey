# Fused x-lane score+PV tile — W==D result + refutation (2026-06-26)

The fused, fast, clean, occupancy-matched tile was built and run through W==D. **It is refuted on speed**,
and the refutation localizes the real wall: generated-codegen *code quality*, not lane layout and not
occupancy.

## W==D (DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE=1, GPU-bound, host-sync <3%)

| ctx | xlane tok/s | baseline | ratio | actual occupancy (this run) |
|---:|---:|---:|---:|---|
| 128 | 82.5 | 82.3 | 1.00× | — |
| 512 | 6.5 | 103.2 | 0.06× | s=6 → 48 wg → 0.5 wg/CU (starved) |
| 1024 | 3.6 | 101.3 | 0.04× | s=11 → 88 wg → 0.9 wg/CU (starved) |
| 4096 | 0.9 | 94.0 | 0.01× | s=43 → 344 wg → **3.58 wg/CU (near-matched)** |

## Two findings, both pointing away from occupancy/layout

1. **Occupancy plumbing bug (real, but not the wall).** The route derives `l_route = ceildiv(MAXC, 48)`
   with `MAXC = 4608` fixed for all ctx (runtime_overhead.py:21), so the *actual* split count is
   `ceildiv(ctx, 96)` — only 6/11 splits at ctx 512/1024 (starved). The route gate's "4.0 wg/CU" was the
   nominal `Hkv·48`, not the runtime value. (A true fix is a concrete `S=48` grid with a symbolic per-split
   `L=ceildiv(Tc_u,48)`; the existing generated routes have the same starvation because they make the grid
   symbolic in `Tc_u`.)

2. **Occupancy is necessary but NOT the binding constraint — refuted.** At ctx 4096 the occupancy was
   already near-matched (3.58 wg/CU vs owned 4.0), yet the route was still 99× slower overall and
   **~1665× over the HBM roofline per layer** (29 ms/layer attention vs a 17.5 µs/layer floor). That is the
   signature of a kernel that is **compute-bound on inefficient generated ISA**, not memory-bound. Fixing
   low-ctx occupancy cannot rescue a kernel that is 1665× over the floor where occupancy is already matched.

3. **Not the lane layout, either.** Every generated whole-cache tile — score-broadcast (0.7), this xlane
   tile (0.9), pall-lifecycle — lands ~16–104× slow at the same ctx. Different tile structures, same
   catastrophic regime → the bottleneck is common to the generated path, not specific to a layout. The
   owned route (hand-written AMDGCN, same S=48, same cache) hits 94 tok/s. **The 99× gap is code quality:
   tinygrad's UOp→ISA codegen for this kernel class vs hand-written assembly.**

## Verdict and label

- Candidate `decode_attention_fused_xlane_score_pv_tile`: **`REJECTED_BY_ECONOMICS`** — correct, route-clean,
  bounded, buffer-identity, fast-primitive-bearing, occupancy-aware, and still 16–104× slower because the
  generated ISA is compute-bound.
- Lane / search label: **`SEARCH_BLOCKED_BY_CODEGEN` at the code-quality (ISA) level** — not by lane layout
  (proven expressible + correct), not by occupancy (matched at ctx 4096, still catastrophic), not by
  materialization (buffer-identity clean). The likely ISA culprits: per-token cross-lane reductions
  (`_warp_reduce_sum_staged` per token per head) and per-token LDS barriers, scalar/uncoalesced V loads, and
  register spill from the placeholder/`.after` recurrence — none of which the hand-written AMDGCN pays.

## Assets kept (the arc was not wasted)

- The **score-once → d-sharded-PV lane layout** is proven expressible + numerically correct
  (microgate PASS, scalar 1e-7 / fdot2 2e-5; in-model tokens match owned). This refutes the prior code's
  "score reuse across the PV column axis is not expressible yet."
- The **occupancy/roofline baseline** (`docs/decode-fused-tile-occupancy-roofline-baseline.md`) is correct
  and reusable — occupancy is *necessary*; it is now also shown to be *insufficient* alone.
- The in-model route + route gate + the cross-lane reducer (P11/P12) all stand.

## Do not reopen / next lever

- Do **not** try another attention lane layout, another whole-cache tile, or occupancy tuning as the fix —
  all three are now refuted as the binding constraint.
- The next lever is **generated-codegen code quality**: make the renderer emit efficient ISA for this tile
  shape — vectorized/coalesced K/V global loads, amortized cross-lane reduction (block of tokens, not
  per-token), fewer barriers, register-tiled accumulation. That is the renderer/lowering frontier
  (the north-star `v_dot2` + cross-lane lowering), and the right next gate is an **ISA-level** microgate
  comparing the generated tile's disassembly against the owned AMDGCN, not another W==D on a new route.
