# Decode-attention pure-search — state + learnings (2026-06-27)

A synthesis after building the decode-attention codegen foundation bottom-up, transferring it in-model, scoping
every structural delta, attacking the levers, and self-reviewing. Written as the authoritative "where we are +
what we learned" record. Authoritative live state: `docs/pure-machine-search-roadmap.md`.

## Where we are

The **decode attention tile is the last default hand kernel** blocking pure machine search (the Q4_K GEMV is
already pure/generated under BubbleBeam G3). The generated block-tile route is correct, route-clean,
token-matched, and now carries a stack of **composable, default-off, proving-ground-tested codegen primitives**:

| primitive | flag | what it does |
|---|---|---|
| recurrence-unroll + list scheduler | `SCHED_UNROLL` / `SCHED_LIST` | scalar-unroll the token loop, re-thread the carry → cross-iteration ILP |
| coalesced-load lowering | `COALESCED_LOAD_LOWERING` | predicate-driven UPCAST of unit-stride load axes (the `OptOps.COALESCE` codegen realization) + `REG_STORE_DEVEC` |
| cooperative-staging LaneMap | `DECODE_STAGE_COALESCE=<W>` | thread→element map: each thread owns a contiguous W-chunk → the global cache staging vectorizes |
| work-removal (exp) | `DECODE_FAST_EXP2` | bare `v_exp2` on the softmax carry (args always ≤0 → range-reduction is dead weight) |

They **compose**, numerically correct (microgate PASS), to **2.54× isolated** (1.024→0.403 / 7.289→2.875 ms
@ctx512/4096) and **~1.75× in-model** (block-tile route 19.0/3.5 → 32.8/6.2 tok/s; transfer dilutes the isolated
gain as expected). All built on the layout-IR seed (`LayoutFn`+composition M1, `CooperativeStageLaneMap` M2, the
`axis_stride` coalescing predicate).

**Honest ceiling:** still **3–15× below the owned hand-asm tile in-model** (owned 103/94 tok/s), gap grows with
ctx. This is a **search-capability** result (the *machine* now generates a far-better tile from generic
primitives), **not a promotion candidate** — owned still ships and is HBM-bound at parity with llama.cpp.

## The diagnostic truths (what the disasm + attacks actually proved)

1. **The gap is LATENCY/ILP-bound, NOT throughput.** The generated tile emits *fewer* instructions (414 vs 561),
   *wider* loads (b64/b128 vs owned d16), *matched* occupancy (vgpr ≤ owned, no spills) — yet is slower. The old
   "scalar loads / global_load_d16=0 / more cross_lane" framing was a premise error.
2. **The tile is OCCUPANCY-BOUND** (vgpr88 on the unroll stack, 4 wg/CU ceiling). ⇒ **levers must REMOVE work,
   not add ILP-via-state.** `SCHED_UNROLL_SPLIT` crashed VGPR 88→144; `Q_HOIST` lost comgr's pressure-aware LICM
   — both *refuted and reverted*. `FAST_EXP2` removed work → *won*.
3. **The ctx-slope is the OUTER `b`-block-loop carry, NOT the inner `tt` carry.** Splitting `tt` left the slope
   unchanged (the baseline unroll already hides the tt-carry in the copies' prologue shadow). Any slope-bending
   split must target `b` and stage its partials in the 8 KB LDS (not VGPR).
4. **`ds_bpermute` cross-lane reduce is at per-token parity with owned** (static count) ⇒ no cross-lane primitive
   warranted (it would have to *beat* owned, which does the identical thing). The dynamic REG-round-trip
   structural-floor risk is noted but the hotloop-diff was inconclusive.
5. **Coalescing IS necessary-but-not-sufficient — but it converted to speed here** where it didn't for the GEMV,
   because the attention staging is a *contiguous bandwidth load* (not a packed gather+dequant), so the LaneMap +
   the codegen lowering the GEMV lacked do turn into real ms.

## The methodology that worked (the durable process learnings)

- **Measure-first / audit-before-attack.** The hotloop-schedule-diff audit *refuted the scope's #1 lever*
  (`SCHED_UNROLL_SPLIT`) before it was built — the carry was already latency-hidden. Building the instrument
  first turned the wall from argument into fact.
- **Build the best the exposed knobs allow, then label (A)/(B).** Refutations are results: `SCHED_UNROLL_SPLIT`,
  `Q_HOIST`, and the ds_permute primitive were all *constructively* refuted, not inferred away.
- **Isolated ≠ in-model — only W==D transfers.** 2.35×→1.75×. Isolated timing is never promotion authority.
- **Discipline = correctness-first + default-off + revert-clean.** Every change is microgate-gated (max_abs
  1.526e-05), env-gated + cache-keyed (byte-identical when off), and reverts clean on failure with an exhaustive
  report. Two of three attacked levers reverted to zero residue.
- **Self-review catches what tests don't.** The adversarial review found that the model-wide-firing flags
  (`SCHED_UNROLL`/`COALESCED_LOAD`) transform *untested sibling kernels* — a silent-miscompile risk the
  block-tile gates can't see. Fixed by declining the unverifiable cases (non-constant bounds, multi-range END)
  and verifying in-model **token correctness** (the W==D harness only checks tok/s). This would have shipped a
  latent hazard otherwise.

## Where to go next (grounded in the diagnoses, not guesses)

1. **`b`-loop LDS-staged split** — the only lever that can bend the ctx-slope without the occupancy tax. Needs
   the recurrence-unroll to select the outer range and a combine epilogue in LDS.
2. **Occupancy guardrail gate** (VGPR/waves-per-CU from the isa descriptor) — auto-abort any pressure-increasing
   change; every partial-state primitive must pass it.
3. **More work-removal levers** (no new state) — they strictly dominate on an occupancy-bound tile.
4. **Split-aware audit tool** so a future split reads the `b`/`tt` carry shadow_fill directly and predicts
   success/failure before implementation.

## One-line state

The pure-search decode-attention foundation is **built, composing, transferring (~1.75× in-model), and
self-review-clean** — a real machine-generated-kernel capability — but the tile remains **3–15× off owned**, and
the diagnosed path to close it is the `b`-loop LDS split + occupancy discipline, not more of what's already shipped.
