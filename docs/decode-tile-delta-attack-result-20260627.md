# Decode-tile delta-attack campaign — 1 closed, 2 refuted, two meta-findings (2026-06-27)

Scope: `docs/decode-tile-structural-deltas-scope-20260627.md`. Method: measure-first audit
(`extra/qk_decode_hotloop_schedule_diff.py`) → sequential attack workflow per Tier-1 delta with an
audit-diagnose-fix-revert loop (each closed-and-committed or reverted-clean + exhaustive failure report).

## Outcome

| delta | flag | status | result |
|---|---|---|---|
| cheaper exp2 (drop range-reduction) | `DECODE_FAST_EXP2` | **CLOSED** (`951455234`) | **+8–9%** isolated; new best stack **2.54×** |
| 2-level split accumulation (tt-carry) | `SCHED_UNROLL_SPLIT` | refuted, reverted clean | every split SLOWER; premise false |
| Q register-hoist + drop f16 convert | `DECODE_Q_HOIST` | refuted, reverted clean | +25%/+21% REGRESS; comgr already does it |

New best isolated stack (`DECODE_STAGE_COALESCE=4 COALESCED_LOAD_LOWERING=1 SCHED_UNROLL=8 SCHED_LIST=1
DECODE_FAST_EXP2=1`): **0.403 / 2.875 ms** @ctx512/4096 (baseline 1.024 / 7.289). Microgate PASS, default-off
byte-identical. ds_permute earlier diagnosed **at parity** (no cross-lane primitive warranted).

## The win — `DECODE_FAST_EXP2` (work-removal)

Online-softmax exp args are always ≤0 (`old_m-new_m`, `sc-new_m`), so the ocml range-reduction (guarding
large/denormal magnitudes) is dead weight on the serial carry chain. The flag lowers `_fexp` to a bare
`__builtin_amdgcn_exp2f`. Block-tile ISA: `v_ldexp_f32` 16→0, `v_cndmask` 64→30, 783→669 lines. Numerics
unchanged (the clamp is exact in the valid range).

## The two meta-findings (these redirect all future work)

1. **The tile is OCCUPANCY-BOUND** (vgpr 88, scratch 0, 4 wg/CU — at the gfx1100 ceiling). Any lever that adds
   live register state regresses: `SCHED_UNROLL_SPLIT` K=8 pushed VGPR 88→144 (occupancy crash), `Q_HOIST`
   pinned the Q half2s and lost comgr's pressure-aware LICM. **Winning levers must REMOVE work / pressure, not
   add ILP-via-state.** `FAST_EXP2` removed work → won. ⇒ build a **default-off occupancy guardrail gate**
   (VGPR/waves-per-CU from the isa-gate descriptor) that auto-aborts any change dropping below baseline waves/CU,
   so the "VGPR erases the win" trap is caught at codegen time, not the bench.

2. **The ctx-slope (gap-grows-with-ctx) is the OUTER block-loop `b` carry, NOT the inner `tt` carry.** The
   recurrence-unroll selects only the innermost REDUCE range (`tt`); splitting it left the slope unchanged
   (7.16→7.14×). The across-block online-softmax carry over `b` (NB = Tc/(S·TK) blocks) is the serial
   ctx-scaling chain. **Any slope-bending split must target the `b` loop, and stage its partials in the
   already-allocated 8 KB LDS tile (not VGPR)** to dodge the occupancy tax that killed the tt-split.

## Why the refutations matter (they are results)

- **`SCHED_UNROLL_SPLIT` refuted the scope's #1 lever.** The baseline `SCHED_UNROLL=8` serial re-thread ALREADY
  hides the tt-carry — the 8 copies' independent score/dot/ds_bpermute prologues overlap in one block and the
  carry sits in their latency-shadow. Splitting added zero ILP + a combine epilogue + the VGPR crash. "More
  split = monotonically slower" is the signature of an overhead-only transform on a non-binding chain.
- **`Q_HOIST` refuted Tier-1 #3.** `q[h·Hd+e2]` is loop-invariant; comgr/LLVM LICM already hoists it
  pressure-aware (re-materializing vs hoisting to keep VGPR low). Forcing it at the UOp level removes that
  freedom on an occupancy-bound kernel → regress. Do not re-chase (mirror "decode combine EXHAUSTED").

## Next attack plan (from the diagnoses)

1. **Re-scope the split to the `b` loop, LDS-staged.** K=2–4 independent block-partition online-softmax partials
   over disjoint `b`-ranges, combined once; partials live in LDS. This is the only split that can bend the
   ctx-slope without the occupancy tax. Requires the recurrence-unroll to select the outer range (today it
   `break`s after the innermost) — a real extension.
2. **Build the occupancy guardrail gate** and require every partial-state primitive to pass it.
3. **Make the audit tool split-aware** (its backward-branch heuristic locked onto the wrong loop under the
   split; ds_bpermute=0) so a future attempt reads the tt/b carry shadow_fill directly and predicts failure
   *before* implementation.
4. Continue work-removal levers (no new state): they strictly dominate on an occupancy-bound tile.

## Discipline held

All three: microgate authority (PASS at max_abs 1.526e-05), default-off byte-identical, revert-clean on failure
(no broken state), exhaustive failure reports with attack + diagnosis plans. Isolated timing is not promotion
authority — the in-model W==D + ctx-slope remain the gate; the closed `FAST_EXP2` is a foundation increment, not
a shipped default.
