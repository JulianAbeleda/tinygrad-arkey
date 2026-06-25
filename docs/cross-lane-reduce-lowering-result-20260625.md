# Cross-lane reduce lowering — Milestone 5 first pass (2026-06-25)

## Result: **`CROSS_LANE_REDUCE_AUTO_LOWERING_EXPOSED`** (opt-in, correct, ds_bpermute replaces the LDS tree)

First real step of "search instead of hand-tuned kernels" for the decode two-kernel problem
(`docs/generic-low-level-search-goal-scope.md` Milestone 5; `docs/pure-machine-search.md` "path to pure"). The
two hand-written decode kernels exist because the scheduler cannot emit `v_dot2` + **cross-lane reduction** + LDS.
This pass makes **cross-lane reduction auto-emittable by the scheduler** — the primitive the hand warp-GEMV is built
on — so a generic reduction the optimizer maps onto the wave now lowers to `ds_bpermute` instead of an LDS tree,
**with no hand-written kernel**.

## What shipped (behind `WARP_REDUCE_LOWERING`, default OFF)

- `extra/qk_warp_reduce_lowering.py` — `pm_warp_reduce`, a PatternMatcher that auto-detects an `Ops.REDUCE` over a
  single power-of-2 (≤32) WARP/GROUP_REDUCE lane axis (ADD/MAX, float) and rewrites it to the staged `ds_bpermute`
  xor-ladder (built on the existing `extra/amd_warp_reduce.py` primitives).
- `tinygrad/codegen/__init__.py` — injects `pm_warp_reduce` into the `expander` rewrite **before**
  `pm_group_for_reduce` (so it claims the lane reduce before the GROUP_REDUCE→LDS machinery), gated by
  `WARP_REDUCE_LOWERING` + AMD. The lane (the WARP/GROUP_REDUCE range) is left intact so `pm_add_gpudims` binds it
  to a real `lidx`. Added `WARP_REDUCE_LOWERING` to the `to_program` cache key.
- Distinction from `extra/amd_warp_reduce.py`: that exposes the ladder as a function you **call by hand** (an
  authoring escape hatch); this is the **automatic** rewrite a scheduler/search needs.

## Evidence (verified end-to-end on gfx1100)

- Unit (deterministic, no GPU): a constructed warp `REDUCE` rewrites to exactly log2(width) `ds_bpermute` shuffles
  and declines non-warp / non-pow2 axes. `test/external/test_warp_reduce_lowering.py` (5 structural tests).
- Pipeline (real renderer): a generic `Tensor` K=16 matvec — whose K-reduce the heuristic maps to `GROUPTOP(16)`,
  a single 16-wide group — auto-lowers with the flag on:
  - **sum correct** (max_err 4.8e-7), **max correct** (max_err 0.0) vs numpy;
  - rendered kernel emits the 4-step `ds_bpermute` ladder and **drops the `__attribute__((shared))` LDS buffer +
    `s_barrier`** the default path uses;
  - flag OFF is byte-for-byte the old LDS path (default behavior unchanged; matmul/reduce/elementwise sanity green).
  3 pipeline tests, all pass.

## Two bugs found and fixed during the attempt (the value of actually building it)

1. **Wrong test vehicle.** A hand-authored `custom_kernel` rejects the rewrite — `ranges ... are leaking out of the
   call`: custom kernels forbid open ranges, so the WARP range can't live there. The rule must run *inside* the
   codegen pipeline where `gpudims` binds the lane. (This is why the rule is a pipeline hook, not a kernel helper.)
2. **Gated-store lane divergence.** The first version was numerically wrong (max_err 6.7): when a reduce result is
   stored from a single lane (`if(lidx0==0)`), an *inline* `ds_bpermute` (plain `warp_reduce_sum`) gets pulled
   *inside* that divergent gate → cross-lane read of a masked lane → garbage. Fix: stage every shuffle into a REG
   (as `warp_reduce_max` already does) so all `ds_bpermute` run unconditionally before the gated store.

## Honest status & next steps (toward default-on / Milestone 6)

This is a **first pass**: the cross-lane primitive is now scheduler-emittable and correct, opt-in. It is **not yet
wired into the model** and **not default-on**. Remaining, in order:
1. **Mixed reduce** (the real GEMV shape): a K=4096 reduce becomes group + serial-K/group, which the rule currently
   *declines* (single-range only). Add the serial-then-ladder split so a real decode weight-GEMV reduce lowers.
2. **W==D / Milestone 6**: compare a search-/scheduler-generated GEMV using this lowering against the owned warp
   GEMV (`q4_k_gemv_primitive.py`) under the existing W==D + byte-exact gate. Only then consider default-on or
   retiring a hand kernel. Caveat (DNR arc): exposing the instruction makes the primitive *searchable* but may not
   alone reach owned-kernel perf — the *schedule* (waitcnt/clause/occupancy) is a separate, deeper gap.
3. **Layering**: the pipeline hook lazy-imports from `extra/`; if this graduates to default, move
   `pm_warp_reduce` (and the ladder) into `tinygrad/` proper. Also generalize beyond the gated-store case and add a
   `warp_size`/`subgroup_size` renderer field (currently assumes wave32).

## Manifest update

`bench/qk-search-spaces/decode_ffn_gemv_gfx1100_v1.json`: `cross_lane_reduction` moves from purely "excluded /
SEARCH_BLOCKED_BY_CODEGEN" to "exposed (opt-in `WARP_REDUCE_LOWERING`); default path still LDS pending mixed-reduce
+ W==D". The owned warp GEMV stays `manual_oracle_not_search_generated` until a search-generated GEMV clears W==D.
