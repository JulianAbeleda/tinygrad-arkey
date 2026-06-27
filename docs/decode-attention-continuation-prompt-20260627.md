# Continuation prompt — decode-attention pure-search (2026-06-27)

You are in `/home/ubuntu/tinygrad-arkey` on AMD gfx1100 (RX 7900 XTX), Qwen3-8B-Q4_K_M. Goal = **pure machine
search**: the *machine* (codegen + BubbleBeam) generates competitive kernels, not hand-asm. The **decode
attention tile is the LAST default hand kernel** blocking this (the Q4_K GEMV is already pure/generated under
BubbleBeam G3). Read first: `docs/decode-attention-pure-search-state-and-learnings-20260627.md` (state+learnings),
`docs/pure-machine-search-roadmap.md` (authoritative live state), `docs/decode-tile-structural-deltas-scope-20260627.md`
+ `docs/decode-tile-delta-attack-result-20260627.md` (deltas + attack outcomes).

## What is built (this session) — all default-off, cache-keyed, proving-ground-tested, committed

A composable codegen-primitive stack on the generated block-tile (`flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel`
in `extra/qk_flash_decode.py`, route-wired via `DECODE_ATTN_BLOCK_TILE=1`):

| primitive | flag | file | commit |
|---|---|---|---|
| recurrence-unroll + list scheduler | `SCHED_UNROLL=<U>` / `SCHED_LIST=1` | `extra/qk_codegen_recurrence_unroll.py`, `extra/qk_codegen_list_scheduler.py` | 522c74eca |
| coalesced-load lowering (`OptOps.COALESCE` realization) | `COALESCED_LOAD_LOWERING=1` | `extra/qk_coalesced_load_lowering.py` | f93ee9cfc |
| cooperative-staging LaneMap | `DECODE_STAGE_COALESCE=<W>` (use 4) | `extra/qk_cooperative_stage_lanemap.py` | 245d82c90 |
| work-removal: bare exp2 | `DECODE_FAST_EXP2=1` | `extra/qk_flash_decode.py` `_fexp` | 951455234 |
| layout-IR M1 LayoutFn+compose | (lib) | `extra/qk_layout_fn.py` | bf8dea852 (pre-session) |
| coalescing predicate | (lib) | `extra/qk_layout_coalesce_check.py` `axis_stride` | M0 + 81b05ad9c |
| hotloop schedule-diff audit tool | (tool) | `extra/qk_decode_hotloop_schedule_diff.py` | 3158b6677 |
| in-model token-correctness check | (tool) | `extra/qk_decode_token_match_check.py` | 9dbefbf7a |

**Best stack** (`DECODE_STAGE_COALESCE=4 COALESCED_LOAD_LOWERING=1 SCHED_UNROLL=8 SCHED_LIST=1 DECODE_FAST_EXP2=1`):
**2.54× isolated** (1.024→0.403 / 7.289→2.875 ms @ctx512/4096), **~1.75× in-model** (route 19.0/3.5 → 32.8/6.2
tok/s vs owned 103.2/93.8). Self-review-clean (hardening 9dbefbf7a: declines non-const-bound + multi-range-END
recurrences, graceful fallbacks). **Still 3–15× off owned in-model — a search-capability result, NOT a promotion
candidate** (owned ships, HBM-bound at llama parity). Do NOT claim it as a shipped win.

## The diagnostic truths (proven — don't re-derive or re-chase)

1. The gap is **LATENCY/ILP-bound, NOT throughput** (generated emits FEWER instr 414 vs 561, WIDER loads, MATCHED
   occupancy). The old "scalar loads / d16=0" framing is dead.
2. **The tile is OCCUPANCY-BOUND** (vgpr88 on the unroll stack, 4 wg/CU ceiling) ⇒ **levers must REMOVE work, not
   add ILP-via-state.** `SCHED_UNROLL_SPLIT` (VGPR 88→144) and `DECODE_Q_HOIST` (lost comgr LICM) were REFUTED+reverted.
   `DECODE_FAST_EXP2` removed work → won (+8–9%).
3. **The ctx-slope is the OUTER `b`-block-loop carry, NOT the inner `tt` carry** (unroll already hides tt-carry).
4. **`ds_bpermute` cross-lane is at per-token parity with owned** → NO cross-lane primitive warranted (it'd only
   match, not beat). ds_permute fully diagnosed — don't reopen.

## THE NEXT LEVER (start here)

**`b`-loop LDS-staged split accumulation** — the only lever that can bend the ctx-slope without the occupancy tax:
- Extend `extra/qk_codegen_recurrence_unroll.py` to select the OUTER `b` recurrence range (today it unrolls one
  loop; selection now prefers the outer token loop — needs to reach `b`), giving K=2–4 independent block-partition
  online-softmax partials over disjoint `b`-ranges, **combined once in the 8 KB LDS tile (NOT VGPR** — that's what
  killed the tt-split). Combine math: M=max_u mx_u; acc=Σ acc_u·exp(mx_u−M); den=Σ den_u·exp(mx_u−M) (reuse
  `flash_*combine` math at `qk_flash_decode.py:1037-1062`).
- **Build the occupancy guardrail gate FIRST** (VGPR/waves-per-CU from the isa-vectorization descriptor decoder,
  `extra/qk_decode_physical_tile_route_integration_gate.py:56-64 _parse_desc`): auto-abort if a change drops below
  baseline waves/CU. Every partial-state primitive must pass it — it would have caught the tt-split VGPR crash
  before the bench.
- **Use the split-aware audit** `extra/qk_decode_hotloop_schedule_diff.py` (now enumerates + classifies
  inner/outer loop candidates, so it can read the `b`/`tt` carry shadow_fill and predict a split's success before
  you implement it). Run it after each ISA capture to confirm the lever actually moved the right loop.
- Also keep pursuing **work-removal levers** (no new state) — they strictly dominate on an occupancy-bound tile.

## Discipline (non-negotiable)

- **Gates (authority):** correctness = `extra/qk_decode_attention_block_tile_microgate.py` → `BLOCK_TILE_MICROGATE_PASS`
  at max_abs 1.526e-05 (run with your flags). Isolated timing = `extra/qk_decode_block_tile_isolated_timing.py`
  (reads `bench/qk-decode-block-tile-isolated-timing/latest.json`). In-model authority = W==D
  `extra/qk_decode_runtime_overhead.py` (`QK_CKPTS=512,4096`) + token-correctness `extra/qk_decode_token_match_check.py`
  (the W==D harness only checks tok/s). **Isolated timing is NEVER promotion authority** (the repo's hard lesson).
- **Default-off + cache-keyed:** every new flag getenv-gated AND added to the `to_program` cache key
  (`tinygrad/codegen/__init__.py` ~line 270). Flag unset ⇒ byte-identical. The flags fire MODEL-WIDE on all AMD
  custom kernels — DECLINE the unverifiable (non-const bounds, untested shapes), and verify token-correctness.
- **Measure-first / audit-before-attack.** Build the instrument, refute the lever, THEN implement (this session's
  `SCHED_UNROLL_SPLIT` refutation saved a blind alley). Refutations are results — record them.
- **Revert clean on failure** + exhaustive report. **Self-review** before declaring done (it caught a model-wide
  miscompile risk this session).
- Commit convention for the layout/pure-search stream: `[nn]` (code) / `[docs]` / `[test]`, **NO Co-Authored-By
  trailer** (matches bf8dea852). Surface SHA+title after committing.

## Verify-it-still-works smoke (run on resume)
```
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python -m unittest test.external.test_coalesced_load_lowering test.external.test_cooperative_stage_lanemap test.external.test_layout_fn
DEV=AMD JIT=1 DECODE_STAGE_COALESCE=4 COALESCED_LOAD_LOWERING=1 SCHED_UNROLL=8 SCHED_LIST=1 DECODE_FAST_EXP2=1 PYTHONPATH=. python3 extra/qk_decode_attention_block_tile_microgate.py   # -> BLOCK_TILE_MICROGATE_PASS
```
Memory `[[pure-machine-search-goal]]` is current (GEMV done, decode-attention frontier, the two meta-findings).
