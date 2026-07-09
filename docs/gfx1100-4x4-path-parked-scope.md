# gfx1100 4x4 Generated Path Parked Scope

Date: 2026-07-08.

## Decision

Stop active development of the generated `4x4` prefill WMMA path on gfx1100.

The active generated path is now:

```text
2x2, 4x2, and 2x4
LDS/DBUF-style reuse and cadence
proof-safe A/B fragment reuse
wait/scheduler tuning only after correctness and density are proven
```

`4x4` is not deleted. It is parked as a diagnostic / future-resource path and should only be reopened when the target
hardware or codegen substrate can afford the register footprint.

## Why

The limiting resource is VGPRs, not VRAM.

On gfx1100, a wave has a hard 256 VGPR-per-lane allocation ceiling. A `4x4` WMMA tile needs:

```text
16 C subtiles * 8 fp32 accumulator VGPRs = 128 VGPRs
```

before any A/B fragments, DBUF phase storage, address carriers, scratch, or epilogue temporaries are allocated.

The DBUF variants then add enough live fragment/address state to hit the wall:

```text
128 C accumulators
+ resident/current A/B fragments
+ second phase or LDS staging carriers
+ address and epilogue scratch
> 256 VGPRs/lane
```

That explains the repeated failure mode:

- no-LDS/no-DBUF `4x4`: cannot compile without newer machinery,
- no-LDS + `PREFILL_DBUF=1` `4x4`: reaches 64 WMMAs but returns `NaN`,
- generated LDS/DBUF `4x4`: blocked by register/lifetime pressure and correctness hazards,
- smaller shapes fit and run.

The handwritten kernel proving that some `4x4` assembly can run does not mean the generated path should keep chasing it
on this target. The generated route has extra lifetime, proof, and bookkeeping constraints; with DBUF it does not have
the same register slack.

## Active 100% Definition

The current fast-prefill target is complete when:

| Gate | Requirement |
| --- | --- |
| A0. Correct active shapes | Generated `2x2`, `4x2`, and `2x4` are GPU-correct under the selected LDS/DBUF route. |
| A1. Packed LDS path | Promoted staging uses packed `global_load_b128`, `ds_store_b128`, and `ds_load_b128`; scalar fragment LDS is fallback-only. |
| A2. Proof-safe reuse | Resident A/B reuse requires role, slot, K phase, row/column identity, byte window, producer epoch, and overwrite epoch. |
| A3. Density improvement | Generated `ds_load/WMMA`, waits/WMMA, and instructions/WMMA move toward the hand LDS2 traces on `2x2`, `4x2`, and `2x4`. |
| A4. Timing win | Same-clock TFLOPS improves over current generated DBUF-safe baselines on active shapes. |
| A5. Default safety | Fast path remains opt-in until correctness and timing pass; parked `4x4` is never selected by default on gfx1100. |

## Reopen Criteria

Reopen `4x4` only if at least one condition changes:

1. A target GPU has a materially larger usable VGPR budget or different tensor-core/register contract.
2. The renderer gains a proven lifetime model that keeps `4x4` DBUF live state below the hardware cap with room for scratch.
3. A measured active-shape ceiling proves `2x2`, `4x2`, and `2x4` cannot reach the required prefill target, and a new
   resource proof shows `4x4` can fit.
4. The work is explicitly scoped as diagnostic research, not the active fast-prefill path.

Until then, do not spend implementation time on:

- no-LDS `4x4` NaN isolation,
- generated LDS/DBUF `4x4` spill fixes,
- `4x4`-specific accumulator or epilogue rewrites,
- scheduler/waitcnt tuning whose only payoff is `4x4`.

## Commands

Active generated-vs-hand comparisons should use:

```bash
python3 extra/qk/prefill/hand_vs_generated_shape_matrix.py \
  --generated-env current --skip-hand --shapes '2,2;4,2;2,4' --loc 2 --unr 2 --pin-clock --json
```

Only run `--shapes 4,4` when explicitly reopening the parked path.
