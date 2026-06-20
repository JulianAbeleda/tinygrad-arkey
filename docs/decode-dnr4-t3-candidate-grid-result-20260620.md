# Decode DNR-4 T3 Candidate Grid Result - 2026-06-20

Verdict: `BLOCKED_DNR4_T3_NO_MATERIAL_NATIVE_LEVER_UNBLOCK_ATT`

T3 built the combined candidate from the two remaining native-side ideas:

- DNR4-T2 low-register q4/q8 preload;
- C7C unpack-all-then-dot issue ordering;
- DNR4-T1 low reduction/tail reuse.

The candidate is correct and structurally valid, but it does not beat C7C and does not clear the material timing gate.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_dnr4_t3_candidate_grid_probe.py --timing-warmups 4 --timing-iters 12 --pmc-warmups 1 --timeout-s 420
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_dnr4_t3_candidate_grid_result.json
```

## Timing

Same-process interleaved timing:

| variant | median us | delta vs native |
|---|---:|---:|
| native DNR-2 | `355.544` | `0.000` |
| best static DNR-3C6 | `339.699` | `-15.845` |
| C7C unpack-dot + dsload_b128 | `325.963` | `-29.581` |
| DNR4-T2 low-band preload | `334.008` | `-21.536` |
| DNR4-T3 low-band unpack-all-then-dot | `328.066` | `-27.477` |

T3 deltas:

| comparison | result |
|---|---:|
| T3 vs native | `+27.477us` |
| T3 vs best static | `+11.632us` |
| T3 vs C7C | `-2.103us` |
| T3 vs T2 | `+5.942us` |

The best row remains C7C at `325.963us`. T3 is close, but it is not the winner and misses the material gates:

- `>=30us` vs native;
- `>=15us` vs best static;
- `>=10us` vs C7C.

## Structural Gates

| gate | result |
|---|---:|
| all variants correct | yes |
| PMC runs OK | yes |
| T3 candidate buildable | yes |
| T3 has no high `v80-v95` band | yes |
| T3 preserves 16 `dot4` ops | yes |
| T3 material timing | no |
| best variant is T3 | no |
| counter predictive signal | no |
| renderer default changed | no |

## PMC Read

The profiler captures succeeded, but they did not produce a trustworthy search objective.

The issue/wait pass shows T3 lowers normalized SQ busy and VALU relative to the other native rows, but `SQ_WAIT_ANY`
does not order the winners cleanly. C7C is still fastest despite a higher issue-pass `SQ_WAIT_ANY` than T2/T3 in this
capture. The LDS pass also shows no bank-conflict story: LDS bank conflict remains zero across the grid.

So the counters are directionally useful for rejecting obvious regressions, but this run does not justify BEAM/search.

## Decision

Native decode is now blocked on attribution, not construction:

- q4/q8 addressing is not the blocker;
- scale/min extraction is not the blocker;
- dot4 selection is not the blocker;
- low-register preload is correct but not enough;
- unpack-all-then-dot is still the best local native schedule row;
- combining T2 and C7C does not create an additive win;
- PMC does not provide a reliable search objective.

Next step: unblock ATT PC timeline decoding, or bring a new route-level decode primitive. Do not continue local
native count-matching rewrites without PC/stage stall attribution.
