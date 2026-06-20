# Decode DNR-4 T3 Issue/Latency Scope - 2026-06-20

Verdict: `PASS_DNR4_T3_ISSUE_LATENCY_SCOPE_READY`

DNR4-T2 made the low-band q4/q8 preload route correct and structurally cleaner, but the timing did not clear the
promotion gate. DNR3-C7D already showed that C7C issue ordering moves wait/busy counters in the expected direction,
but that movement also did not become a material wall-time win.

So T3 is an attribution step, not another static rewrite. The purpose is to decide whether native decode still has a
credible local schedule lever, or whether the branch must stop until ATT gives PC-level stage attribution.

## Current Position

| input | decision |
|---|---|
| DNR4-T2 low-band preload | correct, no high `v80-v95` band, `18.36us` vs native, only `2.69us` vs best static, `10.20us` slower than C7C |
| DNR3-C7D C7C issue ordering | correct, PMC wait/busy movement present, only `6.01us` vs best static |
| oracle semantic map | S3 is the central body: q4 nibble select, waitcnt ladder, 16 dot4 ops, q8/q4 scaling, final fma |
| ATT timeline | blocked because the decoder library is missing |
| DNR3-C9 ledger | native schedule rewrites are parked unless a reopen gate produces new information |

## Tools

| tool | status | use |
|---|---|---|
| same-harness timing | ready | compare native, best static, C7C, T2, and combined candidates in one run |
| native PMC issue/wait/cache | ready, directional | test whether `SQ_WAIT_ANY`, `SQ_BUSY`, VALU/SALU, cache, and LDS counters track wall time |
| oracle semantic map | ready, static | anchor all decode candidates to S0-S5 and especially S3 |
| ATT PC timeline | blocked | required for exact oracle/native PC-stage stall attribution |

## T3 Experiments

| id | work | pass condition |
|---|---|---|
| T3A candidate grid | time native, best static, C7C, T2, and T2+C7C if buildable | one candidate clears `>=30us` vs native, `>=15us` vs best static, or `>=10us` vs C7C |
| T3B native PMC correlation | collect issue/wait/cache/LDS counters for the same grid | at least one counter family orders the timing winners well enough to predict a material candidate |
| T3C combined issue shape | build low-band preload + C7C unpack-all-then-dot + T1 low reduction | correctness holds and timing beats both T2 and C7C materially |
| T3D ATT unblock | install/provide the trace decoder library and rerun ATT | decoded PC timeline joins stalls to S3/S4/S5 instructions |

## Do Not Do

- do not start BEAM/search from static shape similarity;
- do not add more load-count, wait-count, branch-count, LDS-count, or marker-count patches without attribution;
- do not reopen Q4_K addressing, q8 addressing, scale/min extraction, dot4 selection, or gate/up correctness;
- do not promote DNR4-T2 from structural correctness;
- do not claim oracle PC-level attribution until ATT produces decoded timeline packets.

## Next Probe

Build `extra/qk_decode_dnr4_t3_candidate_grid_probe.py`.

Minimum output:

- correctness for every candidate;
- median timing for native, best static, C7C, T2, and T2+C7C if buildable;
- PMC issue/wait/cache table for the same candidate grid;
- a correlation decision: whether counters predict wall time or only move directionally;
- a hard stop if no candidate/counter pair identifies a material native-side lever.

Search remains blocked until this produces a trustworthy objective. ATT remains the route for exact PC-level blame.
