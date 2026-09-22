# Exact predecessor-conditioned K/V and O closure

Date: 2026-08-23  
Branch: `nvidia-bringup-20260731`  
Commit: `6570abc025514273faa100c66b979e531585a1e1`  
GPU: RTX 5090 (sm_120), P0, observed SM 2842 MHz / memory 14001 MHz in all four authority sessions  
Authority:

- `docs/task_workflow/evidence/nv-r-predecessor-conditioned-exact-kv-20260823/closure.json`
- `docs/task_workflow/evidence/nv-r-predecessor-conditioned-exact-o-20260823/closure.json`

Tools:

- `extra/llm_research/decode/nv_r_predecessor_conditioned_exact_kv.py`
- `extra/llm_research/decode/nv_r_predecessor_conditioned_exact_o.py`

## Verdict

- `KV_OCCURRENCE0_CLOSED` with a surviving installed remainder
  `P - C5 = +0.520 us/call`.
- `O_OCCURRENCE0_CLOSED` with a surviving installed remainder
  `P - C5 = +1.032 us/call`.
- The scheduler/PDL microgate is **not justified** for either remainder:
  count-weighted they are `+13.520 us/token` (K/V) and `+37.152 us/token`
  (O), both below the `+50 us/token` promotion bar, and the only scheduler
  axis already bracketed on this commit is wall-negative.
- The mechanism behind the two positive remainders is `[UNMEASURED]`.

No parity, 240 tok/s, or wall-recovery claim is made. No production,
renderer, scheduler, runtime, model, or route code was changed.

## Findings, ordered by severity

### 1. Both rows close the timing identity, then leave a real installed remainder

Each authority row satisfies the required additive closure exactly:

```text
P - C0 = (C2-C0) + (C3-C2) + (C4-C3) + (C5-C4) + (P-C5)
```

Reverse-bracket medians across the two fresh processes per row:

| arm | definition | K/V us | O us |
| --- | --- | ---: | ---: |
| P | installed HCQ graph interval, steady replay | 4.000 | 9.424 |
| C0 | isolated target, exact captured inputs, production weight VA | 3.008 | 8.680 |
| C2 | immediate provider -> target | 2.560 | 8.216 |
| C3 | exact data chain -> target, cloned mutable buffers | 3.104 | 8.376 |
| C4 | same chain on every original live VA | 3.264 | 8.336 |
| C5 | exact production prefix + chain -> target | 3.480 | 8.392 |
| C6 | low-traffic padding at the same position + chain -> target | 3.592 | 8.376 |

- `[MEASURED]` `identity_residual_us == 0.000000` for both rows.
- `[MEASURED]` `P-C5` is positive and reproducible: `+0.520 us` K/V,
  `+1.032 us` O.
- `[MEASURED]` No other kernel overlaps any retained installed target
  interval: `profile_overlap_count == 0` for both rows.

### 2. The Q "live-prefix working set" mechanism does not generalize

For Q, the real production prefix was the lever: `C5-C6 = +0.664 us` and
`P-C5 = +0.008 us` (`nv-r-predecessor-conditioned-exact-result-20260823.md`).

- `[MEASURED]` For K/V, `C5-C6 = -0.112 us`: the real prefix is slightly
  faster than same-position padding.
- `[MEASURED]` For O, `C5-C6 = +0.016 us`: the real prefix is neutral.
- `[MEASURED]` Therefore the live-prefix/working-set mechanism that closed Q
  does not appear in either K/V or O. The positive `P-C5` is a separate,
  prefix-independent installed excess.

### 3. The Q-sibling omission was not the root cause of the K/V remainder

- `[MEASURED]` The retained launch order is now explicit and correct:
  gate/up `1470`, FFN down `1471`, provider `1472`, cooperative Q `1473`,
  cooperative K/V target `1474`. Both `provider.out == Q.in` and
  `provider.out == K/V.in` close on VA.
- `[INVALIDATED]` The earlier hypothesis that omitting the interleaved Q
  sibling was the cause of the K/V `P-C5` gap. After inserting the sibling
  into C5/C6, the remainder is still `+0.520 us` (the pre-fix session showed
  `+0.464 us` with a different P). Including the sibling is necessary for
  exactness, but it does not eliminate the remainder.

### 4. Command depth/padding is mostly neutral, with one small K/V exception

- `[MEASURED]` O: `C6-C3 = 0.000 us` and `C6-C5 = -0.016 us`. Position and
  padding do not move the O target.
- `[MEASURED]` K/V: `C6-C3 = +0.488 us`, so nineteen extra QMDs do add about
  half a microsecond to the K/V target residence. The real prefix adds
  slightly less than the generic pads (`C5-C6 = -0.112 us`).
- `[INFERRED]` The installed K/V excess is therefore not explained by generic
  command depth: the real prefix is the cheap case, and `P-C5 = +0.520 us`
  remains beyond even the padded reconstruction.

## Exact timing deltas

K/V (`26` calls/token):

```text
P-C0    0.992 = -0.448 + 0.544 + 0.160 + 0.216 + 0.520
C2-C0  -0.448
C3-C2  +0.544
C4-C3  +0.160
C5-C4  +0.216
P-C5   +0.520
C6-C5  +0.112
residual 0.000 us
```

O (`36` calls/token):

```text
P-C0    0.744 = -0.464 + 0.160 - 0.040 + 0.056 + 1.032
C2-C0  -0.464
C3-C2  +0.160
C4-C3  -0.040
C5-C4  +0.056
P-C5   +1.032
C6-C5  -0.016
residual 0.000 us
```

Count-weighted projections (attribution only, never booked wall recovery):

| row | calls | P-C5 per token |
| --- | ---: | ---: |
| K/V cooperative | 26 | +13.520 us |
| O projection | 36 | +37.152 us |

## Correctness SHA

- `[MEASURED]` Every arm in all four authority sessions passes the same
  target-output SHA.
  - K/V: `2a5f96aa6908e0e854fdfe17264566e44853a58b0a3542c4390d4a56930fea09`
  - O: `a5ca76164e6902c086f9f961025dee2e8e6b804a2b12ce06ce1632c017dcd946`
- `[MEASURED]` All four processes produce the identical token stream SHA
  `fc90421da972a70fd043dd57f4d30e53c4dd78918a3292725bdb42e63750585a`.
- `[MEASURED]` The two sessions per row share identical cubin SHAs and
  identical prefix names.
- `[MEASURED]` `sha256sum -c sha256.txt` passes for both retained evidence
  directories.

## Measured mechanism observables

- `[MEASURED]` Target interval wall: the arm medians above.
- `[MEASURED]` Installed overlap: zero overlapping intervals in the retained
  profiles for both rows.
- `[MEASURED]` Live VA dependencies: all named edges close exactly.
- `[UNMEASURED]` `node_sum`, union, useful body, wait-exit, kernel entry/exit,
  predecessor gap, and DRAM/L2 counters were not captured by this gate. The
  positive `P-C5` therefore cannot yet be partitioned into QMD admission,
  dependency wait, memory-controller/TLB state, or command-wall versus
  kernel-time mismatch.

## Scheduler microgate justification

- `[MEASURED]` The exact-live residual is real and reproducible, so a
  scheduler microgate is technically eligible.
- `[MEASURED]` Its own promotion bar is not reachable: the largest candidate
  term is O at `+37.152 us/token`, below the `+50 us/token` threshold, and
  the K/V term is `+13.520 us/token`.
- `[MEASURED/REFERENCE]` The same-commit placement microgate
  (`nv-kv-overlap-result-20260822.md`) already bracketed the alternate
  ready-placement candidate and found it wall-negative:
  `207.9 tok/s` candidate versus `210.9 tok/s` landed reference, token SHA
  identical.
- `[INFERRED]` Running another scheduler/PDL bracket now would test an axis
  with no remaining headroom above its gate. The microgate is therefore
  **not justified** for these two rows at this time.

## Ranked next actions

1. Instrument wait-exit and kernel entry/exit timestamps for K/V and O so
   `P-C5` partitions into launch/admission versus execution with zero
   residual.
2. Collect DRAM/L2/TLB counters for the same occurrences under a validated
   reverse-bracket eviction control.
3. Revisit the separate `~14 us` K/V predecessor gap only through a
   producer-to-join microgate if a future row clears the `+50 us/token`
   promotion bar.
4. Do not book the `13.520 + 37.152 = 50.672 us/token` arithmetic as
   recovery; the terms are measured at one occurrence and are attribution
   ceilings, not demonstrated wall.

## What remains unmeasured

- The exact causal mechanism of `P-C5` for K/V and O (L2, TLB, dependency
  wait, QMD admission, or timestamp/harness mismatch).
- Whether occurrence 0 generalizes to the other 25 K/V and 35 O calls.
- Any wall recovery, which would require a fresh token-SHA reverse wall
  bracket above the promotion bar.

## Evidence integrity

- `[MEASURED]` Four authority sessions: forward and reverse arm order for
  each row, fresh processes, `PROFILE=1`, locked clocks, P0.
- `[MEASURED]` Raw per-sample arm timestamps, installed graph profile JSONL,
  token IDs/SHA, output SHA, cubin SHAs, VAs, sizes, and clock state are
  retained for every session.
- `[MEASURED]` `py_compile` passes for all four tools and `git diff --check`
  passes.
