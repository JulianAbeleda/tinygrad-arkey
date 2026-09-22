# Exact predecessor-conditioned Q closure

Date: 2026-08-23  
Branch: `nvidia-bringup-20260731`  
Commit: `6570abc025514273faa100c66b979e531585a1e1`  
GPU: RTX 5090, P0, observed SM 2842 MHz / memory 14001 MHz in both authority sessions  
Authority: `docs/task_workflow/evidence/nv-r-predecessor-conditioned-exact-20260823/closure.json`  
Tool: `extra/llm_research/decode/nv_r_predecessor_conditioned_exact.py`

## Verdict

`Q_OCCURRENCE0_CLOSED`

- [MEASURED] The first `q4k_warp_coop_q8_dp4a_partial_4096_4096`
  production-conditioned interval closes with `0.000000 us` arithmetic
  residual in two fresh-process forward/reverse authority sessions.
- [MEASURED] A replay of the exact 19-command preceding prefix plus the exact
  data-linked FFN/provider chain reproduces installed P to `+0.008 us`:
  `P=8.352 us`, `C5=8.344 us`.
- [MEASURED] Command position is neutral: nineteen low-traffic padding QMDs at
  the identical target position produce `C6-C3=-0.040 us`, while replacing
  those pads with the real prefix produces `C5-C6=+0.664 us`.
- [MEASURED] No other kernel overlaps any of the 18 retained installed-P Q
  intervals across the two authority profiles.
- [INFERRED] The installed Q excess is predecessor working-set state, not
  arithmetic, dependency wait, useful overlap, or QMD depth. The exact split
  among L2 data, TLB/page translation, and instruction/code state is
  unmeasured because this gate uses timing and interval observables, not
  counters for those three structures.
- [UNMEASURED] Generalization from occurrence 0 to all 17 cooperative-Q calls.
  Count-weighted numbers below are projections, never booked wall recovery.

## Findings, ordered by severity

### 1. The earlier C0/C2/C3 artifact did not close production P

- [MEASURED] The old tool measured only C0, C2, and C3; it contained no
  same-session installed-P arm.
- [MEASURED] Its C3 used newly allocated, zero-filled FFN buffers, including
  substitute weight allocations. It therefore measured an exact-cubin,
  same-sized stream, not the exact live predecessor mappings.
- [MEASURED] Its reported output SHA was calculated once for isolated replay
  before the C0/C2/C3 timing arms. It did not checksum the C2 and C3 outputs,
  despite the report wording that the checksum was identical across all three
  runs.
- [MEASURED] The old `sha256.txt` retained only `capture-only.json` and one
  `c0-c2-c3.json`; the first of the two reported sessions was not retained as
  raw evidence.
- [INFERRED] The old `+1.1..1.5 us` result was directionally useful as a cache
  sensitivity signal, but “actual production predecessor” and “gate closed”
  were overstated.

### 2. The exact production prefix, not QMD position, reproduces installed Q

All six micro arms time only the target Q interval. Predecessor bodies are
outside the two target timestamps.

| Arm | Exact definition | Reverse median (us) |
| --- | --- | ---: |
| P | installed HCQ graph interval, normal steady replay | 8.352 |
| C0 | isolated Q, exact captured input, production Q-weight VA | 6.096 |
| C2 | exact provider -> Q | 5.704 |
| C3 | data-linked gate/up -> down -> provider -> Q; immutable production weights, cloned mutable buffers | 7.720 |
| C4 | same chain on every original live VA | 7.656 |
| C5 | exact 19 production-prefix calls -> C3 chain -> Q; Q is command 23 | 8.344 |
| C6 | 19 low-traffic provider pads -> C3 chain -> Q; Q is command 23 | 7.680 |

- [MEASURED] Every dependency VA closes:
  `gateup.out == down.in`, `down.out == provider.in`, and
  `provider.out == Q.in`.
- [MEASURED] Every arm passed the same target-output SHA
  `f957b6531dbd0fd302fc5d59ae925e8f0dd4f36130f93e069c3559567d6bc9a1`.
- [MEASURED] Both authority processes produced token SHA
  `fc90421da972a70fd043dd57f4d30e53c4dd78918a3292725bdb42e63750585a`.
- [MEASURED] C3 versus C4 is wall-neutral at `-0.064 us`; cloned mutable-buffer
  VAs do not explain installed P.
- [MEASURED] C6 versus C3 is wall-neutral at `-0.040 us`; QMD position and a
  longer submission do not explain installed P.
- [MEASURED] The real prefix contributes `C5-C6=+0.664 us`, and C5 then closes
  installed P to `P-C5=+0.008 us`.

### 3. Corrected per-call identity

```text
P - C0 = (C2-C0) + (C3-C2) + (C4-C3) + (C5-C4) + (P-C5)

  2.256 =   -0.392 +    2.016 +   -0.064 +    0.688 +    0.008 us

identity residual = 0.000000 us
```

- [MEASURED] The provider makes the target `0.392 us` faster than isolated C0;
  it is not a target-residence penalty.
- [MEASURED] The exact FFN gate/down/provider chain contributes `2.016 us` of
  target residence.
- [MEASURED] The earlier real prefix adds another `0.664..0.688 us`, depending
  on whether it is referenced to the same-position C6 control or live-VA C4.
- [MEASURED] Installed graph state after the full prefix leaves only
  `+0.008 us`, below one 32 ns timestamp quantum.

### 4. Bridge to the frozen clean-chain ledger

The retained clean chained replay is `C=5.3093 us`; the current installed P is
`8.352 us`, versus frozen P `8.416 us`.

```text
current P - clean C = 3.0427 us/call
clean C -> C0        = 0.7867
C0 -> C2             = -0.3920
C2 -> C3             = 2.0160
C3 -> C4             = -0.0640
C4 -> C5             = 0.6880
C5 -> P              = 0.0080
sum                   = 3.0427 us/call
```

- [MEASURED] Current P is `0.064 us/call` faster than frozen P. This accounts
  for the difference between the current `3.0427 us` and frozen `3.1067 us`
  residuals; it is not an unexplained ledger error.
- [INFERRED/PROJECTED] Multiplying occurrence 0 by all 17 family calls gives
  `51.726 us/token` current clean-C-to-P exposure: `13.374 us` clean-chain
  baseline mismatch, `-6.664 us` provider benefit, `34.272 us` local FFN
  conditioning, `-1.088 us` live-VA difference, `11.696 us` earlier-prefix
  conditioning, and `0.136 us` installed remainder.
- [UNMEASURED] That `51.726 us/token` is not node-sum, union, wall, or useful
  body recovered. It is a count-weighted projection from one occurrence.

## Decision

- [MEASURED] A generic “reduce HCQ dispatch/QMD depth” change has no lever on
  this Q occurrence: the same-position C6 control is `0.040 us` faster than
  C3, and installed P has no overlapping interval or wait gap.
- [INFERRED] The correct construction target is late Q-weight residency:
  prefetch the next Q weight after the destructive prefix, reorder only when
  dependency-legal and residency-preserving, or fuse work only if the
  surviving Q consumer receives a hotter working set.
- [MEASURED CEILING, NOT WALL] `P-C2=2.648 us/call` is the occurrence-level
  hot-target ceiling with the provider retained. It is not a measured
  implementation result and must not be booked.
- [UNMEASURED] K/V and O may have different mechanisms. In particular, this Q
  result does not promote or reject the separate K/V two-queue microgate.

## Corrections to reviewer statements

- [MEASURED] The earlier reviewer criticism that provider and cooperative-Q
  ordinals necessarily select different layers was wrong. Both installed
  families occur 17 times, and occurrence 0 is adjacent after the first FFN.
  The exact sequence is verified here by global call positions 1470..1473 and
  three matching dependency VAs.
- [MEASURED] The provider layout is
  `[packed_q8_output, fp32_hidden_input, bf16_norm_weight]`, not the projection
  layout `[output, weight, input]`.
- [MEASURED] The valid criticism was the synthetic FFN mappings, absent P arm,
  absent per-arm output validation, absent reverse raw session, and resulting
  inability to close the installed residual.

## Evidence integrity

- [MEASURED] Authority Sessions E/F are fresh processes, opposite arm orders,
  P0, persistence enabled, observed SM 2842 MHz and memory 14001 MHz.
- [MEASURED] Each session retains raw per-sample arm timestamps, installed
  graph profile JSONL, token IDs/SHA, output SHA, cubin SHAs, VAs, sizes, and
  clock state.
- [MEASURED] `sha256sum -c sha256.txt` passes for all retained development and
  authority artifacts.
- [MEASURED] No production, renderer, scheduler, runtime, model, or route code
  was changed by this gate.

