# NV R-residual PDL/concurrency adjudication result

> **Blocker update 2026-08-23:** the flush failure was re-derived as a
> measurement-kernel ABI issue: native HCQ reports dynamic NVRTC
> `blockDim.x == 0`. A baked-geometry 128 MiB flush passes spread readback,
> and reverse-bracket cache controls now separate. The production share of R
> remains unmeasured, but it is no longer blocked on a production-runtime
> timestamp/QMD fix. See
> `output/nv-r-hcq-harness-adjudication-result-20260823.md`.

Date: 2026-08-23
Branch: `nvidia-bringup-20260731`
HEAD: `6570abc025514273faa100c66b979e531585a1e1`
GPU: RTX 5090 (sm_120), clocks locked (pm 1, lgc 2850, lmc 14001)

This closes the Phase 11 Tier-0 gate: name the production-conditioned
residual `R = P - C` for the Q, O, K/V, and flash-score cubins as
`dependency_wait`, `launch_gap`, or `cache_state`.

## Verdict

`240_UNMEASURED`. Tier-0 disposition is `PARTIAL_PASS`: a route was identified,
but it was not demonstrated buildable. The prior upgrade to `240_BUILDABLE`
is retracted.

The campaign found a specific candidate schedule (K/V is serialized behind the
Q branch) and made a pure L2-cold explanation less likely, but the decisive
cache-eviction arm of the probe was invalidated by a flush bug, the `P - C`
partition was not measured to a zero residual, and no reverse-bracketed,
token-SHA wall result exists. This is "route identified", not "route
demonstrated buildable".

## What is solid

- [MEASURED] The `n=32` buffer bug is real and corrected: the production
  authority requires `[16384, 16384, 4096]`.
- [MEASURED] The six production cubin hashes match their retained authority
  binaries. Two of the eight rows are controls, so `282.14 us` covers six
  production rows, not every outstanding `R` row.
- [MEASURED] The retained schedule serializes K/V behind the Q branch. The
  approximately 14 us gap contains Q projection, Q completion, Q norm and
  rope work. This is a concrete scheduling target, but it is a schedule gap
  outside the `P - C` interval, not an explanation of `R`.
- [MEASURED] The original 128 MiB flush kernel is broken: the retained
  readback shows it wrote only block 0. See the flush-readback artifact.

## Why `240_BUILDABLE` does not follow

1. [MEASURED] The experiment did not partition `R`. `R = P - C` is inside the
   command interval. The ~14 us K/V and ~2.75 us flash predecessor gaps are
   outside that interval. They are separate schedule gaps, not an explanation
   of `R`.
2. [MEASURED] No wait-exit or actual kernel-entry observable was collected.
   The partition uses retained timestamp boundaries. For Q/O, `pred_gap = 0`
   results from timestamp-chain reuse; it does not distinguish QMD admission,
   launch delay, execution elongation, memory interference, or dependency
   waiting. The original Tier-0 contract was therefore not completed.
3. [MEASURED] The hot/cold timing rule was not reverse-bracketed. The probe
   always runs hot and then cold. Several hot arms visibly warm up
   (`q_coop` hot halves 6.048 -> 5.824 us; `q_g3` 8.288 -> 7.952 us;
   `O` 8.320 -> 8.048 us). The near-zero conclusion is plausible but unproven.
4. [MEASURED] The cache eviction itself is invalidated. The original flush
   kernel wrote only block 0 (`idx 0/1/7` correct; `idx 1023`, `16777216`,
   `33554431` still zero), so the `cold` arm never evicted L2. The
   `cache_state ~= 0` conclusion is therefore `UNMEASURED`, not evidence.
5. [INFERRED] The NCU result is challenged, not refuted. HCQ and NCU disagree,
   but assigning the entire difference to CUDA-context instrumentation
   requires a causal matched-context test. Production predecessor traffic,
   memory-controller state, TLB state and outstanding DRAM work remain
   possible explanations.
6. [INFERRED] Dependency-legal scheduling is not necessarily physically
   recoverable. The prior two-queue implementation measured only 2.1 us of
   overlap and regressed wall by about 44 us. This gate did not overturn that
   result or demonstrate a wait-tolerant native scheduling construction.
7. [UNMEASURED] The projected ceilings remain unbooked and potentially
   interacting. `404.1 us` named plus "some serialization" is still ceiling
   arithmetic. No experiment has shown that those terms add on wall or that
   at least `604.756 us` is recoverable.

## Method

Two measurements were attempted, both on the retained production cubins.

1. Native NV HCQ cold/hot replay (`nv_r_residual_cache_dispatch_probe.py`).
   Each cubin is launched on a real `NVComputeQueue`; the `hot` arm inserts no
   flush and the `cold` arm runs a 128 MiB streaming write before every
   launch. The flush kernel is now known to be broken (writes only block 0),
   so the `cold` arm is not a valid eviction. The `hot` arm remains a clean
   chained-HCQ slope.
2. Offline same-group predecessor-gap partition
   (`nv_r_residual_pred_gap_partition.py`) on the retained full-token capture.
   `pred_gap = start_us - max(same-group predecessor end_us)`. Cross-group
   edges are excluded because their `end_us` sits in a prior capture window;
   raw cross-group deltas of ~456 ms are capture-window artifacts.

Every cubin SHA-256 in the merged artifact matches its authority capture.

## Measured rows

`MEASURED` median of N=32 per arm, one fresh process per row. `B` is the exact
body, `C` the clean chained-HCQ slope, `P` the production command interval,
`R = P - C`. The `cold` arm is retained for the record but is invalidated as
an eviction by the flush bug; `cache_state` is therefore not a real cache
measurement.

| row | count | B us | C us | P us | R us | hot us | cold us | cache_state us | non_cache P-cold us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control norm 8 | 0 | 1.196 | 1.698 | - | - | 1.952 | 1.952 | 0.000 | - |
| control norm 32 | 0 | 1.190 | 1.698 | - | - | 2.176 | 2.176 | 0.000 | - |
| q_coop_4096 | 17 | 4.800 | 5.3093 | 8.416 | 3.107 | 5.872 | 5.920 | +0.048 | 2.496 |
| q_g3_4096 | 19 | 7.488 | 7.7512 | 8.704 | 0.953 | 8.192 | 8.064 | -0.128 | 0.640 |
| o_epi_4096 | 36 | 7.584 | 7.6979 | 9.184 | 1.486 | 8.176 | 8.192 | +0.016 | 0.992 |
| flash_score | 36 | 3.840 | 3.658 | 6.272 | 2.614 | 4.128 | 4.096 | -0.032 | 2.176 |
| kv_coop_1024 | 26 | 2.016 | 2.310 | 3.712 | 1.402 | 2.720 | 2.752 | +0.032 | 0.960 |
| kv_g3_1024 | 28 | 3.328 | 3.798 | 4.768 | 0.970 | 4.064 | 4.064 | 0.000 | 0.704 |

`MEASURED` the fixed-order hot-arm slope is a clean chained-HCQ baseline, and
the `hot`/`cold` arms are near-identical because the broken flush writes only
one block. The weighted `cold - hot = -1.36 us/token` is therefore noise
between two effectively-identical arms, not evidence about L2 residency.
`UNMEASURED` the real cache-state share of `R` until a validated, reverse-
bracketed eviction control is run.

## Predecessor gap

`MEASURED` same-group `pred_gap`:

| symbol | occurrences | pred_gap median us | p10-p90 us |
| --- | ---: | ---: | ---: |
| q_coop / q_g3 Q projection | 17 / 19 | 0.000 | 0.000-0.000 |
| O projection epi_resadd | 36 | 0.000 | 0.000-0.000 |
| flash score | 36 | 2.750 | 1.75-3.50 |
| kv_coop K/V projection | 26 | 14.875 | 14.0-18.25 |
| kv_g3 K/V projection | 28 | 14.125 | 13.0-18.75 |

`MEASURED` Q and O start exactly at their predecessor's end (chained start
reuse). `MEASURED` the K/V projection waits ~14 us after its input norm and
flash score waits ~2.75 us after its K/V completion producer. Those are
queue-placement/serialization gaps. 1203 of 1230 edges are same-group; the 27
cross-group edges are the capture-window boundary. These gaps are schedule
gaps, not a partition of `R = P - C`.

## Reconciliation with prior evidence

`INFERRED` the native-HCQ hot arm (4.096-4.128 us) challenges the Phase 6
CUDA-context NCU cold flash-score result (cold 6.08 us). Because the native
eviction arm is broken, this does not refute a DRAM-cold hypothesis on the
production path; it only shows the CUDA-context cold number is not reproduced
by a non-evicting native replay. A causal matched-context test is still
required.

`MEASURED` this is consistent with the prior HCQ dispatch-slope result
(dispatch floor 0.65 us, `NOT_GLOBAL`, timestamp tax ~0.15 us). The clean HCQ
command path is cheap.

Prior evidence constrains any scheduler route:

- `output/nv-concurrency-ceiling-probe-result-20260821.md`: forcing flash/norms
  onto GPFIFO 1 costs +656.9 us/token; S1 is a serial dependency chain.
- `output/nv-pdl-queue-theories-test-20260820.md`: PDL costs +8-11 us with no
  recovery; a second queue is worth ~113 us but does not close the gap.
- `output/nv-installed-islands-phase10-ledger-result-20260822.md`: critical
  path 4205.376 us (38.7 us above 240), ~472 us dependency-legal overlap, only
  6.420 us realized overlap.

`INFERRED` together these mean "dependency-legal overlap exists" is not the
same as "physically recoverable overlap". A scheduler microgate must
demonstrate producer-to-join recovery without elongating the GEMVs before the
route can be promoted.

## Correct next gate

- [MEASUREMENT REQUIREMENT] Repeat hot/cold as H/C/H and C/H/C, discard
  warmup, and include a cache-sensitive positive eviction control.
- [MEASUREMENT REQUIREMENT] Replay each real predecessor immediately before
  Q, O, K/V and flash, preserving production buffers and ordering. Compare
  isolated, predecessor-conditioned and installed intervals.
- [MEASUREMENT REQUIREMENT] Separate queue admission, actual kernel execution
  and wait residence. `P - C` must close into named observables with zero
  residual.
- [MEASUREMENT REQUIREMENT] Build the smallest scheduler microgate that
  overlaps K/V against the measured Q branch and measure the complete
  producer-to-join span.
- [PROMOTION GATE] Require a positive token-SHA reverse wall bracket before
  declaring the scheduler route buildable.

## Current blocker

The corrected grid-stride flush and the minimal flush check hang with
`Wait timeout: signal not set to X, but X-1`, and `dmesg` shows
`Xid 31 MMU Fault ... FAULT_PTE ACCESS_TYPE_VIRT_READ @ 0x20_1f40c000`. This
is the same fault class as the earlier `q_coop` fault and lives in the native
HCQ timestamp/QMD-chain path (stale-QMD / `BumpAllocator` wrap on the reset
`q.active_qmd = None`), which is production-runtime territory. A valid
cold-arm cache-state measurement is therefore not currently obtainable with
this tooling.

## Evidence

- Probe tool: `extra/llm_research/decode/nv_r_residual_cache_dispatch_probe.py`
- Merge tool: `extra/llm_research/decode/nv_r_residual_merge.py`
- Partition tool: `extra/llm_research/decode/nv_r_residual_pred_gap_partition.py`
- Reverse-bracket tool: `extra/llm_research/decode/nv_r_residual_reverse_bracket.py`
- Flush check tool: `extra/llm_research/decode/nv_r_flush_check.py`
- Merged artifact: `docs/task_workflow/evidence/nv-r-residual-pdl-adjudication-20260823/nv-r-residual-cache-dispatch-probe.json`
- Partition artifact: `docs/task_workflow/evidence/nv-r-residual-pdl-adjudication-20260823/nv-r-residual-pred-gap-partition.json`
- Broken-flush readback: `docs/task_workflow/evidence/nv-r-residual-pdl-adjudication-20260823/nv-r-residual-flush-readback-broken.json`
- Hashes: `docs/task_workflow/evidence/nv-r-residual-pdl-adjudication-20260823/sha256.txt`
- Source capture: `docs/task_workflow/evidence/nv-third-party-theory-audit-20260822/probe2-tinygrad-capture.json`

No production, renderer, scheduler, runtime, model, or route-policy code was
changed by this adjudication.
