# NV catch-llama fully measured ledger result

Date: 2026-08-22
Branch: `nvidia-bringup-20260731`
HEAD: `6570abc025514273faa100c66b979e531585a1e1`
GPU: RTX 5090, UUID `GPU-c800ade9-21ea-2e55-f75c-6d7a458fb186`
Model SHA-256: `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`
Oracle binary SHA-256: `947eb29052871f151719762c2fc265024e14f833b98df7801af9eb09da1625a8`

## 1. Terminal verdict

`NEED_MORE_INFO`

Did tinygrad catch llama: no.

Fresh unprofiled endpoint, same session, d512:

| endpoint | wall us/token |
| --- | ---: |
| tinygrad production control median | 4747.530 |
| tinygrad split-phase candidate | 4735.083 |
| llama control median | 4034.0045 |
| final delta | +713.526 |

The only concurrency candidate built this session is wall-neutral and was
removed. The production stack is unchanged, so the final candidate equals the
fresh control. This is an open, gated state, not a claim that llama cannot be
caught.

## 2. Findings, ordered by severity

1. `MEASURED` `CRITICAL` The remaining gap is device serialization and launch
   placement, not GEMV arithmetic. Exact-image counter replay puts the main
   bodies near oracle parity:

   | row | tinygrad us | llama us | delta us |
   | --- | ---: | ---: | ---: |
   | gate/up | 40.93 | 38.46 | +2.47 |
   | Q | 11.26 | 8.99 | +2.27 |
   | O | 10.78 | 9.06 | +1.72 |
   | V q4 | 5.73 | 4.61 | +1.12 |
   | V q6 | 5.63 | 5.22 | +0.41 |
   | down q6 | 34.72 | 31.42 | +3.30 |
   | vocab | 330.53 | 307.90 | +22.63 |
   | flash combine | 5.63 | 2.56 | +3.07 |

   The sum of those captured body deltas is about 37 us. The wall gap is
   713.5 us. Source of the row table:
   `docs/task_workflow/evidence/nv-catch-llama-ledger-20260822/phase2-row-counters.json`.

2. `MEASURED` `CRITICAL` Device union, not host time, carries the whole fresh
   gap. The corrected ledger closes:

   ```text
   wall_delta          = device_union_delta + host_gap_delta
   713.526             = 785.542            + (-72.016)
   ```

   `INFERRED` from the fresh wall arms and the same-commit profile union.
   The old locked +100.648 us tinygrad host-gap excess does not reproduce; it
   is now a 72.016 us tinygrad advantage. Source:
   `10-corrected-wall-ledger.json` and `08-host-gap-ledger.json`.

3. `MEASURED` `HIGH` The exact-image counter bridge is now qualified. The
   production `DEV=NV` cubin, launch geometry, and HCQ buffer sizes are
   captured verbatim and replayed through a CUDA driver primary context so NCU
   can see them. The bridge is in
   `extra/llm_research/decode/nv_cubin_capture.py:52`,
   `extra/llm_research/decode/nv_cubin_ncu_launcher.py:36`, and
   `extra/llm_research/decode/nv_row_counter_bridge.py:44`. Numeric output
   replay equality remains `UNMEASURED`, so the bridge is counter-domain
   evidence, not a semantic gate.

4. `MEASURED` `HIGH` The edge-aware split-phase concurrency candidate does not
   convert launch-ahead into unprofiled wall. Control A / candidate / control B
   were 4720.438 / 4735.083 / 4774.622 us. The candidate is slower than control
   A and inside the 54.2 us control spread. The scheduler-only arm lives behind
   `NV_SPLIT_PHASE=1` at
   `tinygrad/runtime/graph/hcq.py:27`,
   `tinygrad/runtime/graph/hcq.py:469`,
   `tinygrad/runtime/ops_nv.py:55`, and
   `tinygrad/renderer/cuda.py:38`. It is default-off, which is the correct
   production posture.

5. `MEASURED` `HIGH` Gate/up is a closed kernel-geometry target, not an open
   one. It reads identical 56.64 MiB at 78.49 percent of peak DRAM. A body-only
   candidate cannot cover its 2.47 us deficit without reducing bytes or raising
   bandwidth. `GATE_UP_NO_GO_WALL`.

6. `MEASURED` `MEDIUM` Flash score body is already faster than llama. The
   isolated A/B/A is tinygrad 4.160 / llama 3.744 / tinygrad 4.160 us per call,
   but the installed row gap is 64.540 us, leaving a 49.56 us
   `launch/L2/timeline` residual. That residual is the first named unmeasured
   mechanism. Source:
   `docs/task_workflow/output/nv-third-party-theory-audit-result-20260822.md`.

7. `MEASURED` `MEDIUM` Prior forced-concurrency and queue-placement probes were
   negative, and they constrain the remaining solution space: S1 support is a
   serial data chain, not idle work
   (`docs/task_workflow/output/nv-concurrency-ceiling-probe-result-20260821.md`),
   and the landed K/V pin already captures the small real overlap
   (`docs/task_workflow/output/nv-kv-overlap-result-20260822.md`).

## 3. Corrected endpoint ledger

Wall is unprofiled and measured this session. Union/overlap/node_sum are the
fresh same-commit profile values. Host gap is the inferred matched-route
residual.

| endpoint | wall | node_sum | resident union | resident overlap | host gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| tinygrad production | 4747.530 | 4677.920 | 4671.500 | 6.420 | 76.030 |
| tinygrad split candidate | 4735.083 | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| llama | 4034.0045 | 5008.638 | 3885.9585 | 1122.6795 | 148.046 |
| delta | +713.526 | -330.718 | +785.542 | -1116.260 | -72.016 |

Useful body, useful overlap, and spin-only union are `UNMEASURED` in every
arm. Reconciliation:

```text
device_union_delta + host_gap_delta = 785.542 + (-72.016) = 713.526 us
```

## 4. Work-package verdicts

| package | verdict | exact wall or body delta |
| --- | --- | --- |
| WP-D1 gate/up | `GATE_UP_NO_GO_WALL` | body +2.47 us; DRAM 78.49 percent |
| WP-D2 Q | `NO_GO_WALL_BODY` | main body +2.27 us; completion tail not wall-proven |
| WP-D3 O/down | `NO_GO_WALL_BODY` | O +1.72 us; down q6 +3.30 us |
| WP-D4 vocab | `NO_GO_WALL_BODY` | +22.63 us at 88.42 percent DRAM |
| WP-D5 flash | `UNMEASURED_WALL` | score body -0.32 us; 49.56 us launch/L2/timeline residual |
| WP-D6 K/V | `NO_GO_WALL_BODY` | prior four-warp K wall-neutral; pin removal regresses |
| Phase 4 host | `NO_HOST_TO_RECOVER` | fresh host delta -72.016 us |
| Phase 5 concurrency | `WALL_NEUTRAL` | +14.645 us vs control A |

No work package produced a promoted production change.

## 5. Cumulative stack and reverse ablation

```text
S0 = fresh production control        = 4747.530 us
S1 = S0 + NV_SPLIT_PHASE=1 arm       = 4735.083 us

incremental_stack_gain(S1)           = -12.447 us
promotion rule                       = fail, S1 is slower than control A
final stack                          = S0
cumulative_gain                      = 0.000 us
reverse ablation S1 -> S0            = +12.447 us, no shared-substrate credit
```

## 6. Body / bytes / bandwidth / launch decomposition

For every captured main body, the mechanism is now measured:

| row | bytes MiB | DRAM pct | mechanism |
| --- | ---: | ---: | --- |
| gate/up | 56.64 | 78.49 | DRAM-bound, near roofline |
| Q | 9.45 | 47.71 | launch/occupancy |
| O | 9.47 | 49.89 | launch/occupancy |
| V q4 | 2.38 | 23.61 | launch/occupancy |
| V q6 | 3.46 | 34.89 | near parity |
| down q6 | 41.34 | 67.54 | DRAM-bound-ish |
| vocab | 515.70 | 88.42 | DRAM-bound, 4.66 MiB extra read |
| flash combine | 0.808 | 8.17 | 20x more bytes than llama |

The pseudocode that separates body from wall:

```text
wall          = resident_union + host_gap
resident_union= sum(kernel residence with no double count)
node_sum      = sum(kernel residence, double counted)
overlap       = node_sum - resident_union

body deficit  = sum(replay tinygrad) - sum(replay llama)  # about 37 us
wall deficit  = union deficit + host deficit              # 785.5 - 72.0
launch/timeline residual = wall deficit - body deficit    # the open term
```

The measured body deficit cannot be the wall deficit; the wall deficit is
residence serialization plus the flash score launch/L2/timeline residual.

## 7. Host-gap causal decomposition

```text
host_gap_tinygrad = 4747.530 - 4671.500 = 76.030 us
host_gap_llama    = 4034.0045 - 3885.9585 = 148.046 us
host_gap_delta    = -72.016 us (tinygrad wins)
```

Python/JIT/graph-group/driver/sync/materialization/idle-gap categories are
each `UNMEASURED` because no matched CPU/API/GPU trace was retained this
session, and the `decode_runtime_overhead.py` D arm is slower than W so a W-D
host subtraction is refused. What is now proven is the direction: there is no
host amount left for tinygrad to recover. The remaining problem is on device.

## 8. Prior claims corrected

| prior claim | disposition |
| --- | --- |
| +100.648 us host gap is recoverable | stale; fresh host delta is -72.016 us |
| reduce/residual is a +452 us hotspot | refuted; classifier bucket error and tested fold regressed +59.602 us wall |
| llama overlap is useful concurrency | refuted; on/off changes 1122.7 us overlap but only 7.7 us union and 4352 bytes |
| tinygrad production bytes are about 5.04 GB | UNMEASURED; accounting estimate only |
| flash score gap is 109..112 us | stale; current installed gap is 64.540 us with 49.56 us unmeasured residual |
| split-phase PDL is not yet endpoint-measured | now measured unprofiled and wall-neutral |

## 9. Remaining gap and exact next measurement

`final_remaining_gap = 713.526 us`

It decomposes into `+785.542 us` device union excess and `-72.016 us` host
advantage. The next measurement must move from body counters to a
time-resolved launch/L2/timeline trace of the S1 support chain with per-kernel
wait-exit timestamps. Concretely:

```text
1. Instrument every S1 consumer with a %globaltimer wait-exit after its
   griddepcontrol.wait, matching the llama mmvq/norm/rope/quant wait sites.
2. For the same token, collect consumer grid-start, wait-exit, and useful-body
   intervals on the original timeline, not an NCU replay timeline.
3. Reconcile wall = union + host gap on that same trace, then assign the
   49.56 us flash residual and the remaining S1 serialization to named gaps.
4. Only then test the full llama-equivalent 761-edge programmatic chain,
   including support->support and multi-producer edges; the current 83-edge
   conservative arm is not the same mechanism.
```

This is a scheduler/concurrency and timeline-instrumentation task, not a
kernel-body task.

## 10. Evidence index and manifest

All required ledger artifacts are under
`docs/task_workflow/evidence/nv-catch-llama-ledger-20260822/`:

```text
00-provenance.json
01-baseline-wall-brackets.json
02-baseline-profile-ledger.json
03-counter-bridge-qualification.json
04-row-authority.json
05-individual-ab.json
06-cumulative-stack-ab.json
07-reverse-ablation.json
08-host-gap-ledger.json
09-final-oracle-brackets.json
10-corrected-wall-ledger.json
11-closed-theories.json
phase0/  phase1/  phase2-row-counters.json  phase3/  phase4/  phase5/
sha256.txt
```

`sha256.txt` is generated after all artifacts are final and verified with
`sha256sum -c`.
