# Q/O/K/V roofline campaign result

## Decision

No production recovery is booked from this campaign.

Three previously open constructions were tested through their relevant boundaries:

1. A static one-row-per-CTA Q/K/V stripe was exact but slower in the cold regime and did not improve the measured DRAM service signature.
2. Flash-combine successor prefetch improved the isolated combine-to-O chain, but regressed the full token wall.
3. The first Flash-combine-owned Q8 production campaign was invalid: block-local admission was overwritten and candidate/control captured the same graph. After repairing admission and a reversed multi-output ABI, the genuine route is finite but fails the recurrent full-logit quality contract even at one-layer coverage.

The stripe and prefetch results are composition closures. Combine-owned Q8 is instead a numerical-admissibility closure for the present Q8_1 representation. No valid Q8 token-wall result exists because quality correctly stopped the campaign first.

## Baseline and accounting

The arithmetic authority remains the recorded dense Q/O/K/V ledger:

| Family | Payload | Time | Effective rate |
|---|---:|---:|---:|
| Q | 339.739 MB | 297.216 us | 1.143 TB/s |
| O | 339.739 MB | 304.976 us | 1.114 TB/s |
| K/V | 189.334 MB | 233.568 us | 0.811 TB/s |
| Q/O/K/V | 868.812 MB | 835.760 us | 1.040 TB/s |

The 1.75-TB/s size fit is an asymptotic service model, not a hardware roof. Moving the combined pool to 1.40 TB/s exposes about 215 us/token arithmetically, but ordinary per-layer grouping cannot cross the Flash-to-O dependency or erase every short-service ramp. No part of that arithmetic exposure is booked without a token-wall pass.

## Test 1: static Q/K/V stripe

The new microgate holds output ownership, material weights, row grammar, and launch count fixed while comparing a phased one-task grid with a 4Q:1K:1V interleaved grid.

R9 medians (microseconds):

| Arm | Hot | Rotated cold |
|---|---:|---:|
| Installed Q plus K/V | 8.485 | 14.117 |
| Existing Q-first full grid | 6.888 | 11.759 |
| Phased one-task | 6.817 | 11.992 |
| Interleaved stripe | 6.813 | 11.915 |

The candidate was exact. Under cache-controlled counters, however, interleaving took 13.088 us versus 12.352 us for Q-first, with essentially unchanged DRAM bytes and more instructions. The stripe therefore wins only in the hot regime and is rejected for the streaming production path.

## Test 2: successor O prefetch

Flash combine prefetched the first Q4_K groups of each successor O row without adding a launch or changing numerical output. The isolated chain favored early two-group staging by about 0.52 us/layer in the rotated-cold experiment. The complete token bracket rejected it:

| Boundary | Result |
|---|---:|
| Candidate minus midpoint control | +19.391 us/token |
| Throughput change | -1.227 tok/s |

The cause is composition: moving weight traffic into combine changes the graph/cache schedule enough that the isolated O recovery does not survive the full lifecycle.

## Test 3: combine-owned Q8 O construction

Flash combine emitted both its normal FP16 attention output and the exact Q8_1 packet consumed by the existing Q8 O body. This removes the standalone Q8 producer boundary while retaining the same rounding point.

The original qualification and wall results in this directory are retained as provenance but are **invalid for policy judgment**. Two substrate defects were found:

- Model entry refreshed every block from the model-wide geometry, silently erasing block-local `o_q8_owned` admission. Candidate and control were identical.
- After admission was repaired, the multi-output program passed `(out,q8,partial)` to an emitter expecting `(out,partial,q8)`. This caused out-of-bounds partial reads and consumed an unwritten Q8 packet.

Both defects now have regression coverage. A live one-block census proves one `_q8o` combine, one Q8 residual O consumer, and 35 ordinary O consumers. The corrected route is recurrently finite.

Corrected qualification results:

- All 36 layers: relative L2 0.00460, max absolute 0.21860; argmax/tokens stable but top-10 membership changed. Fail.
- Prefix eight layers: relative L2 0.00172; top-10 membership stable. Fail.
- Prefix four layers: relative L2 0.00331. Fail, demonstrating non-monotonic layer sensitivity.
- Block 0 only: relative L2 0.00267. Fail.
- Block 1 only: relative L2 0.00190. Fail.

The contract limit is relative L2 0.001 plus stable argmax/top-10 membership and bounded decision-margin perturbation. Since even a single tested layer fails and its gross arithmetic exposure is below 0.8 us/token, further layer-sensitivity search is not economically justified now. The correct production-equivalent timing gate and Q8-only combine optimization were deliberately not run after the quality stop.

The earlier R5/R7 layer-count wall curves measured identical graphs and are null-bracket noise. They must not be cited as either a win or a graph-resource regression.

## SASS audit

The current exact FP16 Q/O body is dominated by instruction classes required for Q4_K unpack/dequantization and exact accumulation (including FFMA, LOP3, SHF, integer-to-float conversion, and half arithmetic). The Q8 body removes much of that work because its activation representation changes. No independent exact FP16 instruction class was identified that can simply be deleted while keeping the same representation and association.

## Ledger of tested theories

| Theory | Primitive result | Token result | Booked |
|---|---|---|---:|
| Static one-task QKV stripe | Exact; hot positive; cold/counters negative | Not advanced | 0 us |
| Flash-combine O prefetch | Exact; isolated cold positive | +19.391 us regression | 0 us |
| Exact FP16 instruction lowering | No separable redundant class found | Not advanced | 0 us |
| Shared-Q8 stripe | Conditional on rejected stripe | Closed by prerequisite | 0 us |
| Flash-combine-owned Q8 O | Corrected route finite; useful coverage fails quality | Wall not admissible | 0 us |

## What remains open

The next genuine roofline-scale construction is deeper Flash-to-O segmentation: consume completed head segments and begin exact O partial projection work before the full attention vector is complete. That changes the causal topology rather than moving the same work between adjacent kernels. It is substantially larger engineering and needs a new exact-association design plus a complete-span microgate before production work.

Byte reduction remains a separate model-artifact lane. It should not be mixed into the exact dense-kernel ledger.

## Evidence

Campaign evidence is under `docs/task_workflow/evidence/nv-qokv-roofline-20260827/`. Primary artifacts include the stripe R9/counter captures, O-prefetch chain and wall brackets, Q8 native gates, full-logit qualifications, and the 8/16/20/24/36-layer wall brackets.
