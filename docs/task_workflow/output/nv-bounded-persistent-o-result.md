# Bounded persistent O result

## Verdict

The first-class persistent O substrate is technically feasible but is not a
speed lever for the exact current Q4_K path.

The new research emitter is constructed directly from UOps. It executes the
complete installed vector-load Q4_K arithmetic, retains the installed lane
association and FP32 reduction order, and adds the residual in the same order.
It does not wrap or rewrite the generated O kernel text.

All tested worker counts made progress. The selected 1,024-worker arm is
bit-exact and finite on three independently rotated legal Q4_K fixtures, uses
39 registers, one block barrier, and has no spills.

No production routing changed and no token recovery is booked.

## Test decomposition

Three arms isolate the causal boundary:

1. **Installed O:** readiness is already satisfied and the installed
   4,096-CTA O kernel runs.
2. **Bounded O after readiness:** the new emitter is submitted normally after
   the same Flash producer.
3. **Bounded O launched ahead:** the new worker grid is made resident before
   Flash and waits on the producer's release epoch.

The producer is the validated full 192-CTA Flash score grid with last-CTA
per-head combine. The thirty-second completed head publishes one release
epoch. Control and candidate use separate rotated K/V caches and O weights.

## Worker sweep

The feasibility sweep exposes the expected population tradeoff:

| workers | bounded O cold | installed O cold | Flash-to-O consequence |
| ---: | ---: | ---: | --- |
| 32 | 51.776 us | 8.448 us | Too little row parallelism |
| 64 | 29.280 us | 8.768 us | Too little row parallelism |
| 128 | 16.928 us | 8.736 us | Still serialized |
| 256 | 13.856 us | 9.152 us | Still slower |
| 512 | 10.916 us | 10.796 us | O body reaches approximate parity; launch-ahead span still loses |
| 1,024 | 10.624 us | 10.774 us | Best O body; launch-ahead span still loses |

The 32--256 rows are one-repetition topology discriminators. Worker counts
512 and 1,024 received multi-repetition brackets, and 1,024 received the final
R9 bracket.

## Promotion-grade result

The final R9 medians for the 1,024-worker construction are:

| arm | hot | rotated-cold | recovery versus control |
| --- | ---: | ---: | ---: |
| Installed O only | 8.596 us | 10.646 us | -- |
| Bounded O only | 8.760 us | 10.548 us | +0.098 us/layer cold |
| Flash then installed O | 13.358 us | 14.968 us | -- |
| Flash then bounded O | 14.412 us | 14.910 us | +0.058 us/layer cold |
| Bounded O launched ahead of Flash | 17.572 us | 18.206 us | -3.238 us/layer cold |

The post-readiness apparent recovery is too small and inconsistent with the
hot result to justify integration. Launching ahead is a clear regression.

## Cause

The current O kernel consumes the complete 4,096-element attention vector.
Until every Flash head has combined, a full-row O worker has no useful work.
The launch-ahead grid therefore waits rather than overlapping material O
weight service with Flash.

Enough workers to preserve O's row parallelism also occupy substantial
scheduling capacity while waiting, which slows the producer. Reducing the
worker population lowers that pressure but serializes O rows and loses more
time than it can hide.

The exact partial-readiness alternative was tested separately: two O phases
can begin from half of the heads, but the required scratch round trip, second
O phase, and synchronization cost exceed the measured overlap. Together the
two results close persistent scheduling for the current exact ownership
contract.

This is not a hardware or compiler feasibility wall. It is a dependency and
ownership wall: the current representation does not expose useful O work
early enough without adding more work.

## Next lever

Do not invest in production graph support for this emitter. The next
high-recovery strict-token tests must change the work itself:

1. Co-design a stored quantized layout with its dequantization and execution
   pipeline, rather than rearranging identical Q4_K loads at runtime.
2. Test a smaller material representation or structured sparsity under a
   model-quality contract, then require a measured compulsory-DRAM reduction.
3. Keep cross-request batching separate as a serving-throughput lever; it
   does not improve strict one-token latency.

## Authority

- `extra/llm_research/decode/nv_flash_o_bounded_persistent_emitter_gate.py`
- `docs/task_workflow/evidence/nv-flash-o-bounded-persistent/final-w1024-r9.json`
- `docs/task_workflow/evidence/nv-flash-o-bounded-persistent/final-w1024-artifacts/`
- `docs/task_workflow/evidence/nv-flash-o-bounded-persistent/bracket-w512-r5.json`
- `docs/task_workflow/evidence/nv-flash-o-bounded-persistent/bracket-w1024-r5.json`
