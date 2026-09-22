# NV Q6_K vocabulary terminal route

Decision: **PROMOTE**.

The whole-model C/A/C bracket holds the promoted packed pp512 graph constant and changes only `NV_LLAMA_Q6_VOCAB_PP512`.

| arm | median prefill | throughput |
|---|---:|---:|
| control | 41.064352 ms | 12,516 tok/s |
| candidate 1 | 38.475373 ms | 13,355 tok/s |
| candidate 2 | 38.470494 ms | 13,350 tok/s |

Candidate mean is 38.4729335 ms. Recovery is 2.5914185 ms, or 99.4% of the independently measured 2.6069 ms vocabulary debt.

Correctness:

- token: 198 for control and both candidates
- candidate repeat max absolute difference: 0
- control/candidate logits max absolute difference: 0.2091818
- control/candidate logits mean absolute difference: 0.0322591
- full logits pass `atol=0.5, rtol=0.02`

The terminal route consumes canonical packed Q6_K weights directly, produces the final activation's canonical Q8_1 records in graph space, and launches the standalone HCQ-safe Q6_K dot-product cubin. No expanded weight or hot-path host copy is used.

After this promotion, the measured tinygrad-to-llama wall gap falls from 6.230075 ms to approximately 3.638657 ms when applied to the original 40.910442 ms promoted baseline. This is an accounting estimate; the next cross-runtime run must remeasure the residual categories on the new promoted graph rather than subtract stale regional values.

Primary machine-readable result: `bracket-result.json`.
