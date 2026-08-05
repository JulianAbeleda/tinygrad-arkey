# P5 Q6_K 1024x4096 family CUDA graph A/B — 2026-08-04

## Verdict

`PASS_CORRECTNESS_MATERIAL_WALL_SIGNAL`. Replacing the complete mapped family
of 18 live `q6k_gen_partial_1024_4096_4` nodes with a diagnostic chain
(`fp16->fp32`, exact llama Q8_1, extracted exact llama Q6_K MMQ, scatter back
to four partials) reduced d512 CUDA decode median wall time by **0.184 ms/token
(3.29%)** against bracketed unchanged CUDA controls in the corrected repeat.

This is the first repeated-family causal result. It identifies this Q6_K
1024x4096 family as a real part of the decode gap; it does not authorize a
production route change.

## Closed construction and topology gates

The family count was not hardcoded. The P2 semantic manifest contains 18
`Q6_K, rows=1024, K=4096` records. Fresh d512 CUDA graph construction found
exactly 18 `q6k_gen_partial_1024_4096_4` calls, distributed `[1, 3, 1, 3, 10,
0]` across the six graph groups. The harness aborts unless those counts agree.

For every replacement, the live ABI had to be exactly output float 16,384 B,
weight ushort 3,440,640 B, activation half 8,192 B; the frozen node had to
have at least one source dependency and exactly one consumer. The original
consumer was retained and rewired only from the destroyed partial node to the
new scatter node. Each graph's node-count change had to equal `+3` per mapped
node. All gates passed.

## Measurement

All arms used `DEV=CUDA`, RTX 5090, d512, the same model and prompt, and the
GPU lock. Each arm generated 31 decode tokens; the first (graph build or
reinstantiate) sample was discarded, leaving 30 steady samples. The complete
token sequence was identical in CUDA-control A, diagnostic A/B, and
CUDA-control B.

| arm | steady median ms/token |
|---|---:|
| CUDA tinygrad-role control A | 5.601961 |
| diagnostic 18-role A/B | 5.415198 |
| CUDA tinygrad-role control B | 5.597085 |
| bracket control median | 5.599523 |

The A/B delta is **-0.184325 ms/token**. It exceeds the requested 0.05 ms
resolution threshold and is bracketed by controls that differ by only 0.004876
ms at their medians. P5/p95/MAD for all three 30-sample arms are retained in
the compact payload. An earlier independent bracket measured -0.179373
ms/token; the corrected repeat differs by only 0.004952 ms, confirming the
mechanism-scale signal.

## Meaning and next gate

The isolated 10.49x role advantage did not transfer one-for-one: the graph
replacement adds a boundary conversion, Q8 launch, and scatter, while the
native partial output is consumed by an existing fused reduction/KV node. But
the replicated 3.20-3.29% improvement is stable enough to reject the claim that this
family is wall-neutral. The next diagnostic should isolate the retained
consumer/reduction boundary or repeat the same family at d2048 before judging
whether a production-compatible fusion can preserve the gain.

The compact result is
`docs/task_workflow/output/nv-decode-q6k-llama-family-graph-ab-20260804.json`.
