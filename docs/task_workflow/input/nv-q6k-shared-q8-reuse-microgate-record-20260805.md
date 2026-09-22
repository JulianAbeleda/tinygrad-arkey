# Shared Q8 attention-projection microgate

Date: 2026-08-05
Verdict: **research microgate PASS; llama-equivalence NO; production/token A/B not authorized**

## What was established

The pinned d512 llama graph has 217 `quantize_q8_1` nodes and 217
`mul_mat_vec_q` nodes: six of each per one of 36 layers plus the vocab
projection.  The trace establishes one adjacent Q8 producer for every MMVQ
consumer.  In particular Q, V, and K are each quantized separately even though
they consume the same attention-norm activation.  This is recorded by the
semantic manifest as `llama_observed_cross_mmv_q8_reuse: false` and
`observed_reuse_consumers: 1`.

That is an important correction to the proposed mechanism: shared Q8 is **not
the reason llama is fast**.  It is nevertheless a potentially valid tinygrad
optimization, so long as it is treated as a new route rather than an attempted
llama reproduction.

## Direction-setting experiment

`extra/llm_research/decode/q6k_shared_q8_reuse_microgate.py` compares:

```text
baseline:  3 * (installed Q6_K partial4 + external sum), same fp16 activation
candidate: 1 * tinygrad Q8_1 pack(common activation) + 3 * tinygrad DP4A Q6_K
```

The three Q6 buffers are separate payloads.  A `DEBUG=2` capture schedules
seven baseline calls (three partials, three reductions, scalar combine) and
six candidate calls (three pack-stage calls plus three DP4A Q6 consumers).  It
does not silently CSE the three consumers.  Independently running the pack
twice produces bitwise-identical packed values and scales.

On native `DEV=NV`, RTX 5090, 500 replays and nine samples per arm:

| arm | median us / three projections |
| --- | ---: |
| baseline A/B/A midpoint | 170.080161 |
| shared-Q8 candidate | 80.052252 |
| candidate - baseline | **-90.027909** |

The compact, machine-readable result is
`docs/task_workflow/output/nv-q6k-shared-q8-reuse-microgate-20260805.json`.

## Interpretation and hard boundary

This clears only a *shape-local, three-consumer* included-cost microgate.  It
does not qualify a token route:

1. The model's Q/K/V activation is fp16 while llama's is fp32 at this boundary;
   full model logits must establish the numerical contract.
2. The test uses identical 1024x4096 Q6 shapes and does not yet prove that the
   actual Q/V/K model buffers/lifetimes can share one packed buffer without
   creating materializations or harming scheduling.
3. Llama does not share this buffer, so no llama causal credit is claimed.
4. No Q4 FFN-down work is implied: its activation has a different lifetime and
   must enter through a separately admitted included-cost gate.

The next authorized step is a native, default-off all-18-Q6 attention reverse
A/B with full-logit and token gates, a topology census proving one pack per
actual Q/V/K group, and A/B/A wall timing.  Promote nothing unless all gates
pass.  If the real graph adds conversions or loses the microgate result, record
the failure and leave the installed route unchanged.
