# Train/inference mismatch localization (interim, 2026-09-25 16:15)

Owner: fp16 agent. Harness: `~/scratchpad/fp16/` (numerics.py, consist4.py, layerdiff.py, analyze.py).
Design context: `batch-invariance-design.md` (22122d0e3).

## Method
Exact teacher-forced decode (the production sampler graph with the token forced; matches sampled logprobs to 5e-6)
vs full recompute on the same tokens. A "unified" site computes identical math in fp64 on prefill, decode and
recompute. Real 4B, PL=200 prompt, B=8, 64 steps, seeds 0/1.

## Results (gap = |logp_decode - logp_recompute|, nats)

| Variant | seed 0 mean / max | seed 1 mean / max |
|---|---|---|
| production (hi/lo) | 7.5e-4 / 6.6e-3 | 7.5e-4 / 9.3e-3 |
| **all sites unified** | **2e-8 / 1e-6** | - |
| kv32: K/V storage, q, P kept fp32 (prod kernels) | 3.3e-4 / 2.4e-3 | 3.4e-4 / 2.7e-3 |
| mm: all projections order-invariant | 5.0e-4 / 5.0e-3 | - |
| attnx: one fp64 attention without q/P rounding | 5.4e-4 / 6.0e-3 | - |
| attn: one attention with the model's roundings | ~no change | - |
| mamba: scan / replay / SSD in fp64 | ~no change | - |
| norm, lse | no change | - |

The all-unified row shows the site list is complete.

## Where it comes from
- Blocks 0-11 (Mamba/MLP) diverge only ~1.3e-5 relative (row-count-dependent matmul kernels; 2e-6 with mm unified).
- The first attention block (12) amplifies 40-100x to ~5e-4 and it stays there: **attention is the amplifier**;
  its bf16 rounding of K/V/q/P is the largest single lever (~2.5x).
- **Mamba scan order is not a meaningful source** (contrary to the literature's expectation).

## Real one-stack smoke (4k-token rollouts, blocks 0..40 recomputed in a batched teacher-forced pass)
smoke-001 max 0.207; smoke-002 max 0.306, mean 1.8e-3, p99 0.024, 5/4090 tokens > 0.1 (DayCare runs
`rloo-tinygrad-smoke-00{1,2}`). The long-sequence tail is far larger than the 64-step harness shows.

## Decision
Capture instead of recompute (design change 1): train on the sampler's own fp32 input to block 41 (the LoRA tail),
so blocks 0..40 are identical by construction. DayCare 5c07af7 consumes it; the sampler capture API is in progress.
Deeper invariance (fp32 attention storage/rounding, then order-invariant matmuls) only if LoRA moves below block 41.

## Bugs found
1. JIT=1 NV graph batching corrupts fp64 kernels (JIT=2 fine).
2. Candidate-route warm-start opts (`dense_candidate_gemm.py`) are keyed on shape only and get applied to any
   same-shape matmul (crashed the old fp16 arm with "no tensor core available"). Unassigned.

## Next
Leave-one-out table and 2k-step sequences for prod / kv32 / mm+kv32.
