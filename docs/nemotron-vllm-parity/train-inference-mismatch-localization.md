# Train/inference mismatch localization (updated 2026-09-25 evening)

Owner: fp16 agent. Harness: `~/scratchpad/fp16/` — `sites.py` (per-site unification, monkeypatch only),
`localize.py` (teacher-forced decode vs full recompute), `mambalocal.py` (Mamba layer-local divergence),
`pfcheck.py` (prompt-phase prefill vs recompute), logs `loc_*.log` / `loc_*.json` / `loc_*.npz`, `mambalocal.log`.
Code measured: tinygrad exp @ e67ec30a5 (`NemotronHBatchSampler`, `NemotronHPrefill`, full `block.cached`
recompute). Design context: `batch-invariance-design.md`.

## Method
- **Same tokens for every variant.** Rollouts are sampled once with the production sampler (prefill-primed,
  T=1, Gumbel). Each variant then **teacher-forces those tokens through the production sampler graph** (the
  `_step` graph, with the token fed from a host buffer instead of sampled). This reproduces the sampled logprobs
  to 5e-6 (64 steps) / 2e-5 (2048 steps). The same tokens also go through the **full recompute**
  (`block.cached` over prompt+completion, batch 8 or 4). Gap = |logp_decode - logp_recompute| in nats.
- **Unifying a site.** A unified site computes the same math on all three paths (prefill, decode, recompute),
  with its reductions in fp64 and the result rounded to fp32 at the site boundary. Two fp64 reductions of
  identical inputs differ by about 1e-16 relative, so after the fp32 rounding they agree except with
  probability about 1e-9 per element: the site stops contributing reduction-order noise. The sites are:
  - `mm`: all bf16 projections.
  - `norm`: the block pre-norms and output_norm.
  - `mamba`: the whole mixer (conv, dt, scan / replay / SSD), with fp64 state and ring buffers.
  - `attn`: one reference fp64 attention with the model's q/P→bf16 roundings.
  - `attnx`: the same attention without the q/P roundings.
  - `lse`: the vocab log_softmax.
- **`kv32`** is not an fp64 site. It uses the production kernels, keeps the K/V cache (sampler, prefill and
  recompute) plus q and P in fp32, and runs the prefill attention through the unfused fp32 path.
- **Config.** Real Nemotron-H 4B BF16, deterministic 200-token prompt, B=8, temperature 1.
- **Run flags.** Every run used `JIT=2`, because `JIT=1` NV graph batching corrupts fp64 kernels (see Bugs).

## Results

### 64 steps (PL=200, B=8), single sites
| Variant | seed 0 mean / p99 / max | seed 1 mean / p99 / max |
|---|---|---|
| production (hi/lo) | 7.5e-4 / 5.4e-3 / 6.6e-3 | 7.5e-4 / 3.5e-3 / 9.3e-3 |
| kv32 (fp32 K/V + q + P, prod kernels) | 3.3e-4 / 1.9e-3 / 2.4e-3 | 3.4e-4 / 1.5e-3 / 2.7e-3 |
| mm (order-invariant projections) | 5.0e-4 / 3.0e-3 / 5.0e-3 | 5.5e-4 / 2.4e-3 / 3.7e-3 |
| attnx | 5.4e-4 / 3.8e-3 / 6.0e-3 | 4.7e-4 / 2.6e-3 / 6.2e-3 |
| attn | 7.0e-4 / 4.5e-3 / 6.1e-3 | 6.8e-4 / 3.4e-3 / 6.8e-3 |
| mamba | 7.0e-4 / 4.6e-3 / 9.1e-3 | - |
| norm | 7.3e-4 / 4.3e-3 / 9.2e-3 | - |
| lse | 7.5e-4 / 5.4e-3 / 6.6e-3 (identical to prod) | - |
| **all unified** (mm+norm+mamba+attnx+kv32+lse) | **2.0e-8 / 5.9e-7 / 9.5e-7** | - |

The all-unified row shows that the site list is complete: nothing else differs between decode and recompute.

### 64 steps, leave-one-out (everything unified except the named site)
| Left in production | seed 0 mean / p99 / max | reads as |
|---|---|---|
| projections (noMM) | 3.3e-4 / 1.8e-3 / 2.4e-3 (seed 1: 3.4e-4 / 2.6e-3 max) | **the dominant source** |
| K/V/q/P bf16 rounding (noKV) | 1.6e-7 / 4.1e-6 / 1.7e-5 (seed 1: 1.4e-6 / 1.6e-4 max) | harmless when its inputs agree |
| attention core algorithm (noAttn) | 3.7e-7 / 1.9e-6 / 2.9e-6 | negligible source |
| Mamba decode / SSD / scan (noMamba) | 8.3e-7 / 3.8e-6 / 5.7e-6 | negligible source |
| RMSNorm (noNorm) | 2.4e-6 / 1.1e-5 / 2.8e-5 | negligible source |
| **mm + kv32 only** (all else production) | **2.8e-6 / 1.3e-5 / 2.9e-5** | ~250x better than production |

### 2048 steps (PL=200, B=8, seed 0, 16k tokens)
| Variant | mean / p99 / max |
|---|---|
| production | 4.2e-4 / 4.8e-3 / **2.0e-2** |
| **kv32** | **1.0e-5 / 8.5e-5 / 2.4e-4** (42x mean, 84x max) |
| all-but-mamba (Mamba in production) | 8.0e-7 / 7.9e-6 / 2.5e-5 |
| mamba-only, mm, mm+kv32 | not run yet (the GPU was yielded to the R3 smoke) |

At long length the bf16 rounding of K/V/q/P is nearly the whole tail. Each decode step reads thousands of
rounded keys and values, and every key or value whose rounding flipped moves the attention output.

### Mamba layer-local divergence (compare vLLM RFC #55524: 0.07% fp32 cache, 0.22% bf16 cache)
Setup: the same block input goes into the trainer's chunked `block.cached` (scan_chunk 64) and into the sampler
path (prompt state from the prefill SSD with chunk 256, then the one-token replay step with a 16-slot ring
folded into the checkpoint every 16 tokens). The SSM state and ring are fp32. PL=200, T=256, B=8, blocks
0/6/11/19/26/31/36.

| Projections | mixer-output rel. L2 divergence | elements with any bit difference |
|---|---|---|
| production (row-count-dependent kernels) | 1.2e-5 (0.0012%) | 99.5% |
| order-invariant (`mm`): scan/replay/SSD order only | 1.6e-6 (0.00016%) | 95% |

- Our divergence is 60x (production) to 400-1400x (scan only) below vLLM's, and it does not grow over 256 tokens
  (the first 16 tokens are the same as the average).
- The fraction of elements with any bit difference is near 100% in both cases, so that metric says nothing about
  magnitude. Likely reason for the gap to vLLM (not tested): vLLM's Triton scan kernels carry bf16/fp16
  intermediates, while our scan and state are fp32 end to end.
- End to end, leaving Mamba in production costs at most 2.5e-5 max at 2048 steps (all-but-mamba row).

## Mechanism (one source × one amplifier)
1. **Source: projections pick different kernels by row count.** Decode rows ≤16 use the fp32 matvec, 17-128
   rows use a routed hi/lo, the recompute runs the generic hi/lo tensor-core path, and prefill runs routed pieces
   of up to 1024 rows. Each has a different K-reduction order, which gives about 1.3e-5 relative hidden-state
   noise through blocks 0-11 (2e-6 with `mm`). Mamba scan, RMSNorm and attention-core order add only 1e-6-level
   noise.
2. **Amplifier: bf16 rounding in the 4 attention blocks.** K/V storage, q and P are rounded to bf16. With about
   1e-5 input noise, a fraction of roughly 1e-5/2^-9 of the elements flip by one bf16 ulp. The hidden-state
   divergence therefore jumps 40-100x at block 12 (the first attention block) and stays there. Any single
   remedy gives ≤2.5x at 64 steps, because the other factor still acts. Removing both gives ~250x.
3. **Where the tail lands.** Gaps are larger on low-probability sampled tokens: at p<0.1 the mean gap is ~1e-3
   and the max ~9e-3, versus a mean of 1.3e-4 at p>0.5. They are flat over position within 64 steps.

## Real one-stack smoke (context, from the integration agent)
smoke-001 max 0.207; smoke-002 max 0.306, mean 1.8e-3, p99 0.024, 5/4090 tokens > 0.1 (DayCare runs
`rloo-tinygrad-smoke-00{1,2}`, 4k-token rollouts, blocks 0..40 recomputed in a batched teacher-forced pass through
`NemotronHRolloutSampler`). That tail is larger than this harness's 2048-step production max (2.0e-2). The smoke
differs from the harness in several ways: the rollout sampler's ring KV and `rollout_attention`, a batch-4 recompute
padded to 512-token buckets, and real prompts and lengths. None of these has been measured here.

## Decision (coordinator)
Capture instead of recompute: train on the sampler's own fp32 input to block 41 (the LoRA tail), so blocks 0..40
are identical by construction (DayCare 5c07af7; sampler capture landed in d8924c8d9).

## Recommended invariance plan (only if the trainer recomputes blocks below the capture point)
The capture decision (train the LoRA tail on the sampler's own block-41 input) makes blocks 0..40 identical by
construction and stays first. If a recompute of the frozen stack is ever needed:
1. **kv32 first.** Keep the K/V cache, q and P in fp32 in the sampler, prefill and trainer.
   - Measured: 42x mean and 84x max at 2048 steps, which alone passes max ≤1e-2 by 40x.
   - Memory cost: 4 attention layers × 1024 K + 1024 V × 2 extra bytes = +16 KiB per token (bf16 16 KiB → fp32
     32 KiB). That is 0.5 GB more at 8 × 4k tokens and 8.6 GB more at 128 lanes × 4k.
   - Decode cost: the KV read doubles, so decode attention bandwidth doubles at long context.
   - Kernel work: the flash prefill kernel (fp16 internally) needs an fp32-K/V variant. The kv32 measurement used
     the unfused fp32 prefill attention.
2. **Then row-count-invariant projections**, for the remaining ~1e-5 (mm+kv32 reached 2.8e-6 mean at 64 steps).
   - What: a fixed K-reduction order per output element for every M. That means one K-tile schedule with no
     split-K that varies with M, and the same kernel for decode (B rows) and recompute (B×L rows). The
     alternative is to route both paths through the same route family.
   - Cost: the ≤16-row fp32 matvec is the fastest decode kernel, so the invariant version is likely slower for
     small-batch decode.
3. **Not needed:** Mamba scan/replay invariance, RMSNorm, attention split-KV order, and log_softmax order. Each
   is ≤2.5e-5 max even at 2048 steps.

## Bugs found (side findings)
1. **`JIT=1` NV graph batching corrupts fp64 kernels.** The fp64-norm and fp64-mamba variants gave garbage (gap
   0.3-3 nats). `JIT=0` and `JIT=2` are correct, and `NO_MEMORY_PLANNER=1` does not fix it.
2. **Candidate-route warm-start opts are keyed on shape only** (`dense_candidate_gemm.py:97`
   `_install` keys the opts by `warmstart_key(out_dims, reduce)`; `postrange._warmstart_match` applies them). Any same-shape matmul in another dtype gets the
   forced TC opts and raises "no tensor core available". This crashed the old fp16 arm, and the harness works
   around it by padding K by one.
3. **Symbolic-offset read inside a TinyJit step.** A forced-token read `forced[:, position+1]`, with position a
   bound UOp variable, returned wrong tokens from the second jitted call on (logprob error 0.3-0.6). A host-fed
   buffer works. This may be a harness misuse; it is not confirmed as a tinygrad bug.

## Next (after the R3 smoke)
- At 2048 steps: mamba-only, mm, and mm+kv32.
- Seed 1 for the leave-one-out table.
- The prefill-primed vs prefix-primed split of the prompt phase. Prefill K/V vs recompute K/V is 1-2e-3 relative
  at attention blocks 17+ (bf16 flips), from `pfcheck.py`.
