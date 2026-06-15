# Batched-forward ceiling probe — the true bottleneck, and where machine search re-enters

Date: 2026-06-15. `/tmp/batchprobe.py`: time the full decode forward at query-length T=1..32 (the
speculative-verification shape), Qwen3-8B Q4_K, double-warmed, median of 5.

## Curve (per-token cost amortizes, then plateaus)
| T | fwd ms | ms/token | mem/call (MB) | effective GB/s |
|---|---|---|---|---|
| 1 | 50.0 | 50.0 | 4887 | 98 |
| 2 | 62.0 | 31.0 | 4928 | 79 |
| 4 | 76.6 | 19.1 | 5018 | 66 |
| 8 | 128.9 | 16.1 | 5229 | 41 |
| 16 | 228.6 | 14.3 | 5682 | 25 |
| 32 | 446.5 | 14.0 | 7109 | 16 |

(mem/call = printed total / 5 reps; it stays ~4.9 GB → the weight read IS amortized across T, incremental
~70 MB/token is KV+activations.)

## The true bottleneck MOVES with parallelism
1. **T=1 — memory-latency-bound.** Single-stream weight read with no parallelism to hide latency. ~98 GB/s
   (≈11% of peak). This is the entire single-stream decode story.
2. **Batching amortizes the weight read** → per-token drops 50→14 ms = **~3.5×** (or ~2.4× vs the cli
   graph-replay baseline of 33 ms/tok). Weights are read ~once for all T tokens.
3. **The ~14 ms/token plateau is COMPUTE/kernel-bound, not a memory floor.** Effective bandwidth COLLAPSES
   (98→16 GB/s, i.e. 2% of peak) as T grows — the forward stops being memory-bound and hits tinygrad's
   UNTUNED batched-GEMM + attention throughput. A memory floor would keep dropping ms/tok; instead it
   flattens at the compute wall.

## Why this is the decisive result for the mission
- **Speculation is viable**: batched verification gives a real ~2.4–3.5× raw ceiling (perfect acceptance);
  with realistic 60–70% acceptance, ~1.5–2.3× — and this is on the DENSE fallback path (reads more than
  Q4_K). A Q4_K GEMM verification (B1b exists) reads ~3.5× fewer weight bytes → extends the memory-bound
  sweet spot to higher T → higher ceiling.
- **The plateau is the machine-search target.** The ~14 ms/tok floor is tinygrad's untuned batched
  GEMM/attention (16 GB/s = clearly not optimized). The batched verification GEMMs are EXACTLY the matmul
  opt-schedule space our loop already drives to 33–98% of peak (N1/N2/L0/L1, 42× live). So:
  **speculation turns decode into a batched-GEMM problem → the validated loop tunes the compute-bound
  plateau → machine-search-for-decode, reached.**
- This reconciles the numbers: B0's "13–26× from batching" was the GEMV kernel ALONE; the FULL forward
  ceiling is ~2.4–3.5× because attention + the dense path + the compute plateau cap it. Honest ceiling,
  not the microbench number.

## Path forward (now grounded, not inferred)
1. **Speculative scaffold**: verify K tokens/step (draft = the "fine-tuning" lever — n-gram/tiny draft/
   Medusa heads). Cheap first version: even a trivial draft measures realized speedup against the ceiling.
2. **Q4_K GEMM verification** (route T>1 through B1b's fused Q4_K GEMM instead of dense fp16) → push the
   memory-bound sweet spot out and lower the per-token floor.
3. **Point the loop at the verification GEMMs** (the compute-bound plateau) → machine search tuning the
   decode-serving kernels. This is the loop's proven home, now applied to decode.

The bottleneck question is answered: single-stream decode is memory-latency-bound; batching fixes that and
exposes a compute-bound plateau that is tinygrad-tunable — which is precisely where machine search lives.
