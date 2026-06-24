# Prefill AMD GEMM — GPU-Time Thoroughly Verified: ours is ~10% faster than Tensile

Date: 2026-06-20

## Question

Did we *thoroughly* test "our kernel is GPU-faster than Tensile," or just the batch-isolate (which had a
possible same-buffer WAW-serialization confound)?

## Answer: yes — two independent rigorous methods agree, each reproduced 3×

### Method 1 — gold standard: pure GPU time via HCQ `wait=True` signal timestamps

Both kernels as **raw `AMDProgram`**, each a **single isolated dispatch** timed by on-chip start/end signals.
Removes ALL confounds: no host launch overhead (times only the GPU kernel), no batching, no WAW (one dispatch).
`extra/qk_amd_gemm_gputime_goldstandard.py`, reproduced 3×, **identical median ratio every run:**

| | median TFLOPS | best | GPU time (median) |
|---|---:|---:|---:|
| **ours** | **78.6** | 81.5 | 656 µs |
| Tensile `.co` | 70.9 | 74.4 | 728 µs |
| **ratio** | **1.109** | ~1.09 | — |

Correct: rel RMSE 2.08e-4. Median ratio **1.109 / 1.109 / 1.109** across 3 runs — extremely stable.

### Method 2 — batch-isolate: wall-clock with host overhead amortized

Launch K times back-to-back (K=1,8,32); per-launch time = total/K converges to GPU throughput as K grows.
`extra/qk_amd_gemm_batch_isolate.py`, reproduced 3×: ratio **0.97 (K=1) → 1.09 (K=32)**, both kernels saturate
by K=8.

### The two methods agree

| method | ours/Tensile (GPU) |
|---|---:|
| gold standard (`wait=True`, isolated, GPU-timestamped) | **1.11** |
| batch-isolate (wall, host-amortized) | **1.09** |

Independent mechanisms, same answer. The batch test's potential WAW confound is **ruled out** — the
gold-standard (single isolated dispatch) gives the same ~10%.

## Conclusion (thoroughly verified)

**Our dependency-free kernel executes the GEMM ~10% faster than the vendored Tensile `.co` at the GPU level**
(78.6 vs 70.9 TFLOPS median GPU time), correct, on a shape Tensile never tuned. Verified by two independent
methods, each reproduced 3×, with the gold-standard giving an *identical* 1.109 ratio every run.

The earlier "ours is ~4% behind" (batch=1 / alone-pinned) was **host launch overhead** of the slow
`run_linear` Python path — it vanishes at the GPU level. In real tinygrad inference (JIT-launched, pipelined),
the GPU-throughput number is the representative one.

## What "thorough" means here

| confound | controlled by |
|---|---|
| host launch overhead | `wait=True` times only GPU; batch amortizes it (both agree) |
| WAW serialization (same output buffer) | gold standard = single isolated dispatch (no batch) |
| launch-path asymmetry (run_linear vs tprg) | gold standard = both raw AMDProgram, same `wait=True` path |
| clock | pinned high; ratio is same-session |
| correctness | rel RMSE 2.08e-4 verified each run |
| run-to-run noise | reproduced 3× per method; gold-standard median ratio identical |

## Verdict

`OURS_GPU_FASTER`, thoroughly verified. Dependency-free RDNA3 prefill GEMM is **~10% faster than the vendored
Tensile `.co` at the GPU execution level** — confirmed by pure GPU-timestamp measurement and host-amortized
batching, both reproduced, correct. The win is real and banked.
