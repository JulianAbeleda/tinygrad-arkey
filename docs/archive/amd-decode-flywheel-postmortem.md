# AMD Decode Flywheel — Postmortem

Date: 2026-06-14

A candid account of a multi-phase investigation that started by asking "can a learned
model help optimize AMD decode kernels?", found that the original framing was a dead end
for reasons that took a while to surface, and — by disciplined investigation of *why* —
ended with a real, correctly-measured optimization target. The value of this writeup is
less the result than the failure modes it documents.

## TL;DR

- **The original hypothesis was wrong, and worse, we couldn't tell for a long time** because
  we were optimizing a noise-dominated metric (wall-clock throughput dominated by ~0.27 ms
  launch overhead).
- The learned-model "wins" (3F–4.x triage line) were a combination of a **metric artifact**
  and **wall-clock noise**; a trivial deterministic rule matched the model on every honest
  re-test.
- Re-basing the metric to **device bandwidth vs measured peak** revealed the real picture:
  the batch-1 Q4_K decode GEMV runs at only **~20–47% of peak**, the bottleneck is the
  **dequant compute + occupancy** (not bandwidth or load width), and the one genuine win in
  the whole program is `packed_load` at **+6%** (correctly measured).
- Reducing to primitives showed batch-1 GEMV is **latency-bound with zero weight reuse**, and
  the structural lever is **batching**: B0 measured a **13–26× per-token** amortization, but
  the current fused path leaves most of it on the table, motivating a fused Q4_K GEMM (B1).
- The flywheel's learned-model question is **deferred, not revived**: it only becomes
  meaningful again at B1, the first large, correctly-measured search space in the program.

## The original hypothesis

The "flywheel": a model, learning from accumulated kernel-experiment outcomes, can triage
which experiments to run (and later propose candidates) better than cheap deterministic
methods — a self-improving loop where each run's outcome trains a better selector.

Every clause turned out to be load-bearing in the wrong way.

## The arc (what happened, phase by phase)

**Phase 3F–4.3 — the triage line.** Built a leak-free cost model over kernel-candidate
features and a family-split holdout. XGBoost beat the `mechanism_prior` baseline on the
holdout (macro-F1 0.87 vs 0.48), and in shadow tests appeared to "beat the prior" again.
Each apparent win dissolved under scrutiny:
- The 4.1/4.2 "model beats prior" margins were a **floor-collapse artifact** of the safe-skip
  metric — it penalized a discrete gate for tying a surprise winner with the dead mass, which
  is not a real cost. A fair **deterministic class-skip** gate matched the model exactly (48
  vs 48 experiments saved at 100% recall) in 4.3.
- So at the current feature set, the learned model added **no value** over a cheap
  deterministic rule. (The model also couldn't observe the weight-determined, per-tensor
  variation that drives intra-family outcomes — there was no signal to learn.)

**G0 — generation headroom probe.** Pivoted to "can the model propose better kernels?",
starting with a deterministic headroom probe. It accidentally exposed the deeper problem:
the **metric was unreliable**. The 4.x outcomes were scored on wall-clock `q4_eff` (~28–35
GB/s, dominated by launch overhead); on device timing the same "winning" schedules were
*slower*.

**Phase M — re-base the metric and locate the bottleneck.**
- Measured this GPU's achievable peak: **859 GB/s** (warm streaming copy, 89% of the 960
  datasheet).
- On the device metric `v1_partial` sits at **~20% of peak on attn_q** (~5× gap) and **~47%
  on ffn_gate** (~2×). Real, shape-dependent headroom — it is *not* bandwidth-saturated.
- Re-audited the 4.x "wins" on device: **0 of 7** beat `v1_partial` by >2% (median −38.6%).
  The entire triage-line "win" signal was wall-clock noise. Root cause: the bench captured
  `q4_eff` (wall), not `device_q4_eff`.
- Profiled the bottleneck: loads are already wide `b128`; the kernel body is **dominated by
  Q4_K dequant ALU** (~3862 vector ops/kernel). Both bandwidth and ALU are under-utilized →
  **latency/occupancy-bound on the dequant dependency chain**, worst on small matrices.

**G0′ / G0″ — kernel attempts.**
- A device-metric sweep of all kernel modes found one real win: **`packed_load` (+6% attn_q,
  +2% ffn_gate)** — and it is the 3G `packed_word_lane_unroll` mechanism, so 3G found
  something real while the 4.x schedule work was noise.
- The most-grounded new kernel (`hoist_scale_min`: decode scale/min once, factor it out of
  the reduce) was **correct but regressed −80%** — the full-unroll restructuring *bloated* the
  kernel (more ALU, not less) and serialized the reduce. Lesson: the bottleneck is
  occupancy/latency, **not** decode op-count; ALU reduction is the wrong lever.

**Primitive analysis.** Reduced to three primitives — load, dequant, reduce — the picture is
clear: a **batch-1 GEMV has zero weight reuse** (each dequantized weight used once), the
dequant sits **on the reduction's critical path**, and batch-1 offers too little parallelism
to hide that latency. The hardware idles; neither memory nor compute saturates. No amount of
shuffling work between the load and ALU sub-primitives fixes a latency/occupancy ceiling.

**Phase B0 — the batching lever.** The structural fix is reuse via batching. Sweeping
`B ∈ {1..128}` (measured fp16 compute peak 83.6 TFLOPS): per-token device latency drops
**26× on attn_q** (622→24 µs/tok) and **13× on ffn_gate** (354→26) — the dequant amortizes
exactly as predicted. But the fused quantized path stays at only **17–25% of the dense-fp16
ceiling** at B=128, and even dense matmul reaches only 10–19% of compute peak (untuned tiling).
The fused path is *already* a single kernel reading compressed weights (no fp16 round-trip,
`mem=9.57MB`); it is simply **poorly tiled** (~4% of peak). Real lever; realizing it needs
**B1, a well-tiled fused Q4_K GEMM**.

## What was wrong with the hypothesis (ranked)

1. **Measurement validity.** We optimized a wall-clock metric dominated by host launch
   overhead, so apparent "wins" were noise and we couldn't see ground truth. This invalidated
   every downstream claim until Phase M fixed it. *(Most fatal — and most fixable.)*
2. **Wrong search space / wrong lever.** Even with a good metric, the early work tuned ILP
   knobs (UPCAST/UNROLL) and learned-triage on a kernel whose binding constraint was elsewhere
   (occupancy/latency, then reuse). We polished the dequant because it dominates the profile,
   but the dequant was never the true limiter.
3. **Learnability.** Where outcomes vary, the signal was unobservable (weight-determined
   per-tensor noise) or already captured by a cheap deterministic rule — a lookup equals the
   model. There was nothing for the model to learn that a prior didn't already encode.

The meta-error: we applied a **statistical-learning frame to a deterministic, physics-bound,
measurement-sensitive systems problem**. For a memory-bound, no-reuse, batch-1 GEMV, the
answer is set by the roofline and the reuse structure, not by a learned selector.

## Methodology lessons (the part that generalizes)

- **Validate the metric before optimizing anything.** Wall-clock vs device timing is the
  difference between optimizing noise and optimizing the kernel. For small/fast kernels, host
  launch overhead dominates wall time; use device counters, warm up, fix clocks.
- **Measure against the roofline.** "% of measured peak" is what tells you whether headroom
  even exists. 20% of peak says *keep going*; 95% says *stop*. We almost declared "no headroom"
  without checking — it was actually ~5×.
- **Adversarially verify your own wins.** Every positive in this project that we scrutinized
  turned out smaller or unreal: the floor-collapse artifact, the noise re-audit (0/7), the B=4
  outlier. The wins that survived (packed_load, the batching curve) survived *because* we tried
  to break them.
- **Pre-register failure modes.** Deciding in advance what "the lookup wins" or "no headroom"
  would mean kept us from rationalizing flattering results after the fact.
- **Freeze predictions before outcomes.** The shadow tests committed predictions to git before
  the GPU run, so "the model knew" was never an option — the integrity was provable, not
  asserted.
- **Reduce to primitives when stuck.** The batching lever only became obvious after stripping
  the problem to load / dequant / reduce and asking which primitive actually binds.

## Current state — what is real

- **`packed_load` +6%** on attn_q (+2% ffn_gate), correctly measured on device, is the
  genuine batch-1 kernel win and the adopted batch-1 baseline.
- **A fused Q4_K GEMM (B1b) beats the fp16 dense matmul at small batch** — `1.8–5×` at
  `B≤8`, correctness-gated, reading compressed weights — the right kernel for the
  speculative/Medusa-decode regime and the first hand-authored kernel to beat a real
  baseline. It plateaus at `~5%` of compute peak and loses at `B≥16` (tinygrad's matmul
  tiles fp16 better); a register-blocked GEMM would be needed there.
- **Batching is a 13–26× per-token lever** for the regimes that have it (prefill, batched
  serving, speculative/Medusa decode) — *not* single-stream greedy decode, which is
  irreducibly batch-1.
- **B1 (fused Q4_K GEMM)** is the next build, with a quantified target: close the ~4–6×
  fused-vs-dense gap, then push the dense gap toward the 83.6 TFLOPS roof.
- The **flywheel/learned-model question is deferred to B1's tiling space** — the first large,
  correctly-measured search space where a model could honestly earn its keep.

## Honest bottom line

The flywheel's original triage premise was a dead end: at the current feature set a cheap
deterministic rule matches the model, and the metric we were optimizing was noise. But the
discipline of asking *why it didn't work* — re-basing the metric, locating the bottleneck,
reducing to primitives — converted a negative into a real, correctly-measured optimization
target (batching / fused GEMM) with quantified headroom. That conversion, not the original
hypothesis, is the result.

## References

- Roofline model — Williams, Waterman, Patterson, CACM 2009.
- Memory-bound decode GEMV and dequant amortization over batch — *A Systems Approach to
  Advancing Low-Bit LLM Quantization*, OSDI '25; *Fast 2-bit LLM Inference with Asynchronous
  Dequantization*, arXiv:2311.16442.
- Learned cost models for autotuning — Ansor (OSDI '20), TenSet (NeurIPS '21); KernelBench
  (arXiv:2502.10517) on the difficulty of model-generated kernels.
- Full phase-by-phase detail and artifacts: `docs/amd-decode-flywheel-proof-plan.md`,
  `bench/amd-decode-flywheel-proof-20260614/`.

## Addendum (2026-06-15) — the fused-WMMA line resolved (W1b' / W2)

After re-basing the metric, the frontier moved to: can a FUSED dequant->WMMA kernel be both
memory-light (reads compressed) and fast (tensor cores)? Resolved, end to end:

- **W1b' — the fused primitive WORKS.** A hand-authored custom kernel that dequants the Q4_K weight
  tile ONCE into LDS (`DEFINE_LOCAL`), barriers, then runs a matmul reduce the TC opt turns into WMMA
  reading the staged tile. Correct, reads compressed, uses tensor cores, and fusing is ~FREE vs the
  same-structure fp16 ceiling (mean 1.04x). The W1 28x recompute is structurally gone (dequant is
  pre-barrier, WMMA post-barrier).
- **W2 — but it is NOT competitive, and the wall is fundamental.** Grid parallelism (~70x) and
  split-K K-tiling (real K=4096, correct) still cap the fused custom kernel at ~3-6% of peak, while
  NATIVE tinygrad fp16 matmul reaches 33-98%. The fused kernel is ~10x slower than native — and 5-6x
  slower even at small-N memory-bound decode where reading 3.5x less weight data should win. Root
  cause is not the dequant (the manually-staged fp16 ceiling also caps ~3-8%): a custom kernel that
  manually stages LDS applies only the TC opt, whereas native matmul applies TC+UPCAST*2+LOCAL to
  reach 98%; appending those exact opts barely helps (3.0->3.7%). **The manual LDS staging that makes
  fusion free is exactly what blocks the auto-tiling that reaches peak — in tinygrad's custom_kernel
  + opt model you get fusion OR peak tiling, not both.**

Consequence for the original thesis: "machine search competitive with llama.cpp" via a fused
custom-kernel template is not achievable in tinygrad — the template cannot contain a competitive
point, so the W3/W4 search over it is moot. The honest competitive paths are (c) hand-assembly
(Marlin/rocWMMA, full control of tiling AND fusion) or matmul_decoded (a cheap separate dequant pass
+ native matmul at 33-98%, accepting the fp16 round-trip). The learned-cost-model question, if
revived, lives on the NATIVE matmul opt schedule (already driven near peak by heuristic/BEAM), not on
the fused kernel. This is a clean, well-grounded negative that precisely locates the framework limit.
