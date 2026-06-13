# AMD Decode Optimization: Methodology, Timeline, and Roadmap

Audience: anyone who wants to understand how this campaign has worked, why it
took the turns it did, and where it is going. This is a narrative companion to
the canonical decision state in `amd-decode-current-verdicts.md`; when the two
disagree, the verdicts file wins.

## The Problem

We are trying to make quantized large-language-model **decode** (single-token,
batch-1 autoregressive generation) fast on **consumer AMD (RDNA / gfx1100)**,
inside tinygrad, for GGUF K-quant weights (Q4_K / Q6_K). The yardstick is
`llama.cpp`, whose hand-written, hand-tuned kernels are the practical
state of the art on this hardware.

Where we stand today, all correctness-verified (bit-exact greedy A/B against an
explicit reference):

| Model | generated tok/s | explicit tok/s | gain | % of llama.cpp |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-8B-Q4_K_M  | 52.07 | 50.41 | +3.3%  | ~52% |
| Qwen3-14B-Q4_K_M | 40.55 | 21.77 | +86.3% | ~61.6% |
| Qwen3-32B-Q4_K_M | 17.23 | 11.15 | +54.6% | ~55.9% |

The local win is real and consolidated. The remaining gap to `llama.cpp`
(roughly 38–48%) is the open problem.

## Methodology (the through-line)

The approaches changed many times; the *method* did not. It is the most durable
asset in this work and it is why our negative results are trustworthy:

- **Generated policies, not runtime search.** Optimization choices are emitted
  as explicit policy artifacts (committed JSON), applied deterministically, and
  versioned — not discovered live and thrown away.
- **Reproduce-from-artifact.** Every benchmark matrix regenerates from its
  committed `decision.json` / policy inputs. A result that only reproduces on
  the machine that made it is not evidence; it is local state.
- **Hard correctness gating.** Bit-exact greedy-token parity between the fast
  path and an explicit reference is mandatory. Speed is never accepted without
  identical output.
- **Confirmation-gated acceptance.** No promotion from a single microbench. A
  candidate must survive a confirmation rerun — this is what downgraded an
  apparent 32B win to a tie rather than letting noise through.
- **Negative results are committed, not deleted.** Falsified hypotheses and
  exhausted search frontiers are recorded as findings. The campaign's value is
  as much in what it ruled out as in what it shipped.

## Timeline of Approaches

Each phase below lists the hypothesis, what we did, the result, and the lesson
that pushed us to the next phase.

### 1. The fp32-spill thesis — falsified
- **Hypothesis:** the gap comes from materializing dequantized fp32 weights.
- **Result:** false. tinygrad already fuses Q4_K dequant into the GEMV and never
  materializes fp32 weights.
- **Lesson:** the obvious memory-traffic explanation was wrong; the gap is
  subtler.

### 2. BEAM / generic schedule search — tried and set aside
- **Hypothesis:** tinygrad's built-in autotuner (BEAM) will find the schedule
  that closes the gap.
- **Result:** generic BEAM did not solve it. It is also operationally risky on
  the AMD remote bridge / TinyGPU paths, and its on-device, noisy, ephemeral
  results do not fit the reproduce-from-artifact discipline.
- **Lesson:** guard BEAM off the risky hardware paths, and prefer deterministic,
  pinnable policies. (Deeper lesson, learned later: BEAM searches the
  *schedule-knob space*, which turned out not to contain the answer.)

### 3. Expression-level vectorization in `gguf.py` — rejected
- **Hypothesis:** vectorize the dequant expression to widen loads.
- **Result:** rejected — codegen still emitted scalar byte loads.
- **Lesson:** load width is a *codegen/renderer* property, not something you can
  coax from the expression layer. (Foreshadows the eventual conclusion.)

### 4. Hand-written Q4_K / Q6_K primitives — the stable base
- **Action:** built correctness-gated packed GEMV primitives, verified at the
  kernel boundary and by greedy end-to-end A/B.
- **Result:** a stable, correct fast path to build search and storage work on.

### 5. Generated-policy harness — the current core method
- **Action:** a staged, gated pipeline — descriptor → candidate generation →
  static gate → microbench → confirmation → verdict — emitting pinned policy
  artifacts with reproducibility tests.
- **Result:** accepted an 8B modest win and a **14B strong win (+86%)**. A 32B
  raw accept was correctly downgraded to a tie on confirmation.
- **Lesson:** discipline and reproducibility, dramatically improved over BEAM —
  but the *reach* is the same schedule-knob space.

### 6. Shared primitive storage — a storage-architecture win, not a search win
- **Hypothesis:** 32B failed not from search or correctness but from sidecar
  buffer memory pressure.
- **Action:** added `QK_PRIMITIVE_STORAGE=shared` — typed views over the raw
  GGUF buffer instead of duplicate sidecar buffers. (`q4_ondemand` was tried
  first and rejected as too slow.)
- **Result:** 32B uncapped generated policy now fits (`storage_bytes=0`) and
  passes the full harness with greedy A/B. Validated across 8B/14B/32B.
- **Lesson:** some "performance" problems are architecture problems wearing a
  performance costume.

### 7. Knob-search exhaustion — the negative bound
- **Action:** systematically rejected, with committed evidence, four families of
  schedule-knob search: parts/LOCAL policy search, schedule v0 knobs,
  direct-output Q4 (v1 codegen), and row grouping (Family B).
- **Result:** every family tied within ±3% or regressed.
- **Lesson:** **descriptor / schedule-knob search over this kernel is
  exhausted.** This is itself a result.

## The Inflection: from search to derivation

Phase 7 forced the real question: *why* does schedule search plateau? The
answer is structural, and the literature names it precisely.

At batch-1 decode, weight-only quant GEMV has an arithmetic intensity of about
**2 ops/byte** — it is firmly **memory-bandwidth-bound**
([DecDEC, OSDI'25][decdec]; [GEMV-from-scratch][gemv]). The roofline ceiling is
fixed at `weight_bytes / memory_bandwidth`. Schedule knobs reshuffle *compute*;
they cannot move a kernel whose bottleneck is *bytes moved and load
efficiency*. The plateau was predicted, not unlucky.

The gap is therefore **load/bandwidth efficiency**, and that number is
measurable. On AMD RDNA, a naive 4-bit decode kernel reaches ~49% of peak
bandwidth while an optimized path reaches ~91% on the same silicon
([bitsandbytes #1842][bnb]); batch-1 decode generally lands at
[50–70% of peak][intel-gpu]. We sit at ~52–62% of `llama.cpp` — the same shape.

We tested whether wide loads alone close it (a three-way microbench: v1 vs a
schedulable wide-load path vs an opaque wide-load kernel). With matched
parallelism, the wide-load path came in at **−8.58% vs v1**: wide loads are
necessary plumbing but **not sufficient** — the remaining cost is in the inner
dequant/dot work, not the load width.

The conclusion the field reached, and that we reached independently by
exhausting the search: **you do not search your way to peak low-bit GEMV — you
make the hardware-shaped operation first-class in the compiler and derive it.**
Three reference systems, each escaping the same wall by promoting the
*implicit* thing into a *compiler-controlled* object:

- **Halide** decouples algorithm from schedule and uses a *learned cost model*
  to search without paying for every on-device run ([CACM][halide];
  [2019 autoscheduler][halide-as]). Lesson: the cost model, and "schedule as a
  first-class object."
- **Exo** lets you *declare* a hardware instruction in a user library and
  `replace` matching code with it via *verified, equivalence-preserving
  rewrites* ([PLDI'22][exo]; [design doc][exo-design]). Lesson: declare the
  intrinsic; don't hope search finds it. This maps directly onto tinygrad's
  UOp rewrite engine.
- **Ladder / BitBLAS** make the **low-bit dtype and the hardware's load/compute
  granularity first-class tiles** (`tType` / `tTile`) and *derive* the layout
  transform that matches the hardware's required load width — 8× over cuBLAS on
  INT2 GEMV ([OSDI'24][ladder]; [BitBLAS][bitblas]). Lesson: control the data
  layout so loads hit peak width *by construction*. This is the most direct
  answer to our measured bandwidth gap.

Background that shaped the framing: auto-scheduling is [weak on memory-bound
kernels by measurement][autotune] (1.04–1.30× vs the large compute-bound wins);
the broader low-bit-GEMV space ([Marlin][marlin], [Machete][machete],
[GemLite][gemlite], llama.cpp MMVQ / [wide-K WMMA on RDNA][wmma]) is crowded but
almost entirely CUDA/TensorCore — the RDNA + tinygrad-IR intersection is open.

## Where We Are Now

Two tracks run in parallel:

- **Practical track (use the speed):** the verified generated-policy speedup
  (token-exact with the explicit reference, faster) now feeds a real loop —
  rollouts → SFT-style rows → a tinygrad training/eval gate → trained artifact →
  contract validation. This banks value independent of the research question.
- **Research track (close the gap):** moving from *searching* schedules to
  *deriving* a first-class, compiler-visible packed-QK GEMV operation.

## Where We Are Going

The research direction, concretely:

1. **Measure first.** Establish achieved % of peak memory bandwidth (via
   counters, a near-deterministic metric) for our kernel vs `llama.cpp` on the
   same gfx1100. This bounds the entire opportunity and replaces noisy tok/s as
   the fitness signal.
2. **Make packed-QK GEMV a first-class compiler op** (Ladder direction): a
   hardware-derived weight layout so loads are wide and coalesced *by
   construction*, lowered through tinygrad's renderer rather than a hand kernel.
3. **Declare the intrinsics as verified rewrites** (Exo direction): express
   wide-load / dot primitives as UOp `PatternMatcher` rules, with the existing
   bit-exact A/B as the equivalence check, so the inner dequant/dot work — where
   the −8.58% result says the real cost lives — becomes compiler-visible.
4. **Fix the whole pipeline, not just the load:** load granularity + dequant
   staging + dot/reduction matched to hardware tiles as one transform.

### Open question that gates the research track

The "first-class low-bit compiler op" idea is real and correct — and **already
exists on CUDA** (Ladder/BitBLAS). A defensible contribution is therefore narrow
and specific: *this pattern on consumer RDNA, inside tinygrad's UOp IR, with
reproducible policy artifacts.* Whether that intersection is novel enough to be
a contribution (versus a useful personal reimplementation) is **not yet
checked**. That novelty check is the cheapest decision-relevant action available
and it determines whether the research track is worth its cost or whether the
practical track is the rational focus. It should be run before significant
research-track investment.

## References

These influenced the path and are cited where they did.

- [decdec]: DecDEC — A Systems Approach to Advancing Low-Bit LLM Quantization, OSDI'25. <https://www.usenix.org/system/files/osdi25-park-yeonhong.pdf>
- [gemv]: A Quantized GEMV Kernel from Scratch (2 ops/byte for Q4_K). <https://vijayprabhas9.github.io/gemv_optimization/>
- [bnb]: bitsandbytes #1842 — 4-bit decode at 49% vs 91% of bandwidth on RDNA. <https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1842>
- [intel-gpu]: Pushing the Envelope of LLM Inference on AI-PC and Intel GPUs. <https://arxiv.org/html/2508.06753v2>
- [halide]: Halide — Decoupling Algorithms from Schedules, CACM 2018. <https://andrew.adams.pub/halide_cacm.pdf>
- [halide-as]: Learning to Optimize Halide with Tree Search and Random Programs, 2019. <https://halide-lang.org/papers/halide_autoscheduler_2019.pdf>
- [exo]: Exocompilation for Productive Programming of Hardware Accelerators, PLDI'22. <https://dl.acm.org/doi/10.1145/3519939.3523446>
- [exo-design]: Exo design doc. <https://github.com/exo-lang/exo/blob/main/docs/Design.md>
- [ladder]: Ladder — Hardware-aware Tensor Transformation, OSDI'24. <https://www.usenix.org/conference/osdi24/presentation/wang-lei>
- [bitblas]: BitBLAS. <https://github.com/microsoft/BitBLAS>
- [autotune]: Performance-Portable Autotuning of OpenCL Kernels. <https://www.netlib.org/utk/people/JackDongarra/PAPERS/performance-portable-autotuning.pdf>
- [marlin]: MARLIN — Mixed-Precision Auto-Regressive Parallel Inference. <https://arxiv.org/pdf/2408.11743>
- [machete]: Machete — mixed-input GEMM for Hopper. <https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel>
- [gemlite]: GemLite — Triton low-bit GEMV/GEMM kernels. <https://github.com/mobiusml/gemlite>
- [wmma]: WMMA guide for AMD RDNA — wide-K vector loads. <https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-2/>
- [ansor]: Ansor — Generating High-Performance Tensor Programs, OSDI'20. <https://www.usenix.org/system/files/osdi20-zheng.pdf>
