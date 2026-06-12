# AMD Decode Current Verdicts

Date: 2026-06-12

Status: canonical decision state for the AMD decode optimization campaign.

This document consolidates the current verdicts. Treat older hypothesis and
execution-plan sections as historical unless they agree with this file.

## Bottom Line

The local inference win is real and should be considered consolidated unless the
goal explicitly changes to compiler research.

Current stable paths:

- Qwen3-8B-Q4_K_M: explicit `Q4K_PRIMITIVE=1 Q6K_PRIMITIVE=1` is still
  the boring path, but the reproducible generated-policy pipeline now accepts
  a modest opt-in generated artifact:
  `QK_GENERATED_POLICY=bench/qk-harness-20260612/8b/policy.json`.
  Current harness-matrix result: `53.49 tok/s` generated versus `49.35 tok/s`
  explicit.
- Qwen3-14B-Q4_K_M: use the accepted generated policy
  `QK_GENERATED_POLICY=bench/qk-harness-20260612/14b-rerun/policy.json`.
  Current harness-matrix result: `39.61 tok/s` generated versus `22.76 tok/s`
  explicit, about `60.2%` of the llama.cpp reference.
- Qwen3-32B-Q4_K_M: the uncapped generated policy still OOMs, but a
  tensor-scoped `1536 MB` memory-capped generated policy now fits and is
  accepted against the generic fused baseline:
  `QK_GENERATED_POLICY=bench/qk-policy-cap-20260612/32b-1536mb/policy.json`.
  Stable result: `4.16 tok/s` generated versus `3.44 tok/s` generic baseline,
  `20.98%` gain, `13.5%` of the llama.cpp reference. This is not an
  explicit-full-primitive comparison; full explicit primitive storage remains
  too large for 32B on this card.
- Correctness is verified at the kernel boundary and by greedy end-to-end A/B.
- BEAM/risky schedule search is guarded and must not run on Mac/TinyGPU paths.

Recommendation by default: keep generated policies opt-in and artifact-pinned.
Use `bench/qk-harness-20260612/matrix-summary-rerun.md` as the current
8B/14B/32B source of truth; use the 8B/14B harness artifacts only for the
matching model/hardware path; use the 32B capped artifact only when accepting
the generic-baseline comparison. Stop adding `extra/` q8 arithmetic variants,
and move effort to the next higher-value goal unless compiler research is the
point. Storage accounting and runtime caps are now in place; do not turn this
into another kernel-search loop.

## Verdict Table

| Area | Verdict | Consequence |
|---|---|---|
| Original fp32-spill thesis | False. Q4_K already fuses into GEMV and does not materialize fp32 weights. | Do not pursue fp16-spill/fusion fixes. |
| Generic BEAM | Not enough for this gap, and unsafe on remote/Mac without guards. | BEAM returns only after there is a semantic primitive/candidate space worth tuning. |
| Expression-vectorization probe | Failed. Rewriting byte expressions did not make codegen emit wider useful loads. | Stop trying to garden `gguf.py` scalar byte math. |
| Q4_K/Q6_K v1 primitive | Accepted. It gives a real end-to-end speedup and passed correctness gates. | Keep as the stable local inference path. |
| Generated policy | Model-specific result. The current harness rerun matrix accepts 8B as a modest win (`53.49` vs `49.35 tok/s`) and 14B as a strong win (`39.61` vs `22.76 tok/s`). Both pass 32-token greedy A/B. 32B uncapped policy OOMs, but a tensor-scoped `1536 MB` capped policy accepts versus generic baseline (`4.16` vs `3.44 tok/s`) and passes A/B. | Keep `QK_GENERATED_POLICY` opt-in. Use the 8B/14B artifacts from `bench/qk-harness-20260612/` when running those exact model/hardware paths. Use the 32B capped artifact only with the generic-baseline caveat. Do not make it a global default. |
| QK policy storage | Shape-scoped policy is too coarse for large models; 32B needs tensor-scoped storage decisions. A first memory cap exists and selects `144` primitive tensors under `1.49 GiB` (`64 attn_k`, `64 attn_v`, `16 ffn_down`). Runtime accounting and `QK_PRIMITIVE_MAX_STORAGE_MB` now report/control sidecar bytes. Q4 on-demand storage was tested and rejected as too slow. | Future policy generation must include storage cost, benefit, and fallback decisions. Runtime caps are guardrails, not optimizers. Long-term fix is shared packed storage without per-token copies; otherwise move up to harness work. |
| Ansor-direction harness | Useful. Descriptors, generated candidates, correctness gates, policy cache, manifest-checked pipeline reuse, stage statuses, normalized decisions, and matrix summaries exist. | Continue here only if the goal is making tinygrad generate/select packed quant kernels. Treat storage work as harness-enabling infrastructure, not a 32B/kernel detour. |
| q8_1 representation | Valid and reachable. | Representation is not the blocker. |
| q8_1 algebra/intdot | Correct and improves over the first q8 path, but still loses to v1. | Algebra is not enough; the lowering quality is the blocker. |
| AMD `v_dot4_u32_u8` | Instruction emission works on gfx1100. | Hardware capability exists. |
| Serial vdot candidate | Correct but rejected. It serializes the K loop per row. | Serial custom-C integration is the wrong shape. |
| Parallel vdot candidate | Correct and scheduled, but still rejected on speed. | `Ops.CUSTOMI` inline asm is not a good enough integration layer. |
| v1 roofline premise check | Accepted v1 Q4/Q6 kernels are memory/schedule-bound, not compute-bound. | Do not start isolated packed-dot renderer/core lowering as the next default task. |
| llama.cpp MMVQ comparison | llama.cpp uses q8_1 staging plus packed dot plus RDNA-specific scheduling. | Its advantage is a whole representation/schedule package, not proof that `v_dot4` alone closes the gap. |
| Further q8 `extra/` variants | Stop. | More arithmetic variants repeat a rejected level of abstraction. |
| Next q8 path | Semantic packed-layout plus schedule/codegen generation. | Only justified if continuing compiler research; do not build an isolated `v_dot4` peephole first. |

## Current Hypothesis

The remaining gap is no longer explained by missing fusion or missing hardware.
It is a representation/lowering boundary:

1. tinygrad can already fuse the generic Q4_K dequant expression into GEMV.
2. The v1 primitive proves packed quant GEMV wins when the representation is
   exposed directly.
3. The q8_1 probes prove the activation representation and algebra are valid.
4. The vdot probes prove gfx1100 can emit the relevant packed-dot instruction.
5. The parallel vdot rejection shows that merely inserting inline asm through
   `Ops.CUSTOMI` is not enough; the compiler/searcher needs a semantic packed-dot
   lowering it can optimize around.
6. The v1 roofline check shows the accepted Q4/Q6 kernels are memory/schedule-
   bound. Their logical dot intensity is far below the RX 7900 XTX FP32 ridge,
   and their logical dot throughput is nowhere near peak compute. That makes
   isolated packed-dot lowering a weak next bet.
7. The semantic generated-search pass confirms the useful next abstraction:
   machine-generated schedule/layout policy can matter when it changes coverage
   and split decisions at the model level. It produced a real 14B win while also
   rejecting isolated packed-dot candidates by stop rule.
8. The 14B remeasure audit explains that win: generated policy installs `280`
   wrappers versus `200` explicit, adds Q4 `attn_k`, Q4/Q6 `attn_v` coverage,
   changes FFN split/local choices, and drops batched AMD kernel time from
   `40.84` to `22.95 ms/tok`.
9. The reproducible pipeline adds a stricter repeated-run gate. With an
   adaptive stable-window rerun, 8B is a small generated-policy win, 14B is a
   large generated-policy win, and 32B is blocked by duplicate primitive storage
   pressure rather than by candidate generation.
10. The memory-aware policy cap proves the 32B blocker is architectural, not
    semantic: the same generated search/parity result can be lowered into a
    tensor-scoped policy that fits VRAM by trading coverage for storage. Under a
    `1536 MB` cap, 32B gains `20.98%` over the generic fused baseline while
    preserving greedy output.

So the machine-first research hypothesis is:

> Packed quant decode can move further toward tinygrad's search philosophy only
> if Q4_K/Q6_K layouts, q8_1 staging, packed-dot operations, and RDNA scheduling
> choices become compiler-visible semantic objects, not opaque hand-written
> kernels or inline-asm statements.

That is an Ansor-style direction. It is a different goal from "make my local
Qwen faster this week."

## Strategic Fork

Choose the next step by goal:

| Goal | Track | Recommended next step |
|---|---|---|
| Reliable local Qwen inference | Consolidate | Use the accepted generated-policy artifacts for 8B/14B when you want peak local speed; keep explicit Q4/Q6 flags as the boring fallback. For 32B, use the capped generated artifact only when the generic-baseline comparison is acceptable. |
| More speed on this one GPU | v2 template grind | Write a richer hand template and sweep it, accepting lower ROI. |
| Honor tinygrad's search thesis | Compiler research | Build semantic packed-layout and schedule/codegen generation, then feed it through the generated-search harness. |
| Use the inference win | Training | Validate the smallest real QLoRA/SFT or RLVR stack using the faster decode path for rollouts/eval. |

Default recommendation remains consolidation or training. The compiler path is
worth doing only if the research itself is the goal.

## Stop Rules

- Do not add another q8_1 arithmetic candidate in `extra/` unless it is part of
  a broader semantic layout/schedule/codegen rewrite.
- Do not start isolated renderer/core packed-dot lowering unless a future
  roofline/counter profile overturns the memory/schedule-bound verdict, or the
  packed-dot work is part of a broader semantic layout/schedule rewrite.
- Do not make `QK_GENERATED_POLICY` a global default. The accepted 14B policy is
  model/hardware-specific and must stay an explicit artifact path.
- Do not pursue uncapped 32B generated-policy speed work until primitive-packed
  storage no longer duplicates enough GPU memory to OOM during model load.
- Do not expand 32B caps blindly. Cap selection must report persistent bytes,
  selected tensors, fallback reasons, greedy output A/B, and a stable decode
  window.
- Do not use `QK_PRIMITIVE_STORAGE=q4_ondemand` as a performance path. It is a
  negative storage prototype: lower persistent bytes, unacceptable decode speed.
- Do not run BEAM or risky schedule search on Mac/TinyGPU/remote paths.
- Do not widen tinygrad core optimizer APIs for quant GEMV until an `extra/`
  or renderer-level candidate passes correctness and wins a generated-search
  gate.

## Pointers

- Historical execution plan: `docs/amd-decode-optimization-plan.md`
- Ansor/search direction: `docs/amd-decode-ansor-direction.md`
- Optional v2 template scope: `docs/amd-decode-primitive-v2-design.md`
- Measurement log and detailed verdicts: `docs/amd-rocm-llamacpp-research.md`
- Current generated-search artifacts: `bench/qk-ansor-20260612/README.md`
- Semantic generated-search artifacts: `bench/qk-semantic-20260612/README.md`
- 14B generated-policy audit: `bench/qk-14b-remeasure-20260612/README.md`
- Reproducible generated-policy pipeline: `bench/qk-policy-pipeline-20260612/README.md`
- 32B memory-aware capped policy: `bench/qk-policy-cap-20260612/README.md`
- QK runtime storage-control artifacts: `bench/qk-storage-20260612/README.md`
- QK storage architecture: `docs/amd-decode-qk-storage-architecture.md`
- QK harness architecture: `docs/amd-decode-harness-architecture.md`
- QK harness validation matrix and 14B rerun: `bench/qk-harness-20260612/README.md`
- Vdot premise check: `bench/vdot-premise-20260612/v1-roofline.md`
- llama.cpp MMVQ comparison: `bench/vdot-premise-20260612/llamacpp-mmvq-notes.md`
