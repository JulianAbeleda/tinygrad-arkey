# AMD Decode Current Verdicts

Date: 2026-06-12

Status: canonical decision state for the AMD decode optimization campaign.

This document consolidates the current verdicts. Treat older hypothesis and
execution-plan sections as historical unless they agree with this file.

## Bottom Line

The local inference win is real and should be considered consolidated unless the
goal explicitly changes to compiler research.

Current stable paths:

- Qwen3-8B-Q4_K_M: generated policy is accepted under both sidecar and shared
  storage. Current shared-storage matrix result with
  `QK_PRIMITIVE_STORAGE=shared` and
  `QK_GENERATED_POLICY=bench/qk-shared-storage-20260612/8b/policy.json` gives
  `52.07 tok/s` generated versus `50.41 tok/s` explicit. The older sidecar row
  remains the 8B peak artifact at `53.49 tok/s`.
- Qwen3-14B-Q4_K_M: use the accepted generated policy. Current shared-storage
  matrix result with `QK_PRIMITIVE_STORAGE=shared` and
  `QK_GENERATED_POLICY=bench/qk-shared-storage-20260612/14b/policy.json` gives
  `40.55 tok/s` generated versus `21.77 tok/s` explicit, about `61.6%` of the
  llama.cpp reference. This is slightly above the older sidecar row
  (`39.61 tok/s`).
- Qwen3-32B-Q4_K_M: shared primitive storage now makes the uncapped generated
  policy fit and pass the full harness against an explicit primitive reference:
  `QK_PRIMITIVE_STORAGE=shared` with
  `QK_GENERATED_POLICY=bench/qk-shared-storage-20260612/32b/policy.json`.
  Current shared-storage result: `17.23 tok/s` generated versus `11.15 tok/s`
  explicit, `54.56%` gain, `55.9%` of the llama.cpp reference, with greedy A/B
  passing and `storage_bytes=0`.
- Correctness is verified at the kernel boundary and by greedy end-to-end A/B.
- BEAM/risky schedule search is guarded and must not run on Mac/TinyGPU paths.

Recommendation by default: keep generated policies opt-in and artifact-pinned.
Use `bench/qk-shared-storage-20260612/matrix-summary.md` as the current
8B/14B/32B source of truth. Shared storage is validated across all three rows
and is the recommended generated-policy storage mode when memory behavior or
cross-model consistency matters. Keep it as an explicit runtime mode for now;
sidecar remains available, and is still slightly faster on the 8B peak artifact.
Stop adding `extra/` q8 arithmetic variants, and move effort to the next
higher-value goal unless compiler research is the point. Storage accounting,
runtime caps, and shared storage are now in place; do not turn this into another
kernel-search loop.

## Verdict Table

| Area | Verdict | Consequence |
|---|---|---|
| Original fp32-spill thesis | False. Q4_K already fuses into GEMV and does not materialize fp32 weights. | Do not pursue fp16-spill/fusion fixes. |
| Generic BEAM | Not enough for this gap, and unsafe on remote/Mac without guards. | BEAM returns only after there is a semantic primitive/candidate space worth tuning. |
| Expression-vectorization probe | Failed. Rewriting byte expressions did not make codegen emit wider useful loads. | Stop trying to garden `gguf.py` scalar byte math. |
| Q4_K/Q6_K v1 primitive | Accepted. It gives a real end-to-end speedup and passed correctness gates. | Keep as the stable local inference path. |
| Generated policy | Model-specific result. The current shared-storage matrix accepts 8B as a modest win (`52.07` vs `50.41 tok/s`), 14B as a strong win (`40.55` vs `21.77 tok/s`), and 32B as a strong win (`17.23` vs `11.15 tok/s`). All pass 32-token greedy A/B. The older 8B sidecar row is still slightly faster (`53.49 tok/s`), and the older 32B capped result remains historical evidence that tensor-scoped fallback can fit under sidecar storage pressure. | Keep `QK_GENERATED_POLICY` opt-in and artifact-pinned. Prefer `QK_PRIMITIVE_STORAGE=shared` for generated-policy runs when memory behavior or cross-model consistency matters. Use sidecar only when chasing the exact 8B peak artifact. Do not make generated policies global defaults. |
| QK policy storage | Shape-scoped policy is too coarse for large models under sidecar storage; 32B needs either tensor-scoped storage decisions or shared source storage. Runtime accounting and `QK_PRIMITIVE_MAX_STORAGE_MB` now report/control sidecar bytes. Q4 on-demand storage was tested and rejected as too slow. `QK_PRIMITIVE_STORAGE=shared` references the already-realized raw GGUF buffer through typed views; it has now passed full 8B, 14B, and 32B harnesses with `storage_bytes=0`. | Treat shared storage as the validated generated-policy storage mode, but keep it explicit until it has more runtime soak. Future policy generation should still include storage cost, benefit, and fallback decisions because sidecar remains supported and useful as a performance control. |
| Ansor-direction harness | Useful. Descriptors, generated candidates, correctness gates, policy cache, manifest-checked pipeline reuse, stage statuses, normalized decisions, and matrix summaries exist. | Continue here only if the goal is making tinygrad generate/select packed quant kernels. Treat storage work as harness-enabling infrastructure, not a 32B/kernel detour. |
| Ansor-transition scorecard/descriptors | Started. `bench/qk-ansor-transition-20260612/` freezes the llama.cpp-comparable objective, records the current gap profile, and converts accepted generated policies into machine-readable Q4_K/Q6_K semantic descriptors. | Use this as the next research foundation: reproduce current policies from descriptors, then generate candidate policies/schedules from descriptor data. |
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
   large generated-policy win, and 32B was blocked by duplicate primitive
   storage pressure rather than by candidate generation.
10. The memory-aware policy cap proved the 32B blocker was architectural, not
    semantic: the same generated search/parity result could be lowered into a
    tensor-scoped policy that fit VRAM by trading coverage for storage.
11. Shared primitive storage removes that duplicate-sidecar blocker for the
    current 32B run. The full uncapped generated policy now fits as typed views
    over the already-realized GGUF source buffer, passes greedy A/B, and beats
    the shared explicit primitive reference by `54.56%`.
12. Full 8B/14B shared-storage promotion checks now pass too. 8B shared is
    accepted but slightly below the sidecar peak (`52.07` vs `53.49 tok/s`);
    14B shared is accepted and slightly above sidecar (`40.55` vs
    `39.61 tok/s`). This supports recommending shared storage for generated
    policies without changing the runtime default.

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
| Reliable local Qwen inference | Consolidate | Use accepted generated-policy artifacts when you want peak local speed; keep explicit Q4/Q6 flags as the boring fallback. For uniform generated-policy runs, set `QK_PRIMITIVE_STORAGE=shared`. For exact 8B peak, the older sidecar artifact remains slightly faster. |
| More speed on this one GPU | v2 template grind | Write a richer hand template and sweep it, accepting lower ROI. |
| Honor tinygrad's search thesis | Compiler research | Build semantic packed-layout and schedule/codegen generation, then feed it through the generated-search harness. |
| Use the inference win | Training | Validate the smallest real QLoRA/SFT or RLVR stack using the faster decode path for rollouts/eval. |

For the llama.cpp-comparable research track, use
`bench/qk-ansor-transition-20260612/scorecard.md` as the objective report.
Current generated shared-storage rows are `51.46%` (8B), `61.63%` (14B), and
`55.94%` (32B) of llama.cpp. The first comparable-speed target is `>=70%` on all
three rows; all current rows are below it, so further work needs a real QK
schedule/codegen improvement rather than more rollout/eval plumbing.

Default recommendation remains consolidation or training. The compiler path is
worth doing only if the research itself is the goal.

## Stop Rules

- Do not add another q8_1 arithmetic candidate in `extra/` unless it is part of
  a broader semantic layout/schedule/codegen rewrite.
- Do not start isolated renderer/core packed-dot lowering unless a future
  roofline/counter profile overturns the memory/schedule-bound verdict, or the
  packed-dot work is part of a broader semantic layout/schedule rewrite.
- Do not make `QK_GENERATED_POLICY` a global default. Generated policies are
  model/hardware-specific and must stay explicit artifact paths.
- Do not pursue more 32B generated-policy speed work by hand. Shared storage
  removed the OOM blocker for the current uncapped policy; further 32B claims
  should go through the harness matrix, not bespoke tuning.
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
- QK shared storage artifacts and current 8B/14B/32B matrix:
  `bench/qk-shared-storage-20260612/README.md`
- QK storage architecture: `docs/amd-decode-qk-storage-architecture.md`
- QK harness architecture: `docs/amd-decode-harness-architecture.md`
- Ansor-transition scorecard/gap/descriptors:
  `bench/qk-ansor-transition-20260612/README.md`
- QK harness validation matrix and 14B rerun: `bench/qk-harness-20260612/README.md`
- Vdot premise check: `bench/vdot-premise-20260612/v1-roofline.md`
- llama.cpp MMVQ comparison: `bench/vdot-premise-20260612/llamacpp-mmvq-notes.md`
