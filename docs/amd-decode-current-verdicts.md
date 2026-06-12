# AMD Decode Current Verdicts

Date: 2026-06-12

Status: canonical decision state for the AMD decode optimization campaign.

This document consolidates the current verdicts. Treat older hypothesis and
execution-plan sections as historical unless they agree with this file.

## Bottom Line

The local inference win is real and should be considered consolidated unless the
goal explicitly changes to compiler research.

Current stable paths:

- Qwen3-8B-Q4_K_M: use explicit `Q4K_PRIMITIVE=1 Q6K_PRIMITIVE=1`.
  Same-commit rerun: `51.36 tok/s`; prior stable run: about `58 tok/s`.
- Qwen3-14B-Q4_K_M: use the accepted generated policy
  `QK_GENERATED_POLICY=bench/qk-semantic-20260612/14b-full-level2-skip-stopped-policy.json`.
  Remeasure audit: `39.68 tok/s` mean across three fresh runs (`39.42-40.05`
  range), about `60%` of the llama.cpp reference.
- Correctness is verified at the kernel boundary and by greedy end-to-end A/B.
- BEAM/risky schedule search is guarded and must not run on Mac/TinyGPU paths.

Recommendation by default: keep explicit Q4/Q6 flags for 8B, use the generated
14B policy only for the matching model/hardware artifact, stop adding `extra/`
q8 arithmetic variants, and move effort to the next higher-value goal unless
compiler research is the point.

## Verdict Table

| Area | Verdict | Consequence |
|---|---|---|
| Original fp32-spill thesis | False. Q4_K already fuses into GEMV and does not materialize fp32 weights. | Do not pursue fp16-spill/fusion fixes. |
| Generic BEAM | Not enough for this gap, and unsafe on remote/Mac without guards. | BEAM returns only after there is a semantic primitive/candidate space worth tuning. |
| Expression-vectorization probe | Failed. Rewriting byte expressions did not make codegen emit wider useful loads. | Stop trying to garden `gguf.py` scalar byte math. |
| Q4_K/Q6_K v1 primitive | Accepted. It gives a real end-to-end speedup and passed correctness gates. | Keep as the stable local inference path. |
| Generated policy | Model-specific result. Full-shape stop-gated generation is flat on 8B (`50.94` vs `51.36 tok/s`) and accepted on 14B after remeasure audit (`39.68 tok/s` mean vs current explicit `23.27`; prior `c3315d6ad` explicit also only `22.78`). Both generated policies pass 32-token greedy A/B. | Keep `QK_GENERATED_POLICY` opt-in. Use the 14B artifact when running that exact model/hardware path; do not make it a global default. |
| Ansor-direction harness | Useful. Descriptors, generated candidates, correctness gates, and policy cache exist. | Continue here only if the goal is making tinygrad generate/select packed quant kernels. |
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
| Reliable local Qwen inference | Consolidate | Use explicit Q4/Q6 flags for 8B; use the accepted generated-policy artifact for 14B; document commands and stop decode optimization. |
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
- Vdot premise check: `bench/vdot-premise-20260612/v1-roofline.md`
- llama.cpp MMVQ comparison: `bench/vdot-premise-20260612/llamacpp-mmvq-notes.md`
