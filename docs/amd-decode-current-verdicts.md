# AMD Decode Current Verdicts

Date: 2026-06-12

Status: canonical decision state for the AMD decode optimization campaign.

This document consolidates the current verdicts. Treat older hypothesis and
execution-plan sections as historical unless they agree with this file.

## Bottom Line

The local inference win is real and should be considered consolidated unless the
goal explicitly changes to compiler research.

Current stable path:

- `Q4K_PRIMITIVE=1 Q6K_PRIMITIVE=1 JIT=1 DEV=AMD`
- Qwen3-8B-Q4_K_M: about `58 tok/s`, roughly `57%` of the llama.cpp reference.
- Qwen3-14B-Q4_K_M: about `28 tok/s`, roughly `43%` of the llama.cpp reference.
- Correctness is verified at the kernel boundary and by greedy end-to-end A/B.
- BEAM/risky schedule search is guarded and must not run on Mac/TinyGPU paths.

Recommendation by default: keep the explicit Q4/Q6 primitive path, stop adding
`extra/` q8 arithmetic variants, and move effort to the next higher-value goal
unless compiler research is the point.

## Verdict Table

| Area | Verdict | Consequence |
|---|---|---|
| Original fp32-spill thesis | False. Q4_K already fuses into GEMV and does not materialize fp32 weights. | Do not pursue fp16-spill/fusion fixes. |
| Generic BEAM | Not enough for this gap, and unsafe on remote/Mac without guards. | BEAM returns only after there is a semantic primitive/candidate space worth tuning. |
| Expression-vectorization probe | Failed. Rewriting byte expressions did not make codegen emit wider useful loads. | Stop trying to garden `gguf.py` scalar byte math. |
| Q4_K/Q6_K v1 primitive | Accepted. It gives a real end-to-end speedup and passed correctness gates. | Keep as the stable local inference path. |
| Generated policy | Functional but opt-in. It matches wrapper coverage, but is slightly slower/noisier than explicit flags. | Keep `QK_GENERATED_POLICY` as research infrastructure, not default runtime. |
| Ansor-direction harness | Useful. Descriptors, generated candidates, correctness gates, and policy cache exist. | Continue here only if the goal is making tinygrad generate/select packed quant kernels. |
| q8_1 representation | Valid and reachable. | Representation is not the blocker. |
| q8_1 algebra/intdot | Correct and improves over the first q8 path, but still loses to v1. | Algebra is not enough; the lowering quality is the blocker. |
| AMD `v_dot4_u32_u8` | Instruction emission works on gfx1100. | Hardware capability exists. |
| Serial vdot candidate | Correct but rejected. It serializes the K loop per row. | Serial custom-C integration is the wrong shape. |
| Parallel vdot candidate | Correct and scheduled, but still rejected on speed. | `Ops.CUSTOMI` inline asm is not a good enough integration layer. |
| Further q8 `extra/` variants | Stop. | More arithmetic variants repeat a rejected level of abstraction. |
| Next q8 path | Renderer/core lowering for a semantic packed-dot pattern. | Only justified if continuing compiler research. |

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

So the machine-first research hypothesis is:

> Packed quant decode can move further toward tinygrad's search philosophy only
> if Q4_K/Q6_K layouts and packed-dot operations become compiler-visible
> semantic objects, not opaque hand-written kernels or inline-asm statements.

That is an Ansor-style direction. It is a different goal from "make my local
Qwen faster this week."

## Strategic Fork

Choose the next step by goal:

| Goal | Track | Recommended next step |
|---|---|---|
| Reliable local Qwen inference | Consolidate | Keep explicit Q4/Q6 flags, document run commands, and stop decode optimization. |
| More speed on this one GPU | v2 template grind | Write a richer hand template and sweep it, accepting lower ROI. |
| Honor tinygrad's search thesis | Compiler research | Build renderer/core semantic packed-dot lowering and feed it through the generated-search harness. |
| Use the inference win | Training | Validate the smallest real QLoRA/SFT or RLVR stack using the faster decode path for rollouts/eval. |

Default recommendation remains consolidation or training. The compiler path is
worth doing only if the research itself is the goal.

## Stop Rules

- Do not add another q8_1 arithmetic candidate in `extra/` unless it first
  changes the integration layer to renderer/core semantic lowering.
- Do not make `QK_GENERATED_POLICY` default until it matches or beats explicit
  primitive flags in repeated full-decode runs.
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
