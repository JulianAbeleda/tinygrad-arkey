# Scope - prefill fp16-load vectorization (the dependency-free renderer lever), + CG-W2b gating test

The last open dependency-free prefill lever. CG-W1.5 pinned the in-model warmstarted WMMA matmul's hot-loop
bottleneck to **127 `v_mov` + 16 per-element `global_load_d16` (16-bit) fp16 loads** (not addressing, not conversion);
CG-W2 refuted the kernel copy-pattern fix (contiguous-per-thread still emitted d16 + 123 `v_mov`, and was slower).
The remaining lever: **get the global→LDS copy to use WIDE loads (b128) instead of per-element d16** so the
register-init `v_mov` vanish and issue density rises toward Tensile's 66 TFLOPS.

## Grounding (renderer investigation)
- cstyle.py HAS vectorized load rendering (`float4`/`dtype.vec()` → wide load); a load renders wide **iff its UOp
  carries a vector dtype**, produced by the UPCAST opt on a contiguous axis.
- The transposing copy `A_copy=A_local.permute; A_copy[...].store(a[k_tile][...])` reads global **strided** (the
  gather) → UPCAST can't vectorize it → scalar `half` loads → clang emits `d16_b16`/`d16_hi_b16` + a `v_mov`
  register-init per destination. That's the 127 `v_mov`.
- Tensile reads global **coalesced+wide (b128)** and does the transpose in the **LDS write** (LDS strided writes use
  cheap `ds_store offset:` immediates), so its copy has no per-element d16 / register-init.

## CG-W2b — gating test (cheap, kernel-level): wide-read + LDS-transpose
Restructure amd_copy's copy: read `a[k_tile]` **contiguous + as an explicit vector** (innermost dim = 8 → `half8`/
`b128`), store to LDS **transposed** (strided LDS write with offsets). Measure **ISA-first** (the
`v_mov`/`global_load_b128`/`d16` counts — noise-free), then TFLOPS **fair back-to-back** (CG-W2 lesson: single-run
TFLOPS is clock-ramp noise).
- if wide loads appear (`b128` replaces `d16`) and `v_mov` drops → **the lever is expressible** (a kernel/opt
  structure that vectorizes the read), → CG-W3a (make tinygrad apply it for the matmul copy);
- if it still emits d16 / can't transpose-in-LDS at the UOp level → it's a **renderer/opt change** (CG-W3b,
  project-level).

## CG-W3 — the renderer/opt change (if CG-W2b needs it)
Make tinygrad vectorize the global read of a transposing copy: either
(a) an **opt/lowering** that recognizes a coalesced-wide global read feeding a transposed LDS store and emits the
    `b128` load + the strided LDS store (the Tensile copy shape), or
(b) **auto-coalesce per-element half loads** into vector loads when addresses are a constant-stride run (a renderer
    load-fusion pass).
Lives in: the UPCAST/vectorize opt (`codegen/opt`), the load lowering (`renderer/cstyle.py` + the
devectorizer/vectorizer), and the symbolic-index simplification. **Broad blast radius — all AMD codegen.**

## Gates
| gate | threshold |
|---|---|
| CG-W2b ISA | `b128` global loads replace `d16`; loop `v_mov` ≪ 120; correct (mse < 1e-6) |
| isolated matmul | ≥62 TFLOPS (ffn shapes), **fair back-to-back** vs the strided baseline (not single-run) |
| in-model | warm pp512 ≥1.25× PREFILL_V2 (research)/≥1.35× (strong), dNLL ≤0.01 |
| no decode regression | decode W==D unchanged (shared-codegen change) |
| test suite | tinygrad ops/linearizer/schedule green |
| no dependency / no BEAM | pure tinygrad |

KILL: CG-W2b can't produce wide loads at the UOp level AND the renderer auto-coalesce proves infeasible/regressing →
the d16 fp16-load overhead is a structural tinygrad/clang limit → close the pure-tinygrad prefill path (rest at
PREFILL_V2; Tensile route is the only ≥llama option, with its dependency).

## Against the principles
- *audit before build*: CG-W2b (cheap kernel test) gates the renderer work — exactly as CG-W2 should have led with
  ISA (lesson banked: don't trust single-run TFLOPS).
- *in-model authority + measurement confounds*: fair back-to-back + ISA, never single-run TFLOPS (the 42/65/67
  clock-ramp near-miss).
- *contain dangerous power*: a vectorize/lowering change touches all codegen → full suite + decode-W==D gate.
- *label state*: OPEN/project-level; every kernel-level lever (POWN-1/PWLT/CG-R1/CG-W2) already refuted, so this is
  the last pure-tinygrad prefill lever.

## Effort / risk
CG-W2b: hours-days (kernel probe). CG-W3: 2–4 weeks (vectorize/lowering opt, **high blast radius**), uncertain it
beats the warp-uncoalescing trade. The decode-no-regression + test-suite bar is the true ship constraint.

## Deliverables
CG-W2b probe `extra/qk_wmma_wide_copy.py` + `bench/qk-codegen-wmma/wide_copy.json` (ISA-led); result appended to
`prefill-codegen-wmma-issue-result-20260619.md`; if CG-W2b passes, the CG-W3 opt/renderer patch + the in-model/decode
gates.
