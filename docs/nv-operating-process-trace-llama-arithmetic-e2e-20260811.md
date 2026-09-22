# NV operating process: trace llama -> arithmetic-validate -> e2e dependency trace -> implement

Date: 2026-08-11
Branch: `nvidia-bringup-20260731`.

This is the standing process for every NV decode optimization on this branch.
It exists because iterating one failure mode at a time is slow and generates
churn: each change must first be pinned against the llama.cpp reference, then
proven arithmetically, then mapped across every consumer before a single line
of implementation lands. The M2b copy-elimination session (2026-08-11) was the
first full pass under this process; its artifacts are cited below as the
canonical example.

## 1. Trace llama first (the reference is the ground truth)

Read the llama.cpp graph for the exact task before proposing anything. Record
the findings in `scratchpad/llama_<task>_reference.md` (see
`scratchpad/llama_ffn_residual_reference.md` for the FFN residual example).
Answer, with line citations:

1. Does the reference render a separate kernel for this op, or absorb it into
   an epilogue? (Example: llama.cpp does NOT render the `h + ffn_out` add for
   single-token decode; the mmvq kernel computes `total + h[row]` in fp32.)
2. What buffer does the result land in, and is it reused by the next block by
   pointer (no `ggml_cpy`/`ggml_cont` between blocks)?
3. Is the reference expression bitwise the same shape as the tinygrad
   expression we would replace? Same op order, same dtype promotions, same
   rounding points?

Never guess the reference behavior. If the answer is not in the source tree,
read the source tree (`/home/ubuntu/env/llama.cpp`).

## 2. Validate with arithmetic before implementing

Prove the tinygrad change computes bitwise the same value as the reference
expression. Record the proof in `scratchpad/<task>_arithmetic_validation.md`
(see `scratchpad/m2b_arithmetic_validation.md`), including:

1. The exact expression being replaced and the exact in-kernel expression
   replacing it (same dtypes, same promotion order, same rounding).
2. The census-family swap arithmetic: which program families leave, which
   arrive, and the net kernel/time delta. Example (M2b): -36
   `E_32_32_4_02a9738c` adds, +36 `*_epi_ffnresadd` bodies, and the transport
   copies that appear if the producer boundary is not declared.
3. A bitwise check in the AB (exact-logits SHA gate) so any deviation is
   caught before wall time is spent.

## 3. Trace the end-to-end pipeline (the dependency map)

Before implementing, enumerate every thing the change touches. This is the
anti-patchwork discipline: one dependency map per change, written down before
the code.

1. **Producer side**: which AFTER/buffer is the changed value, and who declares
   its typed layout? (M2b example: the attn_qo residual_add GEMV and the
   ffn_down_resadd GEMV both declare `DeclaredTypedOutput`; without the attn_qo
   declaration the ffn_down residual fold rejects and a copy kernel renders.)
2. **Consumer side**: every residual view / typed input / fold that reads the
   changed value, and the fail-closed validator that gates it
   (`_validated_residual_view`, `_residual_producer_identity`).
3. **Callify boundary**: what the precompiled FUNCTION transform does to the
   result (the transform-time declaration registry is EMPTY - declarations are
   recorded when programs execute, so transform-time redirect gates cannot
   see them; the M2b fix lives in the execution-time fold, not callify).
4. **Census families**: list the families that must move and the families that
   must stay byte-identical between arms.
5. **Gates**: every lease/flag the change depends on, and its fail-closed
   default.
6. **Tests**: the hermetic unit tests that pin the contract, plus the AB gate
   sequence (smoke, exact-logits, census, reverse wall bracket).

## 4. Implement small, fail-closed, and gate it

1. Keep the change minimal and scoped to the dependency map from section 3.
   Every miss keeps the closed graph byte-identical (no partial behavior
   change).
2. Run the hermetic suite
   (`test/unit/test_nv_epilogue_absorption_ffn_resadd.py`,
   `test/unit/test_nv_epilogue_absorption_ab.py`,
   `test/unit/test_llm_kernel_program.py`, `test/unit/test_llm_decode_routes.py`)
   before and after.
3. Run the full AB (`--mode ab`): smoke, exact-logits (bitwise SHA), census,
   reverse wall bracket. Book only when all four pass.
4. Update the ledger row in
   `docs/task_workflow/input/nv-epilogue-absorption-route-scope-20260810.md`
   with measured census/wall numbers, then commit with the bracketed prefix
   (e.g. `[nv]`, `[docs]`) and push.

## 5. Delegation pattern (flash agents)

Research legs that are read-only and well-scoped are flash-agent work:

1. Trace llama for the task (section 1) - one agent, write the reference doc.
2. Arithmetic validation against the AB census artifacts (section 2) - one
   agent, write the validation doc.
3. Dependency-map census walk (section 3) - one agent, produce the family
   table.

The main thread implements (section 4) and runs the gates; agents never edit
`tinygrad/` or `test/unit/` in parallel with each other (write scopes overlap).

## 6. What M2b proved (example of the process working)

- Trace: llama.cpp absorbs the FFN residual add into the ffn_down mmvq
  epilogue and never copies between blocks
  (`scratchpad/llama_ffn_residual_reference.md`).
- Arithmetic: the in-kernel `total + h[row]` fp32 store is bitwise the
  standalone add expression (exact-logits SHA passed).
- Dependency map: the ffn_down residual slot consumes the attn_qo GEMV AFTER;
  the missing piece was the attn_qo typed-output declaration, not a callify
  redirect (probe-3: transform-time registry is empty). With the declaration,
  `_validated_residual_view` accepts and the boundary copy family
   (`E_32_32_4_86a23e1a`, 36) disappears; census 787 -> 751 kernels,
   kernel_us 5922 -> 5882.
