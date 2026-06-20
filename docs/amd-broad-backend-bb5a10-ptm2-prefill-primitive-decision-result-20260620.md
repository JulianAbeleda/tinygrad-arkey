# BB-5a.10 PTM-2 — Prefill Primitive Decision

Date: 2026-06-20

Inputs:
- `bench/amd-broad-backend-roadmap/bb5a10_ptm1_same_harness_authority_bridge_result.json` (PTM-1)
- `bench/qk-tensile-extraction/codegen_oracle.json` (capability delta)

Verdict:
`PTM2_DECIDED_SOFTWARE_PIPELINED_K_LOOP` (with a re-baselined target)

## The decision

Per the roadmap, PTM-1 returned `GAP_REAL_KERNEL_QUALITY` (authority > candidates under one clock), so PTM-2
selects **exactly one** native row. The choice is **`software_pipelined_k_loop`** — the capability the
`codegen_oracle` names as the sole missing piece (double-buffered global→LDS→reg prefetch across the K-loop;
macro-tile and WMMA fragment are already identical between tinygrad and Tensile). Standalone LDS stays
closed; this is the one row.

## But the target is re-baselined by PTM-1 (this is the important part)

PTM-1 changes *what success means*. The prior framing — "tinygrad authority 43 vs hand-ASM 18 = 2.34×, so
build a better hand-ASM candidate" — is **retired**:

- Under one clock the authority-vs-best-candidate gap is **1.33×**, not 2.34×.
- The hand-ASM Route-A candidates (LDS macro 39.9, global-direct 29.6 TFLOPS) are **below** tinygrad's own
  LLVM authority (52.97 this clock). So no hand-ASM candidate built so far beats the existing compile.

Therefore the native row is **not** "another hand-ASM kernel to beat the candidates." The real, remaining
prefill headroom is **tinygrad authority → Tensile**: ~42 nominal (≈53 this high-clock session) up to
Tensile's ~66. The `software_pipelined_k_loop` is the thing that closes *that* span — it must lift
tinygrad's authority itself, not merely catch up to it.

## Honest constraint (carried from MEMORY, not re-litigated)

Building the SW-pipelined K-loop is the known **codegen wall** (BEAM-hang / linearizer-RANGE class: a global
load cannot be hoisted across the loop RANGE, so no software pipeline). POWN/Route-A A1/A2/A3 already
explored the dependency-free hand-ASM version and capped at ~24-32 TFLOPS, below LLVM. So the decision is a
**direction**, not a green-light to build: PTM-3 scopes the row; the actual build is a separate, multi-day,
high-uncertainty codegen effort whose success likely tops out near LLVM's ceiling anyway.

## Stop-rule compliance
- One row chosen (`software_pipelined_k_loop`); no standalone LDS; no mixed-harness comparison (PTM-1 fixed
  the harness); external `.co` route kept separate (that's PTM-4, the user's dependency-policy call).

## Next

PTM-3 native candidate scope (scope only) — see
`amd-broad-backend-bb5a10-ptm3-native-candidate-scope-20260620.md`.
