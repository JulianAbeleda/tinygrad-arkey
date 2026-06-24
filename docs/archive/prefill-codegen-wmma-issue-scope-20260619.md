# Scope - deep WMMA-issue codegen: tinygrad's prefill matmul is ALU-overhead-bound (register marshaling), not WMMA/memory

The pure-tinygrad, no-dependency path to closing prefill 48→66 TFLOPS. After refuting every bounded lever (config
sweep POWN-1, LDS-tiling PWLT-A2, software-pipeline CG-R1 — all IC-served / plateau), a hardware-grounded diagnosis
pins the real gap.

## Diagnosis (measured, the key deliverable) [M]
tinygrad's WMMA matmul kernel (amd_copy, 128×128×16, 48.5 TFLOPS): **188 VGPR / 8 waves, no spill** — *higher*
occupancy than Tensile (256 VGPR / 6 waves), so **not occupancy-limited**. Disassembly of the **hot K-loop body**
(runs 256×, all 16 WMMAs) per iteration:

| op | count | role |
|---|---:|---|
| `v_mov` | **120** | register marshaling (pack d16 global loads; shuffle WMMA fragment operands) |
| `v_add` | 40 | address/index arithmetic |
| `v_wmma_f32_16x16x16` | 16 | **useful work** |
| `ds_load` / `global_load` / `s_waitcnt` | 16 / 18 / 14 | LDS read / global prefetch / sync |

**~160 integer-ALU instructions vs 16 WMMAs (~10 ALU/WMMA) — the ALU overhead exceeds the WMMA compute.** The kernel
is **ALU-issue-bound on register marshaling + addressing**, not WMMA throughput, not memory (IC-served), not occupancy.
(The 880-line epilogue's 376 `v_add` is the one-time 128×128 accumulator store — amortized over 256 iters, irrelevant.)
Tensile's hand-tuned assembly amortizes addressing (pointer-increment, not recompute) and lays fragments in-place (no
marshaling moves) → ~1.37× denser WMMA issue → 66 TFLOPS.

## The lever (concrete, named)
Reduce the per-iteration ALU overhead in tinygrad's WMMA loop, in priority order:
1. **WMMA fragment register layout** — the 120 `v_mov` are mostly shuffling LDS-loaded operands into the VGPRs the
   `__builtin_amdgcn_wmma` intrinsic consumes. If the `ds_load` lands the A/B fragments directly in WMMA-consumable
   layout (matching the intrinsic's operand register expectations), clang emits no shuffle. This is the biggest item.
2. **address strength-reduction** — the 40 `v_add` + the d16-load packing recompute indices each iteration; hoisting
   loop-invariant address math and incrementing pointers (vs recompute) cuts the integer ALU.
3. **d16 global-load packing** — the 18 `global_load_d16_b16/hi` + packs feed `ds_store_b128`; a wider/aligned load
   path reduces the pack moves.

All three are in tinygrad's **AMD codegen / WMMA lowering** (the `Ops.SHAPED_WMMA`→`Ops.WMMA` fragment mapping + the
index/address UOp lowering the renderer emits as HIP C++ for clang). The lever is the **source structure tinygrad
emits**, since clang does the final regalloc/schedule from it.

## Phases
- **CG-W0 (done):** diagnosis above — ALU-marshaling-bound, target = the 120 `v_mov`/iter.
- **CG-W1 — attribute the 120 `v_mov`:** disassemble with source/line mapping (or bisect the kernel) to split the
  marshaling into (a) WMMA-fragment shuffle vs (b) global-load packing vs (c) addressing. Decides which sub-lever
  has the most headroom.
- **CG-W2 — WMMA fragment layout fix:** in `extra/gemm/amd_copy_matmul.py` (the UOp base), restructure the A/B
  fragment `ds_load` + the `SHAPED_WMMA` operand indexing so the loaded registers are already in the intrinsic's
  layout (no `v_mov`). Measure ALU/WMMA + TFLOPS. If it climbs toward 62, the lever is real and probe-expressible.
- **CG-W3 — renderer change (if CG-W2 needs it):** if the marshaling is forced by tinygrad's WMMA lowering /
  index simplification (not the kernel structure), the fix is in the AMD renderer / the symbolic-index strength
  reduction — a tinygrad-internals change. Scope size honestly (bounded pass vs renderer rewrite).
- **CG-W4 — integrate:** make the leaner schedule the matmul default/opt for these shapes; re-measure in-model warm
  pp512 + dNLL vs PREFILL_V2 and vs llama (3394). No dependency.

## Gates
- proof kernel ≥62 TFLOPS isolated before model wiring; KILL a sub-lever if ALU/WMMA doesn't drop or TFLOPS stays ≤50.
- in-model: warm pp512 ≥1.25× PREFILL_V2 (research) / ≥1.35× (strong), dNLL ≤0.01, decode untouched, NO dependency.
- no BEAM (gfx1100 hangs).

## Honest effort/risk
This is the real deep-codegen wall, but it is now a **concrete, located target** (the 120 `v_mov`/iter), not
"diffuse." CG-W2 (kernel-level fragment-layout experiment) is a bounded probe and the right first move — it tests
whether the marshaling is removable at the UOp level (probe-expressible, like CG-R1) or forced by the lowering (CG-W3,
renderer work, weeks). The bounded config/pipeline/LDS levers are all already refuted, so this is the last
pure-tinygrad lever for prefill matmul on gfx1100.

## Deliverables
CG-W1 attribution + CG-W2 fragment-layout kernel in `extra/qk_wmma_issue_kernel.py`, `bench/qk-codegen-wmma/*.json`,
result doc `prefill-codegen-wmma-issue-result-20260619.md` with the verdict (removable at UOp level → proof kernel; or
renderer-forced → CG-W3 scope). No model/default changes until ≥62 TFLOPS isolated.
