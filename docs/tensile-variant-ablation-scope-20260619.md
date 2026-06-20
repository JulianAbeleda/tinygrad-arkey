# SCOPE — Tensile variant-ablation audit (gfx1100, 2026-06-19)

Convert "I'm reciting the primitives that make Tensile win" into a MEASURED attribution. Follow-up to
`prefill-tensile-DEFINITIVE-source-of-truth-20260619.md` and the honest retraction of the
"software-pipeline + dense-issue + occupancy" list (dense-issue already refuted: VALU is equal). We could not ablate
Tensile by recompiling, but the extracted rocBLAS `.co` is **580 kernel variants** = a built-in ablation matrix.

## Established facts (this scope builds on)
- `.co` = 580 kernels; macro-tiles `128x128x16`(244) / `64x64x32`(140) / `32x32x32`(112) / `64x64x16`(64) /
  `32x32x16`(20); ALL `MI16x16x16` (WMMA — kills "Tensile=FMA-only"); various GSU/StreamK/prefetch flags.
- `extra/qk_tensile_hcq_launch.py` loads an ARBITRARY symbol by name (`kd_offset(elf, sym)`); `unbundle()` gives ELF.
- PMC scoreboard (`extra/qk_prefill_primitive_pmc.py`) measures cycles/DRAM/stalls/occupancy/VALU and is CONFIRMED to
  capture Tensile kernels.
- Tensile name encoding to decode per variant: `MT<m>x<n>x<k>` (macro-tile), `MI16x16x16` (WMMA), `GSU<n>`/`SU<n>`
  (split-K / StreamK), `LDSB<0|1>` (LDS buffering), `PGR<n>`/`PLR<n>` (prefetch global/local read), `WG<x>x<y>x<z>`
  (workgroup), `APM`/`ABV` (assertion/buffer flags).

## The core discriminator (why this settles it)
Tensile variants all share LLVM/Tensile **scheduling quality** but differ in **primitive geometry** (tile,
occupancy, LDS-size, pipeline depth). tinygrad's WMMA has poor scheduling but we can pick its geometry. So:
- If even the **most-tinygrad-like Tensile variant** (smallest tile `32x32x32`, least LDS, lowest occupancy) STILL
  beats tinygrad WMMA → the win is **scheduling quality**, NOT the primitive presence (matches A3, which added the
  primitives to tinygrad and got 6 TFLOPS).
- If that variant **degrades toward tinygrad's level** → the **primitive geometry** is the driver, and the sweep
  shows which knob.
This single contrast does most of the epistemic work; the full sweep then quantifies the sensitivity.

## Phase plan
### P0 — Variant inventory + launchability triage
Enumerate all 580 symbols; parse the name encoding into a param table. Select the ablation set: variants computing the
SAME gateup contraction (HHS, Ailk_Bljk) at the SAME shape (M=512,N=12288,K=4096) that differ from the baseline
(`MT128x128x16` selected kernel) in ONE knob — tile ∈ {128x128,64x64,32x32}, K-depth ∈ {16,32}, prefetch/LDSB flags.
Flag non-StreamK / no-workspace variants (simple kernarg contract → directly launchable) vs StreamK (need workspace).
Artifact: `bench/qk-tensile-ablation/variants.json` (symbol, decoded params, launchable, contract).

### P1 — Correctness gate per variant
Launch each candidate on the real gateup A/B/C buffers (reuse `qk_tensile_hcq_launch` grid/kernarg derivation per
variant's `.kd`). Verify `rel_err` vs an fp32 reference (or the baseline Tensile output). DROP any variant that is
wrong or won't launch (record why). Only correct variants proceed. (Different tiles need different grid = ceil(N/mt_n)
× ceil(M/mt_m); StreamK needs the SU workspace + grid.)

### P2 — PMC + disasm ablation sweep
For each correct variant: PMC scoreboard (`GRBM_GUI_ACTIVE` cycles, `GL2C_MC_RDREQ`, `SQ_WAIT_ANY`, `SQ_WAVES`,
`SQ_INSTS_LDS`, `SQ_INSTS_VALU`; 2 passes) + clean cycle timing + static disasm (LDS bytes from `.kd`
group_segment, VGPR from rsrc1, `ds_load`/`s_barrier`/`v_wmma`/`v_fma_mix` counts). Build the matrix:
variant → {tile, K-depth, LDS bytes, waves, prefetch, vgpr} × {cycles, DRAM, stalls, occupancy, VALU}.
Include the tinygrad WMMA point and (from docs) the A3 hand-LDS extremes as anchors. Artifact:
`bench/qk-tensile-ablation/ablation_matrix.json`.

### P3 — Attribution analysis
Read off sensitivities, holding others as constant as the data allows:
- tile ↓ (128→64→32): occupancy ↑, LDS-reuse ↓ — which way do cycles/DRAM/stalls move?
- K-depth 16→32: deeper K-block / pipeline — stalls?
- the **discriminator contrast**: smallest-tile Tensile vs tinygrad WMMA — does Tensile still win?
Output one of: (a) a defensible "throughput scales with X" statement with the measured deltas; or (b) the honest
finding that the gap is ~invariant to the tunable geometry → it's holistic codegen/scheduling quality (consistent
with A3). EXPLICITLY state confounds (tile moves occupancy AND LDS together → attribute at "tile geometry"
granularity, not a single primitive).

### P4 — (gated, optional) SQTT instruction-level
Only if P3 leaves the dominant cause ambiguous: ATT/SQTT trace (`qk_att_primitive_atlas`) on the baseline + one
extreme variant to read actual per-wave stall reasons (mem-wait vs barrier vs dep). Heavyweight; skip if P3 is clear.

### P5 — Verdict + correct the record
Update `prefill-tensile-DEFINITIVE-source-of-truth` §4/§5 with the MEASURED attribution (or the "scheduling-quality,
not primitive" verdict); keep "dense-issue" retracted; close the "how do you know the primitives" question with evidence.

## Gates / hazards
- Verify variant CORRECTNESS before trusting its counters (don't measure a wrong kernel).
- PMC perturbs timing → use cycle/count RATIOS, not PMC-run wall time; ≤8 counters/pass (multi-pass); sum across instances.
- Confounded knobs → attribute at the granularity the data supports; never claim a single isolated primitive when the
  knob moves several.
- If all Tensile variants cluster well above tinygrad regardless of geometry, that IS the answer (scheduling quality);
  don't force a primitive story.

## Deliverables
- `docs/tensile-variant-ablation-result-20260619.md`; `extra/qk_tensile_variant_ablation.py`;
  `bench/qk-tensile-ablation/{variants,ablation_matrix}.json`. No default/route change (measure-only).

## Definition of done
A measured attribution: either "Tensile's win tracks structural knob X by ΔY" (replacing the recited list), OR the
honest "the win is invariant to tunable geometry → it's codegen/scheduling quality" — with confounds stated and the
DEFINITIVE doc corrected. Either outcome ends the primitive-recitation and answers "how do you know."
