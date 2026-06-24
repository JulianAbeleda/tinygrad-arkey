# Scope - winning Tensile prefill kernel transfer table

Date: 2026-06-20

## Objective

Build one decision-grade table for the selected prefill GEMM authority kernel:

`selected Tensile solution params -> emitted ISA features -> tinygrad equivalent/missing primitive`

This is not another broad Tensile audit. It is a concrete transfer ledger for one known winning kernel, starting with
the rocBLAS ffn_gate/up authority kernel on `gfx1100`:

- role: `ffn_gate/up`
- shape: `M=512`, `N=12288`, `K=4096`
- data: fp16 inputs, fp32 accumulation
- symbol family: `Cijk_Ailk_Bljk_HHS_BH_MT128x128x16_MI16x16x16x1_..._PGR1_PLR1_..._TT4_64_..._WGM8`
- launch: grid `[512, 96, 1]`, workgroup `[128, 1, 1]`
- resources: `VGPR=256`, `SGPR=58`, `LDS=25088`, scratch/private `0`
- measured authority: about `62-66 TFLOPS` depending on harness; three-way model route gives about `1.85x` over clean WMMA and about `85%` of llama pp512

The output should make the next implementation decision mechanical: either a tinygrad primitive exists, is partial, or
must be built before search/BEAM can be useful.

## Source Artifacts

Use these as the authority inputs:

| source | role |
|---|---|
| `bench/qk-tensile-extraction/selection.json` | selected rocBLAS and hipBLASLt symbols, launch geometry, resources, timing |
| `bench/qk-tensile-extraction/ffn_gate_up_contract.json` | kernarg, descriptor, code object, workgroup, LDS/private segment contract |
| `bench/amd-broad-backend-roadmap/bb5a10_tensile_layout_audit_result.json` | selected-function ISA counts, LDS offsets, register handoff inference |
| `docs/amd-broad-backend-bb5a10-tensile-layout-audit-20260619.md` | readable selected-kernel layout audit |
| `docs/prefill-clock-threeway-result-20260620.md` | clock-authority comparison against WMMA/Tensile/llama |
| `/home/ubuntu/rocm-libraries-tensile-sparse/shared/tensile/docs/src/conceptual/kernel-parameters.rst` | Tensile parameter meanings |
| `/home/ubuntu/rocm-libraries-tensile-sparse/shared/tensile/Tensile/Common.py` | exhaustive Tensile parameter surface |

## Table Schema

Each row in the table must include:

| column | meaning |
|---|---|
| `Primitive / contract row` | the transferable unit, not a vague optimization label |
| `Selected Tensile parameter evidence` | kernel-name token, contract field, `.dat` field, or generator source anchor |
| `Emitted ISA / runtime evidence` | actual selected-kernel disassembly, metadata, launch, resource, or counter evidence |
| `Why it matters` | performance/correctness reason |
| `tinygrad equivalent today` | exact tinygrad primitive, route, file, or "none" |
| `Gap status` | `present`, `partial`, `missing`, `blocked`, or `not-needed` |
| `Promotion/build gate` | what must be true before it can be considered enough tooling |

## Initial Rows Already Known

These rows are already sufficiently evidenced to enter the table.

| Primitive / contract row | Selected Tensile parameter evidence | Emitted ISA / runtime evidence | tinygrad status |
|---|---|---|---|
| Shape-specialized GEMM contract | `MT128x128x16`, `MI16x16x16x1`, `TT4_64`, `WG32_4_1`, fixed `M/N/K` | stable named symbol, grid `[512,96,1]`, workgroup `[128,1,1]` | partial: tinygrad has WMMA matmul, not this full contract |
| Matrix instruction atom | `MI16x16x16x1`, `ISA1100`, `FMA`, `MIAV1` | selected function has `80` `v_wmma` | present: tinygrad can emit RDNA3 WMMA |
| Explicit LDS operand staging | `TLDS1`, `LDS=25088`, `1LDSB0` | `32` `ds_store_b64`, `40` `ds_load_b128`, LDS offsets `0..18736` | partial: experiments can emit LDS, but not production-quality staged pipeline |
| Global-to-LDS handoff | `BL1`, `GLVWA4`, `GLVWB4`, `GRCGA1`, `GRCGB1`, `PGR1` | `24` `buffer_load_b64`; `32/32` inspected LDS stores reuse recent global-load data regs | missing as a scheduled primitive |
| LDS-to-WMMA handoff | `LRVW16`, `PLR1`, `UMLDSA0`, `UMLDSB1` | `80/80` inspected WMMAs consume recent `ds_load_b128` destination regs | partial: renderer can have registers feed WMMA, but not from a stable LDS read schedule |
| Wait/barrier schedule | `SIA1`, `SLW1`, `PGR1`, `PLR1` | `545` `s_waitcnt`, `6` `s_barrier` in selected function | missing/partial: waits exist, but no resource-aware K-loop wait scheduler |
| Resource envelope | metadata and contract | `VGPR=256`, `SGPR=58`, `LDS=25088`, scratch `0`, private `0` | partial: tinygrad observes resources, but does not reject/schedule to this envelope |
| Workgroup mapping | `WGM8`; kernargs include `NumWorkGroups0/1`, `NumFullBlocks`, `WgmRemainder1`, magic divide | runtime contract has 5 WGM fields at kernarg offsets `104..120` | missing as native scheduler policy |
| No workspace / no opaque runtime | `SU0_SUM0_SUS0`, scratch `0` | single kernel, no workspace, no hidden args | present for extracted route; native route must preserve this |

## Rows That Need More Extraction

These are not table-complete yet and should be extracted before claiming we understand the selected kernel deeply.

| Missing row | What to extract | Why |
|---|---|---|
| Exact Tensile solution YAML/dat fields | Parse selected `.dat`/logic row for all solution fields, not just kernel-name tokens | separates real solution params from inferred abbreviations |
| Full steady-state K-loop schedule | locate prologue, loop body, local read/write cadence, WMMA issue cadence, epilogue | tells us whether tinygrad needs a new loop scheduler or just renderer ops |
| LDS logical tile map | map A/B logical tile coordinates to LDS byte offsets and load vectors | needed for a source-level tinygrad primitive, not only ISA mimicry |
| Bank/padding policy | decode `LdsPad*`, `LdsBlockSizePerPad*`, `TransposeLDS`, `UnrollMajorLDS*` from source/dat | avoids repeating earlier naive LDS regressions |
| Occupancy/resource policy | compute occupancy from VGPR/LDS/waves and compare to tinygrad candidates | prevents a candidate that is correct but below authority |
| Tinygrad authority contrast | disassemble clean WMMA authority under same shape and align counts | makes the "missing equivalent" column concrete |
| Counter contrast | align GL2/DS/SQ wait counters if available | validates that the ISA-level differences are performance-active |

## Execution Plan

### Phase 1 - Normalize Selected-Kernel Provenance

Goal: produce a compact JSON/Markdown row with the exact selected rocBLAS kernel, code object, launch, resources, and
timing.

Pass:

- one symbol, one code object, one shape, one timing band;
- no ambiguity between rocBLAS `MT128x128` and hipBLASLt `MT96x96/UserArgs`;
- table marks hipBLASLt as a comparison row, not the first native-transfer target.

### Phase 2 - Decode Solution Parameters

Goal: turn kernel-name abbreviations and `.dat`/generator fields into a readable parameter ledger.

Pass:

- `MT`, `MI`, `TT`, `WG`, `PGR`, `PLR`, `LRVW`, `TLDS`, `UMLDS*`, `WGM`, `GLVW*`, `GRCG*`, `GSU/SU/SUM/SUS` decoded;
- each decoded field has either a source artifact or a Tensile docs/source anchor.

### Phase 3 - ISA Feature Ledger

Goal: summarize emitted selected-function features with counts and representative offsets.

Pass:

- global loads, LDS stores, LDS reads, WMMA, waits, barriers, scratch/private, resource metadata;
- register handoff evidence for global-load -> LDS-store and LDS-read -> WMMA source operands.

### Phase 4 - Tinygrad Equivalent Audit

Goal: map each primitive row to tinygrad today.

Pass:

- row-level status is one of `present`, `partial`, `missing`, `blocked`, `not-needed`;
- every `partial/missing/blocked` row has a concrete build gate;
- no row says "use BEAM" unless the underlying primitive exists in the search space.

### Phase 5 - Build Readiness Decision

Goal: decide whether we have enough tooling to start native implementation or whether extraction must continue.

Pass:

- if the missing rows are only tunable parameters, proceed to machine search;
- if the missing rows are representation/scheduler primitives, build those first;
- if exact LDS map remains unknown, allow a non-bitexact Tensile-class candidate only if the table makes that explicit.

## Expected Decision

The likely decision is:

- tinygrad has the WMMA atom;
- tinygrad has partial LDS and wait machinery;
- tinygrad does not yet have a first-class Tensile-like GEMM schedule object covering global-to-LDS staging,
  LDS-to-WMMA operand scheduling, WGM policy, and resource-aware K-loop pipelining;
- therefore BEAM/search is premature for this target until those primitive rows are represented.

## Deliverable

Create `docs/prefill-tensile-winning-kernel-transfer-table-20260620.md` with:

1. selected-kernel provenance;
2. decoded parameter ledger;
3. emitted ISA ledger;
4. primitive transfer table;
5. tinygrad build/search readiness verdict.

