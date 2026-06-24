# Prefill Tensile Winning Kernel Transfer Table

Date: 2026-06-20

## Verdict

We have enough evidence to explain the selected rocBLAS ffn_gate/up kernel at the primitive-contract level, but not
enough to start machine search over tinygrad schedules. The missing surface is not the WMMA atom and not merely LDS
instructions. The missing surface is a Tensile-class GEMM schedule object:

`global -> LDS staging + LDS -> WMMA operand reads + steady-state K-loop pipeline + WGM/resource policy`

Until that exists, BEAM/search would tune the wrong space.

## Selected Kernel

Authority target:

| field | value |
|---|---|
| role | `ffn_gate/up` |
| shape | `M=512`, `N=12288`, `K=4096` |
| data | fp16 input, fp32 accumulation |
| library | rocBLAS Tensile |
| code object | `/opt/rocm-7.2.4/lib/rocblas/library/TensileLibrary_Type_HH_HPA_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx1100.co` |
| symbol | `Cijk_Ailk_Bljk_HHS_BH_MT128x128x16_MI16x16x16x1_..._PGR1_PLR1_..._TT4_64_..._WGM8` |
| launch | grid `[512, 96, 1]`, workgroup `[128, 1, 1]` |
| resources | `SGPR=58`, `VGPR=256`, `LDS=25088`, private/scratch `0` |
| kernarg | 128 bytes, 19 by-value args, no hidden args |
| workspace | none; `SU0_SUM0_SUS0`, single GEMM kernel |
| measured band | about `62-66 TFLOPS` isolated; model route is about `1.85x` over clean WMMA and about `85%` of llama pp512 |

Primary artifacts:

- `bench/qk-tensile-extraction/selection.json`
- `bench/qk-tensile-extraction/ffn_gate_up_contract.json`
- `bench/qk-tensile-extraction/ffn_gate_up_schedule_template.json`
- `bench/amd-broad-backend-roadmap/bb5a10_tensile_layout_audit_result.json`
- `docs/prefill-tensile-schedule-template-extraction-result-20260620.md`
- `docs/prefill-tensile-lds-tile-map-sketch-20260620.md`
- `docs/amd-broad-backend-bb5a10-tensile-layout-audit-20260619.md`
- `docs/prefill-clock-threeway-result-20260620.md`

## Decoded Parameter Ledger

| parameter token | decoded meaning | selected value | evidence |
|---|---|---:|---|
| `MT128x128x16` | macro tile M x N x unroll depth | `128x128`, `DepthU=16` | kernel symbol, Tensile docs define `MacroTile` and `DepthU` |
| `MI16x16x16x1` | matrix instruction atom | `16x16x16x1` | kernel symbol; selected disasm has `v_wmma` |
| `WG32_4_1` | workgroup decomposition | `32 x 4 x 1 = 128` threads | kernel symbol and launch contract |
| `TT4_64` | thread tile shape | `4 x 64` | kernel symbol |
| `PGR1` | prefetch global read | enabled one-stage global prefetch | kernel symbol; Tensile `PrefetchGlobalRead` docs/source |
| `PLR1` | prefetch local read | enabled one-stage LDS read prefetch | kernel symbol; Tensile `PrefetchLocalRead` docs/source |
| `LRVW16` | local read vector width | `16` | kernel symbol; `ds_load_b128` evidence |
| `GLVWA4/GLVWB4/GRVW4` | global load vector width | `4` elements | kernel symbol; `buffer_load_b64` on fp16 pairs |
| `GRCGA1/GRCGB1` | global read coalesce group | enabled | kernel symbol; Tensile docs/source |
| `TLDS1` | transpose/use LDS layout policy | enabled | kernel symbol; LDS offsets and load/store evidence |
| `UMLDSA0/UMLDSB1` | unroll-major LDS layout per operand | A false, B true | kernel symbol; needs deeper logical map |
| `1LDSB0` | one-LDS-buffer override | disabled; normal double-buffer-capable policy | kernel symbol; selected LDS allocation `25088` |
| `WGM8` | workgroup mapping | blocking map width `8` | kernel symbol; WGM kernargs at offsets `104..120` |
| `SU0/SUM0/SUS0` | StreamK / split-U / workspace family | disabled | kernel symbol; no workspace/fixup kernels |

## Emitted ISA Ledger

Selected-function facts from `bb5a10_tensile_layout_audit_result.json`:

| feature | selected-kernel evidence |
|---|---|
| global loads | `24` `buffer_load_b64`, plus d16 buffer loads in the function body |
| global-to-LDS stores | `32` `ds_store_b64`, `0` `ds_store_b128` |
| LDS-to-register reads | `40` `ds_load_b128` |
| matrix atom | `80` `v_wmma` |
| waits | `545` `s_waitcnt` |
| barriers | `6` `s_barrier` |
| LDS offsets | stores cover representative offsets `0..17248`; reads cover `0..18736` |
| handoff inference | `32/32` inspected LDS stores reuse recent global-load data registers |
| WMMA source inference | `80/80` inspected WMMAs consume recent `ds_load_b128` destination registers |
| resource envelope | `VGPR=256`, `SGPR=58`, `LDS=25088`, scratch/private `0` |

This proves the selected kernel is not a global-direct WMMA kernel. It stages operands through LDS and feeds WMMA from
LDS-loaded VGPR ranges. It does not prove the source-level logical tile map for every A/B coordinate.

## Primitive Transfer Table

| primitive / contract row | selected Tensile parameter evidence | emitted ISA / runtime evidence | why it matters | tinygrad equivalent today | gap status | promotion/build gate |
|---|---|---|---|---|---|---|
| Shape-specialized GEMM contract | `MT128x128x16`, `MI16x16x16x1`, fixed `M/N/K` | stable named symbol, fixed grid/workgroup | avoids generic runtime choice and lets schedule be fully specialized | clean WMMA matmul exists for same shape | partial | represent this as a first-class AMD GEMM schedule row, not incidental lowered UOps |
| WMMA atom | `MI16x16x16x1`, `ISA1100` | `80` `v_wmma` | tensor op is required but not sufficient | RDNA3 WMMA exists in renderer/runtime experiments | present | keep; not the blocker |
| Macro/workgroup tiling | `MT128x128x16`, `WG32_4_1`, `TT4_64` | launch `[512,96,1]` / `[128,1,1]` | defines occupancy, tile ownership, and output mapping | partial authority macro probes exist | partial | full authority-grid correctness and row/col output mapping without toy-tile shortcuts |
| Global vector/coalesced reads | `GLVWA4`, `GLVWB4`, `GRCGA1`, `GRCGB1`, `PGR1` | `buffer_load_b64`; global-load regs feed LDS stores | keeps memory movement wide and scheduled ahead of compute | normal global loads exist; scheduled prefetch row does not | missing | explicit global-read stage with dependency metadata and vector-width policy |
| Global-to-LDS staging | `TLDS1`, `PGR1`, `SLW1`, `LDS=25088` | `32` `ds_store_b64`; `32/32` handoff pass | moves next K tile toward the compute pipe | handmade LDS candidates exist, but wrong performance family | partial/blocked | staged producer path that overlaps with consumer path and reaches `>=60 TFLOPS`, not just visible LDS stores |
| LDS layout and offsets | `TLDS1`, `UMLDSA0`, `UMLDSB1`, `LRVW16` | selected offsets for stores/loads, LDS range `0..18736` | bank/layout determines whether LDS helps or hurts | non-bitexact layout probes only | partial | recover enough logical A/B tile map or explicitly choose non-bitexact layout with structural gates |
| LDS-to-WMMA operand reads | `LRVW16`, `PLR1` | `40` `ds_load_b128`; `80/80` WMMA source handoff pass | feeds WMMA with wide operand fragments | tinygrad can feed WMMA from registers; LDS-read-fed WMMA exists only in probes | partial | renderer/scheduler primitive for LDS fragment reads whose VGPR outputs are consumed by WMMA |
| Steady-state K-loop pipeline | `DepthU=16`, `PGR1`, `PLR1`, `SIA1` | many waits/barriers; selected body has producer/consumer handoff windows | this is likely the real difference versus naive LDS | no production software-pipelined K-loop scheduler | missing | prologue/steady/epilogue schedule with overlapped global load, LDS store, LDS read, WMMA issue |
| Wait/barrier policy | `PGR1`, `PLR1`, `SLW1`, `SIA1` | `545` waits, `6` barriers | correctness and latency hiding depend on exact dependency waits | probe-level wait insertion exists | partial | dependency-group wait scheduler over staged memory ops, not textual wait insertion |
| Resource policy | contract metadata | `VGPR=256`, `SGPR=58`, `LDS=25088`, scratch/private `0` | prevents a correct kernel from spilling or losing occupancy | resource probes exist, not integrated as a scheduler constraint | partial | reject candidates before timing if scratch/private or bad VGPR/LDS envelope appears |
| Workgroup mapping | `WGM8` | kernargs `NumWorkGroups0/1`, `NumFullBlocks`, `WgmRemainder1`, magic divide | controls tile ordering/cache behavior | extracted route fills it; native scheduler lacks policy | missing | native WGM mapping formula and launch/kernarg policy |
| Workspace-free route | `SU0_SUM0_SUS0` | no workspace, no hidden args, single kernel | makes HCQ/native route maintainable | extracted route proves it | present | preserve no-workspace/no-fixup constraint for first native candidate |
| Clock authority | separate from solution params | WMMA/Tensile/llama all high clock; manual peak unchanged | rules out DPM as explanation | harness exists | present | keep clock telemetry in future comparisons |

## Tinygrad Contrast

The important contrast is now specific:

| route | best known status | implication |
|---|---|---|
| clean tinygrad WMMA | high-clock pp512 about `1436 tok/s`; PTM same-harness authority measured `52.97 TFLOPS` on the synthetic authority comparison | WMMA atom and LLVM scheduling are not absent |
| selected rocBLAS Tensile | about `62-66 TFLOPS`, about `1.85x` over clean WMMA in pp512 model route | dataflow/schedule contract is stronger |
| hand native LDS macro | correct sampled authority tiles, scratch/private `0`, but `18-21 TFLOPS` | visible LDS is not enough and can be the wrong performance family |
| converted DS64 macro | DS64 contract fixed, but still `~97.5%` of slow B128 macro | `ds_store_b64` vs `ds_store_b128` was not the main blocker |

So the transfer target is not "clone these offsets and add barriers." The target is a scheduled GEMM pipeline that
achieves Tensile-like overlap/resource behavior.

## Schedule Template Extraction Update

`extra/qk_tensile_schedule_template_extract.py` decoded the selected rocBLAS `.dat` row and emitted
`bench/qk-tensile-extraction/ffn_gate_up_schedule_template.json`.

Confirmed from `.dat` solution row `52 / 290`, Tensile solution index `1140853605`:

| field | value |
|---|---|
| `macroTile` | `[128, 128, 1]` |
| `threadTile` | `[4, 64]` |
| `workGroup` | `[32, 4, 1]` |
| `depthU` | `16` |
| `workGroupMapping` | `8` |
| `globalSplitU` | `1` |
| `streamK` / `streamKAtomic` | `0 / 0` |
| `globalAccumulation` | `0` |
| `workspaceSizePerElemC` | `0` |

Conservative disassembly segmentation:

| region | line range | key counts |
|---|---:|---|
| prologue before first WMMA | `282071..282554` | `buffer_load_b64=16`, `ds_store_b64=16`, `ds_load_b128=8`, `s_waitcnt=12`, `s_barrier=1` |
| steady first-to-last WMMA | `282555..283735` | `v_wmma=80`, `buffer_load_b64=8`, `ds_store_b64=16`, `ds_load_b128=32`, `s_waitcnt=17`, `s_barrier=5` |
| post-WMMA tail | `283736..289317` | `buffer_load_d16_b16=256`, `s_waitcnt=516` |

This removes the "exact `.dat` row not promoted" blocker. It does not remove the LDS tile-map or exact K-loop
reconstruction blockers.

## LDS Tile-Map Sketch Update

`docs/prefill-tensile-lds-tile-map-sketch-20260620.md` reconciles the selected LDS offsets with the Tensile allocation
formula. The selected non-bitexact layout sketch is:

| logical region | byte base | byte span | role |
|---|---:|---:|---|
| A lower buffer | `0` | `4096` | current/prefetched A tile |
| B lower buffer | `4096` | `4608` | current/prefetched B tile with padding |
| alignment gap | `8704` | `7680` | power-of-two second-buffer alignment gap |
| A second buffer | `16384` | `4096` | alternate A tile for `PGR1` |
| B second buffer | `20480` | `4608` | alternate B tile for `PGR1` |

Total LDS bytes: `25088`, matching selected metadata.

This is enough to design a non-bitexact schedule object. It is not enough to claim a byte-identical Tensile clone.

## What Remains Unknown

These are the rows that block a bit-exact or search-ready native clone:

| unknown | current state | consequence |
|---|---|---|
| exact `.dat` solution fields | decoded and promoted for selected row | no longer a blocker for non-bitexact candidate |
| source-level LDS coordinate map | non-bitexact A/B region map exists; exact per-element coordinates not reconstructed | cannot claim bit-identical Tensile layout, but enough for first schedule-object design |
| steady-state schedule segmentation | conservative first-WMMA/last-WMMA segmentation exists; exact symbolic loop still not annotated | cannot yet encode a faithful scheduler template |
| bank/padding rationale | B padding reconciled as `8` fp16 elems per `128` byte block; full conflict rationale not replayed from generator | native layout still needs structural/perf gates |
| same-shape tinygrad disasm table | performance docs exist; no single aligned table here | needed for final "missing equivalent" proof |
| counter contrast after native candidate | prior PMC direction exists; not tied to every table row | needed to classify whether a candidate is memory, wait, or occupancy limited |

## Build/Search Readiness

| question | answer |
|---|---|
| Do we understand the high-level Tensile primitive? | yes |
| Do we know the selected authority kernel and its runtime contract? | yes |
| Do we have enough to launch the extracted kernel through HCQ? | yes; TPE contract says no hidden args/workspace |
| Do we have enough to implement a bit-identical native Tensile clone? | no |
| Do we have enough to implement a non-bitexact Tensile-class candidate? | partially, but prior native LDS macro shows the naive path is the wrong family |
| Do we have enough to let BEAM/search solve it? | no; the primitive rows are not first-class search dimensions yet |

## Next Implementation Sequence

1. Refine the selected disassembly segmentation from heuristic first-WMMA/last-WMMA into an exact K-loop template.
2. Build a tinygrad AMD GEMM schedule object with explicit stages: global load, LDS store, barrier/wait, LDS read,
   WMMA consume, store.
3. Add resource gates before timing: no scratch/private, bounded VGPR/SGPR, LDS envelope near `25088`.
4. Only then run search over tile/layout/pipeline knobs.

The immediate build gate for "enough tooling" is not a TFLOPS number. It is the existence of the above schedule object
and resource-gated lowering. The performance gate after that remains `>=60 TFLOPS` on `512x4096x12288`.
