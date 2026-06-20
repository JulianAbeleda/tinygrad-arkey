# AMD Broad Backend BB-5a.10 Implementation Plan

Date: 2026-06-19

Generator:

- `extra/qk_amd_bb5a10_implementation_plan.py`

Artifact:

- `bench/amd-broad-backend-roadmap/bb5a10_implementation_plan_result.json`

## Verdict

`PASS_BB5A10_IMPLEMENTATION_PLAN_READY`.

BB-5a.10 now has the full phase list. P0 is complete from the layout audit. P1-P5 should run as one coordinated
implementation batch, not as disconnected one-off probes. P6-P8 are the structural, correctness, and performance gates.
P9 keeps q8 transfer blocked until P8 passes.

## Phases

| phase | track | gate | if blocked |
|---|---|---|---|
| P0 freeze selected authority contract | contract | selected MT128 authority, resource envelope, offsets, and handoff windows captured | rerun layout audit; do not use aggregate Tensile corpus |
| P1 selected-layout lowering spec | B LDS layout | Tinygrad candidate layout has A/B operand regions, selected-kernel-compatible LDS stores, `ds_load_b128` groups, and nonzero LDS budget | fall back to smaller structural layout probe; do not require bitexact byte lanes |
| P2 renderer LDS store/read lowering | B LDS layout | rendered source/disasm has nonzero LDS, LDS stores, `ds_load_b128`, and WMMA source overlap | split vector-read lowering or DEFINE_LOCAL/rangeify preservation only if that row fails |
| P3 K-loop stage scheduler | C scheduler | prologue plus steady-state `depthU=16` loop with producer/consumer LDS stages | emit two-iteration structural kernel first, then generalize |
| P4 semantic waits and barriers | C waits/barriers | dependency-derived `vmcnt`/`lgkmcnt` waits and barriers preserve LDS ordering | add dependency-group metadata; avoid textual wait edits |
| P5 resource policy and rejection | D resource policy | VGPR/SGPR/LDS/private/scratch envelope reported; bad candidates rejected before timing | reduce ambition or reject deterministically; do not time spill candidates |
| P6 structural candidate gate | E candidate gate | P2-P5 pass together: nonzero LDS, LDS stores, `ds_load_b128` feeding WMMA, waits/barriers, scratch-free | route back only to failing row; do not reopen closed LDS/knob sweeps |
| P7 correctness harness | E candidate gate | small WMMA and authority-shape fp16 correctness pass | debug layout mapping or edge predicates only |
| P8 performance gate | E candidate gate | pure tinygrad authority prefill reaches `>=60 TFLOPS` without scratch/private spill | classify counters/instruction mix before changing layout; no blind sweeps |
| P9 q8 transfer reopen decision | F q8 transfer | only after P8 pass, scope q8 downstream transfer with `<=75us` continuation gate and `<=60us` strong pass | keep q8 transfer blocked |

## Execution Shape

Run these together next:

- P1 selected-layout lowering spec;
- P2 renderer LDS store/read lowering;
- P3 K-loop stage scheduler;
- P4 semantic waits and barriers;
- P5 resource policy and rejection.

Then run the gates:

- P6 structural candidate;
- P7 correctness;
- P8 performance;
- P9 q8 transfer reopen decision only if P8 passes.

## Current Boundary

The implementation target is a non-bitexact staged-LDS authority candidate. The first candidate should match the
selected rocBLAS evidence structurally:

- selected-kernel-compatible LDS stores (`ds_store_b64` is the observed authority store path);
- `ds_load_b128` feeding WMMA source registers;
- nonzero LDS around the `25088` byte authority envelope;
- no scratch/private spill;
- semantic waits/barriers;
- correctness plus `>=60 TFLOPS` before q8 transfer reopens.
