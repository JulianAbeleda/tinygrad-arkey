# NV Q6 strict-after panel-1 Gate15 decision (2026-09-01)

## Decision

`REJECT_PERFORMANCE_SECONDARY_LEVER`

The compiler substrate now keeps `StrictAfter` dependencies opaque during commutative index canonicalization. The one-load CUDA/SASS microgate passes, the admitted anchor and fixup rebuild byte-identically, and the full Q6 candidate compiles in about 8 seconds instead of exceeding the 240-second bound.

The full candidate is exact but does not meet the promotion threshold. No route was promoted, committed, or pushed.

## Full-Q6 binary

- Candidate cubin: `51edb7a2406f81c565702597b6f1fb5f934ea659194a0a7b84186da5489e96cc`
- Anchor cubin: `6eb663b3a3fd628e3394a0ce8f8780e108e47f40b887b0a75a0756dcf33e9137`
- Panel-1 loads/stores: `18/18`
- First-load to first-store span: `288` instructions; required at most `160`
- Registers/stack/LDL/STL: `255 / 0 B / 0 / 0`
- Instructions: `5136`
- Frozen arithmetic: `IMMA 256`, `LDSM 32`, `I2FP 1024`, `FMUL 1544`, `FADD 1024`, `FFMA 0`
- Memory-family difference: candidate `LDS 184`; required `176`

The dependency is preserved, but one accumulator update is not a collective phase token: ptxas can schedule that independent chain early. A volatile round-trip did not repair placement and regressed to `40 B` stack with `9/9 LDL/STL`. A post-barrier bridge regressed to `32 B` stack with `8/8 LDL/STL`. Post-barrier transport widths `1, 2, 4, 8, 18` all canonicalized to the same cubin, proving source grouping does not control the final transport schedule.

## Correctness

- Trusted reference: passed with zero failing elements
- Candidate versus anchor partial uint32 identity: exact
- Candidate versus anchor final uint32 identity: exact
- GPU fixup versus CPU fixup: exact
- Active/unused output contracts: finite/NaN exact

## Exploratory locked R31

The static gates were explicitly bypassed only to measure causality; the actual failed gates were recorded before GPU work. These numbers are not promotion evidence.

| Metric | Anchor | Candidate | Paired candidate-anchor |
|---|---:|---:|---:|
| Main median | 229.824 us | 228.384 us | -1.312 us, 27/31 wins |
| Fixup median | 24.608 us | 24.896 us | +0.320 us, 14/31 wins |
| Total median | 254.432 us | 253.216 us | -1.760 us, 23/31 wins |

Required promotion was at least `3 us` improvement and `24/31` wins for both main and total. The candidate fails that bar and recovers only about 4% of the admitted `46.4 us` llama gap.

## Evidence

- `docs/task_workflow/evidence/nv-q6-strict-after-panel1-gate15-20260901/exploratory-r31.json`
- `docs/task_workflow/evidence/nv-q6-strict-after-panel1-gate15-20260901/artifacts`
- `docs/task_workflow/evidence/nv-q6-post-barrier-panel1-gate16-20260901/artifacts`
