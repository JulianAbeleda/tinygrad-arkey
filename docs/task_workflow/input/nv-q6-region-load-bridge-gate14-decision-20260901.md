# NV Q6 RegionLoadBridge Gate14 decision (2026-09-01)

## Goal

Preserve llama's Q8 panel-1 phase boundary through tinygrad UOps and CUDA lowering, then admit the route only if the real Q6 cubin has exactly 18 panel loads and stores, the llama barrier topology, a load-to-store span at most 160 instructions, no spills, and the frozen arithmetic and memory census.

## Verdict

`REJECT_STATIC_COMPILER_SCHEDULE`

The UOp marker-erasure defect is repaired and the real Q6 candidate reaches NVRTC/SASS. Both evidence-supported PTX forms fail the static gate, so correctness, R31 timing, promotion, commit, and push are not released.

## Repaired substrate boundary

- `RegionLoadBridge` survives generic `AFTER` simplification and final program verification.
- Pointer ownership survives new-style pointer-to-buffer scalarization.
- The bridge accepts only one shared typed region, 18 immutable scalar32 global loads, 18 unique direct scalar32 shared stores, one common source and destination, and the existing overwrite barrier.
- The real Q6 builder exposes one explicit bridge flag; the default anchor is unchanged.
- Focused structural/source tests pass: `9 passed, 1 deselected`.

## Synthetic evidence

The raw round-major CUDA/PTX fixture previously passed with an LDG-to-STS span of 104 instructions, 64 registers, and no spills. The equivalent real-UOp fixture renders accumulator chains as nested chain-major expressions; its split bridge is spill-free but spans 256 instructions. This isolates source/phase-body linearization as material to ptxas scheduling.

## Real Q6 Gate14

Both real-Q6 forms compile to the same cubin:

- Cubin SHA-256: `d803ebaa583db01bf7df42ff58c7d1439919d1c06aaf5911c39b6b16ec23d79d`
- Panel loads/stores: `18/18`
- Panel load-to-store span: `621` instructions
- Barrier ordinals: `764, 2714, 2737, 5078`
- First panel load/store ordinals: `2098/2719`
- Registers/stack: `255/32 B`
- `LDL/STL`: `8/8`
- `LDS`: `184` (required `176`)
- Instructions: `5184` (required at most `5144`)
- Frozen families retained: `IMMA 256`, `LDSM 32`, `LDG 109`, `STS 73`, `BAR 4`, `I2FP 1024`, `FMUL 1544`, `FADD 1024`, `FFMA 0`

The fused form was verified in generated CUDA as one inline PTX region containing 18 `ld.global.u32`, `bar.sync 0`, and 18 `st.shared.u32`. ptxas canonicalized it to the same cubin as the split C-barrier form.

## What this proves

The missing lever is not another load marker, pointer qualifier, C lexical region, split inline-PTX bridge, or fused inline-PTX bridge. All can express the logical topology, but ptxas still schedules the panel values across too much live accumulator work and spills them.

The next substrate target is phase-body linearization/lifetime control: tinygrad must retain a round-major or otherwise bounded-live accumulator schedule into CUDA/PTX, or lower the complete compute-and-copy phase below CUDA source where register lifetimes and scheduling can be owned together. That target requires its own synthetic test before another real-Q6 investment.

## Promotion state

- Correctness: not run
- R31 timing: not run
- Commit: not created
- Push: not performed

Artifacts: `docs/task_workflow/evidence/nv-q6-region-load-bridge-gate14-20260901/artifacts`
