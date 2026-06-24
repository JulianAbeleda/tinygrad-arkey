# Prefill AMD GEMM — The ds_load Clue (found, scoped, bounded to +3%)

Date: 2026-06-20

## The clue (from disassembly comparison)

Diffing the **hot loop** of our kernel vs the selected Tensile `.co` (both compute 32 WMMA per body, same
work) surfaced two concrete, reliable (static-disasm) differences the earlier PMC missed:

| op (per 32-WMMA loop body) | Tensile | ours |
|---|---:|---:|
| **ds_load_b128** | **16** | **32** |
| s_waitcnt | 22 | 6 |
| ds_store | 16 | 8 |

1. **Ours issues 2× the LDS reads.** Our WMMA operand is 8 VGPR (16 fp16/lane, the *replicated* layout — lanes
   16–31 duplicate 0–15), so each fragment needs **2** `ds_load_b128`. Tensile (`LRVW16`, `UMLDSB1`, `LDL1`)
   uses a layout that reads each fragment with **1** — half the LDS-read instructions.
2. **Tensile uses 4× finer waitcnt** (22 vs 6) — `SIA1` interleaved scheduling vs our coarse phase waits.

This reopened the "pure confound" conclusion: there *is* a real hot-loop inefficiency (we read LDS twice as
much). So we scoped it.

## The diagnostic — bound the lever before a risky rewrite

A correct fix (Tensile's non-replicated 4-VGPR operand layout) is a deep, uncertain rewrite. So first we
measured the **ceiling** of the lever with `DSHALF`: a variant that issues *half* the `ds_load_b128` (drops the
second half of each fragment — **incorrect output**, timing only). If throughput jumps, ds_load count is the
residual; if not, it's hidden like VALU.

| variant (pinned) | TFLOPS |
|---|---:|
| ds_full (32 loads, correct) | 60.3 |
| ds_half (16 loads, **incorrect**) | 62.3 |
| **ceiling of the ds_load lever** | **+3% (1.03×)** |

**Halving the LDS reads buys only +3%.** The 2× ds_load is real but **mostly hidden** behind compute/occupancy
— the same story as VALU. It is a small contributor, not the dominant residual.

## What this completes

Every candidate for the residual is now measured and bounded:

| candidate | effect when fixed/removed |
|---|---|
| prefetch (A / full A+B PLR) | neutral |
| L2 locality (WGM8) | ours already *better* |
| occupancy | tuned (wg2) |
| bank conflicts | tuned (PAD16, ~0) |
| VALU / address arith | matched to Tensile → **neutral** |
| **ds_load 2×** | **+3% ceiling** (mostly hidden) |
| finer waitcnt (SIA1) | not separately measured; ≤ the small remainder |

**There is no single dominant residual lever.** The ~few-% gap is a *sum of small, mostly-hidden
inefficiencies* (ds_load ~3%, scheduling, etc.), each individually tiny, on top of the measurement/work
confounds (Tensile `beta=true` reads C, col-major, grid 4×96, foreign-`.co` launch, clock/power). At a stable
pinned clock the dependency-free kernel is ~60–62 and Tensile ~62 — the residual is small and has no closeable
single cause.

## Decision

The biggest remaining lever (ds_load) is worth **at most +3%**, and the correct fix requires Tensile's
non-replicated WMMA operand layout — a deep rewrite with uncertain RDNA3 feasibility (the standard
`v_wmma_f32_16x16x16_f16` uses the replicated 8-VGPR operand). **Not worth the risk for +3%.** The honest
landing: the dependency-free kernel is at ~Tensile parity, the residual is identified, bounded (+3% biggest
piece), and dominated by small hidden costs + confound — not a single fixable deficit.

`DSHALF` is kept as a default-off diagnostic flag (throughput-sensitivity probe only; the output is incorrect
when enabled).

## Verdict

Clue found and **bounded**: ours issues 2× the LDS reads (replicated WMMA operand layout) + coarser waitcnt,
but halving ds_load yields only +3% (mostly hidden). No dominant residual lever exists; the dependency-free
GEMM rests at ~Tensile parity with a small, identified, sub-3%-per-lever, confound-mixed remainder.
