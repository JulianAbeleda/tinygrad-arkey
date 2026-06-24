# BB-5a.10 PTM-1 — Same-Harness Authority Bridge Result

Date: 2026-06-20

Artifact:
`bench/amd-broad-backend-roadmap/bb5a10_ptm1_same_harness_authority_bridge_result.json`

Command:

```bash
CNT=30 python3 extra/qk_amd_bb5a10_ptm1_same_harness_bridge.py
```

Verdict:
`PASS_PTM1_SAME_HARNESS_AUTHORITY_BRIDGED` · interpretation `GAP_REAL_KERNEL_QUALITY`

## What PTM-1 did

Timed the captured tinygrad authority kernel and the two hand-ASM P8 candidates under **one process, one
clock, interleaved round-robin** — the bridge the reconciliation result (`bb5a10_p8_timing_authority_
reconciliation`) named as `next_action`. The three kernels (authority shape M=512, N=12288, K=4096) all
reach the GPU through the same `run_linear`/`AMDProgram` path, so a single `synchronize()`+`perf_counter`
loop times them identically. The authority is **recompiled from its exact shape + `_prefill_v2_opts`**
(reproduces kernel `r_16_192_32_2_2_2_2_4_32_2_8`, identity confirmed) rather than the prior `_time_program`
timer; the candidates are `build_converted_macro_insts` (DS64 LDS macro) and `build_gemm_pipe(…,4,2)`
(global-direct pipe).

## Result (best-of-30, two independent runs)

| kernel | best TFLOPS | median |
|---|---:|---:|
| `authority_tinygrad` (LLVM WMMA, global-direct) | **52.97** | 50.73 |
| `lds_macro_ds64` (hand-ASM) | **39.86** | 35.66 |
| `global_direct_pipe_T4x2` (hand-ASM) | **29.59** | 28.28 |

- **authority / best-candidate ratio = 1.33×** (run 1: 1.37×, run 2: 1.33× — reproduced).
- **Prior cross-harness ratio = 2.34×** (43.026 via `_time_program` ÷ 18.383 candidate from a separate,
  lower-clock host-wall session).

## The decisive finding

**The 2.34× "tinygrad authority vs hand-ASM candidate" gap was a measurement artifact. Under one clock the
real gap is ~1.33×.** The prior candidate numbers (18.4 / 17.9) came from a lower-clock session, not from a
worse kernel. Two corroborating facts:

1. **Clock is the dominant confound** — `rocm-smi` showed sclk drifting **1429 → ~1073 MHz within a single
   run**. Interleaving is what makes the ratio trustworthy: every kernel is sampled across the same clock
   trajectory. (Absolute TFLOPS here read high — authority 52.97 > the documented ~42 ceiling — because this
   was a high-clock session. Report ratios, not absolutes; the `clock_provenance` field carries the sclk.)
2. **The hand-ASM Route-A candidates sit BELOW tinygrad's own LLVM authority** (39.86 = 75% of 52.97; the
   global-direct pipe = 56%). So the bb5a10 hand-ASM line has not beaten tinygrad's existing compile — fully
   consistent with the prior POWN/Route-A refutation (capped below LLVM-WMMA).

## Implication for the roadmap

The whole bb5a10 hand-ASM P8 effort was implicitly chasing the inflated 2.34× target. PTM-1 corrects it:
the gap to tinygrad's **own** authority is modest (1.33×) and the hand-ASM candidates are already
sub-authority. The remaining real prefill headroom is **authority (≈53 this clock / ~42 nominal) → Tensile
(~66)**, which is the `software_pipelined_k_loop` capability — not another hand-ASM candidate. This drives
PTM-2.

## Next

PTM-2 prefill primitive decision (see
`amd-broad-backend-bb5a10-ptm2-prefill-primitive-decision-result-20260620.md`).
