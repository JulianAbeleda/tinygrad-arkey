# NV Phase 1 mechanism partition (2026-08-23)

Date: 2026-08-23  
Branch: `nvidia-bringup-20260731`  
HEAD: `6570abc025514273faa100c66b979e531585a1e1`  
GPU: RTX 5090 (`sm_120`), locked SM 2790 / memory 14001 MHz

## Purpose

Partition the current remaining gap into `admission`, `dependency_wait`, and
`useful_body`, using the current Phase 0 ledger, the retained exact-live
P-C5 closures, the Phase 5/6/8 island decompositions, and two new direct
observables (SASS load width and per-kernel timestamp ordering).

## Findings first

1. **[MEASURED] The remaining gap is body/codegen, not admission or wait.**
   The exact-live P-C5 closures are tiny: Q occurrence-0 `+0.008 us/call`,
   K/V `+0.520 us/call` (13.5 us/token), O `+1.032 us/call` (37.2 us/token).
   No row is `DISPATCH_DOMINANT`; clean HCQ dispatch `D` is `<= 0.7 us/call`
   everywhere (Phase 5/6/8).
2. **[MEASURED - NEW] NV quantized GEMVs use scalar loads.** SASS
   disassembly of retained production cubins:

   ```text
   q4k_g3_lanemap_gemv_w1w3fused16_12288_4096 : 32x LDG.E.U16 + 16x LDG.E
   q6k_fp16_mmvq_direct_4096_12288_epi_ffnresadd : 37x LDG.E.U16 + 1x LDG.E
   q6k_gen_coop_151936_4096_inkernel           : 37x LDG.E.U16
   ```

   There are **zero** `LDG.E.64`/`LDG.E.128` vectorized loads in any of the
   three largest GEMV bodies. Weight words are read 32 bits at a time and the
   fp16 activations 16 bits at a time. This is the concrete mechanism behind
   the Phase 5 DRAM-streaming gap (`1.501/1.353/1.357` vs llama
   `1.609/1.449/1.501 TB/s`): tinygrad issues ~4x more narrow load
   instructions than a vectorized llama kernel, lowering memory-level
   parallelism at the same occupancy.
3. **[MEASURED] Flash score is install/cache-bound, not body-bound.** Phase 6:
   exact body `3.840 us` is faster than llama `4.512 us`; the installed
   `6.272 us` command interval is `R = 2.614 us/call` production-conditioned
   residual, and cold NCU replay (`2.13 MB` DRAM, 0.21 waves) matches the
   production interval. Body rewrite has no headroom.
4. **[MEASURED] Flash combine is single-warp body-bound.** Phase 6: body
   `2.304 us` vs llama `1.024 us` (2.25x), `0.79%` SM, `2.08%` warps,
   `0.01` waves. Coarse S splits are already closed; the remaining fix is a
   wider parallel reduction over the 48 splits per head.
5. **[MEASURED - NEW] The vocab tail is serial, not hidden.** Direct
   timestamp ordering from the Phase 0 capture:

   ```text
   q6k_gen_coop_151936_4096_inkernel  8431.75 -> 8431.75 (ends)
   E_1187_32_4 (2nd)                 8431.75 -> 8434.50
   r_32_4_1187                       8434.50 -> 8474.75   (40.064 us)
   r_128_16_8_1187                   8474.75 -> 8485.75   (11.136 us)
   r_16_8                            8485.75 -> 8487.00
   ```

   Every tail kernel starts exactly where its predecessor ends, so the
   `~55 us` tail is real serial wall. This supersedes the 08-19 "hidden mass"
   inference for the current schedule.
6. **[INFERRED] The 4096-norm row and the activation-quant advantage are
   coupled and effectively closed.** `+114.59 - 113.82 = +0.77 us` net; the
   semantic Q/K fusion reduced the combined norm+rope+store delta to
   `+8.48 us`.

## Mechanism map (current, node-sum vs llama PDL-off)

| row | delta us | dominant mechanism | blocked route |
| --- | ---: | --- | --- |
| K/V projections + completion | +118.27 | body: scalar-load DRAM streaming + 26 completion kernels | vectorize loads; fold completion |
| gate/up GEMV | +90.32 | body: scalar-load DRAM streaming (1 warp/row) | vectorize loads; 4-warp w/o cast |
| down GEMV | +79.19 | body: scalar-load DRAM streaming | vectorize loads |
| Q projection + completion | +78.30 | body: scalar-load streaming + completion | vectorize loads |
| flash score | +76.92 | install/cache residual (body faster) | cache/working-set |
| O projection | +75.45 | body: scalar-load streaming | vectorize loads |
| flash combine | +65.44 | body: single-warp latency-bound | wider parallel reduce |
| vocab main + tail | +64.89 | body: single-warp reduction tail (~55 us serial wall) | native fp32+int32 argmax |
| norm/provider/rope | +0.77 / +8.48 | closed | none |

The GEMV rows (K/V, gate/up, down, Q, O) share one generic cause: narrow
scalar global loads. The flash and vocab rows have separate, smaller causes.

## Verdict

`240_UNMEASURED`

The mandatory partition now closes directionally: **body dominates every
remaining row; admission and dependency-wait are measured near zero.** The
single highest-leverage generic fix is vectorized (128-bit) loads in the NV
quantized GEMV emitters, spanning `~440 us/token` of node-sum delta. The next
steps are the Phase 2 candidates below, each requiring a bit-exact microgate
and a fresh reverse wall bracket.

## Evidence

- SASS: retained cubins under
  `docs/task_workflow/evidence/nv-catch-llama-ledger-20260822/phase1/`.
- Timestamps: `docs/task_workflow/evidence/nv-phase0-current-ledger-20260823/production.profile.jsonl`.
- Prior: Phase 5/6/8 island reports and
  `nv-r-predecessor-conditioned-exact-kv-o-result-20260823.md`.

No production, renderer, scheduler, runtime, or route code was changed.
