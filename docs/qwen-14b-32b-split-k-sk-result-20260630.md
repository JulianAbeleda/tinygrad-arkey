# Generated Split-K Q4_K Decode (SK0-SK4A) — result

Date: 2026-06-30

Status: split-K REFUTED for the FFN as the 14B/32B decode lever, and the target is REDIRECTED — the Q4_K FFN GEMV
is already faster than the full-model decode average, so it is not the bottleneck. Route family not globally
refuted; the measured bottleneck moved. Scope: `docs/qwen-14b-32b-generated-split-k-q4k-decode-scope-20260630.md`.
Hardware: gfx1100. Harness: `extra/qk_large_shape_split_k_14b_discovery.py` (synced TinyJit min-of-bursts, role-local).

## SK0 — Amdahl precheck (14B ffn_down 17408→5120, k_blocks=68)

Direct G3 serial depth = ceil(68/4) = 17. Split bounds: parts 2→9 (1.9×), 4→5 (3.4×), 8→3 (5.7×). Theoretically
there is serial-depth room — IF the kernel were latency/serial-bound.

## SK4A — 14B ffn_down role-local microbench (the decisive test)

Synced TinyJit min-of-bursts (the first, un-JIT'd pass was launch-overhead-contaminated at ~2 GB/s and discarded):

| candidate | ms | GB/s | rel_rmse | note |
|---|---|---|---|---|
| **direct G3** (shipped route) | **0.125** | **355.4** | 3.6e-04 | the generated wave kernel |
| split_k_1 (partial substrate) | 0.846 | 52.7 | 3.6e-04 | |
| split_k_2 | 0.837 | 53.3 | 3.6e-04 | |
| split_k_4 | 0.819 | 54.4 | 3.6e-04 | |
| split_k_8 | 0.768 | 58.0 | 2.6e-01 | correctness breaks (masked-tail bug) |

`SK4A_REFUTED_14B_SPLIT_K_NO_ROLE_LOCAL_WIN`.

Two findings:

1. **Split-K does not help.** The available partial substrate (`q4k_gemv_packed_load_partial_kernel`) is **7× slower**
   than the direct G3 wave kernel, and increasing `parts` (2→4→8) barely moves it (53→58 GB/s). It uses an 8-way
   range reduce, not the 32-lane wave — it does not exploit the wave the way G3 does. More K-splits give no benefit,
   consistent with the FFN GEMV being memory-access bound, not serial-latency bound (the ~5120 output-row workgroups
   already hide the 17-deep serial chain).

2. **The FFN is not the bottleneck — TARGET REDIRECTED.** The direct G3 ffn_down kernel is **355 GB/s role-local**,
   which is *above* the full-model 14B decode of **243 GB/s** (27 tok/s × ~9GB). If the Q4_K FFN were the drag, the
   full model could not exceed the FFN's own rate — but the FFN kernel is already faster than the decode average. So
   the 14B/32B gap to llama (585 GB/s) is dominated by **non-Q4_K-FFN work**: the Q6_K lm_head (5120→151936, the
   single largest GEMV), attention, and inter-kernel scheduling across the ~280 decode kernels. Building a
   shape-tuned or split-K FFN kernel would not close the full-model gap, because the FFN is not where the time goes.

## Why this stops the split-K scope

The scope's premise was "14B/32B ffn_down has serial depth 17/25, so split-K K-parallelism is the lever." The
measurement refutes it: the FFN GEMV is already efficient (355 GB/s) and above the full-model average, and split-K
gives no role-local win. Per the scope's own stop rule ("If split-K loses role-local, the large-shape gap is not K
parallelism alone"), the honest outcome is REFUTE + redirect, not force a generated split-K route.

Caveat (fairness): I measured split-K via the existing `q4k_gemv_packed_load_partial_kernel` substrate, not a new
SK2 G3-wave-based split-K kernel. A G3-based split-K might beat the partial substrate role-local — but it cannot
close the full-model gap, because the FFN (355 GB/s) is already above the 243 GB/s full-model rate. The target is
elsewhere, so building the SK2 kernel is not justified.

## Frontier ledger

| field | value |
|---|---|
| candidate_id | `split_k_partial_substrate_14b_ffn_down` |
| profile_id | qwen3-14b Q4_K decode gfx1100 |
| role | ffn_down 17408→5120 |
| status | `REFUTED_SPLIT_K_NO_ROLE_LOCAL_WIN` + `TARGET_CHANGED` |
| measured_delta | split-K 7× slower than direct G3 role-local; parts 2/4/8 ~flat |
| dominant_failed_row | FFN GEMV is memory-bound at 355 GB/s (already > full-model 243); serial depth is not the limiter |
| missing_axis_or_capability | none for FFN — target moved off Q4_K FFN K-parallelism |
| next_bucket | per-role full-decode wall-share attribution: Q6_K lm_head (5120→151936), attention, inter-kernel overhead |
| reopen_condition | if a per-role profile shows a Q4_K FFN role below ~250 GB/s in-model AND latency-bound (low occupancy), revisit a G3-wave split-K for that specific role |
| replay_command | `DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_large_shape_split_k_14b_discovery.py` |

## Bottom line

The Q1432 route-binding win (+8-9%, shipped default-off) remains the correct, promotable FFN result. The path from
14B/32B ~42% → 80% of llama is **not** FFN topology or split-K — the FFN is already efficient. The next honest lever
is a **per-role wall-share attribution of the full decode** to quantify the Q6_K lm_head / attention / scheduling
overhead, which is where the time actually is.
