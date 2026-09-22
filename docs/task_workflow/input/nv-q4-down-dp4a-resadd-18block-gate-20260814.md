# NV Q4 FFN-down DP4A resadd 18-block gate - topology PASS, logits drift, wall +0.50% (2026-08-14)

Date: 2026-08-14. Target: RTX 5090, native `DEV=NV`, sm_120, driver 595.84. Branch
`nvidia-bringup-20260731`, HEAD `0f76ca13a`. GPU idle at start; no process was killed.
Measurement-only; nothing is promoted.

## Verdict

The DP4A resadd route scales to all 18 Q4 FFN-down blocks with a **clean topology** and an
**identical greedy token stream**, but it is **not bit-identical** to the installed scalar route:
the Q8_1 + DP4A arithmetic accumulates `relative_l2 ~= 3e-3` and `max_abs ~= 0.077` over the full
stack, which trips the conservative single-block semantic gate. The wall bracket is a real
**WALL_PASS at +0.50%** (193.92 -> 194.88 tok/s, `-25.6 us/token`).

## Gate 1: qualify, all 18 blocks (FAIL_CLOSED on logit drift, topology clean)

`--mode qualify --indices 4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,28,29 --count 8`.

| field | value |
| --- | --- |
| topology_pass | `true` |
| changed program counts | `_epi_ffnresadd -18`, `q8_1_llama_provider_12288 +18`, `q4k_q8_mmvq_direct_4096_12288_epi_ffnresadd +18` |
| removed materialize programs | `{}` |
| tokens_equal / argmax_equal | `true` / `true` |
| top-k sets equal / order equal | `true` / `false` |
| relative_l2 | `0.002982310` (gate `0.001`) |
| max_abs | `0.076912880` (historical gate `0.01`) |
| semantic_pass | `false` |

The topology is exactly the expected 3-entry delta with zero transport kernels, confirming the
two folds land on every block. The semantic failure is pure accumulation: the single-block lease
measured `relative_l2 5.76e-4`, and 18 blocks compound it to `2.98e-3`. Tokens and argmax stay
identical; only the near-tie top-k ordering shifts.

## Why the logits drift (llama-equivalent, not scalar-equivalent)

The candidate is the llama `quantize_q8_1` + DP4A `mmvq` decomposition (`emit_q8_provider` uses the
live-llama `x/d` roundf spelling; the consumer uses llama's `dp4a` correction). The installed
scalar Q4 GEMV is a different rounding path. Both approximate the same fp32 down-projection, but
they are not bitwise equal, so promoting the DP4A route changes the model's numerics toward llama's
by construction. The greedy decode stream remained identical over the measured 20-token window
(all arm hashes equal), but a longer or sampled run can diverge.

## Gate 2: settled reverse wall bracket, all 18 blocks (WALL_PASS)

`--mode timing --indices <18 blocks> --count 20 --reps 3`. All arms share token stream hash
`6700c07ac628c8d6758a1b16144602fe55b82feae49741c4a3133ab10a091aa6`.

| arm | median ms/token | tok/s |
| --- | ---: | ---: |
| control bracket median | `5.156860425` | `193.92` |
| candidate (18 blocks leased) | `5.131285200` | `194.88` |
| candidate minus control | `-0.025575225` ms | `+0.498%` |

## What this tells the ledger

The Q4 FFN-down GEMV-core deficit is `+97.0 us` (`llama 346.2 vs native 443.2`, 18 blocks). The
full 18-block DP4A route recovers only `~25.6 us/token` of that, i.e. `~1.4 us/block`, not the
`~5.4 us/block` the raw per-shape gap implies. The residual per-block cost is the separate
`q8_1_llama_provider_12288` node plus the four-warp consumer's in-loop inflation over llama's
single-pass `mmvq`. This matches the standing conclusion: per-shape GEMV/DP4A work is near its
ceiling, and the structural lever remains the support-kernel fusion rows (residual/plumbing,
reduce-output epilogue, vocab aux), not more Q4-down tuning.

## Evidence

- qualify JSON: `/tmp/q4-down-resadd-qualify-18blk-20260814.json`
- bracket JSON: `/tmp/q4-down-resadd-bracket-18blk-20260814.json`
