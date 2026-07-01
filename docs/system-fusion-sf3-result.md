# System Fusion SF3 — Experiment Result

Date: 2026-07-01. Candidate: decode_silu_gate_fusion. Flag: DECODE_FUSE_SILU_GATE (default-off).

## Correctness

| metric | value |
|--------|-------|
| method | eager (JIT=0), 3 decode steps |
| rel_rmse | 0.00e+00 |
| pearson | 1.000000 |
| top1_match | True (all steps) |
| verdict | **CORRECTNESS_PASS** |

Logit vectors are exactly identical between flag=0 and flag=1. Removing `.contiguous()` changes kernel scheduling only, not the computation.

## W==D measurement

Method: qk_decode_runtime_overhead.py (W=real decode with .item()/token, JIT=1, NMEAS=40).

| ctx | baseline (flag=0) | fused (flag=1) | delta | delta % |
|-----|-------------------|----------------|-------|---------|
| 128 | 52.3 tok/s | 52.4 tok/s | +0.1 | +0.2% |
| 512F | 50.0 tok/s | 50.4 tok/s | +0.4 | +0.8% |

progs_per_step: 7 (same in both runs at ctx512). The JIT graph structure is unchanged.

## Verdict: SF3_LOW_AMDAHL_NO_MOVEMENT

ctx512 delta: +0.4 tok/s, below the 0.5 tok/s significance threshold.

The improvement is within noise (40-step NMEAS; typical run-to-run variation for this harness is 0.5-1 tok/s). The dominant cost of 14B decode is Q4K GEMV (43%), which is HBM-bound at ~400 GB/s. Removing 40 elementwise kernel launches per step and ~1.4 MB of silu intermediate buffer bandwidth does not move the HBM bottleneck.

## Why this is the expected outcome

The silu_gate pair (E_136_32_4 + E_136_32_4n1) accounts for 1.26% of GPU time at ctx512. Even assuming perfect elimination, 1.26% → ~0.6 tok/s at the current 50 tok/s baseline. The actual improvement must be smaller than this because:
1. The kernel launches overlap with the preceding GEMV (GPU pipeline hides small launch overhead)
2. The intermediate buffer read/write is small (17408 f16 × 40 layers = 1.4 MB) vs. the dominant GEMV bandwidth
3. The JIT progs count is unchanged — the JIT is already amortizing launch overhead effectively

## Promotion decision: DEFER_NOT_DEFAULT_ON

- Correctness: proven
- Code change: correct, kept in repo behind DECODE_FUSE_SILU_GATE=0 (default-off)
- W==D: no detectable improvement at this granularity
- do_not_retry: False — the root cause is real; re-open when a broader scheduler pass aggregates the full elementwise bucket

## Reopen condition

Reopen `decode_silu_gate_fusion` as part of a broader system-fusion scheduler pass that fuses ALL REACHABLE_NOW elementwise groups simultaneously:
- silu_gate: E_136_32_4 + E_136_32_4n1 (1.26%)
- rmsnorm_scale × 2: E_40_32_4 + E_40_32_4n2 (1.28%)
- residual_adds: E_40_32_4n1 + E_40_32_4n3 (0.90%)
- qk_norm_scale: E_5_2_2_16_4_4n1 (1.46%)

Aggregate Amdahl: ~4.9% → ~2.4 tok/s at ctx512. At that scale the improvement should be measurable. The individual pieces are confirmed REACHABLE_NOW; the blocking factor is tooling for a multi-group fusion pass.
