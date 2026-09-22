# NVIDIA pp512 Flash S6 vector substrate gates — 2026-08-29

The initial GPU fault was localized to the wrapper build, not the arithmetic:
the progressive ABI probe passed no-op, sentinel, scalar load, vector load,
score, reduction, and partition-strided output stages with random FP16 inputs.
The wrapper was corrected to compile score and combine as separate NVRTC
programs and to use the non-overlapping `(m,l,acc[128])` layout.

## Measured gates

| gate | score us | combine us | oracle | finite | inputs readonly |
|---|---:|---:|---|---|---|
| 1 head × 1 query × 1 part | 14.272 | — | primitive-only | yes | — |
| 1 head × 1 query × 6 parts | 14.176 | 1.472 | PASS, max abs 0 | yes | yes |
| 32 heads × 1 query × 6 parts | 14.592 | 1.696 | PASS, max abs 0 | yes | — |
| 32 heads × 8 queries × 6 parts | 50.016 | 1.568 | PASS, max abs 7.63e-6 | yes | — |
| 32 heads × 32 queries × 6 parts | 279.776 | 1.920 | PASS, max abs 7.63e-6 | yes | — |
| 32 heads × 512 queries × 6 parts | 52254.368 | 10.048 | PASS, max abs 7.63e-6 | yes | yes |

The full extent is numerically correct and readonly-safe, but this naive
scalar-dot implementation takes 52.254 ms for score plus 0.010 ms combine.
It is therefore a substrate correctness result only, not a performance
candidate. No model wiring or promotion is justified. The next optimization
must replace the redundant per-thread K dot-product work with a real vector/
cooperative tile while preserving this independent oracle and six-part ABI.
