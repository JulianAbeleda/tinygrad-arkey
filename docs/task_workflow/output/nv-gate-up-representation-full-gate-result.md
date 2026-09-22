# Gate/up representation full-gate result

## Outcome

The representation lane reaches a numerical-quality wall before production
admission. The fused kernel mechanism is real and fast, but no byte-reducing
post-hoc contract tested against the installed Q4_K checkpoint satisfies the
one-layer recurrent-logit gate. No route was promoted and the strict endpoint
remains 4065.897 us/token, or 245.948 tok/s.

## Kernel gates

| Gate | Correctness | Control | Candidate | Recovery | Verdict |
|---|---|---:|---:|---:|---|
| One 12288x4096 U4Z8 projection, continuous rotated cold | three finite legal fixtures; independent oracle | 18.898 us | 17.820 us | 1.078 us | pass |
| Conservative pair: two U4Z8 projections plus finish | qualified component projections; composition smoke | 40.039 us | 38.740 us | 1.299 us/layer | pass |
| Fused U4Z8 pair, independent rings, R9 | finite nonzero composition; zero values outside tolerance | 40.232 us | 36.607 us | 3.625 us/layer | pass |

The first fused smoke that reported about 20 us was rejected before admission:
the conservative arm had just warmed the same compressed ring. The authority
R9 uses an independent 16-copy ring for the fused arm. The nonzero gate also
caught and corrected shared shuffle-staging ownership between gate and up.

The fused primitive exposes 130.5 us/token over 36 layers. If it could be
qualified and transferred perfectly, its isolated arithmetic ceiling would be
about 254 tok/s. This is not a token-rate claim.

## Quality gates

All rows below change only block 0 gate/up and compare three recurrent full-logit
rows with a fresh control. The admission limit is stacked relative L2 <= 0.001,
with finite logits and preserved token decisions.

| Contract | Bytes/block | Weight change | Stack relL2 | Recurrent tokens | Verdict |
|---|---:|---|---:|---|---|
| U4Z8 symmetric group-64 | 136 | requantizes weights; removes zero point | severe: first row 0.0775, later 0.43--0.47 | diverges after first row | stop |
| U4 affine group-64 | 140 | scale plus compact zero point | 0.002808 | preserved | stop |
| Q4 codes + 4-bit scale/min metadata | 140 | preserves codes and block scales | 0.001459 | preserved | stop |
| Q4 codes + 5-bit scale/min metadata | 142 | preserves codes and block scales | 0.002094 | preserved | stop |

The five-bit result being worse than four-bit in recurrent logits is not treated
as a measurement anomaly: weight-error magnitude is not a monotonic proxy for
network output error. The recurrent full-logit comparison remains authority.

## Gate disposition

1. Fused primitive performance: **passed**.
2. Nonzero fused semantic composition: **passed**.
3. Post-hoc installed-model quality: **failed at the minimum one-layer dose**.
4. Predecessor-conditioned production transfer: **not admissible after quality failure**.
5. Production route and model artifact: **not implemented**.
6. Strict endpoint bracket: **not run; installed endpoint unchanged**.

This closes post-hoc conversion of the installed Q4_K checkpoint. It does not
prove that a training-aware or higher-precision-source artifact cannot work.
Reopening requires a higher-precision source plus calibration or quantization-
aware optimization; another kernel spelling cannot repair missing weight
information.

## Ledger consequence

There is no promotable low-hanging token win in this lane today. The physical
kernel opportunity is large, but it is separated from the installed endpoint
by an artifact-quality wall. It must remain in the `proven primitive, blocked
representation` bucket and contributes zero booked microseconds and zero booked
tokens per second.
