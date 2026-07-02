# TG-P14 Practical Roofline Audit

Verdict: **PRACTICAL_ROOFLINE_CLOSEOUT**

The practical baseline is the owned HIP attention route measured in the same W==D session. This is the right ceiling
for the promotion question because all non-attention kernels are identical between the generated and owned arms; the
candidate-owned wall delta is the residual attention-route tax.

## Result

| ctx | generated tok/s | owned tok/s | pct of practical | extra ms/token | extra us/layer |
|---:|---:|---:|---:|---:|---:|
| 512 | 109.9 | 111.6 | 98.5% | 0.138 | 3.83 |
| 4096 | 99.8 | 101.0 | 98.8% | 0.122 | 3.39 |

Three-rep floor from the final TG-P14 artifact:

| ctx | reps | worst |
|---:|---|---:|
| 512 | 98.6 / 98.8 / 98.5 | 98.5% |
| 4096 | 98.3 / 98.5 / 98.8 | 98.3% |

So the route clears the 98% practical-ceiling bar at both protected contexts. The remaining gap to owned is only
about **0.12-0.14 ms/token**, or **3-4 us/layer** across 36 layers.

Promotion formula line:

```text
P_worst=98.5%, G_worst=1.5pp, A_worst=0.0pp, basis=owned_same_session_same_scope,
action=promote-if-explicit-replacement-else-closeout
```

## Interpretation

The residual is too small to justify more route tuning for TG-P14:

- The task's >=98% bar is already satisfied.
- The practical owned-route gap is around 1.2-1.5 percentage points.
- To reach 99% at ctx4096, the route needs only about 0.022 ms/token, or 0.61 us/layer.
- To fully equal owned, the maximum available payoff is only about 0.122 ms/token at ctx4096.

The known remaining lever is vectorized PV, but that path is blocked by UOp lowering (`vec4` result extracted into
manual-END REG stores). Given the small maximum payoff, deeper codegen surgery is only worth it if the goal is an
owned-beating speed win, not parity/purity.

## What Is Left

For the original >=98% task bar: **nothing performance-critical remains**.

For BoltBeam promotion: package this as a practical-roofline evidence row and explicitly mark the candidate as a
practical-roofline replacement / handwritten-surface reduction. Without that manifest signal, the right BoltBeam
state is closeout, not default flip.

For an owned-beating win: extend accumulator widening / distinct-slot devectorization to support vectorized PV
`vec-store-to-REG`, then remeasure. Expected ceiling is small: roughly 0.12-0.14 ms/token.
