# Q4-down C0 aggregate report (2026-08-29)

## Decision: PASS aggregate / STOP C0.3

The deterministic aggregate covers all 18 authorized Q4-down roles (ordinals
0 through 17) from the existing completed diagnostic-mirror chunks. No role was
rerun. The stored per-role allclose evidence passes at `rtol=0.02, atol=3.0`;
maximum observed error is `2.7985`.

Aggregate artifact:
`docs/task_workflow/evidence/nv-prefill-q4down-staged-oracle-20260829/c0-aggregate-18.json`

The scoped stage-localization evidence remains STOP: producer, weight, dot,
correction, and epilogue intermediates are unavailable through the production
ABI. No unique actionable failing boundary is established, so C0.3 is not
authorized and no primitive change is made. Source: `result.json` in the
`nv-prefill-q4down-staged-oracle-20260829` evidence directory.
