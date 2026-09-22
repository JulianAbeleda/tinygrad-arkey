# Q6 broad-route binary A/B verdict

Target is the current persistent broad full main (`285.600 us` actual), with
the normalized llama body as the instruction oracle. Both experiments used
the existing full-shape harness under `flock /tmp/nv-q6-oracle-gpu.lock` and
created fresh cubin/SASS/JSON artifacts in
`evidence/nv-q6-binary-ab-20260831/`.

## Results

| arm | exactness | main min / median | regs | stack | LDL/STL | SASS observations | verdict |
|---|---|---:|---:|---:|---:|---|---|
| lifetime separation (`straightline_k256`) | allclose; strict segmented max `6.866e-4` | `356.032 / 356.896 us` | 255 | 72 B | 20 / 20 | 256 IMMA, 192 LDG, 8 BAR | reject |
| contiguous publication (`live_publication`) | allclose; strict segmented max `6.866e-4` | `325.440 / 325.984 us` | 254 | 0 B | 0 / 0 | 384 IMMA, 282 LDG, 12 BAR | reject |

The lifetime arm is `+70.432 us` versus the 285.600-us target and the
contiguous-publication arm is `+39.840 us`. Neither is an investment candidate.
The lifetime arm also does not satisfy a strict bitwise/`1e-6` segmented
criterion, despite passing the harness's declared `rtol=2e-5, atol=2e-3`
allclose criterion.

## Interpretation

The producer/consumer lifetime split did not improve the broad wrapper. It
reduced the generated SASS stack to 72 B but retained 20 local operations and
was substantially slower. This is a direct causal no-go, not a source-level
prediction.

The contiguous-publication arm removed local traffic and reduced register
allocation by one, but added publication/control work (12 barriers and 384
IMMA in the emitted entry) and remained 39.840 us slower than the actual
broad target. Removing local traffic is therefore already falsified as the
dominant lever on this route.

The next work should target the normalized producer excess identified by the
SASS audit: runtime-loop control and repeated Q6 publisher address/decode
work. Do not invest in either lifetime separation or contiguous publication
without a new binary mechanism that changes those counts and passes strict
segmented exactness.
