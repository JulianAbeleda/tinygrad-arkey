# NV Q6 Region A subtest record (2026-08-31)

## Question

Does the Region A cost come primarily from global input loads, or from Q6
decode and shared publication? The existing qualifier was run at R9 on the
same representative shape `(M,N,K)=(512,4096,12288)`, with 128 CTAs and 48
K256 blocks per CTA.

## Results

| Arm | Live-sink median | Min | Max | Registers | Shared | Spills |
|---|---:|---:|---:|---:|---:|---:|
| Global Q6/Q8 loads only | 175.264 us | 174.784 us | 177.184 us | 40 | 0 B | 0 |
| Q6/Q8 decode + shared publication | 226.752 us | 225.760 us | 227.744 us | 64 | 21,504 B | 0 |

The diagnostic bulk-readback arms were 184.096 us and 723.296 us,
respectively, but they perform materially different output traffic and are
not used for attribution.

## Structural checks

Both arms retained 36 Q6 global loads and 12 Q8 global loads. The publication
arm retained 80 shared-publication stores in the live-sink source and two
barriers; the loads-only arm retained no shared stores or barriers. Neither
arm emitted IMMA or local-memory traffic. Readback was finite and nonzero in
both arms.

## Interpretation

Within this diagnostic construction, adding decode and publication costs
approximately `51.5 us` (`226.752 - 175.264`) over the same live-sink load
control. This is large enough to justify producer optimization, but it is not
a claim that the costs add linearly inside the full pipelined kernel.

The test proves that raw global loads alone do not account for the Region A
cost. The next causal arms should split the added work into decode-only and
publication-only while keeping the same live-sink and launch geometry.

Evidence:

- `docs/task_workflow/evidence/nv-q6-region-a-20260831/loads.json`
- `docs/task_workflow/evidence/nv-q6-region-a-20260831/full.json`
