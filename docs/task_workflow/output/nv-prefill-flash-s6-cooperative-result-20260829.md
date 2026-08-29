# Flash S6 cooperative score probe — 2026-08-29

## Scope

Research-only standalone Flash S6 score/reduction substrate. The candidate keeps the
existing six-part `(m,l,acc[128])` ABI and combine kernel. It changes only the score
dot product: one warp computes each 128-wide Q·K dot using lane-strided loads and
warp shuffle reduction; the other warps reuse the score while all 128 threads produce
the value accumulation.

## Results

All rows use the live-shaped fixture (`Hq=32,Hkv=8,Hd=128,T=512,KV=768,parts=6`),
causal masking, exact input-readonly checks, and the independent FP16 attention oracle.

| shape | variant | score us | combine us | max abs | verdict |
|---|---:|---:|---:|---:|---|
| 32×1×6 | naive | 14.944 | 2.432 | 0 | PASS |
| 32×1×6 | cooperative | 11.296 | 1.952 | 0 | PASS |
| 32×8×6 | naive | 53.248 | 1.888 | 7.63e-6 | PASS |
| 32×8×6 | cooperative | 23.104 | 1.920 | 7.63e-6 | PASS |
| 32×512×6 | naive | 52,302.240 | 10.272 | 7.63e-6 | PASS |
| 32×512×6 | cooperative | 4,422.016 | 9.888 | 7.63e-6 | PASS |

The full service row is an 11.8× score-kernel speedup (52.302 ms→4.422 ms), while
the six-part ABI and combine remain intact. The cooperative candidate uses 2,176 B
shared memory versus 1,024 B for the naive body; reported registers remain 40.

## Decision

PASS and retain as the next Flash S6 mechanism candidate. This is standalone evidence
only; no production/model integration is authorized by this gate. The next required
gate is comparison against the installed Flash body/population and, if favorable,
live graph-bound integration with the same oracle and service accounting.

Evidence JSON is in `docs/task_workflow/evidence/nv-prefill-flash-s6-coop-20260829/`.
