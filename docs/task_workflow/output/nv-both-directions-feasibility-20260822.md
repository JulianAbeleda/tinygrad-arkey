# NV both-directions feasibility probe (2026-08-22)

## Question

Is either remaining direction actually possible: a stronger PDL launch-ahead
primitive, or a single-pass fusion that avoids M1's two named failure modes?
This is a feasibility check, not an endpoint bracket.

## Fusion: compute the FFN norm once, not twice

M1 lost `+82.08 us/token` because the norm epilogue was applied per packed-Q4
load, so it re-executed once for the gate dot and once for the up dot, and the
consumer streamed fp32. The probe builds `q4k_w1w3_norm_once16_12288_4096`,
which computes the normalized fp16 activation exactly once per element and
reuses that value for both dots.

| arm | cudaEvent median us/launch | ratio to control |
| --- | ---: | ---: |
| control `fused16` (no norm) | 22.6299 | 1.000 |
| M1 `rms_affine16` (norm twice) | 33.4309 | 1.477 |
| `norm_once16` (norm once) | 33.4499 | 1.478 |

The two fused arms are bitwise identical (`norm_once_vs_m1_bitwise_identical=1`,
observed), and they are equally slow. Removing the redundant second norm did
not help: the cost is not the duplicate epilogue, it is the extra arithmetic
and register pressure of doing the norm inside the memory-bound GEMV at all.
Both fused bodies use 80 registers versus 74 for the no-norm control.

Verdict: **not viable as constructed**. A once-only norm fold is numerically
correct but still costs `~+10.8 us/launch` over the no-norm fused gate/up,
while the separate epilogue it removes is `~2.3 us`. The M1 direction is not
recoverable by removing its redundancy.

## PDL: the two binding missing primitives

Fresh matched-grid native QMD probes under the same HEAD:

| primitive | verdict | meaning |
| --- | --- | --- |
| two producers, one consumer, one latch | named-unavailable | consumer launched `-399.776 us` before producer 2 end with checksums correct; the single latch fired on first arrival, so it is not a true multi-producer merge |
| non-consecutive same-queue A-B-C arming | refuted | consumer waited for the unarmed middle kernel B; the latch does not cross B |

The already-supported rows are one-producer/multi-consumer and replay-flush.
Those do not unblock the `390` multi-producer and `352` adjacency fallback rows
that keep the real decode graph from launching ahead.

Verdict: **not viable on current QMD semantics**. The mechanism works for a
consecutive single-producer pair, but the two primitives needed to cover the
real safe RAW chain are refuted or semantically unproven on this device.

## Comparison

| direction | buildable | measured cost/blocker |
| --- | --- | --- |
| single-pass fusion | yes, bitwise exact | still `~1.48x` the no-norm GEMV; no advantage over M1 |
| PDL launch-ahead | only the consecutive single-producer subset | non-consecutive refuted; multi-producer merge named-unavailable |

Neither direction converts at this head. Fusion is buildable but not
profitable; PDL is partially buildable but missing the exact primitives the
real graph needs. The honest next step is a new construction in one of them:
for fusion, stage the normalized activation into shared memory before the
K-loop and keep occupancy high; for PDL, find a different multi-producer or
cross-kernel ordering primitive. Re-running the same endpoints will not
distinguish them.

## Labels

- `observed`: cudaEvent medians, bitwise equality, register counts, QMD probe
  verdicts and overlap deltas.
- `inferred`: the attribution of the fused cost to arithmetic/register pressure
  rather than the redundant norm.
- `unmeasured`: an endpoint wall for the once-only fusion, and any alternative
  staging or ordering primitive.

## Evidence

- `docs/task_workflow/evidence/nv-w1w3-norm-once-20260822/result.json`
- `docs/task_workflow/evidence/nv-w1w3-norm-once-20260822/pdl_two_producer.json`
- `docs/task_workflow/evidence/nv-w1w3-norm-once-20260822/pdl_non_consecutive.json`
- `docs/task_workflow/evidence/nv-w1w3-norm-once-20260822/sha256.txt`
