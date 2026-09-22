# NV exact streaming service-rate campaign

## Outcome

No token-wall improvement is booked.  The campaign found that the earlier
`~270 tok/s` measured-rate ceiling is not a claimable exact-optimization
ceiling: it normalizes every projection to a long-body steady-state rate and
therefore deletes the latency ramp paid by small Q/O/K/V streams.

The installed projection families fit the following size-aware model:

```text
body time = fixed stream/launch ramp + payload / asymptotic stream rate
```

A least-squares fit across gate/up, down, Q, O, and K/V gives a fixed term of
about `3.27 us/launch` and an asymptotic rate of about `1.75 TB/s`.  Across 180
layer projection launches, the fixed term is about `589 us/token`—nearly all
of the old model's `653 us/token` supposed headroom.  The fit reproduces the
observed projection mass (`2966.656 us`) at the displayed precision.

This does not establish a hard hardware floor.  It changes the accounting:
the low aggregate GB/s of a small body is primarily a size/ramp effect, not
evidence that its steady-state issue/dequant loop has hundreds of recoverable
microseconds.

## Causal tests

The first exact discriminator packed multiple independent Q4 output-row warps
into one CTA while retaining the installed vector loads, one warp per row,
lane ownership, arithmetic order, and bytes.  All variants were elementwise
bit-exact, but all lost to the installed one-warp CTA in the isolated gate:

| rows per CTA | installed vector | candidate | result |
| ---: | ---: | ---: | --- |
| 2 | 4.897 us | 5.178 us | no-go |
| 4 | 4.882 us | 5.211 us | no-go |
| 8 | 4.902 us | 5.649 us | no-go |

The Q6-V follow-ups closed both obvious scheduling axes.  Packed lane ownership
was slightly faster but changed fp32 association and failed elementwise bit
equality.  Exact two-block unrolling retained bit equality but was flat/slower
in the amortized primitive bracket.  Neither qualified for production or a
whole-token wall test.

Rejected spellings were removed from production code.  The installed route and
token endpoint are unchanged.

## What the wall means now

The old uniform-rate construction remains useful as a mathematical answer to
“what if every byte were charged at the long-body rate,” but not as the next
optimization budget.  A size-aware reconstruction returns approximately the
current device union once the measured non-weight mass is retained.  Therefore
we cannot honestly say that another generic dequant scheduler should recover
the distance to 270 tok/s.

The remaining ways to move the wall are now narrower:

1. Remove an entire physical stream/ramp while preserving its service rate.
   Gate/up fusion and QKV producer work already exercised the obvious exact
   instances; the latter failed installed whole-wall composition.
2. Reduce numerical weight bytes under an explicit quality contract.  This
   changes the variable payload term and is now the largest unclosed lever.
3. Reduce the retained non-weight device mass with a new causal construction,
   not by re-opening projection rate based only on aggregate GB/s.

## Decision

`UNIFORM_RATE_CEILING_REJECTED_SIZE_AWARE_RAMP_WALL_PROMOTED`

Evidence:

- `docs/task_workflow/evidence/nv-exact-streaming-service-rate/ceiling-correction.json`
- `docs/task_workflow/evidence/nv-exact-streaming-service-rate/q4-r2-4096x4096-hot.json`
- `docs/task_workflow/evidence/nv-exact-streaming-service-rate/q4-r4-4096x4096-hot-installed-control.json`
- `docs/task_workflow/evidence/nv-exact-streaming-service-rate/q4-r8-4096x4096-hot.json`

