# NV 227 push: Q versus K/V producer concurrency result

Date: 2026-08-24

## Decision

The physical overlap exists, but it is not a remaining production lever.
Tinygrad already places the tested Q and K/V branches on separate compute
GPFIFOs. Reapplying that placement is a structural no-op, and submitting the
auxiliary queue first regresses the production wall. No latency is booked and
the installed endpoint remains `4515.395719 us/token = 221.464532 tok/s`.

## Cold physical gates

Both gates time the complete shared-Q8 RMSNorm/Q8 provider through the Q/K/V
join and evict projection weights before every sample.

| gate | serial control | Q || K/V | recovery/layer | exact |
| --- | ---: | ---: | ---: | --- |
| rendered CUDA streams, 256 MiB eviction, 80x9 | 18.428001 us | 16.311600 us | 2.116401 us | 0/6144 mismatches |
| live cubins/buffers/VAs, HCQ, 2x24 steady samples | 18.640 us | 16.592 us | 2.048 us | all four buffers |

The live gate captured the exact consecutive installed chain at sequence
positions 1509-1511. Its cubins used the real immutable weight allocations;
the provider-to-Q and provider-to-K/V VA identities were both proved. The
isolated nine-layer projection was `18.432 us/token`, or 222.372 tok/s if it
were wholly additive. Extending the same 2.048 us/layer to all 36 layers was
an unvalidated `73.728 us/token` ceiling, or 225.141 tok/s.

## Production adjudication

The construction census invalidated the assumption that the shared pair still
needed a placement route:

```text
17/17 shared-Q8 providers              queue 0
17/17 shared-Q8 Q projections          queue 0
 9/9  shared-Q8 Q4/Q4 K/V pairs        queue 1
```

Those counts are identical in the installed control and targeted candidate.
The apparent midpoint result (`4.496621 -> 4.490118 ms/token`) therefore has
no causal topology change; the candidate is only 0.383 us below the closing
control. It is classified `NO_OP_TOPOLOGY`, not a wall pass.

The aux-first replay test preserves the installed graph and reverses only the
two compute-queue submit calls:

| arm | ms/token |
| --- | ---: |
| control A, q0 first | 4.489211 |
| candidate, q1 first | 4.491439 |
| control C, q0 first | 4.477316 |
| control midpoint | 4.483263 |

Candidate regresses the midpoint by `8.176 us/token` and loses to both
controls. All arms have token stream SHA-256
`f25083e5d0a754131283b40c03f52e688fee9f175bea7ae106805e7d628d7905`.

## Interpretation

The live microgate proves the hardware can overlap the branches when their
small sequence is submitted in isolation. It does not prove that its recovery
can be added to the full-token endpoint. Production already expresses the
same dependency fork, while graph-wide arbitration, contention, and the other
ready work consume the isolated opportunity. Neither an extra placement rule
nor K/V-first submission improves the wall.

Verdict: `NO_GO_ALREADY_PLACED_AUX_FIRST_WALL_NEGATIVE`.

Evidence: `docs/task_workflow/evidence/nv-227-qkv-concurrency-20260824/`.
