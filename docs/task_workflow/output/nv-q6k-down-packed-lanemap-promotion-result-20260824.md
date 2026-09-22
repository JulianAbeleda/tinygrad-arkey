# NV Q6_K FFN-down packed-lane promotion

Date: 2026-08-24
GPU: RTX 5090 (`NV`, `sm_120`)
Model: Qwen3-8B-Q4_K_M, single-token decode

## Outcome

The packed-lane Q6_K FFN-down spelling is promoted on NV `sm_120`. It is
bit-exact, raises the cold isolated streaming rate, preserves the full graph
topology and typed output contract, and passes two current composed production
wall brackets. `TINYGRAD_Q6K_FFN_DOWN_PACKED_LANEMAP_DISABLE=1` restores the
prior four-warp spelling.

The earlier wall no-go is superseded. That bracket forced direct greedy and
feedback ping-pong off, so it did not measure the current composed production
path. Its own profiled device result was already positive (`-13.848 us` Q6
body and `-18.438 us` union), making the neutral obsolete-path wall result an
unresolved accounting contradiction rather than a valid closure.

## Causal gates

The production-rendered CUDA microgate compares the prior four-warp fp16
consumer with the packed lane-ownership spelling. The candidate preserves the
Q6 block bytes and arithmetic result while deduplicating load ownership.

| gate | control | candidate | result |
| --- | ---: | ---: | --- |
| hot isolated | 22.228 us | 14.139 us | `-8.090 us/layer` |
| cold, 256 MiB eviction | 40.908 us | 35.223 us | `-5.685 us/layer` |
| cold rate | 1.010 TB/s | 1.172 TB/s | higher rate, same weights |
| numerical comparison | — | — | finite, accepted exact-compatible output |

The current typed-output route remains one kernel per Q6 down projection and
absorbs the residual add. The candidate changes neither the 18-call population
nor the graph boundary contract.

## Composed production wall

Both reverse brackets use fresh A/B/A processes, locked clocks, 32-token
settled windows, and the current direct-greedy plus feedback-ping-pong graph.
The candidate must beat both controls and preserve the token stream.

| bracket | control A | candidate | control C | midpoint recovery | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| d512, reps 9 | 4356.931 us | 4346.557 us | 4357.397 us | 10.607 us | pass |
| d512, reps 15 | 4395.279 us | 4369.211 us | 4403.634 us | 30.246 us | pass |

All arms within each bracket have identical token-stream hashes. The smaller
independent recovery, `10.606875 us/token`, is booked conservatively.

## Ledger booking

Starting from the post-gate/up conservative endpoint:

```text
4462.066719 - 10.606875 = 4451.459844 us/token
throughput                         = 224.645405 tok/s
remaining to 227                  = 46.173500 us/token
                                  = 2.354595 tok/s
```

This is wall credit, not the `102.326 us/token` isolated cold projection.

## Verification

- focused route/model tests: `41 passed`;
- policy is restricted to `NV/sm_120`;
- loader census: default `18/18` packed admissions; explicit rollback `0/18`;
- both composed wall brackets beat both controls with matching token hashes.

Evidence:

- `docs/task_workflow/evidence/nv-227-q6k-fp16-packed-lanemap-20260824/microgate.json`
- `docs/task_workflow/evidence/nv-ranked-parity-campaign-20260824/04-q6-packed-lanemap-composed-wall-r9.json`
- `docs/task_workflow/evidence/nv-ranked-parity-campaign-20260824/04-q6-packed-lanemap-composed-wall-r15.json`

Verdict: `PROMOTED_Q6K_DOWN_PACKED_LANEMAP_BOOK_10_606875_US`.
