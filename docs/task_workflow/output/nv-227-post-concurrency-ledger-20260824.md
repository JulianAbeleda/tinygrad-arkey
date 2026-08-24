# NV 227 ledger after producer-concurrency closure

Date: 2026-08-24

## Endpoint

| item | latency | throughput |
| --- | ---: | ---: |
| conservative installed endpoint | 4.515396 ms/token | 221.465 tok/s |
| 227 target | 4.405286 ms/token | 227.000 tok/s |
| remaining | 110.109 us/token | 5.535 tok/s |

No Q/K/V concurrency recovery is booked. The physical microgate's 225.141
tok/s all-layer number is a non-additive ceiling, not a new checkpoint.

## Closed topology surface

- Ordinary and shared Q4/Q4 K/V launch fusion are installed.
- Shared mixed Q4/Q6 K/V fusion is exact in isolation but wall-negative.
- The installed ready scheduler already assigns the shared Q4/Q4 fork to
  queue 0 for provider/Q and queue 1 for paired K/V.
- Reapplying that placement is a no-op; auxiliary-first replay is 8.176
  us/token slower at the wall.
- Prior broad two-queue support placement and host submit-ahead are also
  wall-negative or neutral.

Launch topology and queue ordering are therefore closed as the next parity
lever. Wider loads, K/V fusion, and queue scheduling do not reduce the streamed
weight bytes.

## Next measured order

Return to production-conditioned body rate and bytes, using current-route
exact cubins and cold cache state. Rank Q, O, gate/up, down, K/V, and
flash-score by recoverable rate gap against llama rather than by total kernel
time alone. The next implementation target must meet one of two conditions:

1. prove a same-byte DRAM-rate deficit with enough weighted recovery to clear
   a meaningful part of `110.109 us/token`; or
2. reduce weight bytes through a new representation/consumer contract.

The immediate action is a refreshed current-route streaming-rate table. It
must identify a concrete body, byte count, measured tinygrad bandwidth,
measured llama bandwidth, and projected us/token/tok/s before code changes.
