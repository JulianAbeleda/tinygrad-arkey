# NV 227 ledger after both Q4/Q4 K/V producer landings

Date: 2026-08-24
Installed checkpoint basis: ordinary and shared-Q8 Q4/Q4 K/V pairs plus the
producer-owned K/V cache sink

## Endpoint

| item | latency | throughput |
| --- | ---: | ---: |
| conservative installed endpoint | 4.515396 ms/token | 221.465 tok/s |
| 227 target | 4.405286 ms/token | 227.000 tok/s |
| remaining | 110.109 us/token | 5.535 tok/s |

The remaining latency is `2.58%` of the installed `4271.072 us` device node
sum. Cross-session endpoint movement is not used for per-route attribution;
the two Q4/Q4 routes have their own positive same-session brackets.

## Installed topology

```text
nodes                         462
device node sum       4271.072 us/token
device union          4258.500 us/token
measured overlap        13.202 us/token
ordinary Q4/Q4 pairs          9
shared-Q8 Q4/Q4 pairs          9
terminal producer sinks       36
generic cache stores           0
```

## Producer-side closure and next order

1. The eight shared-Q8 mixed Q4-K/Q6-V pairs passed native/profile gates but
   failed the strict wall gate. The ten ordinary mixed pairs have incompatible
   installed 32-thread K and 128-thread V geometries. K/V launch fusion is now
   exhausted as a parity route; the endpoint is unchanged.
2. Test actual producer concurrency, not timestamp reordering: a cache-cold
   complete-span microgate comparing serialized Q then K/V with Q and K/V on
   separate queues, joined before flash. This is the outstanding test named by
   the retained HCQ adjudication and has a tens-to-hundreds-of-microseconds
   ceiling across 36 layers. Promotion still requires a token-exact production
   wall bracket; dependency legality alone books nothing.
3. If concurrent memory streamers only contend and do not shorten the join
   span, return to production-conditioned streaming rate/cache-state on the
   Q, O, gate/up, down, K/V, and flash-score bodies. Small launch fusions and
   wider flash-combine topology are already adjudicated.
4. Weight-byte reduction remains the larger roofline lever, but it requires a
   new representation or consumer contract. Wider loads and launch fusion do
   not reduce DRAM bytes.

The immediate clean probe is therefore producer-to-join concurrency under a
validated cold-cache protocol, before any scheduler/runtime route is changed.
