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

## Next producer-side order

1. Test mixed Q4/Q6 K/V pairs as two separate grammars: ten ordinary blocks
   and eight shared-Q8 blocks. The population is 18 pairs, but weight bytes do
   not fall, so advancement requires a native rate/launch win and then a wall
   bracket; population alone is not evidence.
2. If mixed pairs cannot recover a material fraction of `110.109 us`, widen
   ownership to an attention Q/K/V producer boundary. That can amortize more
   launch/control work than K/V-only fusion, but has a higher register and
   semantic-boundary risk.
3. Keep topology-only overlap work behind producer body reduction. Installed
   overlap is only `13.202 us/token`; timestamp reordering cannot supply the
   missing `110.109 us/token` without creating genuinely concurrent work.
4. Weight-byte reduction remains the larger roofline lever, but it requires a
   new representation or consumer contract. Wider loads and launch fusion do
   not reduce DRAM bytes.

The immediate clean probe is ordinary mixed Q4-K/Q6-V pairing, followed by
the shared-Q8 mixed pair only if the isolated arithmetic/occupancy gate is
positive.
