# NV Q6 direct shared-Q8 g12 record

Status: **WALL_NO_GO; closed-default lease remains unpromoted.**

The flat four-warp Q6/Q8 DP4A direct-output consumer was integrated behind the
explicit `SharedQ8AttentionAdmission.q6_direct_output` lease.  It is restricted
to native NV sm_120, the real Qwen 1024x4096 Q6 V shape, and preserves the
existing shared Q8 provider plus Q4 consumers.  Existing g12/max-subset
artifacts contain zero direct-Q6 programs, so this experiment is disjoint from
the booked Q4 g12 recovery.

## Semantic progression

Against the composed g0 control (P1/P2/P5), every admitted stage was finite,
had exact token stream/argmax/top-10 sets, and passed the relative-L2 and
margin contract:

| Q4/Q6 lease depth | relative L2 | direct Q6 consumers | result |
| ---: | ---: | ---: | --- |
| 1 | 0.000389914 | 2 | PASS |
| 4 | 0.000540336 | 6 | PASS |
| 8 | 0.000529770 | 8 | PASS |
| 12 | 0.000573415 | 12 | PASS |

The expected direct-consumer count is derived from actual `Q6KPrimitiveLinear`
V blocks, not provider count: Qwen mixes Q4/Q4/Q4 and Q4/Q4/Q6 groups.

## Settled g12 reverse bracket

All arms used `d512`, 32-token uninterrupted windows, five repetitions, two
feedback captures, the cooperative-Q4 g12 shared route, and identical stream
hashes.

| arm | ms/token |
| --- | ---: |
| partial-Q6 control A | 5.395371719 |
| direct-Q6 candidate B | 5.404545156 |
| partial-Q6 control A2 | 5.399700406 |
| control midpoint | 5.397536063 |
| B minus midpoint | **+0.007009094 ms/token** |

The direct consumer is **+7.009094 us/token slower** than the existing shared
Q8 partial route.  This is a real-token included-cost reverse bracket, so the
route receives zero recovery credit.  Do not run max17, promote the lease, or
add its primitive win over the older installed Q6 route to the ledger.

Raw machine artifacts remain in `/tmp/nv-q6-direct-*-20260805.*`.
