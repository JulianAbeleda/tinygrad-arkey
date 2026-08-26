# NV post-horizon short-context ledger

## Authority

The installed pre-cliff authority is a depth-512 composed production run with
16-token windows and 12 repetitions.  Its 192 timed tokens remain below the
context-769 flash alignment cliff.  No output-horizon hint was supplied, so
the newly booked static-S64 policy was inactive.

| authority | latency | throughput | relative to 240 | relative to retained llama |
| --- | ---: | ---: | ---: | ---: |
| tinygrad pre-cliff S48 | 4.261471 ms/token | 234.66 tok/s | 94.804 us / 5.34 tok/s behind | 213.146 us / 12.36 tok/s behind |
| retained llama d512 | 4.048325 ms/token | 247.02 tok/s | 118.342 us faster | reference |

The previous approximately 236.6 tok/s row is retired as the current
authority.  It came from a different position/window sample.  A separate
288-token check crossed the alignment boundary and is retained only as regime
evidence; its median was 4.265855 ms/token and its final window rose to
4.406773 ms/token.

## Device ledger

Full profiling is a body census, not an unprofiled wall decomposition.  It
inflated the measured wall to 5.578 ms.  Within that profile domain, the
kernel union is 4.1095 ms and measured overlap is only 3.364 us.  Aggregated
body mass is:

| lifecycle role | profiled device mass | interpretation |
| --- | ---: | --- |
| FFN gate/up | 1,274.048 us | largest body, but exact load-width and tested geometry changes are wall-closed; near streaming/ramp wall |
| FFN down | 853.280 us | byte lever is real; current Q5/Q4 post-hoc artifacts fail the recurrent quality contract |
| Q/O projections | 603.680 us | exact stream-rate opportunity only if a new issue/dequant or complete-stream construction passes cold counters |
| norms/reductions | 425.120 us | mostly compulsory; largest exact rewrite tested slower |
| flash attention | 336.000 us | S48 is the measured pre-cliff winner; S64 is admitted only by request horizon |
| vocabulary projection | 314.272 us | large streaming body; serial argmax tail already removed |
| K/V projections | 235.808 us | several pair/triple producer topologies already adjudicated; only new ownership/rate mechanisms remain open |
| elementwise/tail | 69.344 us | too small as a first-order pool |

A marker-light nine-token run measured a 4.264256 ms device window against the
4.261471 ms clean wall.  The 2.785 us difference proves that the clean token
wall is effectively the device critical path at this authority.  Marker wall
inflation and overlapped host preparation must not be booked as additive
headroom.  Recovering 240 therefore requires approximately 95 us of device
service, not Python or graph bookkeeping.

## Ranked next actions

1. **New exact issue/dequant scheduling for Q/O/K/V.** Rebuild a current cold
   rate census and advance only a body with a measured size-aware rate deficit
   and a causal stall mechanism.  The candidate must change load-level
   parallelism, dequant scheduling, or CTA aggregation; instruction-only
   unrolling is already closed.
2. **Removal of a complete physical stream/ramp.** A new producer-consumer
   ownership construction remains admissible, but tested ordinary QKV full
   grid, shared producer variants, queue order, and broad PDL placements may
   not be repeated as if untested.
3. **Training/calibration-aware byte reduction.** This has the largest
   theoretical ceiling, but requires a new model artifact.  Post-hoc Q5/Q4
   and coarse row selection are closed under the current quality contract.
4. **Vocabulary projection service rate.** Reopen only with a construction
   that raises the full 151936x4096 stream rate; the reduction tail is already
   near floor.
5. **Broader context geometry table.** Useful for consistency beyond context
   1024, but it does not raise the present pre-cliff endpoint.

Decision:
`SHORT_CONTEXT_DEVICE_WALL_CONFIRMED__NEXT_GATE_QO_KV_RATE_CAUSAL_CENSUS`.
