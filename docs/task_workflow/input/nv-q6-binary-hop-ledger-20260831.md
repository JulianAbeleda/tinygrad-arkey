# NVIDIA Q6 binary hop ledger

## Fixed target

- Shape: `M=512, N=4096, K=12288`.
- Generated route: 170-owner wide Stream-K, phase unroll two.
- Generated main: 318.8 us in the direct repeated-launch measurement.
- Llama main oracle: 201.216 us.
- Llama five-percent gate: 211.2768 us.
- Main gap: 117.584 us; 107.5232 us must be recovered to reach the gate.
- Generated fixup: approximately 11.7 us; llama fixup: 8.64 us.

## Execution hops

| hop | operation | llama | generated | state |
|---:|---|---|---|---|
| 1 | CTA/tile/K ownership | approximately 170 persistent CTAs | 170 Stream-K owners | matched |
| 2 | canonical Q6/Q8 global input | direct packed input | direct packed input | matched |
| 3 | Q6 low/high-bit and scale decode | scheduled inside large epoch loop | repeated scalar unpack/permutation | unresolved |
| 4 | shared-memory publication | packed warp-oriented layout | scalar-oriented generated staging | unresolved |
| 5 | producer/consumer synchronization | software-pipeline boundaries | phase-oriented barriers | barrier pressure rejected alone |
| 6 | warp fragment load | `LDSM` | scalar `LDS` | substitution rejected alone |
| 7 | signed INT8 tensor work | large interleaved `IMMA` body | smaller nested-loop body | unresolved |
| 8 | FP32 scale and accumulation | interleaved with tensor work | more separately materialized work | unresolved |
| 9 | owner partial store | partial workspace | partial workspace | functionally matched |
| 10 | deterministic fixup | 8.64 us | approximately 11.7 us | approximately 3 us secondary gap |

## Binary topology

The generated route contains an inner K/phase loop and an outer owner/tile
loop. Its observed backedges are `0x62c0 -> 0x0960` and `0x6af0 -> 0x02e0`.
The llama route has one much larger software-pipelined loop with its principal
backedge near `0x20780 -> .L_x_137`. Llama interleaves loads, decode, fragment
preparation, `IMMA`, and FP32 work inside this body.

## Matched counter evidence

| counter | generated wide baseline | llama | interpretation |
|---|---:|---:|---|
| executed instructions | 281.8M | 176.7M | generated executes 105.2M more instructions |
| L2 traffic | 650.9 MB | 517.3 MB | generated transfers 133.6 MB more through L2 |
| active warps | 16.66% | 16.66% | occupancy is not the primary gap |
| SM/tensor throughput | 38.1% | 49.5% | llama feeds useful tensor work more densely |
| long scoreboard | 7.46% | 4.73% | generated waits more on dependencies |
| math-pipe throttle | 11.36% | 25.80% | llama spends more time saturating tensor execution |

## Rejected explanations

- DRAM bandwidth alone: matched routes read approximately the same DRAM bytes.
- Occupancy alone: active-warp occupancy is equal.
- Barrier pressure alone: measured barrier stalls are too small.
- Wide-route spills: the qualified wide route has no local-memory spills.
- Persistent Q6 cache alone: reduced L2 traffic but increased instructions and runtime.
- Native `LDSM` substitution alone: exact K256 A/B was 1.98% slower than scalar `LDS`.
- More geometric ownership: 170 owners remains the best measured configuration.
- Four-phase source unrolling: crossed the register boundary, spilled, and regressed.

## Region tests

### Region A: Q6 decode and shared publication

Hold canonical blocks, CTA population, decoded byte count, and shared output
layout fixed. Compare the current producer with a readback/checksum consumer
that removes `IMMA` and FP32 epilogue work. Report time per K256 block, global
loads, shared stores, barriers, registers, and spills.

### Region B: fragment preparation and IMMA

Feed already prepared shared operands and execute the same `IMMA` population
as the full route. Exclude canonical Q6 decode and FP32 scale application.
Report exact int32 output, latency, `LDS`/`LDSM`, `PRMT`, `IMMA`, registers,
and spills. The scalar/native fragment A/B is a subtest, not the complete
region measurement.

### Region C: FP32 scaling and accumulation

Feed deterministic int32 MMA tiles and canonical Q6/Q8 scales. Execute the
same scale pairs, FP32 accumulation chains, and output ownership as the full
route without global packed decode or tensor work. Report exact FP32 output,
latency, conversion/arithmetic instruction census, registers, and spills.

## Attribution rule

Region timings are diagnostic and may overlap in a software pipeline, so they
must not be summed as a performance prediction. A region qualifies as the next
build target only if its generated-minus-oracle or generated-minus-lower-bound
cost is large enough to explain a material fraction of the 117.584 us main gap,
and a full-route A/B then preserves exactness while reducing latency.

## Region B result (2026-08-31)

The isolated prepared-operand gate passed exact int32 output. It stages the
already-packed operand windows and issues the same 16-group IMMA population,
without canonical Q6 decode or FP32 scaling. With 4096 independent replicas:

- median `23.136 us`, minimum `22.944 us`
- `16 IMMA`, `16 LDSM`, `0 LDS`, `0 PRMT`
- `95` registers, `576` bytes local frame, `6144` bytes shared allocation
- `0 LDL`, `0 STL`

This proves the fragment-preparation plus IMMA subgraph is exact and spill-free.
It is a diagnostic lower-level measurement and must not be summed with Regions
A or C; the full-route causal test remains required.
