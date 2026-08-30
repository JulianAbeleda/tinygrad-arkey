# NV pp512 substrate test ledger

## Decision summary

Only two new substrate investments pass: the llama-style packed gate/up main and a replacement packed FFN-down main built on the same stream-K substrate. Q8 pair reuse is correct but below threshold. Every proposed standalone epilogue is either below its service upper bound or regresses as a real fused candidate.

| Hypothesis | Evidence | Decision |
|---|---:|---|
| Packed gate/up main | 20.468-20.470 ms removable oracle wall; 13.688 ms matched llama-service target | PASS |
| Packed FFN down main | 14.088-14.093 ms removable oracle wall | PASS |
| Gate/up Q8 record reuse | exact logits; 0.316-0.321 ms median win | STOP standalone |
| Gate SiLU/multiply plus FP16 cast | 0.728 ms summed service; real fused candidate regresses 1.122-1.140 ms median | STOP |
| Down residual epilogue | 0.174 ms summed service upper bound | STOP |
| Q/O direct conversion/copy | 0.164 ms attributable service upper bound | STOP |
| K/V layout/transpose copies | about 0.196 ms attributable service upper bound | STOP |
| Standalone FP16 gate output | bit-exact, but included fused candidate regresses | STOP |

## Gate/up oracle

R9 control/oracle/control medians are 65.298799, 44.830975, and 65.301318 ms. The oracle preserves 72 Q8 producers and replaces exactly 72 packed gate/up mains. Every K, Q/O, Q4 V, and Q6 V population is unchanged; total admitted mains and producers remain 216 each; 20-cycle replay is exact.

The 20.47 ms recovery is the full removable main service, not the expected real-kernel gain. The matched tinygrad-versus-llama delta is 190.112 us per projection, or 13.688064 ms for the exact 72-call population.

## FFN-down oracle

R9 control/oracle/control medians are 65.333233, 51.240559, and 65.328923 ms. Exactly 36 down matmuls are replaced while retaining 36 distinct FP16 overlay-weight inputs, the real down activations, output allocations, and residual positions. All 216 packed-route mains/producers remain unchanged and deep replay is exact.

The prior Q4-down and Q6-down implementations remain rejected: Q4 failed full-output correctness and Q6 regressed 1.6-1.7 ms. This PASS qualifies a new down shape on the future stream-K substrate; it does not revive either old implementation.

## Q8 gate/up pair reuse

The candidate shares one graph-owned Q8 record across each ordered gate/up pair. It has exactly 36 producers and record allocations feeding 72 unchanged packed mains. Full logits are bit-exact (`max_abs=0`), token 198 is unchanged, and deep replay is exact.

R9 control/candidate/control medians are 65.350663, 65.034509, and 65.355422 ms. The 0.316154 and 0.320913 ms wins fail the frozen 0.5 ms standalone investment threshold. Reuse may be folded into the future gate kernel only when it adds negligible complexity; it is not a separate workstream.

## Epilogue and layout service

Live finalized graph buffers and runtimes were measured for nine device-timed samples per exact call.

- 36 `SiLU(gate) * up` calls: 455.520 us summed median service.
- 36 FP32-to-FP16 down-input casts: 272.768 us.
- 36 down residual additions: 174.176 us.
- Q-projection direct FP32 copy population: 164.000 us attributable service.
- 36 attention transpose/layout calls: 77.344 us.
- K/V-adjacent FP16 layout copies: about 118.880 us.

Only the grouped gate epilogue exceeded 0.5 ms service, so it advanced to a real candidate. A bit-exact native fused SiLU/multiply/cast program replaced the 72 old calls with 36 graph-owned calls. Full logits remained bit-exact, but R9 control/candidate/control medians were 65.330881, 66.452672, and 65.312621 ms. It regressed 1.121791 and 1.140051 ms and is closed `STOP`.

## Build order

1. Build one shape-qualified Q4_K x Q8 stream-K tensor-core main for `M=512, N=12288, K=4096` and the exact 72 gate/up population.
2. After the gate kernel passes correctness, matched service, and full-model A/B/A, extend the same scheduler/reduction substrate to `M=512, N=4096, K=12288` for the exact mixed Q4_K/Q6_K down population.
3. Do not create separate Q8-reuse, SiLU, cast, residual, Q/O-copy, or K/V-layout projects. Re-evaluate fusion only inside a passing packed main where it is nearly free.

Gate and down oracle recoveries are destructive upper bounds and are not additive performance forecasts. The real investment target remains closing the measured llama service differences with correct packed kernels.
