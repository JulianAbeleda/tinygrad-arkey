# Flash cold-service cause

## Current answer

The residual Flash loss is no longer an undifferentiated memory wall. Two
mechanisms are measured:

1. The installed S8 kernel performs about one quarter more horizon work than
   llama's production S6 kernel at depth 512. The closed S6 lease removes that
   work, but the generic token selector did not pass wall.
2. At matched S6 horizon and essentially matched DRAM/L2 bytes, tinygrad uses a
   less efficient load/reduction grammar: more global-load sectors and severe
   shared-memory wavefront expansion in the final PV reduction.

The second mechanism is the remaining machine-level target.

## Matched S6 counter split

These counters compare the tinygrad S6/wide-KV body with llama's corrected
production-flags S6 body in the full-entry residency state. Cross-runtime NCU
duration is not used as timing authority; the counters identify work and stall
structure.

| measure | tinygrad S6 | llama S6 | interpretation |
|---|---:|---:|---|
| DRAM reads | 3.183 MB | 3.163 MB | matched |
| L2 traffic | 13.178 MB | 13.174 MB | matched |
| global-load sectors | 491,520 | 417,792 | tinygrad +17.65% |
| global-load warp instructions | 36,864 | 34,560 | tinygrad +6.67% |
| sectors/global-load request | 13.33 | 15.11 | llama gets more useful sectors per request |
| shared-load instructions | 13,824 | 19,968 | tinygrad executes fewer |
| shared-load data-pipe wavefronts | 101,383 | 19,968 | tinygrad 5.08x |
| wavefronts/shared-load instruction | 7.33 | 1.00 | bank-conflict/replay signature |
| long-scoreboard stall | 65.84% | 58.98% | tinygrad waits longer on load dependencies |
| achieved occupancy | 9.14% | 9.47% | essentially matched |
| registers/thread | 56 | 162 | register occupancy is not tinygrad's limiter |

This rules out extra DRAM bytes, L2 bytes, instruction count, registers, and
occupancy as the sole equal-horizon explanation. Llama's advantage is better
memory-level service: fewer/more productive global requests and conflict-free
shared consumption of the final warp partials.

## First construction result

Transposing only the internal PV shared array reduced shared-load wavefronts
from about 101k to 52k and shared-store wavefronts from about 26k to 14k. It did
not pass the primitive gate. The changed index stopped the compiler from
statically expanding the final PV loop:

| measure | control | transposed candidate |
|---|---:|---:|
| hot primitive | 3.812 us | 4.360--4.363 us |
| shared-load instructions | 13,824 | 50,688 |
| dynamic instructions | 801,600 | 838,080 |
| cold long-scoreboard stall | about 63% | 72--80% |

Forced source-level unroll pragmas did not change that generated machine
grammar. This candidate is a no-go, but it proves the bank behavior is
turnable and identifies the constraint on the next implementation: preserve
the original compile-time/vector expansion while changing shared layout.

## Next admissible test

Build the final cross-warp PV exchange with an explicitly vectorized/static
shared ABI rather than a dynamically indexed scalar loop. The primitive gate
must jointly require:

- shared-load wavefronts close to the number of shared-load instructions;
- no increase in global-load sectors;
- no increase in dynamic instructions;
- hot and cold body improvement;
- identical partial outputs before any production investment.

If that gate passes, test it first on both S6 and installed S8, then profile the
36-layer score sum and finally run the token bracket. This is generic to the
dense Flash shape family: it changes internal reduction layout, not model
weights or an 8B-specific layer policy.

## Evidence

- `docs/task_workflow/evidence/nv-flash-fast-math/s6-detailed-counters.json`
- `docs/task_workflow/evidence/nv-flash-fast-math/llama-s6-full-entry-detailed.ncu-rep`
- `docs/task_workflow/evidence/nv-flash-fast-math/s6-transposed-pv-smem-r1.json`
- `docs/task_workflow/evidence/nv-flash-fast-math/s6-transposed-pv-smem-unroll-r1.json`
- `docs/task_workflow/evidence/nv-flash-fast-math/s6-transposed-pv-smem-static-r1.json`
- `docs/task_workflow/evidence/nv-flash-fast-math/s6-transposed-pv-smem-nested-timing-r1.json`

