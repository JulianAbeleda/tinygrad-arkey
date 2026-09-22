# Q6_K owner path operand-residency audit

Date: 2026-08-31
Route: generated tinygrad Q6_K Stream-K owner main
Shape: `M=512,N=4096,K=12288`
Oracle: llama Q6 down main `201.216 us`, pair total `209.856 us`

## Finding

The generated owner kernel is numerically exact but measures `689.157 us`
minimum. The dominant gap is shared-operand residency, not MMA availability or
CTA occupancy. Both routes use 170 CTAs, 256 threads, 128x128 tiles, 2,048
IMMA operations per K=256 unit, 255 registers/thread, and 58,880 bytes shared.

## Hop accounting per CTA K=256 work unit

| hop | generated | llama design | excess |
|---|---:|---:|---:|
| shared A fragment reads | 524,288 B | 65,536 B | 8.0x |
| shared B fragment reads | 262,144 B | 131,072 B | 2.0x |
| shared Q6 D reads | 131,072 B | 8,192 B | 16.0x |
| shared Q6 scale reads | 262,144 B | 16,384 B | 16.0x |
| shared Q8 scale reads | 524,288 B | 131,072 B | 4.0x |
| total shared reads | 1,703,936 B | 352,256 B | 4.84x |

Across 6,144 useful work units this is approximately 10.913 GB generated
shared reads versus 2.608 GB for llama. The generated route performs the same
25.770B MACs and 2,048 IMMA operations per work unit, so arithmetic count is
not the explanatory gap.

## Ranked theories and admission tests

1. Make A fragments, packed Q6 scales, and D phase-resident; load each B
   fragment once per column group and reuse it across both output rows. Target
   16 ldmatrix calls, 64 B loads, 8 packed Q6-scale loads, and 4 D loads per
   phase. Predicted save: 350-450 us.
2. Attach phase-1 Q8 global loads to the phase barrier so source values are not
   hoisted before phase-0 completion. Check local reservation and local-memory
   sectors. Predicted save: 20-80 us.
3. Structure the two tile segments explicitly with one accumulator bank and one
   boundary flush/reset, removing 64 predicated stores and per-hop div/mod.
   Predicted save: 40-100 us.
4. Strength-reduce tile coordinates to incrementing block indices inside each
   segment. Predicted save: 10-40 us.

Do not prioritize accumulator batching, owner-count tuning, async copies, a new
MMA primitive, or CTA geometry until the operand-residency counts and local
traffic are measured. Register count alone cannot improve occupancy here:
shared memory and the 170-CTA grid already limit the route to one CTA per SM.

## Required ledger fields for every experiment

Record exact values/IDs, minimum and median time, residual to the 211.277-us
5% gate, useful work units, IMMA count, global Q6/Q8/scale bytes, shared reads
by operand, actual local load/store sectors, barriers, boundary stores,
register/shared/local reservations, resident CTAs/warps, predicted delta, and
observed delta.

## E1 result: materialized A-fragment residency

The backend-neutral materialized native-fragment path reduces emitted A-fragment
loads from 256 to 32 per K=256 work unit while preserving 256 WMMA calls and
four barriers. The corrected target is 32 total loads: 16 per phase.

Representative qualification remains bit-exact across 5,570,560 FP32 values
and all 340 tile IDs (`max_abs=0`). Minimum owner-main time improves from
689.157 us to 678.516 us, an observed 10.641-us (1.54%) win. Resources remain
255 registers and 58,880 shared bytes; local reservation rises from 984 to
1,088 bytes. The predicted 350-450-us gain is falsified for A residency alone.
The remaining investigation must separately account for B, Q6 D/scales, Q8
scales, accumulator spill/liveness, and per-hop boundary control.

## Concrete UOp refactor plan

The current `q6_streamk_owner_kernel` has the right arithmetic but its graph
topology makes residency impossible to preserve. `qwords` and `qscales` are
constructed inside the static `for kphase in range(2)` body, while `mma()` and
the epilogue create fresh `sh.after(...)` / `shq.after(...)` consumers for every
`(cg,n,r,g)` path. The generated graph therefore exposes repeated shared reads
even when the logical value is unchanged.

Implement the first pass in a separate owner-kernel constructor so the exact
CTA primitive remains unchanged during optimization. The graph should have
this order for each active owner segment:

```text
segment metadata -> Q6 phase 0 stage -> barrier
                 -> Q8 phase 0 stage -> barrier
                 -> phase-0 MMA for all 8 column groups
                 -> Q8 phase 1 stage -> barrier
                 -> phase-1 MMA for all 8 column groups
                 -> one boundary flush/reset -> final store
```

Use one shared arena with named, non-overlapping regions: Q6 bytes/scales/D,
Q8 records for the current phase, and a small resident cache for the values
consumed by all eight column groups. The Q8 record producer must be a single
`UOp.group` per phase, followed by one `UOp.barrier`; all MMA nodes in that
phase must consume the same barrier-ordered `ydst` view. Likewise, bind one
`sx = sh.after(ready_q6)` and one `sy = shq.after(ready_y)` per phase and pass
those exact UOps through the MMA and scale epilogue, instead of recreating
ordered aliases inside each output loop.

The first admission target is structural, before timing: per K=256 work unit
the generated graph must expose no more than 16 A `ldmatrix` loads, 64 B/Q8
word loads, 8 packed Q6-scale loads, 4 D loads, and 4 Q8-scale loads per
phase. The current accounting is 32 A loads, 128 B/Q8 word loads, 16 Q6-scale
loads, 8 D loads, and 8 Q8-scale loads per phase. The target is therefore the
llama-shaped 2x reduction for A/B and 4x reduction for Q6 metadata. Count
source UOps and emitted PTX memory instructions separately; CSE alone is not
an admission criterion if local sectors remain unchanged.

The owner loop must use a single accumulator bank and a single transition
predicate. On transition, store the completed bank to `owner*2` (row-major
tile layout), clear the bank, and begin the next tile. Do not attach the
transition predicate to every MMA or operand load. For the initial experiment,
keep the existing dynamic `lo/hi` and fixup ABI unchanged; only operand graph
topology is allowed to change. This isolates the causal effect.

## Experiment sequence and accounting gates

1. Baseline the current exact owner graph and save source counts, PTX counts,
   local sectors, and timing.
2. Apply phase-resident aliases and shared Q8/Q6 staging. Require exact output
   and unchanged owner/tile IDs. Reject if any operand count or local sector
   increases.
3. Add explicit two-segment control flow and strength-reduced tile indexing.
   Require the same exactness checks and no more than one boundary flush per
   segment.
4. Benchmark only after gates 2 and 3 pass. The performance gate is `220.349
   us` total (5% above the `209.856 us` llama pair), with `201.216 us` as the
   main-kernel reference. Record minimum and median, not a single sample.

Expected outcome is a reduction from `1,703,936 B` to approximately `352,256 B`
shared reads per work unit and from the measured `689.157 us` toward the
`220.349 us` total gate. The first residency-only pass is expected to save
`350-450 us`; this is a hypothesis, not a result, until the ledger records
actual sectors and timing. If shared reads meet target but time does not,
investigate local-memory sectors and register allocation next; do not infer a
new primitive is needed.
