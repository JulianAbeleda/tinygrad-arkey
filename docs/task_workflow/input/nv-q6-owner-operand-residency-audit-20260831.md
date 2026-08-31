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
