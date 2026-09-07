# Q4 metadata half-load composition: rejected

On top of ordinary packed Q8 loads and packed Q4 publication, replacing the exact 32 aligned Q4 metadata byte pairs with half loads reduced PRMT 176 to 48 while keeping IMMA 256, LDG 152, LDS 384, and zero local spill. A production-shaped alternating R15 was bit exact/read-only and measured 339.167 us control versus 331.033 us candidate (+8.134 us/call).

The full graph failed replay from the first cycle. Token stayed 198, but logits and downstream stages diverged (`max_abs` 0.079740 then 0.135366); the first gate output mismatch was layer 5. Every one of the 64 removed scalar declarations had exactly one consumer, which rules out a loose matcher. This is a production publication/scheduling hazard, so the candidate was reverted and no enable path remains.

## Bounded hazard audit

The metadata writers use `alu7`; readers use `alu13`. Both resolve to aligned byte-pair bases (multiples of 80 plus even offsets). A full CTA `__syncthreads()` occurs after every metadata and payload store and immediately before the candidate loads; SASS retains `BAR.SYNC.DEFER_BLOCKING`. Thus a missing warp/CTA publication fence is not the cause, and adding `__syncwarp` would not strengthen the contract.

A three-activation, 10-cycle production-shaped kernel fixture was deterministic and bit exact for all 30 candidate/control comparisons. The failure is therefore specific to composed graph scheduling/lifetime rather than the isolated fragment arithmetic or an absent in-kernel barrier. The route remains rejected.
