# Q4 metadata half-load composition: rejected

On top of ordinary packed Q8 loads and packed Q4 publication, replacing the exact 32 aligned Q4 metadata byte pairs with half loads reduced PRMT 176 to 48 while keeping IMMA 256, LDG 152, LDS 384, and zero local spill. A production-shaped alternating R15 was bit exact/read-only and measured 339.167 us control versus 331.033 us candidate (+8.134 us/call).

The full graph failed replay from the first cycle. Token stayed 198, but logits and downstream stages diverged (`max_abs` 0.079740 then 0.135366); the first gate output mismatch was layer 5. Every one of the 64 removed scalar declarations had exactly one consumer, which rules out a loose matcher. This is a production publication/scheduling hazard, so the candidate was reverted and no enable path remains.
