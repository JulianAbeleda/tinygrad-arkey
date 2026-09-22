# NV 227 push: shared-Q8 mixed Q4-K/Q6-V pair scope

Date: 2026-08-24
Base checkpoint: `ae72610ba`

The conservative starting endpoint is `4.515396 ms/token = 221.465 tok/s`;
227 requires another `110.109 us/token` (`5.535 tok/s`).

The ten ordinary Q4-K/Q6-V pairs are not a clean first fusion target: their
installed K kernel uses 32 threads per CTA and V uses 128. A single launch
would change arithmetic geometry or strand three warps for every K row. The
eight shared-Q8 mixed blocks (`1,2,3,6,9,12,15,18`) instead have matching
128-thread four-warp Q4 and Q6 consumers and one existing packed-Q8 provider.

Gate: preserve each consumer's block assignment, accumulation order, shuffle
ladder, and four-partial merge; use separate caller-owned K/V outputs; require
bit equality, no spills, exactly eight removed production nodes, identical
tokens, and a candidate below both reps>=7 wall controls. This is a research
lease only unless every gate passes.
