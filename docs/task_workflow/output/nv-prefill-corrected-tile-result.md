# NV dense prefill candidate restoration

## Result

The dense Qwen3-8B pp512 candidate path is restored on ordinary `DEV=NV`, but
the historical 44--46 ms measurement is retired.  It was fast because it was
numerically invalid, not because a valid optimization had become disconnected.

The corrected installed path is finite, bit-exact against the safe tensor-core
projection path for all four production shapes, and produces the identical
whole-model logits SHA and argmax.  A fresh complete R7 synchronized authority
bracket measures 84.025 ms per 512-token chunk, or 6,093 prompt tok/s.

Authority: `docs/task_workflow/evidence/nv-prefill-corrected-tile-20260828/authority-complete-r7.json`.

## Root cause

Two independent faults existed in the retained candidate:

1. A lane store into a loaded local-memory vector rendered as an assignment to
   the loaded C temporary.  The shared-memory publication was lost, yielding
   all-NaN projections.
2. The candidate already owned a complete 128x128 output tile, but its
   warmstart also applied generic output upcasts.  A CTA then covered more
   logical rows than its declared tile and distinct rows aliased the same LDS
   slots.  Repairing only the first fault made the output finite but wrong.

The valid schedule retains only the tensor-core option.  Candidate geometry
owns all output subtiles.  Cooperative LDS stores use explicit scalar lane
addresses so the renderer cannot turn their destinations into loaded
temporaries.

## Correctness gates

| gate | result |
| --- | --- |
| Q/O projection | bit-exact; finite |
| K/V projection | bit-exact; finite |
| gate/up projection | bit-exact; finite |
| down projection, nonzero deterministic input | bit-exact; finite |
| complete model logits | identical SHA-256 and argmax to context-disabled safe path |
| candidate route census | all four identities selected; no missing or unexpected roles |
| focused unit tests | 54 passed |

The all-ones, within-tile K-map, and across-tile ownership fixtures also pass.
They establish complete work coverage but are diagnostic evidence rather than a
substitute for the real-weight and whole-model comparisons above.

## Performance interpretation

The valid path improves the prior fused-attention/safe-projection wall of about
143 ms to about 84 ms.  It does not reproduce the invalid 46 ms row and does
not match the retained same-session llama.cpp pp512 result near 14k prompt
tok/s.  The honest current claim is therefore restored, correctness-qualified
fast prefill at roughly 6.1k prompt tok/s, not prefill parity.

The remaining performance gap is no longer a missing route hookup.  Recovering
it requires a new candidate whose declared tile and physical LDS ownership
cover additional output work without aliasing; re-enabling the retired generic
upcasts is forbidden.
