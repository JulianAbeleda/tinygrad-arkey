# TG-P9 Terminal: Pure Generated 8B Attention Primitive Route

Verdict: **TG_P9_BLOCKED_SPLIT_PRESERVING_COMBINE**

The primitive route split cleanly in two. One half is now a real generated capability; the other is pinned to a concrete tinygrad codegen gap.

## Phases
- TG-P9.1 **PASS**: live-context split geometry IR is expressible in generated UOp (TG-P8 thought it was EMITTER_BLOCKED; the real blocker was a typed-const bug).
- TG-P9.2 **PASS**: live-split tile is byte-identical and scales with Tc -- ctx512 tile 36.6->8.0us (4.6x); **full-model ctx512 87.7% -> 96.7% of owned**, token-identical, route-bound.
- TG-P9.3/9.4 **BLOCKED (EMITTER)**: the split-preserving combine (remove the per-d fexp redundancy without collapsing Hq*S or Hq*Hd) cannot be lowered -- every weight-sharing/gmax-fusing shape mis-vectorizes the reduction-accumulator REG to a non-assignable `make_float4(...) = ...`; `REG_STORE_DEVEC=1` compiles them but returns NaN.
- TG-P9.5 **REFUTE_STILL_SLOW**: full candidate 96.7%/95.3% (up from 87.7%/95.9%), still <98% at both -- combine-capped.

## Live-split candidate vs owned (full model)
| ctx | owned tok/s | live-split gen tok/s | % owned | token_match |
|---|---|---|---|---|
| 512 | 107.6 | 104.0 | 96.7% | True |
| 4096 | 97.9 | 93.3 | 95.3% | True |

## Outcome
Owned HIP stays the 8B attention default. TINYGRAD_DEFAULT_PURITY_PASS remains blocked, but the impurity is now a **single, precise compiler primitive gap**: the AMD backend cannot keep a combine's reduction-accumulator REG scalar when the kernel shares softmax weights across d or fuses the gmax max-reduce. Fix that lowering and the combine (hence full parity, hence full purity) becomes reachable -- the live-split tile already closes the ctx512 half.
