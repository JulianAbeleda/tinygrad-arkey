# TG-P11 Terminal: Reduce/Upcast Accumulator Widening

Verdict: **TG_P11_8_BLOCKED_REGRESSION_SURFACE_TOO_BROAD**

The core fix was diagnosed to a sharper mechanism and a minimal attention-free reproduction, but a correct, generic, SAFE version could not be landed in-session.

## Refined root cause
The hand flash/gemv kernels express reductions with a **manual END/AFTER accumulator** idiom (`acc.set(acc.after(s)+v, end=s)`), **not** `Ops.REDUCE` (confirmed: END=2/AFTER=4, REDUCE=0). So `reduce_to_acc`/`horizontal_reduce` -- which correctly folds upcast reduce lanes -- never runs on them. When a contiguous load in the reduce body is folded to float4, `acc + float4` broadcasts the scalar-REG accumulator to a vector and emits the invalid `make_float4(acc,acc,acc,acc) = <4 partials>` store; `REG_STORE_DEVEC` aliases the lanes into slot 0 -> NaN. `gmax` (strided load) and the shipped combine (inline fexp blocks the fold) compile clean.

## Candidate fixes (both gated by REDUCE_ACC_UPCAST_FIX, both need model-wide validation)
1. Horizontal-reduce the vectorized END-accumulator value in the devectorizer (op-specific: max = plain HREDUCE, add = HREDUCE - (n-1)*acc).
2. Prevent the load-fold in a scalar-REG reduce body (keep the reduce scalar like gmax; measure the combine still beats the shipped fexp path).

## Why not landed
Both touch the core reduce/devectorize lowering used by **every** kernel; a subtle regression silently corrupts outputs model-wide. Responsible landing needs reliable UOp-graph pattern detection + a full token-parity/W==D regression pass -- a dedicated compiler effort. Owned HIP stays the 8B attention default.

## Delivered
- TG-P11.0 baseline pinned; TG-P11.1 minimal attention-free microgate (`extra/qk_tg_p11_reduce_upcast_microgate.py`).
- Root cause sharpened from 'reduce+upcast' to 'manual END-accumulator bypasses horizontal_reduce under load-fold' -- with two concrete, scoped fix approaches.
