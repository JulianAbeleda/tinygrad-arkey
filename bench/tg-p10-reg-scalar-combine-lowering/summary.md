# TG-P10 Terminal: REG Scalar Combine Lowering

Verdict: **TG_P10_BLOCKED_REG_SCALAR_LOWERING_DIAGNOSED**

TG-P10 made the final purity blocker MECHANICALLY classifiable and pinned it to one exact compiler invariant.

## Shipped
- **TG-P10.0 PASS**: BoltBeam is ready with a small typed extension.
- **TG-P10.1 PASS**: `tinygrad.reg_scalar_lowering.v1` minimal generated-UOp repro (control + 2 compile-fail + REG_STORE_DEVEC-NaN).
- **TG-P10.2 PASS**: BoltBeam adapter + classifier -> `EMITTER_BLOCKED`; a fixed artifact flips it to `REACHABLE` (254 tests pass).

## Blocked (diagnosed, not landed)
- **TG-P10.3**: root cause pinned. When an output axis is UPCAST by 4, `reduce_to_acc` keeps a size-1 scalar REG accumulator, but `num = Σ w·pv[d]` varies along the upcast `d` -> the devectorizer emits an invalid `make_float4(acc,acc,acc,acc) = <4 partials>` store. The shipped combine only compiles because its inline `fexp` suppresses that upcast. `REG_STORE_DEVEC=1` scalarizes but collides on slot 0 -> NaN.
- Fix location: `tinygrad/codegen/late/expander.py fix_reduce_unroll` / `devectorizer.py reduce_to_acc` — widen the accumulator per upcast lane for varying axes; keep scalar for invariant axes. Generic, no kernel special-casing.
- Not landed here: the change touches the core reduce lowering used by **every** kernel; a subtle regression would silently corrupt outputs model-wide. It needs a dedicated full-model regression pass.

## Distance to the north star
One generic reduce+upcast accumulator-sizing fix (validated model-wide). Then the shipped live-split tile + the unblocked split-preserving combine should clear >=98% at ctx512 and ctx4096, and strict final purity passes. Owned HIP stays default until then.
