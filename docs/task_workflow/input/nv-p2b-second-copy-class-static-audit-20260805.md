# P2b second one-per-fused-block copy class: static audit

## Verdict

**UNPROVEN; no generic alias/ownership rule was added and no GPU census is
authorized.**

The owned invocation-input contract is already the only class with an exact
CPU proof. It reduced the composed native census from 71 to 36 `E_86a2`
programs: one common baseline copy plus one surviving copy for each of the 35
fused attention-O blocks. The count is real, but it does not identify a safe
second alias class.

## Evidence inspected

* `nv-attention-o-owned-invocation-input-census-20260805.json`: 841 programs,
  36 `E_86a2`, 35 fused O calls.
* Native boundary snapshots in `/tmp/nv_p2b_attention_o_census.json` and
  `/tmp/p2b_composed_census.json`.
* The two attempted post-callify compiled-trace artifacts:
  `/tmp/nv_attention_o_postcallify_trace_census_corrected.json` and
  `/tmp/nv_attention_o_compiled_trace_census_final.json`.

The native residual-side snapshot for blocks 1--35 contains an apparent
precompiled-output spelling through `RESHAPE`, `MEMORY_SEMANTIC`,
`CONTIGUOUS`, and `GETTUPLE(FUNCTION)`. It is directional only: it is recorded
before the exact post-memory-plan writer/consumer mapping required for an
alias decision.

The alternate activation-side snapshot contains `CAST` and `PERMUTE` in its
chain. It is computation/layout work, not an equal-dtype, equal-span,
zero-offset ownership candidate, and is categorically excluded from a generic
alias rule.

Most importantly, both retained post-callify artifacts have
`post_callify_86a2_trace: []` (the earlier snapshot similarly has empty
`adapter_slots`). Therefore there is no established writer output slot, no
unique reader slot, and no physical-buffer chain from a surviving `E_86a2`
copy to a fused-O argument. Ordinal/template counts cannot substitute for
those facts.

## CPU gate

`test/unit/test_projection_epilogue_boundary_census.py` passes 17 tests. It
continues to prove the existing owned invocation-input rule and its
wrong-span/movement/multi-output protections, but does not reproduce a new
distinct survivor with a complete output-slot and reader-lifetime proof.
Consequently it cannot authorize changing `callify.py`.

## Reopen condition

Before any new generic rule or GPU run, obtain either:

1. a bounded real compiled-linear trace for every surviving `E_86a2` writer,
   showing one invocation-owned precompiled output slot, an identical
   dtype/span/zero-offset read slot, and a unique `AFTER(CALL)` dependency; or
2. a hermetic nesting that reproduces the survivor separately from the class
   already removed, with the same concrete output/read slots and lifetime.

Only then can a closed-default rule be tested against an exact CPU topology
target (producer plus fused O, no adapter). Until then the 841 census is a
partial recovery and the <=804 full-path gate remains closed.
