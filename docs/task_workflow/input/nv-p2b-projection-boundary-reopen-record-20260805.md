# P2b projection-epilogue boundary reopen record

Date: 2026-08-05. Target: native NV decode, Qwen3-8B-Q4_K_M. Status:
**the callify-owned precompiled-output redirect independently passes topology,
full-logit, replay, and reverse-wall gates but remains closed by default;
the composed attention-O epilogue fails its adapter gate and is closed.**
FFN-down prelude remains closed for its independent recomputation defect. No
production policy was promoted.

## Question

M4's old attention-O/FFN-down epilogue variants lost because an opaque program
materialized extra inputs. M5 subsequently added one narrow typed `AFTER` view
ABI for the flash-combine -> attention-O activation. This record asks the
necessary narrower question before reopening timing: can today's primitives
pass the *other* inputs to a fused projection with no adapter?

## Current boundary rule

Before the amendment, `UOp.custom_kernel` preserved an input only if it was an
`AFTER`, or a `MEMORY_SEMANTIC` view that already had physical buffer identity.
Every other input was wrapped in `CONTIGUOUS`, which schedules its producer as
a concrete adapter. M5 does not change that general rule. Its typed ABI is intentionally
limited to the fp16 declared flash-combine output and its attention-O activation
slot; it cannot certify a fp32 residual, a function result, or a lazy
arithmetic value.

The new hermetic census
`test/unit/test_projection_epilogue_boundary_census.py` uses the real Q4K
emitters and only lazy scheduling. It proves both sides of the rule:

| family/input | present graph form | emitted adapter | result |
| --- | --- | --- | --- |
| attention-O residual | nested block `FUNCTION` result (`GETTUPLE`, no physical identity) | one `test` materialization before `q4k...epi_resadd` | blocked |
| same residual shape from exact Q4K producer `AFTER` | concrete producer identity | none | direct, but not the model's value form |
| FFN down gate/up | Q4K producer outputs | none | direct |
| FFN down `h` residual | lazy fp32 add | one `test` materialization | blocked |
| block-output / next consumer | function-result boundary has no outward `AFTER` contract | no generic view proof exists | blocked |

The test suite is deliberately a construction gate, not a performance claim:
`42 passed` across the new census, M4 render/admission tests, and M5 typed ABI
tests.

## Amendment: precompiled invocation identity

The original conclusion was too broad. Callify already assigns a fresh output
buffer to every `FUNCTION(precompile=True)` and rewrites its result to
`AFTER(BUFFER, CALL)`. The old adapter happened earlier, while
`custom_kernel` could see only `GETTUPLE(FUNCTION)`. The new narrow
`UOp.has_precompiled_output_identity()` preserves only that exact returned
slot, plus `RESHAPE` and `MEMORY_SEMANTIC` wrappers. It rejects non-precompiled
calls, permutations, shrinks, expands, and offsets. The CPU proof shows raw
`GETTUPLE` versus `CONTIGUOUS` for the admitted/control arms; after callify
the producer output buffer is the epilogue residual argument by object
identity, with no adapter call.

## Family decisions

### Attention-O residual

The exact `AFTER` control demonstrates that the current transport itself can
avoid an adapter. The model residual is not that control: its block/call output
is `GETTUPLE`, therefore `has_buffer_identity()` is false and the existing
rule materializes it. This restates the mechanism behind M4's +36-node row
using the current code, after M5 landed.

The old typed-view route remains inapplicable, but the precompiled-output
contract is a safe consumer hook. The next gate force-opens only the existing
closed M4 attention-O route and requires full logits plus a census which removes
the 36 adds without new adapters before timing. The GPU lock was occupied at
amendment time; no measurement is claimed here.

### FFN-down prelude + residual

Gate and up can reach the old fused kernel directly today. `h` cannot. More
importantly this exact emitter remains independently rejected: it recomputes
`silu(gate) * up` inside every 4096 output row (M4 decomposition: 98.16 us vs
26.23 us legacy per call). Removing its one remaining adapter cannot repair
that algorithmic defect. No GPU arm is admitted.

### Block-output contiguous

The exact precompiled result now has an output-buffer contract, but this route
still needs a consumer-specific alias/lifetime proof. The 37.513 us/token row
remains a location, not recovery credit.

## What would reopen P2b

The missing capability was exposure of the existing precompiled output-buffer
contract, not another emitter. Attention-O can now proceed. A residual-only
FFN-down emitter and block-output contiguous still require their own real-chain
proof; the old FFN prelude emitter cannot reopen.

## Real-model attention-O census amendment

The d512 native-NV census was run in fresh control and attention-O-only
processes after P0 became green. It fails the topology gate:

| arm | programs/token | E programs | residual adds | fused O | new `86a2` adapters |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 946 | 490 | 72 | 0 | 0 |
| attention-O | 982 | 526 | 36 | 36 | 72 |

The generated token is identical (`38835`), but token equality cannot promote
a construction which grows the graph. Full logits and wall timing were
therefore not run.

The two adapters per fused O are now identified exactly:

1. activation: `CONTIGUOUS -> CAST(fp16) -> ... -> PERMUTE -> ... -> AFTER`;
   this is real fp32-to-fp16 work unless the independently measured M5 fp16
   flash-combine/output-layout route is active;
2. residual: `RESHAPE -> RESHAPE -> MEMORY_SEMANTIC -> CONTIGUOUS ->
   GETTUPLE(FUNCTION(precompile=True))`; the invocation output promise exists,
   but the explicit `CONTIGUOUS` sits outside it and is intentionally not
   treated as a view without an output-layout proof.

This closes the attention-O emitter in isolation. Its exact reopen condition
is a composed construction where (a) the M5 fp16 combine contract removes the
activation adapter, (b) callify proves the block result's explicit contiguous
is the same zero-offset/equal-span invocation output buffer, and (c) the real
census becomes at most 910 programs/token (`946 - 36 residual adds`) with zero
new adapters. Only then may full logits and a reverse native A/B run.

FFN-down stays closed because its old emitter additionally recomputes SiLU per
output row. Block-output contiguous stays closed on the same missing explicit-
contiguous/output-buffer proof. No remaining safe projection family has a
topology-shrinking construction under the current contracts.

### Composed reopen attempt

The exact M5 + attention-O composition was then constructed. Its first real
census, before changing callify, reached 911 programs with 37 `86a2` adapters:
the fp16 combine removed its original activation-copy population, but explicit
block-output materializations still fed the residual epilogue. A proposed
consumer-side elision recognized only `CONTIGUOUS(GETTUPLE(precompiled
FUNCTION))` and rebuilt the enclosing `RESHAPE`/`MEMORY_SEMANTIC` views. CPU
tests passed for a plain function result, but the real d512 schedule failed
before execution with `ValueError: bad reshape: () -> (1, 1, 4096)` in
`rangeify.cleanup_dead_axes`. No artifact, logits, or wall result was accepted.

The failure is reproduced hermetically by the real nested ownership spelling:
the precompiled function returns `MEMORY_SEMANTIC(CONTIGUOUS(...))`, and the
caller adds another `MEMORY_SEMANTIC(CONTIGUOUS(GETTUPLE(...)))`. Callify's
`_precompiled_output_redirect` only redirects a top-level `CONTIGUOUS` or a
physical BUFFER/MULTI. It cannot prove this nested owned result aliases the
invocation output; the unmodified CPU schedule retains four ordinary
materialization programs before the epilogue. Stripping the caller's wrapper
alone is therefore both insufficient and, on the real graph, shape-unsafe.

The experimental elision was reverted. The composed route is closed. Its exact
reopen condition is a callify-owned extension which understands
`MEMORY_SEMANTIC(CONTIGUOUS(result))`, preserves the resolved output-slot shape
and ownership, and returns the dependency-bearing `AFTER` without a second
store. That requires dedicated callify alias/lifetime tests; it cannot be
implemented safely as broader `custom_kernel` movement stripping. Block zero
must remain unfused because its embedding/gather residual is not a precompiled
block output. No further GPU retry is authorized until the hermetic nested
schedule has exactly producer + epilogue and no extra materialization.

### Callify-owned reopen amendment

The required extension is now implemented at callify, without broadening
`custom_kernel` movement stripping. A function-owned
`MEMORY_SEMANTIC(CONTIGUOUS(value))` is redirected into its resolved invocation
output slot only when dtype and exact shape agree. After `FUNCTION -> CALL`, a
caller `CONTIGUOUS` may collapse through only `RESHAPE` and
`MEMORY_SEMANTIC`; it must retain one precompiled `CALL` dependency, refer to
exactly one output argument of that call, and preserve dtype and element span.
Permutes, offsets, wrong spans, non-precompiled calls, and ambiguous output
slots fail closed.

The hermetic matrix is green:

* the exact nested block spelling schedules only the producer and projection
  epilogue, with no materialization program between them;
* two-output functions retain distinct output buffers, the same invocation
  dependency, and their respective semantic owners;
* wrong-span and intervening-movement constructions are rejected;
* TinyJit warmup/capture/replay with changing physical inputs returns the
  current invocation's value, not a stale output allocation.

This reopens only the real construction census. It does not yet reopen logits
or timing. The next accepted row must keep block zero ordinary, fuse blocks
1--35, contain zero `86a2` adapters, produce token `38835`, and have at most
875 programs/token (`946 - 36 residual adds - 35 redundant block-output
materializations`). A census miss stops before numerical or wall qualification.

### Independent redirect qualification

The redirect was isolated from every projection route behind
`CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT` (default `0`). The OFF arm is the
exact legacy materialization behavior; the ON arm changes only this callify
contract. Fresh d512 native-NV censuses give:

| arm | programs/token | E programs | residual-add class | `86a2` copies | token |
| --- | ---: | ---: | ---: | ---: | ---: |
| redirect OFF | 946 | 490 | 72 | 1 | 38835 |
| redirect ON | 875 | 419 | 36 | 1 | 38835 |

The exact executable-histogram delta is one 71-count materialization class,
`E_fab82...`, disappearing. The old `E_02a...` class also falls by 36, but a
new `E_81c...` class appears 36 times, so that hash substitution is net zero
dispatches and must not be claimed as separate recovery without semantic
classification. There is no adapter increase. The one `86a2` row is present
in both arms and is unrelated baseline work.

Full-vocabulary qualification is bit exact over 8 rows of shape
`[1, 151936]`: both arms have SHA-256
`71c0a2b092cbc2e40c22b42cd4f6f3c84fe56fd40f2bfd008efc5b76be0ae0f0`,
`array_equal=True`, and maximum absolute difference `0.0` across 1,215,488
float32 elements. Native TinyJit replay also passes with changing physical
inputs.

The reverse wall sequence discarded the compile-contaminated first sample in
each fresh process. Settled OFF samples were `5.5475752`, `5.54904515`,
`5.5566532`, and `5.55401185` ms/token; their bracket median is
`5.5515285`. Settled ON samples were `5.4705162` and `5.4824793`; their median
is `5.47649775`. The isolated recovery is `0.07503075` ms/token, or `1.3515%`
of the OFF wall. Every timing repetition has the same token hash. This is a
qualified candidate, but the switch stays default-off pending explicit
promotion.

### Composed attention-O verdict

The pre-scoped composition was then measured relative to redirect ON, keeping
block zero ordinary and enabling the existing fused attention-O emitter only
for blocks 1--35. It fails before numerical or wall qualification:

| arm | programs/token | E programs | residual adds | fused O | `86a2` copies | token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| redirect ON | 875 | 419 | 36 | 0 | 1 | 38835 |
| attention-O composed | 874 | 418 | 1 | 35 | 71 | 38835 |

The one-program gross shrink is misleading. The exact histogram delta is:
`E_02a... -35`, `E_0a5e... -36`, `E_86a2... +70`, fp32-combine to
fp16-combine `-36/+36`, and plain-GEMV to fused-GEMV `-35/+35`. Thus 70 new
copies--exactly two per fused block--replace 71 other E programs for a net
`-1`; the kernel substitutions are count-neutral. The current adapter-slot
tracer did not resolve those copies to concrete epilogue arguments, so their
precise ownership remains an analysis question, not a reason to waive the
gate. No full-logit or wall run was made for this arm. The emitter stays
closed; none of the independent redirect recovery is credited to it.

## Boundary

No code/default/policy has been promoted. This result does not debit the
parity ledger and does not reclassify llama's fused epilogues. It closes only
the currently available custom-kernel/typed-boundary constructions.
