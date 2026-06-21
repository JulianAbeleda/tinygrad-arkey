# Performance Primitive Research Principles

This document is a focused supplement to [Coding Principles](coding-principles.md).

It does not replace the core project principles. It applies only to GPU performance work,
quantized inference primitives, machine-search experiments, and kernel/codegen research
where isolated wins can easily mislead.

The core principles still govern the work:

- centralize authority
- modularize execution
- abstract for simplicity
- keep concerns orthogonal
- encode invariants
- contain dangerous power
- test behavior at the boundary

This supplement explains how those rules apply when the work is performance-primitive
research instead of ordinary product code.

## Core Rule

A performance primitive is not just an operation.

A performance primitive is:

```text
math
+ data layout
+ activation format
+ memory path
+ work decomposition
+ reduction strategy
+ compiler lowering
+ scheduling/register behavior
+ integration boundary
```

Do not declare a primitive understood until all required parts are included in the
measurement.

## Measure The Whole Primitive

Isolated kernels are useful for diagnosis, not final truth.

A benchmark is incomplete if it excludes required work such as:

- activation quantization
- q8 packing
- score materialization
- dequant/unpack work
- scale/min decode
- partial reductions
- graph/JIT dispatch behavior
- data layout conversion
- quality loss

Examples from this repo:

- `READRAW` reached llama-class bandwidth because it skipped dequant/unpack.
- `sudot4` improved the Q4_K kernel but lost whole-linear once q8 pack was included.
- TC prefill attention won in isolation but failed in-model because score materialization dominated.
- ring overlap worked in microbenchmarks but decode remained HBM-bound.
- speculative decode acceptance was strong but the runtime loop made the real path slower.

Rule:

```text
If required work is excluded, call the result diagnostic, not a win.
```

## Prefer In-Model Gates Over Proxy Wins

The final gate for runtime performance work is the real model path.

Use this order:

1. source/render audit
2. microkernel diagnosis
3. isolated role benchmark
4. whole-linear or whole-primitive benchmark
5. in-model W==D benchmark
6. quality/correctness gate
7. default decision

Do not skip directly from a microkernel win to a model route.

A path may ship only when the relevant boundary passes:

- decode: in-model token/s, W==D, greedy correctness or explicit quality gate
- prefill: pp throughput, warm forward, dNLL where lossy, no decode regression
- quantized matvec: q8/dequant/pack/reduction costs included
- runtime overlap: application workload, not only independent-kernel overlap

## Separate Diagnostic, Candidate, And Shipped States

Every performance result should be labeled as one of:

- diagnostic: explains a bottleneck, not a candidate
- candidate: passes an isolated or whole-primitive gate, not model-shipped
- shipped: passes in-model gate and has fallback/control
- refuted: failed the relevant gate with a recorded reason
- deferred: promising but blocked by a specific missing capability

Do not let diagnostic artifacts become implicit authority.

## Use Reference Implementations As Oracles, Not Ceilings

llama.cpp, CUDA, ROCm, Metal, BitBLAS, Marlin, AWQ, SmoothQuant, and similar systems are
references.

They are not ceilings and not specifications.

Use them to answer:

- what math is actually performed?
- what data layout is used?
- when is activation quantized?
- how is work mapped to lanes/warps/blocks?
- where are reductions performed?
- which instructions are emitted?
- how much required work is avoided, fused, or amortized?

Do not copy a surface feature without auditing the full primitive.

Bad:

```text
llama uses dp4a, so add dp4a.
```

Good:

```text
llama uses packed q4 extract, q8 activation, signed dot4, qsum/min correction,
per-group scale, and a specific scheduler. Which of those are missing here?
```

## A Primitive Name Must Include Its Dataflow

Do not name search rows by one instruction.

Bad names:

- `dp4a`
- `TC attention`
- `ring2`
- `flash`

Better names:

- `q4k_ffn_mmvq_q8_with_pack_cost`
- `prefill_attention_tc_materialized_scores`
- `decode_attention_gqa_coop_vec`
- `q6k_lm_head_coop_k_partial_sum`
- `q4k_mmvq_sudot4_128row_with_q8_pack`

The name should make the required data movement and integration boundary visible.

## Audit Before Building Deeper

When a primitive fails, do not immediately build the next variant.

First ask which layer failed:

- math/formula
- data layout
- memory coalescing
- activation format lifecycle
- reduction strategy
- instruction selection
- compiler lowering
- register scheduling
- graph/runtime integration
- quality tolerance

Build the next probe only after the failing layer is named.

The MMVQ campaign followed this pattern:

- coalescing-bound Q6_K roles shipped
- Q4_K FFN local coop refuted
- q8/dp4a refuted until signedness was audited
- signed dot4 fixed
- 128-thread scheduler tested
- whole-linear q8 pack refuted the route
- next frontier became activation format lifecycle, not another dot kernel

## Activation Format Is Part Of The Primitive

A fast int-dot kernel is not useful if the activation format is expensive to produce.

For W4A8-style paths, include:

- q8 pack time
- q8 scale computation
- q8 layout conversion
- reuse count across linears
- quality error
- whether the graph can common or cache the packed value
- whether the previous operation can emit q8 as an epilogue or side-channel

Rule:

```text
Kernel speed minus activation-pack cost is the primitive speed.
```

If q8 pack cancels the kernel win, the target becomes activation lifecycle, not another
matvec kernel.

## Coalescing, Register Tightness, And Reduction Are Tradeoffs

Do not optimize one in isolation.

Common tradeoff:

```text
one-thread-per-row -> register-tight but uncoalesced
cooperative lanes -> coalesced but needs partial reduction and shared state
hand-tuned backend -> balances both with scheduler/register control
```

When testing a scheduler shape, record:

- threads per row
- rows per block
- K split
- reduction location
- output writes
- scale decode location
- q8 reuse
- native instruction emission
- occupancy implications

A variant that improves one side but regresses the other may be useful diagnostically but
should not be routed.

## Decode T=1 Must Preserve Parallelism

Decode has a different shape than prefill. Prefill has a multi-query/token axis. Decode usually
has `T=1`, so a primitive must create enough parallel work from other axes.

Rule:

```text
For decode T=1, a primitive must manufacture enough parallel work from KV splits and
GQA/query-head columns while preserving reuse. Fusion or LDS reuse that reduces workgroups
is harmful, even if it removes kernels or memory reads.
```

This rule was added after the llama decode audit corrected the attention hypothesis:

- llama's T=1 decode attention path is a non-WMMA vector `flash_attn_tile`, not the WMMA prefill path;
- its win comes from many KV-split parallel blocks, LDS K/V staging, GQA query-head column packing, and register
  online softmax;
- tinygrad's raw fused flash and fused LDS+GQA scalar tiles were byte-exact but slower because they collapsed the
  wrong axes or failed to preserve enough parallel work;
- prefill-proven LDS tile structure did not transfer because decode lacks the prefill multi-query axis.

Do not treat these as sufficient objectives for decode:

- fewer kernels;
- fusion alone;
- LDS reuse alone;
- GQA reuse by serializing G heads;
- beating a weak global-reread baseline.

Before timing a decode attention candidate, answer:

- how many workgroups does it create at ctx512/1024/4096?
- does it preserve query-head/GQA parallelism instead of serializing it?
- does reuse reduce redundant memory traffic without collapsing occupancy?
- is online-softmax/combine work hidden under enough parallel KV work?
- does it compare against the current winner, not a weaker baseline?

For this repo, the current decode-attention comparator is `gqa_coop_vec`, and a new vector-tile candidate must beat
that comparator before any in-model route work is justified.

```text
Apply this principle to the existing winning primitive and its split parameters before building a new hand-tile.
```

This was learned the hard way (2026-06-21): new hand-tiles (scalar fused LDS+GQA, warp-cooperative) were
byte-exact but slower than `gqa_coop_vec`'s **matmul** q·k. The principle's lever — more KV-splits — was instead
cheapest and most effective applied to the existing winner: lowering `FLASH_L` (128→64) gave more workgroups and a
~1.08× attention win @ctx1024 (passing the standalone gate), where a from-scratch tile could not. **But that win
did not promote:** whole-decode W==D was +1.8%@1024 and **−1.2%@4096** — below the ≥5% bar and regressing long
context, because tinygrad's split-combine cost caps the useful split count well below llama's many-split regime.
So: the bounded decode-tile lane is rested; the remaining lever is the full llama-style `flash_attn_tile`
lifecycle with an **efficient many-split / stream-k combine** (north-star), not another bounded tile or flag
sweep. Do not promote `FLASH_L=64` by default. See `docs/decode-vector-flash-tile-realigned-result-20260621.md`.

## Value Semantics Beat Source Emission

A lowering test must validate computed values, not just generated source or disassembly.

This matters for architecture-specific instructions.

Example from this repo:

- a helper labeled signed dot4 emitted a native dot4-looking instruction
- source/disassembly checks passed
- value tests later proved it behaved unsigned
- several conclusions were distorted until value-level tests were added

Rule:

```text
Every new lowering or intrinsic needs value tests with signed, unsigned, negative, zero,
and edge-case lanes.
```

## Machine Search Needs Rows, Not Wishes

A machine-search row must state:

- primitive name
- current implementation
- reference implementation
- required dataflow
- legal knobs
- correctness/quality gate
- isolated gate
- in-model gate
- expected Amdahl impact
- known refutations
- required fallback

Do not search a space until the row names the full primitive boundary.

Example shape:

```json
{
  "primitive": "q4k_ffn_mmvq_sudot4_with_q8_lifecycle",
  "phase": "decode",
  "required_work": ["q8_pack", "packed_q4_extract", "signed_dot4", "qsum", "scale_epilogue"],
  "ship_gate": "in-model W==D >=5% and quality accepted",
  "kill_gate": "whole-linear fails after pack cost included"
}
```

## Stop Conditions Must Match The Goal

There are two valid modes.

Shipping mode:

```text
If the scoped gate fails, stop and do not route.
```

Research mode:

```text
If the scoped gate fails, identify the next layer and decide explicitly whether to fund it.
```

Do not confuse the two.

A failed shipping gate does not prove the research space is exhausted.

A research continuation does not justify shipping or defaulting a failed candidate.

## Keep Fallbacks And Authority Centralized

If a performance path ships:

- route it from one authority point
- add an explicit fallback flag
- test that unsupported shapes fall back
- test that unrelated roles are untouched
- update the current-state docs/cache
- mark older diagnostics as superseded or refuted

Do not leave multiple unofficial ways to activate the same experimental path.

## Quality Is A First-Class Gate

Lossy paths need quality gates even when speed passes.

Examples:

- q8 activation paths need dNLL or equivalent validation before defaulting
- demotion policies need measured quality budgets
- fp16/TC prefill paths need dNLL, not only greedy smoke

If speed fails, quality can be skipped for that candidate because it will not ship.

If speed passes, quality is mandatory before default.

## Document Refutations As Assets

A refutation should explain:

- what was tested
- what passed
- what failed
- why the old hypothesis changed
- what should not be reopened
- what next layer, if any, remains

Refutations are part of the search map. They prevent repeated work.

## Summary

For GPU primitive research in this repo:

1. Define the full primitive boundary.
2. Include required data movement and format work.
3. Audit reference implementations before copying them.
4. Use source, value, micro, whole-primitive, and in-model gates in order.
5. Treat activation lifecycle as part of matvec primitives.
6. Do not ship lossy or proxy-only wins.
7. Separate shipping discipline from research continuation.
8. Convert findings into machine-search rows.
9. Keep authority centralized and fallbacks explicit.
10. Record refutations as durable knowledge.
