# NV attention-O post-callify copy trace scope

Date: 2026-08-05.  This is a CPU/static amendment to the composed P2b
attention-O verdict in `nv-p2b-projection-boundary-reopen-record-20260805.md`.
It neither opens the route nor claims a performance recovery.

## Why the old census is insufficient

The composed redirect-on / fp16-combine / fused-O census is real and remains a
hard fail:

| arm | programs | residual adds | fused O | new `E_86a2` copies |
| --- | ---: | ---: | ---: | ---: |
| redirect-on | 875 | 36 | 0 | 0 new |
| composed O | 874 | 1 | 35 | 70 |

The previous tracer reported `adapter_slots: []`.  It compared every copy's
`src[1]` with every epilogue input.  That is not a valid post-rangeify
attribution rule: a scheduled opaque CALL has no declared output list, a
writer may have more than one store, and a copy can feed an intermediary before
it reaches the epilogue.  The old JSON therefore establishes the count, not
the ownership of any of the 70 copies.

The older pre-callify input snapshot is useful direction only: it showed a
flash activation cast/layout path and a nested block-output path.  It cannot
establish that either present `E_86a2` writer is an alias-only operation after
callify/rangeify.  In particular, a dtype conversion, non-zero offset, layout
permutation, or a non-output invocation allocation must continue to copy.

## New read-only trace

`extra/llm_research/decode/nv_projection_epilogue_qualification.py` now
contains `post_callify_copy_trace(linear)`.  It observes the exact executable
linear graph after recursive callify and rangeify, without changing its graph,
allocator, schedule, or execution:

1. It finds each `E_86a2` CALL.
2. It reads STORE destinations in the scheduled call body to recover the exact
   written PARAM argument slot(s).  It does **not** assume slot zero is output.
3. It follows the written physical buffer identity through all consumer CALL
   argument slots, including any intermediary transport calls, until branches
   terminate at a `q4k...epi_resadd_4096_4096` call or another sink.
4. It records program name, argument slot, shape, and dtype at every edge.

The next composed *census only* artifact must include
`post_callify_86a2_trace`.  Its 70 records must account for every writer and
every consumer branch.  This instrumentation is covered on CPU by
`test_post_callify_trace_links_a_materialization_writer_to_epilogue_slot` in
`test/unit/test_projection_epilogue_boundary_census.py`: a non-precompiled
boundary is mapped through its actual scheduled writer buffer to the residual
epilogue argument, with the 4096-fp32 ABI asserted.  The full file passes
(`11 passed`).

## Decision rule

There is no new generic contract in this amendment.  A contract may be
proposed only if every eliminated copy has all of the following proof in the
trace and a corresponding hermetic construction:

- its producer writes exactly one invocation-owned precompiled output slot;
- the consumer reads that same physical output buffer at identical dtype,
  element span and zero offset;
- the chain between them contains only `RESHAPE` and an owner-preserving
  `MEMORY_SEMANTIC` carrier—no cast, permute, shrink, expand, or ordinary
  materialization;
- producer/consumer dependencies remain attached to the same `AFTER(CALL)`;
  multi-output slots and aliased input/output buffers are rejected.

If and only if a trace class meets that rule, the narrow implementation site is
callify's owned precompiled-output contract, not `custom_kernel` movement
stripping or an emitter change.  The implementation must be closed-default,
have separate wrong-span/movement/multi-output/replay tests, and be scoped to
the proven call argument role.  A class containing an fp32<->fp16 conversion
or layout change is computation/layout work, not an ownership contract; it
requires a separately typed producer/consumer ABI and does not qualify here.

## Gate before any GPU timing

The trace itself can be collected alongside the already-authorized census;
no numerical or wall run is admitted by this document.  Before any GPU timing
of a proposed contract:

1. CPU hermetic tests must reproduce each observed real trace class and prove
   that only the exact alias class is removed.
2. Redirect-on + fp16-combine + fused-O census must have **at most 804
   programs/token** (`875 - 35` residual adds `- 36` combine/cast class), with
   zero new adapters.  The target is deliberately stricter than the old net
   874 count: the 70 copies cannot be hidden by a gross dispatch subtraction.
3. Block zero remains unfused.  Full logits and reverse A/B/A wall timing are
   then required as separate gates.

Until the trace makes the producer chains concrete, the correct result is
`UNPROVEN`, not an inferred residual or activation alias.  The composed route
therefore remains closed and has zero ledger credit.

## First traced-census attempt

One fresh composed d512 census was run after the Q4 microgate released the GPU
lock.  It hard-failed the construction gate before logits or timing:

| programs | E programs | residual adds | fused O | total `E_86a2` | generated token |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 876 | 420 | 1 | 35 | 71 | 38835 |

One `E_86a2` is common baseline work, so the composed route still introduces
70.  The result is 72 programs above the required 804 ceiling.  No numerical,
wall, or ledger claim follows.

The attempt also found an instrumentation attachment error: during TinyJit
capture, `Tensor.linear_with_vars` hands its real schedule to
`capturing[0].add_linear` and deliberately returns an empty LINEAR sentinel.
The first observer examined that empty sentinel, producing an empty
`post_callify_86a2_trace` despite the independently counted 71 executions.
An authorized observability retry read those newly appended capture LINEARs.
It reproduced the same `876 / 71 / 1 / 35` structure and token `38835`, but
also returned an empty trace.  The second attachment point was still too
early: pre-memory-plan captured SINKs call generic elementwise bodies named
`test`; the rendered `E_86a2` identity and immutable output-slot table exist
only in `TinyJit.captured.linear` after `jit_lower` and compile.

The observer now reads retained compiled decode captures containing the exact
`epi_resadd_4096_4096` family.  For PROGRAM calls it uses `ProgramInfo.outs`
rather than reconstructing STORE PARAMs; it continues to use physical buffer
identity for the producer-consumer edges.  A second CPU proof compiles the
synthetic boundary and establishes that its materialization writer reaches
epilogue argument slot 3.  The focused file now passes (`14 passed`).  No
further GPU census is authorized by this record.  Consequently the exact
real writer/consumer ownership remains unproven and no alias contract was
implemented.

A final authorized observability census again reproduced the exact
`876 / 71 / 1 / 35` structure and token `38835`, but the retained-capture trace
was empty.  The measured decode was TinyJit's `cnt == 0` eager arm: the first
prompt yield is produced by prefill, and the DEBUG-wrapped next yield executes
the first decode before any `CapturedJit` exists.  Thus a retained compiled
capture cannot be the authority for this particular census.

The instrumentation is now attached to the eager `compile_linear` boundary,
after memory planning and PROGRAM rendering and before execution, and restores
the original compile functions immediately after the census.  This is the
first point holding all three facts simultaneously: the rendered `E_86a2`
identity, `ProgramInfo.outs`, and the actual memory-planned producer/consumer
buffers.  Under the explicit no-fourth-census stop it has CPU proof only; it
has not produced a real-model trace.  The terminal evidence classification is
therefore unchanged: copy ownership `UNPROVEN`, no safe alias contract, route
closed with zero credit.
