# Typed semantic CALL-input reopen: CPU structural record

Date: 2026-08-05
Status: **CPU structural gate passes; production topology gate fails; route closed**

## Exact blocker addressed

The latest production trace has a typed RMSNorm marker below an opaque
consumer call rather than a selector-visible store:

```text
REDUCE_OUTPUT(fp16)
  -> MEMORY_SEMANTIC(owner=RUNTIME_SCRATCH)
  -> RESHAPE(equal span)
  -> CONTIGUOUS
  -> CALL
```

Consequently, a `STORE(..., REDUCE_OUTPUT)` selector cannot be reached by
peeling another wrapper at the store stage.  The boundary is the call input.

## Closed construction

`CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER` is a new default-zero gate.  Before
ordinary early callify materializes `CONTIGUOUS`, a top-down prepass admits
only the literal spelling above and creates one precompiled producer
invocation:

```text
CALL(SINK(STORE(output_slot, REDUCE_OUTPUT(...))), x_param, weight_param, output_slot)
  -> AFTER(output_slot, CALL)
  -> RESHAPE(equal span)
  -> original consumer CALL
```

The producer has exactly two direct PARAM input views and one explicit output
slot `(2,)`; its output retains the original `RUNTIME_SCRATCH` owner.  There
is no custom kernel, movement stripping, inferred capture, or transport
COPY/CONTIGUOUS adapter.  The ordinary late `REDUCE_OUTPUT` store selector
then owns the producer body.

The matcher rejects default-off operation, PERMUTE movement, a non-scratch
owner, and aliased input/weight.  It also requires equal dtype/span and the
exact single semantic carrier.

## CPU evidence

`test/unit/test_reduce_output_rmsnorm.py` passes **32 tests**.  The new
end-to-end structural case performs the same top-down prepass followed by the
normal early callify pass, recursively schedules the producer `SINK`, and
observes exactly:

```text
["reduce_output_rmsnorm_1_4096"]
```

Its consumer input is `RESHAPE(AFTER(...))` and its transitive graph contains
neither `COPY` nor `CONTIGUOUS`.

## Next gate (not authorized/run here)

Run the existing production capture census with this one gate enabled and all
other route settings held fixed.  Admit no wall test unless it reports the
predeclared topology: **36 `reduce_output_rmsnorm_1_4096` programs, at most
839 total programs, and zero adapter copies**.  Then require full-logit
correctness and reverse A/B/A wall timing; until those gates pass, recovery is
zero and the feature remains default-off.

## Production topology result

Authorized direct-greedy d512 capture ran with both
`CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1` and
`CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1` under the GPU lock.  It reported:

```text
dynamic reduce_output_rmsnorm programs: 0 (required 36)
total programs:                         875 (required <=839, but irrelevant)
```

The surviving fp16 chain was byte-for-byte still:

```text
REDUCE_OUTPUT -> MEMORY_SEMANTIC(RUNTIME_SCRATCH) -> RESHAPE
  -> CONTIGUOUS -> CALL
```

Thus the new top-down callify prepass did not encounter the actual parent
CALL in the production construction phase.  The CPU construction proves the
mechanism once that parent relation exists, but not the earlier construction
stage at which production creates it.  This is a construction-order blocker,
not a numeric or kernel verdict.  The fixed 36/839 topology gate failed, so
no logits or wall A/B/A test was run and the route remains default-off with
zero booked recovery.
