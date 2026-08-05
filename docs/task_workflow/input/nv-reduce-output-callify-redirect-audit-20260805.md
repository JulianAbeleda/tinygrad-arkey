# REDUCE_OUTPUT audit after generic precompiled-output redirect

Date: 2026-08-05
Verdict: **three production censuses admitted zero; invocation-local hermetic correction did not reach the real model graph; production route NO-GO**

## Question

Does `CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT`, which independently reduced
the closed-route production graph from 946 to 875 programs without changing
the token, also provide the missing contract needed by the ordinary-UOp
`REDUCE_OUTPUT` RMSNorm selector?

## Static answer

It provides two previously missing facts:

1. the caller consumes the precompiled invocation's actual output allocation,
   rather than a copied `CONTIGUOUS` adapter; and
2. the consumed value retains its producer dependency as
   `AFTER(output_buffer, CALL)`.

That is enough for `lower_reduce_output_store` when it runs directly on the
post-callify graph. A hermetic device-tagged NV construction reaches
`reduce_output_rmsnorm_1_4096` in `get_kernel_graph`; no GPU is involved.

Before the follow-up implementation, it was not enough for the real recursive
scheduler. The same construction through `create_linear_with_vars` produced
three generic `test` programs and zero `reduce_output_rmsnorm` programs.

## Exact loss point

The loss is after callify bufferization but before the outer consumer's
rangeify selection:

```text
callify:
  AFTER(output_buffer, CALL(SINK, ..., output_buffer), precompile=True)
  -> physical identity and dependency are both exact

recursive pm_schedule:
  CALL(SINK, ..., output_buffer)
  -> CALL(LINEAR, ..., output_buffer)

outer lower_reduce_output_store:
  _precompiled_output_after_view(...)
  -> rejects because it can no longer prove the output-slot STORE
```

`_precompiled_output_after_view` currently reopens the producer body and
requires a `STORE(PARAM[output_slot], ...)`. Once the body is a scheduled
`LINEAR`, the visible STORE belongs to an inner scheduled call and its PARAM
numbering is no longer the outer precompiled invocation's slot numbering.
The proof therefore disappears even though the exact buffer and CALL
dependency remain unchanged.

The existing test
`test_callified_precompiled_output_retains_exact_after_dependency` stops before
recursive scheduling, which is why it passes while production still falls
back.

## Generic implementation

The follow-up carries the fact callify already proved instead of rediscovering
it from a later lowered body:

1. An immutable `precompiled_output_slots: tuple[int, ...] = ()` field was added to
   `CallInfo`.
2. `transform_precompiled_call` populates it with the exact appended output
   argument slots whose body PARAMs were constructed and stored by callify.
3. Scheduling preserves the field when rebuilding `CALL(SINK)` as `CALL(LINEAR)`.
4. `_precompiled_output_after_view` continues requiring exact dtype/span,
   the dependency-bearing `AFTER`, and physical equality with the referenced
   CALL argument; accept a scheduled `LINEAR` body only when that argument slot
   is present in `precompiled_output_slots`.

This is invocation metadata, not a custom kernel or a new value-path op. It
does not infer identity from shape, PARAM position, or a generic `AFTER`, and
the redirect remains closed by default.

## Decisive hermetic tests

The full-scheduler positive now passes:

```python
with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1):
  linear, _ = marked_precompiled_producer_output.linear_with_vars()
assert [call.src[0].arg.name for call in linear.src].count(
  "reduce_output_rmsnorm_1_4096") == 1
```

The device-tagged NV construction executes no GPU work and produces exactly
two programs: the producer plus `reduce_output_rmsnorm_1_4096`. Separate tests
prove that the redirect-off default, a movement view, an unlisted multi-output
slot, an input/output alias, and an extra AFTER dependency all decline.

## Validation performed

- Direct post-callify selector: REDUCE_OUTPUT body present.
- Full recursive scheduler after implementation: `test` producer plus exactly
  one `reduce_output_rmsnorm_1_4096` body.
- Redirect-off full scheduler: zero REDUCE_OUTPUT bodies.
- Focused callify, projection ownership, JIT replay, REDUCE_OUTPUT, movement,
  multi-output, alias, and lifetime suite: 29 passed, 2 deliberately excluded
  because they are device-dependent numerical tests rather than static
  selector contracts.
- `CallInfo` and `DiagnosticCallInfo` pickle round trips retain the new field.
- No GPU, timing, default change, route promotion, commit, or push.

## Production census result

The first authorized production census ran under timeout and the shared GPU
flock, writing only to `/tmp`:

```bash
flock -n /tmp/gpu-bench.lock env DEV=NV PYTHONPATH=. \
  CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1 \
  python3 scratchpad/nv_reduce_output_rmsnorm_census.py \
  --depth 512 --max-context 1024 --chunk-size 32 \
  --out /tmp/nv-reduce-output-rmsnorm-redirect-census-20260805.json
```

The gate failed decisively:

| field | result |
| --- | ---: |
| total production calls | 875 |
| REDUCE_OUTPUT RMSNorm programs | 0 |
| forced marker observations | 290 |
| markers with buffer identity | 0 |
| markers with precompiled-output identity | 0 |

The relevant production norm spelling appeared 72 times as
`MEMORY_SEMANTIC` with a `CONTIGUOUS` base. Other observations were 72 owned
ADD values, 72 Q PERMUTEs, 72 K PERMUTEs, and 2 owned CAST values. Every marker
therefore recorded `ReduceOutputSpec.input_identity_at_marker=False` before
callify could install the newly durable slot proof.

This means the implementation correctly closes the later proof-lifetime gap,
but it is not sufficient to reopen production: the next seam is the
pre-callify owned `MEMORY_SEMANTIC -> CONTIGUOUS` identity contract. Because
the required reducer count and strict topology shrink both failed, no logits,
wall timing, route change, or promotion ran.

## Final CPU-only construction

The follow-up construction exposes only the exact owned spelling as an early
candidate, without claiming that it is already a physical identity:

```text
MEMORY_SEMANTIC(CONTIGUOUS(value), owner=<present>)
```

`ReduceOutputSpec.owned_contiguous_candidate` records that narrow fact at the
Tensor marker. It is deliberately separate from
`input_identity_at_marker`, which remains false. The late rangeify selector
must still prove all of the following after callify:

1. the original dependency-bearing `MEMORY_SEMANTIC` carrier is present;
2. its single inner value is an exact equal-span precompiled-output `AFTER`;
3. the referenced CALL argument is the unique matching physical allocation;
4. the scheduled invocation metadata lists that argument as an output slot;
5. dtype and span are unchanged.

Consequently, the hint cannot admit an ordinary buffer, movement view,
non-CALL materialization, unlisted output, input/output alias, or extra
dependency. The redirect-off default remains byte-for-byte closed.

The expanded CPU-only suite passes 49 tests with 2 device-dependent numerical
tests deliberately excluded. It includes the real nested spelling
`runtime_activation(precompiled_producer(x).contiguous())` through the full
recursive scheduler, where the marker records identity false and candidate
true, and the final schedule contains exactly the producer plus
`reduce_output_rmsnorm_1_4096`. `git diff --check` is clean.

## Predicted production gate

The failed census gives a falsifiable count before another GPU run. Its
program histogram contains 73 ordinary `r_16_256` RMSNorm reductions and 72
matching `E_32_32_4_f14` epilogues. The 72 candidate observations were
collected across two traces, so they represent exactly 36 attention-norm
sites in the final decode graph. The 36 FFN norms remain ADD-rooted and the
single output norm remains CAST-rooted; neither is eligible for this route.

The next production census must therefore satisfy all three gates:

| field | required result |
| --- | ---: |
| `reduce_output_rmsnorm_1_4096` programs | 36 |
| total production calls | 839 |
| topology delta from redirect-only control | -36 |

The arithmetic is `875 - 36 = 839`: each accepted site replaces one generic
reduction plus its epilogue with one REDUCE_OUTPUT program. Any different
count is a construction mismatch and stops before logits or timing. No GPU
recensus has run for this final construction yet; the shared lock remains
with the active timing owner.

## Second production census and corrected loss point

The candidate-only recensus completed under the shared flock and falsified the
36/839 prediction:

| field | observed |
| --- | ---: |
| `reduce_output_rmsnorm_1_4096` programs | 0 |
| total production calls | 875 |

It therefore hard-stopped before logits or wall timing. The GPU lock was
released immediately.

The failed prediction came from a non-production-shaped hermetic test. That
test placed `REDUCE_OUTPUT` outside the precompiled producer. In the model, the
attention norm marker lives inside the next block's precompiled `_run`
consumer. `_function` substitutes the external owned block output with
`PARAM[0]` in that function body:

```text
caller argument:
  MEMORY_SEMANTIC(CONTIGUOUS(GETTUPLE(previous FUNCTION)))

consumer body after function input substitution:
  REDUCE_OUTPUT(..., PARAM[0], ...)
```

The candidate bit survived on the marker, but its prior producer AFTER and
physical output proof existed only on the enclosing CALL argument. The late
selector correctly rejected the bare PARAM. This exactly accounts for the
second zero-admission census.

The corrected construction carries invocation-local proof across that
boundary without treating PARAM as identity. During callify, a candidate
marker is annotated with an input slot only when:

1. its body source is the exact equal-span `PARAM[slot]`;
2. the same enclosing invocation argument is an exact owned pre-call
   precompiled output or a unique dependency-bearing post-call `AFTER`;
3. the referenced prior CALL lists that unique argument as a durable output
   slot; and
4. the redirect context is explicitly enabled.

Rangeify then accepts only the exact PARAM whose slot equals that annotation.
The outer CALL still consumes the producer AFTER, so recursive call resolution
retains the actual dependency; the body never infers lifetime from PARAM shape
or position alone.

A production-shaped nested test now changes from three generic programs to
exactly `test` producer plus `reduce_output_rmsnorm_1_4096`. Redirect-off,
non-call owned-contiguous, and movement-view nested inputs all remain rejected.
The expanded focused suite passes 51 tests with 2 device-dependent numerical
tests excluded, and `git diff --check` is clean. This authorized one final
census-only 36/839 hard gate; it did not establish a numerical or wall claim.

## Third census and closure

The invocation-slot construction received one final census-only authorization.
It again completed with zero `reduce_output_rmsnorm` programs and 875 total
calls. This falsifies the production effect of the nested hermetic correction
even though that correction accurately reproduces one real function boundary.
The hard stop ran: no logits, wall timing, route change, or promotion followed,
and the shared GPU lock was released.

## Instrumented production-stage trace (amendment)

The prior three censuses established *that* the selector admitted zero.  They
did not establish whether the marker was erased before rangeify, whether the
marker reached a concrete STORE in a different spelling, or whether the
selector rejected an otherwise visible candidate.  A diagnostic-only trace was
therefore added.  It is gated by `REDUCE_OUTPUT_TRACE`; it is not part of UOp
identity, eligibility, cache keys, defaults, or the route policy.

The final d512 production census was run under the already-authorized
redirect-on construction, with no logits or wall timing:

```text
after FUNCTION input substitution: candidate 70, ordinary 74
after callify:                    candidate 70, ordinary 74
before rangeify STORE selection:  candidate  3, ordinary  4
parent of each of those 7:         MEMORY_SEMANTIC 7
selector entries:                  0
final programs:                    0 reduce_output_rmsnorm; 875 total
```

The `70 -> 3` comparison is **not a rewrite loss count**.  The first two
counters are dynamic marker observations over the captured trace; the final
counter is over the small set of unique function bodies which reach
`get_kernel_graph` after schedule caching.  It must not be interpreted as 67
production candidates being erased.  The stable, like-for-like fact is that
all seven unique surviving markers still exist immediately before the late
selector.

The first exact loss of selector visibility is therefore the **semantic
ownership carrier boundary**, not FUNCTION substitution, callify, invocation
slot proof, or rangeify fallback:

```text
visible immediately before selector:
  STORE(..., MEMORY_SEMANTIC(REDUCE_OUTPUT(...)), owner=<present>)

implemented selector pattern:
  STORE(..., REDUCE_OUTPUT(...))
```

Every surviving marker is still under `MEMORY_SEMANTIC`; no direct
`STORE(..., REDUCE_OUTPUT)` exists.  Consequently `lower_reduce_output_store`
is never entered and cannot accept or reject a proof.  This also explains why
the preceding invocation-slot work had no production effect: that work fixed
a proof *inside* a marker, while the actual production marker is hidden from
the only selector pattern before that proof is examined.

### Cheapest valid reopen

Do **not** broaden an identity predicate or promote the route.  The smallest
new scope is a CPU-only structural construction which reproduces exactly one
production spelling:

```text
STORE(target, MEMORY_SEMANTIC(REDUCE_OUTPUT(...), same explicit owner))
```

It must prove that peeling only this carrier preserves the original owner,
does not peel a movement view, arbitrary nested `MEMORY_SEMANTIC`, non-call
materialization, alias, or extra dependency, and still requires every existing
late output/input/weight/device proof.  Only if that hermetic contract selects
one reducer should a new census verify the predeclared `36 reducers / 839
calls` topology gate.  Until then the route remains closed and its recovery is
zero.

The evidence now supports only these bounded conclusions:

1. The generic precompiled-output redirect independently changes production
   topology from 946 to 875 calls.
2. The ordinary REDUCE_OUTPUT selector admits zero production norms in all
   three censuses attempted here.
3. Marker creation, late proof lifetime, the owned-contiguous marker spelling,
   and one nested consumer PARAM substitution boundary have each been
   reproduced hermetically, but closing those seams did not alter production.
4. The next exact loss point in the real model graph is therefore **UNPROVEN**.
   It may be a deeper nested-call rewrite, schedule-cache specialization, STORE
   formation/order, or another concrete argument mismatch; the current census
   does not distinguish them.

This iteration is closed rather than widening PARAM, movement, alias, or
ownership predicates. A future reopen requires read-only stage instrumentation
on the actual model construction that counts candidate markers at each of:
post-function substitution, post-callify, pre-rangeify STORE, selector entry,
and selector rejection reason. Until that ledger identifies the first count
drop, the 36/839 prediction is withdrawn and the REDUCE_OUTPUT production route
remains closed.
