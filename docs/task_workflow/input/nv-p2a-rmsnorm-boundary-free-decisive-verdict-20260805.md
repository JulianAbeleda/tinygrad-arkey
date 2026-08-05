# P2a RMSNorm boundary-free decisive verdict

Date: 2026-08-05  
Target: Qwen3-8B-Q4_K_M, d512, RTX 5090 / native `DEV=NV`  
Status: **NO-GO for the cheapest scheduler-native construction; policy remains closed.**

## Question

Can the existing default-zero typed-call-input construction turn the live
RMSNorm chain into an ordinary, graph-replayed one-program producer without a
new materialization?

The exact admitted construction was deliberately narrow:

```text
REDUCE_OUTPUT
  -> MEMORY_SEMANTIC(RUNTIME_SCRATCH)
  -> RESHAPE(equal span)
  -> CONTIGUOUS
  -> opaque consumer CALL
```

It would replace the two ordinary RMSNorm programs with the existing ordinary
`reduce_output_rmsnorm_1_4096` program, preserving the consumer dependency as
an `AFTER` edge.  It is closed by default and has no custom kernel, source
adapter, view stripping, or relaxed movement predicate.

## Fresh decisive production census

The gate was rerun from the current campaign worktree, under the shared GPU
lock, with only the two required closed contexts enabled:

```bash
flock -w 30 /tmp/gpu-bench.lock env DEV=NV PYTHONPATH=. \
  python3 scratchpad/nv_reduce_output_rmsnorm_census.py \
  --depth 512 --max-context 1024 --chunk-size 32 \
  --typed-semantic-producer \
  --out /tmp/nv-p2a-rmsnorm-production-census-20260805.json
```

Result:

| required topology condition | observed |
| --- | ---: |
| `reduce_output_rmsnorm_1_4096` programs | **0** |
| total programs | **875** |
| new materializations attributable to this route | 0 (because no route admitted) |

The trace proves the marker is present through function substitution and
callify (`70` candidate observations and `74` ordinary observations), but only
three candidate and four ordinary structural representatives remain before
rangeify.  None becomes a selector-visible `STORE(REDUCE_OUTPUT)`.  The two
live terminal shapes are:

```text
fp32: REDUCE_OUTPUT -> RUNTIME_SCRATCH -> RESHAPE -> CAST -> STORE
fp16: REDUCE_OUTPUT -> RUNTIME_SCRATCH -> RESHAPE -> CONTIGUOUS -> CALL
```

The fp32 form has an output-dtype cast before its store.  The fp16 form enters
an opaque consumer call before the late STORE selector.  The exact typed
producer prepass does not create a producer invocation for that live parent
relation, so its ordinary late lowering never has a body to select.  This
reproduces the prior zero-admission result on the current composed campaign
state; it is not a stale-baseline inference.

## Why no predicate widening is admissible

Widening the selector through `CAST`, `CONTIGUOUS`, or arbitrary `CALL` would
only hide the construction boundary.  It would not preserve the required
allocation/lifetime proof and repeats the class of unsafe late-view inference
that previously produced an Xid 31.  Making `PARAM` sufficient is likewise
unsound: function substitution removes the caller allocation, byte-offset,
and dependency proof.  Neither is a boundary-free scheduler primitive.

The next actual abstraction is materially larger: a consumer-owned,
multi-phase scheduler carrier that keeps a producer's workgroup-shared reduce
state through post-barrier lane restoration *and* lets an opaque consumer own
the producer output contract.  That is a new generic function/scheduler ABI,
not the smallest P2a construction.  It cannot be honestly implemented as a
late wrapper or opaque adapter, and it must be designed independently with
alias/lifetime and multi-phase program proofs before reopening this route.

### Compiler-boundary follow-up

A later CPU-only construction tested whether the campaign's typed
`PostBarrierRegion` could supply that missing lane-restoration lifetime.  It
cannot: that primitive predicates work after an already-expressible barrier,
whereas RMSNorm must restore *all* lanes whose axis was consumed by the scalar
reduction.  The first exact compiler failure is earlier than rendering:
linearization consumes the reopened `LOOP` range, and the current IR does not
allow a loop-control `RANGE` to depend lexically on a void `BARRIER`.

The minimum honest reopen is therefore a first-class post-barrier loop/control
form spanning type validation, range ownership, `END` pairing, CFG ordering,
linearization, and renderer support.  A partial default-off carrier was not
retained: an API whose only valid outcome is a linearization failure would add
surface area without adding capability.  `PostBarrierRegion` remains useful
for producer-only warp retirement, but receives no RMSNorm recovery credit.

## Validation

The hermetic route suite was run on the current worktree:

```text
PYTHONPATH=. pytest -q \
  test/unit/test_reduce_output_rmsnorm.py \
  test/unit/test_reduce_output_rmsnorm_route.py \
  test/unit/test_nv_boundary_free_ordinary_uop_gate.py \
  test/unit/test_rmsnorm_native_lowering_gate.py \
  test/unit/test_rmsnorm_semantic_lowering.py
```

Result: **59 passed**.  These prove the closed fallback and synthetic topology
contracts; the production census above falsifies their reachability for the
current model graph.  Consequently no full-logit test, wall A/B/A, d2048, or
d4096 run is authorized, and P2a receives **zero** recovery credit.
