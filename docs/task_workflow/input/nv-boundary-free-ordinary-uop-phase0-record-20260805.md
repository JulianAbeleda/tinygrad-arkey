# NV boundary-free ordinary-UOp Phase 0 record

Status: **construction gap confirmed; no candidate promoted or GPU A/B run.**

This reconciles Path 3, M3, and M4 before a new fusion route is attempted.
They all used a custom-program boundary.  That boundary either materialized lazy
inputs or changed replay transport; its one-kernel RMSNorm body was not the missing
generic scheduler feature.  M4 additionally changed projection input/output contracts
and, for FFN-down, recomputed activation.  Neither is an ordinary-UOp epilogue route.

## Exact Phase-0 baseline

`extra/llm_research/decode/nv_boundary_free_ordinary_uop_gate.py` constructs the
ordinary fp16 `(1,4096)` RMSNorm twice: once from a realized activation and once from a
lazy `base + base` producer.  Both lower to exactly two ordinary scheduler programs,
with no `CUSTOM` or `CONTIGUOUS` node in the requested expression:

| input | programs | result |
| --- | ---: | --- |
| realized | 2 | reduction followed by dependent epilogue |
| lazy add | 2 | same reduction/epilogue boundary; producer may fuse into its consumers |

The two-program boundary is semantically necessary in the current generic lowering:
one workgroup computes the row's global sum-of-squares scalar; the output-wide epilogue
then consumes it.  Inlining that reduction into every output lane recomputes the full
4096-wide reduction 4096 times.  Existing rangeify/codegen has no generic cooperative
"reduce once, broadcast within the output workgroup, then store vector output" primitive.

The semantic native route has already proven the contrast: it produces one body but is
`UOp.custom_kernel`, is absent from the ordinary HCQ graph-profile path in the realized
microgate, and adds materializations in a real token.  It therefore fails this phase's
construction contract rather than serving as a candidate.

## Decision

Do **not** modify Path 3, M3, M4, or the generic buffer-removal heuristic.  Removing a
buffer there can only choose recomputation or a materialized scalar; it cannot create a
cross-thread broadcast.  Such a change would be a hidden semantic/performance regression,
not boundary-free fusion.

The smallest admissible next implementation is a generic scheduler/codegen primitive with
all of these properties:

1. A reduction result may feed an elementwise output in the same ordinary `SINK` program.
2. It maps one decode row to one cooperative workgroup, broadcasts the reduction result
   through existing local/register reduction machinery, and writes the output lanes.
3. It is selected structurally (one reduced row feeding an elementwise consumer), preserves
   lazy input indexing, and rejects matmul, multi-row/prefill, movement views it cannot
   index, and arbitrary custom programs.
4. It first passes an isolated realized-and-lazy topology/profile gate, then a single
   RMSNorm family real-token A/B at d512.  Only a positive reverse bracket advances it to
   d2048/d4096 and to one disjoint projection epilogue family.

### Concrete implementation seam (reviewed against current lowering)

The required primitive cannot be implemented by `remove_bufferize`.  At that stage, the
ordinary scalar reduction has been scheduled as a separate buffer result.  Removing that
buffer substitutes the 4096-wide reduction at every epilogue output lane, which is
recomputation, not a workgroup broadcast.

The minimal generic design is instead a new, default-off `Ops.REDUCE_OUTPUT` semantic
carrier, lowered before `run_rangeify` only when all of the following structural facts are
proven:

* one reduced scalar (or fixed small vector) has exactly one elementwise consumer output;
* the consumer also reads the reduction's unreduced input through an indexable identity
  view; and
* one logical output row has a static bounded width and no aliasing/movement view that the
  generated indexer cannot preserve.

Its codegen lowering needs a provider-independent AST recipe: map one row to one group,
reduce chunks into `AddrSpace.LOCAL` using the existing group-reduce machinery, barrier,
let every output lane read the final scalar, and write its disjoint output elements.  CUDA
and AMD/Metal providers only supply existing group/barrier syntax; targets with no provider
return the normal two-program source.  The carrier's first source must remain that normal
source, exactly as `Ops.RMSNORM` does today.

Required hermetic tests before any NV execution:

1. carrier validator rejects multiple consumers, non-identity/lazy-unindexable views,
   multiple rows, prefill, matmul/reduce chains, and unsupported dtype;
2. lowering emits one normal `SINK`/CALL, not `CUSTOM`/`custom_kernel`, with no
   `CONTIGUOUS` or intermediate global scalar store;
3. rendered CUDA AST contains one local reduction, barrier, and output stores; CPU/other
   unsupported targets retain the exact two-program fallback; and
4. a small interpreter/reference test proves each output lane uses the same reduced scalar,
   rather than an inlined per-lane reduction.

Only after those pass is the NV included-cost microgate authorized.  A failed construction
routes back to this design seam; it must not be promoted through an environment flag.

### P2 construction attempts and precise compiler blocker

Two concrete scheduler-native constructions were exercised after this scope was written:

1. A single `FUNCTION` body containing a `GROUP_REDUCE` over the 4096 input values and a
   dependent 4096-wide ordinary output store.  The function inlines normally and uses no
   custom source or assembly, but rangeify splits it into two programs: a scalar reduction
   store followed by the output epilogue.  This is the same boundary as the original graph.
2. The same construction with a one-slot `CompositeReduce`,
   `DEFERRED_REDUCE_OWNER`, and `DEFERRED_REDUCE_SLOT`, mirroring the proven attention
   state path.  It also becomes two programs.  A scalar deferred slot has scalar logical
   ownership; it cannot keep a wider elementwise output inside its reduction kernel.

This localizes the missing abstraction more precisely than the original `REDUCE_OUTPUT`
name.  Current scheduler IR has no way to say:

> keep this workgroup-shared scalar live after its reduction barrier while restoring the
> reduced lane axes as output ownership in the same program.

`DEFERRED_REDUCE_SLOT` works for attention because its primary accumulator already owns
the output vector lanes; the scalar denominator is secondary state.  RMSNorm is the
opposite: the only reduced state is scalar, while the output lanes are the same axes that
were reduced.  Making the RMS input a vector accumulator would either require a physical
4096-lane register value or incorrectly reduce distinct output elements across lanes.

The exact implementation seam is therefore **post-reduction lane restoration**, not a
new renderer instruction.  It needs a typed carrier whose metadata records:

* which `GROUP_REDUCE` axes are restored for the post-barrier epilogue;
* the scalar state slots that are broadcast to those lanes; and
* the original logical input views that may be re-indexed in the restored phase.

Rangeify must keep the carrier and its epilogue in one kernel rather than bufferizing the
scalar.  Late reduction lowering must emit `END(reduce axes) -> BARRIER -> restored lane
phase -> output END`.  `gpudims` must map the restored axes to the same local IDs, not
allocate a second launch dimension.  This is a new multi-phase kernel lifetime primitive,
akin to the existing pipeline/state phase ABI, not an extension of ordinary REDUCE or a
target-specific provider.

No NV timing was run for these attempts: both fail the required hermetic one-program gate.
The existing one-program hand body remains evidence that the hardware algorithm is valid,
but routing it through an opaque call would repeat Path 3 and violate this scope.

This is substrate work, not a small model-route patch.  Until that primitive exists, the
correct action is to prioritize the shared-Q8 V/K and FFN-down tracks plus predispatch;
the 574.654-us norm ownership allocation must not be booked as recoverable fusion credit.

## Validation

`pytest -q test/unit/test_nv_boundary_free_ordinary_uop_gate.py` passes.  The GPU
realized-input A/B was rerun under the campaign lock on 2026-08-05: custom semantic
topology remains one program and exact, but is +58.324 us/replay versus ordinary, so it
does not reopen this gate.
