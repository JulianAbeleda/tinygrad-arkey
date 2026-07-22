# Shared flash attention tile-WMMA completion scope

## Feasibility gate

The bounded `Q=16, KV=16, Hd=64` path passes the worth test on AMD with
`USE_TC=1 TC_OPT=2 NOOPT=0`:

- one carrier-owning fused compute `CALL`
- no full score or probability buffer shape
- fp16 output versus fp32 reference: `max_abs_err ~= 5.95e-05`
- scalar fallback and current AMD semantic suites remain green

This proves the score-resident route is worth completing. It does not prove
dual WMMA: the production guard remains fail-closed for composite WMMA.

## Objective

Enable an opt-in tile lowering that keeps QK and PV in registers, performs
online softmax over score tiles, and emits two ordinary AMD WMMA contractions
without materializing the `T x KV` score or probability tensors. The same
semantic and measurement path must serve fp16 and non-fp16 routes.

## Primitive contracts

1. `CompositeTileCarrier` owns score-tile, V-tile, output-tile, lane-axis,
   lane-group, source-range, and `(m,l,acc)` slot metadata.
2. QK produces one scalar score per outer KV tile; raw QK fragment lanes must
   never be consumed as independent softmax scores.
3. V loads use the source rank and output-Hd range. Scalar broadcasting is
   forbidden for lane-shaped V.
4. `m` and `l` are scalar slots; `acc` is a slot-shaped vector/register tile.
5. The tile state merge returns `new_m`, `new_l`, and corrected `new_acc` so
   multi-tile online softmax is mathematically equivalent to the scalar path.
6. `SHAPED_WMMA` descriptors must validate dimensions, lane grouping, source
   ownership, and AMD thread geometry before admission.

## Implementation phases

### A. Scheduler ownership

- Preserve nested QK reduction ownership through pre-bufferization.
- Keep the composite producer in the outer tile range context.
- Ensure cleanup and dead-axis passes preserve slot-specific tuple shapes.

### B. Typed lowering

- Carry `CompositeTileCarrier` through expander and no-range paths.
- Lower grouped V loads from source-axis ranges.
- Allocate slot-specific vector accumulators in `reduce_to_acc`.
- Resolve `REDUCE_SLOT` projections with scalar and vector logical shapes.
- Keep the current scalar fallback selected unless all tile contracts validate.

### C. Tile math

- Compute tile max and correction against running `m`.
- Update `l` and rescale `acc` using the correction factor.
- Normalize score weights in registers.
- Form normalized-weight times V as a WMMA-ready PV operand.

### D. AMD backend

- Map validated score/V/acc fragments to existing WMMA operands.
- Emit QK and PV WMMA in one fused call.
- Reject unsupported shapes, packed layouts, or lane provenance rather than
  silently falling back inside a supposedly fused kernel.
- Capture generated source and AMD ISA markers for both contractions.

### E. Model and geometry promotion

- Exercise shared 8B and 14B routes at contexts 512, 2048, and 4096.
- Cover GQA, causal/additive masks, `Hd=64/128`, fp16, and non-fp16.
- Reuse one geometry policy and one benchmark/evidence schema.
- Verify no route-specific attention implementation is introduced.

## Required gates

1. Scalar CPU and AMD correctness remain green.
2. Exact bounded fp16 comparison stays below the established tolerance.
3. One fused compute `CALL` owns the composite.
4. No full score/probability allocation appears in the allocation census.
5. QK and PV each have source and AMD ISA WMMA evidence in that call.
6. Multi-tile synthetic and bounded attention tests pass for all admitted Hd.
7. Both model profiles have warmed 200-sample timing artifacts, allocation
   census, correctness, FLOP/byte accounting, and roofline efficiency.
8. Promotion remains fail-closed if any artifact is missing.

## Primitive fallback rule

When a phase is blocked, implement the smallest reusable primitive and test it
in isolation: carrier metadata, source-axis resolver, lane-aware load,
slot-shaped tuple projection, tile state merge, or WMMA descriptor validation.
Do not add an attention-specific optimizer carve-out, scalar broadcast, manual
kernel, or unverified performance claim.

## Completion definition

The work is complete only when all gates pass on one immutable revision and the
authority report changes from `NO-GO` to `PROMOTE` for both 8B and 14B. Until
then, the scalar fused path is the production-safe fallback and the tile path
is research-only.
