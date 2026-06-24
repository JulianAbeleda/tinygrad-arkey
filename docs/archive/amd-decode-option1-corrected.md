# Option 1, scoped — the TC blocker is DTYPE (fp32), not fusion. It's a model fix, not codegen surgery.

Date: 2026-06-15. Scoping "option 1" (let the decode forward use tensor cores) corrected the root-cause
diagnosis and made the fix far simpler than feared.

## The correction (I had it wrong)
The earlier "fusion blocks TC" diagnosis was WRONG -- it came from a standalone `(x@W.T).silu()` test where
`helper_realized_ast` returned the split-off silu kernel (no reduce), not the matmul. Capturing the ACTUAL
error on the model's matmul kernels: **"no tensor core available"**, and dumping the reduce structure:

    reduce.arg=Ops.ADD dtype=float  src0=MUL  inputs=[dtypes.float, dtypes.float]   <- fp32 x fp32

The model's verification matmul runs in **fp32**. RDNA3 WMMA only supports fp16/bf16 inputs. Verified
cleanly:
- **fp16 matmul + TC = 16.31 TF (APPLIES)**
- **fp32 matmul + TC = "no tensor core available" (ERRORS)**

So the no-TC plateau is a **dtype** problem, not fusion, not layout, not symbolic batch. (Those earlier
walls were real but downstream of this -- once the dtype is fp16, TC is reachable.)

## The fix is a model-side fp16 cast (NOT a tinygrad codegen change)
Casting the FFN matmul inputs to fp16 (`model.py _feed_forward`, Q4K_UNFUSE path) -> the MUL becomes
fp16xfp16 (verified: dump shows `inputs=[dtypes.half, dtypes.half]`), and **TC applies** (warmstart
`apply:1`). No renderer/scheduler surgery needed -- the existing TC opt works once the dtype is right.

## What remains (engineering, not walls)
1. **Opts that apply to all matmuls**: the down matmul errored on `UNROLL:0:2` ("2 can't divide 3" -- the
   factored dim 12288=256x16x3). Fix: pick TC opts that divide the factored kernel dims (or PADTO). The
   loop should search the model's ACTUAL kernel layout, not isolated A@B.
2. **Cast/overhead vs payoff**: at T=16 the fp16 cast + unfuse + only-1/3-TC'd netted SLOWER (25 vs 18
   ms/tok). The matmuls are a fraction of the forward (Amdahl), and the per-matmul cast adds overhead.
   The win needs: (a) the WHOLE verification forward in fp16 (no per-matmul cast), (b) TC on all matmuls,
   (c) a batch large enough that matmul compute dominates.

## Corrected option-1 scope (the real, tractable path)
NOT "add fused-TC primitives to tinygrad". Instead:
1. **Run the verification forward in fp16 end-to-end** (so every matmul is WMMA-eligible and there's no
   per-matmul cast overhead). This is a dtype-policy change in the model's decode/verification path.
2. **Tune the model's actual (factored) matmul kernels** with the loop -- find TC opts that apply to all of
   them (divisibility-aware), via the curated-config search on the real kernel layout.
3. **Measure net e2e** at the speculative batch (K=16+), where the matmul fraction is largest.
Honest expectation: the per-kernel win is 2x (proven), but the e2e plateau drop is Amdahl-bounded by the
matmul fraction; the realization is now an engineering/measurement question, not an expressibility wall.

## Why this matters
The whole "tinygrad can't express TC for decode" conclusion was too pessimistic: TC IS reachable on RDNA3
for the verification matmul -- the blocker was a dtype policy (fp32 compute), fixable in the model. The
TileLang-class vocabulary argument still holds for FUSION (the genuine W2 wall), but the IMMEDIATE
decode-TC lever is a dtype fix on our stack, no DSL needed. That reorders the options: try the fp16
verification path FIRST (cheap, our hardware), before any tinygrad-codegen or tile-DSL investment.

Artifacts: model.py Q4K_UNFUSE (now fp16 cast), warmstart hook + WARMSTART_DUMP diagnostic (postrange.py),
extra/qk_decode_warmstart.py. All default-off; normal decode unchanged.
