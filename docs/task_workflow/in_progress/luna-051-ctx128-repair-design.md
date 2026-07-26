# LUNA-051: Context-128 STORE Semantic Repair Design

Verdict: `TOOL_FAILURE`.

Start commit: `923f6ff02` (feature base). Evidence commits were applied on this
child branch before this analysis; the static causal comparison remains blocked.

## Evidence boundary

The reported 14B ctx128 failure is a pre-timed-decode COMGR rejection of a
constructed vector on the destination side of a HIP assignment. It is not a
decode-flash failure. The retained scope records `_warm_depth -> _prefill ->
Transformer.generate -> compile_linear`; it does not retain the failing UOp
STORE, the generated HIP function, its route identity, or an 8B/512 control
from the same capture regime.

The required LUNA-030, LUNA-031, and LUNA-034 artifacts are absent. LUNA-021
also records `TOOL_FAILURE`: installed tracing collection did not yield a
positive-control dispatch. Therefore a source-only design cannot truthfully
name the semantic owner of the illegal STORE.

## Exact source route boundary

For an attached direct/bounded-packed prefill linear, the production decision
is in `tinygrad/llm/prefill_routes.py`:

1. `_attached_production_route` authorizes only an immutable attachment.
2. `_attached_direct_packed_spec` additionally requires active prefill scope,
   matching role, and exact `(m,n,k)` binding.
3. `route_prefill_linear` calls `route_packed_wmma_prefill` first, except for
   the exact Q6_K vocabulary exception.
4. `route_packed_wmma_prefill` declines when the candidate, its frozen
   `(quant, role)` geometry, its canary, or its entry creation declines.
5. Only then does `route_prefill_linear` call `route_direct_packed_prefill`.
6. If the direct route is not exact-bound or declines, execution falls through
   to the attached FP16 graph GEMM or ordinary `Tensor.linear`.

This establishes a *candidate fallback*, not attribution: a ctx128 32-token
prefill chunk cannot reach direct-packed unless its attachment has that exact
physical `m=32` shape. The retained source map says the production custom
prefill-attention route itself is admitted only at `q_tokens=512`; that does
not identify the failing linear kernel.

## Known source-level mitigation already present

`extra/qk/prefill/packed_wmma_prefill_candidates.py` keeps the packed-WMMA
matmul output rank two with `.contiguous()` before the metadata-only
`reshape(1,m,n)`. Its stated invariant is that the producer STORE owns a
scalar-addressable rank-two output lane; it must not fuse a final batch view
into a vector accumulator store. This is a plausible repair layer only for a
selected packed-WMMA candidate. It does not prove that the failing ctx128
program selected that candidate, nor does it establish token/KV correctness.

The direct-packed Q4_K LM-head path separately removes output upcasts for the
full-vocabulary extent in `_direct_packed_opts`. That is a route-specific
safety rule, not a generic renderer repair and not evidence for ctx128.

## Intended destination semantics and invariants

The only sound repair, if the captured owner is packed-WMMA output lowering,
is to preserve a materialized rank-two global output buffer at the matmul ABI,
then apply the `(1,m,n)` batch reshape as metadata after the STORE. The repair
must preserve all of the following:

- Every logical `(m,n)` output lane has one addressable global destination.
- Vector lanes are values used to compute/store scalar lanes, never STORE
  destinations unless lowered to an addressable vector memory view.
- The final batch reshape does not allocate, reorder, or change output values.
- Activation shape, quant packing, role binding, causal/KV positions, and
  prefill route identity remain unchanged.
- 8B and ctx512+ code objects stay unchanged unless separately captured and
  validated.

Naming or materializing the constructed vector merely to make HIP compile is
rejected: without the UOp ancestry it could discard or redirect lane updates.

## Candidate decision

No implementation candidate is admitted. The smallest safe action is *not* a
new global compiler transformation or a blanket fallback guard. A fail-loud
guard at `route_direct_packed_prefill` is admissible only if LUNA-034 positively
shows that 14B ctx128 reaches this fallback after a packed candidate decline;
it must report the declined `(quant, role, m, n, k)` and must not be presented
as a ctx128 recovery. If the capture instead names an `r_toks_*` generic
lowering owner, the candidate belongs in that lowering stage and must be
reviewed as a distinct repair.

## Required focused controls before code

1. Retain the 14B ctx128 semantic STORE ancestry and HIP excerpt.
2. Capture a passing 14B ctx512 selected packed-WMMA route.
3. Capture a passing 8B ctx128 program under the same collector policy.
4. Positively capture an allowed non-14B direct-packed selection.
5. Compile the three programs and compare destination address space, lane
   count, STORE source indices, and final output shape.
6. Only after a candidate compiles, run deterministic finite/token and KV
   bounds controls. GPU decode/prefill execution was not run for this card.
