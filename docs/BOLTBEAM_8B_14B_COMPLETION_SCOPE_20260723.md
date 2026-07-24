# Boltbeam 8B/14B Completion Scope

## Objective

Produce truthful end-to-end Boltbeam evidence for both Qwen3 8B and Qwen3 14B prefill while reusing the shared compiler, attention, route, and timing assets. Do not duplicate attention implementations or weaken fail-closed admission and promotion gates.

## Shared invariants

- Attention uses the centralized score-resident composite path; no model-specific attention fork is allowed.
- The one-buffer LDS candidate is candidate-only until its exact model-forward identity appears in route census and numeric artifacts.
- Every authoritative timing run uses `--pin-clock`, CPU affinity core 0, identical warmups/rounds, synchronized samples, and recorded clock provenance.
- Every artifact joins model profile, workload, route identity, source hash, binary hash, numeric proof, and timing samples.
- A process exit, compile-only capture, memory plan, or partial log is not a Boltbeam result. A JSON artifact is required.

## Track A: 8B runtime route completion

### Current facts

- Model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`.
- Profile: `qwen3_8b_q4k_m_gfx1100`.
- Target attention route: exact one-buffer `attn_qo` candidate at `(M,N,K)=(512,4096,4096)`.
- Q6 vocabulary projection: `output.weight`, `lm_head`, `(512,151936,4096)`.
- The direct scalar Q6 vocabulary primitive and attachment guard exist.
- Full-model runs still fail before artifact creation on a `make_half32(...)` vector lvalue, proving the runtime tensor is falling through to another packed-WMMA or logits path.

### Policy activation finding

The current live 8B model policy is `prefill_v2=True`, `prefill_tc_attn=False`, `candidate_id=direct-packed-baseline`, `strategy=DIRECT_PACKED_FALLBACK`, `measured=False`, and `graph_gemm=None`. `_graph_gemm_binding` correctly declines because no audited graph-GEMM candidate registry or policy rows exist. Enabling the shared attention route therefore requires constructing and admitting an audited graph-GEMM candidate policy; a binding-only toggle would be dishonest.

### Required implementation

1. Start the exact smoke command under `CCACHE=0 DEBUG=4` and capture the complete generated source and route census.
2. Construct or locate the audited graph-GEMM candidate policy for the score-resident attention route. Require canonical payload/source identity, exact geometry, resource proof, and numeric proof before attaching it.
3. Set `prefill_tc_attn` only through that admitted policy; do not flip the flag without a candidate registry row.
4. Record the runtime linear’s immutable attachment identity, `PrefillDirectPackedBinding`, route id, role, shape, and selected callable.
5. Prove whether `_exact_q6k_vocab_direct_prefill` executes for `output.weight`; if it does not, fix the attachment/selection guard rather than the emitter.
6. Ensure the exact Q6 vocabulary path bypasses packed-WMMA and uses scalar-addressable fp32 output storage.
7. Ensure final logits consumers preserve fp32 scalar output; no generic AMD vector-store patch is permitted.
8. Compile the full 8B forward with `CCACHE=0`; assert no `make_float32`, `make_half32`, or prior failing kernel identity remains.
9. Run full-output nonconstant numeric parity for attention and vocabulary/logits output.
10. Run pinned Boltbeam smoke with 3 warmups and 10 synchronized rounds at `pp512`.
11. Require a complete JSON artifact containing route census, model-forward one-buffer identity, source/binary hashes, clock provenance, numeric parity, raw samples, and promotion decision.

### 8B completion criteria

- JSON artifact exists and is non-empty.
- Route census passes with no missing, unexpected, or identity-mismatched entries.
- Model-forward census names the exact one-buffer attention identity.
- Q6 vocabulary route is the exact scalar route, not packed-WMMA fallback.
- Full-output numeric parity passes on nonconstant inputs.
- Timing has pinned clock provenance and 10 raw synchronized samples.
- Any failure remains explicit and fail-closed; no fallback is labeled as promotion.

## Track B: 14B memory and exact-route completion

### Current facts

- Model: `/home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf`.
- Profile: `qwen3_14b_q4k_m_gfx1100`.
- Exact vocabulary shape: `(M,N,K)=(512,151936,5120)`.
- Q6 scalar reduction proof exists for `151936x5120`, but no exact model-forward candidate is registered.
- Current admission at `max_context=4608` declines before model load because available VRAM and the planner’s budget are insufficient.

### Required implementation

1. Establish a memory baseline before loading the model: free VRAM, weights, KV bytes per 1K context, prefill peak, allocator reserve, and route workspace.
2. Choose a bounded test context that fits, starting with `512`; do not silently reduce the requested context in an authority artifact.
3. If needed, free VRAM by terminating stale GPU processes or use a clean process; record the before/after device state. Do not change unrelated system settings.
4. Add an exact generated candidate for `qwen3_14b_q4k_m_gfx1100`, Q6_K, `output.weight`/`lm_head`, `(512,151936,5120)` only after source, binary, numeric, and resource evidence exist.
5. Bind the candidate through the same shared route and attachment machinery as 8B. Do not create a second attention implementation.
6. Compile the complete 14B forward with `CCACHE=0 DEBUG=4`; assert exact route identity and absence of vector-lvalue diagnostics.
7. Run nonconstant full-output numeric parity for vocabulary and attention.
8. Run pinned Boltbeam at the bounded context with 3 warmups and 10 synchronized rounds.
9. If memory admission or exact route evidence is unavailable, emit a structured fail-closed JSON artifact naming the missing resource or identity join. Never claim 14B promotion from an ordinary fallback.

### 14B completion criteria

- Memory plan is accepted for an explicitly recorded context.
- Exact `K=5120` candidate payload/source/binary identity is registered and joined.
- Route census proves the candidate was selected in the real model forward.
- Full-output numeric parity passes.
- Pinned clock provenance and raw timing samples exist.
- If any prerequisite remains unavailable, the artifact is structured fail-closed with `promotion_eligible=false` and precise missing fields.

## Shared test matrix

Run each model in this order:

1. Memory preflight and route inventory.
2. `CCACHE=0 DEBUG=4` compile/source capture.
3. Nonconstant numeric canary.
4. Pinned Boltbeam smoke: `taskset -c 0`, 3 warmups, 10 rounds, `--pin-clock`, explicit artifact path.
5. Route-census and artifact completeness validation.
6. Promotion decision: pass only if every identity, parity, resource, and timing gate joins.

## Non-goals

- No generic renderer scalarization without a captured owning route.
- No 14B registration by copying the 8B payload or identity.
- No hidden fallback acceptance.
- No unpinned timing substituted for pinned authority.
- No duplicated attention or model-specific compiler path.
