# Boltbeam Q6_K Vocabulary Output Route Scope

## Objective

Make the real model-forward Q6_K vocabulary projection compile on AMD gfx1100 after the one-buffer attention route is attached, then validate both Qwen3 8B and 14B through the pinned Boltbeam-compatible prefill harness.

## Known failure

Both pinned smoke runs reach model compilation and fail with the same HIP diagnostic:

```text
expression is not assignable
make_float32(val43.x,...) = make_float32((*(buf0+...))*val44...)
```

The generated kernel has `151936` output columns, which is the Qwen vocabulary projection. The isolated one-buffer attention kernel is not the failing operation.

## Corrected route identity

The full model does not reach the Q4_K vocabulary emitter; it selects the Q6_K cooperative route. The independent route trace shows:

- 8B uses `q6k_gen_coop_151936_4096`.
- 14B uses `q6k_gen_coop_151936_5120`.
- The model route is Q6_K, selected through `q6k_primitive_linear_call` and the Q6 cooperative emitter.
- The Q4_K scalarization commits are therefore dead code for this failure.
- The likely failing boundary is the downstream `partial.sum(axis=1)` reduction after Q6 partial output, not the Q4 direct-output store.

## Why the previous fixes were insufficient

The named and untagged full-vocabulary guards were added to the known `extra/qk/q4k_prefill_route_spec.py` direct-output emitter. Its compile-only probe is clean, but the complete model forward still emits the old vector lvalue. The actual model route is the Q6_K cooperative emitter and its downstream reduction.

## Required work

1. Capture the complete failing model-forward program identity and source with a debug compile, including kernel name, output shape, route family, and emitter module.
2. Trace that identity to the Q6 cooperative vocabulary emitter and its downstream partial-sum reduction; do not patch by error text alone.
3. Add the smallest structural guard for the vocabulary width at the actual Q6 boundary. Use scalar-addressable reduction stores for the vocabulary output. Preserve existing tiled/upcast schedules for ordinary prefill roles.
4. Prove the changed emitter in a compile-only model-forward capture. The old `make_float32(val43...)` assignment must be absent, and the candidate identity/source hash must be recorded.
5. Run an isolated numeric check for the vocabulary projection and the one-buffer attention path. Require full-output parity and no device faults.
6. Run pinned Boltbeam-compatible smoke for:
   - 8B: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`, profile `qwen3_8b_q4k_m_gfx1100`.
   - 14B: `/home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf`, profile `qwen3_14b_q4k_m_gfx1100`.
   Use CPU affinity core 0, `--pin-clock`, identical warmup/round settings, and separate artifacts.
7. Require route census, model-forward producer/body/tail provenance, whole-model numeric/quality parity, pinned clock provenance, and exact binary/candidate identity joins. Keep 14B fail-closed if no exact generated candidate is registered.

## Non-goals

- Do not weaken candidate identity, parity, route-census, or clock-pinning gates.
- Do not register the 14B `K=5120` route without an exact generated payload/hash and matching evidence.
- Do not modify the one-buffer attention math or duplicate a second attention implementation.
- Do not treat a compile-only pass or an unpinned smoke as model promotion.

## Completion criteria

The scope is complete only when both model smokes compile and produce artifacts. 8B must show the exact one-buffer attention identity in the model-forward census and pass pinned numeric/timing gates. 14B may either pass with its own exact generated route or produce a structured fail-closed report naming the missing candidate/evidence; it must not silently fall back while claiming coverage.

## Integration completion addendum

The route integration itself is now established for the 8B attention projection: the exact one-buffer candidate is attached only under its phase, invocation, role, and shape guards. The remaining model-forward work is the vocabulary/logits boundary that runs after attention and is independent of attention math.

### Required call-path invariants

- `output.weight` must be identified as `lm_head` before route selection.
- Q6_K vocabulary projection must use the exact direct scalar-output primitive for `(M,N,K)=(512,151936,4096)` on 8B.
- The route must not enter packed-WMMA for that exact projection.
- The final logits conversion must write scalar-addressable lanes; no constructed `make_float32` or `make_half32` value may appear on the left-hand side of a store.
- Ordinary Q4/Q6 prefill roles, attention routes, and decode routes must retain their existing schedules.
- 14B `(512,151936,5120)` remains declined until an exact generated payload, identity, and model-forward evidence join exists.

### Execution sequence

1. Capture the full model program source under `CCACHE=0 DEBUG=4` and record the kernel identity, input/output shapes, and route owner.
2. Trace the final half conversion from `tinygrad/llm/prefill_routes.py` through the selected Q6/direct-output primitive and any logits consumer.
3. Add one structural scalar-lane guard at the actual producer/consumer boundary. Do not add another parallel attention route or patch by generated variable name.
4. Compile the complete 8B forward and assert the old vector-lvalue forms are absent.
5. Run full-output nonconstant numeric parity for the vocabulary projection and attention output.
6. Run pinned Boltbeam 8B with 3 warmups and 10 synchronized rounds, recording clock provenance, route census, model-forward identity, binary/source hashes, and timing samples.
7. Run the same Boltbeam command for 14B. If admission declines, emit a structured fail-closed artifact naming the missing exact candidate/evidence; do not treat fallback as promotion.

### Stop conditions

- Any fix that changes ordinary roles or decode behavior is out of scope and must be reverted from the candidate path.
- A compile-only pass without a model-forward route census is insufficient.
- A smoke exit without a JSON artifact is incomplete.
- An unpinned timing result cannot satisfy the promotion gate.
