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

The full model does not reach the Q4_K vocabulary emitter. The independent route trace shows:

- 8B uses `q6k_gen_coop_151936_4096`.
- 14B uses `q6k_gen_coop_151936_5120`.
- The model route is Q6_K, selected through `q6k_primitive_linear_call` and the Q6 cooperative emitter.
- The Q4_K scalarization commits are therefore dead code for this failure.
- The likely failing boundary is the downstream `partial.sum(axis=1)` reduction after Q6 partial output, not the Q4 direct-output store.

## Why the previous fixes were insufficient

The named and untagged full-vocabulary guards were added to the known `extra/qk/q4k_prefill_route_spec.py` direct-output emitter. Its compile-only probe is clean, but the complete model forward still emits the old vector lvalue. Therefore the model is selecting a second Q4_K vocabulary route or a separate downstream emitter/cache path.

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
