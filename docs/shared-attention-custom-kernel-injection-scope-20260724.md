# Fused Attention via Custom-Kernel Injection — Scope (2026-07-24)

## Strategic pivot
Stop fighting the general composite-reduce lowering (the entire class-2 saga). Instead, inject the
already-proven fused kernel directly via the `custom_kernel` escape hatch, and let the existing (working)
compiler handle everything else. "Use what we have as a base" = the captured kernel IS the base.

## Why this dodges class-2 (and the whole reach-through/forwarding/cycle mess)
Class-2 is caused ONLY by the composite reduce fusing the projection/KV-cache provenance of Q/K/V into
itself. The custom_kernel approach makes attention an **opaque kernel with buffer inputs**: the compiler
realizes Q/K/V as ordinary buffers (which it already does correctly — the SDPA path proves the full model
schedules fine) and treats the attention as a `CALL`. No composite reduce → no reach-through, no
store-forwarding, no split_store cycle. The problem is architecturally removed, not patched.

## Grounding (verified)
- `Tensor.custom_kernel(*inputs, fxn=...)` exists (tensor.py:194, ops.py:1256) and is already used in
  production (prefill_routes.py:197,201 — the Q4_K vocab route builds `kernel = format.emit(route_spec)`
  then `out.custom_kernel(packed_weight, activation, fxn=kernel)`).
- The proven kernel source EXISTS (capture harness emits `.hip.cpp` + `.amdisa.s` + JSON metadata). Its
  signature is already a clean 4-buffer ABI:
  `void E_1024_32_32_<hash>(half* out[32*512*128], half* Q[32*512*128], half* K[8*512*128], half* V[8*512*128])`
  Scale + causal are baked as CONST (matches the postrange admission contract). fp16 in, fp16 out.

## DTYPE: keep it ORTHOGONAL (answer to the open question)
Do NOT do dtype lowering inside the attention path, and it is worth being deliberately orthogonal now:
- The attention kernel is a **pure fp16 island**: `half*` Q/K/V in, `half*` out. The model already casts
  Q to half at the boundary (model.py:616) and K/V are fp16 (cache). So the fp16 contract is already met
  by the existing path — no new dtype work.
- ALL Q4_K/dequant/quant dtype handling stays UPSTREAM in the existing (working) projection kernels. That
  separation is precisely what avoids class-2 (the bug was the Q4_K dequant leaking INTO attention).
- Consequence: dtype and attention are decoupled. We can land fp16 attention now and revisit any
  dtype/precision changes (e.g. bf16, fp8 KV) later as an independent axis, without touching this kernel.
- Only re-open dtype if a geometry needs a non-fp16 kernel; then it's a separate capture, not a lowering
  inside this path.

## What we REUSE (the base)
- The captured kernel (`amd_gfx1100_q16_grid_hd128_loop_attention`) — the hard, proven part (254 VGPR/0
  spills, 3.7-4.4x, numerically correct to ~6e-5).
- `generate_shared_attention_captures` as the per-geometry kernel GENERATOR (content-addressed already).
- The eligibility/admission logic (grid_shape checks, AMDAttentionGridSpec.validate), the loud class-2
  guard, the candidate-context/profile plumbing.
- `custom_kernel` + the `format.emit(route_spec)` fxn-builder pattern from prefill_routes as the template.

## What is NEW (the rewrite — a thin injection route)
1. A `fused_attention_custom_kernel(q, k, v, *, scale, causal, ctx)` helper: geometry lookup -> select the
   captured kernel -> build the custom_kernel `fxn` (kernel source + launch/grid/block + param binding,
   analogous to `format.emit`) -> `Tensor.empty(out_shape).custom_kernel(q, k, v, fxn=...)[0]`.
2. Model integration: model.py fused branch calls this instead of `shared_prefill_attention(...)`. Same
   eligibility guard; SDPA fallback otherwise. Output cast back to q.dtype (as today).

## Tasks (dependency-ordered)
- B1. Reverse the custom_kernel `fxn` contract from prefill_routes (`format.emit`): what the callable must
  return (a kernel UOp / Program from source + launch dims + param ownership). Smallest de-risking step.
- B2. Kernel packaging: generate + store captured kernels for needed geometries (8B Hq32/Hkv8, 14B
  Hq40/Hkv8) x prefill KV lengths. The capture is per-candidate_context; a matrix or a JIT-symbolic KV.
- B3. Build `fused_attention_custom_kernel` (geometry -> kernel -> custom_kernel call). Start with the ONE
  captured geometry (T=512,KV=512) to prove the path end-to-end.
- B4. Wire into model.py; guard + SDPA fallback.
- B5. A4 acceptance: custom_kernel output vs SDPA next-token numerics (8B, then 14B). The real gate.
- B6. A3: confirm the injected kernel actually fires (CALL in schedule, kernel name).
- B7. Variable KV length: capture matrix vs parameterized kernel (the kernel bakes q_tokens/kv_tokens).
- B8. A5-A8 tail (benchmarks KV 512..4096, decode-nonregression, proof collector, BoltBeam route) — now
  reachable because attention no longer crashes scheduling.

## Risks / unknowns
- custom_kernel `fxn` API details (how the raw kernel source + launch config are injected; grid/block
  binding). Mitigation: prefill_routes' packed-weight route is a WORKING template to copy.
- Geometry coverage: kernel bakes q_tokens/kv_tokens (and scale as CONST). Prefill KV varies 512..4096 and
  q_tokens per chunk. Either a capture matrix (finite, content-addressed) or a JIT-symbolic-dim kernel.
  Causal/scale baked -> per-(causal,scale) capture or parameterize.
- Decode is a SEPARATE path (flash_decode_attention_route already exists); this scope is prefill only.
- Perf: custom_kernel adds a Q/K/V materialization boundary (extra buffers) vs a hypothetical fully-fused
  path -- but the isolated kernel already assumes materialized Q/K/V, so this matches its proven profile.

## Relationship to prior work
Class-1 fix (d51bd3e92) and the loud class-2 guard (2ebdb2e15) stay. The composite-reduce lowering is left
as-is (not deleted) but is no longer on the critical path for enablement. The full class-2 diagnosis
(scope-A doc) remains the record of WHY the general path was abandoned for enablement.
