# Generated Quant Runtime Execution Map

Date: 2026-07-05.

Primary scope: `docs/generated-quant-runtime-architecture-scope-20260705.md`.

Purpose: organize the next work so Q4_K/14B pressure does not create another route-specific fast path.
This document is the short execution map. The architecture scope remains the north-star design.

## Current Position

Completed organizing work:

- Canonical throughput entry: `extra/qk/bench.py`.
- Prefill harness policy: `extra/qk/prefill_harness.py`.
- Decode harness policy: `extra/qk/decode_harness.py`.
- Gate/probe runner: `extra/qk/gate_registry.py`.
- Route inventory/provenance ledger: `extra/qk/route_manifest.py`.
- 8B generated prefill route is closed.
- Q4_K/Q8_1 int8-WMMA Tensor substrate is correct for parity/codegen, but blocked as a full 14B route by graph/materialization scale.

The next runtime work must be descriptor/registry work first. Do not add another Q4_K-specific route branch.

## Execution Rule

Every new optimized runtime path must answer these questions before route wiring:

```text
What RuntimeOpSpec is being served?
What QuantizedTensorSpec / ActivationQuantSpec describes the data?
What GeneratedCandidate owns the lowering?
What provenance allows it?
What gate proves correctness?
What authority harness proves or blocks performance?
What ordinary tinygrad fallback exists if no generated candidate is eligible?
```

If those answers do not exist, the work is still substrate/probe work, not a promotable route.

## Phase 0: Keep Harness And Gate Boundaries Stable

Status: mostly done.

Owners:

- Throughput authority: `extra/qk/bench.py`.
- Prefill policy: `extra/qk/prefill_harness.py`.
- Decode policy: `extra/qk/decode_harness.py`.
- Per-kernel timing: `extra/qk/harness_contract.py::time_fn`.
- Gates: `extra/qk/gate_registry.py`.

Remaining cleanup:

- Migrate older diagnostic `*_ab.py` / `*_wd.py` timing clones to `time_fn` when touched.
- Unify llama-bench wrappers only when model-level bench ownership is decided.
- Do not change `prefill_whole_synced.py` or `decode_runtime_overhead.py` methodology during quant-runtime refactor.

## Phase 1: Runtime Binding Audit

Goal: create a machine-readable and human-readable inventory of route-bound implementations.

Inputs:

- `tinygrad/llm/prefill_routes.py`
- `tinygrad/llm/decode_routes.py`
- `tinygrad/llm/route_ops.py`
- `extra/qk/*`
- `extra/qk/route_manifest.py`

Output:

```text
bench/generated-quant-runtime-binding-audit/latest.json
docs/generated-quant-runtime-binding-audit-20260705.md
```

Classifications:

```text
allowed.tensor_expression
allowed.generated_uop_builder
allowed.codegen_lowering
allowed.renderer_intrinsic
banned.handwritten_kernel_body
banned.source_string_kernel
banned.shape_specific_gpu_path
unknown.investigate
```

Exit criteria:

- Every route manifest row has a current classification.
- Every `custom_kernel` reachable from model runtime is classified.
- Unknowns are explicitly listed with file/function references.

## Phase 2: Descriptor Types

Status 2026-07-05: implemented in `tinygrad/llm/runtime_specs.py`.

Goal: add types only. No behavior changes.

Target module:

```text
tinygrad/llm/runtime_specs.py
```

Initial types:

```text
QuantizedTensorSpec
ActivationQuantSpec
RuntimeOpSpec
GeneratedCandidate
```

Minimal enum/string fields:

```text
family: QuantizedLinear | DenseLinear | FlashAttention | KVCache | ActivationFusion
phase: prefill | decode
role: ffn_gate_up | ffn_down | attn_qo | attn_kv | lm_head | attention | unknown
quant_format: Q4_K | Q6_K | fp16 | fp8 | int8 | unknown
activation_format: fp16 | fp32 | Q8_1 | none
lowering_strategy: packed_dequant_dot | grouped_int_dot_correction | iu8_wmma_grouped_dot | online_softmax_flash
provenance: machine_authored_generated | tinygrad_scheduler_generated | banned | unknown
```

Exit criteria:

- Unit tests cover serialization, validation, and generated-only provenance checks.
- No runtime route behavior changes.

## Phase 3: Build Specs In Existing Routes

Status 2026-07-05: prefill direct-packed route analysis exports `RuntimeOpSpec` metadata through
`PrefillLinearRouteSpec.runtime_op_spec(...)`; runtime behavior and defaults are unchanged.

Goal: make route files describe the operation before selecting any implementation.

First target:

```text
tinygrad/llm/prefill_routes.py::_direct_packed_spec
```

Target behavior:

```text
current: shape/env/quant checks -> concrete implementation
next: shape/env/quant checks -> RuntimeOpSpec -> existing implementation
```

Exit criteria:

- Q4_K and Q6_K prefill both produce a `RuntimeOpSpec`.
- Existing tests still pass.
- Route behavior and defaults are unchanged.
- Strict mode can report the op spec when no route binds.

## Phase 4: Generated Candidate Registry

Status 2026-07-05: implemented in `tinygrad/llm/generated_candidates.py`; the audit gate validates generated-only
provenance and route-id references.

Goal: register generated candidates by op semantics, not by route-specific flags.

Target module:

```text
tinygrad/llm/generated_candidates.py
```

Initial registered candidates:

- `quant_linear_prefill.prefill_v2_scheduler_matmul_default`
- `quant_linear_prefill.q4k_int8_wmma_tensor_substrate`
- `quant_linear_decode.q4k_g3_lanemap`
- `quant_linear_decode.q6k_generated_coop`
- `attention_decode.live_split_flash`

Rules:

- Registry rejects banned provenance.
- Registry can return `blocked` with a reason instead of silently falling through.
- Environment flags remain debug overrides, not primary architecture.

Exit criteria:

- Unit tests prove banned candidates cannot register.
- Candidate selection can be run without importing GPU/tinygrad-heavy modules.

## Phase 5: Quant Specs

Status 2026-07-05: implemented in `tinygrad/llm/quant_specs.py` for Q4_K, Q6_K, and Q8_1.

Goal: move Q4_K/Q6_K properties out of route logic into reusable descriptors.

Target module:

```text
tinygrad/llm/quant_specs.py
```

Initial specs:

- Q4_K
- Q6_K
- Q8_1 activation

Exit criteria:

- Block sizes, group sizes, scale/min layout, signedness, and supported activation formats are represented as data.
- Existing Q4_K/Q6_K route code can reference specs instead of repeating constants where practical.

## Phase 6: First End-To-End Conversion

Status 2026-07-05: executed through `generated_q4k_prefill_e2e`.

Target: Q4_K prefill.

Reason:

- It is the active performance pressure point.
- It already has authority harnesses and parity gates.
- It has both positive generated substrate evidence and a clear blocker.

Do this only after phases 1-4 are in place:

```text
RuntimeOpSpec(QuantizedLinear, prefill, Q4_K, Q8_1)
  -> candidate registry
  -> q4k_int8_wmma generated candidate
  -> correctness gate
  -> authority smoke
  -> full authority only if smoke is route-clean and bounded
```

Success is not immediate speed. Success is:

- no banned implementation reachable from the selected candidate,
- route-bound correctness,
- full-model blocker classified as missing lowering/search/codegen/runtime/hardware.

Observed result:

- Candidate selection reaches `quant_linear_prefill.q4k_int8_wmma_tensor_substrate`.
- `prefill_mmq_parity_gate.py` passes for `mmq`, `sdot4`, and `wmma_generated`.
- `int8_wmma_codegen` passes with full-range Q8_1 int8 activations and emits `wmma_i32_16x16x16_iu8`.
- Canonical 14B smoke reaches `PREFILL_Q4K_Q8=wmma` and stops at the full-model graph guard:
  `RAW groups*m*n=419430400 > limit=67108864` for `attn_qo`.

Classification: blocked on lowering/runtime graph explosion in `group_tensor_matmul_v0`; the next implementation target
is a fused/tiled generated emitter that keeps the Q4_K/Q8_1 grouped dot route-bound without materializing many Tensor
matmul graph fragments.

Reuse scan result:

- Existing `q4k_q8_1_sdot4_coop_gemm_kernel` already owns the fused Q4_K/Q8_1 dot4 algebra.
- Existing generated packed-tile direct-warp code already owns the in-kernel 8-lane direct-output reduction pattern.
- Combining those produced `PREFILL_Q4K_Q8=mmq_direct`, which is bounded and correct but not fast:
  canonical 14B pp512 smoke = 85 tok/s.

So the next route should not duplicate the scalar dot4/direct-output topology. The remaining gap is a generated tiled
lowering that keeps the bounded direct-output shape while using a throughput-appropriate dot substrate.

Tiled WMMA route scope implemented:

- New route descriptor: `prefill_q4k_int8_wmma_tiled_research`.
- New candidate: `quant_linear_prefill.q4k_int8_wmma_tiled_substrate`.
- New env: `PREFILL_Q4K_Q8=wmma_tiled`.
- Unknown `PREFILL_Q4K_Q8` modes now raise instead of falling through to the scalar Q4_K/Q8_1 GEMM route.
- `q4k_wmma_tiled_lowering_feasibility` passes on AMD:
  bounded `16x16x32` RAW tile lowers to `wmma_i32_16x16x16_iu8` and matches int32 reference.
- `q4k_wmma_tiled_microgate` passes on AMD:
  one bounded Q4_K/Q8_1 tile with scale/min correction has rel_rmse ~= `1.3e-7` vs the q8-dequant reference.
- `q4k_wmma_tiled_role_shape` classifies all 14B role shapes as `blocked.full_route_lowering_missing`.
- `generated_q4k_prefill_e2e` now reports:
  `GENERATED_Q4K_PREFILL_E2E_TILED_BLOCKED_FULL_ROUTE`.

Current classification: one-tile tiled WMMA is correct and codegen-valid, but there is still no direct tiled full-role
scheduler/codegen lowering. The next implementation is not another Tensor chunk/cat wrapper; it must map role shapes to
bounded tiles without route-local WMMA source/asm and without falling back to `prefill_q4k_direct_tile4x4_default`.

## Do Not Start Yet

Do not start until the direct tiled full-role lowering exists:

- MoE-specific expert offload or expert-cache route machinery.
- MLA-specific branches.
- New environment-variable control planes.
- New benchmark entry points.

## Immediate Next Commit Target

Implement Phase 1 audit scaffolding:

```text
extra/qk/generated_quant_binding_audit.py
test/unit/test_generated_quant_binding_audit.py
gate_registry row: generated_quant_binding_audit
```

The audit should start conservative. It can classify known route manifest rows first and mark deeper custom-kernel
call graph reachability as `unknown.investigate`. The value is forcing every next route discussion to name
provenance before writing code.
