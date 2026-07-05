# Generated Quant Runtime Architecture Scope

Date: 2026-07-05.

## Goal

Build a holistic, model-agnostic quantized runtime substrate where fast paths are generated from
Tensor/UOp/codegen/search infrastructure only.

Hard rule:

```text
No handwritten kernels.
No handwritten runtime fallbacks.
No handwritten oracle kernels.
No custom source-string kernels.
No shape-specific hand-authored GPU kernels.
```

Allowed implementation surface:

```text
Tensor expressions
UOp graph construction
codegen passes
scheduler/search spaces
renderer intrinsic lowering
hardware vocab descriptors
mathematical reference code in tests
```

The goal is not to hardcode Q4_K, Qwen, or any single benchmark path. Q4_K is one packed weight
format and one useful proving case. The architecture must also account for Q6_K, IQ formats, MXFP4,
FP8/INT8, MoE experts, shared experts, MLA/DeepSeek-style attention, flash attention, KV-cache
quantization, dense prefill, and decode.

## Current Design

The current design is route-first and format-specific.

High-level flow:

```text
tinygrad/llm/model.py
  -> tinygrad/llm/prefill_routes.py or tinygrad/llm/decode_routes.py
  -> tinygrad/llm/route_ops.py
  -> extra/qk concrete implementation
  -> custom_kernel / generated UOp / special route
```

Important current files:

- `tinygrad/llm/model.py`: model execution, MoE wiring, attention paths, prefill toggles, admission.
- `tinygrad/llm/prefill_routes.py`: prefill route selection and implementation binding.
- `tinygrad/llm/decode_routes.py`: decode route selection and implementation binding.
- `tinygrad/llm/route_ops.py`: lazy bridge from runtime code into `extra/qk`.
- `tinygrad/llm/route_policy.py`: BoltBeam/FutureSight policy loading and route selection.
- `tinygrad/codegen/opt/tc.py`: tensor-core and lane-map vocab, including RDNA3 iu8 WMMA.
- `tinygrad/renderer/cstyle.py`: HIP/C-style renderer, including dot helpers and iu8 WMMA lowering.
- `extra/qk/*`: mixed namespace for generated UOp builders, experiments, gates, audits, policy tools,
  custom-kernel paths, quant semantics, flash routes, and probes.

This structure has produced useful acceleration work, but the runtime boundary is too weak: route code
often knows concrete quant formats, shapes, environment flags, schedule knobs, and implementation names.

## Problems With The Current Design

### 1. Quant formats leak into architecture

Q4_K and Q6_K are currently more than data formats. They influence route structure, implementation
names, environment flags, and special-case control flow.

That makes future formats such as IQ*, MXFP4, FP8, or model-specific expert formats harder to add
without expanding route-specific branches.

### 2. Route files know too much

`prefill_routes.py` and `decode_routes.py` combine:

- semantic detection,
- shape gating,
- quant selection,
- policy override handling,
- schedule selection,
- fallback behavior,
- concrete implementation binding.

These files should describe and bind generated candidates, not encode a growing implementation matrix.

### 3. There is no single operation descriptor

The runtime does not consistently represent:

```text
QuantizedLinearPrefill(
  weight_format=Q4_K,
  activation_format=Q8_1,
  shape=(m,n,k),
  role=ffn_gate_up,
  device=AMD:gfx1100
)
```

Instead, route code often jumps directly from model/shape checks to a concrete implementation.

### 4. Search is constrained by manually exposed route families

BubbleBeam/FutureSight can select candidates today, but the search space is still shaped by manually
exposed route IDs, environment flags, and existing implementations.

The target is different: declare the primitive/lowering space first, then let generated candidates
compete inside that declared space.

### 5. MoE, MLA, and flash are not unified under one candidate model

The repo already has MoE model support and generated flash UOp builders, but they do not share a single
operation/candidate abstraction with dense quantized linear paths.

The architecture should treat these as operation families:

```text
DenseLinearPrefill
DenseLinearDecode
MoEExpertLinearPrefill
MoEExpertLinearDecode
AttentionPrefill
FlashDecodeAttention
MLAAttention
KVCacheReadWrite
ActivationFusion
```

### 6. The no-handwritten-kernel rule makes the current structure invalid long term

Any route that depends on handwritten or shape-specific GPU kernel logic is not eligible in the target
runtime. Fast paths must exist because tinygrad can generate them.

## New Design

The new design is operation-first, quant-format-agnostic, and generated-only.

Target flow:

```text
model semantics
  -> runtime operation descriptor
  -> quant/layout descriptor
  -> generated candidate registry
  -> BubbleBeam/FutureSight policy/search
  -> Tensor/UOp/codegen lowering
  -> correctness/perf/provenance gates
  -> promoted generated route or ordinary tinygrad fallback
```

There is no optimized handwritten fallback. If a generated fast path does not exist or does not pass
gates, the route falls back to ordinary tinygrad graph execution or is classified as blocked.

## Core Abstractions

### QuantizedTensorSpec

Describes storage and math, not runtime policy.

Fields should cover:

```text
format
ggml_type or equivalent external type
block_elems
block_bytes
group_size
scale_layout
zero_or_min_layout
signedness
packing
codebook or linear quant rule
correction_terms
preferred_dot_families
supported_activation_formats
```

Examples:

```text
Q4_K
Q6_K
IQ3_XXS
IQ4_XS
MXFP4
FP8
INT8
```

### RuntimeOpSpec

Describes what computation the model needs.

Fields should cover:

```text
family
phase
role
shape
input_dtype
input_layout
weight_spec
activation_quant_spec
device
architecture_traits
quality_constraints
```

Example:

```text
RuntimeOpSpec(
  family=QuantizedLinear,
  phase=prefill,
  role=ffn_gate_up,
  shape=(m=512, n=17408, k=5120),
  weight=QuantizedTensorSpec(format=Q4_K),
  activation=Q8_1,
  device=AMD:gfx1100
)
```

### GeneratedCandidate

Describes a generated implementation family. A candidate is eligible only if it is generated from
Tensor/UOp/codegen infrastructure.

Fields should cover:

```text
candidate_id
op_family
supported_quant_formats
supported_activation_formats
shape_constraints
device_constraints
lowering_strategy
required_codegen_features
search_space_id
provenance
rollback_behavior
authority_gates
```

Lowering strategy examples:

```text
packed_dequant_dot
grouped_int_dot_correction
iu8_wmma_grouped_dot
fp8_tensorcore_matmul
online_softmax_flash
grouped_expert_quant_matmul
kv_cache_quantized_attention
```

### Candidate Registry

The registry answers:

```python
candidate = select_generated_candidate(op_spec, device, policy)
```

It should reject candidates that are not generated-only.

### Route Policy

Route policy should select among declared generated candidates. Environment variables should be debug
and override tools, not the primary architecture.

Policy entries should carry:

```text
selected_route
op_family
shape
quant_format
candidate_id
search_space_id
route_params
provenance
gate_artifacts
```

## Old vs New

### Old

```text
Q4_K branch in prefill_routes.py
  -> env flag says sdot4/mmq/direct
  -> route_ops imports concrete extra/qk implementation
  -> custom_kernel call
```

### New

```text
model builds QuantizedLinearPrefill op spec
  -> weight_format=Q4_K is a descriptor
  -> registry finds generated candidates
  -> policy/search chooses a generated lowering
  -> codegen emits the runtime path
```

### Old

```text
Implementation names encode quant and shape:
q4k_q8_1_sdot4_direct_packed_gemm
prefill_q4k_direct_packed_load_direct_out_gemm
```

### New

```text
Candidate names encode generated lowering family:
quant_linear_prefill.grouped_int_dot_correction
quant_linear_prefill.iu8_wmma_grouped_dot
quant_linear_decode.packed_dequant_dot
attention_decode.online_softmax_flash
```

### Old

Adding a new quant format risks adding another route branch.

### New

Adding a new quant format adds a `QuantizedTensorSpec` and any missing generated lowering support.

### Old

MoE, flash, dense FFN, and Q4/Q6 routes evolve as separate surfaces.

### New

They are operation families sharing the same registry, policy, generated-candidate, and gate model.

## Code-Level Refactor Plan

This refactor should avoid hierarchy churn at first. Introduce the design through code boundaries and
types, then move files only when the boundaries are proven.

### Phase 1: Audit current runtime bindings

Audit all `custom_kernel` and route implementation calls in:

```text
tinygrad/llm/prefill_routes.py
tinygrad/llm/decode_routes.py
tinygrad/llm/route_ops.py
extra/qk/*
```

Classify each implementation as:

```text
Tensor expression                 allowed
generated UOp builder             allowed
codegen lowering                  allowed
renderer intrinsic lowering       allowed
handwritten kernel body           banned
source-string/native body         banned
shape-specific hand-authored path banned
unknown                           investigate
```

Output should be a machine-readable and human-readable report.

### Phase 2: Introduce specs without changing route behavior

Add the minimal descriptor types:

```text
QuantizedTensorSpec
ActivationQuantSpec
RuntimeOpSpec
GeneratedCandidate
```

Initially these can live in existing `tinygrad/llm` modules or one small new module. The first
objective is not file hierarchy. The first objective is making route decisions operate on explicit
semantic objects.

### Phase 3: Make prefill and decode routes build specs

Refactor route code so it constructs `RuntimeOpSpec` before selecting an implementation.

For example:

```text
prefill_routes.py
  current: shape/env/quant checks -> concrete implementation
  target: shape/env/quant checks -> RuntimeOpSpec -> registry selection
```

At this phase, behavior may remain equivalent, but the route path should expose which generated
candidate or fallback was selected.

### Phase 4: Build generated-only candidate registry

Introduce a registry that can register generated candidates and reject banned ones.

Initial candidates should wrap only generated/Tensor/UOp/codegen paths. Any implementation that cannot
prove generated-only provenance should not register.

### Phase 5: Separate quant semantics from route selection

Move Q4_K/Q6_K knowledge into reusable quant specs:

```text
block_elems
block_bytes
scale/min layout
signedness
correction math
activation compatibility
```

Route code should ask the quant spec for these properties instead of encoding format internals.

### Phase 6: Promote required codegen vocab

The following capabilities belong in codegen/lowering, not route-specific implementations:

```text
sdot4 / dot4 lowering as first-class generated op
iu8 WMMA tensor-core vocab
groupwise scale/min correction lowering
activation quantization lowering
packed int/FP fragment handling
lane-map and cross-lane reductions
LDS staging and double-buffer candidates
online softmax / flash attention primitives
grouped expert dispatch and grouped quant matmul
```

### Phase 7: Convert one route end-to-end

Pick one target route and convert it fully:

```text
RuntimeOpSpec
  -> generated candidate registry
  -> generated lowering
  -> route provenance
  -> correctness gate
  -> authority timing gate
```

Likely first target: Q4_K prefill, because it has strong measurement pressure and clear semantics.

The success condition is not "beat every old path immediately." The success condition is:

```text
No banned implementation is reachable.
The generated path is route-bound and numerically correct.
Performance blocker, if any, is attributed to missing codegen/search capability.
```

### Phase 8: Extend to MoE and flash

MoE should reuse quantized linear machinery:

```text
router/top-k
  -> grouped selected experts
  -> QuantizedLinear over expert weights
  -> probability weighting/reduction
  -> optional shared expert path
```

Flash should use attention operation specs:

```text
AttentionOpSpec
  -> online softmax candidate
  -> score/PV/lifecycle generated candidates
  -> KV-cache layout/quant descriptors
```

MLA/DeepSeek-style attention should be represented through attention descriptors, not model-name
branches:

```text
q_lora_rank
kv_lora_rank
key_length_mla
value_length_mla
compressed KV layout
projection decomposition
```

### Phase 9: Remove banned route access

After generated equivalents exist, remove runtime access to banned implementations.

If no generated equivalent exists, the optimized route is not eligible. The system should either use
ordinary tinygrad execution or report a generated-codegen blocker.

## Promotion Gates

A generated candidate can become default only when it has:

```text
generated-only provenance
no hidden banned fallback
correctness evidence
route-bound evidence
whole-decode or whole-prefill authority timing
rollback to ordinary tinygrad graph execution
roofline/PMC attribution when performance is the question
```

Local microbenchmarks are diagnostic. Promotion requires lifecycle authority.

## Success Criteria

- No handwritten kernels are reachable from model runtime.
- No custom source-string kernels are used as optimized routes.
- Q4_K is represented as a quant descriptor, not as an architecture.
- New quant formats add specs and lowerings, not route spaghetti.
- Dense FFN, MoE experts, MLA, flash attention, and KV cache use the same operation/candidate model.
- BubbleBeam/FutureSight selects among declared generated candidates only.
- Route policy carries provenance and gate artifacts.
- Performance failures are classified as missing vocab, missing lowering, schedule/search gap, runtime issue, or hardware limit.

## Non-Goals

- Do not reorganize the whole file hierarchy before the code boundaries exist.
- Do not add another Q4_K-specific fast path.
- Do not preserve handwritten kernels as oracle/fallback runtime paths.
- Do not make environment flags the core control plane.
- Do not claim search ownership unless the search space includes the primitive and lowering decisions used.

## North Star

```text
If it is fast, it is fast because tinygrad can generate it.
```

Q4_K, Q6_K, MXFP4, FP8, MoE, MLA, and flash should be inputs to the same generated runtime substrate,
not separate hand-built runtime architectures.
