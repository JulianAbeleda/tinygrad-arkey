# Shared attention compiler-phase ABI: exhaustive implementation scope

Date: 2026-07-23

## Objective

Introduce one reusable compiler abstraction for ending the lifetime of a typed fragment/state region, publishing selected state to LDS, and reloading it later with explicit logical ownership. Use that abstraction to implement one-pass attention with:

```text
QK + online softmax
  -> publish probability/correction and selected state
  -> end QK-state lifetime
  -> rotate small PV accumulator windows through LDS
  -> drain complete output
```

The abstraction must be general enough for other composite reductions. It must not be an attention-only collection of special cases.

## Why the pivot is required

The mathematical implementation and tensor-core path are already proven. The remaining failure is representational:

- The production single-wave kernel reports about `254 VGPR`.
- Slicing PV state lowers allocation to `196` but leaves an early QK/softmax peak.
- Full score-state/PV splitting lowers resources but recomputes QK and is slower (`0.8387 ms` vs `0.5423 ms` at KV512; `5.1502 ms` vs `2.0550 ms` at KV4096).
- A compile-only LDS accumulator pressure probe reaches `197 VGPR`, `8704 B LDS`, and zero spills.
- The old `m/l` staging attempt fails before resource emission because the current UOp shape/address contract produces `INDEX` vector/scalar mismatches.

The repeated pattern is that local kernel changes cannot express a true lifetime boundary. The compiler therefore keeps state coupled or rejects the reload graph.

## Non-goals

- Do not rewrite the whole scheduler.
- Do not change the composite online-softmax math.
- Do not add route-specific 8B/14B kernels.
- Do not materialize the full score or probability matrices.
- Do not rely on compiler spills as an optimization.
- Do not change generic WMMA allocation for ordinary kernels.
- Do not promote a candidate based on instruction count without resource and replay evidence.

## Baseline that must remain unchanged

The current production default and proof baseline are the single-wave score-resident fused attention path:

- 8B: `Hq=32, Hkv=8, G=4, Hd=128`.
- 14B: `Hq=40, Hkv=8, G=5, Hd=128`.
- QK and PV both use WMMA.
- Full score/probability buffers are absent.
- Real AMD numeric captures pass.
- Corrected replay is roughly `4x` to `26x` faster than ordinary GQA across the measured matrix.

Every ABI change must be gated behind an experimental context or constructor flag. The old ABI must continue to compile and pass its existing tests.

## Compiler layers in scope

The work must keep these responsibilities separate:

### 1. Scheduler/state construction: `tinygrad/schedule/wmma.py`

Construct logical state regions and phase transitions. This layer owns algorithmic meaning:

- QK fragment roles;
- online `m/l` state;
- probability/correction state;
- PV accumulator logical blocks;
- output ownership;
- phase ordering.

It must not encode physical VGPR numbers.

### 2. Typed UOp metadata: `tinygrad/uop/ops.py` and `tinygrad/uop/spec.py`

Add validated, immutable descriptors for phase/state transitions. These descriptors carry logical identity and storage layout, not backend-specific register assignments.

### 3. Range/post-range handling: `tinygrad/codegen/opt/postrange.py`

Preserve phase sideband metadata through loop conversion, register allocation, and opaque post-regalloc markers. Metadata loss here is a correctness failure.

### 4. AMD lowering: `tinygrad/renderer/isa/amd.py` and the HIP mirror

Lower publish/reload operations to existing LDS loads/stores and typed waits. Physical fragment mappings are applied only here. Generic ordinary-WMMA lowering must remain untouched.

### 5. Proof/capture: `extra/qk/shared_attention_capture.py`

Record phase IDs, logical state ownership, output block ranges, graph identity, numeric evidence, and resource evidence. Aggregate validation must fail closed on overlap, gaps, mismatched graphs, or missing phase records.

## Proposed ABI

The ABI consists of three typed concepts.

### `StateRegionSpec`

Describes a logical state region:

```text
name: stable logical name, e.g. "online_ml" or "pv_acc"
dtype: scalar element type
shape: logical per-wave element shape
lane_layout: explicit lane-major or fragment-major ownership
blocks: logical block count
block_width: elements per block
storage: REG or LDS
owner: stable construction identity
```

Validation requirements:

- `dtype`, shape, and block width agree.
- Every logical element has exactly one lane owner.
- LDS layout size is computable and aligned.
- A region cannot silently change logical block identity when its physical output offset changes.

### `PhaseBoundarySpec`

Describes a lifetime transition:

```text
name: stable phase name
before: tuple of StateRegionSpec names
publish: tuple of region names and LDS offsets
after: tuple of region names allowed to remain live
wait: typed wait requirement
barrier: wave-only or workgroup requirement
version: ABI version
```

For the target attention transition:

```text
before: QK fragments, online m/l, probability/correction temporaries
publish: probability/correction slab, optionally old m/l
after: reload handles, compact PV accumulator window, V fragment
wait: lgkmcnt=0
barrier: none for the single-wave path
```

The boundary is a semantic lifetime frontier. It must not be implemented as an untyped dependency marker that leaves the original values reachable after the boundary.

### `StateHandle`

Represents a post-publish reload:

```text
region: logical region name
phase: producing phase ID
block: logical block ID
lane: logical lane owner
lds_offset: validated byte/element offset
dtype: exact reload dtype
shape: exact scalar/vector shape
```

The handle must preserve vector shape through lowering. Scalar LDS addresses may be used for individual lane elements, but a vector reload must be rebuilt as an explicit vector stack with the expected dtype before softmax or WMMA consumers.

## Logical versus physical identity

This is mandatory.

```text
logical_state_block  !=  physical_output_block
```

For sliced PV:

- Logical accumulator blocks remain `0..N-1` inside the kernel.
- `output_block_base` identifies where the slice is written in the final output.
- A pre-biased V pointer or compile-time input offset handles physical V input ownership.
- The drain applies output offset only at final stores.

Changing `output_block_base` must not widen the logical register state or create extra runtime address pairs.

## Phase transition semantics

The compiler must implement this sequence explicitly:

1. Construct QK and online-softmax state.
2. Publish selected values to LDS using typed stores.
3. Emit `WAIT(lgkmcnt=0)` through the typed wait ABI.
4. End reachability of the pre-boundary values. They must not remain in the backward slice of PV consumers.
5. Reload state through typed `StateHandle`s.
6. Rebuild vectors with exact shape and lane ownership.
7. Run compact PV WMMA and LDS accumulator updates.
8. Wait before reading LDS values that were just written.
9. Drain with logical block ownership plus output base metadata.

For the final one-pass target, QK is computed once per KV tile. No second QK kernel or full score buffer is permitted.

## Synthetic compiler tests before attention integration

Each test must compile through the generic UOp pipeline and AMD HIP/ISA lowerers.

### Test A: scalar publish/reload

- One scalar fp32 state per lane.
- Publish to LDS.
- Typed wait.
- Reload and store.
- Verify exact resource metadata and no shape mismatch.

### Test B: vector publish/reload

- Eight fp32 values per lane.
- Publish lane-major.
- Reload as eight scalar loads.
- Rebuild with explicit `STACK` into `float.vec(8)`.
- Consume in a vector multiply and WMMA-compatible path.

This test specifically prevents the current `INDEX (vec8) * scalar` failure.

### Test C: phase lifetime test

- Create QK-like temporaries before the boundary.
- Publish only the required state.
- Ensure post-boundary consumers cannot reach the pre-boundary temporaries.
- Compare compiler resource metadata against a control graph.

### Test D: logical/physical slice test

- Compile low and high output slices.
- Keep identical logical state blocks and non-V loads.
- Change only output store base and pre-biased input pointer.
- Verify no high-slice VGPR premium.

### Test E: multi-block LDS rotation

- Back eight logical accumulator blocks with correctly sized lane-major LDS.
- Rotate a one- or two-block register window.
- Use typed waits and no workgroup barrier for a single wave.
- Verify LDS byte count and no spills.

No attention integration is allowed until Tests A-E pass.

## Attention integration sequence

### Stage 1: publish old online state

Add an experimental `stage_old_ml_lds` path in the helper only after vector publish/reload passes. Store old `m/l` with explicit scalar lane addresses and reload into `float.vec(8)` handles. Do not attach `AFTER` to raw constants; use typed phase UOps or a non-constant carrier accepted by the type verifier.

### Stage 2: phase-bound QK and PV

Use a `PhaseBoundarySpec` after QK/softmax publication. Ensure the QK zero seed, old `m/l`, and temporary score carriers are not reachable from the PV-side graph except through validated handles.

### Stage 3: rotate PV accumulators

Keep one or two PV blocks in registers. Back inactive blocks in LDS. Apply the online correction to each reloaded block and store it back before advancing to the next block. Preserve one QK/softmax computation per KV tile.

### Stage 4: full output drain

Drain all logical blocks with disjoint output ownership. Use the existing v2/v3 proof schemas, extended only for the new phase records.

## Admission gates

Every candidate must pass in this order.

1. Structural UOp validation.
2. HIP and AMD ISA compile.
3. Resource metadata:
   - target `VGPR <= 192` for the first admitted candidate;
   - no scratch;
   - zero VGPR/SGPR spills;
   - LDS size explicitly recorded and within device limits;
   - max referenced VGPR and fixed fragment ranges recorded.
4. Numeric correctness:
   - full output, not a sampled element;
   - causal first and prefix positions;
   - Hd128;
   - maximum absolute and relative error recorded.
5. Proof:
   - QK/PV role attribution;
   - no score/probability materialization;
   - complete logical output ownership;
   - phase IDs and graph/source/ISA hashes;
   - no spills.
6. Corrected TinyJit replay:
   - graph captured once;
   - inputs and outputs reused;
   - synchronized replay;
   - at least 10 samples after warmup;
   - total multi-phase closure timed as one unit;
   - baseline measured under the same protocol.
7. Whole-model promotion:
   - native lowering route census proves candidate selected;
   - 8B and 14B geometries;
   - KV512, KV1024, KV2048, KV4096;
   - full-output numeric checks;
   - separate fp16-attention and Q4 whole-model roofline accounting.

## Fallback decision tree

```text
vector publish/reload fails
  -> fix StateHandle shape/ownership ABI

phase boundary compiles but VGPR remains >192
  -> attribute the peak from ISA/liveness
  -> reduce QK fragment ownership or rematerialize only the identified state

one-pass LDS accumulator compiles but numeric fails
  -> verify correction alpha and block-local state before changing geometry

one-pass LDS accumulator is correct but slower
  -> measure LDS traffic and wait cost
  -> reject if no total replay gain
  -> retain only reusable ABI if structurally valuable

candidate replay improves but whole-model route is ordinary_sdpa
  -> do not claim promotion; wire native lowering/census first

candidate improves one model/context only
  -> do not promote; complete both 8B/14B matrices
```

## Commit boundaries

Keep commits small and independently revertible:

1. Add `StateRegionSpec`, `PhaseBoundarySpec`, and `StateHandle` validation.
2. Add synthetic scalar/vector publish/reload tests.
3. Preserve phase sideband through postrange and opaque markers.
4. Add AMD/HIP lowering and wait tests.
5. Add logical/physical slice tests.
6. Add attention phase integration behind an experimental flag.
7. Add LDS accumulator rotation and numeric evidence.
8. Add proof/capture phase aggregation.
9. Add composite replay closure and resource admission.
10. Add model route census and whole-prefill artifacts.

Every commit uses a bracketed prefix and records the focused command/result. Do not push a partially compiling ABI.

## Ownership for future agents

- Compiler ABI agent: `ops.py`, `spec.py`, `postrange.py`, synthetic tests.
- AMD lowering agent: `amd.py`, HIP mirror, wait/address lowering tests.
- Attention integration agent: `schedule/wmma.py`, experimental helper only after ABI gates.
- Proof agent: capture schema, phase aggregation, fail-closed negative tests.
- Benchmark agent: composite replay closure, route census, whole-prefill matrix.

Agents must not edit the same subsystem concurrently. GPU timing is serialized after all structural and resource gates pass.

## Completion definition

This phase is complete only when one shared compiler ABI supports the phase transition and the resulting one-pass attention candidate satisfies all gates through whole-model promotion for both target routes. A structurally elegant ABI without replay improvement is not completion; a faster isolated kernel without proof-gated model integration is also not completion.

