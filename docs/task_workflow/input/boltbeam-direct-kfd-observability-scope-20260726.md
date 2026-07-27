# BoltBeam Direct-KFD Kernel Observability and G5 Search Scope

Date: 2026-07-26
Repository: tinygrad-arkey
Target: AMD gfx1100, wave32

## Objective

Extend BoltBeam so it can evaluate tinygrad kernels submitted through direct KFD queues, without depending on HIP/HSA interception. The system must expose generated program identity, final code-object resources, launch geometry, synchronized runtime timing, and optional hardware counters as separate evidence layers. Use it to search the 14B G5 decode depth-decay problem only after the evidence joins are valid.

This scope is not permission to change G5 geometry or promote a faster candidate. Every candidate remains isolated, correctness-gated, and fail-closed.

## Why the current path stops short

- `extra/qk/prefill/prefill_boltbeam_trace.py` records profile events and optional PMC data, but it is prefill-oriented and depends on runtime events that do not see every direct-KFD launch.
- `extra/qk/decode/current_decode_execution_adapter.py` already proves that tinygrad can compile a program, inspect the ELF descriptor, disassemble AMD ISA, and emit VGPR/LDS/ISA evidence, but it deliberately forbids dispatch.
- `extra/qk/mmq_compile_evidence.py` and `tinygrad/viz/amd.py` provide resource and ISA parsers, but there is no immutable per-dispatch ledger joining program identity to launch geometry and measured wall time.
- rocprofv3 sees llama HIP launches but produced no tinygrad kernel trace. This is an observability boundary, not proof that tinygrad emitted no kernels.
- Direct KFD submission bypasses HIP/HSA interception. BoltBeam must instrument tinygrad's own `AMDProgram`/KFD launch boundary or consume a queue-level trace.

## Evidence model

Every record has one immutable `program_id` derived from source hash, binary hash, kernel name, target, compiler identity, and route/candidate identity.

### Layer A: compile and code-object evidence

Capture source, ELF binary, AMD kernel descriptor, allocated VGPR/SGPR, LDS, scratch, spills, local size, grid, ISA digest, and parser/tool versions. Unknown resource fields remain unknown; they cannot be inferred from source or occupancy formulas.

### Layer B: launch and timing evidence

At the direct-KFD launch boundary record queue, packet, program_id, grid, workgroup, arguments' buffer identities, submit timestamp, completion timestamp, synchronization mode, and process/boot/device identity. Use GPU timestamp packets or device events where available; host elapsed time is labeled diagnostic only.

### Layer C: hardware-counter evidence

Add a backend adapter with two implementations: an approved ROCm profiler path when it sees the launch, and a direct SQTT/performance-counter path when it does not. Counters are optional for route attribution but mandatory for claims about occupancy, spills, cache, LDS conflicts, or memory stalls. A missing counter is `UNAVAILABLE`, never zero.

## Work plan

### Phase 0: safety and identity

1. Add a `BoltBeamObservationSession` that pins boot ID, device, ROCm/LLVM versions, git commit, model hash, compiler environment, GPU lock owner, and cache policy.
2. Make candidate sets immutable during a session. Reject mixed source/binary/compiler identities.
3. Add positive controls: a known tinygrad elementwise kernel, a known decode flash kernel, and a llama HIP kernel. Each must produce the expected Layer A/B record before any G5 run.

### Phase 1: direct-KFD launch adapter

1. Instrument the narrowest `AMDProgram` submission/completion boundary in `tinygrad/runtime/ops_amd.py`; do not alter packet contents or scheduling.
2. Emit one JSONL dispatch record per program with a content-addressed artifact directory.
3. Join the record to `ProfileProgramEvent` and route observer data when present; absence is explicit.
4. Add a subprocess-safe collector API so isolated benchmark workers write sidecars without parent-process monkey patches.

### Phase 2: resource and ISA join

1. Reuse `current_decode_execution_adapter.py`, `mmq_compile_evidence.py`, and AMD ELF parsing rather than duplicating parsers.
2. Store a complete resource/ISA artifact for every distinct program_id.
3. Reject a dispatch ledger if source hash, binary hash, target, or kernel name does not match the resource artifact.
4. Add tests for missing fields, digest mismatch, duplicate program IDs, stale cache artifacts, and direct-KFD records with no HIP profiler row.

### Phase 3: timing and counter adapters

1. Implement synchronized per-program timing from device timestamps or explicit queue completion.
2. Integrate rocprofv3 as an optional external adapter; never require it for direct-KFD identity.
3. Implement or document the direct SQTT/PMC adapter. It must identify counter scope, sampling period, normalization, and whether values are per-wave, per-SE, or aggregate.
4. Add a counter positive control and a negative control proving that an empty rocprof file does not erase a valid Layer A/B ledger.

### Phase 4: BoltBeam candidate search

1. Extend the candidate schema with `program_identity`, `resource_evidence`, `launch_geometry`, `timing_evidence`, and `counter_evidence` references.
2. Search only bounded candidates: current G5, G4 control, K_ONLY, and one explicitly justified lifetime/address-recomputation variant. Do not invent four-wave G5 or serial fifth-head variants without a separate correctness proof.
3. Run each candidate in a fresh process under `/tmp/gpu-bench.lock`, with pinned clocks, fixed token fixture, three warmups, and at least ten synchronized samples per depth.
4. Randomize A/B order across repeated sessions to avoid thermal/cache ordering bias. Never run candidates concurrently.
5. Gate promotion on numerical parity, route identity, complete Layer A/B evidence, and a statistically stable depth slope. Counter evidence is required for any mechanism claim.

## G5-specific discriminator

The first search question is not “which kernel is faster?” It is “which program family owns the additional ctx4096 wall time?” Compute per-family contribution at ctx512 and ctx4096, then require an Amdahl bound before changing that family. A candidate is inadmissible if its measured improvement is smaller than timing noise or if it changes KV traffic, route identity, or output semantics.

## Acceptance criteria

- Direct-KFD tinygrad positive controls produce nonempty Layer A and Layer B artifacts.
- A missing rocprof trace does not invalidate direct-KFD records.
- Every resource/timing/counter row joins by immutable program identity.
- G4/G5 ctx512/4096 ledgers contain route, grid, workgroup, source/binary hash, resources, timing, and counter availability.
- BoltBeam emits `SUPPORTED`, `REFUTED`, or `INCONCLUSIVE`; it never treats missing evidence as clean.
- No G5 production change is promoted unless numerical, authority, and depth-slope gates pass.

## Explicit non-goals

- Do not infer occupancy from VGPR count alone.
- Do not treat source-level UOps as proof of final ISA or hardware stalls.
- Do not use host wall time as a hardware-counter substitute.
- Do not bypass the GPU lock, kill live AMD processes, or reset the GPU from a candidate runner.
