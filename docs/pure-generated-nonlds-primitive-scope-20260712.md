# Pure generated non-LDS primitive scope

## Goal

Turn the validated hybrid non-LDS strategy into a compiler-owned pure primitive
that can be machine-searched and graph-bound for Qwen3-8B prefill.

The hybrid route is the performance teacher: it reaches approximately 4.4k
tok/s and uses handwritten backend atoms for the lean roles. The pure route is
the target: no handwritten `Ops.INS`, no external backend atom, and exact
compiler/candidate provenance.

## Known results

- Generated buffer2 `ffn_gate_up` candidate: proven correct and fast.
- Hybrid lean routes for `attn_qo`, `ffn_down`, and `attn_kv`: the correct
  resource strategy, but handwritten.
- Generated diagnostic WMMA cores for all three shapes: structurally valid and
  numerically correct.
- Naive ordinary generated `A @ B.T` transport: valid generated route but only
  about 1,902 tok/s at ctx512; it does not preserve the lean pipe lifecycle.

Therefore the missing primitive must preserve the hybrid pipe strategy while
remaining compiler-owned.

## Primitive contract

### Input

`WMMAPipeSpec` contains only declarative workload and strategy data:

- exact M/N/K;
- tile M/N and K step;
- wave partition and workgroup size;
- register-resident A/B policy;
- pipeline stages and K cadence;
- vector load/store widths;
- wait policy;
- role, target, and candidate identity.

The route supplies tensors and exact workload identity. It does not supply
instructions or a binary.

### Lowering

Add a compiler-owned lowering stage:

```text
WMMAPipeSpec
  -> typed pipe IR / graph UOps
  -> postrange and AMD codegen
  -> generated Program and binary
  -> ordinary graph launch
```

The typed pipe IR must express register-resident fragments, global b128 loads,
WMMA operands, targeted waits, K-stage progression, and output stores without
using `Ops.INS`. The AMD renderer/codegen owns final instruction selection and
wait scheduling.

### Runtime ABI

The generated program must launch with the normal graph ABI:

1. output buffer;
2. A input buffer;
3. B input buffer;
4. exact dtype and contiguous-stride contract;
5. normal launch dimensions;
6. candidate context attached to `KernelInfo`/Program metadata.

The program key and compiler cache must include schema version, canonical
candidate identity, role, exact shape, target, and primitive parameters.

## Vertical implementation phases

### P0: typed context

Extend candidate context for non-LDS pipe geometry while leaving LDS contexts
unchanged. Validate identity, role, shape, target, pipeline, and cache key.

### P1: typed pipe IR

Define the smallest graph/compiler IR needed to represent one generated pipe
epoch. Add host tests for stage order, load ownership, wait policy, and output
store ownership. No AMD route promotion yet.

### P2: `attn_qo` graph transport

Lower `512x4096x4096` into the existing graph/codegen pipeline. Prove:

- no `Ops.INS` or native-ISA source;
- compiler-generated source and binary;
- ordinary buffer ABI and launch dimensions;
- exact candidate context and cache identity;
- no scratch and expected register/resource limits;
- nonconstant full-output correctness;
- runtime binary equals compiled candidate;
- pinned kernel timing;
- gate/up-only + `attn_qo` whole-model A/B.

### P3: remaining roles

Reuse P2 for:

- `ffn_down`: `512x4096x12288`;
- `attn_kv`: `512x1024x4096`.

`attn_kv` gets an independent small-N occupancy and resource gate. No geometry
is assumed transferable merely because the primitive is shared.

### P4: machine search

Seed each role from the hybrid strategy, then search only parameters represented
by the pure primitive contract: tile, waves, K step, upcast/register residency,
vector widths, wait policy, and pipeline cadence. Every candidate has an exact
role/shape/target identity and isolated evidence.

### P5: combined pure authority

Combine the three passing non-LDS candidates with generated `ffn_gate_up` and
run the established pinned whole-prefill authority at ctx512/1024/2048/4096.
Require full-output parity, route census, binary/resource joins, and no hybrid
fallback. Compare against hybrid only as the teacher/reference.

## Non-goals

- Do not wrap or copy handwritten hybrid instruction lists.
- Do not pass a precompiled diagnostic binary directly as a graph candidate.
- Do not label ordinary generated `A @ B.T` as equivalent to the lean primitive;
  the 1,902 tok/s negative test disproves that assumption.
- Do not change the proven generated buffer2 `ffn_gate_up` candidate during P0-P2.

## Acceptance and stop conditions

The primitive is accepted only after P2 passes all compiler, correctness,
resource, binary, timing, and whole-model gates. If the compiler cannot express
the pipe epoch in graph UOps without a new backend ABI, document that missing
interface and stop the pure implementation at that boundary. A hybrid result
may be retained as a teacher/reference, but it must never be reported as pure.

## 100% definition

Pure non-LDS transport is complete when all three roles use generated graph-bound
pipe candidates, every candidate has exact provenance and executed-binary proof,
the combined pure route passes parity, and the pinned whole-model result either
reaches 4.4k or identifies a measured next ceiling.

## Spark execution packets

Each packet is intentionally small enough for one Spark agent. Agents must not
silently combine packets, change defaults, or claim completion from a host-only
test. Every packet returns a commit, tests, changed files, and an explicit
pass/blocked verdict.

### S0 — Contract inventory (read-only)

Owner files: `tinygrad/uop/ops.py`, `tinygrad/codegen/__init__.py`,
`tinygrad/codegen/opt/postrange.py`, `tinygrad/engine/realize.py`,
`tinygrad/runtime/graph/hcq.py`, `extra/qk/wmma_pipe_spec.py`.

Deliverable: a field-by-field map of the existing `KernelInfo`, sink/program
metadata, graph capture, cache key, and runtime argument ABI. Identify the exact
function where a typed pipe primitive can enter before `to_program`. No edits
unless a missing type annotation is necessary.

#### Compiler insertion-point inventory (C4)

The implementation must cross these existing boundaries in order; adding an
`Ops` enum alone is insufficient:

1. **Typed IR and UOp ownership:** define the immutable `WMMAPipeIR` beside
   `WMMAPipeSpec` in `extra/qk/wmma_pipe_spec.py`. If a core UOp is required,
   add it in `tinygrad/uop/ops.py` together with `range_start`, shape/dtype
   inference, `type_verify`, `toposort`/graph traversal, and serialization
   behavior. The IR must lower to ordinary loads, WMMA, waits, and stores; it
   must never carry source text or native instructions.
2. **Verifier and rangeification:** teach `tinygrad/shape/` verification and
   `tinygrad/codegen/opt/rangeify.py`/`symbolic.py` the operation's axis and
   buffer contracts. A malformed A/B/output dtype, stride, tile divisibility,
   or wait/stage count must fail before scheduling.
3. **Postrange/sink:** lower or expand the typed operation in
   `tinygrad/codegen/opt/postrange.py` before `apply_opts` finalizes the sink.
   Preserve `KernelInfo.candidate_context`, `opts_to_apply`, and pipe metadata
   through every `replace`; warmstart state must restore on both normal and
   exceptional exits.
4. **Program/cache boundary:** `tinygrad/codegen/__init__.py::to_program` must
   see an ordinary sink and produce `ProgramInfo`; candidate identity must enter
   the compiler cache key in `tinygrad/device.py::Compiler.compile_cached`.
   Distinct identities with identical source must produce distinct program and
   binary cache entries.
5. **AMD lowering:** the existing WMMA/load/store UOps then flow through
   `tinygrad/renderer/amd.py` (and renderer pattern tables), followed by normal
   metadata extraction. No `Ops.INS`, assembly source, or precompiled binary is
   permitted. Resource output must include launch dimensions, VGPR/SGPR, LDS,
   scratch, and wave size.
6. **Runtime/graph ABI:** `tinygrad/engine/realize.py::get_runtime`,
   `tinygrad/runtime/graph/hcq.py`, and the HCQ call tuple must retain ordinary
   A/B/output argument order, dtypes, contiguous strides, and candidate
   identity. Graph capture/replay must not rebind buffers across roles.

Acceptance is a host structural test at each boundary, followed by AMD compile,
resource/binary identity, nonconstant correctness, and pinned timing. Until all
six boundaries pass, the primitive remains a scoped blocker and cannot enter
route selection or the pure default.

### S1 — Typed context/schema review

Owner: `extra/qk/wmma_pipe_spec.py`, `extra/qk/runtime_specs.py`, focused tests.

Deliverable: immutable pipe IR schema, exact identity validation, role/shape/
target checks, and cache-key tests. It must reject wrong stage count, wait policy,
dtype, and target. Existing LDS candidate contexts must remain byte/behavior
compatible.

### S2 — Postrange propagation

Owner: `tinygrad/codegen/opt/postrange.py`, tests only for propagation.

Deliverable: pipe IR survives sink creation, postrange, and graph capture through
`KernelInfo.candidate_context`; warmstart opts, contexts, local-stage allow/deny
keys restore after success and exception. No renderer or runtime changes yet.

### S3 — Graph-owned pipe IR lowering

Owner: compiler lowering module plus `extra/qk/wmma_pipe_spec.py`.

Deliverable: lower one `WMMAPipeIR` into ordinary graph UOps before backend
rendering. It must not return a precompiled binary, raw instruction tuple, or
`Ops.INS`. Host structural test must show global b128 loads, WMMA, wait policy,
stores, and no native-ISA source. If existing UOps cannot represent a required
operation, document the smallest new compiler IR operation and its semantics.

### S4 — AMD renderer/codegen integration

Owner: AMD codegen/renderer insertion point identified by S0/S3.

Deliverable: compile the S3 UOps to a normal AMD Program with launch dimensions,
argument order, source/binary hashes, resource metadata, and candidate context.
No route promotion. Test two distinct identities in one process and prove cache
and binary separation.

### S5 — Buffer ABI gate

Owner: graph route and authority tests.

Deliverable: exact output/A/B argument order, fp16 input/output contract,
contiguous strides, shape, M/N/K divisibility, and normal graph replay. Reject
non-512 M only when the candidate says so; do not hardcode M in transport. Test
paired layers with different buffers using one role binary.

### S6 — `attn_qo` proof ladder

Run, in order: source purity, resource/no-spill, nonconstant full-output
correctness, runtime binary equality, pinned kernel timing, then gate/up-only plus
`attn_qo` whole-model A/B. The A/B must include route census and no fallback.
Failure blocks later role work.

### S7 — `ffn_down` and `attn_kv`

Parameterize only after S6 passes. Run separate proofs for 512x4096x12288 and
512x1024x4096. KV requires independent occupancy/LDS/tail checks and cannot
inherit attn_qo resource assumptions.

### S8 — Candidate-set assembly

Add three exact candidates to the existing set format, preserving generated
`ffn_gate_up`. Validate duplicate identities, weak-key collisions, role policy,
cache keys, and all four route census entries.

### S9 — Combined pure authority

Run Qwen3-8B pinned K8/warm4/round3 at ctx512 first, then 1024/2048/4096.
Require full-output parity, strict pure provenance, no rollback, exact binary
joins, clean commit, and route census. Compare against hybrid only as reference.

### S10 — Machine search and promotion

Seed role-specific populations from the passing pipe primitive. Search only
compiler-owned knobs. Correctness precedes timing; timing precedes whole-model
promotion. Retain passing role winners and reject isolated-only improvements.

### S11 — Completion review

Review all packet diffs and artifacts for route ownership, no `Ops.INS`, cache
identity, state restoration, ABI, resource, and provenance. Completion requires
all three roles, combined pure authority, and either 4.4k or a measured pure
ceiling with the remaining interface documented.

## Packet handoff rules

Packets S0–S2 may run in parallel. S3 depends on S0–S2. S4–S5 depend on S3.
S6 depends on S4–S5 and must be reviewed before S7. S8–S10 depend on all role
proofs. S11 is always performed by the parent/reviewer, not the implementing
agent. If a packet is blocked, spawn a smaller investigation packet against its
named interface; do not bypass the gate with a hybrid implementation.

## Current compiler standstill

Packets S0-S4a are complete. The typed context, immutable pipe IR, graph sink
attachment, and typed pipe-op contract are implemented and host-tested. S4b is
blocked because the current UOp/AMD renderer has no first-class operation for
staged WMMA pipe lifecycle or per-stage `vmcnt` waits. Waits are currently
derived from backend load scheduling, while `Ops.WMMA` represents only the
aggregate tensor-core operation.

Completing S4b requires new compiler scheduling semantics, AMD renderer
lowering, and resource accounting together. A partial mapping is not an
acceptable pure candidate, and copying the hybrid handwritten emitter would
invalidate provenance. This is the named interface boundary for the current
pure 4.4k effort.
