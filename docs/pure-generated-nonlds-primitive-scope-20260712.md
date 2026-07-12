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
