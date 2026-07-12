# Pure non-LDS Program-to-graph transport scope

## Objective

Bind the compiler-generated non-LDS WMMA pipe program to the ordinary prefill
graph ABI so `attn_qo`, `ffn_down`, and `attn_kv` can become exact pure
machine-search candidates. This is the missing step between the validated
diagnostic compiler core and a whole-model pure 4.4k test.

## Proven starting point

The diagnostic compiler already produces and numerically validates generated
WMMA cores for:

- `attn_qo`: 512x4096x4096;
- `ffn_down`: 512x4096x12288;
- `attn_kv`: 512x1024x4096.

The generated structure has global b128 loads, WMMA operations, stores, and the
targeted wait policy. It does not use the handwritten pipe oracle.

The production blocker is architectural:

- diagnostic lowering returns a backend `Program`/binary;
- the candidate route expects ordinary graph UOps or raw custom-kernel tuples;
- candidate contexts currently describe the LDS path only;
- `lower_wmma_pipe_spec` is intentionally fail-closed;
- copying the handwritten `Ops.INS` emitter would fabricate pure provenance.

## Required transport contract

Implement a compiler-owned transport that accepts a typed `WMMAPipeSpec` and
returns an ordinary graph-bound program with:

1. A/B/output buffers in the normal graph argument order;
2. exact M/N/K, dtype, contiguous-stride, and output-shape validation;
3. launch dimensions and target metadata;
4. a typed `KernelCandidateContext` containing schema and canonical identity;
5. candidate-specific compiler cache identity;
6. scoped warmstart options and exception-safe restoration;
7. binary/source/resource evidence available to the existing authority;
8. no `Ops.INS`, native-ISA source, or handwritten route-local emitter.

The transport must preserve graph replay and support paired tensors/layers with
the same role binary but different buffers.

## Vertical implementation sequence

### T1: typed pipe context

Extend candidate context construction to represent the non-LDS pipe geometry,
pipeline, wait policy, and role. Keep the existing LDS context unchanged. Add
exact identity/cache-key tests and reject mismatched role/shape/target.

### T2: Program/UOp ownership boundary

Choose one compiler-owned representation that can be inserted into an ordinary
graph before backend compilation. It must not accept a precompiled binary as a
substitute for graph ownership. Add a minimal `attn_qo` lowering and prove the
program is generated from the typed spec.

### T3: buffer ABI and state scope

Bind A/B/output arguments with explicit dtype, shape, stride, and order checks.
Derive M from the admitted workload; do not hardcode `M=512`. Scope all env,
getenv-cache, warmstart opts, contexts, local-stage keys, and deny keys around
capture and restore them on exceptions.

### T4: attn_qo authority

Require, in order:

- source-only generated proof;
- no forbidden Ops/source markers;
- resource/no-spill proof;
- full-output nonconstant correctness;
- executed binary hash equals compiled candidate hash;
- pinned kernel timing;
- gate/up-only + attn_qo whole-model A/B.

### T5: ffn_down and attn_kv

Reuse the transport only after T4 passes. Add exact candidates for the long-K
down role and small-N KV role separately. KV requires an independent occupancy,
LDS, and tail-resource gate.

### T6: combined pure authority

Assemble only passing exact candidates with generated `ffn_gate_up`, prove the
candidate census and executed binary identities, then run pinned ctx512/1024/
2048/4096 with full-output parity. Compare against the banked hybrid reference
without calling the hybrid route pure.

## Review checklist

The first attempted transport must not be accepted if it:

- hardcodes M or assumes only ctx512;
- mutates global env/warmstart state without full restoration;
- binds a precompiled diagnostic binary directly as a graph candidate;
- reports scheduler-generated provenance while using `Ops.INS` or hand ASM;
- proves only route selection without executed binary/source evidence;
- lacks argument order/stride/dtype assertions;
- aliases two role identities in compiler cache.

## Completion and standstill

The transport phase is complete only when all three non-LDS roles pass T4/T5 and
the combined pure authority runs. It is a genuine standstill if the compiler
cannot expose a graph-insertable generated Program/UOp without a new backend
ABI or hand emitter. At that point the correct deliverable is the typed missing
interface and tests, not a relabeled hybrid result.
