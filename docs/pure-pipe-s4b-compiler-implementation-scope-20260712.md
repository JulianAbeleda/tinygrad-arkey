# Pure pipe S4b compiler implementation scope

## Objective

Implement the compiler-owned non-LDS WMMA pipe primitive required to bind the
validated generated pipe core into the normal graph/runtime ABI. The first
vertical slice is `attn_qo` at `M=512,N=4096,K=4096`; no route defaults change
until that slice passes every gate.

The hybrid handwritten route is a teacher/reference only. It must not be copied,
wrapped, or used as a purity shortcut.

## Existing interfaces

The implementation must fit these existing ownership points:

- `extra/qk/wmma_pipe_spec.py`: declarative `WMMAPipeSpec`, typed IR, and
  candidate context;
- `tinygrad/uop/ops.py`: `KernelInfo`, `KernelCandidateContext`, and operation
  definitions;
- `tinygrad/codegen/opt/postrange.py`: candidate context and warmstart
  propagation;
- `tinygrad/codegen/__init__.py`: sink-to-Program lowering and cache identity;
- AMD renderer/isel/regalloc/waitcnt scheduling: instruction and resource
  ownership;
- `tinygrad/engine/realize.py`: runtime argument tuple and launch ABI;
- `tinygrad/runtime/graph/hcq.py`: graph capture/replay;
- existing pure execution/timing authorities and route census.

## Required semantics

The primitive must express, in compiler-owned IR:

- global row-major fp16 A and transposed/global row-major fp16 B;
- 128-bit cooperative global loads;
- register-resident A/B fragments;
- two in-flight K stages;
- K-step progression and stage ownership;
- targeted VM wait after the required load group;
- `v_wmma_f32_16x16x16_f16` operations;
- fp32 accumulation and fp16/fp32 output contract as appropriate;
- final stores and loop/tail behavior;
- exact role/shape/target metadata;
- register, LDS, scratch, workgroup, and occupancy estimates.

The IR must not contain rendered ISA text, instruction tuples, or `Ops.INS`.

## Work packets

### C0 — Semantic contract

Define the operation fields, invariants, and lifecycle state machine. Specify
which values are compile-time constants and which are graph buffer arguments.
Add invalid-contract tests for shape divisibility, unsupported waits, stage
count, dtype, target, and tail behavior.

### C1 — UOp representation

Add the smallest typed UOp or compiler-side IR node needed to represent one pipe
epoch. Define its source/child UOps, output dtype, buffer references, and graph
replacement behavior. Prove ordinary matmul graphs remain unchanged when the
node is absent.

### C2 — Lowering and scheduling

Lower the typed node before backend rendering. Implement stage ownership,
register fragment lifetime, load grouping, K-loop progression, and targeted
wait placement. The scheduler must expose enough metadata for resource checks.

### C3 — AMD instruction mapping

Map the scheduled operation to existing AMD renderer primitives for b128 loads,
WMMA, stores, and waitcnt. Add the missing mapping only where no existing
primitive exists. Do not import or call the hybrid emitter. Add compile-only
attn_qo source/ISA structure tests.

### C4 — Resource accounting

Compute and verify global/local sizes, register usage, LDS allocation (expected
zero for the non-LDS pipe), scratch, wave count, and occupancy-relevant limits.
Fail before binary launch on unsupported or overflowing plans.

### C5 — Graph ABI binding

Attach the typed candidate context to `KernelInfo` and preserve it through
postrange, `to_program`, runtime realization, cache, and HCQGraph. Prove output,
A, and B argument order, dtype, contiguous strides, and exact shape. Test two
candidate identities in one process and paired tensors using one binary.

### C6 — `attn_qo` authority

Run the complete ladder:

1. host contract and graph propagation;
2. compiler source/ISA purity;
3. resource/no-spill proof;
4. full-output nonconstant correctness;
5. executed binary/source identity;
6. pinned kernel timing with compile excluded;
7. gate/up-only plus `attn_qo` whole-model A/B.

No later role work is accepted until C6 passes.

### C7 — Role parameterization

Reuse the compiler primitive for `ffn_down` and `attn_kv`. Keep exact candidate
identities and separate resource proofs. KV must validate its smaller N grid,
tail behavior, and occupancy independently.

### C8 — Combined pure authority

Assemble generated `ffn_gate_up` plus all passing generated non-LDS roles. Run
ctx512 first, then 1024/2048/4096 under pinned K8/warmup4/round3 authority.
Require parity, route census, no rollback, no hybrid fallback, clean commit,
and executed-binary joins.

## Required tests

- typed operation schema and invalid contracts;
- ordinary graph unchanged without the primitive;
- stage/wait ordering and K-loop lifecycle;
- source contains no `Ops.INS` or native-ISA injection;
- exact A/B/output ABI and strides;
- candidate context/cache separation;
- warmstart/env restoration on exceptions;
- resource limits and no scratch;
- full-output correctness for nonconstant matrices;
- runtime binary identity;
- route census and strict pure gate;
- pinned timing and whole-model A/B.

## Review gates

Every packet must include changed files, host tests, compiler artifact, and a
purity statement. Review rejects any packet that introduces a second emitter,
uses precompiled diagnostics as graph kernels, mutates global state without
restoration, or claims whole-model improvement from isolated TFLOPS.

## Completion

S4b is complete only when C6 passes for `attn_qo`, C7 passes for `ffn_down` and
`attn_kv`, and C8 produces a pure combined authority result. If C2/C3 requires
a backend ABI redesign larger than this scope, stop with the exact missing
operation/resource semantics and keep the hybrid reference clearly labeled as
hybrid.
