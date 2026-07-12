# Native AMD Register Graph Scope

Date: 2026-07-12

## Purpose

Close the remaining pre-binary blocker for the sequential, one-slot pure
register route on AMD gfx1100. The route must pass the normal native AMD
pipeline without weakening UOp validation, WMMA ABI checks, register ownership,
or the pure-route evidence gate.

## Reproduction

Use the existing `RegisterPipeTemplate` fixture with `schedule="sequential"`,
the existing `RegisterStorageAdapter`, a full-K graph, and a multi-output WMMA
shape (`subtile_count=8`, `accumulator_elements=64`). Run:

```text
SPEC=1 full_rewrite_to_sink(..., AMDISARenderer(Target.parse("AMD:ISA:gfx1100")))
```

The failure is a UOp type-verification error on `Ops.STACK half.vec(16)` with
16 children that are themselves `Ops.STACK half.vec(16)`. The same nested
shape is observable with `SPEC=0`; disabling verification does not make it a
valid native graph.

Measured rewrite boundary:

| Rewrite | Nodes | Nested vector stacks |
|---|---:|---:|
| raw graph | 450 | 0 |
| add loads | 390 | 0 |
| combined devectorize matcher | 814 | 4 |
| later rewrites / control flow | 814 | 4 |

The failure is therefore introduced by the existing combined devectorizer
pass, not by AMD register allocation or the typed wait lowering. The native
isel stage can map the static one-slot A/B stage elements when the graph is
otherwise valid.

The four malformed parents are WMMA A/B carriers. `correct_load_store` splits
the vector register load into scalar loads, then the fixed-point matcher revisits
the carrier through the existing load/GEP folding rules and wraps an already
stack-shaped target a second time. The resulting `STACK(half.vec16)` contains
16 `STACK(half.vec16)` children instead of a flat 16-lane carrier. This is an
interaction between existing matcher families, not a malformed producer UOp.

## Ownership map

1. `register_pipeline.py` owns the logical A/B stage carriers and their
   readiness provenance. It must continue to emit scalar-correct
   `half.vec(16)` stage contracts.
2. `kernel_pipeline.py` owns epoch/slot/reuse ordering. It must not flatten or
   reinterpret vector data.
3. `devectorizer.py` owns load folding, vector splitting, GEP/STACK lowering,
   and WMMA scalarization. The fix belongs at this boundary if the input graph
   is valid.
4. `amd.py` owns conversion of valid stage carriers to pinned VGPRs and
   `WaitCount` to `s_waitcnt`. No route code may add raw ISA.
5. `spec.py` remains the authority for legal UOp shapes. Do not suppress the
   nested-stack error or broaden `Ops.STACK` to accept malformed children.

## Required investigation

### N0: Freeze the graph

Capture the pre-devectorize graph, post-devectorize graph, nested-stack count,
node tags, dtype/count, and the first backward slice for each malformed parent.
The fixture must include both A and B roles and the multi-output C ownership
shape; single-output WMMA is not sufficient evidence.

### N1: Isolate the matcher

Run the devectorizer components independently in existing order:

- `sym`
- `devectorize_alu`
- `devectorize_buf_and_index`
- `load_store_folding`
- `correct_load_store`
- `load_store_indexing`

Record the first component that changes a valid `half.vec(16)` carrier into a
`half.vec(16)` stack of vector children. Determine whether the cause is:

- load grouping/folding of the producer input;
- vectorized `DEFINE_REG`/stage-buffer index lowering;
- WMMA source scalarization (`no_vectorized_wmma` / `GEP` rendering); or
- a tag/lifetime rewrite that loses the stage-carrier identity.

### N2: Choose the smallest reusable fix

Preferred order:

1. Preserve the existing scalar/vector contract through the owning matcher.
2. Use an existing tag or codegen-extension seam to prevent only the invalid
   grouping for register-pipe carriers.
3. Add a narrowly scoped devectorizer pattern that converts the exact valid
   carrier form to the expected scalar/vector form while preserving tags and
   readiness.
4. Reject the form with a typed error if correctness cannot be proven.

Do not flatten nested stacks generically, disable all global load grouping, or
remove ABI/spec verification. Those changes would hide shape errors or alter
unrelated kernels.

### N3: Compiler proof

The fix must prove:

- no nested vector stacks at the post-devectorize/spec boundary;
- A/B stage values remain exactly `half.vec(16)`;
- no `DEFINE_LOCAL`, LDS stage allocation, or scratch appears;
- producer wait and overwrite dependencies survive;
- WMMA A/B/C contracts and multi-output ownership remain unchanged;
- dynamic/two-slot VGPR indexing still fails closed;
- existing LDS and non-register routes are byte/graph-equivalent.

### N4: Native AMD lowering

Run normal `to_program`/linearization with a real `KernelInfo` and the native
AMD renderer. Verify:

- static A/B stage accesses select `STAGE_READ`/`STAGE_WRITE`;
- stage VGPR spans do not overlap WMMA fragments or accumulators;
- typed waits lower to `s_waitcnt` with the expected immediate;
- no raw `Ops.WAIT`, `Ops.STACK`, or route-owned pseudo-op remains in the final
  instruction stream;
- assembler/resource descriptor generation succeeds.

Synthetic graphs with incomplete `KernelInfo` or missing workgroup metadata
may be used for local matcher tests, but cannot count as binary evidence.

### N5: Evidence and runtime

After a native binary exists, reuse the existing authorities for:

- final `AMDResourceArtifact` (VGPR/SGPR/LDS/scratch/spills and source/binary
  identity);
- single-role numerical correctness;
- pinned-clock synchronized timing;
- remaining dense roles;
- whole-model pure attribution with no hybrid fallback;
- typed machine-search admission.

## Exit criteria

This native-graph scope is complete only when N0-N4 pass. It does not by itself
claim 100% pure 8B execution; N5 remains required for correctness, timing,
whole-model purity, and machine search.

## Current status

The failure is isolated to the combined devectorizer boundary with a stable
reproduction and a clear ownership boundary. No generic flattening or spec
weakening is acceptable. Until N3/N4 pass, the route remains a structural
compiler proof rather than a runnable GPU candidate.
