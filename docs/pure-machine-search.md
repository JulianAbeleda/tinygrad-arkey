# Pure Machine Search

Pure machine search means the selected hot runtime path is generated or spec-driven, selected by BubbleBeam/FutureSight
policy, and verified by tinygrad gates. Handwritten/owned routes may exist as rollback, historical baseline, or ceiling
comparison, but they are not pure machine-search routes.

Practical project target: **machine search over reusable compiler primitives**. Strict purity remains the audit label for
routes whose executing topology is generated/spec-owned. Backend primitives may be hand-authored; complete
model-specific kernel schedules may not be called pure.

This document is the reviewer contract. If a route cannot be classified with these rules, the route is `unknown` and
cannot claim pure machine-search provenance.

## Verdict Vocabulary

| Provenance | Pure final default? | Meaning |
|---|---:|---|
| `machine_authored_generated` | yes | A search/profile/spec descriptor owns the kernel topology and lowering parameters; code is emitted from that descriptor or grammar. |
| `tinygrad_scheduler_generated` | yes | Ordinary tinygrad graph lowering/scheduler output, with no route-local custom kernel or instruction emitter. |
| `hand_authored_uop_template` | no | A human wrote a Python `Tensor.custom_kernel` / UOp kernel body. It may be temporary runtime debt, but it is not pure. |
| `external_handwritten_kernel` | no | HIP/CUDA/C/C++/ASM/source-string/precompiled binary/raw-instruction emitter. Never a pure final default. |
| `rollback_oracle` | no | A handwritten, owned, or specialized route retained only for rollback/reference comparison. |

Only `machine_authored_generated` and `tinygrad_scheduler_generated` are allowed by `FINAL_DEFAULT_PROVENANCE` in
`extra/qk/route_manifest.py`.

## Classification Rule

Classify the implementation that actually executes on the selected route, not the benchmark story around it.

1. If the route injects raw instructions, source strings, inline asm, precompiled binaries, or hand-emitted native ISA, it
   is `external_handwritten_kernel`.
2. Else if the route body is a human-authored `Tensor.custom_kernel` / UOp template, it is `hand_authored_uop_template`.
3. Else if the route is emitted from a structured search/spec descriptor whose degrees of freedom are data and whose
   emitter is shared infrastructure, it is `machine_authored_generated`.
4. Else if it is normal tinygrad graph lowering with no route-local custom kernel body, it is
   `tinygrad_scheduler_generated`.
5. Else it is `unknown` and must not be promoted as pure.

Search selecting a handwritten implementation does not make it pure. A route becomes pure only when the implementation
itself is generated/spec-owned under the rules above.

## ASM Is Not The Boundary

Assembly is allowed as a compiler/backend target. The boundary is kernel authorship, not whether the final code contains
AMD ISA.

| Use | Pure route allowed? | Meaning |
|---|---:|---|
| backend-emitted ASM | yes | tinygrad/codegen lowers generated IR/specs into target instructions such as WMMA, DS loads, stores, or waitcnt. |
| backend intrinsic lowering | yes | A reusable renderer/backend primitive emits one instruction family for generated callers. |
| ASM probe | not a product route | A temporary diagnostic kernel used to establish hardware semantics. |
| hand-authored full-kernel schedule | no, except oracle/escape hatch | A human writes the concrete load/LDS/WMMA/wait/store lifecycle for a hotspot. |
| raw instruction or binary injection | no, except oracle/escape hatch | The selected runtime path injects prebuilt instructions or binaries instead of compiler-owned lowering. |

So a generated kernel may end in ASM. That is normal. A hand kernel is different: the concrete kernel schedule is authored
by a human rather than produced by the compiler/search/spec path.

## Compiler Primitive Compromise

The preferred compromise is not to make the compiler rediscover every hardware trick from first principles. It is to
expose hardware tricks as reusable compiler primitives, then let search/spec compose them.

Allowed primitives include:

- WMMA intrinsic lowering,
- targeted `waitcnt` lowering,
- b128 global/DS load-store lowering,
- LDS staging operations,
- DBUF scheduling idioms,
- register/layout constraints for WMMA fragments,
- DS offset folding and address-lifetime rules.

These are not hand kernels by themselves. They become a hand kernel only when a route-local implementation manually emits
the complete executable lifecycle for a model/shape-specific hotspot.

The goal for prefill is therefore:

```text
machine search chooses primitive composition
backend emits AMD ISA
no route-local full-kernel raw instruction list
```

## Concrete Markers

These markers force `external_handwritten_kernel` unless the code is only a non-runtime test fixture:

- `Ops.INS`
- `UOp(Ops.PROGRAM, ...)` with `Ops.BINARY`
- source strings containing `asm volatile`
- route-local `__builtin_amdgcn_*` source strings
- `.cu`, `.hip`, `.s`, `.asm`, `.cpp`, or precompiled kernel blobs selected by the route

These markers usually force `hand_authored_uop_template`:

- route-local `Tensor.custom_kernel(..., fxn=some_python_kernel)`
- Python functions named like `*_kernel(...)` that manually construct UOp loops, reductions, loads, stores, or custom
  expressions
- `Ops.CUSTOM` or `Ops.CUSTOMI` used inside a route-local kernel template

These markers can support `machine_authored_generated`, but only with descriptor evidence:

- dataclass/spec/manifest rows defining the candidate shape, schedule, tile, lowering strategy, and rollback
- an emitter that lowers the descriptor through shared generator infrastructure
- a gate proving route-bound execution and generated-only provenance

Renderer-owned intrinsics are not automatically impure. For example, a backend helper such as `_sdot4` or WMMA lowering
can be pure when the route reaches it through generated tinygrad/codegen lowering. The same operation becomes impure if a
route-local kernel directly emits a source string or instruction list to force it.

## Examples In This Repo

| Route or file | Classification | Reason |
|---|---|---|
| `decode_q4k_g3_generated` | `machine_authored_generated` | Generated G3 LaneMap route selected by policy and lowered from route/search descriptors. |
| `decode_q6k_coop_generated` | `machine_authored_generated` | Spec-driven Q6_K route emitted from `Q6KGEMVRouteSpec`. |
| `decode_flash_live_split_g4_8b_kvboth` / `decode_flash_block_tile_g5_konly` | `machine_authored_generated` | Live-split decode attention is now bound through `FlashDecodeAttentionSpec` (`FlashDecodeTileSpec` + `LiveSplitGeometrySpec` + `FlashCombineSpec`) and registered as descriptor-owned generated UOp codegen. |
| `prefill_pipe_role_selective_generated` | not pure under this strict implementation rule while it lowers through raw `Ops.INS` | The schedule is spec-selected, but the executing implementation still uses `extra/qk/prefill/wmma.py` instruction-list emitters. This is generated schedule selection over a handwritten substrate, not final pure machine search. |
| `prefill_q4k_direct_tile4x4_default` | `machine_authored_generated` | Default Q4_K direct-packed prefill is emitted from `Q4KPrefillRouteSpec`; `PREFILL_Q4K_REDUCE_OUT=1` remains separate research debt. |
| Retired `PREFILL_Q4K_Q8=sdot4/mmq/mmq_direct` routes | — | Removed 2026-07-06 (no backups): these scalar/MMQ modes are no longer valid route envs. Any remaining helper symbols are historical/non-default fixture debt, not manifest routes. |
| `extra/qk/prefill/wmma.py` raw builders | `external_handwritten_kernel` | Explicit instruction-list emitter using RDNA3 instructions and `Ops.INS`. |
| `route_q4k_graph_gemm` fused Q4_K WMMA path | `external_handwritten_kernel` | Calls `build_gemm_lds2_q4k` and wraps the emitted instruction list in `Ops.INS`. |
| `native_isa_block_tile_graph_node.py` | `external_handwritten_kernel` for selected runtime use | Injects a precompiled native-ISA `Ops.BINARY` program into the HIP runtime. |
| Retired handwritten decode rollback routes | — | Removed 2026-07-06 (no backups): bubblebeam-off / generated-off now falls to the ordinary tinygrad graph or no manifest kernel route. |

## Audit Rule

A route can claim strict pure-machine-search status only if all of this is true:

- Its `route_manifest.py` row uses `machine_authored_generated` or `tinygrad_scheduler_generated`.
- Its selected runtime path has no raw ISA/source-string/precompiled-binary injection.
- Its selected runtime path has no route-local human-authored UOp template, unless that template has been converted into
  a descriptor-owned generated emitter and the manifest provenance was updated accordingly.
- Its authority gate proves route-bound execution, no hidden fallback, correctness, and timing against the relevant
  rollback/baseline.
- Its rollback route is present and explicitly classified as rollback/reference if handwritten.

If any item fails, the route can still ship as transitional engineering debt, but the manifest must say so with
`hand_authored_uop_template`, `external_handwritten_kernel`, or `rollback_oracle`, and a `replacement_scope` is required
for any default route.

## Naming Trap

The word `generated` in a route id is not authoritative. A route can be generated in one layer and handwritten in another:

- generated policy selecting a handwritten kernel is not pure,
- generated schedule parameters feeding a raw instruction-list emitter are not pure,
- generated UOp descriptors lowered through normal tinygrad/codegen can be pure,
- ordinary tinygrad lowering can be pure even when the backend renderer emits hardware instructions.

The audit question is always: who owns the executing kernel topology and instruction/body construction?

## External Alignment

`pure machine search` is local repo vocabulary. The closest public terms are `auto-scheduling`, `autotuning`,
`tensor program generation`, and `compiler-generated kernels`.

This contract intentionally follows the strict end of that public spectrum:

- TVM Ansor-style auto-scheduling removes per-operator manual schedule templates and constructs/searches schedules from
  tensor expressions plus general search rules.
- TVM MetaSchedule distinguishes manual schedules, template-based design spaces, and automatically generated design
  spaces; this repo treats manual/template-only points as non-pure unless the executing implementation is generated from
  descriptor-owned search space.
- Triton/compiler-generated kernel work treats backend scheduling, memory layout, synchronization, and specialization as
  compiler/search/lowering responsibilities; this matches the repo rule that backend-owned lowering can be pure.

So the repo definition is stricter than ordinary `autotuning`: tuning knobs on a handwritten kernel is not pure here.
The route must have generated/spec-owned executing topology, not just generated parameter selection.

## Current State

- Q4_K decode GEMV defaults to generated G3 LaneMap where structurally eligible.
- Q6_K decode GEMV defaults to a spec-driven generated coop route.
- 8B long-context decode attention defaults to generated live-split/KV_BOTH.
- 14B-style G=5 decode attention uses the generated block-tile route for its validated shape.
- Prefill has generated/spec-driven role selection, but any path that still executes through the raw WMMA instruction
  emitters remains non-pure until the substrate is replaced by generated tinygrad/codegen lowering.

The local authority is `bench/qk-search-spaces/default_route_manifest.json`; the runtime census is
`extra/audit/pure_machine_search_default_path_census.py`.

## Ownership Boundary

- BoltBeam owns candidate generation, route policy, evaluation, roofline attribution, and ledgers.
- tinygrad owns runtime execution, backend/compiler lowering, and focused hardware gates.
- tinygrad should not grow a second search-policy/evaluator stack.

## Promotion Rule

A generated route can become default only when it has:

- correctness evidence,
- route-bound/no-hidden-fallback evidence,
- rollback,
- W==D or equivalent authority timing,
- practical-roofline justification when absolute parity is the question.

Local microbenchmarks and isolated kernels are diagnostic only.

## What Remains

The hard work is not adding more handwritten kernels. It is improving the generator, lowering, route policy, and
measurement stack until generated candidates can cover more shapes and move closer to practical roofline without
manual special cases.
