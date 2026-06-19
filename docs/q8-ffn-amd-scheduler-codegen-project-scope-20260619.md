# q8 FFN AMD scheduler/codegen project scope (2026-06-19)

Purpose: completely scope the next real unlock after DSO closed bounded q8 native ownership.

The q8 decode primitive is proven useful, but not yet natively ownable by tinygrad:

- A4 research artifact route: W==D decode `1.051-1.063x`, dNLL `+0.002887`;
- hipcc/LLD artifact lifecycle: `114.12us`, clears the route gate;
- tinygrad COMGR fused-C: correct but `146.88us`;
- tinygrad AMD DSL/ASM: correct but `166.649us`;
- DSO variant ladder: body-insensitive `~0.151-0.153ms` variants vs full `0.166ms`;
- final DSO classifier: `wait_scheduler_bound`.

Therefore the remaining unlock is **not** q8 primitive search. It is one of:

1. teach tinygrad to emit a hipcc-quality AMD schedule; or
2. import/host the mature schedule as an artifact while keeping the route research-flagged.

These routes are not mutually exclusive. Route B can provide an oracle and deployment bridge while Route A learns the
compiler capability.

## Definition of "hipcc-quality" for this project

For the fused Q4_K x q8 gate/up consumer, "hipcc-quality" means:

| property | target |
|---|---:|
| correctness | max_abs `<=2e-3` vs q8 proxy on real GGUF |
| fused gate/up device time | `<=60us` |
| lifecycle producer + gate/up | `<=129.2us` |
| W==D decode | `>=3%` sustained |
| dNLL | `<=0.01` |
| runtime | HCQ/no in-process HIP runtime |
| default | unchanged/off |

The current fast artifact is the schedule oracle, not a default route.

## Evidence boundary

Closed facts:

- Dot4 is not missing: all consumers emit `v_dot4_i32_iu8`.
- Static instruction count is not the core explanation: tinygrad ASM has fewer static instructions than hipcc/LLD.
- Load shape is a visible delta but not enough alone: DSO load-only variants remain near full ASM time.
- COMGR source reshuffling is closed: fused-C is correct but slow.
- Current AMD DSL/ASM expression is correct but does not transfer the fast schedule.
- Producer ownership should not proceed until the consumer scheduler wall is solved.

Open project-level facts:

- whether tinygrad can add a scheduler/codegen capability narrow enough to justify maintenance;
- whether artifact/import is acceptable for a research flag;
- whether a native scheduler can generalize beyond q8 decode to prefill/Tensile-class matmul/other AMD kernels.

## Route A — native tinygrad AMD scheduler/codegen transfer

Goal: make tinygrad generate the fast q8 consumer schedule without external hipcc/LLD artifacts.

This is compiler work. It should be evaluated as a reusable backend capability, not a one-off kernel stunt.

### A0 — schedule contract extraction

Deliverable:

- `bench/q8-ffn-amd-scheduler-project/oracle_contract.json`

Tasks:

- normalize the hipcc/LLD oracle schedule into a machine-readable contract:
  - launch geometry;
  - kernarg layout;
  - resource fields;
  - instruction counts;
  - global-load widths;
  - waitcnt placement;
  - reduction pattern;
  - register live ranges if recoverable;
  - workgroup decomposition.
- compare it against COMGR and AMD DSL/ASM.

Gate:

- the contract identifies concrete codegen features, not just "LLVM is better".

Kill:

- if the oracle cannot be reduced to a stable contract, prefer Route B artifact hosting only.

Status: **DONE/PASS**. See `q8-ffn-route-a-scheduler-codegen-result-20260619.md`.

### A1 — AMD DSL capability map

Deliverable:

- `bench/q8-ffn-amd-scheduler-project/dsl_capability_map.json`

Tasks:

- map which oracle features current AMD DSL can express:
  - `global_load_b128` / D16 / byte loads;
  - exact waitcnt placement;
  - `s_clause` / delay-alu / scheduling annotations;
  - EXEC masking / scalar-lane stores;
  - local id Y/Z descriptor enablement;
  - wave32 constraints;
  - vector register allocation constraints;
  - bpermute/LDS reduction alternatives.
- classify each feature as:
  - expressible now;
  - expressible with small assembler extension;
  - renderer/scheduler feature;
  - not worth owning.

Gate:

- at least one feature has a credible `>=30us` contribution or enables the full oracle schedule.

Kill:

- if every feature is "renderer/scheduler feature" with no bounded first milestone, Route A is a larger compiler
  roadmap item and should not be funded just for q8 decode.

Status: **DONE/FAIL_A1_NO_BOUNDED_FEATURE**. No A2 candidate clears the `>=30us` gate. Route A remains project-level
AMD scheduler/codegen roadmap for q8 decode.

### A2 — one feature proof

Deliverable:

- `bench/q8-ffn-amd-scheduler-project/feature_proof.json`

Candidate proofs:

- vector/coalesced load selection;
- descriptor/local-y support;
- alternate reduction schedule;
- explicit wait scheduling / grouped waits;
- register allocation/live-range improvement;
- schedule annotation emission if supported by assembler format.

Gate:

- improves a q8-shaped microbench by `>=30us`, or improves full consumer by `>=25us`;
- correctness still passes;
- no external compiler.

Kill:

- if best feature proof moves `<15us`, stop Route A for q8. DSO already showed local variants are body-insensitive.

### A3 — native q8 consumer rebuild

Deliverable:

- `bench/q8-ffn-amd-scheduler-project/native_consumer_rebuild.json`

Tasks:

- rebuild the fused gate/up consumer using the new scheduler/codegen feature;
- compare against:
  - hipcc/LLD oracle;
  - COMGR fused-C;
  - old AMD DSL/ASM.

Gate:

- `<=75us` to continue;
- credible path to `<=60us`;
- real-GGUF correctness `<=2e-3`.

Kill:

- if still `>100us`, Route A is not viable for q8 without a broad compiler rewrite.

### A4 — producer capability reopen

Only if A3 passes.

Tasks:

- reopen fused RMSNorm + q8 side-channel producer;
- add staged reduction + post-barrier multi-output stores only if the consumer route is now viable.

Gate:

- producer + gate/up lifecycle `<=129.2us`;
- no separate q8 pack kernel.

Kill:

- if producer requires a separate pack/lifecycle tax, keep A4 research artifact as the only q8 route.

### A5 — graph/model integration

Only if A3/A4 pass.

Gate:

- W==D decode `>=3%`;
- dNLL `<=0.01`;
- default off.

## Route B — artifact/import route

Goal: use the mature schedule directly while preserving HCQ/no-HIP-runtime execution.

This is a research artifact route, not a tinygrad-native codegen win. It may still be valuable because A4 already
proved the route improves decode.

### B0 — artifact policy decision

Question:

- Is an external hipcc/LLD-produced HSACO acceptable behind a research flag?

If no:

- Route B stops immediately.

If yes:

- continue only as a reproducible research flag, not default.

### B1 — reproducible artifact build

Deliverable:

- `bench/q8-ffn-amd-scheduler-project/artifact_build_manifest.json`

Tasks:

- pin source strings;
- pin compiler commands;
- record ROCm/LLVM version;
- record code hash;
- record disassembly summary;
- record launch contract.

Gate:

- artifact is reproducible from repo scripts on this machine;
- no in-process HIP runtime.

Kill:

- if the artifact cannot be rebuilt deterministically, keep only the current A4 proof and do not maintain Route B.

### B2 — named artifact loader

Deliverable:

- `bench/q8-ffn-amd-scheduler-project/artifact_loader.json`

Tasks:

- generalize the current artifact loading path enough to avoid fragile runtime-cache hacks;
- support named kernel symbols if needed;
- validate kernarg size/resource metadata;
- keep launch geometry explicit.

Gate:

- eager launch correctness and timing match current A4 artifact;
- no model default change.

Kill:

- if loader becomes a broad binary importer, stop. This route should stay narrow.

### B3 — graph-safe research route

Deliverable:

- `bench/q8-ffn-amd-scheduler-project/artifact_graph_route.json`

Tasks:

- make producer + consumer graph-capturable with stable buffer lifetime;
- remove per-token Python overhead;
- keep `Q8_FFN_HANDWRITTEN=1` or successor flag.

Gate:

- W==D decode reproduces A4 `>=3%`;
- dNLL `<=0.01`;
- no HIP runtime;
- flag off unchanged.

Kill:

- if graph route is fragile or shape-specific beyond the intended Qwen3-8B/Q4_K_M research target, keep it as a
  benchmark-only artifact.

### B4 — maintenance boundary

The artifact route must declare:

- supported model shape(s);
- supported GPU arch;
- supported ROCm/LLVM build path;
- exact source code;
- no guarantee as default tinygrad functionality.

If this cannot be written cleanly, do not ship the route even as a research flag.

## Route C — schedule import as compiler training data

Goal: use Route B artifacts to teach Route A.

This is a bridge route, not a runtime route.

Tasks:

- store oracle schedule contracts from hipcc/LLD and Tensile-style kernels;
- build a diff tool from tinygrad output to oracle output:
  - instruction groups;
  - load widths;
  - wait placement;
  - resource metadata;
  - runtime timing;
  - variant sensitivity.
- add machine-readable labels:
  - `load_shape`;
  - `wait_schedule`;
  - `reduction_shape`;
  - `resource_encoding`;
  - `register_schedule`;
  - `body_insensitive`.

Gate:

- the diff tool predicts at least one successful change on a tiny q8-shaped probe.

Kill:

- if the oracle only says "LLVM scheduled better" without actionable feature labels, keep it as documentation only.

## Priority recommendation

Do **B0/B1 first** if the goal is a research number or a working route:

- A4 already proves artifact q8 improves decode;
- the loader path already exists in rough form;
- cost is low and bounded;
- risk is maintenance/policy, not unknown performance.

Do **A0/A1 first** if the goal is tinygrad compiler progress:

- it is the only dependency-free long-term route;
- it may generalize to other AMD kernels;
- it is weeks-scale compiler work, not a decode primitive task.

Do not start A2/A3 until A0/A1 produce a concrete feature with credible `>=30us` movement. DSO says blind local tuning
is the wrong move.

## Final decision matrix

| outcome | decision |
|---|---|
| Route B accepted and B3 passes | keep q8 decode as research flag; default off |
| Route B rejected, Route A A1 has no bounded feature | q8 native ownership remains closed |
| Route A A3 reaches `<=75us` | reopen producer ownership and lifecycle route |
| Route A A3 reaches `<=60us` and A5 passes | native q8 route becomes maintainable behind flag |
| Route C yields predictive feature labels | feed them into AMD renderer/scheduler roadmap |
| none pass | decode primitive work is exhausted; focus on prefill/Tensile or broader compiler roadmap |

## What "completion" looks like

This project is complete when it can say one of:

1. **Artifact route accepted:** q8 decode has a reproducible, graph-safe research flag with documented external artifact
   policy.
2. **Native route viable:** tinygrad emits a consumer `<=75us` with a credible path to `<=60us`, so producer ownership is
   worth reopening.
3. **Compiler roadmap only:** no bounded feature moves enough, so the remaining work is a general AMD scheduler/codegen
   project outside the q8 primitive arc.

The current evidence makes outcome 1 or 3 more likely than outcome 2.
