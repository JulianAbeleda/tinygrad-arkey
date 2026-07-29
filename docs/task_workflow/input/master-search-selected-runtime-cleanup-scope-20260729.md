# Master Search-Selected Runtime and Fallback Cleanup Scope

Date: 2026-07-29

Status: scoped for low-agent execution on `exp`; implementation and promotion are not authorized by this document

Repository tiers:

- production: `/Users/julianabeleda/env/tinygrad-arkey` (`master`)
- qualification: `/Users/julianabeleda/env/tinygrad-arkey-dev` (`dev`)
- research/integration: `/Users/julianabeleda/env/tinygrad-arkey-exp` (`exp`)

This scope supersedes the no-human/full-program-synthesis interpretation in
`master-pure-machine-search-decoupling-scope-20260729.md`. It does not discard that scope's useful closure inventory,
hash validation, runtime trace, generic fallback, or README-boundary work. It changes the classification rule and the
resulting migration plan.

## 1. Intended outcome

`master` remains the fast, runnable product that produced the published README results. It contains:

- production configurations selected by the BoltBeam + BubbleBeam/FutureSight + tinygrad workflow;
- the human-maintained tinygrad compiler, lowerers, descriptor interpreters, runtime guards, and backend primitives
  required to execute those selected configurations;
- ordinary tinygrad graph/scheduler fallback for unsupported or declined workloads;
- the smallest stable correctness, route-identity, benchmark, and maintenance surfaces required to defend the product.

`master` does not contain handwritten specialized rollback/oracle kernels, abandoned alternatives, candidate-search
development, raw sweeps, or qualification-only harnesses. Those belong on `dev` or `exp`.

The cleanup must preserve the benchmark-producing hot path. It is not a kernel redesign, new search campaign, or
attempt to remove all uses of `Tensor.custom_kernel`.

## 2. Practical machine-search definition

A route is machine-search-selected when:

1. humans define a valid candidate space, compiler/lowering vocabulary, objective, correctness rules, and promotion
   policy;
2. the combined search workflow explores, ranks, scores, or measures alternatives;
3. a performance-sensitive configuration is selected from that process;
4. tinygrad lowers and executes the selected configuration; and
5. humans may review and approve promotion.

Human authorship of the compiler, lowerer, descriptor schema, candidate grammar, or promotion gate does not invalidate
machine search. A `Tensor.custom_kernel` call is an execution mechanism, not an authorship or fallback classification.

A handwritten fallback/oracle is a separately fixed specialized implementation retained for rollback, comparison, or
debugging rather than the selected production configuration. It belongs on `dev`/`exp` even if it is correct or fast.

Missing modern provenance packaging means `historically_unverified`; it does not prove `not_machine_search`.

### 2.1 Required independent classification axes

Every route and file must be classified independently on four axes:

| Axis | Values |
|---|---|
| execution | ordinary tensor graph, scheduler/codegen, descriptor lowering, `Tensor.custom_kernel`, raw external kernel |
| selection | search-selected, compiler heuristic, manually fixed, unresolved |
| runtime role | primary production, conditional production, generic fallback, specialized rollback/oracle, test/research |
| evidence | current complete, historically recoverable, incomplete, absent |

No agent may infer one axis from another. In particular, `custom_kernel` does not imply handwritten, `extra/` does not
imply test-only, and incomplete evidence does not imply manual selection.

## 3. Fact base from the published benchmarks

The README numbers are the preservation authority for this cleanup.

### 3.1 Decode

README commit `8b98e803b` records the promoted-master fixed-depth results:

| Model | ctx512 | ctx4096 | Production dependencies |
|---|---:|---:|---|
| Qwen3-8B Q4_K_M | 114.19 tok/s | 103.07 tok/s | Q4_K G3 decode GEMV plus G4 live-split flash attention |
| Qwen3-14B Q4_K_M | 69.70 tok/s | 62.45 tok/s | Q4_K G3 decode GEMV plus G5 QG2/S32 width-4 live-split flash attention |

Commit `8deca39bb` promoted the G5 configuration immediately before the README result. The committed campaign result
explicitly records SDPA at ctx128 and flash at ctx512/4096. Therefore G3 and G4/G5 are production dependencies, not
test-only fallback kernels.

Q4_K G3 has a recoverable combined-system lineage: BubbleBeam/FutureSight enumerated and ranked lane-layout candidates,
tinygrad lowered and measured the selected LaneMap route, and BoltBeam later owned route-policy binding. Its human-written
lowerer is compatible with the practical machine-search definition and must be retained as production functionality.

### 3.2 Prefill

README commit `f4d08fa90` was created after:

- `6ca798568`: packed-WMMA default-on;
- `aeeb65e9a`: admitted 8B/14B fused prefill attention through custom-kernel injection by default;
- `411410710`: promotion/census record for that attention route.

The current README records:

| Model | Four-point authority | Later bounded checks | Production dependencies |
|---|---|---|---|
| Qwen3-8B Q4_K_M | 3694/3624/3485/3236 | 3768 pp512 smoke | compiler-generated WMMA-LDS selected linears plus fused prefill attention |
| Qwen3-14B Q4_K_M | 1945/1920/1875/1785 | 2026 pp512, 1880 pp4096 | packed-WMMA selected linears plus fused prefill attention |

The Q4/Q6 direct-packed rollback implementations did not produce these headline results. On 14B, explicitly disabling
packed-WMMA selected direct-packed and exposed the documented rollback-path fault. Direct-packed is nevertheless still
reachable today and cannot be deleted before fallback rewiring.

## 4. Branch ownership contract

| Branch | Required contents |
|---|---|
| `master` | Search-selected production routes, selected descriptors/policies, required lowerers and generic compiler/backend primitives, ordinary tinygrad fallback, production CLI/README, compact route and benchmark authorities. |
| `dev` | Everything needed to qualify production plus specialized handwritten fallbacks/oracles, A/B adapters, fault reproducers, debug-only tests, and detailed qualification harnesses. |
| `exp` | Search-space and candidate development, BubbleBeam/FutureSight/BoltBeam adapters, rejected alternatives, topology searches, raw sweeps, blocked prototypes, and temporary evidence tools. |

Promotion remains destination-based. Never merge `exp` wholesale into `dev` or `dev` wholesale into `master`.

## 5. Corrected route disposition

| Route/family | Runtime role | Disposition |
|---|---|---|
| Q4_K G3 decode GEMV | Search-selected primary decode | Keep on master; move its implementation from the `extra` seam into a maintained `tinygrad/llm` owner without logic change. Preserve historical search lineage. |
| Q6_K cooperative decode | Conditional production for Q6 models | Keep working route while provenance is classified; promote required runtime implementation into `tinygrad/llm`. Do not label it fallback merely because its spec was human-authored. |
| G4/G5 live-split decode attention | Search/tuning-selected production attention at ctx512+ | Keep on master; promote descriptor/executor/lowering closure into production owners. G4/G5 values remain distinct selected configurations. |
| WMMA-LDS graph prefill | Search-selected/compiler-generated 8B linears | Keep on master; promote runtime binding into `tinygrad/llm`; retain ordinary tinygrad lowering. |
| Packed-WMMA prefill | Selected/compiler-generated 14B linears | Keep on master; extract live selector, table, canary admission, warmstart builder, and execution subset from research lineage. Preserve exact current behavior before improving provenance. |
| Fused prefill attention | Promoted production attention for admitted 8B/14B grids | Keep on master. Its custom-kernel injection is part of the measured path. Do not remove it under fallback cleanup. |
| Q4 direct-packed prefill | Specialized conditional rollback/fallback; not headline route | Replace master use with ordinary tinygrad fallback, then retain implementation only on dev as oracle. |
| Q6 direct-packed prefill | Specialized conditional rollback/fallback; not headline route | Replace master use, including exact-Q6-vocab handling, with ordinary tinygrad semantics, then retain implementation only on dev. |
| Historical owned Q4 warp/bridges | Retired rollback/reference | Dev oracle or Git-history recovery only; never reintroduce into master. |
| Refuted/blocked topology candidates | Research only | Exp until conclusion is banked; then delete if no maintained owner. |

## 6. File-level disposition

Paths are current `exp` names. Destination names may be refined inside the named packet, but ownership may not change
without updating this scope and the machine-readable ledger.

### 6.1 Keep or promote as master production

| Current surface | Production purpose | Target |
|---|---|---|
| `tinygrad/llm/{model,decode_routes,prefill_routes,qk_primitives,fused_attention}.py` | Model dispatch, selected route binding, quant storage, generic fallback | Retain and simplify after extra seam removal. |
| `extra/llm_research/gemv_g3_codegen_lowering.py` plus selected LaneMap/reduction dependencies | Benchmark-producing Q4 G3 route | Promote cohesive runtime subset to `tinygrad/llm/decode_kernels.py` or equivalent. |
| `extra/llm_research/q6k_route_spec.py` plus decode-only quant helpers | Q6 production decode | Promote decode runtime subset; do not move shared helpers wholesale until consumers are split. |
| `extra/llm_research/decode/{flash_decode_attention_executor,flash_decode_attention_spec}.py` and required `flash_common`, `flash_kernels`, `live_split_geometry` subset | Benchmark-producing G4/G5 attention | Promote descriptor/executor to `tinygrad/llm/flash_decode_attention.py`; generic primitives to `tinygrad/schedule/wmma` or compiler owners. |
| `extra/llm_research/prefill/prefill_graph_gemm_route.py` runtime subset | 8B WMMA-LDS selected linears | Promote to `tinygrad/llm/prefill_graph_gemm.py` or consolidate into `prefill_routes.py`. |
| `extra/llm_research/prefill/packed_wmma_prefill_candidates.py` and required current-execution subset | 14B packed-WMMA selected linears | Promote minimal shipped selector/warmstart/canary/runtime closure to `tinygrad/llm/packed_wmma_prefill.py`. |
| `tinygrad/schedule/wmma/{flash_prefill,kernels,fragments,softmax,loop_state,composite}.py` and generic AMD lowering | Benchmark-producing fused prefill attention/compiler substrate | Retain. `kernels.py` is not removed merely because it constructs UOps. |
| Minimal route policy/model facts currently sourced from `route_manifest.py` and memory-adaptive collector | Load-time production admission | Extract immutable production facts/interfaces into `tinygrad/llm`; keep research report generation on exp. |

### 6.2 Move to dev after rewiring

| Current surface | Dev role | Prerequisite |
|---|---|---|
| `extra/llm_research/prefill/q4k_prefill_route_spec.py` | Q4 direct-packed fallback/oracle | Master direct-packed calls removed; generic fallback and route trace proven. |
| `extra/llm_research/prefill/q6k_prefill_route_spec.py` | Q6 direct-packed fallback/oracle | Exact Q6 vocab and general declines use generic master semantics. |
| Prefill-only portions of `quant/q4_k_gemv_primitive.py` and `quant/q6_k_gemv_primitive.py` | Direct-packed oracle helpers | Decode/shared consumers split into master-owned generic modules. |
| Current decode/prefill A/B adapters, detailed timing/resource capture, fault reproducers, clock controls | Qualification and diagnosis | Canonical production authority extracted first. |
| Handwritten historical rollback implementations reachable only from Git | Optional oracle fixture | Restore only if a named dev qualification test requires them. |

### 6.3 Keep on exp or delete after banking

- BubbleBeam/FutureSight candidate-generation and ranking development;
- BoltBeam interchange/search adapters and raw ledgers;
- `decode/*topology_search*`, `prefill/*topology_search*`, direct-packed M4 search, packed-WMMA M2b search;
- MMQ candidate lineage, refuted DS4 experiments, pure-register experiments, Hd/split sweeps, and microbench probes;
- the uncommitted M2a/M3 full-kernel prototypes created under the superseded no-human interpretation;
- raw benchmark outputs after compact route/evidence summaries are retained.

Uncommitted superseded prototypes must be inventoried and removed with explicit patches; do not use a broad reset or
delete anything whose ownership is uncertain.

### 6.4 Unresolved mixed files

The quant primitive, packed-WMMA, route-manifest, memory-adaptive, and flash helper modules contain mixed runtime and
research concerns. Agents must extract production symbols by consumer closure; moving the apparent file wholesale is
forbidden.

## 7. Target master architecture

```text
tinygrad/llm README or CLI
  -> production model/load policy
  -> selected route descriptors/configurations
  -> master-owned route executor interface
  -> tinygrad compiler/lowering/backend
  -> ordinary tinygrad fallback on decline

dev qualification
  -> same production interface
  -> optional handwritten oracle/fallback adapter
  -> generated-vs-oracle A/B and fault reproduction

exp search
  -> BoltBeam + BubbleBeam/FutureSight + tinygrad candidate evaluation
  -> selected configuration/policy update
  -> bounded qualification patch to dev
```

Final `master` production import closure must not require `extra/llm_research`. This is an ownership rule, not a ban on
human-written compiler code or custom-kernel execution.

## 8. Dependency-ordered low-agent packets

Each packet is small enough for one low agent, owns a disjoint primary file set, and ends with a committed checkpoint
only after review. Agents do not commit unless explicitly assigned commit authority.

### P0: correct definitions and freeze the inventory

CPU-only. No runtime change.

- Mark the superseded strict scope/audit as a historical stricter interpretation, not current authority.
- Generate a machine-readable ledger with the four classification axes, current caller, benchmark dependency,
  disposition, destination, prerequisite, recovery commit, and confidence.
- Inventory every uncommitted superseded prototype and classify retain, rewrite, or discard.
- Add a documentation consistency test preventing `custom_kernel == fallback` and `missing provenance == not search`.

Gate: all benchmark-producing routes are explicitly protected; all specialized rollback candidates name a removal
prerequisite.

### P1: canonical production benchmark and README boundary

CPU-only until final measurement.

- Finish `python -m tinygrad.llm.bench` as the stable production entrypoint.
- Make `tinygrad/llm/README.md` sufficient from clone through model load, generic control, route trace, and authority
  invocation.
- Extract stable orchestration from `extra/llm_research/bench.py`; detailed campaign harnesses stay dev/exp.
- Record commit, dirty state, model identity, target facts, route IDs, selected descriptors, warmups, samples, and
  throughput in one versioned JSON schema.
- Add mocked/CPU smoke coverage; do not manufacture performance evidence.

Gate: fresh-clone CPU smoke imports no `extra/llm_research`; real numbers remain explicitly pending GPU recertification.

### P2: production policy and route interface

CPU-only.

- Define master-owned route descriptors and execution interfaces without requiring autonomous source artifacts.
- Extract minimal model facts, selected route table, structural guards, and admission data from research manifests.
- Preserve route IDs and selected parameter values exactly.
- Keep optional hash/trace support from `generated_runtime.py`, but do not require a new full-program search export to
  retain a historically selected production route.

Gate: all current supported shapes resolve to the same selected route/configuration in table-driven tests.

### P3: WMMA-LDS graph-prefill promotion

CPU implementation; GPU required only for promotion.

- Extract the live `route_pf16_graph_gemm` runtime subset into `tinygrad/llm`.
- Preserve selected candidate payloads, compiler options, cache identity, decline behavior, and route observer output.
- Remove only its corresponding `route_ops` adapter after direct callers switch.

CPU gate: compile/import, selected-shape binding, unsupported-shape generic fallback, no extra import.

GPU promotion gate: route-bound numerical parity and same-session performance on the published 8B shapes.

### P4: packed-WMMA production extraction

CPU implementation; GPU required only for promotion.

- Split the live packed-WMMA selector, current geometry, correctness canary admission, warmstart builder, execution
  adapter, and required packed-weight semantics from the broader MMQ research lineage.
- Preserve `PACKED_WMMA_GEOM` behavior and identities in this organizational pass; provenance improvement is separate.
- Do not pull refuted DS4/MMQ searches or qualification harnesses into master.
- Remove the selector/warmstart `route_ops` adapters only after direct master ownership exists.

CPU gate: all admitted quant/role/shape combinations select identical options and decline identically.

GPU promotion gate: canary parity, 14B route trace, pp512/pp4096 A/B, and healthy post-run device.

### P5: replace direct-packed master fallback

CPU implementation first; GPU only for regression confirmation.

- Split generic packed-weight storage/attachment from the direct-packed executor attachment.
- Replace a packed-WMMA decline with ordinary tinygrad graph/scheduler fallback on master.
- Change forced `direct_packed` compatibility behavior to a documented generic fallback or fail-loud result; it may not
  import a dev implementation.
- Replace the exact Q6 vocab special case with ordinary tinygrad semantics while preserving lazy final-token behavior.
- Remove Q4/Q6 direct describe/emit adapters from `route_ops.py`.
- Move direct-packed specs and prefill-only helpers to dev as optional oracles.

CPU gate: load/dispatch tests cover disabled mode, missing candidate, failed canary, unknown shape, Q6 vocab, and no
hidden dev import.

GPU promotion gate: 8B/14B supported defaults retain route identity and speed; explicit generic fallback produces
correct tokens without invoking direct-packed. The known 14B direct rollback is not executed on master.

### P6: Q4 G3 and Q6 decode production promotion

CPU implementation; GPU required only for promotion.

- Move the G3 selected LaneMap lowering and required generic Q4 semantics into a master-owned decode module without
  changing generated UOps or selected geometry.
- Preserve the Q4 historical search lineage and BoltBeam policy identity as documentation/evidence, not runtime search.
- Extract the Q6 decode spec/emitter and shared quant semantics separately from prefill fallback helpers.
- Remove corresponding `route_ops` adapters after direct imports switch.

CPU gate: program identity/name, shape guards, emitted UOp/source digest where stable, and fallback behavior match.

GPU promotion gate: fixed-depth 8B/14B Q4 results and any supported Q6 authority retain correctness and route identity;
published Q4 decode endpoints remain within declared noise.

### P7: G4/G5 decode-attention production promotion

CPU implementation; GPU required only for promotion.

- Move the descriptor, executor, and required flash/lane/geometry closure into master-owned modules.
- Separate generic flash/warp/WMMA primitives from research timing and alternative-candidate code.
- Preserve exact G4 and G5 selected values, kernel identity, ctx threshold, SDPA behavior below threshold, and fail-loud
  behavior for unsupported KV-quant/rope-at-read shapes.
- Remove only the flash-decode `route_ops` adapter after callers switch.

CPU gate: G4/G5 descriptor identity, emitted program/resource expectations, ctx128 SDPA selection, ctx512+ flash
selection, and no extra import.

GPU promotion gate: 8B and 14B ctx128/512/4096 token parity, route identity, three-sample A/B, and device health.

### P8: fused prefill-attention ownership audit

Mostly CPU; GPU only if executable code changes.

- Retain the promoted route and custom-kernel execution.
- Verify every production dependency already resides under `tinygrad`; move only remaining research-owned runtime facts.
- Separate performance/parity campaigns from production descriptor and compiler primitives.
- Do not rewrite the attention kernel merely to eliminate `Tensor.custom_kernel`.

Gate: admitted Hq32/Hq40 grids select the same descriptor and generated program; ordinary SDPA remains fallback for
other grids; published prefill route identity is unchanged.

### P9: eliminate the production `route_ops`/`extra` seam

CPU-only after P2-P8.

- Replace the final route-manifest and memory-collector adapters with master-owned production interfaces.
- Delete `tinygrad/llm/route_ops.py` only after it has zero production callers.
- Strengthen the boundary audit from "extra imports only through route_ops" to "master production imports no
  extra/llm_research".
- Keep canonical production authorities under `tinygrad/llm`; dev/exp tools may import richer tiers only downward.

Gate: static import/source closure and clean-clone import both pass with `extra/llm_research` absent.

### P10: tier placement and branch derivation

CPU-only.

- Apply destination-based patches from exp to dev, retaining direct-packed oracles and qualification tools.
- Derive master from its current tip with only production routes, generic fallback, required tests, README, and compact
  evidence.
- Do not use whole-branch merges or equality as a goal.
- Record exact tips, tracked counts, retained oracle paths, and recovery commits.

Gate: master cannot import dev/exp assets; dev retains named oracles; exp retains active searches; all worktrees clean.

### P11: final AMD recertification and README publication

GPU required and separately operator-authorized.

- Run only after CPU/closure gates pass on exact candidate tips.
- Acquire the GPU lock and use the canonical authority.
- Verify route identity before accepting timing.
- Re-measure 8B/14B decode ctx512/4096 and prefill pp512/4096 at minimum; run full four-point prefill if README keeps
  that curve.
- Compare against the retained pre-migration numbers and investigate any change outside declared noise.
- Update README numbers only from the exact promoted commit.

Gate: correctness, route identity, performance, GPU health, clean tree, and artifact record all pass.

## 9. Parallelization rules

With four slots including the coordinator:

- P0 and P1 may run in parallel because they own docs/ledger versus CLI/README code.
- P3 and P4 may run in parallel only after P2 freezes the production route interface and only if their primary files do
  not overlap `model.py`, `prefill_routes.py`, or shared quant modules. Shared integration belongs to the coordinator.
- P6 and P7 may run in parallel after shared quant/warp destinations are frozen.
- P5 must integrate after P3/P4 because it changes decline behavior shared by both prefill families.
- P8 may audit in parallel but must not change shared compiler files while P7 edits them.
- P9-P11 are sequential integration/release packets.

Every agent receives exact owned paths, forbidden paths, CPU/GPU authority, required tests, and a stop condition. Agents
must report overlaps rather than editing another packet's primary files.

## 10. Tests and gates

### 10.1 Always-on CPU gates

- focused unit tests for the touched route family;
- route-selection truth table for supported/unsupported shapes;
- ordinary tinygrad fallback positive control;
- static import and source-closure scan;
- no production import from `extra/llm_research` for completed slices;
- route ID and selected descriptor stability;
- generated/source digest stability where deterministic;
- `python3 -m json.tool` for records;
- `git diff --check`;
- documentation link gate;
- organization and size-budget audits;
- clean-clone `python -m tinygrad.llm` and benchmark-smoke commands.

### 10.2 GPU gates

GPU work occurs only at the promotion checkpoints named above, never during file classification or mechanical moves.
Each run requires:

- `/tmp/gpu-bench.lock` ownership;
- exact clean commit/worktree identity;
- model and target identity;
- observed route/configuration identity;
- correctness/token parity before timing;
- matched warmups and samples;
- health before and after;
- no concurrent source changes.

### 10.3 Negative gates

The cleanup fails if:

- a published route silently becomes generic fallback;
- `custom_kernel` is used as a deletion predicate;
- a missing provenance file is treated as proof that search did not occur;
- master imports a dev oracle;
- direct-packed removal breaks model load or exact Q6 vocab semantics;
- a mechanical move changes selected parameters, emitted program identity, or benchmark route;
- a richer branch is merged wholesale into a cleaner branch;
- README numbers survive without exact-route recertification after executable changes.

## 11. Existing work disposition

Retain for review from the earlier `exp` work:

- generic artifact/hash validation;
- runtime route trace and explicit generic fallback records;
- strict import/source-closure tooling;
- canonical `tinygrad/llm` benchmark scaffolding;
- historical lineage recovery records.

Do not promote without reclassification:

- the `0/9` result as an authorship verdict;
- requirements for zero human-written lowering;
- BoltBeam-only full-kernel search infrastructure as the replacement architecture;
- blocked M2a/M3/M4 replacement grammars created only because historical search was presumed absent.

Committed additive work remains on exp until packet P0 records retain/rewrite/discard decisions. Uncommitted
superseded work is not committed merely to preserve it; Git history and explicit recovery records are sufficient.

## 12. Required records

- this scope;
- machine-readable route/file ownership ledger;
- benchmark-to-route dependency map;
- old path to production/dev/exp destination map;
- per-packet test and commit record;
- per-route observed identity before and after migration;
- direct-packed generic-fallback parity record;
- final master/dev/exp tips and cleanliness;
- exact README benchmark records from the final master tip.

## 13. Completion definition

The task is complete when:

- master retains every benchmark-producing search-selected route and its required lowering;
- master uses ordinary tinygrad fallback rather than a specialized handwritten rollback/oracle;
- Q4/Q6 direct-packed fallback implementations remain available on dev, not master;
- master production runtime has no dependency on `extra/llm_research`;
- search development and rejected candidates remain on exp;
- `tinygrad/llm/README.md` provides the supported clone-to-inference and benchmark path;
- model load and inference work on supported models with both selected and generic paths;
- exact route identities and published performance are recertified on the final master commit;
- master, dev, and exp are clean, pushed, and recorded.

The concise target is: **preserve the searched fast path, remove specialized fallback/oracle debt from master, and make
the production runtime self-contained under `tinygrad`.**
