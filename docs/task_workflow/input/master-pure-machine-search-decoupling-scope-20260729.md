# Master Pure-Machine-Search Hot-Path Decoupling Scope

Date: 2026-07-29

Status: superseded for implementation by `master-search-selected-runtime-cleanup-scope-20260729.md`; retained as the
record of the rejected no-human/full-program-synthesis interpretation

Authority: this document supersedes the repository-organization completion record wherever that record permits a
hand-authored custom kernel or an unproven search claim on `master`. It does not supersede hardware recovery, TinyGPU,
or unrelated compiler work.

Correction: the user-approved practical definition permits human-authored search spaces, descriptors, lowerers,
compiler primitives, and promotion decisions. Machine search means the performance-sensitive configuration was
explored/ranked/measured by the combined workflow. Do not execute this document's replacement-search packages or use
its `0/9` proof result as a historical authorship verdict. The corrected scope preserves search-selected production
routes and moves only specialized handwritten fallback/oracle implementations out of master.

## Implementation checkpoint (2026-07-29)

This migration was realigned after a history audit. The project generator was not BoltBeam alone: the historical
pipeline combined BoltBeam policy/evaluation, BubbleBeam/FutureSight candidate generation and ranking, and tinygrad
lowering/runtime. A missing current-tree export is therefore not evidence that the route was never generated.

- M0 and M1 remain useful and are implemented on `exp`: the nine-route provenance audit, route migration ledger,
  strict purity gate, generated-artifact catalog/loader, hash validation, runtime trace, and fail-closed generated-plan
  dispatch seam are committed.
- The audit result `0/9` is only the count of routes that satisfy the **new current-checkout end-to-end proof contract**.
  It must not be read as a historical authorship verdict.
- Q4_K decode G3 has a real, reachable combined-system lineage: a declared workload/search space, two BubbleBeam
  lane-layout candidates, deterministic FutureSight ranking, selected lane-map descriptor, G3 lowering, correctness
  and performance gates, default promotion, and later BoltBeam route-policy binding. The first action for G3 is to
  recover and replay that chain, not design a replacement search.
- Scheduler prefill has a recovered four-role candidate set plus separate correctness and timing records. It needs a
  recovered or replayed top-level request/run/ranking/export binding, not a new kernel implementation.
- Q6 decode, direct-packed Q4/Q6 prefill, packed-WMMA policy, G4/G5 decode attention, and prefill flash attention do not
  yet have equivalent route-bound search exports in the examined history. Their current labels remain claims to audit;
  no new search is authorized until their histories have been exhausted.
- BoltBeam `f6ee2763f47316112fbba40b91b859e0e7068a6d` and the uncommitted M2/M3/M4 prototypes are quarantined additive
  infrastructure. They are not the migration plan and must not be promoted merely because they satisfy the new schema.
- `dev` and `master` remain at their prior published commits. Runtime defaults, GPU state, and hardware recovery are out
  of scope during this recovery/package phase.

The machine-readable route-by-route historical result is
`docs/task_workflow/output/combined-generator-lineage-recovery-20260729.json`.

## 1. Outcome

`master` must be a runnable product branch whose performance-critical LLM path is selected by machine search and
lowered by tinygrad. It must not ship a model/shape-specific custom kernel body written or tuned by a person, even as a
disabled rollback. A user cloning `master` must be able to follow the README, run the supported hot path, observe the
selected generated route identities, and reproduce the performance claims from committed authority commands and
evidence.

The branch split is:

| Branch | Kernel ownership |
|---|---|
| `master` | Machine-search-selected plans/artifacts, generic runtime dispatch, ordinary tinygrad codegen, reusable model-agnostic backend primitives, and ordinary tinygrad graph fallback. |
| `dev` | Everything in `master`, plus hand-authored fallback/oracle kernels, debug adapters, qualification harnesses, and generated-vs-hand A/B controls. |
| `exp` | Everything in `dev`, plus search-space development, candidate generators, failed/refuted candidates, raw sweeps, and disposable probes. |

No hand-authored custom kernel implementation is retained on `master` merely because it is default-off, called a
fallback, wrapped by a dataclass, selected by generated policy, or useful for comparison.

## 2. Alignment decisions

These definitions are fixed for this migration.

1. **Search must author the selected plan.** Search selecting a human implementation is not pure machine search.
2. **A descriptor is not provenance.** A dataclass around a hand-written UOp function does not change authorship.
3. **A generated name is not provenance.** `generated`, `machine_authored_generated`, and route-manifest status are
   claims to verify, not evidence.
4. **The executing closure decides.** Classification follows runtime dispatch through every emitter, helper, manual
   geometry table, source string, and backend extension used to build the selected kernel.
5. **Generic compiler code is allowed.** tinygrad's scheduler, graph lowering, renderer, and model-agnostic hardware
   primitives are hand-maintained software that generates kernels; they are not route-local custom kernels.
6. **Backend ISA is allowed.** Generated kernels may lower to AMD ISA. The forbidden boundary is human ownership of a
   complete model/shape-specific kernel lifecycle, not the presence of WMMA, LDS, waitcnt, or assembly.
7. **Generic fallback is allowed.** Unsupported shapes may use ordinary tinygrad graph/scheduler lowering or fail
   loudly. A specialized hand-authored custom fallback is not allowed on `master`.
8. **Private search is compatible with a public runtime.** BoltBeam plus BubbleBeam/FutureSight may remain the search
   system, but its
   exported candidate must carry sufficient immutable provenance for this repository to prove that no manual post-edit
   became the shipped implementation. Users do not need the private search system to run the selected artifact.
9. **Numbers follow the exact artifact.** README throughput belongs to a route id, artifact digest, Git commit, target,
   model digest, measurement schema, and command. A result cannot survive a route replacement by narrative inheritance.
10. **No hidden `extra/` runtime.** The production default path on `master` must not import executable kernel bodies from
    `extra/llm_research`. Production generated artifacts and their loader live under `tinygrad/`.

## 3. Current result: the existing purity pass is insufficient

At `master` commit `ddc0e9784db6ba9512538772336497d53e1271b7`, the current census reports:

```text
TINYGRAD_DEFAULT_PURITY_PASS
9 named default routes
7 machine_authored_generated
2 tinygrad_scheduler_generated
0 final-default purity debt
```

That result is not sufficient for the new contract. The checker derives purity primarily from the route manifest's own
provenance strings and a hand-maintained overlay. It does not prove:

- the source closure that actually constructs the kernel;
- who authored the concrete loop/load/LDS/WMMA/reduction/store topology;
- a search-run identity or generator revision;
- hashes connecting search input, selected candidate, generated source/plan, runtime binding, and measured binary;
- that a generated file has not been manually edited;
- that every authority gate and cited artifact still exists on `master`;
- that the README command runs the same route/artifact represented by the table.

The current strict audit therefore fails under this scope: **0 of the 9 named default routes is presently packaged with
the complete proof chain required for final `master`**. This is a current-checkout packaging/provenance verdict, not a
historical authorship verdict and not evidence that the combined generator never ran. The supporting machine-readable
strict audit is
`docs/task_workflow/output/master-hot-path-provenance-audit-20260729.json`.

## 4. Current default-route audit

| Route | Current claim | Actual executing owner or selection source | Strict result | Required disposition |
|---|---|---|---|---|
| `decode_q4k_g3_generated` | machine-authored/generated | Recoverable BubbleBeam/FutureSight lane-map selection, G3 lowering, promotion gates, and BoltBeam policy binding | historical combined lineage recovered; final package incomplete | Replay/export the historical selected descriptor and deterministic lowering with immutable hashes; move any non-generated bridge/oracle to `dev`/`exp`. |
| `decode_q6k_coop_generated` | machine-authored/generated | `q6k_route_spec.py` wraps a hand-written lowering and dequant grammar | unproven/hand template | Replace with generated plan/module plus generic quant primitives; retain the old emitter only as a dev oracle. |
| `decode_flash_live_split_g4_kvboth` | machine-authored/generated | `flash_kernels.py` manually owns the full tile, LDS, softmax, PV, and store lifecycle | hand-authored UOp kernel | Generate the topology; remove the route-local builder from master. |
| `decode_flash_live_split_g5_kvboth` | machine-authored/generated | Same hand-authored flash closure with separate G5 geometry | hand-authored UOp kernel | Generate separately identified G5 plan and evidence; no reuse of a hand builder hidden behind the spec. |
| `prefill_flash_attention_generated` | machine-authored/generated | `tinygrad/schedule/wmma/kernels.py` explicitly declares fixed-geometry hand kernels | hand-authored UOp kernel | Replace the full attention builder with search output composed from generic primitives; split the current implementation to dev/exp. |
| `prefill_wmma_lds_dbuf_generated` | tinygrad scheduler-generated | Ordinary tinygrad matmul plus recovered four-role selected candidate set, timing, and correctness records | compiler-generated; historical export partially recovered | Preserve the compiler path and recovered bundle; recover or replay only the missing request/run/ranking/export binding. |
| `prefill_q4k_direct_tile4x4_default` | machine-authored/generated | Hand-written route spec/lowering and fixed options | unproven/hand template | Search/export the plan; current lowering becomes dev oracle. |
| `prefill_q6k_direct_generated` | machine-authored/generated | Hand-written route spec/lowering using hand dequant helpers | unproven/hand template | Search/export the plan; current lowering becomes dev oracle. |
| `packed_wmma_prefill_generated` | tinygrad scheduler-generated | Ordinary tinygrad matmul, but `PACKED_WMMA_GEOM` is a frozen manually copied table | hand-tuned schedule table | Rerun a real search and bind the resulting shape-keyed plan; the current table stays only on dev/exp. |

The repository already documents the last problem accurately in
`docs/packed-wmma-14b-machine-search-claim-scope-20260725.md`: the current packed-WMMA kernel is compiler-generated,
but the schedule constants have no demonstrable search provenance. This scope applies that honesty consistently to all
default routes.

## 5. File-level ownership boundary

### 5.1 Master-forbidden implementation surfaces

The following current surfaces are presumptively forbidden on final `master` unless rewritten as generated artifacts or
generalized into genuinely model-agnostic compiler infrastructure:

| Current path | Why it is in scope | Final owner/form |
|---|---|---|
| `tinygrad/schedule/wmma/kernels.py` | 383 lines of fixed-geometry hand-authored attention kernel builders | Hand implementation moves to dev/exp; master consumes generated plans/artifacts. |
| `tinygrad/llm/fused_attention.py` | Route-local custom-kernel injection currently composes the hand builder | Split: generic generated-artifact dispatch may stay on master; hand-kernel construction/fallback moves to dev/exp. |
| `extra/llm_research/flash_kernels.py` | Full decode attention UOp lifecycle written in Python | Dev/exp oracle and research implementation only. |
| `extra/llm_research/live_split_geometry.py` | Route-local geometry/combine implementation supporting the hand kernel | Search-space/spec data may remain on exp; generated runtime data moves under `tinygrad/llm/generated/`. |
| `extra/llm_research/gemv_g3_codegen_lowering.py` | Manually constructs Q4_K GEMV kernel topology | Dev oracle; generated replacement under `tinygrad/llm/generated/`. |
| `extra/llm_research/gemv_g2_lanemap.py` | Manually fixed lane ownership and packing geometry | Exp search-space input or dev oracle; selected plan exported as immutable generated data. |
| `extra/llm_research/q6k_route_spec.py` | Spec wrapper plus hand-authored executable lowering | Exp/dev; master retains only generated plan and generic quant/compiler primitives. |
| `extra/llm_research/prefill/q4k_prefill_route_spec.py` | Hand-authored direct-packed UOp lowering | Exp/dev. |
| `extra/llm_research/prefill/q6k_prefill_route_spec.py` | Hand-authored direct-packed UOp lowering | Exp/dev. |
| `extra/llm_research/prefill/packed_wmma_prefill_candidates.py` | Manually copied frozen schedule table and runtime binding | Search/oracle on dev/exp; shape-keyed generated plan/catalog on master. |
| `tinygrad/llm/route_ops.py` | Production seam dynamically imports executable route bodies from `extra` | Eliminate after generated artifacts have direct core owners. |

### 5.2 Conditional compiler/backend surfaces

These files are not automatically forbidden merely because they contain AMD-specific operations. Each must pass a
reusability audit:

| Surface | Keep on master only if |
|---|---|
| `tinygrad/codegen/late/warp_reduce.py` | It exposes a shape/model-independent compiler operation used by generated callers, contains no selected model geometry, and has generic semantic tests. Otherwise move it to dev. |
| `tinygrad/renderer/isa/amd_attention_abi.py` | Attention-specific descriptors are generalized into reusable backend primitives whose semantics do not encode the complete 8B/14B kernel lifecycle. Otherwise it is dev-only substrate. |
| `tinygrad/renderer/isa/amd_wmma_residency.py` | Register allocation policy is target-generic across generated WMMA callers and is not a hidden fixed-kernel schedule. |
| `tinygrad/schedule/wmma/{fragments,softmax,loop_state,composite}.py` | Each module is a reusable semantic/compiler primitive with multiple generated consumers; no module may assemble a complete model-specific kernel. |
| `tinygrad/renderer/{cstyle,isa/amd}.py` modifications | They lower general operations and do not recognize a route id, model shape, or hand-authored hot-kernel identity. |

If a conditional surface cannot satisfy this rule, the generated path must use a more generic compiler primitive or the
surface must leave `master`.

### 5.3 Master-allowed production surfaces

Final `master` may contain:

- `tinygrad/llm` model loading, policy, admission, dispatch, and measurement entrypoints;
- a generated artifact catalog and immutable generated plans/modules under `tinygrad/llm/generated/`;
- generic validation and loading code for those artifacts;
- ordinary tinygrad tensor graphs and scheduler/codegen/renderers;
- reusable hardware primitives with semantic, model-independent contracts;
- generated-file markers and deterministic drift checks;
- production correctness, route-binding, provenance, and benchmark tests;
- compact authority records needed to reproduce README claims.

## 6. Target architecture

```text
BoltBeam policy/evaluation + BubbleBeam/FutureSight candidate generation/ranking (outside runtime)
  -> public search request + search-space digest
  -> selected candidate export + rank/objective/result digest
  -> deterministic artifact generator
  -> tinygrad/llm/generated/{catalog,plans,artifacts,provenance}
  -> generic validator/loader
  -> tinygrad runtime route selection
  -> generic tinygrad compiler/backend primitives
  -> emitted AMD code object
  -> runtime trace + correctness + performance evidence
```

There are two acceptable final artifact forms:

1. **Generated plan data (preferred):** a JSON/data plan describes semantic graph, primitive composition, schedule,
   layouts, resource constraints, and shape guards. A generic master-owned compiler lowers it.
2. **Deterministically generated source module:** the search exporter emits a Python module or other compiler input with
   an `@generated` banner and content digest. No manual edits are permitted. The module may compose reusable primitives,
   but must not call a separate hand-authored route-local kernel builder.

A handwritten descriptor feeding a handwritten emitter is not either form.

### 6.1 Proposed master layout

```text
tinygrad/llm/
  README.md
  cli.py
  model.py
  route_selection.py
  generated_runtime.py          # generic validation/dispatch only
  generated/
    catalog.json                # generated route ids, guards, artifact/provenance hashes
    plans/*.json                # selected machine-search plans
    artifacts/*.py              # optional deterministic generated compiler inputs
    provenance/*.json           # search/export chain and evidence pointers
  bench.py                      # canonical user-facing authority command
```

The exact filenames may change during implementation, but these ownership boundaries may not.

## 7. Generated artifact and provenance contract

Every master hot-path route requires one provenance record containing at least:

- schema/version;
- route id and workload role;
- target backend and architecture;
- supported model-independent shape/quant guards;
- search-space id and SHA-256;
- public search request/workload digest;
- search system name and revision/commit identifier;
- search run id, timestamp, objective, budget, and candidate count;
- selected candidate rank and objective values;
- complete selected plan payload and SHA-256;
- deterministic exporter name/revision;
- generated source/plan path and SHA-256;
- declared reusable primitives;
- forbidden fallback/kernel identities;
- correctness evidence paths and hashes;
- performance evidence paths and hashes;
- runtime route trace schema and expected identity;
- recovery commit for the previous implementation;
- an explicit `manual_post_edit: false` field enforced by regeneration/drift checks.

If BoltBeam remains private, the exported record must still expose the non-secret request, selected plan, objective,
rank, run id, generator revision, and hashes. The tinygrad repository must be able to verify that the checked-in
artifact is byte-identical to the exported result. A prose statement that a private tool produced it is insufficient.

## 8. Strict purity audit

Add a new master gate; do not weaken the existing census in place until the new gate independently proves closure.

The gate must:

1. Enumerate actual default routes from runtime guards and the generated catalog.
2. Resolve static and permitted dynamic imports from every selected route to its executable closure.
3. Reject any master default closure containing:
   - route-local `Tensor.custom_kernel` with a human Python builder;
   - full-kernel UOp loops/loads/stores not marked and verified as generated;
   - route-local `Ops.CUSTOM`, `Ops.CUSTOMI`, raw source, ISA, or binary injection;
   - manual geometry tables without generated provenance;
   - imports from `extra/llm_research`;
   - missing generator/search/evidence hashes;
   - generated-file drift;
   - a default route whose observed runtime identity differs from the catalog.
4. Permit reusable compiler/backend primitives only through an explicit reviewed allowlist with model/shape-independent
   semantic contracts.
5. Verify every cited authority path exists at the audited master commit.
6. Produce a machine-readable closure graph showing why every default route passed.
7. Fail when the manifest says `pure` but the executing closure is unproven.

The final verdict vocabulary is:

| Verdict | Meaning |
|---|---|
| `SEARCH_GENERATED_REPRODUCIBLE` | Search and deterministic generation can be rerun from repository-visible inputs. |
| `SEARCH_GENERATED_ATTESTED` | Private search generated the immutable exported plan; public hashes and export metadata prove the checked-in artifact is unchanged. |
| `TINYGRAD_GENERIC_GENERATED` | Ordinary generic tinygrad graph fallback, not the claimed optimized main hot path. |
| `HAND_AUTHORED_CUSTOM` | Human-owned complete kernel or schedule; dev/exp only. |
| `UNPROVEN` | Evidence or closure incomplete; cannot ship as the claimed master hot path. |

Both `SEARCH_GENERATED_*` states may execute on the optimized master hot path. `TINYGRAD_GENERIC_GENERATED` is allowed
as unsupported-shape fallback, but it does not establish the project's machine-search performance claim.

## 9. Route migration work packages

The packages are recovery-first. A replacement grammar, runner, or kernel is permitted only after the relevant package
has a written history result showing that the combined generator path cannot be recovered or replayed. Missing fields
in the new strict schema do not, by themselves, authorize replacement engineering.

### M0: freeze claims and install the strict gate

- Record the current nine-route audit.
- Change the current census/reporting so the old self-declared pass cannot be mistaken for the new strict verdict.
- Add closure, artifact-existence, and generated-drift tests.
- Do not delete or repoint a route in this package.

Gate: strict audit fails for named, expected reasons and has positive/negative fixtures proving it cannot be passed by
renaming a hand kernel or adding a dataclass.

### M1: generated catalog and loader

- Define the artifact/provenance schemas.
- Add `tinygrad/llm/generated_runtime.py` and generated catalog layout.
- Add deterministic catalog generation and `--check` drift mode.
- Add runtime trace fields: route id, plan digest, generated artifact digest, target, shape, and fallback reason.
- Keep the existing routes active until parity is proven.

Gate: a synthetic generated candidate loads, binds, traces, and fails closed on any hash/shape/target mismatch.

### M2: recover and replay historical generator outputs

- Restore the Q4_K G3 search space, BubbleBeam candidates, FutureSight ranking, selected LaneMap descriptor, lowering,
  promotion evidence, and BoltBeam route policy from reachable commits into an isolated replay surface.
- Re-run the static selector and deterministic lowering without GPU access and compare the selected payload/source
  identity with the historical route.
- Preserve the archived four-role scheduler-prefill candidate set, correctness record, and timing record by content
  hash; recover the missing request/run/ranking/export header where history permits.
- Produce one lineage map per route distinguishing recovered source facts, reproducible replay, missing proof, and
  genuinely absent search stages.
- Do not bind or change production defaults in this package.

Gate: the recovery result is independently replayable from reachable repository objects, and every unrecovered field is
named without converting absence into a claim of human authorship.

### M2b: scheduler-generated prefill candidates

Migrate `prefill_wmma_lds_dbuf_generated` first because the kernel already lowers through ordinary tinygrad. Export its
recovered selected candidate plan and complete its search provenance. Rerun search for
`packed_wmma_prefill_generated` only if M2 proves that its source cannot be recovered; the current `PACKED_WMMA_GEOM`
table cannot be promoted merely because it is labeled generated and its keys omit shape.

Requirements:

- shape-key candidate keys include `(target, quant, role, m, n, k)`;
- no copied constants without a search-result record;
- selected plan is immutable and route-bound;
- ordinary tinygrad fallback remains available for non-admitted shapes;
- the old geometry table remains only as a dev A/B oracle.

### M3: Q4_K/Q6_K decode GEMV

- Recover/replay and export the historical Q4_K lane-map route before attempting any new Q4 search. Export or regenerate
  Q6_K coop/partial plans only after its combined-system history is exhausted.
- Move concrete loop/lane/load/reduce/store decisions out of master hand-written Python.
- Retain only quant-format semantics and reusable packed-load/dequant compiler primitives on master.
- Move old builders to dev/exp as oracles with explicit debug-only flags.
- Prove all 8B/14B roles select the generated digest and do not reach the dev oracle.

### M4: Q4_K/Q6_K direct-packed prefill

- Search topology, parts, output layout, local/reduce axes, and opts from declared spaces.
- Export shape-keyed selected plans.
- Remove `_direct_packed_opts`-style human constants from the master decision path.
- Keep the old spec/lowering as a dev oracle until token parity and performance gates pass.

### M5: decode flash attention

- Make tile size, split geometry, staging, query grouping, combine topology, and reduction structure search-plan data.
- Replace `flash_kernels.py` as the executing source with generated plan/module artifacts.
- Generalize any retained LDS/warp/reduction operations into reusable compiler primitives.
- Generate independent G4 and G5 route artifacts; neither may inherit purity from the other.
- Prove fixed-depth 8B and 14B route binding at ctx512 and ctx4096.

### M6: prefill flash attention

This is the hardest package and the current largest master violation.

- Search/export the full primitive composition for the admitted Hq=32/Hkv=8 and Hq=40/Hkv=8 paths.
- Remove the final runtime dependency on `tinygrad/schedule/wmma/kernels.py`.
- Split `fused_attention.py` into generic generated dispatch and dev-only hand implementation.
- Generalize or remove attention-specific ABI/residency helpers that encode the complete fixed kernel.
- Preserve exact causal, GQA, accumulation, and V-layout semantics.
- Prove 8B and 14B token parity, full-output numerical bounds, route identity, source/binary hashes, and measured speed.

### M7: remove the production `extra` seam

- Promote immutable generated catalog/plans/provenance into `tinygrad/llm/generated/`.
- Remove `tinygrad/llm/route_ops.py` dynamic imports from the production closure.
- Remove all master tests/docs that require hand-authored or research kernel bodies.
- Keep research generators, failed candidates, raw sweeps, and hand oracles on dev/exp.
- Run the strict import and source-closure audit from a fresh clone.

### M8: README and canonical benchmark

- Make `tinygrad/llm/README.md` the runnable hot-path quickstart.
- Add a canonical `python -m tinygrad.llm.bench` authority entrypoint under `tinygrad/`; it must not import research
  measurement policy from `extra` on master.
- The command emits one versioned JSON record containing commit, dirty state, model digest, device/driver facts, route
  ids, plan/artifact hashes, correctness result, warmups, samples, and throughput.
- Root README tables link to committed records generated by that command.
- Remove or demote any number whose route/artifact cannot be reproduced from the final master tip.

## 10. Dev/exp hand-fallback contract

Hand-authored kernels remain useful as correctness or performance oracles. Their ownership must be explicit:

- place reusable hand oracles under a dev-owned debug/qualification namespace;
- require an opt-in debug flag that does not exist on master;
- never import them during ordinary master module import or route discovery;
- preserve exact recovery commits before moving/deleting old paths;
- maintain generated-vs-hand A/B tests on dev;
- retain failed/refuted candidates and raw search sweeps only on exp;
- prevent dev-only test or document references from re-entering master through a broad cherry-pick.

There is no specialized custom-kernel rollback on master. Master rollback means selecting the previous generated artifact
or ordinary tinygrad graph lowering. If neither is safe, admission fails loudly.

## 11. Correctness and purity gates

Every route package must pass before its old implementation leaves the dev comparison path:

- generated artifact/schema/hash validation;
- generated drift check;
- source-closure purity audit;
- forbidden master import/path scan;
- runtime route-bound trace with expected route and artifact digest;
- no-hidden-fallback assertion;
- CPU compile/import tests for all supported shapes;
- GPU numerical comparison against ordinary tinygrad and the dev hand oracle;
- end-to-end greedy token parity for 8B and 14B where the route applies;
- exact-context decode authority at ctx512 and ctx4096;
- whole-prefill authority at pp512/1024/2048/4096;
- GPU health and post-run reset/recovery checks;
- `sz.py`, organization audit, runtime-boundary audit, doc-link gate, and `git diff --check`;
- clean-clone README command test.

Generated and hand A/B runs use isolated processes and the same model, target, clock policy, route shape, warmups, and
sample count. A result without route/artifact trace is not admissible.

## 12. Performance and README acceptance

The current README claims are baselines to revalidate, not values automatically inherited by the new artifacts:

| Workload | 8B current claim | 14B current claim |
|---|---:|---:|
| Decode ctx512 | 114.19 tok/s | 69.70 tok/s |
| Decode ctx4096 | 103.07 tok/s | 62.45 tok/s |
| Prefill pp512 (2026-07-24 curve) | 3694 tok/s | 1945 tok/s |
| Prefill pp4096 (2026-07-24 curve) | 3236 tok/s | 1785 tok/s |
| Newer bounded prefill endpoints | 3768 tok/s pp512 smoke | 2026 pp512 / 1880 pp4096 medians |

For promotion:

- run at least three same-session samples per reported endpoint;
- compare against the current path, ordinary tinygrad fallback, and llama.cpp where the README makes that comparison;
- require token/numerical correctness and a healthy GPU before considering speed;
- do not retain a README value if the exact generated artifact does not reproduce it within the declared noise policy;
- if purity requires a slower artifact, report the new measured number honestly and keep the faster hand path on dev;
- do not claim that machine search beats llama.cpp unless the final generated master artifact wins the same-session
  authority comparison.

## 13. Branch execution and commit topology

1. Complete M2 historical recovery on `exp`; freeze or discard replacement-search prototypes that duplicate recovered
   combined-system functionality.
2. Implement only the remaining M2b-M6 gaps on `exp` with both generated and hand paths available.
3. Promote qualified generated runtime/catalog code and hand-oracle debug support to `dev` with destination-based
   patches.
4. Run generated-vs-hand A/B and all qualification gates on the exact dev commit.
5. Derive `master` from the qualified dev result using destination-based patches that omit every hand implementation,
   debug flag, research generator, raw artifact, and dev-only test.
6. Run master gates from a clean clone before updating README numbers.
7. Commit final evidence and update the generated catalog hashes.
8. Push `exp`, `dev`, and `master`; record exact tips, counts, recovery commits, and branch cleanliness.

Never merge a richer tier wholesale into a cleaner tier. Never delete the dev/exp oracle before the generated master
route has independent correctness and performance evidence.

## 14. Required records

- this scope document;
- `master-hot-path-provenance-audit-20260729.json`;
- strict purity closure output for each branch;
- generated artifact/catalog schema and drift output;
- route-by-route old-to-new ownership and recovery map;
- search export/provenance record for every promoted route;
- generated-vs-hand and generated-vs-generic A/B evidence;
- final README benchmark records;
- final branch identities and tracked counts.

## 15. Stop conditions

Stop a route promotion and leave it on dev/exp if any of the following holds:

- the search/export chain cannot distinguish generated output from manual post-editing;
- the generated plan still calls a hand-authored route-local kernel builder;
- a required authority artifact is absent or points outside the audited commit;
- runtime trace cannot prove the generated artifact executed;
- correctness differs beyond the frozen tolerance;
- GPU health fails;
- the generated artifact misses the supported README shape;
- the only way to preserve performance is to retain the hand implementation on master;
- private search metadata cannot be exported without secrets and no safe attested record can be produced.

A stopped route may continue as a dev/exp candidate. It may not remain a master default under an optimistic provenance
label.

## 16. Completion definition

This migration is complete only when all of the following are true:

- every optimized master hot-path route is `SEARCH_GENERATED_REPRODUCIBLE` or
  `SEARCH_GENERATED_ATTESTED` under the closure audit;
- every unsupported-shape fallback on master is ordinary `TINYGRAD_GENERIC_GENERATED` or fails loudly;
- master contains no hand-authored custom kernel, manual performance geometry table, specialized custom rollback, or
  production import of `extra/llm_research`;
- hand kernels and A/B controls remain available on dev/exp with recovery records;
- `tinygrad/llm/README.md` and root README provide one supported command path from clone to measurement;
- every published number binds to the exact final master commit and generated artifact digest;
- correctness, route binding, performance, boundary, size, link, and clean-clone gates pass;
- all three branches are clean, pushed, and recorded.

Until those gates pass, the honest status is: **the current hot path may be fast and may have been informed by machine
search, but `master` has not yet proven that its main hot path is purely machine-search authored.**
