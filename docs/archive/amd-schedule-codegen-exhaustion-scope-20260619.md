# AMD schedule/codegen exhaustion scope (2026-06-19)

Purpose: exhaust the remaining **AMD scheduler/codegen** question by primitive, not by open-ended compiler ambition.

This is the project-level follow-up after q8 Route A closed as a bounded decode edit and Tensile prefill proved a
mature backend schedule can run through tinygrad HCQ. The question is no longer "is there a faster kernel?" That is
already proven for both q8 decode and prefill. The question is:

> Can tinygrad learn the schedule classes needed for the proven oracles, or should those schedules remain external
> artifacts / policy-bound backend imports?

## Non-goals

- Do not search all possible AMD schedules.
- Do not reopen q8 decode as a single-feature A2 without new attribution.
- Do not reopen pure-tinygrad prefill WMMA knobs already killed by POWN-1.
- Do not route external artifacts by default.
- Do not mix decode and prefill quality gates.
- Do not call a schedule "owned" until it is generated or assembled by tinygrad without relying on hipcc/rocBLAS/Tensile
  runtime selection.

## What "exhaustive" means here

Exhaustive means every relevant schedule/codegen feature is classified for each primitive oracle:

| classification | meaning | allowed next step |
|---|---|---|
| `expressible_now` | current UOps/renderer/AMD ASM can emit it | build only if it clears the primitive gate |
| `bounded_extension` | one small renderer/assembler/runtime feature can emit it | run a feature proof with a movement gate |
| `project_level` | requires scheduler/register allocation/software pipeline/compiler architecture | keep as roadmap unless multiple primitives justify it |
| `artifact_only` | recovered mature backend schedule is usable but not natively generated | research flag / policy decision |
| `not_worth_owning` | Amdahl or quality does not justify ownership | close |
| `tooling_blocked` | attribution/disassembly/counters cannot name the feature yet | build tooling first or keep the claim lower-confidence |

The project is exhausted when every row in the oracle matrix has one of these labels, a gate, and a stop condition.

## Authority oracles

### Oracle 1 - q8 decode MMVQ lifecycle

Primitive:

- fused Q4_K x q8 gate/up int-dot consumer;
- producer-side q8 activation side-channel from RMSNorm/apply;
- in-model W==D decode and dNLL gate.

Known authority:

| line | result |
|---|---|
| artifact route | PASS_RESEARCH: W==D `1.051-1.063x`, dNLL `+0.002887` |
| hipcc/LLD lifecycle | `115.24us`, graph-safe, no in-process HIP |
| tinygrad COMGR fused-C | correct but `146.88us` |
| tinygrad AMD DSL/ASM | correct but `166.649us` |
| DSO ladder | body-insensitive `~0.151-0.153ms` variants vs full `0.166ms` |
| Route A A0/A1 | no bounded `>=30us` native feature |
| PMU/SQTT pass | capture works; SQTT decode unusable; no A2 reopen |

Current state:

- `artifact_only` works as a research route.
- Native ownership is `project_level` unless stronger attribution names a bounded feature.

### Oracle 2 - prefill Tensile-class fp16 GEMM

Primitive:

- PREFILL_V2 fp16 realized-weight matmul bucket;
- ffn_gate/up, ffn_down, attn_q/o;
- warm pp512/pp1024 and dNLL gate.

Known authority:

| line | result |
|---|---|
| pure tinygrad POWN-1 sweep | KILL: best `42.0 TFLOPS`, below `>=62 TFLOPS` gate |
| extracted Tensile ffn_gate/up | `66.8-66.9 TFLOPS`, correct, no copies, HCQ |
| extracted Tensile ffn_down | `68.9 TFLOPS`, StreamK, no workspace |
| extracted Tensile attn_q/o | `58.9 TFLOPS`, correct/stable |
| weighted model | ~`1.40x` full pp512, ~95% llama if graph-routed |
| TPE-6 block transfer | exact/copy-free, GPU math `1.53x`, host-sync route redirected to graph helper |

Current state:

- `artifact_only` works at kernel level and block math level.
- Native ownership is `project_level` unless the Tensile schedule anatomy yields a bounded codegen feature.

### Optional Oracle 3 - long-prefill attention

Only activate if long-prompt prefill profiling shows attention again dominates.

Current state:

- reuse-free custom attention is refuted;
- real frontier would be LDS/register-locality flash-style scheduling;
- not part of the first exhaustion pass unless its Amdahl share rises above the project gate.

## Cross-primitive feature matrix

This is the finite audit surface. Each feature must be classified for q8 and prefill.

| feature | q8 decode oracle | prefill Tensile oracle | likely class today |
|---|---|---|---|
| special instruction selection | `v_dot4_i32_iu8` already emitted | WMMA/MFMA already emitted | mostly `expressible_now` |
| vector/global load shape | oracle uses wider/coalesced loads | Tensile global-read vectorization | `bounded_extension` or `project_level` depending on allocator/schedule coupling |
| waitcnt placement | oracle differs; grouped wait standalone moved ~0.8us | software-pipelined waits likely central | `project_level` if global scheduler needed |
| `s_clause` / `s_delay_alu` | present in oracle, absent in tinygrad ASM | likely part of mature scheduling | `project_level` unless semantics can be encoded locally |
| register allocation / live ranges | suspected gap; no bounded q8 feature isolated | likely central to 42 -> 67 TFLOPS | `project_level` |
| occupancy / VGPR / SGPR policy | q8 needs attribution before claiming | POWN knobs did not solve plateau | `project_level` |
| software pipelining | not yet named as q8 feature | likely key Tensile advantage | `project_level` unless one-shape proof is possible |
| LDS staging / layout | q8 producer needs staged reductions; consumer less clear | Tensor-class schedule may use LDS carefully; noLDS regressed | `project_level` |
| reduction topology | q8 standalone reductions moved ~13us, below A2 | prefill reductions not dominant for GEMM | q8 `not_worth_owning` standalone |
| launch/kernarg contract | artifact route proven for q8/prefill | named descriptor + raw kernarg proven | `artifact_only` / bounded runtime support |
| graph/rebind boundary | q8 graph-safe | TPE graph route still open | bounded runtime work, not codegen |
| attribution tooling | PMU/SQTT capture works, SQTT decode fails | HCQ attribution Level 3 works, Level 4 absent | `tooling_blocked` for stall-level search |

## Exhaustion phases

### SCE-0 - ledger authority refresh

Goal: make a single machine-readable schedule/codegen ledger across q8 and prefill.

Deliverables:

- `bench/amd-schedule-codegen-exhaustion/oracle_matrix.json`
- row for q8 decode oracle;
- row for prefill Tensile oracle;
- row for optional attention only if activated.

Required fields:

- primitive name;
- oracle artifact/doc;
- current tinygrad implementation;
- speed gap;
- correctness/quality gate;
- known schedule deltas;
- current classification for each feature;
- next allowed build;
- stop condition.

Gate:

- every feature in the matrix has a non-empty classification and evidence pointer.

Kill:

- if a row cannot cite a measured oracle or tinygrad baseline, it stays out of the exhaustion set.

### SCE-1 - oracle schedule anatomy extraction

Goal: normalize the two schedule oracles into comparable contracts.

q8 tasks:

- reuse `oracle_contract.json`;
- add PMU/SQTT status from `pmu_sqtt_evidence.json`;
- include artifact route timing and in-model gate.

prefill tasks:

- consume TPE-5 shape matrix and TPE-6 block result;
- extract or summarize Tensile macro tile, workgroup, depthU, vector widths, LDS bytes, VGPR/SGPR, StreamK/GSU,
  launch geometry, kernarg shape, and disassembly instruction classes for the three roles.

Gate:

- "Tensile does X, tinygrad does Y, missing capability is Z" table exists for ffn_gate/up at minimum.

Kill:

- if Tensile anatomy cannot be recovered beyond "external kernel fast", keep prefill as `artifact_only` and do not
  start native transfer.

### SCE-2 - tinygrad capability map

Goal: map oracle schedule features against current tinygrad surfaces.

Surfaces:

- UOps and `custom_kernel`;
- AMD renderer;
- AMD DSL/ASM `Ops.PROGRAM`;
- HCQ runtime / raw kernarg / named descriptor helpers;
- TinyJit/HCQGraph capture and rebind.

Output classes:

- `expressible_now`;
- `bounded_extension`;
- `project_level`;
- `artifact_only`;
- `not_worth_owning`;
- `tooling_blocked`.

Gate:

- at least one cross-primitive feature is classified as `bounded_extension` with a credible movement budget, or the
  audit explicitly says no such feature exists.

Kill:

- if all material features are `project_level` or `artifact_only`, stop native builds and write the project roadmap.

### SCE-3 - bounded feature proof, only if SCE-2 finds one

Goal: prove one reusable codegen feature moves a primitive enough to justify ownership.

Candidate features:

- named-descriptor/raw-kernarg runtime helper if artifact import is the accepted route;
- vectorized load selection that survives q8/prefill microbench gates;
- explicit wait/schedule annotation if it can be encoded semantically;
- a one-shape software-pipelined WMMA/MFMA loop if it is not just hand-maintained assembly.

Gates:

| primitive | feature-proof gate |
|---|---:|
| q8 decode | `>=30us` q8-shaped microbench movement or `>=25us` full consumer movement |
| prefill GEMM | ffn_gate/up `>=62 TFLOPS` isolated or `>=1.25x` block/model transfer if runtime-only |
| graph/runtime | removes TPE-6 host-sync loss and keeps block `>=1.20x` after overhead |

Kill:

- q8 feature moves `<15us`;
- prefill proof stays near `42 TFLOPS`;
- graph/runtime proof does not transfer to block/model;
- feature is per-shape hand assembly with worse maintainability than the artifact route.

### SCE-4 - native schedule rebuild, only after SCE-3 passes

Goal: rebuild a full primitive with the new native capability.

q8 gate:

- fused gate/up `<=75us` to continue, credible path to `<=60us`;
- lifecycle `<=129.2us`;
- W==D decode `>=3%`;
- dNLL `<=0.01`.

prefill gate:

- ffn_gate/up `>=62 TFLOPS`;
- full matmul bucket model `>=1.25x`;
- warm pp512/pp1024 pass after graph route;
- dNLL `<=0.01`.

Kill:

- native rebuild remains below artifact/oracle by enough that artifact import is clearly the better research route.

### SCE-5 - roadmap closeout

Goal: close the project-level question.

Possible final verdicts:

| verdict | meaning |
|---|---|
| `artifact_policy_decision` | mature schedules are usable through HCQ, but native generation is not worth funding now |
| `bounded_codegen_feature_found` | one reusable feature passes SCE-3 and should be built |
| `native_transfer_pass` | tinygrad emits a schedule class that reaches the oracle gate |
| `project_level_renderer_rewrite` | no bounded feature exists; only a broader AMD scheduler/register allocator/software pipeline project remains |
| `tooling_first` | SQTT/PMU/disassembly attribution must be fixed before more codegen claims are honest |

## First execution slice

The correct next slice is **SCE-0 + SCE-1**, not a renderer rewrite.

Deliverables:

- `extra/amd_schedule_codegen_exhaustion.py`
- `bench/amd-schedule-codegen-exhaustion/oracle_matrix.json`
- `docs/amd-schedule-codegen-exhaustion-result-20260619.md`

The script should be read-only:

- ingest existing JSON artifacts;
- summarize q8 and prefill rows;
- classify known features using the table above;
- fail if required evidence files are missing;
- make no model route or runtime change.

Pass:

- q8 and prefill rows are populated with evidence;
- every feature has a classification;
- next build is either "none, roadmap only", "graph/runtime helper", or one named bounded codegen feature.

Expected result before running:

- q8 remains `artifact_only` for research and `project_level` for native generation;
- prefill remains `artifact_only`/graph-runtime for near-term measurement and `project_level` for native generation;
- no native renderer build starts unless SCE-1 finds a tighter Tensile feature than POWN-1 exposed.

## Why this follows the principles

- It defines the primitive boundary before building.
- It treats artifact oracles as references, not defaults.
- It keeps in-model decode and warm prefill gates separate.
- It records refutations as stop conditions.
- It does not overclaim PMU/SQTT: capture exists, decoded attribution does not.
- It gives native codegen a fair path, but only through measured reusable schedule features.
