# q8 FFN dynamic scheduler observability scope (2026-06-19)

Purpose: exhaust the **visible dynamic gap** behind the q8 FFN decode route before reopening any S1-S4 hand tuning.

This is option 2 from the post-S0 discussion: build a tinygrad-native trace/counter bridge around HCQ for this exact
primitive. It is not a generic profiler and not a new decode route.

## Starting point

S0 closed native q8 decode ownership as bounded primitive work:

- tinygrad AMD DSL/ASM fused gate/up is correct but slow: `166.649us`;
- target is `<=60us`;
- hipcc/LLD oracle is the schedule target;
- COMGR fused-C is also slow: `146.88us`;
- tinygrad ASM and hipcc/LLD both emit `16` native signed dot4 ops;
- tinygrad ASM has fewer static instructions overall (`218` vs `336`);
- visible static deltas are load shape and address/bit-manipulation:
  - global loads `22` vs `11`;
  - grouped VALU `+37`;
  - DS only `+3`.

S0 proves this is not a missing primitive, missing dot4, or obvious static instruction-count blowup. What remains is
dynamic scheduler behavior: load coalescing, latency hiding, dependency ordering, occupancy/resource limits, cache
behavior, and wait placement.

## Existing observability assets

| asset | status | use here |
|---|---|---|
| `extra/q8_ffn_asm_schedule_audit.py` | S0 PASS/CLOSE | static instruction and load-shape authority |
| `extra/qk_hcq_attribution.py` | PMU-4 PASS | probe-local HCQ launch/graph attribution |
| `docs/primitive-pmu-observability-result-20260619.md` | PMU works for HIP, not rocprof-visible HCQ | explains why this scope is tinygrad-native first |
| `tinygrad/runtime/support/hcq.py::hcq_profile` | existing timestamp mechanism | device-time rows without external profiler |
| `tinygrad/runtime/ops_amd.py` `PMC` / `SQTT` / `PROFILE` | built-in AMD counter/thread-trace hooks | optional Level-4 evidence if stable power/tooling works |
| `tinygrad/renderer/amd/sqtt.py` + `tinygrad/viz` | SQTT decode/display support | optional instruction trace parsing |

## Non-goals

- Do not tune the q8 consumer yet.
- Do not route model defaults.
- Do not reopen q8 producer ownership.
- Do not build a broad profiler clone.
- Do not claim PMU-level root cause from Level-3 launch/timing metadata.
- Do not require full model decode until microbench evidence is stable.

## Evidence levels

| level | evidence | allowed conclusion |
|---|---|---|
| L2 static | instruction counts, resource metadata, load shape | "the code shape differs" |
| L3 HCQ dynamic | launch rows, device timestamps, graph/eager boundary, variant timings | "this variant moves device time / host boundary" |
| L4 AMD counters/trace | PMC/SQTT counters or instruction trace for HCQ | "this is memory/issue/occupancy/cache/wait dominated" |

S0 already provides L2. This scope first builds L3 for q8, then attempts L4 only if the built-in AMD hooks work.

## Target question

Explain which sentence is true:

1. **load-shape bound**: `global_load_b32` / scalarized q4/q8 loads dominate; vector/coalesced load selection is the
   first compiler feature to fund.
2. **address-math bound**: per-lane address/scale/min extraction and bit manipulation dominate; specialization or
   renderer CSE is the first feature to fund.
3. **reduction bound**: wave/LDS reduction dominates enough to justify a reduction rewrite.
4. **wait/scheduler bound**: static shape is close enough, but dynamic issue/latency hiding is poor; this needs a real
   AMD scheduler/codegen project.
5. **resource/occupancy bound**: VGPR/SGPR/LDS/private segment or local-id descriptor choices cap occupancy.
6. **unobservable with current hooks**: we need PMU/SQTT support before reopening.

The scope is complete when one sentence wins, or when it proves we cannot distinguish them with available hooks.

## DSO-0 — preflight and authority snapshot

Tasks:

- rerun S0 and store the static result path;
- record tinygrad commit, GPU target, `PROFILE/PMC/SQTT` availability, and whether stable power state is required;
- record the existing authority timings:
  - hipcc/LLD oracle `<=60us` target;
  - COMGR fused-C `146.88us`;
  - tinygrad ASM `166.649us`.

Artifact:

- `bench/q8-ffn-dynamic-scheduler-observability/preflight.json`.

Gate:

- can import/build the q8 ASM consumer and S0 artifact;
- no model route changes.

Kill:

- if the q8 ASM consumer no longer builds or correctness has regressed, stop and repair provenance first.

## DSO-1 — q8 HCQ dynamic attribution row

Extend the PMU-4 probe-local attribution style to the q8 ASM consumer:

- wrap `HCQProgram.__call__` and/or the direct runner context;
- capture program name, runtime class, code hash, launch geometry, local size, kernarg size, buffer count;
- capture `group_segment_size`, `private_segment_size`, `wave32`, `rsrc1/2/3` when runtime exposes it;
- force `wait=True` for microbench rows and record returned device time plus host wall time;
- run repeated warm launches of:
  - tinygrad ASM full real-GGUF gate/up;
  - hipcc/LLD artifact gate/up;
  - COMGR fused-C gate/up.

Artifact:

- `bench/q8-ffn-dynamic-scheduler-observability/hcq_rows.json`.

Gate:

- artifact contains comparable L3 rows for all three consumers;
- medians match the banked timing class within reasonable noise;
- host/device gap is not the dominant explanation for the q8 ASM microbench.

Kill:

- if wait/device time cannot be collected for the direct q8 consumers, fall back to DSO-3 variant timing only and mark
  L3 timing incomplete.

## DSO-2 — resource and occupancy metadata audit

Tasks:

- extract resource fields for tinygrad ASM, hipcc/LLD, and COMGR:
  - kernarg size;
  - group segment / LDS;
  - private segment / scratch;
  - wave32/wave64;
  - SGPR/VGPR if available from metadata or disassembly;
  - launch grid/local;
  - workgroups per CU estimate from VGPR/LDS/local size when possible.
- join these fields with S0 static counts.

Artifact:

- `bench/q8-ffn-dynamic-scheduler-observability/resource_audit.json`.

Gate:

- can state whether occupancy/resource metadata is a plausible first-order explanation.

Kill:

- if metadata is missing for tinygrad's minimal ELF, document that gap and keep going with variant timing.

## DSO-3 — controlled variant ladder

Do not tune the full consumer blindly. Build minimal variants that isolate one suspected dynamic cause at a time.

Variants:

| variant | purpose | expected signal |
|---|---|---|
| reduction-only | measures wave/LDS reduction cost | if close to full miss, reduction bound |
| dot-loop synthetic q4/q8 | keeps dot4/load schedule, removes real scale/min | address/scale-min cost |
| fixed-scale/min real loads | keeps real q4/q8 loads, removes scale/min select chain | address-math vs load-shape |
| q8-only load sweep | scalar vs packed q8 loads if expressible | q8 load shape |
| q4-only load sweep | scalar vs wider q4 loads if expressible | q4 load shape |
| wait placement sweep | conservative waits vs grouped waits where correctness permits | wait/scheduler bound |
| local shape smoke | `(128,1,1)` vs any safe descriptor/local-y variant | descriptor/local-id cost |

Rules:

- each variant must use the same HCQ timing collector;
- each variant must state whether it preserves real GGUF dataflow or is synthetic;
- no variant proceeds to model route;
- stop early if one variant explains `>=50us` of the `~106us` miss to target.

Artifact:

- `bench/q8-ffn-dynamic-scheduler-observability/variant_ladder.json`.

Gate:

- at least three variants run and classify reduction/address/load/wait as high, medium, or low explanatory power.

Kill:

- if all variants move `<15us`, classify the route as broader scheduler/codegen, not local primitive tuning.

## DSO-4 — built-in AMD PMC/SQTT attempt

Only after DSO-1..3.

Use tinygrad's own AMD hooks before writing new low-level counter code:

- run the q8 ASM consumer with `PROFILE=1 PMC=1` and a small `PMC_COUNTERS` set;
- if stable power state blocks PMC, record the exact blocker and do not force system-level changes silently;
- run a minimal `PROFILE=1 SQTT=1` trace only if buffer size and runtime overhead are acceptable;
- parse available `ProfilePMCEvent` / `ProfileSQTTEvent` rows into a compact JSON summary.

Candidate counter/trace questions:

- are VMEM/cache counters unusually high for tinygrad ASM vs oracle/control?
- is issue dominated by VMEM waits, VALU, or LDS?
- does SQTT show long gaps around `global_load_b32 -> s_waitcnt -> dot4`;
- do vector/coalesced loads in the oracle produce materially different issue cadence?

Artifacts:

- `bench/q8-ffn-dynamic-scheduler-observability/pmc_attempt.json`;
- `bench/q8-ffn-dynamic-scheduler-observability/sqtt_attempt.json` if trace succeeds.

Gate:

- either produce useful L4 rows, or produce a precise blocker such as `stable_power_required`,
  `sqtt_buffer_overflow`, `profile_event_missing`, or `minimal_elf_unmapped`.

Kill:

- if PMC/SQTT is unstable or too intrusive, do not depend on it for the q8 verdict. Use DSO-3.

## DSO-5 — classifier and final decision

Build a single result file that joins:

- S0 static audit;
- HCQ dynamic rows;
- resource metadata;
- variant ladder;
- optional PMC/SQTT rows.

Artifact:

- `bench/q8-ffn-dynamic-scheduler-observability/result.json`;
- doc: `docs/q8-ffn-dynamic-scheduler-observability-result-20260619.md`.

Classifier output must be one of:

| label | meaning | action |
|---|---|---|
| `load_shape_bound` | load width/coalescing explains enough of the miss | scope a tinygrad load-selection feature |
| `address_math_bound` | address/scale-min work explains enough | scope specialization/CSE feature |
| `reduction_bound` | reduction cost explains enough | scope reduction rewrite |
| `wait_scheduler_bound` | timing changes depend on wait/load/dot ordering | project-level scheduler |
| `resource_occupancy_bound` | resource metadata or occupancy explains enough | scope resource/register fix |
| `unobservable_l4_required` | L3 variants inconclusive; PMU/SQTT missing | stop until L4 works |
| `closed_project_level` | no bounded cause moves enough | keep q8 native ownership closed |

Final gate:

- only reopen q8 native ownership if a bounded label predicts `>=50us` improvement and preserves the one-kernel fused
  gate/up lifecycle.

Otherwise:

- keep A4 as research artifact only;
- keep tinygrad native q8 decode closed;
- record the compiler roadmap item precisely.

## Recommended execution

Run in this order:

1. DSO-0/1 in one probe: establish q8-specific HCQ rows.
2. DSO-2 metadata join.
3. DSO-3 variant ladder with early stop.
4. DSO-4 only if DSO-3 cannot classify.
5. DSO-5 result doc and update the performance-primitive source of truth.

Expected outcome from current evidence: `wait_scheduler_bound` or `closed_project_level` is most likely. The only
bounded alternative worth testing is `load_shape_bound`, because S0 showed `22` vs `11` global loads.
