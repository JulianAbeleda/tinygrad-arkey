# 14B decode G5 steady-state throughput recovery scope

Date: 2026-07-26

Status: evidence-qualified implementation scope

Delegation status: **NOT YET READY for autonomous low-reasoning agents**. The technical direction is qualified, but the reproducible harnesses, atomic task packets, dependency graph, and task-level acceptance criteria listed below must be completed first.

Hardware: AMD RX 7900 XTX, gfx1100, wave32

Source handoff: `docs/HANDOFF_14b_decode_depth_decay_20260726.md`

## Objective

Recover the deep-context throughput lost by the 14B G=5 flash-decode geometry without regressing 8B G=4 correctness or speed.

The target is approximately 620 GB/s effective model bandwidth at ctx4096, corresponding to roughly 64 tok/s for the 14B model. The implementation target is to bring the G5 KV_BOTH flash tile's deep steady-state cost close to G4 while preserving the current one-read cooperative K/V staging contract.

This scope also repairs the route attribution defect that currently reports the G5 execution as a G4-promoted route.

## Confirmed baseline

### Full-model defect

| Configuration | ctx512 | ctx4096 | Effective bandwidth trend |
|---|---:|---:|---:|
| 8B ours, G=4 | 113.86 tok/s | 102.57 tok/s | 581 -> 578 GB/s |
| 14B ours, G=5 | 68.39 tok/s | 59.41 tok/s | 621 -> 575 GB/s |

The 14B depth-dependent latency slope is approximately 0.617 microseconds per added context token, versus 0.270 microseconds for 8B. Normalized by layer count, the G5 slope is approximately 2.06x the G4 slope.

### Static production-kernel resources

Compile-only probe: MAXC=4608, S=48, KV_BOTH, Hkv=8, Hd=128, materialized Tc=4096. Both expected generated kernel names matched exactly once. No GPU dispatch or PMC was used.

| Shape | Workgroup | VGPR | SGPR | LDS | Scratch | Spills |
|---|---:|---:|---:|---:|---:|---:|
| G4, Hq=32 | 128 threads, 4 waves | 54 | 29 | 8192 B | 0 B | 0 |
| G5, Hq=40 | 160 threads, 5 waves | 91 | 27 | 8192 B | 0 B | 0 |

The G5 kernel uses 68.5% more VGPRs per thread. Spilling is refuted for this materialization. LDS is identical, so K_ONLY's LDS reduction is not the primary lead.

### Controlled GPU timing discriminator

The probe used the production KV_BOTH emitter plus fused combine, Hkv=8, Hd=128, MAXC=4608, S=48. Every GPU command held `/tmp/gpu-bench.lock`. The power profile was `auto` before and after and was never changed. Each captured linear contained both expected flash-tile and fused-combine names; expected-name match count was 2 for every configuration. All outputs were finite and nonzero.

| Shape | Tc | Median synchronized replay | Launched tiles | ns/tile |
|---|---:|---:|---:|---:|
| G4, Hq=32 | 512 | 0.085751 ms | 384 | 223.310 |
| G5, Hq=40 | 512 | 0.085551 ms | 384 | 222.789 |
| G4, Hq=32 | 4096 | 0.129293 ms | 2304 | 56.117 |
| G5, Hq=40 | 4096 | 0.171072 ms | 2304 | 74.250 |

At shallow depth, fixed launch, synchronization, and combine costs hide the difference. At Tc=4096, G5 costs 32.3% more per launched tile. This confirms a G/depth interaction inside the current flash path, independent of model GEMVs and dispatch growth.

The timing brackets tile plus fused combine and includes host synchronization. It does not by itself distinguish VGPR residency, SIMD scheduling, barrier tails, or memory-issue rate.

### Measurement authority reconciliation

The later G3-G8 direct-executor sweep is not comparable evidence. It divided median milliseconds by a hard-coded 16 rather than the launched-tile count and included direct graph construction/dispatch/output synchronization. Its near-flat values therefore cannot refute the 32.3% TinyJit replay delta.

The clean authority remains the TinyJit synchronized replay result above, with `Hkv*S*ceil_to_16(ceil(Tc/S))/16 = 2304` launched tiles at Tc=4096. The owned harness now reproduces it with one lock owner, dynamic AMD power-control discovery, exact source-match count 2, finite/nonzero outputs, and 15 samples per configuration. Full raw samples are recorded in [14b-decode-g5-authority-timing-20260726.json](14b-decode-g5-authority-timing-20260726.json). The auxiliary cached-binary JIT runner-name field is empty and is recorded as unavailable; it is not used as the positive control.

The authority result is complete: G5 is 2.17% slower than G4 at Tc=512 and 32.23% slower at Tc=4096. The direct-executor G sweep remains non-authoritative because it used an invalid denominator and different timing path.

### Route attribution defect

The singleton candidate in `tinygrad/llm/decode_routes.py` is named `decode_flash_live_split_g4_kvboth` but admits every positive integral Hq/Hkv ratio. Hq=40 therefore executes G5 geometry under a G4 route identity.

The manifest's `decode_flash_block_tile_g5_konly` entry claims `promoted_default`, but runtime never selects it, its named promotion artifacts are missing, and its flatness claim measured allocation rather than depth. K_ONLY correctness is not established for G5.

## Conclusions authorized by the evidence

1. A G5-specific KV_BOTH geometry/resource search is justified.
2. Static spilling is not the current lead.
3. VGPR-driven residency or a five-wave scheduling/barrier cliff is the leading mechanism class.
4. K_ONLY is not authorized as a default or as the first performance experiment.
5. PMC collection is not the first step; static resources and controlled geometry experiments are cheaper discriminators.
6. G4 and G5 require separate route identity, admission, tuning, and evidence even if they initially share the same emitter.

## Work track A: repair route identity without changing execution

Introduce explicit KV_BOTH candidates for the two admitted production shapes:

- G4: B=1, Hq=32, Hkv=8, Hd=128, split=48, KV_BOTH.
- G5: B=1, Hq=40, Hkv=8, Hd=128, split=48, KV_BOTH.

Initially both candidates must call the current emitter with byte-equivalent parameters. This change establishes honest attribution and separate evidence boundaries; it is not a performance change.

Required route tests:

- Hq=32 binds the G4 route ID.
- Hq=40 binds the G5 route ID.
- Unsupported Hq values fail admission unless separately evidenced.
- Generated program identity and outputs remain unchanged for the two existing shapes.
- The G5 route is visible to the route census and manifest checks.

Do not merely tighten the G4 guard. There is no flash fallback, so doing so without adding G5 would break 14B.

Demote the unreachable G5 K_ONLY manifest entry to blocked research. Do not classify it as refuted because G5 correctness has not been tested, and do not retain `promoted_default` because the runtime and evidence do not support that status.

## Work track B: establish the resource cliff

Capture the actual worker-produced G4 and G5 programs at Tc=512 and Tc=4096:

- Kernel name and program fingerprint
- Source and code-object fingerprint
- VGPR and SGPR counts
- LDS/group segment
- Scratch/private segment
- Spill counts
- Code size
- Workgroup and grid dimensions

The G4/G5 timing authority is now established. A broad G sweep is optional follow-up evidence, not a prerequisite for a G5-specific candidate search; direct-executor timings or hard-coded denominators remain diagnostic only.

Every capture must include expected-name positive controls and nonzero match counts. If the same shape produces different programs by depth, explain that before comparing occupancy.

Produce a controlled G sweep using the same KV_BOTH emitter:

- G in {3, 4, 5, 6, 7, 8}
- Hkv=8
- Hd=128
- Identical MAXC, split, Tc, dtype, and staging
- Static resources for every generated program
- Synchronized replay time normalized per launched 16-token tile
- Finite-output check for every run

The sweep must determine whether the VGPR increase and steady-state timing show a discontinuity at G=5 or vary smoothly with G.

## Work track C: explain and reduce G5 VGPR pressure

Diff the generated source, lowered UOps, and resource allocation for G4 versus G5. Identify the first compiler or representation stage where G5's register requirement diverges materially from G4.

Investigate these mechanism classes in order:

1. Range/lane lowering changes caused by WARPS=5 or THREADS=160.
2. Register lifetime extension across cooperative staging and barriers.
3. Online-softmax accumulator or reduction values retained across the tile loop.
4. Address/index expressions that fail to fold for G5 constants.
5. Register allocation granularity or a high-index virtual that inflates the descriptor.
6. Five-wave workgroup scheduling imbalance after register pressure is reduced.

Prefer representation or lifetime fixes that benefit the general emitter. Do not add a G5-only compiler exception unless a general invariant cannot express the correction.

For each candidate, record:

- Static resources
- Deep ns/tile
- Shallow ns/tile
- Numeric result versus KV_BOTH baseline/reference
- Generated source or UOp delta
- Whether G4 changes

## Work track D: G5 candidate search

Search only after the baseline G sweep and resource divergence are understood. Keep split_size=48 fixed initially because the existing sweep already established it as the best tested split for the current G5 geometry.

Candidate dimensions may include:

- Query-head/warp partitioning
- Cooperative staging ownership independent from query-head ownership
- Tile-loop scheduling and barrier placement
- Register lifetime reduction
- Equivalent accumulator/reduction organization
- Workgroup geometry that avoids a five-wave scheduling cliff

Reject candidates that improve timing by duplicating K/V HBM traffic, weakening bounds, changing semantics, or relying on unverified cache residency.

## Work track E: PMC occupancy, gated

Use occupancy PMCs only if one of these conditions holds:

- Static resources predict a residency difference but timing does not follow it.
- Two candidates have similar resources but materially different deep timing.
- A candidate reduces VGPR use and the remaining gap needs hardware attribution.

Required counters include SQ waves and busy cycles sufficient to estimate waves in flight. Collection must run inside the isolated worker, hold the GPU lock, verify a positive kernel match, print event counts, and restore `power_dpm_force_performance_level=auto` even on interruption.

Do not reuse the prefill PMC path as though it were decode authority.

## Work track F: K_ONLY, separately gated

K_ONLY removes V from LDS but loads V through the per-query-head path, relying on cache reuse for G copies. Halving LDS is not sufficient evidence that it improves occupancy or HBM behavior.

The compile-only paired screen now makes K_ONLY a legitimate candidate for correctness qualification. At Hq=40/Hkv=8/Hd=128/MAXC=4608/S=48/Tc=4096, same-run metadata was:

| Variant | VGPR | SGPR | LDS | Scratch/spills |
|---|---:|---:|---:|---:|
| KV_BOTH | 93 | 30 | 8192 B | 0 / 0 |
| K_ONLY | 58 | 27 | 4096 B | 0 / 0 |

K_ONLY reduced VGPR by approximately 37.6% and LDS by 50% in this pair. This is static evidence only. It does not authorize runtime use, promotion, or a performance claim.

The isolated correctness gate passed all required boundaries and depths, with K_ONLY versus KV_BOTH exactly equal and both within approximately 1e-8 of an independent NumPy GQA reference. The positive controls matched the expected tile and combine kernels in every subprocess.

The first clean timing comparison did not convert the resource reduction into a speedup. In the same direct executor and fresh locked subprocesses, K_ONLY was 1.78% slower than KV_BOTH at Tc=512 and 0.39% slower at Tc=4096. Outputs were finite and byte-identical per context. The normalized tile denominator in this probe differs from the earlier production microbenchmark, so use this result for the A/B delta only, not for absolute ns/tile comparison.

Current K_ONLY verdict: **correctness-qualified, performance-neutral/slightly negative, not promotable**. Keep it blocked research unless a later worker-level measurement demonstrates a real residency benefit that this direct executor does not expose.

Before any K_ONLY performance claim, require:

- Hq=40 K_ONLY versus KV_BOTH/reference numerical parity
- Multiple seeds and randomized q/cache values
- Contexts around tile and split boundaries: 15/16/17 and 47/48/49
- Target depths 512 and 4096
- End-to-end multi-token/logit parity
- Guarded or canary-backed cache/output checking
- Ring/wrap coverage when applicable
- Static VGPR, SGPR, LDS, scratch, spill, and workgroup comparison

Keep KV quantization and rope-at-read outside the first K_ONLY gate. They share the load site and require separate evidence.

Only benchmark K_ONLY if correctness passes and static resources predict a useful residency change. The static screen satisfies the resource condition; correctness remains entirely open.

## Promotion gates

A G5 candidate is promotable only when all conditions hold:

1. Correctness passes against the current KV_BOTH/reference path across the admitted G5 shape and required depth cases.
2. No out-of-bounds, canary, or GPU-fault signal appears.
3. Deep Tc=4096 normalized tile time improves materially over the 74.250 ns/tile baseline.
4. G4 shallow and deep timing do not regress beyond measurement noise.
5. Full-model 14B fixed-depth ctx4096 reaches the performance target or the remaining gap is accounted for outside flash decode.
6. Full-model ctx512 does not lose the current lead.
7. Route identity, manifest status, and authority artifacts describe the actual selected implementation.
8. Power state, kernel-match counts, program identity, and timing samples are recorded with the verdict.

Primary performance target:

- G5 deep tile+combine cost within 10% of G4 under the controlled probe, or
- 14B fixed-depth ctx4096 at approximately 64 tok/s with representative repeated measurements.

## Explicit non-goals

- Repeating the split-size sweep before geometry changes
- Re-proving cooperative KV_BOTH global-load count
- Rechecking route or dispatch-count growth
- Treating model-wide effective bandwidth as direct flash-kernel HBM bandwidth
- Promoting K_ONLY based on LDS size alone
- Broad attention rewrites before the G5 resource cliff is localized
- General PMC infrastructure work unrelated to this decode question
- Fixing the pre-existing multi-checkpoint vectorized-store compile error

## Measurement discipline

- Hold `/tmp/gpu-bench.lock` for every GPU run.
- Verify the power profile before timing and restore/verify `auto` afterward.
- Instrument inside isolated workers, never only in the parent process.
- Require a known-positive kernel match and print match counts beside every verdict.
- Treat empty captures as probe failures, never clean results.
- Use fixed-depth authority, not max-context allocation labels.
- Report total effective model bandwidth and normalized per-tile kernel time separately.
- Keep raw samples and distinguish host-synchronized replay from device-only timing.

## Recommended sequence

1. Land the behavior-preserving G4/G5 KV_BOTH route split and manifest correction.
2. Capture exact worker artifacts at shallow and deep Tc.
3. Run the static-resource and normalized-timing G sweep.
4. Locate the source of the 54-to-91 VGPR jump.
5. Test the smallest general resource/lifetime corrections.
6. Search G5 geometry only after the cliff is localized.
7. Run correctness gates on the leading candidate.
8. Run fixed-depth 14B full-model validation at ctx512 and ctx4096.
9. Use PMCs only if attribution remains ambiguous.
10. Evaluate K_ONLY only as a separately correctness-qualified candidate.

## Low-agent delegation readiness

This document currently defines the technical campaign, not yet an exhaustive low-agent execution protocol. Low-reasoning agents may perform bounded mechanical work only after the relevant task packet exists and its dependencies are complete.

Current readiness checklist:

- [x] Full-model defect and target recorded.
- [x] Static G4/G5 resource discriminator completed with positive kernel-name controls.
- [x] Controlled G4/G5 deep timing discriminator completed under the GPU lock.
- [x] Route and K_ONLY structural audit completed.
- [x] Spilling refuted for the tested Tc=4096 materialization.
- [x] G5-specific KV_BOTH search justified.
- [x] Existing opt-in static candidates screened: inline reduction and split-score are resource-worse; stage coalescing is byte-identical/neutral.
- [x] K_ONLY compile-only screen shows a materially lower-resource G5 code object and therefore justifies a correctness-only gate.
- [x] K_ONLY correctness gate passes boundary/depth/multi-seed parity with positive kernel controls.
- [x] K_ONLY timing gate is complete: no positive performance result; route promotion is blocked.
- [x] Measurement discrepancy classified: the direct G sweep used an invalid tile denominator and is not comparable to the TinyJit authority.
- [x] Static-resource probe preserved as a committed reusable command or harness.
- [ ] GPU microbenchmark moved from `/tmp/g5_decode_microbench.py` into an owned reusable harness or existing harness mode.
- [x] Raw baseline samples, metadata, hashes, environment, and expected match counts stored in a stable artifact.
- [x] One parameterized TinyJit timing-authority worker reproduces G4/G5 with a single lock owner and hard timeout.
- [ ] Exact task dependency graph instantiated under `docs/task_workflow/input/`.
- [ ] Every task packet names allowed and forbidden files.
- [ ] Every task packet contains exact commands, positive controls, output paths, schemas, pass thresholds, and stop conditions.
- [ ] Finite G-sweep and candidate matrices pinned so low agents cannot invent unreviewed geometry.
- [ ] Reviewer ownership assigned for semantic/compiler decisions.
- [ ] GPU concurrency and recovery protocol copied into every GPU task packet.
- [ ] Probe lifecycle and end-of-campaign deletion decisions recorded for every temporary or campaign-specific executable.

The earlier direct G sweep is **non-authoritative**: it passed finite-output checks but divided milliseconds by a hard-coded 16 and used a different executor. Its timings are not used for candidate selection. The owned authority above is the measurement of record.

Until every unchecked item required by a task is resolved, a low agent must not infer missing procedures or broaden the experiment.

## Execution task DAG to instantiate

Create these as separate Markdown task packets under `docs/task_workflow/input/`. Each task moves independently through `input/`, `in_progress/`, and `output/`.

### G5-T1: behavior-preserving route identity split

Dependencies: none.

Agent class: low, CPU-only.

Purpose: add explicit G4 KV_BOTH and G5 KV_BOTH candidate identities while preserving current generated behavior. Add exact bind tests. Do not optimize geometry.

### G5-T2: manifest and census correction

Dependencies: G5-T1.

Agent class: low, CPU-only.

Purpose: record the actual G5 KV_BOTH route, demote unreachable K_ONLY to blocked research, and make route-census attribution honest. Do not claim new promotion evidence.

### G5-T3: reusable static-resource capture

Dependencies: none.

Agent class: low, CPU-only/compile-only.

Purpose: preserve the successful resource extraction method in an existing owned harness or minimal reusable audit surface. It must emit kernel identity, hashes, VGPR, SGPR, LDS, scratch, spills, code size, and launch geometry with positive match counts.

### G5-T4: reproducible timing baseline

Dependencies: G5-T3.

Agent class: low, serialized GPU.

Purpose: replace the ephemeral `/tmp` probe with a reproducible owned mode, rerun the four baseline configurations, and store raw samples plus environment and power-state evidence.

### G5-T5: fixed G sweep

Dependencies: G5-T3 and G5-T4.

Agent class: low, serialized GPU.

Purpose: run the pinned G in {3,4,5,6,7,8} KV_BOTH matrix. No compiler changes are allowed. Produce static resources and normalized shallow/deep timing for every point.

### G5-T6: G4/G5 divergence artifact

Dependencies: G5-T3 and G5-T5.

Agent class: low for capture, stronger reviewer for interpretation.

Purpose: capture source, UOp, lowering-stage, and allocation differences at the first reproducible divergence. The low agent records facts only and must not choose a compiler fix.

### G5-T7: candidate matrix approval

Dependencies: G5-T6.

Agent class: strong reviewer.

Purpose: define a finite candidate matrix addressing the proven divergence. Record expected invariant, allowed files, and rollback for each candidate before low agents implement anything.

### G5-T8: candidate implementation and microgates

Dependencies: G5-T7.

Agent class: low only for one pre-approved candidate per task packet.

Purpose: implement exactly one approved candidate, run its static and timing microgates, and report. No opportunistic compiler or route changes.

### G5-T9: correctness qualification

Dependencies: a G5-T8 candidate passing its resource/timing threshold.

Agent class: low for prescribed test execution, strong reviewer for verdict.

Purpose: run numerical, token, bounds/canary, depth-boundary, and multi-token correctness gates. Any mismatch blocks promotion.

### G5-T10: full-model promotion gate

Dependencies: G5-T9.

Agent class: serialized GPU with strong review.

Purpose: run fixed-depth 14B ctx512/ctx4096 authority and 8B regression measurements, then update route evidence only if all promotion gates pass.

### G5-T11: probe consolidation and pruning

Dependencies: G5-T10, or a reviewer-approved terminal negative verdict for the campaign.

Agent class: low for reference inventory and prescribed deletion; strong reviewer approves retention decisions.

Purpose: inventory every probe, microbenchmark, capture script, temporary harness mode, and campaign-only artifact introduced or used by this scope. Promote only generally reusable measurement capability into an existing owned harness. Delete completed one-off probes after their commands, raw evidence, and verdict are preserved. Do not leave executable campaign history in the active tree.

Required classifications:

- `promote_reusable`: generally useful capability with a stable owner and more than one durable consumer.
- `retain_regression`: a small automated test defending the fixed invariant.
- `retain_reproducer`: the unique or materially smallest reproducer for unresolved behavior.
- `delete_ready`: completed one-off probe whose evidence and recovery identity are preserved.
- `delete_after_capture`: probe removable after its missing durable verdict or command is banked.

Required outputs:

- Complete probe inventory with path, purpose, owner, consumers, LOC, and classification.
- Replacement owner for every promoted reusable capability.
- Stable evidence or concise verdict pointer for every deleted probe.
- Gross and net LOC removed.
- Confirmation that documentation and command references no longer point to deleted probes.
- Confirmation that no archive directory was created as a substitute for deletion.

### G5-K1: optional K_ONLY qualification

Dependencies: G5-T3 and explicit reviewer approval. Independent from the KV_BOTH critical path.

Agent class: prescribed low-agent tests followed by strong review.

Purpose: run the separate K_ONLY correctness/resource matrix. Do not benchmark or promote K_ONLY until correctness and static-resource gates pass.

Current status: static-resource gate **passes**; runtime correctness gate **pending**.

## Required task-packet schema

Every delegated task must state:

- Task ID and dependency IDs
- Objective and non-goals
- Agent class and whether GPU access is allowed
- Exact files allowed to change
- Exact files or subsystems forbidden to change
- Starting commit identity and required input artifacts
- Exact commands and environment
- Positive controls and expected nonzero counts
- Output paths and machine-readable schema
- Baseline values and tolerances
- Pass, blocked, and failure criteria
- Stop conditions
- Power and GPU-lock protocol when applicable
- Required final report fields
- Commit boundary and rollback instruction
- Probe lifecycle classification and cleanup owner

Low agents must not fill missing fields by judgment. A packet missing any execution-critical field remains blocked.

## Shared GPU protocol for delegated tasks

- Only one GPU task may run at a time.
- Every GPU command must hold `/tmp/gpu-bench.lock`.
- CPU/static tasks may run concurrently only when they do not edit the same files.
- Verify `power_dpm_force_performance_level` before measurement and restore/verify `auto` afterward.
- A GPU reset, page fault, non-finite output, missing expected kernel match, empty capture, or non-`auto` final power state stops the task.
- Do not retry unexplained failures automatically.
- A killed or interrupted run has no performance authority until power state and GPU health are re-established.
- Raw samples must be retained; summary-only results are insufficient.

## Probe lifecycle and cleanup policy

Every probe must have a declared expiration condition when it is created. A probe is not permanent merely because it found a result once.

Keep executable probe code only when it is:

- A current automated regression test.
- A reusable owned measurement facility with multiple durable consumers.
- The unique or materially smallest reproducer for an unresolved issue.
- An operational diagnostic that is still invoked and documented.

Delete probe code when it is:

- Superseded by a stronger or more general harness.
- Associated with a completed or refuted candidate.
- Used only to produce an already-banked artifact or verdict.
- Duplicating setup, capture, timing, or report machinery now owned elsewhere.
- An ephemeral task-specific wrapper around an existing reusable facility.

Before deletion, preserve only what remains durable:

- Exact command and environment when needed to interpret the result.
- Raw result artifact or stable authority pointer.
- Positive-control counts and relevant program identity.
- Concise conclusion and limitations.
- Audited commit identity so Git history can recover the implementation if necessary.

Do not preserve dead probes by moving them to `archive/`, `legacy/`, `scratchpad/`, or another inactive directory. Git history is the implementation archive. Durable evidence belongs in the campaign artifact or final scope report, not in an unowned executable.

Temporary files under `/tmp` must be deleted or deliberately recreated as an owned facility before campaign completion. A `/tmp` path must never become the only reproducible authority for a promoted result.

## Delegation boundary

Low agents are authorized for deterministic capture, prescribed matrix execution, mechanical route/manifest changes, and pre-approved single-candidate implementations.

Low agents are not authorized to:

- Invent compiler fixes or new geometry.
- Interpret an empty or partial capture as a negative result.
- Promote K_ONLY or any new route.
- Change candidate matrices or thresholds.
- Broaden admission beyond explicitly evidenced shapes.
- Combine NFC route attribution with performance changes.
- Decide that similar code represents identical knowledge.
- Retain a completed probe without an explicit lifecycle classification and owner.

Resource-mechanism interpretation, candidate approval, and promotion verdicts require stronger review.

## Definition of complete

This scope is complete when either:

- A correct, honestly attributed G5 route meets the deep-context performance target without G4 regression, or
- The G5 flash path is brought within 10% of G4 normalized deep tile cost and remaining full-model loss is conclusively attributed elsewhere.

In either outcome, completion additionally requires:

- G5-T11 probe inventory and pruning is complete.
- No completed one-off probe remains in the active tree without an explicit owner and retention criterion.
- Reusable measurement capability has one centralized owner rather than cloned campaign scripts.
- Unique unresolved reproducers are explicitly labeled and linked to the unresolved issue.
- Temporary `/tmp` probes are removed and are not the sole authority for any conclusion.
- Net probe/harness LOC change is reported.

A negative result is complete only when static resources, normalized timing, route identity, power state, positive controls, and raw measurements are all preserved well enough to prevent the same hypotheses from being repeated.

## Campaign closeout checkpoint (2026-07-26)

The low-agent execution has closed the evidence and probe lifecycle for this scope:

- The owned timing harness and raw JSON preserve the authoritative G4/G5 result: approximately +2.17% G5 cost at Tc=512 and +32.23% at Tc=4096 after exact tile normalization.
- Route admission, explicit G4/G5 identities, manifest state, resource capture, and positive-control requirements are recorded.
- K_ONLY is correctness-qualified but not performance-qualified: the direct A/B result was slightly slower in both tested contexts, so it is not promoted.
- Direct-executor G-sweep output is explicitly non-authoritative and is not used to choose a candidate.
- All campaign-specific `/tmp/g5_*.py` probes were deleted after their evidence was migrated; no repository files or `docs/task_workflow/` files were touched.

This closes the low-risk measurement and cleanup portion of the scope. No candidate has met the promotion gate, so full-model promotion remains intentionally pending. The remaining work is a new bounded kernel-geometry investigation (warp ownership, partitioning, or equivalent G5-specific code generation), not another rerun of the rejected probes.

## Geometry review result (2026-07-26)

The final low-cost review found no production-safe one-line G5 warp-ownership change. In `extra/llm_research/flash_kernels.py`, `WARPS=G`, `THREADS=32*G`, query-head ownership, cooperative staging, and both barriers are coupled. Reducing G5 to four waves would serialize the fifth query head or require a second pass/workgroup that duplicates KV traffic.

The only bounded follow-on experiment worth scoping is compile-only: retain five query warps but assign K/V staging to a four-wave loader subset with explicit predicates and barriers. It is admissible for GPU timing only if it materially lowers VGPRs while preserving LDS, grid, output shape, and global-load count. Current staging has only about 1.5% final-tile lane slack, so this is expected to be neutral or worse and is not a fix claim.

Low-agent execution is complete. The performance defect is not closed: the authoritative deep G5 cliff remains, K_ONLY is not promotable, and full-model promotion was not run. The next action is a separate strong-review scope for compiler allocation/lifetime diagnosis.
