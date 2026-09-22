# NV catch-llama fully measured ledger scope (2026-08-22)

Date: 2026-08-22  
Branch: `nvidia-bringup-20260731`  
Starting HEAD: `6570abc025514273faa100c66b979e531585a1e1`  
Hardware: NVIDIA GeForce RTX 5090, UUID `GPU-c800ade9-21ea-2e55-f75c-6d7a458fb186`  
Model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`, SHA-256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`  
Depth: d512  
Status: **execution scope for a new implementation-and-measurement agent.**

## 0. Directive

Catch the pinned llama.cpp oracle on Qwen3-8B-Q4_K_M at d512 on the RTX 5090, or produce a fully measured account of the remaining gap and the exact information needed to close it.

This scope is not another theory report. The executing agent may make narrowly targeted production kernel, route, runtime, or host-submission changes required by the work packages below. It must preserve model semantics, keep every experimental route default-off until promoted, and must not change the model architecture, quantized weights, prompt, tokenizer semantics, or oracle implementation to manufacture a win.

The final result must be fully **measured and ledgered**:

- every endpoint claim has a fresh-process reverse bracket;
- every correctness gate has retained token/logit hashes;
- every row change has a topology and device-time explanation;
- every promoted change is measured alone and in the cumulative stack;
- every final wall value is reconciled as `wall = resident_union + host_gap`;
- every artifact is retained with SHA-256;
- no projected ceiling is reported as recovered wall.

Read these two documents fully before acting:

1. `docs/task_workflow/output/nv-third-party-theory-audit-result-20260822.md`
2. `docs/task_workflow/evidence/nv-third-party-theory-audit-20260822/audit-summary.json`

Re-derive their arithmetic from the raw evidence. Treat them as the starting hypothesis, not as authority for the new session.

## 1. Exact success question

```text
Can a semantically identical tinygrad DEV=NV production decode route achieve
wall_us_per_token < the pinned llama.cpp oracle wall_us_per_token at d512,
in a same-session reverse bracket, and reproduce the win in a second session?
```

There are three allowed terminal verdicts:

1. `CAUGHT_LLAMA`: tinygrad is faster by the final acceptance rule in Section 16.
2. `PARITY_ONLY`: tinygrad and llama are statistically indistinguishable, but tinygrad does not clear the catch margin.
3. `NEED_MORE_INFO`: tinygrad has not caught llama, the remaining gap is reconciled to named measured rows plus explicitly named unmeasured mechanisms, and the report specifies the exact measurement or evidence required to re-attempt. This verdict keeps the attempt open; it is not a finding that llama cannot be caught.

“Promising,” “projected to win,” and “sum of ceilings exceeds the gap” are not terminal verdicts. `NEED_MORE_INFO` is an open, gated state, not a claim that the gap is unrecoverable.

## 2. Starting arithmetic envelope — planning input only

The third-party audit retained this cross-epoch planning ledger:

| planning term | us/token |
| --- | ---: |
| locked wall gap | 729.430 |
| ranked device-excess ceilings | 646.838 |
| locked host-gap ceiling | 100.648 |
| total paper ceiling | 747.486 |
| paper slack over gap | 18.056 |

The implications are severe:

```text
required total conversion = 729.430 / 747.486 = 97.6%
minimum device recovery if host reaches parity = 628.782 us
minimum host recovery if every listed device ceiling converts = 82.592 us
```

These are **inferred planning constraints**, not a performance forecast. The wall baseline is from tinygrad commit `07e9b2abe`; the row measurements are from current dirty HEAD `6570abc0`; the row ceilings are non-additive. Phase 0 must replace this envelope with a one-session ledger before any candidate is credited.

One dependency is already clear enough to control order: the measured flash launch/L2/timeline residual of 49.564 us is the first named information gap. If no body mechanism recovers it and the counter bridge reveals no omitted row, the paper envelope is about 31.5 us short even with full host recovery, and the correct verdict is `NEED_MORE_INFO` on that named residual, not a dead end. Likewise, perfect device-row parity without at least 82.6 us of host recovery cannot catch the locked oracle. Device and host work are both mandatory.

## 3. Non-negotiable definitions

Use these names in every JSON, table, and conclusion:

```text
mass[i]             = end[i] - start[i]
node_sum            = sum(mass[i])
resident_union      = measure(union of all resident kernel intervals)
resident_overlap    = node_sum - resident_union
span                = max(end) - min(start)             # never call this union
host_gap            = unprofiled_wall - resident_union
unprofiled_wall     = resident_union + host_gap

spin_sum            = integral (resident_count - useful_count) dt
useful_body         = integral useful_count dt
useful_overlap      = integral max(useful_count - 1, 0) dt
useful_union        = integral 1[useful_count > 0] dt
spin_only_union     = integral 1[resident_count > 0 and useful_count = 0] dt

resident_union      = useful_body - useful_overlap + spin_only_union
```

Never use `max(end)-min(start)` for interval union. Never publish
`resident_union = useful_body - useful_overlap` unless `spin_only_union` was measured as zero. Useful-body fields must be `null`/`UNMEASURED` unless per-kernel wait-exit timestamps are cross-clock calibrated and the corrected identity closes.

## 4. Claim labels

Every factual sentence and every result row in the final report must be labeled one of:

- `MEASURED`: directly reconstructed from a retained artifact or fresh run;
- `INFERRED`: a conclusion or ceiling computed from measured inputs;
- `UNMEASURED`: required quantity not obtained by the experiment.

Source code statements must cite `file:line`. Measurement statements must cite an artifact path and SHA-256. If a number crosses sessions, commits, backends, profiler domains, or binaries, label that fact beside the number.

## 5. Session and provenance contract

Before each GPU session, record:

- date/time/timezone;
- `git rev-parse HEAD`, branch, full `git status --short`, tracked diff SHA-256, and untracked analysis helpers used;
- Python executable and SHA-256;
- model path, size, and SHA-256;
- GPU name, UUID, driver, CUDA version, P-state, power limit, and clock-policy output;
- `nsys --version`, `ncu --version`, `nvcc --version`;
- every relevant environment variable, especially `DEV`, `PROFILE`, `HCQ_*`, `NV_*`, `JIT`, and graph/replay flags;
- exact command arrays, not paraphrased commands;
- llama repository HEAD, dirty state, tracked diff SHA, build command, binary path, and binary SHA-256.

The prior oracle label `ac4cddeb0` is insufficient because the llama tree was dirty. Pin the executable by SHA-256. Rebuild only if required; if rebuilt, the new binary becomes a new oracle and the baseline must be rerun.

Use `/tmp/gpu-bench.lock` for the entire lifetime of each measurement process. Do not overlap Nsight, compilation that uses the GPU, desktop workloads, or another benchmark. The consumer application-clock interface reports deprecated on this GPU, so control drift with reverse brackets and retain the clock/power samples rather than claiming a fixed frequency.

## 6. Statistical and correctness contract

### 6.1 Fresh processes and brackets

Every wall comparison uses fresh processes in this order:

```text
control A -> candidate -> control B
```

For an implementation-versus-oracle catch test, run both reverse orientations in the same session:

```text
llama A -> tinygrad -> llama B
tinygrad A -> llama -> tinygrad B
```

Use settled continuous windows, at least 32 timed tokens per repetition and at least five repetitions per arm unless memory/context limits require more conservative values. Record every accepted and rejected sample. Fix the outlier rule before seeing candidate results; do not discard a slow candidate sample under a rule not applied to both controls.

### 6.2 Promotion significance

The old `50 us` gate is too coarse for a ledger with only 18 us paper slack. Use both conditions:

1. candidate is faster than **each** bracketing control median; and
2. bracket-median gain is at least `max(10 us, 3 * control MAD)`.

Repeat any promotable result in a second fresh session. Report a bootstrap confidence interval over per-repetition medians as supporting evidence; do not substitute the interval for the two-control rule.

Retain positive results below the gate in the explanatory ledger, but do not stack them into production without a second confirmation. Retain neutral and negative results as closures.

### 6.3 Semantic gate

For every candidate:

- require identical tinygrad control/candidate token arrays and SHA-256 over at least 128 deterministic decode tokens across qualification and timing routes;
- require the sampled token to equal full-logit argmax at each qualification step;
- retain full-logit arrays for at least eight tokens, require finite values, and report bitwise equality, max absolute error, L2 error, and SHA-256;
- if bitwise equality is not expected for a legal arithmetic reassociation, require identical tokens plus the predeclared numeric bound; do not relax the bound after seeing the result;
- compare tinygrad and llama token streams using the same prompt/tokenizer semantics for the final catch gate.

No performance result survives a failed semantic gate.

## 7. Artifact layout

Create:

```text
docs/task_workflow/evidence/nv-catch-llama-ledger-20260822/
docs/task_workflow/output/nv-catch-llama-ledger-result-20260822.md
```

Required top-level machine-readable artifacts:

```text
00-provenance.json
01-baseline-wall-brackets.json
02-baseline-profile-ledger.json
03-counter-bridge-qualification.json
04-row-authority.json
05-individual-ab.json
06-cumulative-stack-ab.json
07-reverse-ablation.json
08-host-gap-ledger.json
09-final-oracle-brackets.json
10-corrected-wall-ledger.json
11-closed-theories.json
sha256.txt
```

Retain raw JSONL, stdout/stderr logs, `.nsys-rep`, exported SQLite, graph dumps, `.ncu-rep`, cubin/PTX/source, compiler logs, token arrays, and full-logit NPZ files underneath named phase directories. A summary without raw inputs is not evidence.

Generate `sha256.txt` only after the artifacts are final, then run `sha256sum -c` and retain the successful verification output.

## 8. Phase 0 — rebuild the same-session authority ledger

Do not edit production code before this phase closes.

### 8.1 Freeze the two endpoints

Measure the current tinygrad production route and pinned llama executable in the same locked session. Use identical model, d512 prompt semantics, deterministic decode, context capacity, and enough tokens to reach settled replay behavior.

Collect unprofiled wall in both reverse orientations from Section 6.1. Retain token SHA for tinygrad and, if the existing llama benchmark does not expose token IDs, build a measurement-only oracle harness that does. Do not change llama kernels or scheduling for the baseline.

### 8.2 Profile the two endpoints separately

Tinygrad profile requirements:

- full pre-split decode DAG;
- exact rendered program name, layer index, start/end, queue, graph group, launch geometry, and dependency edges;
- true interval union, node sum, overlap, span, and graph-group dead gaps;
- anchors Q, K, V, O, gate/up, down, flash score/combine, vocab, and all completion reductions.

Llama profile requirements:

- fresh PDL-on trace of the pinned oracle executable;
- a PDL-off diagnostic trace with identical 762-node topology, if the measurement seam remains available;
- exported SQLite and weighted logical DAG;
- true interval union rather than span;
- logical role mapping for normalization, quantization, Q/K/V/O/G/D, rope/store, flash, combine, activation, residual, and vocab.

Profiled spans are not wall. Quantify profiler tax with paired unprofiled runs, but never “correct” a profile by subtracting a global tax and then call it measured.

### 8.3 Close the new baseline ledger

Publish, for both endpoints and their delta:

| quantity | tinygrad | llama | delta | status |
| --- | ---: | ---: | ---: | --- |
| wall | | | | measured unprofiled |
| node_sum | | | | measured profile |
| resident_union | | | | measured profile |
| resident_overlap | | | | measured profile |
| host_gap | | | | inferred only from matched wall/union route signature |
| useful_body | | | | measured or UNMEASURED |
| useful_overlap | | | | measured or UNMEASURED |
| spin_only_union | | | | measured or UNMEASURED |

Required reconciliation:

```text
abs(node_sum - resident_overlap - resident_union) <= 0.001 us
abs(wall - resident_union - host_gap) <= declared cross-domain uncertainty
```

If route signatures, kernel census, graph groups, or token SHA differ between profiled and unprofiled arms, do not combine them. Fix the measurement seam first.

### 8.4 Refresh the catch constraint

Recompute every row ceiling in Section 10 from the new same-session profiles. Do not carry forward `729.430`, `646.838`, or `100.648` except as historical comparison. Publish:

```text
fresh_gap
fresh_device_union_gap
fresh_host_gap
fresh_sum_of_nonoverlapping_row_ceilings
fresh_required_conversion_fraction
fresh_slack_or_shortfall
```

If the refreshed, alternate-path-aware ceiling is smaller than the fresh gap, identify the missing mechanism before implementation. Do not start a stack that is mathematically incapable of catching the oracle.

## 9. Phase 1 — build the production-kernel counter bridge

The audit could not profile production `DEV=NV` kernels with Nsight Compute: the driver-side profiler gate returned error 3 and NCU saw no kernels. Full measurement requires a bridge before claiming a bandwidth or instruction mechanism.

Build a measurement-only harness that executes the **same compiled kernel image** through an NCU-visible CUDA/driver context:

1. capture/export the exact PTX or cubin produced for the production NV program;
2. retain its SHA-256 and production launch geometry;
3. load that exact image in a standalone harness without rewriting the kernel source;
4. reproduce production argument layout, shapes, strides, quant types, depth-dependent cache extent, and alignment;
5. validate output against the production kernel with SHA-256 and numeric error;
6. prove grid, block, dynamic shared memory, registers, local memory, and image hash match;
7. run NCU only after this equivalence gate passes.

If exact-image execution is impossible, a source-recompiled harness is labeled `SOURCE_EQUIVALENT_FALLBACK`, not production evidence. Record compiler/version/flags and SASS differences.

For each row collect at minimum:

- `gpu__time_duration.sum`;
- `dram__bytes.sum` and read/write split where available;
- L2 sectors/bytes and hit rate;
- achieved DRAM throughput;
- registers per thread, local load/store sectors, spills;
- achieved occupancy and active warps;
- executed instruction counts by major class;
- branch/divergence indicators;
- launch grid/block/shared-memory values.

NCU replay duration is a counter-domain body measurement, not wall and not original-timeline overlap. Retain warm-cache and production-order/cold-cache variants separately.

## 10. Phase 2 — refresh the row authority

Use the common semantic boundaries below. Do not assign every `r_*` to reduce or every `E_*` to residual. Classify from model role metadata, dataflow, exact adjacency, and program implementation.

For every row publish:

```text
tinygrad node mass
llama PDL-off body mass
tinygrad resident-union contribution
critical-path sensitivity with alternate-path takeover
real DRAM bytes on both sides
effective bandwidth in the same counter domain
launch-count and cold-L2 residual
legal mechanism ceiling
wall A/B result, initially null
```

Starting rows to refresh, in execution order:

| order | semantic row | current tinygrad spelling | historical device ceiling us | main question |
| ---: | --- | --- | ---: | --- |
| 1 | gate/up | `q4k_g3_lanemap_gemv_w1w3fused16_12288_4096` | 101.326 | Is the body slower from real bytes, bandwidth, instruction mix, or occupancy? |
| 2 | Q | Q `q4k_*_4096_4096` plus `r_32_32_4_4_*` | 84.412 | Can completion reduction be removed or made direct without losing the fast main body? |
| 3 | O | `q4k_g3_lanemap_gemv_epi_resadd_4096_4096` | 75.231 | Why does the already-fused residual epilogue trail llama's complete O body? |
| 4 | down | q4/q6 `*_4096_12288_epi_ffnresadd` | 74.551 | Is the deficit in dot work, activation prelude, residual epilogue, or quant-type strata? |
| 5 | vocab | `q6k_gen_coop_151936_4096_inkernel` plus reduction tail | 67.612 | Can top-1 completion avoid the multi-kernel reduction tail? |
| 6 | flash combine | `flash_fused_gmax_combine_f16_32_128` | 66.943 | Is combine body necessary at d512, and what is body versus launch/cold-L2? |
| 7 | flash score | `flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128` | 64.540 | Can the 49.564-us installed residual move after isolated body is accounted? |
| 8 | K | K projection plus `r_8_32_4_4_*` | 58.368 | Which Q4/Q6/layer strata require partial completion? |
| 9 | V | V projection plus `r_8_32_4_4_*` | 53.855 | Which direct/partial route is actually slower than oracle? |
| H | host | graph-group/replay submission handoff | 100.648 | Which CPU/API interval creates wall-minus-union, and can at least the refreshed minimum be removed? |

Historical ceilings are navigation only. Replace them before implementing.

## 11. Phase 3 — device work packages

Each package follows this fixed loop:

```text
measure incumbent -> isolate mechanism -> build one-variable candidate
-> semantic qualification -> exact topology diff -> isolated A/B/A
-> full-token control/candidate/control wall -> promote or close
-> add promoted candidate to cumulative stack -> remeasure stack
```

Do not run a broad parameter search directly against full-token wall. Use an isolated body gate to select at most the top two variants, then pay the full model qualification cost.

### WP-D1. Gate/up — first decision point

Required measurements:

- all 36 per-layer durations, quant type, weight bytes, grid/block, registers, spills, DRAM bytes, L2 behavior, and bandwidth;
- matching llama G rows with PDL disabled for body residence;
- production-order cold-cache versus repeatedly isolated hot-cache timing;
- split body cost between activation quant/provider, W1/W3 packed-weight reads, dot accumulation, SiLU/multiply, and output store where the implementation permits attribution;
- prior `nv-w1w3-norm-once-20260822` and `nv-w1w3-norm-smem-20260822` results, so an unchanged losing construction is not repeated.

Candidate families may include load/vector geometry, warp/CTA mapping, dual-output accumulation scheduling, q8 activation reuse, register-pressure reduction, or a legally equivalent fused W1/W3 organization. Change one mechanism per microgate.

Promotion requirements:

- exact semantic gate;
- isolated body improvement explains at least 50% of the refreshed G-row deficit;
- no new local-memory spill or extra DRAM traffic unless wall proves it beneficial;
- full-token wall passes Section 6.2 twice.

If the isolated exact-image body reaches llama parity but full wall does not move, stop kernel geometry work and measure launch/L2/timeline placement before another variant.

### WP-D2. Q projection plus completion

Stratify the 36 layers by the direct G3 route versus warp-cooperative partial route. The comparable Q body includes `r_32_32_4_4_*` completion when present; never compare only the main partial kernel to llama's complete Q.

Test, in order:

1. whether a direct route can replace partial+completion for the affected layers;
2. whether the final reduction can be completed in the producing kernel without increasing main-body time more than it saves;
3. whether completion launch/cold-L2, rather than arithmetic, explains the installed deficit;
4. whether per-layer route selection beats a global route.

Preserve q/k normalization and rope dependencies. A candidate that moves work into an uncounted provider is not a win; topology accounting follows the data.

### WP-D3. O and down epilogue rows

Residual adds are already absorbed. Preserve the exact epilogue semantics and compare complete rows.

For O, isolate base dot versus residual epilogue and store type. For down, stratify Q4/Q6 layers and isolate dot, fused SiLU/up prelude if present, residual add, and final store. Measure real bytes; do not infer that epilogues are free from their absence as separate kernels.

Test common substrate changes first. If one lanemap/accumulator/load change improves G, O, and down, qualify it per shape and then as one shared production change. Retain per-row contribution; do not count the same shared change three times.

### WP-D4. Vocab top-1 chain

The semantic boundary is the entire final chain, historically about 370.848 us tinygrad versus 303.236 us oracle—not only `q6k_gen_coop`.

Measure main GEMV, auxiliary elementwise work, and each reduction stage. Test a legal single-kernel or shorter-tail top-1 completion only if it preserves deterministic tie-breaking and sampled/full-logit agreement. Retain full logits; token-only equality is insufficient for this package.

### WP-D5. Flash combine and score

Do not reopen S=4/S=2 coarse split without new current-HEAD evidence; those shapes were previously slower. Do not search score tiles first: fresh isolated body explains only about 14.976 us of the historical 64.540-us installed score gap.

Required decomposition:

- repeated hot isolated body;
- cold first launch after the preceding Q/K/V/rope sequence;
- launch-to-start gap;
- score-to-combine gap;
- score and combine real bytes/L2 hit rate;
- exact number of chunks and whether combine is semantically required at d512;
- standalone score+combine versus legally fused completion;
- installed full-token row versus body prediction.

The residual bucket must remain named `launch/L2/timeline` until one-variable experiments separate it. A subtraction is not a mechanism.

### WP-D6. K and V

Map every layer by quant type and route: direct, warp-cooperative partial, and its exact `r_8_32_4_4_*` completion. Compare complete K and V semantic bodies to oracle. Because K/V may be off the weighted Q/O/G/D spine, calculate alternate-path wall sensitivity before implementation.

Promote only if the candidate moves full wall or enlarges useful overlap without increasing the critical path. Node-mass savings off path do not qualify by themselves.

## 12. Phase 4 — host-gap package (mandatory, start instrumentation early)

The prior submit-ahead candidate was wall-neutral. Do not repeat it unchanged.

### 12.1 Measure the host gap causally

For the fresh authority route, retain a CPU/API/GPU timeline that attributes wall-minus-union to:

- Python/generator work before submission;
- JIT dispatch and variable binding;
- graph-group selection and five replay submissions;
- cross-group synchronization;
- driver enqueue latency;
- final device synchronization and token materialization;
- idle gaps between graph groups;
- any profiler or logging effect.

The categories must reconcile to host gap within a declared clock-domain uncertainty. “Five replays versus one” is not a cause until the trace shows which four handoffs are exposed on wall.

### 12.2 Candidate order

Test only mechanisms supported by the trace:

1. precompute/submit the next graph group while the current group is resident, without forming one oversized GPU replay;
2. persistent or reusable submission descriptors that remove Python/JIT work but retain the current GPU group boundaries;
3. safe adjacent-group handoff coalescing where dependency and variable-binding contracts permit it;
4. token materialization or sync deferral that does not change generator semantics;
5. a bounded multi-token handoff only if token feedback dependencies remain correct and token latency is still the reported unit.

Every host candidate must also collect a device profile. Reject it if host time falls but resident union grows by an equal or larger amount, as happened with earlier one-large-replay designs.

### 12.3 Host gate

After refreshing the ledger, compute the minimum host recovery required even under perfect device conversion. If no traced host mechanism has at least that ceiling, do not stack further; name the missing device row or measurement that would change the envelope and mark the host package `NEED_MORE_INFO` before continuing.

## 13. Phase 5 — cumulative stack and interaction accounting

Individual A/B gains are not additive. Maintain a cumulative stack:

```text
S0 = fresh production control
S1 = S0 + first promoted change
S2 = S1 + second promoted change
...
Sn = final candidate
```

After every addition, run `S(n-1) / Sn / S(n-1)` with fresh processes and the same token SHA. Record:

```text
individual_gain_i
incremental_stack_gain_i
interaction_i = incremental_stack_gain_i - individual_gain_i
cumulative_gain
remaining_gap_to_same_session_llama
```

If a promoted change turns neutral or negative in the stack, remove it unless a measured positive interaction elsewhere outweighs the loss.

At the final stack, perform reverse ablation: remove each promoted change one at a time from `Sn`, in reverse order and then in at least one order-independent check for shared GEMV substrate changes. This prevents crediting one wall reduction to multiple row ceilings.

Re-profile the cumulative stack after every material topology change and at minimum after each two promotions. Refresh node sum, union, overlap, host gap, row mass, and critical-path sensitivity. Never carry the original profile ledger through a changed kernel topology.

## 14. Stop/reopen rules

Close a candidate family when any of these holds:

- semantic gate fails and there is no exact correction that preserves the mechanism;
- exact-image isolated body is not faster in A/B/A;
- full-token candidate is not faster than both controls in two attempts;
- a candidate reduces node mass but not wall and profiling shows the row is off path or replaced by alternate-path takeover;
- real bytes/instructions prove the proposed mechanism cannot cover the refreshed deficit;
- the same construction was already measured negative and no changed premise is demonstrated.

Reopen a closed historical theory only if the agent identifies the exact changed fact: commit, route, geometry, dependency edge, quant stratum, profiler defect, or prior classification error.

Never stack `WALL_NEUTRAL` or `NO_GO_WALL` arms merely because the paper ledger needs their ceiling.

## 15. Required final ledgers

### 15.1 Endpoint wall ledger

For fresh control, final candidate, and llama:

| status | endpoint | wall | node_sum | resident union | resident overlap | host gap | useful_body | useful_overlap | spin-only union |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

Every unmeasured field is explicitly `UNMEASURED`, never zero.

### 15.2 Row recovery ledger

| status | row | fresh oracle deficit | isolated body gain | individual wall gain | stack incremental gain | final ablation value | remaining ceiling |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |

The sum of `final ablation value` is an interaction-sensitive explanation, not automatically the cumulative gain. Publish the interaction residual and explain it.

### 15.3 Gap closure

```text
fresh_gap
- measured cumulative device/row wall recovery
- measured host wall recovery
- measured interactions
= final_remaining_gap
```

The right side must equal the direct final tinygrad-versus-llama wall bracket within the declared uncertainty. Any residual larger than 10 us remains open; do not allocate it to launch, overlap, bandwidth, or noise without evidence.

### 15.4 Prior-evidence corrections

List every historical value invalidated by the fresh authority session, including baseline drift, row remapping, body/launch attribution, counter-byte differences, and closed mechanisms. Preserve both old and new artifact citations.

## 16. Final catch acceptance

`CAUGHT_LLAMA` requires all of:

1. same model SHA, d512 prompt semantics, deterministic generation, and matching token stream;
2. tinygrad final candidate uses the production `DEV=NV` route, not `DEV=CUDA` or an isolated harness;
3. both reverse implementation/oracle bracket orientations show tinygrad faster than each bracketing llama median;
4. tinygrad advantage is at least `max(10 us, 3 * combined control MAD)`;
5. the result reproduces in a second fresh locked session;
6. no profiler is active during the endpoint wall runs;
7. final topology, route gates, binary/model/commit/diff hashes, raw samples, and token hashes are retained;
8. the corrected wall ledger closes and no measured regression is hidden outside the reported row set.

If tinygrad is within the uncertainty band but does not clear condition 4, verdict is `PARITY_ONLY`.

If neither `CAUGHT_LLAMA` nor `PARITY_ONLY` holds, and the remaining gap is reconciled to named measured rows plus named unmeasured mechanisms, verdict is `NEED_MORE_INFO` with the exact next measurement listed.

## 17. First ten actions for the executing agent

1. Read and independently check the third-party report and summary.
2. Inventory current tinygrad and llama dirty state; freeze binary/model/tool hashes.
3. Create the evidence directory and provenance record.
4. Run same-session unprofiled tinygrad/llama reverse brackets and token gates.
5. Capture fresh tinygrad and llama weighted profiles; compute true interval union.
6. Publish the refreshed baseline and catch constraint before code changes.
7. Build and qualify the exact-image NCU counter bridge.
8. Refresh all nine device row ceilings and the host minimum.
9. Execute WP-D1 gate/up; stop and report its individual wall verdict before proceeding.
10. Start host causal tracing while continuing the remaining rows in the prescribed order.

After action 9, the agent must issue an explicit checkpoint:

```text
GATE_UP_WALL_PASS
GATE_UP_WALL_NEUTRAL
GATE_UP_NO_GO_WALL
```

with exact device-body, wall, token-SHA, topology, and counter deltas. Gate/up is the first decision point for whether the current GEMV substrate can convert cross-implementation body deficits into wall.

## 18. Deliverable

Produce one findings-first report at:

`docs/task_workflow/output/nv-catch-llama-ledger-result-20260822.md`

It must contain, in this order:

1. terminal verdict and exact final tinygrad/llama wall delta;
2. severity-ordered findings tied to code lines or artifact hashes;
3. corrected same-session endpoint ledger;
4. individual work-package verdicts with exact wall deltas;
5. cumulative stack and reverse-ablation ledger;
6. body/bytes/bandwidth/launch decomposition for every tested row;
7. host-gap causal decomposition and result;
8. every wrong, stale, overstated, or unmeasured prior claim;
9. remaining gap and the exact next measurement needed if llama was not caught;
10. complete raw-evidence index and verified SHA-256 manifest.

The report must answer, without projections:

```text
Did tinygrad catch llama?
Which exact rows supplied the measured wall recovery?
How much did each contribute after stack interactions?
How much device union and host gap remain?
What is still unmeasured?
What exact measurement would unlock the next attempt?
```

The task is complete only when those questions are answered from a same-session, semantics-gated, hash-retained ledger.
