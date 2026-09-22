# NV first-principles installed-island sequential scope (2026-08-22)

Date: 2026-08-22  
Repo: `/home/ubuntu/tinygrad-arkey`  
Branch: `nvidia-bringup-20260731`  
Scope-authoring HEAD: `6570abc025514273faa100c66b979e531585a1e1`  
Python: `/home/ubuntu/tinygrad-arkey/.venv/bin/python`  
Backend: `DEV=NV`  
GPU: RTX 5090 / sm_120  
Model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`  
Workload: single-token decode, fixed context depth 512  
Status: **sequential measurement and causal-design scope; no production-code
authorization.**

## 0. Mission and boundary

The mission is to determine, in sequence, which installed producer-to-consumer
islands can recover enough end-to-end wall time to reach 240 tok/s without
specializing behavior to a model identity. The selected mechanisms may be
specialized to target facts, quant format, semantic role, and tensor shape.

The first-principles rule is:

```text
Do not begin with a kernel rewrite or a fusion idea.
Begin with endpoint wall, isolate an installed dependency island, decompose
its cost, then select the mechanism implied by the measured dominant term.
```

This scope authorizes:

- read-only inspection of production, renderer, scheduler, runtime, and model
  code;
- GPU measurements in fresh processes;
- new or modified measurement tools under `extra/llm_research/**`;
- reports under `docs/task_workflow/output/**`;
- raw evidence under `docs/task_workflow/evidence/**`;
- build-ready follow-on scopes under `docs/task_workflow/input/**`.

This scope does **not** authorize edits to:

- `tinygrad/runtime/**`;
- `tinygrad/renderer/**`;
- `tinygrad/codegen/**`;
- `tinygrad/schedule/**`;
- `tinygrad/llm/**` production paths;
- the model implementation, route policy, generated promotion records, or
  scheduler/runtime behavior.

If a phase identifies a buildable candidate that requires one of those edits,
the agent must write a separate implementation scope and stop that candidate.
Research-only construction is allowed only when it does not change production
source and the result is explicitly labelled `RESEARCH_CONSTRUCTION`.

No earlier conclusion is trusted merely because it appears in a report. Every
claim in the final result must be labelled `MEASURED`, `INFERRED`, or
`UNMEASURED`.

## 1. Exact objective and recovery budget

The latest endpoint supplied to this scope is:

```text
tinygrad wall       4747.5 us/token
240 tok/s target    4166.666667 us/token
required recovery   580.833333 us/token
```

The first fresh reverse bracket in Phase 1 replaces `4747.5` as the working
control if it differs. The target remains `1e6 / 240`.

Maintain two ledgers, never one:

```text
booked_recovery = sum(measured reverse-bracket wall improvements)
remaining_to_240 = current_control_wall - 4166.666667 - booked_recovery

projected_ceiling = simulated or attribution-only upper bound
```

A projected ceiling never enters `booked_recovery`. A microbenchmark result,
node deletion, body delta, command-interval delta, or critical-path simulation
is not a booked result.

## 2. The measurement model

### 2.1 Terms that must remain distinct

For a complete token:

```text
wall       = endpoint elapsed time per accepted decode token
node_sum   = sum of per-node measured intervals
union      = interval union of measured device-command intervals
overlap    = node_sum - union
host_gap   = wall - union
```

The word `host_gap` is retained for compatibility but must be annotated: when
one backend's interval excludes an inter-kernel device-front-end gap, that gap
can land in `host_gap` even though it is not host execution.

For one tinygrad kernel or same-binary kernel family:

```text
B = exact-cubin pure GPU body duration
C = clean chained HCQ duration
P = production-installed HCQ command interval

D = C - B       # clean dispatch/front-end component
R = P - C       # production-conditioned residual

P = B + D + R
```

`R` is named only `production-conditioned residual` until a counter or
controlled predecessor experiment separates cache state, dependency wait,
memory visibility, QMD scheduling, placement, or another mechanism.

For an island containing multiple nodes:

```text
installed_span = consumer_done - latest_required_producer_ready
installed_union = interval_union(nodes in the island boundary)
weighted_path = longest legal dependency path inside the boundary
```

Do not sum `P` across overlapping nodes and call it wall leverage. Recompute
the weighted path and union after every simulated change.

### 2.2 Like-for-like comparison rule

The frozen role census compares tinygrad HCQ command intervals against llama
nsys kernel-body durations. It remains a localization artifact, not a causal
body ledger.

For every row selected below, publish both:

```text
tinygrad installed command interval vs llama installed island span
tinygrad pure body                 vs llama pure body
```

If an installed llama interval cannot be reconstructed, label it
`UNMEASURED`; do not substitute kernel-body duration silently.

### 2.3 Cause-selection rule

After decomposition, assign exactly one triage verdict:

| verdict | evidence requirement | next mechanism |
| --- | --- | --- |
| `BODY_DOMINANT` | exact tinygrad body accounts for at least 60% of the positive installed delta and stays slow under matched predecessor conditions | target-derived kernel topology/codegen scope |
| `INSTALL_DOMINANT` | production-conditioned residual accounts for at least 60% | producer/consumer handoff, placement, boundary, or fusion scope |
| `DISPATCH_DOMINANT` | clean dispatch component accounts for at least 60% | launch elimination or command-path scope |
| `MIXED` | no component reaches 60% | whole-island candidate with separate predicted terms |
| `BODY_PARITY` | exact body is within 10% of the oracle body and the installed delta is elsewhere | prohibit arithmetic rewrite absent new evidence |
| `UNMEASURED` | identity cannot close or observables are not comparable | stop and repair measurement |

The percentages are routing thresholds, not physical laws. Emit the raw values
so a later reviewer can change the threshold without repeating the run.

## 3. Locked evidence to verify, not blindly repeat

At Phase 0 the agent must hash-check and reconstruct these records:

1. Full-token role census:
   `docs/task_workflow/evidence/nv-full-token-role-census-20260822/role-census.json`
   expected SHA-256
   `0326f0d21e10059a92196a439431f5bd58fb04353a6b20d972e94b3cece494cf`.
2. Q/K exact-cubin result:
   `docs/task_workflow/evidence/nv-qk-head-norm-predecessor-20260822/`.
3. HCQ dispatch-slope result:
   `docs/task_workflow/evidence/nv-hcq-dispatch-slope-20260822/`.
4. The retained tinygrad and llama captures indexed by
   `docs/task_workflow/output/nv-full-token-role-census-result-20260822.md`.
5. The corrected S1 audit:
   `docs/task_workflow/output/nv-s1-blackbox-reaudit-result-20260822.md`.

Known calibrated facts, subject to successful reconstruction:

```text
Q exact body                         ~1.190 us
K exact body                         ~1.196 us
clean chained Q/K HCQ                ~1.698 us/kernel
faithful per-kernel HCQ profile      ~1.696 us/kernel
plain no-op HCQ floor                ~0.649 us/kernel
production Q/K command interval      ~2.5 us/kernel median
```

The accepted conclusion is limited to:

```text
the Q/K compiler-body hypothesis is rejected;
the global ~1.4 us fixed-HCQ-tax hypothesis is rejected;
the remaining Q/K installation mechanism is unmeasured.
```

Do not restate the `~0.8 us` Q/K residual as cache or serialization without a
new adjudicating measurement.

## 4. Global measurement protocol

Every GPU phase obeys all of the following.

### 4.1 Process and machine discipline

- Acquire `flock /tmp/gpu-bench.lock` for every GPU measurement.
- Use a fresh process for every accepted timing sample.
- Record the exact clock-lock command and `nvidia-smi -q` clock/power state
  before and after each bracket. Fail closed if the established clock lock
  cannot be applied.
- Record temperature, power limit, graphics clock, memory clock, driver, and
  active compute processes.
- Reject samples containing an unrelated GPU process, clock-state transition,
  thermal throttle, ECC/Xid event, compile, or first-use allocation in the
  timing region.
- Use the same model file, prompt construction, depth, generated-token count,
  and sampler settings across arms.
- Capture commit, branch, `git status --short`, complete environment, and
  command line for every process.

### 4.2 Wall timing

- Use at least three independent `control / candidate / control` brackets.
- Each arm is a fresh process.
- Exclude declared warmup/capture tokens identically in all arms.
- Report every accepted and rejected sample with a rejection reason.
- Primary delta is the median candidate wall minus the median of the
  bracketing-control midpoint values.
- Also report tok/s, but use microseconds/token for arithmetic.
- A candidate is `WALL_PASS` only when the wall delta is negative, token gates
  pass, the controls are stable, and no cost moved outside the measured token.
- The existing `+50 us/token` promotion bar remains a policy threshold. A
  smaller measured win is retained as `WALL_POSITIVE_BELOW_BAR`, never silently
  promoted.

### 4.3 Correctness

- Persist the generated-token stream for every arm and hash it with SHA-256.
- Require identical token SHA for exact candidates.
- When a governed non-bit-exact candidate is explicitly scoped later, retain
  full-logit SHA plus relative L2, max absolute error, argmax, top-k membership,
  and top-k ordering. This scope does not itself authorize a relaxed gate.
- Pin the first generated token and accepted token count.

### 4.4 Raw evidence

Every phase writes to:

```text
docs/task_workflow/evidence/nv-installed-islands-20260822/<phase>/
```

Retain:

- raw stdout/stderr;
- timing samples, not only medians;
- cubins and cubin hashes where size permits;
- nsys `.nsys-rep` and `.sqlite` files;
- NCU raw reports for selected counter probes;
- HCQ profile JSONL;
- DAG/census exports;
- environment and command manifests;
- token streams and hashes;
- one `sha256.txt` covering every retained file except itself.

Large immutable GGUF or weight buffers are not duplicated. Instead retain the
GGUF SHA, tensor identity, byte offset or loader identity, quant type, shape,
and fixture-generation command.

## 5. Common artifact schema

Create one machine-readable file per measured semantic row using this minimum
shape:

```json
{
  "schema": "tinygrad.nv_installed_island.v1",
  "island_id": "...",
  "semantic_role": "...",
  "commit": "...",
  "model_sha256": "...",
  "depth": 512,
  "cardinality": {"tinygrad": 0, "llama": 0},
  "production": {
    "tiny_command_sum_us": 0.0,
    "tiny_command_median_us": 0.0,
    "tiny_installed_span_us": 0.0,
    "llama_body_sum_us": 0.0,
    "llama_installed_span_us": null
  },
  "exact_binary": {
    "name": "...",
    "sha256": "...",
    "grid": [1, 1, 1],
    "block": [1, 1, 1]
  },
  "clean_hcq": {
    "slope_us": 0.0,
    "intercept_us": 0.0,
    "r2": 0.0,
    "profiled_median_us": 0.0
  },
  "pure_body": {
    "tiny_us": 0.0,
    "llama_us": 0.0,
    "method": "nsys exact cubin"
  },
  "conditions": {
    "hot_us": 0.0,
    "fill_us": 0.0,
    "exact_predecessor_us": null,
    "flush_us": 0.0
  },
  "decomposition": {
    "body_us": 0.0,
    "clean_dispatch_us": 0.0,
    "production_residual_us": 0.0,
    "identity_residual_us": 0.0
  },
  "wall_sensitivity": {
    "zero_cost_ceiling_us": 0.0,
    "legal_mechanism_ceiling_us": null,
    "alternate_path": "..."
  },
  "verdict": "BODY_DOMINANT|INSTALL_DOMINANT|DISPATCH_DOMINANT|MIXED|BODY_PARITY|UNMEASURED"
}
```

`identity_residual_us` must be zero within timestamp resolution. If it is not,
the phase verdict is `UNMEASURED`.

## 6. Sequential execution plan

Only one phase may be `IN_PROGRESS`. Later phases may not reinterpret an open
earlier identity as a fact.

### Phase 0 — provenance and no-change gate

#### Work

1. Record branch, HEAD, dirty state, Python, driver, GPU facts, model SHA, and
   all relevant environment variables.
2. Hash-check every artifact in Section 3 from its intended working directory.
3. Regenerate the role census from its script and require byte-identical JSON.
4. Recompute the HCQ slope OLS values and R-squared from raw points rather than
   trusting the reported fit.
5. Record the initial list of modified files. No pre-existing user change may
   be overwritten or attributed to this scope.

#### Pass gate

```text
role census byte-identical
Q/K cubin hashes match
HCQ slope hashes match
plain no-op slope ~0.649 us and Q/K slope ~1.698 us reconstruct
no unauthorized source path changed
```

#### Stop

Any hash, cardinality, or commit mismatch is `BLOCKED_PROVENANCE`; repair the
record before GPU work.

### Phase 1 — fresh endpoint and profiler-tax bracket

#### Work

1. Run a fresh unprofiled tinygrad endpoint control.
2. Run the same workload with the minimum profile configuration required for
   the installed timeline.
3. Run the unprofiled control again.
4. Run the pinned llama endpoint and PDL-off trace in the same locked session.
5. Persist token SHA for every arm.

#### Required outputs

- fresh tinygrad and llama wall medians;
- `PROFILE=0` versus `PROFILE=1` endpoint delta;
- full node counts, command union, node sum, overlap, and wall;
- a statement of which profiler observables include dispatch/front-end time;
- updated `remaining_to_240`.

#### Pass gate

The unprofiled controls must bracket stably, token SHA must match, and the
profiled timeline must preserve route/cardinality. If profiler tax is
material, it must be measured and never subtracted from individual rows by an
unverified uniform factor.

### Phase 2 — common island manifest and matched boundaries

#### Work

Create a semantic manifest for every layer and the tail. At minimum it must
name these boundaries:

```text
I_Q:     Q projection partials -> completion/norm -> Q rope complete
I_K:     K projection partials -> completion/norm -> K rope/store complete
I_V:     V projection -> V handoff/store ready
I_ATTN:  Q/K/V ready -> flash score -> combine -> O input ready
I_O:     O input ready -> O projection/residual complete
I_FFN:   FFN norm ready -> gate/up -> activation -> down/residual complete
I_TAIL:  final norm -> vocab -> sampler feedback ready
```

For each tinygrad and llama node, emit:

- layer;
- semantic role;
- island membership;
- exact producer and consumer edge;
- fused-into annotation where applicable;
- command/body observable type;
- queue/stream;
- kernel/cubin identity;
- physical and semantic cardinality.

Confirm the positional mapping of the current `rope + K/V store` bucket. The
three `E_*` families currently assigned by position may not be used for a
candidate until this confirmation passes.

#### Pass gate

Every node is assigned exactly once, all island boundaries resolve, and the
full-token node sums close with zero residual. Cardinality differences remain
visible rather than averaged away.

### Phase 3 — Q/K installed-island completion

Q/K is first because the exact body and clean HCQ calibration already exist.
Do not repeat tree-reduction or compiler-codegen work unless this phase
invalidates the retained cubin result.

#### Work

1. Reconstruct per-layer production `P` distributions for Q and K, not only
   means. Report median, p10, p90, min, max, and layer correlation.
2. Re-run exact predecessor conditions:
   - hot repeated target;
   - simple fill producer;
   - faithful captured projection-completion predecessor where constructible;
   - L2 flush as a diagnostic only.
3. Attach L2/DRAM and launch/wait-exit counters to a bounded representative
   layer set: early, shared-Q8 lease, precision-boundary, and tail layers.
4. Split the production-conditioned residual among only measured mechanisms.
   Any unsplit remainder stays `UNMEASURED_RESIDUAL`.
5. Compute wall sensitivity for these legal launch-eliminating shapes:
   - Q completion + RMSNorm + rope;
   - K completion + RMSNorm + rope/store;
   - one combined Q/K completion-support kernel after both partial producers.
6. Account for the body transferred into the surviving kernel. The legal
   ceiling is not the deleted command interval.

#### Required verdicts

```text
Q body: BODY_PARITY or reopened with evidence
K body: BODY_PARITY or reopened with evidence
Q installation mechanism: named or UNMEASURED
K installation mechanism: named or UNMEASURED
best legal island construction: one, ranked, with alternate-path ceiling
```

#### Candidate gate

If a research-only candidate already exists without production edits, run its
token and wall bracket. Otherwise write
`nv-qk-completion-norm-rope-installation-scope-<date>.md` and stop the build.

Do not book the historical `~101 us` until a wall bracket measures it.

### Phase 4 — O projection, the simplest unresolved projection row

O is next because it has 36 semantically matched calls, no mixed physical
completion count, and a measured census attribution delta of `+75.23 us`.

#### Work

1. Enumerate every O kernel spelling and layer; prove which geometry and
   epilogue route each uses.
2. Capture exact production cubins, launch geometry, register count, shared
   memory, and arguments.
3. Measure `B`, `C`, and `P` for each spelling.
4. Replay under:
   - hot O input;
   - simple fill;
   - exact flash-combine predecessor where constructible;
   - residual-input hot/cold controls.
5. Collect real DRAM bytes, L2 sectors/hit rate, issue stalls, achieved
   occupancy, and memory-pipe activity for representative layers.
6. Compare like-for-like with llama O body and installed combine-to-O span.
7. Simulate separately:
   - body parity only;
   - removal of combine-to-O production residual only;
   - SM120-native Q4 topology with current residual epilogue preserved;
   - combine-output layout absorbed without changing O arithmetic.

#### Decision

- `BODY_DOMINANT`: write a generic sm_120 Q4 `attn_qo` topology-search scope.
- `INSTALL_DOMINANT`: write a flash-combine-to-O typed-boundary/handoff scope.
- `MIXED`: write one island scope with independent body and boundary arms.
- zero-cost legal ceiling below 20 us/token: demote O and proceed.

No AMD launch geometry or hardcoded model dimension may be copied as a selected
answer. AMD topology is a candidate axis only.

### Phase 5 — gate/up and down installed FFN island

The frozen attribution ceilings are `+101.33 us` for gate/up and `+74.55 us`
for down. These rows overlap causally inside one FFN island and must not be
summed as wall recovery without path recomputation.

#### Work

1. Separate every Q4 and Q6 spelling, fused gate/up variant, fp16-store
   variant, four-warp down variant, and residual epilogue.
2. Capture `B`, `C`, and `P` for gate/up and down by spelling.
3. Preserve the current shared/provider advantages and current promoted
   epilogues in all controls.
4. Run predecessor conditions:
   - norm/provider -> gate/up;
   - gate/up output -> down;
   - down -> residual/next norm.
5. Collect real bytes and issue/occupancy counters.
6. Reconcile the known result that the quad gate/up topology is faster
   standalone in one configuration but slower in-loop. Name the counter or
   predecessor condition that reproduces the reversal; otherwise label the
   mechanism unmeasured.
7. Compute island sensitivities for:
   - faster gate/up body only;
   - faster down body only;
   - elimination of activation handoff;
   - elimination of down residual/next-norm handoff;
   - a role-conditioned SM120 topology retaining exact output contracts.

#### Closed candidates

Do not repeat without a changed measured premise:

- quad-u128 gate/up as previously composed;
- separate Q8 provider for single-consumer FFN down;
- producer-folded DP4A arm that measured wall-negative;
- norm arithmetic inserted into the bandwidth/issue-limited GEMV epilogue.

#### Decision

Write at most one gate/up scope and one down/boundary scope. Rank them by legal
wall sensitivity, not raw node mass.

### Phase 6 — flash score, combine, and the attention handoff

The frozen attribution rows are `+64.54 us` score and `+66.94 us` combine.
Earlier coarse splits are closed, but the installed-body decomposition is not.

#### Work

1. Capture exact score and combine cubins for all layer/depth spellings.
2. Measure pure body, clean HCQ, production interval, and the complete
   Q/K/V-ready-to-O-input island span.
3. Collect per-kernel:
   - real DRAM bytes;
   - L2 bytes and hit rate;
   - shared-memory transactions and barrier stalls;
   - register count and occupancy;
   - issue-active and memory-pipe utilization;
   - launch/wait-exit timestamps.
4. Reconcile the previous isolated score timing with the installed row. Do not
   assume either launch or body dominates.
5. Attribute combine separately from its fp16 output absorption and the O
   consumer handoff.
6. Simulate and rank:
   - score-body parity;
   - combine-body parity;
   - score->combine boundary removal;
   - combine->O boundary removal;
   - one attention support program retaining online max/sum stability;
   - register-Q/narrow-group topology with no coarse split assumption.

#### Closed candidates

- coarse split `S=4` and `S=2` as already measured;
- geometry-only variants that fail token SHA;
- overlap projections based on llama PDL node_sum shadow;
- a projected single-stage ceiling without an installed wall arm.

#### Decision

Choose `BODY_DOMINANT`, `INSTALL_DOMINANT`, or `MIXED` for score and combine
separately, then produce one non-double-counted island ranking.

### Phase 7 — mixed Q and K/V projection families

These rows have mixed physical cardinality and promoted shared-Q8/four-warp
leases. They are deferred until the simple O/FFN/flash decompositions validate
the measurement method.

#### Work

1. Partition by exact route and block set:
   - ordinary Q4 G3;
   - shared-Q8 cooperative Q4;
   - Q6 direct V;
   - Q4/Q6 K variants;
   - completion kernels;
   - precision-boundary and tail blocks.
2. Emit counts and sums for each partition. Never use one per-call average
   across differing kernels.
3. Measure `B`, `C`, `P`, counters, and predecessor conditions for every
   partition contributing at least 10 us/token or 10% of its semantic row.
4. Preserve the activation-quant provider advantage when evaluating an
   alternative projection route.
5. Identify whether the blocks outside the shared-Q8 lease are body-limited or
   installation-limited.
6. Compute legal ceilings for generic target/quant/role/shape dispatch rules;
   model name or block-index-only promotion is disallowed as the final design.

#### Decision

Produce a partitioned ranked list. A family-wide recommendation is forbidden
unless all material partitions have the same measured mechanism.

### Phase 8 — vocab and sampler tail

The vocab row is `+67.61 us` in the localization census, but tinygrad has five
physical nodes and llama has two. Previous top-1 fusion was wall-negative.

#### Work

1. Decompose the Q6 vocab main consumer, partial/reduction nodes, and sampler
   tail individually.
2. Measure exact body and clean/production HCQ cost for the main consumer and
   every node above 2 us/token total mass.
3. Define an installed boundary from final-norm output ready to sampler
   feedback ready on both implementations.
4. Compute real bytes and reduction traffic.
5. Test by simulation—not implementation—whether a different main-output
   topology reduces partials while retaining the existing sampler contract.

#### Closed candidate

Do not repeat top-1 fusion into the current main route without evidence that a
changed main-output topology invalidates the prior wall-negative result.

#### Decision

If the legal tail ceiling is below 20 us/token after alternate-path takeover,
close vocab for the 240 campaign. Otherwise write one main/reduction topology
scope.

### Phase 9 — coupled 4096 norm/provider and residual support audit

The 4096 norm `+125.44 us` and activation quant `-114.43 us` rows are coupled.
Their current net attribution is only `+11.01 us`.

#### Work

1. Preserve the 17 shared norm+Q8 providers as a frozen control partition.
2. Separate standalone 4096 norms from fused providers by semantic site.
3. Measure only the standalone partition's `B`, `C`, `P`, and installed
   boundary cost.
4. Reconfirm the positional residual/rope mappings from Phase 2.
5. Reject any candidate accounting that removes a norm cost while reintroducing
   the `114.43 us` activation-quant advantage as separate work.

#### Decision

This phase may outrank an earlier island only on a non-double-counted legal
wall ceiling. The raw `125.44 us` row is never an independent recovery claim.

### Phase 10 — corrected causal wall ledger

#### Work

Build one ledger whose rows are disjoint installed islands. For each row emit:

- tinygrad installed span/union;
- llama installed span/union where reconstructible;
- exact-body delta;
- clean-dispatch delta;
- production-conditioned residual delta;
- overlap/path interaction;
- zero-cost wall-sensitivity ceiling;
- legal mechanism ceiling;
- measured candidate wall result if one exists;
- confidence and evidence paths.

Required identities:

```text
wall = union + host_gap
node_sum = union + overlap
P = B + D + R                         # every decomposed tinygrad spelling
sum(disjoint island wall attribution) = measured device-path delta
```

Any remaining cross-backend residual is labelled `UNMEASURED`; do not assign
it proportionally.

#### 240 feasibility gate

Recompute alternate paths while selecting legal mechanisms. Publish:

```text
measured booked recovery
sum of non-overlapping legal ceilings
remaining recovery to 240
```

Verdict:

- `240_MEASURED`: only after endpoint wall is at or below 4166.666667 us.
- `240_BUILDABLE`: disjoint legal mechanisms have measured/simulated ceilings
  sufficient to close the remaining gap, but implementation is pending.
- `240_UNCLOSED`: legal ceilings do not close the gap.
- `240_UNMEASURED`: causal ledger or identity remains open.

### Phase 11 — implementation handoff scopes

This phase writes documents only. Select the smallest set of disjoint
mechanisms whose conservative legal ceilings can reach the target.

Each implementation scope must contain:

1. exact production row and dependency edge;
2. measured dominant term (`B`, `D`, or `R`);
3. code paths that would change;
4. target-derived legality facts;
5. fallback and closed-default admission;
6. numerical contract;
7. isolated and installed measurement arms;
8. node/copy/materialization census gates;
9. reverse wall bracket and promotion threshold;
10. rollback and cross-target non-regression plan;
11. projected ceiling clearly labelled unmeasured;
12. prohibition on model-name dispatch.

Order implementation scopes by conservative non-overlapping wall leverage,
not by authorship convenience.

## 7. Required scripts and reuse policy

Prefer extending measurement tools rather than duplicating parsers. Reuse:

- `extra/llm_research/decode/nv_full_token_role_census.py` for role closure;
- `extra/llm_research/decode/nv_cubin_capture.py` for exact binaries;
- `extra/llm_research/decode/nv_cubin_ncu_launcher.py` for retained cubin replay;
- `extra/llm_research/decode/nv_hcq_dispatch_slope.py` for clean HCQ slopes;
- `extra/llm_research/decode/nv_hcq_profile_per_kernel.py` for faithful profile intervals;
- `extra/llm_research/decode/nv_duration_attach.py` and
  `cuda_duration_attach.py` for timeline attachment;
- `extra/llm_research/decode/full_token_dag_capture.py` for dependency capture;
- `extra/llm_research/decode/dag_critical_path_sim.py` for alternate-path
  recomputation;
- `extra/llm_research/decode/nv_full_token_dram_counters.py` for real-byte
  counters;
- the established reverse-bracket harnesses for endpoint gates.

If a shared tool's semantics are wrong, fix it only under `extra/llm_research`
and record old versus new output. Do not change production profiling semantics
inside this scope.

Create one orchestrator, if useful, under
`extra/llm_research/decode/nv_installed_island_campaign.py`. It must:

- run one named phase at a time;
- default to dry-run command emission;
- fail closed on dirty unauthorized paths, missing lock, hash mismatch, token
  mismatch, or unstable controls;
- never promote a route or edit production files;
- emit commands and manifests before execution;
- support resuming from retained evidence without rerunning closed phases.

## 8. Stop conditions and anti-patterns

Stop the current phase immediately when:

- the exact binary or route differs between supposedly matched arms;
- token SHA differs under an exact gate;
- a control bracket drifts beyond its declared stability threshold;
- the model, depth, clock state, driver, or GPU process set changes;
- `P = B + D + R` fails beyond timestamp resolution;
- cardinality cannot be reconstructed;
- a proposed candidate moves cost outside the measured island;
- a required implementation would modify a prohibited production path.

Do not:

- infer body speed from HCQ command intervals;
- infer wall recovery from node count;
- infer a global dispatch tax from one small-kernel family;
- call the no-op dispatch floor fully removable;
- assign production residual to cache without counters;
- sum overlapping island ceilings;
- use isolated bandwidth as installed bandwidth;
- compare a hot tinygrad replay to a cold llama production node;
- reopen a closed candidate without naming the changed premise;
- promote on projected tok/s;
- specialize dispatch to `Qwen3-8B`, a model filename, or a fixed block list as
  the final mechanism.

## 9. Per-phase report template

Every phase report begins with findings ordered by wall severity and uses this
table:

| field | required value |
| --- | --- |
| verdict | one named verdict |
| claim class | measured / inferred / unmeasured |
| production row/island | exact semantic boundary |
| cardinality | physical and semantic counts |
| endpoint control | us/token and tok/s |
| installed tinygrad | span, union, command sum |
| installed llama | span, union, body sum |
| exact body | tinygrad and llama |
| clean HCQ | slope, intercept, R-squared |
| production residual | value and named/unnamed mechanism |
| zero-cost ceiling | alternate-path-aware |
| legal ceiling | mechanism-specific, projected |
| measured wall delta | bracket or `UNMEASURED` |
| token evidence | SHA and gate |
| raw evidence | paths and hashes |
| next action | exactly one action or stop |

End every report with:

```text
node_sum = ...
union = ...
overlap = ...
wall = ...
host_gap = ...
useful_body = measured | unmeasured
booked_recovery = ...
remaining_to_240 = ...
```

## 10. Final deliverables

1. Fresh same-session endpoint and profiler-tax record.
2. Full semantic island manifest with exact physical cardinality.
3. Q/K installed-island causal completion.
4. O projection decomposition.
5. Gate/up/down FFN-island decomposition.
6. Flash score/combine/O-handoff decomposition.
7. Partitioned Q and K/V projection decomposition.
8. Vocab/tail decomposition.
9. Coupled standalone-norm/provider audit.
10. Corrected disjoint causal wall ledger.
11. 240 feasibility verdict.
12. Ranked implementation scopes sufficient to execute the next campaign.
13. One evidence index with SHA verification commands.
14. One explicit list of every prior claim found wrong, overstated,
    cross-domain, double-counted, or still unverified.

## 11. Completion condition

This scope is complete only when another reviewer can reconstruct, from raw
evidence, the following statement with zero unexplained residual:

```text
At the fresh control wall W, reaching 240 requires W - 4166.666667 us.

For each disjoint installed island:
  X us is intrinsic body difference,
  Y us is clean dispatch/front-end difference,
  Z us is production-conditioned residual,
  O us is overlap/path interaction,
  and L us is recoverable by this legal mechanism before alternate-path takeover.

Measured booked gains total M us.
The remaining target is R us.
The selected non-overlapping build scopes do or do not close R.
```

If the data cannot support that statement, the correct final verdict is
`240_UNMEASURED`, not optimism and not a projected benchmark.
