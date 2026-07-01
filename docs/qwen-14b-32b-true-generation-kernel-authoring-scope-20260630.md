# Qwen 14B/32B True-Generation Decode Kernel Authoring Scope

Date: 2026-06-30

Status: execution scope for Claude. No default route changes until the promotion gates pass.

## Why This Scope Exists

The true-generation track proved the machinery can author topology from a grammar:

- `TG2_PASS_G3_REDISCOVERED_BY_GRAMMAR`: the topology grammar rediscovered the Q4_K G3 LaneMap without route-id matching.
- `TG6_PASS_TEMPLATE_EVALUATOR_REPLAYS_CONTROLS`: grammar candidates can flow through the evaluator.
- `TG7_PASS_FIRST_NEW_PROFILE_SEARCH_RESULT`: the pipeline ran on a new Q6_K profile and honestly exhausted the space without manufacturing a win.

That is still not the same as producing a new winning kernel for a new model shape. Qwen3-14B and Qwen3-32B are the right next targets because they expose a real measured gap:

| model | quant | ctx512 tinygrad | ctx512 llama.cpp | ratio |
|---|---|---:|---:|---:|
| qwen3-8b | Q4_K_M | 103.5 tok/s | 98.7 tok/s | 104.9% |
| qwen3-14b | Q4_K_M | 25.0 tok/s | 65.08 tok/s | 38.4% |
| qwen3-32b | Q4_K_M | 11.8 tok/s | 30.78 tok/s | 38.3% |

The 8B G3 route is speed-equivalent to owned on its tracked shapes, but the current model guard is explicitly shape-specific:

```text
g3_bubblebeam_shape:
  in=4096, out in {4096,12288}
  or in=12288, out=4096
```

So the first job is not "tune harder." It is to prove whether 14B/32B are falling off because the promoted generated route is not covering their larger Q4_K decode shapes.

## Source Citations

Read these before implementation:

| claim | citation |
|---|---|
| 14B/32B decode gap vs llama.cpp | `bench/models/qwen/amd-rx7900xtx-gfx1100.md` |
| Current G3 shape guard and default route | `tinygrad/llm/model.py`, `Q4KLinear.__call__` G3 branch |
| Short-context flash policy and ctx512 crossover | `tinygrad/llm/model.py`, `should_use_flash_decode` |
| TG north-star loop | `docs/pure-machine-search-true-generation-agnostic-scope-20260630.md` |
| G3 rediscovery scope | `docs/true-generation-rediscover-g3-scope-20260630.md` |
| TG2 author artifacts | `bench/qk-topology-author/summary.md`, `extra/qk_topology_candidate_author.py` |
| TG6 evaluator bridge | `bench/qk-template-candidate-gate/summary.md`, `extra/qk_template_candidate_gate.py`, `extra/qk_candidate_evaluator.py` |
| TG7 new-profile result | `bench/qk-new-profile-search/qwen3_8b_q6k_ffn_down_gfx1100/summary.md`, `extra/qk_new_profile_search.py` |
| LaneMap IR | `extra/qk_lanemap_template.py` |
| G3 codegen emitter | `extra/qk_gemv_g3_codegen_lowering.py` |
| Quant semantics data | `bench/qk-search-spaces/quant_semantics.json` |
| gfx1100 target facts | `bench/qk-search-spaces/targets/amd_gfx1100.json` |
| Route manifest | `bench/qk-search-spaces/default_route_manifest.json`, `extra/qk_route_manifest.py` |
| Artifact cache design | `docs/pure-machine-search-artifact-cache-scope-20260630.md` |
| Runtime/prefill VRAM planner scope | `docs/model-agnostic-runtime-route-planner-scope-20260630.md` |

## Goal

Use Qwen3-14B and Qwen3-32B as the first real "new winning generated kernel" attempt:

```text
model profile
  -> role/shape/quant census
  -> route-miss proof
  -> topology grammar candidates
  -> generated Q4_K kernel emission
  -> route-bound correctness
  -> W==D and llama-matched speed
  -> promote/refute with rollback
```

Primary target: decode Q4_K GEMV for the large Qwen3 dense models.

Secondary target: prefill only if the VRAM planner says the tuned prefill path fits on the active device. Do not force prefill flags to chase a benchmark if the memory math says it does not fit.

Default outcome if successful: promote the generated route as the default for the exact passing profile class, with owned/current routes retained as rollback. This is the intended pure-machine-search direction: profile-authored generated kernels should become the shipped route after they prove correctness, speed, and memory fit.

## Non-Goals

- Do not hand-write a 14B/32B custom kernel.
- Do not widen `g3_bubblebeam_shape` by hardcoded model dimensions and call that pure search.
- Do not touch attention unless route attribution shows attention is the dominant measured gap.
- Do not force `PREFILL_V2` for 14B/32B on a 24GB card if the auto planner rejects it.
- Do not assume multi-GPU VRAM can be summed. A loaded model and its KV/cache/compiled buffers must fit on the device that executes the kernels unless a real sharding backend exists.
- Do not change defaults until route-bound correctness and W==D gates pass.
- Do not edit `autogen/**`.

## Success Definitions

There are two different bars:

| bar | meaning |
|---|---|
| Promotion bar | Candidate beats current tinygrad shipped default with token/logit equivalence and no protected-context regression. Use the tiered policy: TIER_A >= 5%, TIER_B >= 2%, protected-context regression <= 1%. |
| Track closure bar | Candidate brings 14B/32B decode near external parity: target >= 80% of matched-context llama.cpp, stretch >= 95%. If promotion passes but llama parity does not, keep searching or record the remaining gap. |
| Memory-fit bar | tinygrad candidate must fit the same model, quant, context, and active GPU target that llama.cpp fits under full GPU offload. If llama.cpp fits and tinygrad OOMs, do not promote; open a memory-planner/storage issue. If llama.cpp also OOMs under the same full-offload target, the model/context is outside the device envelope and the candidate can still pass for smaller contexts. |

The external llama number is a sanity target, not the only promotion authority. The internal promotion authority is shipped tinygrad default vs generated candidate under the same W==D harness.

## Outcome Taxonomy: Refuted vs Frontier

A failed generated route is not automatically a dead end. It must be classified into one of these outcomes:

| outcome | meaning | next action |
|---|---|---|
| `PROMOTED` | Generated route is correct, route-bound, memory-fit, and speed passes. | Make it default for the passing profile class with rollback. |
| `SPEED_EQUIVALENT` | Generated route matches shipped route and is more pure/search-owned. | Promote if it reduces hand-written surface and has no regression. |
| `REFUTED_CANDIDATE` | This exact topology/codegen candidate was fairly tested and lost. | Ledger candidate id, measured loss, and refuted axis. |
| `SEARCH_SPACE_INCOMPLETE` | The measured bottleneck points to a knob/topology/codegen capability that the grammar could not express. | Do not mark the route family dead. Add the missing axis/capability and reopen from that frontier. |
| `CODEGEN_CAPABILITY_BLOCKED` | Grammar can author the idea, but the emitter/lowerer cannot generate it correctly or efficiently yet. | Scope the missing lowering/regalloc/scheduler feature, then replay the cached candidate. |
| `MEMORY_BLOCKED` | Speed/correctness may pass, but the route does not fit where the baseline/llama full-offload fits. | Fix storage/planner/residency before promotion. |
| `TARGET_CHANGED` | Route attribution shows the measured gap is not in Q4_K GEMV. | Move to the measured bucket instead of forcing this scope. |

This is the main discipline for the 14B/32B work: the loop should preserve promising failed ideas as frontier items when the machine lacks the knobs needed to express the winning variant. Only an actually represented and fairly measured axis can be refuted.

## Memory And llama.cpp Fit Policy

The comparison must distinguish performance from residency:

```text
valid comparison:
  tinygrad fits on GPU
  llama.cpp fits on GPU with full layer offload
  same quant
  same context
  same active GPU device

invalid comparison:
  llama.cpp silently spills layers/KV to CPU
  tinygrad uses GPU-only execution
  then llama "fits" but the speed number is not a like-for-like GPU target
```

For every measured context, collect:

| row | tinygrad | llama.cpp |
|---|---|---|
| model file bytes | GGUF size | GGUF size |
| GPU weight residency | loaded tensor/storage bytes | fully offloaded layer bytes |
| KV bytes | model config × context × dtype | llama reported or estimated KV bytes |
| temporary/compile/cache bytes | runtime cache/precompile sidecars | scratch/KV/graph allocations if visible |
| peak VRAM | sampled with ROCm SMI or runtime metric | sampled with ROCm SMI during `llama-bench` |
| fit verdict | PASS/OOM | PASS/OOM/full-offload-not-achieved |

Promotion rule:

```text
if tinygrad_passes_speed_and_correctness and tinygrad_peak_vram <= device_budget and llama_full_offload_fits:
  candidate is promotable for that profile/context envelope
elif tinygrad_oom and llama_full_offload_fits:
  block promotion and fix memory/storage/planner first
elif tinygrad_fits and llama_oom:
  promotion may proceed; tinygrad has a fit advantage
elif both_oom:
  record max-supported context/model envelope; do not promote for the failing envelope
```

This keeps "pure machine search default" honest: a generated kernel is not production-ready if it only wins speed by requiring a memory layout that makes the target model unusable.

## Phase Q1432-0: Baseline And Route-Miss Proof

Goal: prove the measured gap and identify which route is actually running before building anything.

Build or extend:

```text
extra/qk_large_model_decode_route_gap_audit.py
```

Inputs:

```text
qwen3-14b Q4_K_M GGUF
qwen3-32b Q4_K_M GGUF
optional: qwen3.5-27b only as an unsupported-architecture diagnostic, not a primary target
```

Measure:

| measurement | requirement |
|---|---|
| W==D decode | ctx128, ctx512, ctx2048, and ctx4096 if runtime is practical |
| llama.cpp comparison | matched depth, same GGUF where possible |
| route attribution | Q4_K role counts: G3, owned/warp, fallback, bridge, unknown |
| host-sync | prove GPU-bound, not host-loop dominated |
| flash route | prove attention route state at ctx512+ |
| memory fit | peak VRAM and OOM status for tinygrad and llama full-offload |

Outputs:

```text
bench/qwen-14b-32b-truegen/q1432_0_baseline/latest.json
bench/qwen-14b-32b-truegen/q1432_0_baseline/summary.md
bench/qwen-14b-32b-truegen/q1432_0_baseline/route_counts.json
bench/qwen-14b-32b-truegen/q1432_0_baseline/llama_compare.json
bench/qwen-14b-32b-truegen/q1432_0_baseline/memory_fit.json
```

Pass verdict:

```text
Q1432_0_PASS_GAP_AND_ROUTE_MISS_PINNED
```

Block verdicts:

```text
Q1432_0_BLOCKED_MODEL_MISSING
Q1432_0_BLOCKED_MEASUREMENT_NOISE
Q1432_0_ABORTED_NO_ROUTE_MISS
Q1432_0_BLOCKED_MEMORY_COMPARISON_INVALID
```

Abort if the current generated route already fires for the major 14B/32B Q4_K roles and the gap comes from a different bucket.

Memory-specific stop condition: if llama's published row is not full GPU offload, do not use it as the external parity target until a full-offload llama row is captured. CPU spill is allowed as a separate "fits by spilling" note, but it is not a fair GPU decode comparator.

## Phase Q1432-1: Shape, Role, And Quant Census

Goal: turn each large model into a search profile, not a model-name special case.

Build or extend:

```text
extra/qk_large_model_profile_census.py
```

For each model, extract:

| field | examples |
|---|---|
| architecture | qwen3 dense vs qwen3.5 hybrid |
| quant by tensor role | Q4_K, Q6_K, Q8_0, fp16 |
| linear role | ffn_gate_up, ffn_down, attn_qo, attn_kv, lm_head |
| shape | in_features, out_features, parts, block count |
| decode wall share | from Q1432-0 attribution |
| route eligibility | whether current G3 guard covers it |
| memory footprint | model bytes, KV bytes by context, extra fp16/precompile bytes |
| fit envelope | max context that fits on the active GPU under tinygrad and llama full-offload |

Create profile descriptors:

```text
bench/qk-search-spaces/profiles/qwen3_14b_q4_k_m_gfx1100_decode.json
bench/qk-search-spaces/profiles/qwen3_32b_q4_k_m_gfx1100_decode.json
```

Schema requirements:

- No hardcoded route names in the profile.
- Quant facts come from `quant_semantics.json`.
- Target facts come from `targets/amd_gfx1100.json`.
- Model shape facts come from GGUF metadata/tensor census.

Pass verdict:

```text
Q1432_1_PASS_PROFILE_CENSUS
```

## Phase Q1432-2: Topology Grammar Extension For Large Q4_K Shapes

Goal: author candidate topology specs for the larger shapes using the same TG grammar style that rediscovered G3.

Extend:

```text
bench/qk-search-spaces/topology_grammar_v1.json
extra/qk_topology_candidate_author.py
```

The grammar must span:

| dimension | examples |
|---|---|
| row ownership | 1 row/warp, 2 rows/warp, row group tiling |
| K decomposition | factor `K / qk_k` into block groups, words per group, local block, group pair |
| lane ownership | packed-word lane index, contiguous word index, coalesced block group |
| reduction | cross-lane wave reduce, partials plus reduce |
| store shape | direct out, partials only when justified |
| wave target | wave32 on gfx1100 |

Candidate count must be bounded:

```text
max_candidates <= topology_grammar_v1.json:max_candidates
```

Required positive controls:

1. The grammar still rediscovers 8B G3.
2. The grammar can generate candidates for 14B and 32B without adding shape-specific code.
3. If no valid candidates exist, the tool returns an honest blocked verdict.

Outputs:

```text
bench/qwen-14b-32b-truegen/q1432_2_topology_author/qwen3_14b/latest.json
bench/qwen-14b-32b-truegen/q1432_2_topology_author/qwen3_32b/latest.json
bench/qwen-14b-32b-truegen/q1432_2_topology_author/candidate_rows.json
```

Pass verdict:

```text
Q1432_2_PASS_CANDIDATES_AUTHORED
```

Block verdicts:

```text
Q1432_2_BLOCKED_GRAMMAR_MISSES_VALID_SHAPES
Q1432_2_BLOCKED_CANDIDATE_EXPLOSION
```

Anti-cheat audit:

- Candidate generation cannot reference model names like `14b` or `32b` except when loading the profile.
- Candidate generation cannot reference route ids like `g3`.
- The winning candidate cannot be injected by a `(K,N)` if-statement.
- Verification may compare to known routes; generation may not.

## Phase Q1432-3: Generated Kernel Emission And Microgates

Goal: emit runnable Q4_K GEMV kernels from authored candidate specs.

Extend:

```text
extra/qk_lanemap_template.py
extra/qk_gemv_g3_codegen_lowering.py
extra/qk_candidate_template_gen.py
```

Requirements:

- Generated function name is derived from `(quant, shape, target, topology hash)`.
- The kernel body is emitted from `LaneMapTemplate`, not copied from a hand route.
- Existing G3 output remains byte-identical for its 8B profile.
- New candidate kernels run as standalone custom kernels for each targeted role shape.

Microgates:

| gate | requirement |
|---|---|
| random vector correctness | generated output matches current fallback route within established tolerance |
| edge shape correctness | rows not divisible by candidate row grouping handled or rejected loudly |
| quant correctness | Q4_K dequant semantics identical to existing primitive |
| no fallback | route spy proves the emitted kernel ran |
| key stability | candidate hash and generated name are stable across runs |

Outputs:

```text
bench/qwen-14b-32b-truegen/q1432_3_codegen/latest.json
bench/qwen-14b-32b-truegen/q1432_3_codegen/microgate_rows.json
```

Pass verdict:

```text
Q1432_3_PASS_KERNELS_EMIT_CORRECT
```

## Phase Q1432-4: Default-Off In-Model Route Binding

Goal: bind generated candidates in `model.py` without hardcoding the large shapes.

Preferred mechanism:

```text
QK_GENERATED_ROUTE_POLICY=path/to/policy.json
```

or an equivalent route-manifest driven selector that maps:

```text
model_profile + quant + role + shape + target -> candidate topology hash
```

Requirements:

- Default-off until promotion.
- Rollback flag exists.
- If policy is absent or incomplete, current shipped route runs unchanged.
- Route attribution must show the generated candidate fires only for targeted roles.
- Token/logit equivalence at ctx128 and ctx512 before speed gates.

Outputs:

```text
bench/qwen-14b-32b-truegen/q1432_4_route_bound/latest.json
bench/qwen-14b-32b-truegen/q1432_4_route_bound/route_attribution.json
bench/qwen-14b-32b-truegen/q1432_4_route_bound/token_match.json
```

Pass verdict:

```text
Q1432_4_PASS_ROUTE_BOUND_DEFAULT_OFF
```

## Phase Q1432-5: Decode Speed Authority

Goal: measure candidate movement against both current tinygrad default and matched-context llama.cpp, while enforcing the memory-fit bar.

Contexts:

```text
ctx128
ctx512
ctx2048
ctx4096 if practical
```

Arms:

| arm | meaning |
|---|---|
| shipped_default | current tinygrad without generated large-shape route |
| generated_candidate | policy-bound generated kernel |
| llama_matched | external matched-context reference |

Metrics:

- W==D tok/s.
- route counts.
- token/logit correctness.
- host-sync percent.
- spread/noise.
- effective model-byte GB/s.
- per-role kernel-time attribution where available.
- peak VRAM and fit verdict for each arm.
- missing-axis diagnosis: if the candidate loses, identify whether the lost delta maps to a represented axis or an unrepresented knob/capability.

Pass verdicts:

```text
Q1432_5_PASS_TIER_A_WD_MOVEMENT
Q1432_5_PASS_TIER_B_RESIDUAL_MOVEMENT
Q1432_5_PASS_SPEED_EQUIVALENT_TO_SHIPPED
```

Refute verdicts:

```text
Q1432_5_REFUTED_SPEED_REGRESSION
Q1432_5_REFUTED_NO_WD_MOVEMENT
Q1432_5_INCONCLUSIVE_NOISE
Q1432_5_BLOCKED_TINYGRAD_OOM_LLAMA_FITS
Q1432_5_BLOCKED_LLAMA_COMPARISON_NOT_FULL_GPU
Q1432_5_SEARCH_SPACE_INCOMPLETE_MISSING_AXIS
Q1432_5_CODEGEN_CAPABILITY_BLOCKED
```

Promotion can pass if it beats shipped tinygrad under the tiered policy. The track is not "closed" until the remaining gap to llama is also explained or reduced to the practical memory/dequant ceiling.

When a candidate fails speed, the tool must answer:

| question | required evidence |
|---|---|
| Did this candidate cover the dominant role? | route counts and role wall-share |
| Was the losing work in a represented axis? | topology spec fields and grammar row |
| Was the losing work in an unrepresented axis? | dynamic/static attribution row mapped to "missing knob" |
| Is it an emitter quality gap? | generated source/ISA difference, instruction mix, memory pattern |
| Is it a memory-fit issue? | tinygrad/llama fit table |
| Is it noise? | repeated runs/spread |

If the answer is "unrepresented axis" or "emitter quality gap", do not write `REFUTED_NO_WD_MOVEMENT` for the route family. Write `SEARCH_SPACE_INCOMPLETE` or `CODEGEN_CAPABILITY_BLOCKED`, add the missing axis to the frontier, and stop.

## Phase Q1432-6: Ledger, Cache, And Promotion Package

Goal: make the result durable and avoid regenerating candidates unnecessarily.

Update:

```text
bench/qk-search-spaces/default_route_manifest.json
bench/qk-search-spaces/refuted_axes.json
bench/qk-search-spaces/search_profiles.json
bench/qk-search-spaces/open_frontier.json
docs/README.md
docs/pure-machine-search-remaining-hot-kernels-scope-20260630.md
```

Cache requirements:

Fingerprint must include:

```text
model profile hash
quant_semantics hash
target descriptor hash
topology grammar hash
LaneMapTemplate/codegen version
route manifest version
benchmark policy version
```

Cacheable artifacts:

- candidate topology specs;
- generated kernel source;
- standalone microgate verdicts;
- in-model route-bound verdicts;
- speed authority verdicts.
- frontier rows for blocked/missing capabilities.

Cache invalidation:

- Any change to quant semantics, target facts, grammar, codegen emitter, or model profile invalidates downstream generated kernels.
- Benchmark-only reruns should reuse candidate generation and microgate artifacts if their fingerprints match.

Pass verdict:

```text
Q1432_6_PASS_PROMOTION_OR_REFUTATION_LEDGERED
Q1432_6_PASS_FRONTIER_LEDGERED
```

Default-flip requirements:

1. `Q1432_4_PASS_ROUTE_BOUND_DEFAULT_OFF`.
2. `Q1432_5_PASS_TIER_A_WD_MOVEMENT`, `Q1432_5_PASS_TIER_B_RESIDUAL_MOVEMENT`, or `Q1432_5_PASS_SPEED_EQUIVALENT_TO_SHIPPED`.
3. No protected-context regression greater than 1%.
4. No `Q1432_5_BLOCKED_TINYGRAD_OOM_LLAMA_FITS`.
5. Route policy is profile/shape/target driven, not model-name hardcoded.
6. Rollback flag documented and tested.
7. llama.cpp fit comparison is either full-GPU-valid or explicitly marked not applicable.

If all seven hold, promote generated route default-on for the passing profile class. Keep the old route as rollback/reference.

Frontier ledger requirements:

For every non-promoted candidate, write a row with:

| field | meaning |
|---|---|
| `candidate_id` | stable topology/codegen id |
| `profile_id` | model/quant/target profile |
| `status` | `REFUTED_CANDIDATE`, `SEARCH_SPACE_INCOMPLETE`, `CODEGEN_CAPABILITY_BLOCKED`, `MEMORY_BLOCKED`, or `TARGET_CHANGED` |
| `measured_delta` | W==D delta and per-role delta |
| `dominant_failed_row` | the exact metric that lost |
| `represented_axis` | whether current grammar/codegen could express the needed fix |
| `missing_axis_or_capability` | knob/topology/lowering needed if not represented |
| `replay_command` | command to replay from cached artifacts |
| `reopen_condition` | precise condition under which this becomes searchable again |

Examples:

```text
REFUTED_CANDIDATE:
  "2rows_per_warp_bg8_wpg4_cross_lane" loses 7% W==D and the lost metric maps to a represented row.

SEARCH_SPACE_INCOMPLETE:
  static/dynamic audit says the winner needs mixed row grouping, but grammar only has fixed rows_per_warp.

CODEGEN_CAPABILITY_BLOCKED:
  topology is valid, but emitter cannot generate vectorized coalesced Q4_K loads for K=8192 without spilling.
```

## Phase Q1432-7: Prefill Follow-On Gate

Goal: do not conflate decode with prefill. Decide prefill work separately from VRAM and wall-share facts.

Run after decode phases:

```text
extra/qk_prefill_authority_refresh.py
extra/qk_prefill_whole_role_attribution.py
```

Requirements:

- Use the runtime route planner memory math.
- If tuned prefill does not fit on the active device, record `PREFILL_FAST_PATH_NOT_FIT` and stop.
- If it fits, profile role wall-share and only then open a prefill topology/codegen scope.

Expected on 24GB gfx1100:

- 8B tuned prefill fits and already wins.
- 14B/32B likely do not fit the fp16-covered fast prefill path without memory pressure; this must be measured, not guessed.

Pass/refute verdicts:

```text
Q1432_7_PASS_PREFILL_PROFILE_OPENED
Q1432_7_REFUTED_PREFILL_FAST_PATH_NOT_FIT
```

## Claude Execution Instructions

Execute strictly in order:

1. Run Q1432-0. Stop if the route miss is not real.
2. Run Q1432-1. Stop if profiles cannot be generated from GGUF facts.
3. Run Q1432-2. Stop if the grammar cannot author bounded large-shape candidates.
4. Run Q1432-3. Stop at the first correctness/codegen blocker.
5. Run Q1432-4. Stop if route binding is not clean.
6. Run Q1432-5. Promote only on measured W==D movement and no protected-context regression. If it fails, classify the failure using the outcome taxonomy before calling anything refuted.
7. Run Q1432-6 to ledger the outcome: promoted, refuted, memory-blocked, codegen-blocked, or open frontier.
8. Only then run Q1432-7 for prefill.

Do not skip to kernel implementation before Q1432-0 and Q1432-1 prove the target. Do not report "pure machine search solved" unless the candidate was authored from profile/grammar data, emitted as a generated kernel, route-bound in-model, and promoted by authority measurements. Do not report "route family refuted" unless the missing-axis audit proves the needed lever was represented and fairly measured.

## Expected Outcomes

Most likely:

- Q1432-0 proves that 14B/32B miss the current generated G3 route because their dense/FFN shapes are outside the current 8B guard.
- Q1432-2 authors larger Q4_K topology candidates from grammar/profile data.
- Q1432-3 or Q1432-5 is the real hard point: either the emitted topology does not generalize cleanly, or it is correct but not fast enough.

Best case:

- A generated large-shape Q4_K candidate closes a major part of the 14B/32B decode gap, moving them from about 38-40% of llama toward 80%+.

Honest failure:

- The grammar exhausts without a faster candidate, or the gap is not Q4_K GEMV route coverage. In that case, ledger the outcome precisely. If every relevant knob was represented, mark the candidate/axis refuted and move to the measured next bucket. If the audit shows the route could plausibly win with an unexposed knob, write an open frontier row instead of closing the family. Do not force a fake win, but also do not erase a real future win because today's grammar was too small.
