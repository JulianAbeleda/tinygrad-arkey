# Pure Machine Search: True Generation + Agnostic Generalization Scope

Date: 2026-06-30

Status: future execution scope. This starts after the current PMS-R0/R1/R3-style route/profile substrate. It is not a
kernel implementation request by itself.

## Verdict On The Claim

The claim is **directionally true**, but it needs a sharper boundary:

```text
true:
  The current hot-kernel problem is no longer "missing a route selector."
  The repo now has route manifests, profile descriptors, refuted-axis rows, and a generator check that can refuse
  out-of-profile or already-refuted work.

not yet true:
  The machine still does not author the key topology itself.
  G3 is generated code, but the winning lane-map/topology was still human-authored and then templated.
  The agnostic profile machinery is a substrate, not proof that another quant/model/GPU will work.
```

So the next north-star gap is:

```text
profile -> candidate topology author -> generated kernel -> evaluator -> promote/refute
```

not:

```text
human writes lane map -> machine selects it
```

## Source Citations

Read these before execution:

| claim | citation |
|---|---|
| Current pure-search boundary and remaining hot-kernel scope | `docs/pure-machine-search-remaining-hot-kernels-scope-20260630.md` |
| Route manifest exists | `extra/qk_route_manifest.py`, `bench/qk-search-spaces/default_route_manifest.json` |
| Profile exists for Qwen3-8B-Q4_K_M/gfx1100 | `bench/qk-search-spaces/profiles/qwen3_8b_q4_k_m_gfx1100.json` |
| Manifest-driven candidate generator passed | `bench/qk-search-spaces/pms_r3_candidate_generator_check.json` |
| G3 generated route is promoted/speed-equivalent | `bench/amd-isa-backend-g3-weight-promotion/summary.md`, `bench/amd-isa-backend-g3-weight-promotion/latest.json` |
| G3 writer/codegen path | `extra/qk_gemv_g3_codegen_lowering.py` |
| Prefill role-selective current default | `bench/qk-prefill-pipe-role-selective/summary.md`, `extra/qk_prefill_pipe_role_selective.py` |
| Current high-level handoff | `docs/claude-active-work-audit-and-agnostic-search-scope-20260630.md` |
| Generic search labels | `docs/generic-low-level-search-goal-scope.md` |

## Current State

Current `bench/qk-search-spaces/pms_r3_candidate_generator_check.json` reports:

```text
PMS_R3_PASS_MANIFEST_DRIVEN_CANDIDATES
next_candidate = NO_UNTRIED_CANDIDATE_TARGETS_A_FAILED_ROW
declared_rows = 11
open_failed_rows = []
promoted_or_shipped_rows = 11
do_not_search_axes = 7
```

Meaning:

- For the current Qwen3-8B-Q4_K_M/gfx1100 profile, the hot-kernel route space is closed under the declared candidates.
- Reopening current hot kernels requires a new measured premise.
- The right next work is not another local knob sweep; it is making the candidate author able to generate new topology
  families for new profiles.

## Goal

Build the first true-generation loop:

```text
profile descriptor
  -> quant/role/shape census
  -> theoretical ceiling + wall-share audit
  -> topology search space
  -> lane-map / tile-map candidate author
  -> generated kernel
  -> route-bound correctness
  -> W==D / whole-prefill authority
  -> promote/refute/rollback ledger
```

The positive control is current Q4_K G3:

```text
Given only the profile + quant semantics + candidate grammar,
can the machine regenerate a G3-equivalent lane-map route and prove it?
```

If it cannot rediscover G3 on the solved profile, it is not ready to claim agnostic search.

## Phase TG0: Provenance Audit For G3

Goal: split G3 into generated, templated, and still-human parts.

Build:

```text
extra/qk_g3_true_generation_audit.py
```

Outputs:

```text
bench/qk-g3-true-generation-audit/latest.json
bench/qk-g3-true-generation-audit/summary.md
bench/qk-g3-true-generation-audit/provenance_rows.json
bench/qk-g3-true-generation-audit/manual_topology_gaps.json
```

Required questions:

1. Which pieces are generated mechanically by `qk_gemv_g3_codegen_lowering.py`?
2. Which pieces are encoded as human topology knowledge?
3. Which shape guards are Qwen/gfx1100 specific?
4. Which quant semantics are Q4_K-specific?
5. Which parts would need to vary for Q5_K, Q6_K, Q8_0, 14B/32B, MoE, NVIDIA, or Metal?

Acceptance:

- The artifact identifies the exact remaining hand-authored objects.
- It distinguishes "generated code" from "generated topology."
- It names the minimal topology grammar needed to reproduce G3.

Verdicts:

```text
TG0_PASS_G3_PROVENANCE_PINNED
TG0_BLOCKED_G3_ROUTE_PROVENANCE_AMBIGUOUS
```

## Phase TG1: LaneMap/TileMap Template IR

Goal: create a small declarative IR that can express the current winning route without embedding it as Python control
flow.

Build:

```text
extra/qk_lanemap_template.py
bench/qk-search-spaces/templates/lanemap_v1.schema.json
bench/qk-search-spaces/templates/q4k_g3_lanemap_v1.json
```

Minimum IR fields:

```json
{
  "template_id": "q4k_g3_lanemap_v1",
  "quant": "Q4_K",
  "role_family": "decode_gemv",
  "wave_size": 32,
  "block_elems": 256,
  "packed_word_bytes": 4,
  "lane_ownership": {"axis": "packed_word", "lanes": 32},
  "dequant": {"source": "Q4_K", "ops": ["load_qs", "load_scales", "unpack_nibbles", "scale_min", "fma"]},
  "accumulation": {"dtype": "fp32", "reduction": "lane_partition_reduce_sum"},
  "store": {"mode": "direct_out"},
  "shape_guards": [{"K_mod": 256, "N_roles": ["4096", "12288"]}],
  "target_guards": {"vendor": "AMD", "wave": 32}
}
```

Acceptance:

- The existing G3 route can be reconstructed from the template.
- The reconstructed route is token-equivalent and route-clean.
- No speed claim is required in TG1 unless reconstruction changes codegen output.

Verdicts:

```text
TG1_PASS_TEMPLATE_ROUNDTRIP
TG1_BLOCKED_TEMPLATE_NOT_EXPRESSIVE
```

## Phase TG2: Candidate Topology Author

Goal: make the machine author lane-map candidates from a grammar instead of selecting a hand-written map.

Build:

```text
extra/qk_topology_candidate_author.py
bench/qk-search-spaces/topology_grammar_v1.json
bench/qk-topology-author/latest.json
```

Candidate grammar dimensions:

```text
lane ownership axis:
  packed_word | block_group | output_row | token | split

lane grouping:
  1 row/warp | 2 rows/warp | half-warp | subgroup

load pattern:
  coalesced packed-word | strided packed-word | scalar fallback

dequant placement:
  per-lane in-register | shared predecode | split dequant/consume

reduction:
  lane_partition_reduce_sum | ds_bpermute tree | LDS partial + reduce

output:
  direct_out | partials + external sum

target features:
  wave32 | wave64 | subgroup32 | subgroup_simdgroup | vector dot availability
```

Positive-control task:

```text
Generate a candidate set for Q4_K decode GEMV from profile qwen3_8b_q4_k_m_gfx1100.
The set must include a G3-equivalent topology without hardcoding route_id=decode_q4k_g3_generated.
```

Acceptance:

- The generated candidate set includes a topology matching the promoted G3 route.
- The author can explain why refuted axes are excluded.
- The author emits bounded candidate count, not combinatorial explosion.

Verdicts:

```text
TG2_PASS_G3_REDISCOVERED_BY_GRAMMAR
TG2_BLOCKED_GRAMMAR_MISSES_G3
TG2_BLOCKED_CANDIDATE_EXPLOSION
```

## Phase TG3: Quant Semantics Library

Goal: make quant support data-driven so new quant types are not hardcoded one by one.

Build:

```text
extra/qk_quant_semantics.py
bench/qk-search-spaces/quant_semantics.json
bench/qk-quant-semantics-audit/latest.json
```

Minimum quant rows:

```text
Q4_K
Q5_K
Q6_K
Q8_0
fp16
```

For each quant:

```text
block_elems
block_bytes
scale/min layout
packed bit layout
dequant ops
preferred load width
natural lane extent
known good route families
known refuted route families
quality constraints if lossy/demotion is involved
```

Acceptance:

- Q4_K semantics reproduce G3.
- Q6_K semantics reproduce the current shipped coop route and mark the direct half-warp route refuted as built.
- Unsupported quants fail as `SEARCH_SPACE_INCOMPLETE`, not by falling into Q4_K assumptions.

Verdicts:

```text
TG3_PASS_QUANT_SEMANTICS_READY
TG3_BLOCKED_QUANT_LAYOUT_UNKNOWN
```

## Phase TG4: New-Profile Opener

Goal: make adding a model/quant/GPU start from audits, not flags.

Build:

```text
extra/qk_profile_opener.py
bench/qk-search-spaces/profiles/<new_profile>.json
bench/qk-profile-opener/<new_profile>/latest.json
```

Required flow for every new profile:

```text
1. model/shape census
2. quant mix census
3. GPU target feature census
4. default-path route census
5. theoretical ceiling
6. wall-share attribution
7. declared search rows
8. do_not_search inherited from matching prior profiles
9. first candidate recommendation or NO_ACTION
```

Acceptance:

- It can regenerate the current Qwen3-8B/gfx1100 profile exactly.
- It can create a draft profile for one additional target without manual route flags.
- It refuses to generate candidates if any of model shape, quant layout, or GPU feature requirements are missing.

Candidate next profiles:

```text
qwen3_14b_or_32b_q4_k_m_gfx1100
llama_8b_q4_k_m_gfx1100
qwen3_8b_q5_k_m_gfx1100
qwen3_8b_q4_k_m_nvidia
qwen3_8b_q4_k_m_metal
```

Pick only one first. Prefer another AMD/gfx1100 profile before cross-vendor, because it isolates quant/model shape from
GPU backend differences.

Verdicts:

```text
TG4_PASS_NEW_PROFILE_OPENER_READY
TG4_BLOCKED_PROFILE_MISSING_MODEL_METADATA
TG4_BLOCKED_PROFILE_MISSING_QUANT_SEMANTICS
TG4_BLOCKED_PROFILE_MISSING_TARGET_FEATURES
```

## Phase TG5: Cross-Target Feature Model

Goal: separate algorithmic route families from target-specific lowering.

Build:

```text
bench/qk-search-spaces/targets/amd_gfx1100.json
bench/qk-search-spaces/targets/nvidia_sm*.json
bench/qk-search-spaces/targets/apple_metal_*.json
extra/qk_target_features.py
```

Feature fields:

```text
wave/subgroup size
coalescing granularity
vector dot / matrix core availability
shared memory/LDS size
barrier model
register file limits
occupancy model
native ISA backend availability
external compiler ownership
profiling availability
```

Acceptance:

- AMD gfx1100 target reproduces current route permissions.
- NVIDIA/Metal profiles are allowed to exist as `TARGET_BACKEND_INCOMPLETE` until lowering/profiling gates pass.
- Candidate author can say "this topology is algorithmically plausible but target lowering is missing" instead of
  silently pretending portability.

Verdicts:

```text
TG5_PASS_TARGET_FEATURE_MODEL_READY
TG5_BLOCKED_TARGET_BACKEND_INCOMPLETE
```

## Phase TG6: Template Search Evaluator

Goal: evaluate topology-authored candidates with the same strict gates as current handoff routes.

Build or extend:

```text
extra/qk_candidate_evaluator.py
extra/qk_template_candidate_gate.py
```

Gate ladder:

```text
1. template schema validation
2. generated kernel builds
3. route attribution proves intended candidate fired
4. token/logit correctness
5. W==D or whole-prefill authority
6. attribution/ceiling explains movement
7. ledger update
```

Acceptance:

- G3 rediscovery candidate passes and maps to the existing promoted route.
- A known bad candidate, such as Q6_K direct half-warp as built, remains refuted.
- A missing-target candidate reports `SEARCH_BLOCKED_BY_RUNTIME` or `SEARCH_BLOCKED_BY_CODEGEN`.

Verdicts:

```text
TG6_PASS_TEMPLATE_EVALUATOR_REPLAYS_CONTROLS
TG6_BLOCKED_EVALUATOR_ROUTE_ATTRIBUTION
```

## Phase TG7: First New-Profile Search

Goal: prove the system is actually agnostic on one new profile.

Recommended first target:

```text
same GPU, different model/shape or quant mix
```

Rationale:

- Same GPU avoids conflating search generality with backend portability.
- Different shape/quant tests whether profile-driven topology generation works.
- Cross-vendor comes after the target feature model is real.

Process:

```text
TG4 opener -> TG2/TG3 candidate author -> TG6 evaluator -> ledger
```

Acceptance:

- The system either finds a promotable route or emits a precise blocker:
  - `SEARCH_EXHAUSTED_SPACE`
  - `SEARCH_SPACE_INCOMPLETE`
  - `SEARCH_BLOCKED_BY_CODEGEN`
  - `SEARCH_BLOCKED_BY_RUNTIME`
  - `NO_ACTION_UNDER_CEILING`
- It does not require editing model route code by hand for the first candidate.

Verdicts:

```text
TG7_PASS_FIRST_NEW_PROFILE_SEARCH_RESULT
TG7_BLOCKED_MANUAL_ROUTE_EDIT_REQUIRED
```

## What Would Make This "True Machine Search"

A credible claim requires all of:

1. The machine authors at least one topology from grammar/profile, not by selecting a pre-existing hand route.
2. The authored topology compiles to a route-bound generated kernel.
3. The evaluator promotes/refutes it with correctness and authority speed gates.
4. The same pipeline works on a second profile without hand-editing route logic.
5. The old owned/specialized kernel remains only fallback/reference for promoted cases.

The first milestone is not beating current G3. The first milestone is:

```text
rediscover current G3 from a topology grammar
```

That proves the search space can express a known optimum. Only then should we ask it for unknown wins.

## Non-Goals

- Do not reopen current Q4_K G3 speed tuning while parity holds.
- Do not reopen Q6_K direct half-warp as built.
- Do not reopen broad decode attention under the current low-leverage ceiling.
- Do not claim NVIDIA/Metal portability until target lowering and profiling are gated.
- Do not treat "profile file exists" as proof of agnostic search.
- Do not merge new defaults without route attribution, correctness, authority speed, and rollback.

## Claude Execution Prompt

Use this prompt for the next agent:

```text
You are in /home/ubuntu/tinygrad-arkey on master. Read:

1. docs/pure-machine-search-true-generation-agnostic-scope-20260630.md
2. docs/pure-machine-search-remaining-hot-kernels-scope-20260630.md
3. extra/qk_route_manifest.py
4. bench/qk-search-spaces/default_route_manifest.json
5. bench/qk-search-spaces/search_profiles.json
6. bench/qk-search-spaces/profiles/qwen3_8b_q4_k_m_gfx1100.json
7. bench/qk-search-spaces/pms_r3_candidate_generator_check.json

Task: execute TG0 and TG1 only. Do not implement new kernels or change defaults.

TG0:
- Build extra/qk_g3_true_generation_audit.py.
- Emit bench/qk-g3-true-generation-audit/{latest.json,summary.md,provenance_rows.json,manual_topology_gaps.json}.
- Pin exactly which pieces of G3 are generated, templated, or still hand-authored.

TG1:
- If TG0 passes, define the smallest LaneMapTemplate IR that can losslessly express existing G3.
- Emit bench/qk-search-spaces/templates/lanemap_v1.schema.json and
  bench/qk-search-spaces/templates/q4k_g3_lanemap_v1.json.
- Round-trip/check the template against the current G3 route. Correctness/route identity is enough; no speed re-run
  unless generated code changes.

Stop at first hard blocker.

Acceptable verdicts:
- TG0_PASS_G3_PROVENANCE_PINNED + TG1_PASS_TEMPLATE_ROUNDTRIP
- TG0_BLOCKED_G3_ROUTE_PROVENANCE_AMBIGUOUS
- TG1_BLOCKED_TEMPLATE_NOT_EXPRESSIVE
```

## End State

After this scope, the project has a clean ladder:

```text
current route/profile substrate
  -> G3 topology provenance
  -> LaneMapTemplate IR
  -> topology candidate author
  -> quant semantics library
  -> new-profile opener
  -> cross-target feature model
  -> template evaluator
  -> first new-profile search
```

That is the path from "generated selected route" to "machine-authored route."
