# Decode Attention Primitive-Complete Online-Softmax+PV Scope

## Goal

Build or precisely block the generated/search-owned decode-attention primitive that can replace the owned AMDGCN tile.

The target is not another metadata fusion. The target is the full decode-attention performance primitive:

```text
split-KV decode attention tile
+ whole-cache KV identity
+ T=1 parallelism from Hkv*S workgroups
+ v_dot2 / packed fp16 QK score
+ cross-lane reduction
+ register-resident online softmax state: m, l, acc[D]
+ PV accumulation in the same tile lifecycle
+ combine lifecycle accounted for as part of the candidate
```

Success means the route is generated/search-owned, lifecycle-clean, token-correct, and W==D-competitive with the current owned route.

## Why this scope exists

A2 proved the generated whole-cache route can be lifecycle-clean:

- owned tile absent
- owned combine absent
- no `E_49152`
- token sample matches
- whole-cache KV identity preserved

A3.6 through A3.10 then showed that incremental stage replacement is not enough:

| Step | Result | Meaning |
|---|---|---|
| A3.6 tile max | no transfer | `flash_max_32` is not the material gap |
| A3.7 tile prob | no transfer | max/prob metadata fusion is not enough |
| A3.9 partial PV | no transfer | one-for-one partial-PV replacement is not enough |
| A3.10 prob+partial PV | regression | naive producer-consumer fusion loses the owned route's performance shape |

The core issue is therefore not too many kernels by itself. The generated route lacks the owned tile's work decomposition, lane ownership, reduction schedule, and register-resident online-softmax+PV dataflow.

## Canonical inputs

- `docs/pure-machine-search-roadmap.md`
- `docs/decode-attention-pure-search-scope.md`
- `docs/decode-attention-a3-performance-primitive-lowering-scope.md`
- `docs/decode-attention-a3-10-tile-prob-partial-pv-result.md`
- `docs/decode-two-kernel-problem-audit-result-20260625.md`
- `structure/Development/performance-primitive-research-principles.md`

## Current comparator

The current runtime winner is still the owned attention route:

```text
owned_flash_tile_gqa_whole
owned_flash_combine
```

This route is acceptable as shipped runtime code, but it is not pure generated/search-owned code.

The generated candidate must compare against:

- A2 generated whole-cache skeleton first, to prove local transfer;
- current owned route second, to prove promotion viability.

Do not compare only against weaker or stale baselines.

## Non-goals

- Do not promote a route that uses `owned_flash_tile_gqa_whole` or `owned_flash_combine`.
- Do not claim pure machine search from a hand-written whole kernel.
- Do not chase metadata-only fusions again unless a profiler artifact says metadata re-entered the critical path.
- Do not treat fewer launches as sufficient evidence.
- Do not route a candidate that reintroduces `E_49152`.
- Do not run broad search over a manifest that excludes the primitive described here.

## Required primitive properties

| Property | Requirement |
|---|---|
| KV access | Read the whole KV cache buffer directly; preserve buffer identity |
| Parallelism | Preserve or manufacture enough T=1 work through `Hkv*S` split-KV workgroups |
| Query/GQA mapping | Preserve query-head/GQA parallelism; do not serialize G heads in a way that collapses occupancy |
| QK score | Use generated/search-owned packed fp16 dot behavior or classify the dot lowering wall |
| Reduction | Use generated/search-owned lane/cross-lane reduction or classify the reduction wall |
| Softmax state | Keep `m`, `l`, and `acc[D]` in the tile lifecycle, preferably registers |
| PV | Accumulate PV inside the tile lifecycle instead of materializing avoidable intermediate stages |
| Combine | Account for combine as part of the primitive lifecycle, even if emitted as a second program |
| Runtime | No hidden materialization, no extra host/device sync, no debug/profile contamination |

## Implementation phases

### P0: Freeze the A3.10 failure as the baseline

Goal: make sure the next work starts from the correct lesson.

Build:

- Use A3.10 as the negative control.
- Record A2, A3.10, and owned route signatures in the new gate output.
- Ensure the gate explicitly says `A3_10_REGRESSION_IS_NEGATIVE_CONTROL`.

Gate:

- A2 route clean.
- A3.10 route clean.
- A3.10 still slower than A2 at the measured ctx points.

Kill:

- If A3.10 unexpectedly becomes faster after unrelated code changes, stop and rerun attribution before building new code.

### P1: Define the primitive-complete candidate manifest

Goal: make the search space honest before building the candidate.

Build:

- Add a search-space manifest for this primitive family.
- Include instruction, memory, scheduling, dataflow, and runtime primitive fields.
- Mark unresolved items explicitly instead of hiding them.

Suggested manifest:

```text
bench/qk-search-spaces/decode_attention_online_softmax_pv_tile_v1.json
```

Required fields:

- `search_space_id`
- `primitive_family`
- `supported_profiles`
- `exposed_instruction_primitives`
- `exposed_memory_primitives`
- `exposed_scheduling_primitives`
- `exposed_dataflow_primitives`
- `exposed_runtime_primitives`
- `excluded_primitives`
- `proof_of_coverage`
- `known_negative_controls`

Gate:

- Manifest names the full primitive boundary.
- Manifest does not claim coverage for unimplemented lowerings.
- A failed search over this manifest can truthfully produce `SEARCH_SPACE_INCOMPLETE` or `SEARCH_BLOCKED_BY_CODEGEN`.

Kill:

- If the manifest cannot describe the owned route's required behavior, do not run search yet.

### P2: Build a structural generated tile skeleton

Goal: create a generated/search-owned program shape that owns the full online-softmax+PV lifecycle, even if initially slow.

Build:

- Add an opt-in route, for example `DECODE_ATTN_ONLINE_PV_TILE=1`.
- Program name should distinguish it from A3 metadata fusions, for example:

```text
flash_online_pv_tile_whole_cache_32_128
```

- Inputs should preserve the whole-cache ABI.
- Outputs should match the existing combine/lifecycle contract or emit an explicitly documented new contract.

Gate:

- Route fires.
- Owned tile/combine absent.
- No `E_49152`.
- Tokens match owned baseline sample.
- Program attribution proves the online-PV tile exists.

Kill:

- If correctness requires falling back to owned tile/combine, classify `SEARCH_BLOCKED_BY_CODEGEN` or `SEARCH_BLOCKED_BY_RUNTIME`.
- If whole-cache identity cannot be preserved, reject the route.

### P3: Add lane ownership and reduction mapping

Goal: avoid the A3.10 failure mode where simple fusion loses the owned route's parallel shape.

Build:

- Define the lane ownership map for head, split, token, and `D` lanes.
- Reuse GEMV LaneMap lessons only where semantically valid.
- Add explicit reduction ownership for score max, denominator, and PV accumulation.

Gate:

- Structural artifact reports lane map.
- Cross-lane or reduction lowering is visible in source/ISA attribution when intended.
- No scratch/spill regression beyond the stated resource budget.

Kill:

- If the scheduler can only express scalar per-lane duplicate work, record `SEARCH_BLOCKED_BY_CODEGEN`.
- If cross-lane lowering works but W==D remains flat and attribution shows reduction is not material, stop this lane.

### P4: Add packed dot and memory shape

Goal: close the score-path gap without collapsing decode parallelism.

Build:

- Attach generated/search-owned `v_dot2` or equivalent packed fp16 dot behavior to the tile.
- Preserve whole-cache memory shape.
- Add LDS only if the candidate keeps occupancy and avoids the known decode LDS trap.

Gate:

- ISA/resource audit sees intended dot primitive or records why it cannot be emitted.
- Workgroup count and occupancy proxy remain compatible with T=1 decode.
- W==D improves over A2 at least at one ctx without regressing the others beyond spread.

Kill:

- If `v_dot2` appears but does not transfer, record `NO_TRANSFER` and do not keep stacking complexity blindly.
- If LDS staging slows the candidate by reducing occupancy or duplicating cache-served reads, reject LDS for this candidate.

### P5: Integrate combine lifecycle accounting

Goal: avoid repeating the tile-only proxy-win problem.

Build:

- Treat tile plus combine as one candidate lifecycle.
- Report split count, tile workgroups, combine workgroups, intermediate bytes, and per-ctx combine share.
- Keep separate programs if necessary, but the evaluator must score the whole lifecycle.

Gate:

- Whole lifecycle W==D is measured.
- Unknown bucket remains closed or explicitly explained.
- Combine does not erase a local tile win.

Kill:

- If tile local speedup does not transfer to W==D because combine/runtime dominates, reject the candidate even if the tile is faster.

### P6: BubbleBeam binding and promotion decision

Goal: only after the candidate is real, make it searchable/selectable.

Build:

- Bind the candidate to BubbleBeam with the manifest from P1.
- Mark owned route as fallback/reference.
- Add promotion metadata to benchmark artifacts.

Gate:

- BubbleBeam can select the generated candidate from the manifest.
- Generated route is token-correct.
- No owned attention route fires.
- No `E_49152`.
- W==D meets the owned-route threshold across ctx points.

Promotion verdict:

```text
DECODE_ATTENTION_ONLINE_PV_TILE_SEARCH_PROMOTABLE
```

Failure verdicts:

```text
DECODE_ATTENTION_ONLINE_PV_TILE_NO_TRANSFER
DECODE_ATTENTION_ONLINE_PV_TILE_BLOCKED_BY_CODEGEN
DECODE_ATTENTION_ONLINE_PV_TILE_BLOCKED_BY_RUNTIME
DECODE_ATTENTION_SEARCH_SPACE_INCOMPLETE
```

## Evidence bundle required

A complete result must include:

| Artifact | Purpose |
|---|---|
| Search-space manifest | Defines what was actually searchable |
| Route capture JSON | Proves which programs fired |
| Correctness sample | Proves token/sample match |
| `E_49152` check | Proves no hidden full-cache materialization |
| ISA/resource audit | Proves or blocks dot/reduction/LDS lowering |
| W==D ctx sweep | Final decode authority |
| Split-KV lifecycle economics | Prevents tile-only proxy wins |
| Result doc | Human-readable verdict and next action |

## Decision logic

Use this table to prevent ambiguous outcomes:

| Observation | Decision |
|---|---|
| Candidate cannot be represented in manifest | `SEARCH_SPACE_INCOMPLETE` |
| Candidate can be represented but renderer cannot emit required lowering | `SEARCH_BLOCKED_BY_CODEGEN` |
| Candidate emits but route falls back to owned | `SEARCH_BLOCKED_BY_RUNTIME` |
| Candidate is route-clean and correct but slower than A2 | `NO_TRANSFER` |
| Candidate beats A2 but not owned | continue only if resource attribution points to a concrete next primitive |
| Candidate meets owned W==D threshold | bind to BubbleBeam and prepare promotion |

## First executable task

Start with P1 before more codegen work:

```text
Create bench/qk-search-spaces/decode_attention_online_softmax_pv_tile_v1.json
and a small checker that validates the manifest contains the full primitive boundary.
```

Reason: if the manifest cannot honestly represent the owned route's primitive boundary, more generated experiments will keep producing false search failures.
