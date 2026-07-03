> **⚠ SUPERSEDED (2026-06-21) — historical provenance only.** Current state lives in `docs/current-project-state-handoff-20260621.md` (+ `docs/README.md`). Do NOT treat this as authority. Many scripts/paths it references were removed in the active-surface reduction (`docs/perf-probe-active-surface-reduction-result-20260621.md`, 291 perf files deleted). Kept for history.

# Machine Search Path: Decode And Context

Date: 2026-06-16

Purpose: record the current performance state and the recommended path for a
principled machine-search layer. This is a handoff for later implementation,
not an active runtime contract.

> **Execution status (2026-06-16) — see `docs/amd-decode-banked-20260616.md`.**
> - **Scaffold built** (now ported to `boltbeam.search.spec`): the schema authority (rows, constraints,
>   accepted-policy records) this plan prescribes. ✅
> - **Track 1 (decode sequential tax): investigated → overlap GATED.** Profiled (72% GEMV / 28% non-GEMV,
>   single-queue sum); overlap is reclaimable (~+30%, HBM ~58% idle) but tinygrad's AMD backend has one
>   compute ring, so it needs a `[runtime]` 2nd-ring build first (`docs/amd-decode-two-queue-probe-...`).
>   Norm-fusion refuted.
> - **B3 (per-tensor bit-width) DONE** as the first real search run (historical tinygrad demotion search; BoltBeam now owns search-policy work) on the
>   scaffold): the quality-gated demotion search mapped the frontier and the gate rejected lm_head on
>   quality; in-pattern lever tapped at ~64 tok/s (`docs/amd-decode-demotion-search-...`).
> - **Track 2 (flash policy) / Track 3 (prefill LDS)**: not started. Decode is **banked**; these + the two
>   gated builds (overlap 2nd-ring, sub-4-bit kernel) are the open frontiers.

## Current Performance State

Decode is no longer blocked primarily by the standalone Q4_K GEMV kernel.

Current roofline:

| target | tok/s | percent of HBM peak | percent of llama.cpp |
|---|---:|---:|---:|
| HBM peak bandwidth floor | 183 | 100% | 173% |
| llama.cpp fixed hand-tuned path | 105.7 | 57% | 100% |
| tinygrad current default-on policy | 60.9 | 33% | 58% |
| tinygrad out-of-box, no flags | 55 | 30% | 52% |
| tinygrad before Q6_K work | 23 | 13% | 22% |

Progression:

| step | tok/s | percent of llama.cpp | output |
|---|---:|---:|---|
| baseline Q4_K primitive only | 23.1 | 22% | exact |
| Q6_K primitive coverage | 53.5 | 51% | byte-identical |
| Q6K_COVER_MORE for attn_v and lm_head | ~53.5 | 51% | byte-identical |
| ffn_down Q6 to Q4 demotion, B3 | 60.9 | 58% | free quality, dNLL -0.003 |

Standalone Q4_K GEMV:

| path | percent of HBM peak |
|---|---:|
| tinygrad int-dot, v_dot4 | 76% |
| tinygrad fp-dequant | 56% |
| llama.cpp end-to-end decode | 57% |

Interpretation: the kernel can already reach or exceed the effective bandwidth
level represented by llama.cpp end-to-end decode. The remaining decode gap is
therefore likely dominated by non-GEMV sequential tax, dispatch/runtime overhead,
unfused small operations, attention/KV/logits/sampling overhead, or policy
selection.

Per-model generated policy with shared storage:

| model | tinygrad tok/s | percent of llama.cpp |
|---|---:|---:|
| Qwen3-8B | 60.9 peak | 58% |
| Qwen3-14B | 40.55 | 61.6% |
| Qwen3-32B | 17.23 | 55.9% |

Long-context decode:

| context | dense SDPA tok/s | flash tok/s | speedup |
|---:|---:|---:|---:|
| 8 | 56.2 | 47.5 | 0.84x |
| 1024 | 27.6 | 34.3 | 1.24x |
| 3072 | 9.4 | 22.7 | 2.41x |

Interpretation: flash decode has a context crossover around 400 tokens and
should remain default-off or policy-controlled rather than globally enabled.

Decode and prefill are different problems:

| phase | tinygrad | llama.cpp | percent of llama.cpp |
|---|---:|---:|---:|
| decode | 60.9 tok/s | 105.7 tok/s | 58% |
| prefill | ~67 tok/s | ~3000 tok/s | ~2% |

Interpretation: prefill is the outlier. It needs LDS/cache-blocked codegen and
schedule search, not more QK primitive coverage alone.

## Strategic Conclusion

Machine search is feasible here, but only as bounded machine search.

The project already has:

| requirement | current state |
|---|---|
| stable benchmark targets | decode tok/s, prefill tok/s, context buckets |
| baselines | llama.cpp, HBM roofline, current tinygrad runs |
| search levers | Q4/Q6 policy, demotion, flash threshold, storage caps |
| quality gates | byte identity, dNLL, exactness checks |
| backend boundary | QK primitive path is AMD-guarded |
| artifact discipline | generated benchmark sprawl is ignored/backed up |
| precedent | generated policy already produced real wins |

The viable form is:

```text
search spec -> candidate generator -> isolated runner -> scorer -> accepted policy
```

The non-viable form is:

```text
one-off script mutates runtime until a benchmark improves
```

## Principle Mapping

The machine-search layer should follow the local coding principles directly.

| principle | machine-search implication |
|---|---|
| Centralize authority | one schema owns search rows, constraints, metrics, and accepted policies |
| Modularize execution | candidate generation, runner, scorer, and artifact writer are separate |
| Abstract for simplicity | runtime consumes a boring accepted-policy interface |
| Orthogonalize | decode, long-context decode, and prefill have separate search spaces |
| Encode invariants | backend, exactness, quality, memory cap, and ctx range are schema fields |
| Contain dangerous power | search runs isolated and cannot freely mutate runtime code |
| Test boundaries | parsers, scorers, schemas, and accepted-policy loading need tests |
| Reduce knowledge duplication | new experiment means a new row, not a new cloned script |

## Recommended Architecture

Create a single search authority rather than adding more standalone scripts.

Suggested components:

| component | role |
|---|---|
| search spec table | declares model, phase, backend, search space, constraints, objective |
| candidate generator | expands a bounded row into candidate policies |
| isolated runner | executes each candidate in a subprocess with explicit env ordering |
| scorer | compares throughput, exactness, dNLL, memory, and baseline regression |
| accepted policy table | durable runtime-consumable result |
| report writer | produces human-readable summaries without becoming runtime authority |

Suggested row shape:

```text
phase: decode | long_context_decode | prefill
model: qwen3_8b | qwen3_14b | qwen3_32b
op_scope: q4k_gemv | q6k_gemv | attention | ffn_down | lm_head | scheduler
backend: AMD
search_space: primitive_policy | demotion | flash_threshold | storage | schedule | lds_blocking
objective: tok_s | hbm_pct | serving_latency
constraints: exact_required | dNLL_epsilon | max_storage_mb | ctx_range | no_remote_execution
```

Accepted policy artifacts should contain at least:

```json
{
  "model": "qwen3-8b",
  "phase": "decode",
  "backend": "AMD",
  "ctx_range": [1, 399],
  "objective": "tok_s",
  "baseline_tok_s": 55.0,
  "accepted_tok_s": 60.9,
  "quality_gate": "dNLL <= baseline + epsilon",
  "exactness": "byte-identical or declared quality delta",
  "memory_cap_mb": null,
  "hardware": "required",
  "commit": "required"
}
```

## First Search Tracks

### Track 1: Decode Sequential Tax

Goal: explain and reduce the gap between strong standalone GEMV and weaker
end-to-end decode.

Measure:

```text
per-token op timeline
kernel launch count
non-GEMV op cost
attention overhead
KV/cache update cost
logits and sampling overhead
runtime/Python overhead
device sync count
storage movement
```

Candidate levers:

```text
fuse adjacent cheap ops
reduce host round trips
specialize decode path per model
cache policy decisions once per model
precompute primitive dispatch
reorder decode subgraph where legal
reduce dynamic shape/control overhead
```

Expected value: this is the most direct route from 58% of llama.cpp toward
competitive end-to-end decode.

### Track 2: Context-Aware Flash Policy

Goal: replace manual flash toggling with a searched context threshold policy.

Known state:

```text
ctx 8: flash loses
ctx 1024: flash wins
ctx 3072: flash wins strongly
approximate crossover: ctx 400
```

Search dimensions:

```text
model size
head count
head dim
batch size
GPU
dtype or quant policy
ctx bucket
```

Runtime result should be a table lookup, not a global default.

### Track 3: Prefill LDS/Cache Blocking

Goal: address the separate prefill gap.

This should come after the policy-search framework is clean because it touches
codegen-level power.

Search dimensions:

```text
tile sizes
LDS usage
vector widths
wave layout
block K/N/M choices
prefetching
accumulation layout
memory coalescing
attention blocking
```

Commit ownership should likely be `[codegen]`, not `[nn]`, unless the change is
strictly in model/decode policy.

## Can This Surpass llama.cpp?

Yes in targeted slices, not automatically across the whole stack.

Most plausible:

| target | surpass llama.cpp? | confidence |
|---|---|---|
| standalone Q4_K GEMV | already plausible | high |
| specific long-context decode bucket | plausible | medium-high |
| specific model/hardware decode policy | plausible | medium |
| general end-to-end decode | possible after sequential-tax work | medium-low |
| prefill soon | unlikely without major codegen work | low |

The current numbers mean llama.cpp is not an impossible ceiling. They do not
mean tinygrad will beat llama.cpp end-to-end without reducing the non-GEMV
sequential tax and separately fixing prefill.

## Explicit Non-Goals

Do not:

```text
make QK_GENERATED_POLICY a global default
merge decode and prefill search spaces too early
add more standalone one-off benchmark scripts
refactor frozen qk_flywheel_shadow.py
chase LOC reduction as the objective
let search write arbitrary runtime code
run BEAM or risky schedule search on Mac, TinyGPU, or remote paths
```

## Recommended Next Claude Task

Ask Claude to implement only the scaffold first.

Suggested prompt:

```text
Read structure/Development/coding-principles.md,
structure/Development/tinygrad-coding-overrides.md, and
structure/Development/machine-search-decode-context-plan-2026-06-16.md.

Implement the smallest table-driven machine-search scaffold for decode/context
policy work. Do not run hardware benchmarks and do not change runtime behavior.

Requirements:

1. Add one authoritative schema/spec location for search rows and accepted
   policy records.
2. Encode phase, model, backend, ctx range, objective, constraints, and quality
   gates as data, not ad hoc script logic.
3. Add loader/validator tests using synthetic data only.
4. Add no new benchmark artifacts except tiny durable fixtures required by tests.
5. Preserve AMD env-ordering invariants. Any future runner must set DEV/JIT/QK
   flags before importing tinygrad.
6. Do not make QK_GENERATED_POLICY global default.
7. Do not refactor qk_flywheel_shadow.py.
8. Use one owning commit prefix per commit.

Expected output: a small schema + tests + documentation that make future
decode/context machine search row-driven rather than script-driven.
```
