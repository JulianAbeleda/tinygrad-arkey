# Project North Star: Beat llama With Closed Lifecycle Search

Date: 2026-06-20

## Completion Definition

The project is complete when both conditions are true:

1. **Performance:** tinygrad beats the current llama.cpp reference on the target Qwen3-8B decode benchmark, under
   trusted W==D measurement, with correctness/quality gates intact.
2. **Method:** the win is not a one-off hand patch. It is produced or made repeatable by a **closed lifecycle
   machine-search system** that can reduce kernel count and choose fused routes inside a bounded, measurable domain.
3. **Clean execution repo:** the working system exists in a clean `tinygrad-v2` repo/workspace that contains only the
   code, docs, artifacts, and harnesses needed to execute and maintain the llama-beating lifecycle-search path.

In short:

```text
complete = beat llama + lifecycle-level machine search + clean tinygrad-v2 execution repo
```

A single benchmark win without the search system is not enough. A search system that cannot beat llama is not enough.
A research repo full of stale probes without a clean execution surface is not enough.

## Why This Is The End Goal

The last month changed the project shape:

- prefill moved from unclear/broken to llama-class through graph/lifecycle work;
- decode weight GEMV/MMVQ moved from suspected blocker to mostly solved in-model;
- host/runtime overhead was closed by W==D (`host-sync 0%`);
- current decode gap is now localized to attention lifecycle and unfused elementwise.

That means the remaining hard problem is not "find one faster kernel." It is:

```text
search over the decode lifecycle:
what becomes a kernel,
what gets fused,
what gets materialized,
what gets reused,
what gets routed by shape/context,
and what is safe by quality policy.
```

This is the point where machine search should move up a level.

## What "Closed Lifecycle Search" Means

Closed lifecycle search is not open-ended BEAM over arbitrary code. It is a bounded system:

| axis | closed domain |
|---|---|
| model | Qwen3-8B-Q4_K_M first |
| hardware | AMD gfx1100 / RX 7900 XTX first |
| phase | decode first; prefill as solved/reference |
| target roles | attention, FFN activation elementwise, q8/route policy where compatible |
| contexts | 512, 1024, 4096 |
| authority | full W==D tok/s for promotion |
| local timing | PROFILE GPU timestamps for attribution only |
| quality | greedy/dNLL/exactness gates as appropriate |
| output | route templates, fusion choices, kernel schedules, and policy thresholds |

The search object is a **route/lifecycle plan**, not only a kernel schedule.

Example route candidate:

```json
{
  "role": "decode_ffn",
  "template": "gate_up_activation_down",
  "fusion": ["silu_mul_into_down_input"],
  "layout": "down_consumer_native",
  "q8_compatible": true,
  "ctx_policy": [512, 1024, 4096],
  "schedule": {
    "kernel_count_target": "remove_E_49152",
    "local_gate_ms_recovered": 0.5
  }
}
```

Example attention candidate:

```json
{
  "role": "decode_attention",
  "template": "gqa_grouped_flash",
  "fused_stages": ["partial", "reduce_fixup", "softmax_stats"],
  "short_kv_specialization": true,
  "ctx_policy": [512, 1024, 4096],
  "kernel_count_target": "reduce_stat_kernels"
}
```

## Why Kernel Search Alone Is Not Enough

The latest timed attribution says:

| family | ctx1024 gap vs llama | implication |
|---|---:|---|
| attention | `+2.73 ms/tok` | lifecycle/reduce/stat overhead |
| elementwise | `+1.83 ms/tok` | missing fusion, especially FFN activation |
| weight-GEMV/MMVQ | `+0.41 ms/tok` baseline, negative in q8/ctx4096 | not the main target |
| host/runtime | `0.0%` host-sync | not the target |

Kernel search can make one existing kernel faster. Lifecycle search can delete kernels, fuse stages, avoid
materialization, or pick a different route by context. That is where the current measured gap lives.

## Required Search Stack

The intended system has five layers:

| layer | job |
|---|---|
| route search | choose the lifecycle template, e.g. current flash vs grouped flash vs short-KV path |
| fusion search | choose which graph boundaries disappear, e.g. FFN activation, attention stats, residual adds |
| layout search | choose intermediate layouts that let the next consumer avoid copies or standalone elementwise |
| kernel schedule search | tune the remaining kernels: tiles, waves, vectorization, LDS, waits |
| policy search | choose thresholds and defaults by context, VRAM, quality, and server/one-shot mode |

The evaluation ladder:

```text
generate candidate
-> structural checks
-> correctness/quality
-> local timed A/B
-> full W==D decode
-> policy/default decision
```

## Clean Repo / v2 Requirement

The current repo is the research ledger. It contains valuable history, refutations, one-off probes, stale scopes, and
artifact archaeology. That is useful for provenance, but it is not the right final execution surface.

The project must create a clean clone/workspace, tentatively:

```text
/home/ubuntu/tinygrad-v2
```

The v2 repo should be a refactored execution repo, not a history dump.

### v2 Goals

| goal | requirement |
|---|---|
| minimal execution surface | keep only files needed to run the winning decode/prefill paths, lifecycle search, gates, and docs |
| clean lifecycle-search system | route templates, candidate schemas, evaluators, structural checks, and promotion gates are first-class |
| reproducible benchmarks | W==D decode, llama comparison, quality gates, and policy tables are runnable from documented commands |
| provenance bridge | link back to the research repo/docs for refutations and historical evidence, but do not carry every stale probe forward |
| maintainability | remove abandoned experiment scripts, duplicated harnesses, stale default-policy docs, and dead route fragments |

### v2 Initial Keep List

Exact list should be generated by audit, but the expected kept classes are:

- core tinygrad files needed by `tinygrad.llm` and AMD execution;
- model route files for Qwen3/Q4_K/Q6_K decode and `PREFILL_V2`;
- shipped/default or supported opt-in routes:
  - `PREFILL_GRAPH_GEMM`;
  - `PREFILL_TC_ATTN`;
  - `PREFILL_CONCRETE_KV`;
  - q8 FFN opt-in if still policy-supported;
- lifecycle-search code:
  - route candidate schema;
  - attention/elementwise splitters;
  - A/B evaluators;
  - W==D promotion harnesses;
  - quality/dNLL/greedy gates;
- docs:
  - this north-star doc;
  - current benchmark index;
  - current prefill state;
  - current decode attribution;
  - current attention/elementwise scope;
  - short refutation index.

### v2 Drop Candidates

Do not blindly delete from the research repo. In v2, exclude or archive:

- stale Q6/MMVQ-first scopes refuted by timed attribution;
- old prefill contradiction docs superseded by Increment 0/Branch B;
- one-off `/tmp`-style harnesses copied into `extra/`;
- failed flash v1 as hot-path code, unless retained in an `archive/` or `references/` folder;
- old external Tensile shipping paths unless a policy explicitly accepts them;
- duplicated benchmark scripts whose only job is historical reproduction;
- dead q8 lifecycle probes that do not feed the current opt-in or attribution gates.

### v2 Migration Gates

Before calling v2 the active repo:

1. `bench/README.md` equivalent exists and names the current default, opt-in, and llama reference rows.
2. Full W==D decode benchmark runs in v2 and matches current repo within measurement noise.
3. Prefill Branch B / Increment 0 policy checks run in v2.
4. Lifecycle-search candidate generation runs in v2.
5. Attention and elementwise attribution splitters run in v2.
6. Quality gates pass.
7. The old repo remains as research provenance and is linked from v2 docs.

### Why v2 Is Part Of Completion

The end goal is not just to discover a fast path; it is to own a maintainable performance system. A clean v2 repo
turns the month of research into an executable product surface:

```text
research repo = what we learned
tinygrad-v2 = what we run, search, maintain, and ship
```

## Relationship To Existing Docs

This doc is the project-level completion condition. It should be read with:

- `docs/primitive-lifecycle-search-scope-20260619.md` — seed lifecycle candidate ledger and schema.
- `docs/decode-current-route-attribution-result-20260620.md` — latest timed reason to target attention/elementwise.
- `docs/decode-role-tensor-kernel-attribution-result-20260620.md` — lane ranking and explicit drops.
- `docs/decode-attention-elementwise-solution-scope-20260620.md` — immediate execution scope.
- `docs/prefill-increment0-shipped-result-20260620.md` — proof that lifecycle/integration can close a phase.

## Near-Term Milestones

1. **Attention split:** classify current attention cost into partial compute, reduce/fixup, softmax-stat, and other.
2. **Elementwise split:** confirm `E_49152_32_3` / FFN activation and residual/RoPE shares.
3. **First lifecycle candidate:** remove or fuse at least one material attention or FFN activation stage.
4. **Stacked decode route:** combine attention + elementwise + q8 where compatible.
5. **Closed-search prototype:** encode the successful route families as searchable templates with gates.
6. **tinygrad-v2 cutover:** clone/refactor into a clean execution repo with only relevant runtime, search, gate, and
   current-doc files.

## Success Gates

Performance gates:

- near-term: recover `>=2.5 ms/token` from attention + elementwise;
- medium-term: reach `>=80 tok/s` at ctx1024 with correctness intact;
- completion: beat llama.cpp reference under W==D on the target benchmark.

Search-system gates:

- candidates are generated from explicit route templates, not ad hoc scripts only;
- refutations prune the search space;
- every candidate names correctness, timing, and policy gates;
- the system can reproduce the winning route or regenerate an equivalent one after cache/artifact deletion;
- results are stored as machine-readable artifacts, not only prose.
- the clean v2 repo can run the winning path and search/gate loop without depending on stale research scripts.

## Non-Goals

- Do not chase another one-off kernel if it cannot be represented as a route template.
- Do not reopen Q6/MMVQ/host/q8 lifecycle without new timed evidence.
- Do not count a benchmark-only hack as project completion.
- Do not require the first version of lifecycle search to be fully general across all models and GPUs. It should be
  closed, correct, and effective first.
- Do not delete research provenance while creating v2. v2 is a clean execution clone, not a destructive cleanup of the
  historical repo.
