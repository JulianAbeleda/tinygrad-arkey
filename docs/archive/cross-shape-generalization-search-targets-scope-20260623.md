# Cross-Shape / Generalization Search Targets — Scope (2026-06-23)

## Mission
Decode and prefill are at/above llama.cpp on **one** validated path (Qwen3-8B-Q4_K_M, gfx1100, B=1). Machine search
should eventually generalize beyond it — but **only after each target has a baseline oracle and a correctness
harness**, never speculatively. Status: `CROSS_SHAPE_SEARCH_NEEDS_TARGETS` → this selects targets and defines the
unlock condition; execution is deferred per target until its oracle exists.

## Target axes (and current eligibility)
| axis | candidate targets | unlock condition | priority |
|---|---|---|---|
| model size | **14B**, 32B | model file present + per-shape baseline oracle + correctness harness; owner go (memory: no 14B/32B without ask) | high (owner-gated) |
| context | longer decode (ctx 8k+), varied prefill lengths | extend `WD_CTXS` + whole-prefill harness; oracle per ctx | medium |
| GPU | other RDNA/CDNA if available | hardware present + ISA tooling for that arch (the wrapper is AMD-only today) | low (hardware-gated) |
| quant / model | other Q4_K_M shapes, Q6/KV variants | shape inventory + route eligibility check + oracle | medium |

## Per-target required sections (before any search)
1. **Target selection** — which axis/value, and why.
2. **Baseline oracle** — the comparator: llama.cpp ref (if available) + the current tinygrad route's frozen W==D /
   whole-prefill on that target. **No stale baselines.**
3. **Route eligibility** — does the owned tile / graph-GEMM route even *fire* for the target's shape (B, Hq, Hkv, Hd,
   G, N/K divisibility)? If not, the result is a diagnosed "route doesn't fire" — itself the finding.
4. **Shape inventory** — the dominant GEMM/attention shapes at the target (the per-role table).
5. **Correctness harness** — byte-identical greedy (decode) / rel_rmse + greedy (GEMM) against the oracle.
6. **Authority benchmark** — W==D (decode) / whole-prefill synced (prefill); the only promotion authority.
7. **Bounded knobs** — the same enumerated schema as 8B (S, min_ctx, combine, tile constants, per-shape GEMM config).
8. **Expected cost** — model load + per-candidate gate time × grid.
9. **Stop rules** — no oracle → stop; route doesn't fire and can't be cheaply made to → record + stop.

## The generalization question worth searching first
The **prefill per-shape GEMM config** generalizes most naturally: the kv_proj WG-starvation fix
(`out_f≤1024 → BN64`) was a hand-picked per-shape config. Across 14B/32B the role shapes change (more heads, larger
FFN), so the **per-shape tile-config map** is exactly the kind of bounded, oracle-checkable space machine search is
good at — *once a 14B baseline oracle exists*. That is the recommended first cross-shape search, gated on the owner
authorizing a 14B target and a baseline being built.

## Unlock condition (all must hold)
- target model/data available;
- baseline oracle built (frozen, contract-stamped);
- correctness harness exists;
- current route works **or** fails for a diagnosed reason;
- bounded knobs known.

## Verdicts
- `CROSS_SHAPE_TARGETS_SELECTED` (this doc) → per-target `CROSS_SHAPE_NEEDS_BASELINES` until oracles exist →
  `CROSS_SHAPE_SEARCH_READY` once a target's oracle + harness are built → `CROSS_SHAPE_DEFERRED` if owner declines.

## Current state
`CROSS_SHAPE_DEFERRED` pending owner authorization — per the standing rule, **no 14B/32B without an explicit ask**.
The path is now mapped; pulling the trigger is a target-selection + oracle-build task, not a blind search.
