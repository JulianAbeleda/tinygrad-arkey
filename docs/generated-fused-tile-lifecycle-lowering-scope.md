# Generated fused tile lifecycle lowering scope

## Goal

Make the generated code path capable of representing and lowering a fused tile lifecycle with nested reductions, recurrence state, local/cooperative output ownership, and compact metadata stores.

This is the lower-level blocker discovered while trying to build generated fused score + online-state + PV decode attention.

Current blocker verdict:

```text
FUSED_SCORE_STATE_PV_TILE_BLOCKED__MULTI_REDUCTION_STORE_SHAPE
```

Immediate goal verdict:

```text
FUSED_TILE_LIFECYCLE_LOWERING_ACCEPTS_NESTED_REDUCE_STATE_STORE
```

This is not a speed project yet. It is an expressibility/lowering project.

## Why this exists

The generated fused PV tile improved the catastrophic split x-lane route, but W==D still lost:

| ctx | owned baseline tok/s | fused PV tile tok/s | delta |
|---:|---:|---:|---:|
| 512 | 103.5 | 72.1 | -30.34% |
| 1024 | 101.8 | 70.1 | -31.14% |
| 4096 | 94.6 | 58.5 | -38.16% |

The next obvious step was fusing score + online state + PV. A target builder was added:

```python
flash_fused_score_state_pv_tile_whole_cache_kernel
```

The builder has the required shape, but tinygrad fails before execution during lowering/estimation:

```text
IndexError: pop from empty list
renderer/__init__.py -> Estimates.from_uops
```

So the wall moved below attention routing into UOp/codegen lifecycle representation.

## Minimal failing lifecycle shape

The failing generated shape is:

```text
GLOBAL axes:
  kvh, split

LOCAL axes:
  d

REDUCE axes:
  j/token
  e/dot

STATE:
  dot[g]
  acc[g]
  den[g]
  mx[g]

STORE:
  pout[(h, split, d)] where:
    d < Hd      -> acc[d]
    d == Hd     -> den/l
    d == Hd + 1 -> mx/m
```

Attention-specific meaning:

```text
score_reduce(e) inside token_loop(j)
online_update(m,l,acc[d])
compact_store(acc[d], l, m)
```

Generic compiler meaning:

```text
nested reduce + recurrence tuple + local output axis + compact side metadata store
```

## Required lower-level capability

| Capability | Why required |
|---|---|
| nested reduction scope ownership | q.k needs `e` reduction inside token recurrence `j` |
| recurrence state tuple | online softmax needs `(m,l,acc[d])` updated together |
| local/cooperative output axis | PV column `d` must be local/coalesced, not global scalar columns |
| compact metadata stores | output has data columns plus metadata columns `l,m` |
| scope-balanced lowering/estimation | current failure is an END-scope stack mismatch in `Estimates.from_uops` |
| route-safe custom kernel identity | later gates need a distinct generated program name |

## Non-goals for this stage

Do not solve these yet:

| Non-goal | Reason |
|---|---|
| W==D promotion | impossible until lowering works |
| LDS tuning | later performance layer |
| `v_dot2`/packed dot lowering | later performance layer |
| default route changes | this is a default-off codegen probe |
| owned/precompiled binary fallback | violates pure generated/search target |

## Candidate abstractions

### Option A: repair generic UOp scope lowering

Make existing UOp nested reductions/store groups lower correctly.

Pros:

| Pros |
|---|
| most generic |
| benefits future search spaces |
| least attention-specific |

Risks:

| Risks |
|---|
| touches core lowering/renderer behavior |
| easy to destabilize unrelated kernels |
| needs careful minimal reproduction |

### Option B: introduce `FusedTileLifecycle` helper abstraction

Add a constrained helper that emits a known-good UOp pattern for:

```text
nested_reduce -> recurrence_state -> local_axis_store
```

Pros:

| Pros |
|---|
| narrower surface |
| easier to gate |
| still generic enough for search if parameters are explicit |

Risks:

| Risks |
|---|
| may become attention-shaped if not parameterized |
| still depends on renderer accepting emitted UOps |

### Option C: attention-specific primitive

Add a direct generated attention tile op/builder with special lowering.

Pros:

| Pros |
|---|
| fastest route to decode result |
| clear correctness target |

Risks:

| Risks |
|---|
| least pure/searchable |
| repeats owned-kernel problem at another layer |
| not the preferred direction unless A/B fail |

Recommended path:

```text
A minimal reproduction first, then B if generic repair is too risky.
```

## Gates

| Gate | Requirement | Failure verdict |
|---|---|---|
| P0 scope gate | scope + blocker artifact exists | `FUSED_TILE_LIFECYCLE_SCOPE_INCOMPLETE` |
| P1 minimal repro | tiny standalone nested-reduce/state/local-store kernel reproduces or avoids failure | `FUSED_TILE_LIFECYCLE_REPRO_MISSING` |
| P2 classify lowering site | failure localized to estimate/render/scope balance | `FUSED_TILE_LIFECYCLE_BLOCKED__UNCLASSIFIED` |
| P3 expressibility fix | minimal repro compiles and numerically passes | `FUSED_TILE_LIFECYCLE_FAIL__MINIMAL_NUMERIC` |
| P4 attention builder retry | fused score-state-PV standalone gate passes | `FUSED_SCORE_STATE_PV_TILE_FAIL__NUMERIC` |
| P5 route gate | target route fires and old score/max absent | `FUSED_SCORE_STATE_PV_TILE_FAIL__ROUTE` |

## First executable artifact

Add:

```text
extra/qk_fused_tile_lifecycle_lowering_gate.py
bench/qk-fused-tile-lifecycle-lowering/latest.json
```

The first gate should:

1. Reference the attention blocker artifact.
2. Build a minimal reproduction of nested reduce + recurrence + local metadata store.
3. Run it if possible.
4. If it fails, classify whether it matches the known `Estimates.from_uops` scope-stack failure.
5. Emit a machine-readable verdict.

Expected first verdict:

```text
FUSED_TILE_LIFECYCLE_BLOCKED__ESTIMATE_SCOPE_STACK
```

or, if the minimal reproduction is too weak:

```text
FUSED_TILE_LIFECYCLE_REPRO_MISSING
```

## Success definition

This stage is complete when a minimal generic lifecycle kernel lowers and runs with numeric correctness:

```text
FUSED_TILE_LIFECYCLE_MINIMAL_NUMERIC_PASS
```

Only then should the decode attention builder be retried.
