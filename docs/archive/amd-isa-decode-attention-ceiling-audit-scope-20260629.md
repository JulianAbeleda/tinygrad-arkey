# AMD ISA Decode Attention Ceiling Audit Scope - 2026-06-29

## Purpose

Build an audit that starts from the mathematical lower bound of decode attention and derives the rest of the loss stack:

```text
math floor -> owned hand ASM -> native ISA tile -> full decode wall
```

The goal is to decide whether further native attention-tile work is worth doing, or whether the search should move to
another decode component.

This is audit-only. Do not optimize kernels in this phase.

## Why This Audit Is Needed

The resource-lever loop is now mostly exhausted:

| lever | result |
|---|---|
| route binding / fallback | solved |
| token correctness | solved |
| grid parallelism | solved |
| hardware exp | solved |
| dynamic-S | solved |
| scheduler/waitcnt broad pass | small / not dominant |
| occupancy/LDS cut | refuted |
| register accumulators | real feature; DS `31 -> 9`; ctx512 +5.4%; ctx4096 neutral |
| LDS descriptor reclaim | resource-correct; W==D no movement |
| address scalarization | refuted/dead |
| local FMA/move cleanup | too small by R0 |

The remaining gap needs a different audit frame:

```text
How much speed is even available if attention became perfect?
How close is owned to the mathematical floor?
How much of native's overhead is real vs irrelevant to full decode wall?
```

## Current Known Points

Baseline N7 / RA / RL figures:

| route | ctx512 | ctx4096 |
|---|---:|---:|
| native dynamic-S baseline | 67.09 | 57.40 |
| native + reg accum RA4 | 70.72 | 56.73 |
| native + reg accum + LDS reclaim RL2 | 70.74 | 56.70 |
| owned | about 103 | about 94 |

N4 says native attention tile is the native-vs-owned kernel delta, but N3F/N4 also showed tile work is only a fraction of
full decode wall. That means a tile-local win can be real and still not move end-to-end W==D much.

## New Tool

Add:

```text
extra/amd_isa_decode_attention_ceiling_audit.py
```

Artifacts:

```text
bench/amd-isa-backend-decode-attention-ceiling/latest.json
bench/amd-isa-backend-decode-attention-ceiling/summary.md
bench/amd-isa-backend-decode-attention-ceiling/math_floor.json
bench/amd-isa-backend-decode-attention-ceiling/loss_stack.json
```

## Required Inputs

Reuse existing artifacts where available:

```text
bench/amd-isa-backend-phase-n4/latest.json
bench/amd-isa-backend-phase-n2b/latest.json
bench/amd-isa-backend-pc-source-trace/latest.json
bench/amd-isa-backend-regalloc-accum/ra4_latest.json
bench/amd-isa-backend-regalloc-accum-lds-reclaim/rl2_latest.json
bench/amd-isa-backend-phase-i/latest.json
bench/amd-isa-backend-phase-n0/latest.json
```

If an artifact is missing or stale, the audit should either re-run the relevant tool or record the input as unavailable.

## Math Floor Model

Compute a lower-bound-ish operation and byte model for decode attention at each context:

```text
ctx in {512, 1024, 2048, 4096}
head_dim
num_query_heads
num_kv_heads
kv_grouping
split length L
valid split count
dtype / qtype for K/V path
```

Required modeled work:

| component | required lower-bound model |
|---|---|
| score dot | minimum dot products / FMAs or dot2 equivalents for Q·K |
| K reads | minimum K bytes read from valid context |
| V reads | minimum V bytes read from valid context |
| online max | minimum max comparisons / updates |
| denominator | minimum exp2 + add/update operations |
| PV accumulation | minimum multiply/add work for P·V |
| partial writes | minimum partial state/output writes |
| combine/gmax | minimum split-combine work if split count > 1 |

This does not need to be a perfect theoretical model. It must be explicit about assumptions and conservative enough to
avoid overstating available headroom.

## Owned / Native Measurement Model

Compare the floor against measured and static data:

| route | required comparison |
|---|---|
| owned | static ISA mix, measured kernel time, whole-step owner share, W==D |
| native dynamic-S | static ISA mix, PMC counters, PC/source rows, measured kernel time, W==D |
| native + reg accum/RL | DS reduction, LDS/VGPR/group segment, W==D |

For each route, compute:

```text
static_over_floor_ratio
dynamic_counter_over_floor_ratio
kernel_time_share
full_decode_wall_share
owned_gap_to_floor
native_gap_to_floor
native_gap_to_owned
```

If exact dynamic instruction counts are unavailable, use the current N2B PMC counters and clearly label the value as
PMC-derived, not per-PC exact.

## Loss Stack

Emit a `loss_stack.json` with at least these layers:

| layer | meaning |
|---|---|
| `math_floor` | modeled minimum work |
| `owned_over_floor` | overhead owned pays above model |
| `native_shared_over_floor` | overhead both routes pay |
| `native_specific_over_owned` | native-only overhead vs owned |
| `full_decode_non_attention` | wall share outside attention tile |
| `max_gain_match_owned_tile` | max W==D if native attention matched owned |
| `max_gain_hit_math_floor_tile` | max W==D if native attention hit modeled floor |
| `max_gain_non_attention_zero` | sanity bound showing non-attention dominates if applicable |

The audit must answer:

```text
If native attention matched owned, what tok/s would we expect?
If native attention hit the math floor, what tok/s would we expect?
If all remaining full-decode non-attention kernels stayed unchanged, what is the hard W==D ceiling?
```

## Decision Rules

The audit should produce one of:

```text
AMD_ISA_ATTENTION_CEILING_PASS_CONTINUE_TILE
AMD_ISA_ATTENTION_CEILING_PASS_MOVE_TO_NON_ATTENTION
AMD_ISA_ATTENTION_CEILING_PASS_NEEDS_OWNED_ALGO_RETHINK
AMD_ISA_ATTENTION_CEILING_INCONCLUSIVE_MISSING_MEASUREMENTS
```

Decision criteria:

| condition | decision |
|---|---|
| matching owned tile would yield >=10% W==D improvement | continue attention tile work |
| hitting math floor would yield >=10% W==D but owned is far above floor too | consider owned-algorithm rethink / machine search beyond owned |
| matching owned tile yields <5% W==D | move to non-attention decode components |
| native-specific overhead is mostly refuted/resource exhausted | move to non-attention unless math floor says large untapped algorithmic gain remains |
| inputs are stale/missing | inconclusive and name exact missing measurement |

## Output Requirements

`latest.json` must include:

```json
{
  "verdict": "...",
  "contexts": [512, 1024, 2048, 4096],
  "math_floor": {},
  "owned_vs_floor": {},
  "native_vs_floor": {},
  "loss_stack": {},
  "max_gain": {
    "match_owned_tile": {},
    "hit_math_floor_tile": {},
    "move_non_attention": {}
  },
  "decision": {
    "next_target": "...",
    "reason": "..."
  },
  "input_artifacts": {}
}
```

`summary.md` must include:

- one table of current W==D;
- one table of floor/owned/native ratios;
- one loss-stack table;
- one decision table;
- explicit caveats.

## Non-Goals

- Do not optimize kernels.
- Do not change flags.
- Do not make native attention default.
- Do not treat estimated PC/source rows as measured per-PC stalls.
- Do not claim exact math optimality; this is a conservative floor model.

## Claude Prompt

Use this prompt verbatim:

```text
You are working in /home/ubuntu/tinygrad-arkey.

Read and follow:

  docs/archive/amd-isa-decode-attention-ceiling-audit-scope-20260629.md

Task:
Build an audit-only ceiling/decomposition tool:

  extra/amd_isa_decode_attention_ceiling_audit.py

Artifacts:

  bench/amd-isa-backend-decode-attention-ceiling/latest.json
  bench/amd-isa-backend-decode-attention-ceiling/summary.md
  bench/amd-isa-backend-decode-attention-ceiling/math_floor.json
  bench/amd-isa-backend-decode-attention-ceiling/loss_stack.json

Goal:
Start from the mathematical lower-bound model for decode attention and derive:

  math floor -> owned hand ASM -> native ISA tile -> full decode wall

The audit must answer:

  1. How far is owned above the modeled floor?
  2. How far is native above the modeled floor?
  3. How much native-specific overhead remains vs owned?
  4. What is the max W==D if native attention matched owned?
  5. What is the max W==D if native attention hit the modeled floor?
  6. Should search continue on attention tile or move to non-attention decode components?

Use existing artifacts when available:

  bench/amd-isa-backend-phase-n4/latest.json
  bench/amd-isa-backend-phase-n2b/latest.json
  bench/amd-isa-backend-pc-source-trace/latest.json
  bench/amd-isa-backend-regalloc-accum/ra4_latest.json
  bench/amd-isa-backend-regalloc-accum-lds-reclaim/rl2_latest.json
  bench/amd-isa-backend-phase-i/latest.json
  bench/amd-isa-backend-phase-n0/latest.json

If an input is missing or stale, record that explicitly instead of guessing.

Model at ctx512/1024/2048/4096:

  score dot work
  K/V bytes read
  online max/update work
  exp2/denominator work
  PV accumulation work
  partial writes
  gmax/combine work

Compare owned and native against the floor using static ISA, PMC counters, N4 per-kernel attribution, and W==D. Label estimates clearly. Do not claim hardware per-PC exactness unless that data exists.

Allowed verdicts:

  AMD_ISA_ATTENTION_CEILING_PASS_CONTINUE_TILE
  AMD_ISA_ATTENTION_CEILING_PASS_MOVE_TO_NON_ATTENTION
  AMD_ISA_ATTENTION_CEILING_PASS_NEEDS_OWNED_ALGO_RETHINK
  AMD_ISA_ATTENTION_CEILING_INCONCLUSIVE_MISSING_MEASUREMENTS

Do not optimize anything. This is an audit-only decision tool.
```


## Result

Verdict: `AMD_ISA_ATTENTION_CEILING_PASS_MOVE_TO_NON_ATTENTION`
(tool `extra/amd_isa_decode_attention_ceiling_audit.py`; artifacts under `bench/amd-isa-backend-decode-attention-ceiling/`).

Decode is **weight-memory-bound**. Streaming all Qwen3-8B-Q4_K_M weights (5.03 GB) once per token at the XTX peak
~960 GB/s = 5.24 ms/token, a ~191 tok/s ceiling (real ~80% bw ≈ 153). The attention KV-read floor is **< 1%** of that
(ctx4096: ~35 µs vs 5.24 ms, 0.67%).

| route | ctx512 | ctx4096 | % of weight floor (512/4096) |
|---|---:|---:|---|
| native (reg-accum + LDS reclaim) | 70.74 | 56.70 | 37% / 30% |
| owned | 103.5 | 94.4 | 54% / 50% |

Amdahl on the **measured** N3F dynamic-S cut (not eager GPU-compute, which overstates via no overlap) puts the attention
tile wall-share at ~10% @ctx512 and ~0 @ctx4096. So:

- Matching owned's tile → **+10.5% @ctx512 (borderline), +2.9% @ctx4096 (< 5%)**.
- Hitting the attention math floor → same bound (the floor is even cheaper than owned's tile).
- And it would now require an **owned-level algorithmic rewrite**, because every tile *resource* lever is exhausted or
  refuted (grid, hardware exp, dynamic-S, scheduler/waitcnt, occupancy/LDS, address scalarization, register
  accumulators, LDS reclaim, FMA/mov cleanup).

Decision: **move search to the non-attention FFN / weight-memory path.** The FFN/projection GEMVs dominate the wall, are
**shared and identical** between the native and owned routes (N4: `q4k_gemv` 7109 vs 7157), and sit at only ~50–54% of
the weight floor (~2× headroom) — versus attention's diminishing ~10% / ~0. The native AMD-ISA attention route is
complete, correct, and net-positive (~60–68% of owned) and is now at its practical ceiling; further attention-tile work
is low-leverage.

Caveats: peak HBM bw gives optimistic ceilings (real ~80%); the math floor is a conservative lower-bound work model;
tile wall-share is from the measured dynamic-S Amdahl, not hardware per-PC stalls (ATT per-PC is walled under HCQ).
