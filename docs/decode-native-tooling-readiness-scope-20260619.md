# Decode Native Tooling Readiness Scope

Date: 2026-06-19

Purpose: define exactly what tooling is still missing before native decode scheduler/renderer implementation is allowed.
This is a tooling scope, not a kernel scope. The current state is:

```text
ATT/HCQ visibility exists; timing authority exists; native scheduler feature attribution does not.
```

Until the attribution gap is closed, native decode scheduler/renderer work remains roadmap-only unless the project
explicitly funds broad AMD backend work without a bounded feature.

## Authority

Start from these documents:

- `gpu-performance-first-principles.md`: every kernel is limited by bytes, math, or overhead; measure the bucket before
  attacking it.
- `what-makes-a-performance-primitive-efficient-20260618.md`: a performance primitive includes math, layout,
  activation lifecycle, memory path, reductions, lowering, scheduling, graph routing, and model transfer.
- `decode-complete-tooling-result-20260619.md`: complete enough to prevent direct-output/reduce-fusion from being the
  wrong next build, but still has explicit gaps.
- `decode-native-mmvq-scheduler-renderer-full-scope-20260619.md`: native route start criteria require one attributed
  `>=30us` q8 feature, `>=5%` W==D projection, shared decode+prefill oracle movement, or explicit broad backend funding.
- `decode-n1-attribution-result-20260619.md`: `N1_COMPLETE_NO_N2_START`; largest bounded attribution is `14.087us`;
  `sqtt_decode_usable=false`.
- `amd-att-primitive-attribution-result-20260619.md`: ATT body attribution works on real HCQ primitive surfaces, but is
  not timing authority.

## Current Tooling Inventory

| Tooling layer | Current state | Enough for | Not enough for |
|---|---|---|---|
| Correctness/value gates | Available for shipped/research routes | reject wrong kernels and lossy q8 without dNLL | root-cause attribution |
| W==D / role timing | Available for promotion gates | decide if a candidate moves decode | explain which scheduler feature caused movement |
| HCQ program capture | Available | prove runtime/cache identity and role launch surface | quantify resource stalls |
| AQLprofile ATT replay through HCQ | Available | body-attribute real tinygrad kernels | timing, stall attribution, or feature budget by itself |
| In-model role join | Partial: Q4 `attn_q/o` full, Q6 surface fallback, `ffn_gate/up` missing ATT | close fallback/runtime identity hypotheses | explain q8 scheduler gap |
| llama/oracle launch contracts | Available | compare geometry, VGPR, SGPR, LDS, kernargs, instruction mix | prove which static delta matters |
| PMC/SQTT evidence | Partial: PMC runnable; old local SQTT decode failed; imported ATT now body-decodes | support visibility and static/dynamic hypotheses | timeline-grade scheduler/resource attribution |
| Feature attribution | Insufficient | below-gate pruning of load/wait/reduction standalone work | starting N2 implementation |

## What Is Missing

### M1 - Feature-Level Timing Attribution

We need a tool that converts trace/counter/static evidence into a table like:

```json
{
  "feature": "register_lifetime",
  "role": "ffn_gate/up",
  "evidence": ["counter or trace backed facts"],
  "movement_us": 37.2,
  "confidence": "timing_grade",
  "safe_first_patch": "one bounded renderer/scheduler change"
}
```

The current `n1_attribution.json` has only below-gate bounded rows and unknown scheduler/resource rows. Unknown is not
implementation authority.

### M2 - q8 `ffn_gate/up` Role-Joined Body Evidence

The q8 route is the only measured decode speed route and the clearest native scheduler oracle. The complete tooling
atlas still marks `ffn_gate/up` as `runtime_identity_only` / `ATT_MISSING`. We need role-joined ATT/PMC evidence for
the exact high-share q8/native surface, not just Q4 `attn_q/o` or Q6 fallback surfaces.

Required output:

- HCQ program names and launch geometry for the exact q8/native `ffn_gate/up` role;
- body trace metrics for the interval;
- static ISA/resource metadata for the same program binary;
- paired timing authority for the same role or full W==D route.

### M3 - Oracle-Diff Normalization

The oracle rows exist, but the tooling must normalize all rows into the same schema:

| Field | Required |
|---|---|
| role and shape | semantic role, in/out features, quant format |
| lifecycle | producer, consumer, reduce, glue, graph replay boundary |
| launch | grid, local, kernarg size, LDS, VGPR, SGPR, scratch |
| ISA | load widths, dot ops, waits, clauses, delays, reductions, stores |
| timing | standalone, role-local, W==D where available, trust label |
| counters/trace | PMU/ATT/SQTT metrics, with authority label |
| verdict | shipped/refuted/deferred/open/project-level |

Without this, static diffs such as `s_clause=3 vs 0` or `global_load_b128 vs b32` remain interesting observations, not
patch authority.

### M4 - Bucket Classification Per Role

Per the first-principles doc, every candidate must be assigned to one binding bucket before implementation:

- bytes: HBM/L2 bandwidth, coalescing, load width, cache hit/miss, bytes moved;
- math: dot/FMA issue, VALU utilization, dependency chains, instruction mix;
- overhead: launch, graph boundary, reduce/glue, synchronization, lifecycle round trips.

The missing tool should output one primary bucket and reject patches in the wrong bucket. Example: a wait scheduler
patch is not justified if the role is proven HBM transaction-bound and wait placement has sub-gate movement.

### M5 - Movement Budget And Amdahl Projection

Every proposed backend feature needs both local movement and model movement:

| Gate | Required |
|---|---:|
| q8 local attributed movement | `>=30us` to start N2 |
| q8 N2 proof movement | `>=25us` consumer improvement |
| high-share role-local movement | `>=10%` |
| W==D projected movement | `>=5%` for meaningful native decode work, `>=3%` for research flag |
| lossy q8 dNLL | `<=0.01` |

The tool must compute the Amdahl projection from role share and measured local movement. Packet counts alone are not
accepted.

### M6 - Confidence Labels

Every row must label its authority:

| Label | Meaning |
|---|---|
| `timing_grade` | interleaved role A/B or W==D timing with matching binary/role |
| `counter_grade` | PMU/SQTT/ATT evidence tied to the same timed interval |
| `static_grade` | ISA/resource diff only |
| `surface_grade` | direct surface fallback, valid for visibility but not promotion |
| `inferred` | hypothesis, not implementation authority |

N2 can start only from `timing_grade` plus at least `counter_grade` or a clearly bounded dynamic ablation. Static-grade
rows can prioritize tooling, not codegen changes.

## Tooling Readiness Definition

We have enough tooling to start native scheduler/renderer implementation when one of these is true:

1. `bench/qk-decode-native-tooling/feature_attribution.json` contains a `timing_grade` feature with `movement_us >= 30`
   on q8 `ffn_gate/up`, and the feature names one bounded implementation surface.
2. The same artifact projects `>=5%` W==D movement from a role-local measured change and passes the timing policy.
3. A shared decode+prefill backend feature has independent oracle movement in both domains and a concrete first patch.
4. The project explicitly records that it is funding broad AMD backend scheduler work without bounded attribution.

If none is true, the correct state is:

```text
tooling incomplete for native implementation; continue attribution/tooling only.
```

## Proposed Artifact Schema

Create `bench/qk-decode-native-tooling/readiness.json`:

```json
{
  "verdict": "TOOLING_NOT_READY|TOOLING_READY_FOR_N2|BROAD_BACKEND_ACCEPTED",
  "roles": [],
  "oracles": [],
  "bucket_classification": [],
  "feature_attribution": [],
  "start_gate": {
    "n2_candidate_count": 0,
    "max_timing_grade_movement_us": 0,
    "max_projected_wd_pct": 0,
    "missing": []
  }
}
```

Create `bench/qk-decode-native-tooling/feature_attribution.json`:

```json
{
  "feature": "s_clause_s_delay_alu",
  "role": "ffn_gate/up",
  "bucket": "math|bytes|overhead",
  "movement_us": null,
  "authority": "static_grade",
  "evidence": [],
  "implementation_surface": null,
  "decision": "tooling_only|start_N2|closed|project_level"
}
```

## Execution Plan

### DTR-0 - Freeze Existing Evidence

Build `readiness.json` from existing artifacts only:

- `bench/qk-decode-complete-tooling/result.json`;
- `bench/q8-ffn-amd-scheduler-project/oracle_contract.json`;
- `bench/q8-ffn-amd-scheduler-project/n1_attribution.json`;
- `bench/amd-scheduler-tooling-backend/r1p2_hcq_replay.json`;
- `bench/qk-att-primitive-atlas/result.json`;
- `bench/qk-att-inmodel-role-join/*.json`.

Gate: artifact says `TOOLING_NOT_READY` and names the exact missing rows. If it says ready from current evidence, that
is a bug unless it names a `>=30us` timing-grade feature.

### DTR-1 - Fill q8 `ffn_gate/up` Role Join

Extend `extra/qk_att_inmodel_role_join.py` or add a small wrapper to trace the exact q8/native `ffn_gate/up` role.

Gate:

- program identity captured;
- body trace present;
- matching static ISA/resource row linked;
- timing authority linked or explicitly missing.

Kill:

- if full role capture is impossible, record why and whether a surface fallback is equivalent enough for visibility
  only. Do not promote timing from fallback.

### DTR-2 - Counter/Trace-To-Feature Join

Create a table-driven joiner that aligns:

- timed role interval;
- program binary hash;
- ISA/resource metadata;
- ATT/PMC/SQTT metrics;
- oracle row;
- candidate feature labels.

Gate: every candidate feature has evidence and an authority label. Unknown scheduler/resource rows stay unknown unless
the tool ties them to timing or counters.

### DTR-3 - Dynamic Ablation Matrix

For each candidate feature that is statically different, run or reuse a bounded ablation:

- load shape/coalescing;
- wait grouping;
- reduction topology;
- instruction order / scheduler markers;
- VGPR/register lifetime where controllable;
- graph/lifecycle boundary.

Gate: at least one ablation or counter-backed attribution reaches `>=30us`, or all rows are closed/below gate.

### DTR-4 - Readiness Decision

Emit:

- `docs/decode-native-tooling-readiness-result-20260619.md`;
- `bench/qk-decode-native-tooling/readiness.json`;
- `bench/qk-decode-native-tooling/feature_attribution.json`.

Outcomes:

| Outcome | Meaning |
|---|---|
| `TOOLING_READY_FOR_N2` | one bounded feature clears start criteria |
| `TOOLING_NOT_READY` | keep building attribution tooling; no codegen patch |
| `ROADMAP_ONLY` | tooling is enough to show no bounded feature exists |
| `BROAD_BACKEND_ACCEPTED` | project explicitly funds backend work without bounded attribution |

## Non-Goals

- Do not add a scheduler/renderer patch in this scope.
- Do not treat ATT packet count as a speed metric.
- Do not reopen imported Q4 routing as a speed route.
- Do not build reduce/glue fusion unless the new timing-grade ledger clears the existing gate.
- Do not rerun old env knob searches.

## Done Criteria

This tooling scope is done when a reader can answer all of the following from one artifact directory:

1. Which decode roles still matter by Amdahl share?
2. For each role, is the binding bucket bytes, math, or overhead?
3. Which oracle row is the target, and what exact contract differs?
4. Which feature has measured or attributed movement, and how much?
5. Is the evidence timing-grade, counter-grade, static-grade, surface-grade, or inferred?
6. Does any feature clear the native implementation start gate?
7. If not, what exact tooling row is missing next?

Until those answers exist, the native scheduler/renderer project is not ready for implementation.
