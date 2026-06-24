# Decode Native Tooling Pass Scope

Date: 2026-06-19

Purpose: exhaustively define what remains for the decode-native tooling verdict to pass. "Pass" means the tooling can
make a final implementation decision, not necessarily that native scheduler/renderer implementation starts.

Current verdict:

```text
TOOLING_NOT_READY_FOR_N2
```

Current state after `decode-native-tooling-completion-result-20260619.md`:

- q8/native `ffn_gate` and `ffn_up` in-model body visibility exists.
- The exact default native program is `q4k_gemv_partial_12288_4096_1`, hash `236fd9e8841b577f`.
- The native-to-oracle q8 gap is still `73.109us`.
- Known bounded ablations are below gate: load shape `14.087us`, wait grouping `0.837us`, reduction topology
  `13.305us`.
- The unresolved bucket is scheduler/resource behavior: `s_clause` / `s_delay_alu`, instruction order, register
  lifetime, occupancy/resource scheduling, or a compound backend effect.

## Definition Of A Passing Tooling Verdict

The tooling passes when it can emit one of these final outcomes:

| Outcome | Meaning |
|---|---|
| `TOOLING_READY_FOR_N2` | one bounded feature has timing/counter-grade `>=30us` movement and names an implementation surface |
| `ROADMAP_ONLY` | tooling is strong enough to prove no bounded feature clears the gate; native work is broad backend only |
| `BROAD_BACKEND_ACCEPTED` | project explicitly accepts broad AMD backend scheduler work without bounded attribution |

The tooling does **not** pass if it merely says:

- SQTT decode failed;
- PMC captured bytes exist;
- ATT body packets exist;
- static ISA diffs exist;
- the gap is probably scheduler/resource behavior.

Those are current-state observations, not final tooling authority.

## The Remaining Missing Evidence

### P1 - Counter-Grade q8 Role Metrics

Need parsed counters for the exact q8/native role or a justified equivalent surface.

Existing status:

- `PROFILE=1 PMC=1` runs.
- PMC events exist:
  - `SQ_BUSY_CYCLES`
  - `SQ_INSTS_VALU`
  - `SQ_INSTS_SALU`
  - `SQC_LDS_IDX_ACTIVE`
  - `SQC_LDS_BANK_CONFLICT`
  - `GRBM_GUI_ACTIVE`
  - `GL2C_HIT`
  - `GL2C_MISS`
- Current artifacts record event names and blob layouts, but not decoded counter values tied to a feature verdict.

Required output:

- `bench/qk-decode-native-tooling/pmc_decode.json`

Required rows:

| Row | Required content |
|---|---|
| raw decode | per-event counter values from PMC blobs, or explicit proof the blob format cannot be decoded locally |
| normalization | per-kernel or per-wave normalized rates where possible |
| bucket | bytes/math/overhead classification from counters |
| feature link | map counter pattern to candidate features: load shape, wait schedule, register/resource, scheduler markers |
| authority | `counter_grade` if decoded; `blocked_counter_decode` if not |

Pass gate:

- decoded values identify one feature with plausible `>=30us` movement; or
- decoded values show no bounded feature can explain the gap, supporting `ROADMAP_ONLY`.

Kill gate:

- PMC blobs cannot be decoded into values and no external parser path is available. Then pass must come from SQTT or
  dynamic ablations, not PMC.

### P2 - SQTT / ATT Timeline Attribution

Need instruction/timeline attribution for the scheduler/resource rows.

Existing status:

- tinygrad SQTT capture is runnable.
- local tinygrad RDNA3 SQTT decode fails on q8 blobs with:

```text
ValueError('unknown cdna format word=0xf4080100')
```

- imported AQLprofile ATT replay works and decodes body packets for HCQ kernels.
- ATT packet counts are visibility only, not timing.

Acceptable paths:

| Path | Description | Pass condition |
|---|---|---|
| local SQTT decoder repair | teach `tinygrad.renderer.amd.sqtt` the RDNA3 packet format seen in q8 blobs | q8 SQTT maps instructions for `q4k_gemv_partial_12288_4096_1` |
| external decoder bridge | export tinygrad SQTT/ATT to a ROCprofiler/RGP-compatible decoder | decoded instruction timeline joins to program hash and role |
| AQLprofile ATT role interval extension | use the working AQLprofile ATT path plus instruction/resource summaries where possible | emits feature-level evidence beyond packet counts |

Required output:

- `bench/qk-decode-native-tooling/timeline_attribution.json`

Required rows:

| Row | Required content |
|---|---|
| program join | role, program name, hash, launch geometry |
| decoded timeline | instruction groups or stall/resource events |
| scheduler markers | evidence for or against `s_clause` / `s_delay_alu` value |
| register/resource | evidence for or against VGPR/occupancy/resource model as dominant |
| movement | feature movement estimate or explicit "unattributed compound" |

Pass gate:

- timeline evidence attributes `>=30us` to one bounded feature; or
- timeline evidence proves the movement is compound/backend-level, supporting `ROADMAP_ONLY`.

Kill gate:

- no local or external timeline decoder can produce q8 role-level instruction/resource attribution. Then pass must come
  from P1/P3/P4, or the result remains `TOOLING_NOT_READY`.

### P3 - Same-Interval Role Timing Join

Need timing that matches the captured role/program interval closely enough to use with counters/traces.

Existing status:

- q8 oracle timings exist from prior standalone artifacts.
- `ffn_gate` / `ffn_up` ATT captures include `target_wall_ms`, but this wall time includes ATT/profiling overhead and
  is not promotion timing.
- Complete tooling policy requires role-local same-process interleaved A/B or W==D for timing authority.

Required output:

- `bench/qk-decode-native-tooling/role_timing_join.json`

Required measurements:

| Measurement | Purpose |
|---|---|
| native `ffn_gate` role-local timing | current in-model native surface |
| native `ffn_up` role-local timing | second consumer, same binary |
| q8/native standalone authority | existing q8 native baseline |
| q8 artifact oracle authority | target timing |
| timing/tracing alignment | prove the timed binary/hash is the traced binary/hash |

Pass gate:

- same binary/hash joins timing with trace/counter evidence; and
- movement projection is computable.

Kill gate:

- only ATT/profiler-wall timings exist. Then the row remains visibility-only.

### P4 - Feature-Level Dynamic Ablations

Need to price the still-unattributed scheduler/resource rows, or prove they are too compound to own as bounded patches.

Existing below-gate ablations:

| Feature | Movement |
|---|---:|
| load shape/coalescing | `14.087us` |
| wait grouping | `0.837us` |
| reduction topology | `13.305us` |
| dot4 instruction selection | `0us`, already matched |

Remaining ablations to scope:

| Feature | Question | Required evidence |
|---|---|---|
| `scheduler_markers` | do `s_clause` / `s_delay_alu` explain a large part of the gap? | controlled variant or counter/timeline evidence |
| `instruction_order` | does load/dot/reduce order, with same instruction multiset, move time? | variant with same semantics and isolated order change |
| `register_lifetime` | does VGPR/live range/occupancy explain gap? | resource metadata plus timing variant or compiler control |
| `resource_descriptor` | does launch/resource contract differ in a timing-relevant way? | same binary under varied resource descriptors, or proof not controllable |
| `compound_scheduler` | are single-feature variants too weak but combined schedule is large? | artifact labels compound and routes to broad backend, not N2 |

Required output:

- `bench/qk-decode-native-tooling/scheduler_ablation_scope.json`
- updated `bench/qk-decode-native-tooling/ablation_matrix.json`

Pass gate:

- one single-feature ablation reaches `>=30us`; or
- all feasible single-feature ablations are below gate/uncontrollable, supporting `ROADMAP_ONLY`.

Kill gate:

- an ablation changes multiple features and moves. Label `compound`; do not start a single-feature N2 from it.

### P5 - Amdahl / W==D Projection

Need to convert feature movement into model movement.

Existing status:

- q8 research route has W==D movement: `1.05-1.06x`.
- native feature movement projections are currently `0%` because no feature clears gate.

Required output:

- `bench/qk-decode-native-tooling/wd_projection.json`

Required rows:

| Row | Required content |
|---|---|
| role share | fraction of decode time affected by role |
| local movement | feature-level microseconds |
| graph/lifecycle cost | producer, consumer, glue, graph replay boundary |
| projected W==D | expected model speedup |
| confidence | timing/counter/static/inferred |

Pass gate:

- feature projects `>=5%` W==D for meaningful native work, or `>=3%` for research-flag-only work.

Kill gate:

- feature moves locally but projects below W==D gate. Close as local-only.

## Exhaustive Execution Plan

### P0 - Regenerate Current Readiness

Command:

```bash
python3 extra/qk_decode_native_tooling_readiness.py
```

Expected current output:

```text
TOOLING_NOT_READY
missing:
- timing-grade feature attribution >=30us
- counter/timing join that converts scheduler-resource unknowns into a bounded feature
```

### P1 - Decode PMC Blobs Or Prove Blocked

Implement `extra/qk_decode_native_pmc_decode.py`.

Inputs:

- `bench/q8-ffn-amd-scheduler-project/pmu_sqtt_evidence.json`
- `bench/q8-ffn-amd-scheduler-project/pmu_sqtt_pmc_q8_gateup_full.json`
- `bench/q8-ffn-dynamic-scheduler-observability/pmc_q8_gateup_full.json`

Outputs:

- `bench/qk-decode-native-tooling/pmc_decode.json`

Acceptance:

- if counters decode: update feature attribution with `counter_grade` rows;
- if counters do not decode: record exact blob-format blocker and move to P2.

### P2 - Repair Or Bridge SQTT Timeline Decode

Implement one of:

- `extra/qk_decode_native_sqtt_decode_probe.py` for local decoder repair;
- or `extra/qk_decode_native_external_trace_bridge.py` for external decoder export/import.

Inputs:

- SQTT blobs from `bench/q8-ffn-amd-scheduler-project/pmu_sqtt_evidence.json`;
- program lib from profile events;
- role hash from `bench/qk-att-inmodel-role-join/ffn_gate.json`.

Outputs:

- `bench/qk-decode-native-tooling/timeline_attribution.json`

Acceptance:

- decoded q8 role timeline maps instructions to feature rows; or
- result proves decoder path is blocked and names what external dependency/tool is missing.

### P3 - Same-Binary Timing Join

Implement `extra/qk_decode_native_role_timing_join.py`.

Inputs:

- `bench/qk-att-inmodel-role-join/ffn_gate.json`
- `bench/qk-att-inmodel-role-join/ffn_up.json`
- q8 oracle/native timing artifacts

Outputs:

- `bench/qk-decode-native-tooling/role_timing_join.json`

Acceptance:

- timing rows link to program hash `236fd9e8841b577f` or clearly document why only proxy timing exists.

### P4 - Scheduler Ablation Decision

Implement `extra/qk_decode_native_scheduler_ablation_scope.py`.

Inputs:

- `feature_join.json`
- `pmc_decode.json`
- `timeline_attribution.json`
- existing dynamic scheduler observability artifacts

Outputs:

- `bench/qk-decode-native-tooling/scheduler_ablation_scope.json`
- updated `ablation_matrix.json`

Acceptance:

- each remaining scheduler/resource feature is one of:
  - `start_N2`;
  - `closed_below_gate`;
  - `compound_project_level`;
  - `blocked_by_missing_counter_decode`;
  - `blocked_by_missing_timeline_decode`.

### P5 - Final Readiness Result

Update `extra/qk_decode_native_tooling_readiness.py` to consume P1-P4 outputs.

Emit:

- `docs/decode-native-tooling-pass-result-20260619.md`
- updated `readiness.json`
- updated `feature_attribution.json`

Allowed final verdicts:

- `TOOLING_READY_FOR_N2`
- `ROADMAP_ONLY`
- `BROAD_BACKEND_ACCEPTED`
- `TOOLING_NOT_READY`

`TOOLING_NOT_READY` is allowed only if an external blocker remains, such as no available RDNA3 SQTT decoder and no
usable PMC blob parser.

## Exact Start Conditions For N2

Native implementation can start only if all are true:

1. Feature row authority is `timing_grade` plus `counter_grade`, or timing-grade plus an isolated dynamic ablation.
2. `movement_us >= 30` for q8 consumer/lifecycle, or projected W==D `>=5%`.
3. Feature maps to one bounded implementation surface.
4. Correctness and dNLL gates are defined.
5. The feature is not labeled `compound`.

Examples that pass:

- `register_lifetime`: counter/timeline proves occupancy/VGPR policy accounts for `>=30us`, and a bounded allocator
  experiment is named.
- `scheduler_markers`: controlled `s_clause` / `s_delay_alu` placement moves `>=30us` with no other changes.
- `activation_lifecycle`: same-binary timing plus W==D projection shows producer/consumer boundary removal reaches gate.

Examples that do not pass:

- static diff shows oracle has `s_delay_alu` and tinygrad does not;
- ATT body packet counts differ;
- PMU profile ran but counters are not decoded;
- SQTT blob exists but decoder fails;
- combined handwritten artifact is faster but cannot be decomposed into one bounded native feature.

## Exact Conditions For ROADMAP_ONLY

The tooling can pass as `ROADMAP_ONLY` if:

1. q8 `ffn_gate/up` role evidence exists;
2. PMC and/or SQTT paths either decode or are proven unavailable;
3. all feasible single-feature ablations are below `30us`;
4. remaining movement is labeled `compound_scheduler` or `broad_backend`;
5. the result doc explicitly says no bounded N2 patch is allowed.

This is a successful tooling outcome: it prevents wasted native implementation work.

## Blocker Taxonomy

Every non-ready row must use one of these blockers. Do not write a free-form "blocked" verdict without one of these
labels.

| Blocker | Meaning | Next action |
|---|---|---|
| `blocked_counter_decode` | PMC blobs exist but local code cannot decode counter values | implement/parser audit or mark PMC unavailable |
| `blocked_timeline_decode` | SQTT/ATT timeline exists but instruction/resource decode is unavailable | local decoder repair or external decoder bridge |
| `blocked_same_binary_timing` | role evidence exists but timing cannot be joined to same program/hash | add role-local timing join or label proxy-only |
| `blocked_uncontrollable_feature` | feature cannot be isolated by current compiler/runtime knobs | classify as compound/project-level |
| `blocked_external_dependency` | required ROCm/RGP/decoder tool is absent or incompatible | record package/tool/version needed |
| `blocked_hardware_state` | AMD device state/OOM/other process prevents rerun | record process/device state and avoid changing conclusions |
| `closed_below_gate` | measured feature movement is below start threshold | do not reopen as N2 |
| `compound_project_level` | movement exists only as a multi-feature/backend effect | roadmap/broad-backend only |

## Required Artifact Schemas

### `pmc_decode.json`

```json
{
  "schema": "decode_native_pmc_decode_v1",
  "verdict": "PASS_COUNTER_GRADE|BLOCKED_COUNTER_DECODE|NO_USEFUL_COUNTER_SIGNAL",
  "inputs": [],
  "program": {
    "name": "q8_b2b_fullrow_reduce",
    "hash_or_tag": null
  },
  "events": [
    {
      "name": "SQ_BUSY_CYCLES",
      "raw_values": [],
      "decoded": false,
      "unit": "cycles|count|unknown"
    }
  ],
  "derived": {
    "valu_per_busy_cycle": null,
    "salu_per_busy_cycle": null,
    "l2_hit_rate": null,
    "lds_bank_conflict_rate": null
  },
  "feature_implications": [
    {
      "feature": "register_lifetime",
      "authority": "counter_grade|blocked_counter_decode",
      "movement_us": null,
      "decision": "start_N2|closed_below_gate|compound_project_level|blocked_counter_decode"
    }
  ]
}
```

### `timeline_attribution.json`

```json
{
  "schema": "decode_native_timeline_attribution_v1",
  "verdict": "PASS_TIMELINE_ATTRIBUTION|BLOCKED_TIMELINE_DECODE|NO_SINGLE_FEATURE",
  "program_join": {
    "role": "ffn_gate/up",
    "program_name": "q4k_gemv_partial_12288_4096_1",
    "lib_sha16": "236fd9e8841b577f"
  },
  "decoder": {
    "path": "local_sqtt|external_decoder|aqlprofile_att",
    "ok": false,
    "error": null
  },
  "timeline": [],
  "feature_implications": []
}
```

### `role_timing_join.json`

```json
{
  "schema": "decode_native_role_timing_join_v1",
  "verdict": "PASS_SAME_BINARY_TIMING|PROXY_ONLY|BLOCKED_SAME_BINARY_TIMING",
  "rows": [
    {
      "role": "ffn_gate",
      "program_name": "q4k_gemv_partial_12288_4096_1",
      "lib_sha16": "236fd9e8841b577f",
      "timing_us": null,
      "timing_authority": "same_process_interleaved|standalone_proxy|att_wall_not_authority",
      "usable_for_projection": false
    }
  ]
}
```

### `scheduler_ablation_scope.json`

```json
{
  "schema": "decode_native_scheduler_ablation_scope_v1",
  "verdict": "START_N2|ROADMAP_ONLY|TOOLING_NOT_READY",
  "features": [
    {
      "feature": "scheduler_markers",
      "isolation_possible": false,
      "movement_us": null,
      "authority": "timing_grade|counter_grade|static_grade|inferred",
      "decision": "start_N2|closed_below_gate|compound_project_level|blocked_uncontrollable_feature"
    }
  ]
}
```

### `wd_projection.json`

```json
{
  "schema": "decode_native_wd_projection_v1",
  "verdict": "PASS_WD_GATE|BELOW_WD_GATE|NO_PROJECTABLE_FEATURE",
  "rows": [
    {
      "feature": "register_lifetime",
      "local_movement_us": null,
      "affected_role_share": null,
      "projected_wd_pct": null,
      "confidence": "timing_grade|counter_grade|static_grade|inferred",
      "decision": "start_N2|research_only|closed_local_only|no_projection"
    }
  ]
}
```

## Final Command Sequence

The full pass attempt should run in this order:

```bash
python3 extra/qk_decode_native_tooling_readiness.py
python3 extra/qk_decode_native_pmc_decode.py
python3 extra/qk_decode_native_sqtt_decode_probe.py
python3 extra/qk_decode_native_role_timing_join.py
python3 extra/qk_decode_native_scheduler_ablation_scope.py
python3 extra/qk_decode_native_tooling_readiness.py
```

If an external decoder path is chosen instead of local SQTT repair, replace the second SQTT command with:

```bash
python3 extra/qk_decode_native_external_trace_bridge.py
```

Each command must be safe to rerun and must preserve prior artifacts unless it is intentionally regenerating its own
output path.

## Implementation Boundaries

Allowed in this pass:

- new `extra/qk_decode_native_*` tooling scripts;
- JSON artifacts under `bench/qk-decode-native-tooling/`;
- result docs under `docs/`;
- small read-only adapters around existing profiling artifacts.

Not allowed in this pass:

- renderer/scheduler/codegen changes;
- changing q8 route behavior;
- making `Q8_FFN_HANDWRITTEN` default-on;
- changing model outputs or quality gates;
- rerunning destructive device cleanup to free VRAM without explicit user direction.

If AMD device state blocks a rerun, write `blocked_hardware_state` and preserve the last valid artifact. Do not
overwrite a passing role artifact with a later OOM failure.

## Current Best Guess

Based on existing evidence, the likely final passing outcome is:

```text
ROADMAP_ONLY
```

Reason:

- body-insensitive dynamic ladder already suggests no single small body feature owns the gap;
- load/wait/reduction ablations are below gate;
- resource metadata does not show a simple descriptor mismatch;
- remaining movement likely belongs to broad AMD scheduler/resource/codegen quality.

But this must still be proven by the P1-P4 tooling rows, not assumed.

## Done Criteria

This pass scope is done when a reader can open `bench/qk-decode-native-tooling/readiness.json` and see:

1. q8 role body evidence: present.
2. same-binary timing join: present or explicitly proxy-labeled.
3. PMC decode: decoded or formally blocked.
4. SQTT/timeline attribution: decoded or formally blocked.
5. feature attribution: every feature has authority and decision.
6. final verdict: `TOOLING_READY_FOR_N2`, `ROADMAP_ONLY`, `BROAD_BACKEND_ACCEPTED`, or externally blocked
   `TOOLING_NOT_READY`.

Until then, no native scheduler/renderer implementation is justified.
