# Decode Native Tooling Completion Scope

Date: 2026-06-19

Purpose: finish the tooling work after DTR-0 so the project can make a defensible native scheduler/renderer start
decision. This scope covers DTR-1 through DTR-4 from `decode-native-tooling-readiness-scope-20260619.md`.

Current DTR-0 verdict:

```text
TOOLING_NOT_READY
```

Current blockers:

1. q8 / native `ffn_gate/up` role-joined ATT/PMC/body evidence is missing.
2. No timing-grade feature attribution reaches `>=30us`.
3. Scheduler/resource unknowns are not joined to timing/counters.
4. q8 `ffn_gate/up` lacks a bytes/math/overhead bucket classification.

## Principle

Do not build a scheduler/renderer feature until the tool can answer:

```text
which exact feature, on which exact role, in which performance bucket, with how much movement?
```

The answer must be artifact-backed. Static ISA differences are enough to guide attribution work, but not enough to
start N2 implementation.

## Existing Surfaces

| Surface | File | State |
|---|---|---|
| readiness generator | `extra/qk_decode_native_tooling_readiness.py` | DTR-0 complete |
| role join | `extra/qk_att_inmodel_role_join.py` | supports `attn_output`, `ffn_down`, `lm_head`; missing `ffn_gate/up` |
| ATT interval/replay | `extra/qk_att_primitive_atlas.py` | reusable AQLprofile ATT interval wrapper |
| AQLprofile packet export | `extra/amd_rocprofiler_r1p2_hcq_replay.py` | working HCQ ATT packet factory |
| q8 oracle contract | `bench/q8-ffn-amd-scheduler-project/oracle_contract.json` | static/timing oracle exists |
| N1 attribution | `bench/q8-ffn-amd-scheduler-project/n1_attribution.json` | no N2 start |
| complete tooling atlas | `bench/qk-decode-complete-tooling/result.json` | role/lifecycle/timing policy authority |

## DTR-1 - q8 / Native `ffn_gate/up` Role Join

### Goal

Produce role-joined body evidence for the high-share `ffn_gate/up` path. This is the missing role in the complete
tooling atlas and the only role tied to the measured q8 research speed route.

### Required Captures

Capture two related surfaces, because they answer different questions:

| Capture | Purpose | Authority |
|---|---|---|
| default native `ffn_gate` and `ffn_up` linears | prove current in-model program/lifecycle identity for the high-share role | visibility / bucket classification |
| q8 fused gate/up research route, if enabled by `Q8_FFN_HANDWRITTEN=1` | connect the measured speed route to ATT/program/resource evidence | feature attribution target |

If the q8 fused route cannot be role-joined directly, record why and fall back only to visibility. Do not promote
fallback evidence to timing authority.

### Implementation Shape

Extend `extra/qk_att_inmodel_role_join.py` instead of starting a separate one-off script unless that becomes
unwieldy.

Add role modes:

- `ffn_gate`
- `ffn_up`
- `ffn_gateup_pair`
- optional `q8_gateup` if the research flag path exposes one fused call boundary

For default native role capture:

1. Run block 0 through attention to produce the FFN input.
2. Capture the shared `ffn_norm(h)` activation.
3. Wrap `block.ffn_gate` and/or `block.ffn_up` with `CaptureLinear`.
4. Warm compile outside the ATT interval.
5. Trace the role call inside `ATTInterval`.
6. Capture HCQ programs with `ProgramCapture`.

For q8 fused route capture:

1. Enable the exact env/flag surface used by the q8 research route.
2. Capture the same normalized FFN activation.
3. Trace the fused route boundary, not only one internal consumer if a fused call exists.
4. Record any producer/side-buffer setup separately from the timed interval.

### Outputs

Add:

- `bench/qk-att-inmodel-role-join/ffn_gate.json`
- `bench/qk-att-inmodel-role-join/ffn_up.json`
- `bench/qk-att-inmodel-role-join/ffn_gateup_pair.json`
- optional `bench/qk-att-inmodel-role-join/q8_gateup.json`

Also update:

- `bench/qk-decode-native-tooling/readiness.json`
- `bench/qk-decode-native-tooling/feature_attribution.json`

### Gates

| Gate | Pass |
|---|---|
| ATT start/stop sync | start and stop packets synchronize |
| body packets | nonzero body-like packets |
| program capture | HCQ program rows captured |
| program identity | main program matches runtime/cache identity or q8 route expectation |
| activation provenance | activation comes from actual block-0 FFN path, not random-only surface |
| timing label | row says whether timing is role-local, W==D, or visibility-only |

### Kill / Boundary Cases

| Case | Action |
|---|---|
| full in-model capture OOMs | record OOM and surface fallback, visibility only |
| ATT misses body due to target WGP/SIMD sampling | repeat/enlarge dispatch if legal; otherwise record sampling miss |
| q8 fused route has no clean role boundary | record graph/lifecycle boundary as missing tooling |
| capture changes graph/program identity | kill the row; no inference from perturbed path |

### DTR-1 Done

DTR-1 is done when q8/native `ffn_gate/up` no longer appears in readiness as `runtime_identity_only` with missing ATT.
It may still be `TOOLING_NOT_READY`; the missing row should move from "no body evidence" to either bucket
classification or a sharper blocker.

## DTR-2 - Counter/Trace/ISA Feature Join

### Goal

Build a joiner that aligns the exact role/program from DTR-1 with:

- oracle timing and ISA;
- tinygrad timing and ISA;
- ATT metrics;
- PMC/SQTT metrics where available;
- launch/resource metadata;
- lifecycle rows from complete tooling.

### Implementation Shape

Extend `extra/qk_decode_native_tooling_readiness.py` or add
`extra/qk_decode_native_feature_join.py` if the join grows beyond DTR-0.

The join key must include:

- role;
- shape;
- program name;
- `lib_sha16`;
- launch geometry;
- capture mode;
- timing source.

Do not join rows only by role name if multiple binaries or launch geometries exist.

### Output

Add:

- `bench/qk-decode-native-tooling/feature_join.json`

Each row:

```json
{
  "role": "ffn_gate/up",
  "program_name": "q4k_gemv_partial_12288_4096_1",
  "lib_sha16": "...",
  "oracle": "q8_hipcc_lld_artifact",
  "bucket": "bytes|math|overhead|unknown",
  "candidate_features": [],
  "timing": {},
  "trace": {},
  "isa_diff": {},
  "resource_diff": {},
  "authority": "timing_grade|counter_grade|static_grade|surface_grade|inferred"
}
```

### Candidate Feature Labels

Use exactly these labels unless the result doc adds a new one with justification:

- `load_shape`
- `wait_schedule`
- `instruction_order`
- `scheduler_markers`
- `register_lifetime`
- `resource_descriptor`
- `reduction_shape`
- `activation_lifecycle`
- `graph_boundary`

### Gates

| Gate | Pass |
|---|---|
| exact binary join | DTR-1 program row joins to static/resource row by `lib_sha16` or documented equivalent |
| timing join | timing source is attached or explicitly missing |
| oracle join | target oracle row attached |
| authority labels | every feature has one of the approved authority labels |
| unknown preservation | scheduler/resource unknowns remain unknown unless tied to timing/counter evidence |

### DTR-2 Done

DTR-2 is done when `feature_join.json` can regenerate `feature_attribution.json` without hand-written prose
interpretation.

## DTR-3 - Dynamic Ablation Matrix

### Goal

Price named features. This is where static differences become movement budgets, or get closed.

### Required Rows

| Feature | Existing state | Required next evidence |
|---|---|---|
| `load_shape` | `14.087us`, below gate | keep closed unless DTR-1 role timing contradicts |
| `wait_schedule` | `0.837us`, closed standalone | keep closed unless counter evidence says it gates a larger scheduler feature |
| `reduction_shape` | `13.305us`, below gate | keep closed unless cross-role Amdahl reaches gate |
| `scheduler_markers` | static diff only: `s_clause` / `s_delay_alu` | dynamic ablation or counter-backed attribution |
| `register_lifetime` | unknown | resource/VGPR/occupancy plus timing attribution |
| `activation_lifecycle` | q8 route measured, native unclear | role-local timing plus W==D projection |
| `graph_boundary` | graph-safe routes exist for some artifacts | only reopen if DTR-1 shows lifecycle/timing loss at graph boundary |

### Measurement Rules

- Prefer same-process interleaved role A/B for role-local movement.
- Use W==D ctx sweep for final promotion.
- ATT packet counts never become timing.
- If a dynamic ablation changes multiple features, label it `compound` and do not start a single-feature N2.
- If the strongest movement is still below `30us`, close it as a bounded N2 target.

### Outputs

Add:

- `bench/qk-decode-native-tooling/ablation_matrix.json`
- updated `bench/qk-decode-native-tooling/feature_attribution.json`

Each ablation row:

```json
{
  "feature": "scheduler_markers",
  "variant": "inserted_s_delay_alu_pattern_probe",
  "role": "ffn_gate/up",
  "baseline_us": 166.649,
  "candidate_us": null,
  "movement_us": null,
  "changed_features": ["scheduler_markers"],
  "correctness": "PASS|FAIL|NOT_RUN",
  "authority": "timing_grade|static_grade",
  "decision": "start_N2|closed|compound|project_level"
}
```

### Gates

| Gate | Pass |
|---|---|
| start N2 | one feature has `movement_us >= 30` and single-feature attribution |
| keep researching | feature is plausible but measurement missing due to a named tooling blocker |
| close bounded route | all measured features below gate or compound/unattributed |

## DTR-4 - Readiness Decision

### Goal

Regenerate readiness and write a result doc that decides whether native implementation can start.

### Outputs

- `bench/qk-decode-native-tooling/readiness.json`
- `bench/qk-decode-native-tooling/feature_attribution.json`
- `docs/decode-native-tooling-completion-result-20260619.md`

### Outcomes

| Outcome | Meaning | Next |
|---|---|---|
| `TOOLING_READY_FOR_N2` | one bounded feature clears gate | implement exactly that feature behind a research flag/probe |
| `TOOLING_NOT_READY` | a specific tooling row is still missing | keep tooling only |
| `ROADMAP_ONLY` | tooling is enough and no bounded feature clears gate | stop bounded native decode work |
| `BROAD_BACKEND_ACCEPTED` | project chooses broad backend investment without attribution | start backend project with explicit risk |

### Completion Criteria

The final result must answer:

1. Which role dominates the remaining native opportunity?
2. Is that role bytes-, math-, or overhead-bound?
3. Which oracle row is the target?
4. Which exact backend feature has movement?
5. How many microseconds and what W==D projection?
6. Is the evidence timing-grade, counter-grade, static-grade, surface-grade, or inferred?
7. What implementation surface is allowed, if any?

If any answer is missing, implementation remains blocked.

## Commit Discipline

Suggested commits:

1. `[test] Scope decode native tooling completion gates`
2. `[test] Add q8 ffn gateup role-joined ATT tooling`
3. `[test] Join decode native feature attribution artifacts`
4. `[test] Execute decode native tooling readiness decision`

Do not mix scheduler/renderer source changes into these commits.

## Immediate Next Step

Implement DTR-1. The smallest useful change is to extend `extra/qk_att_inmodel_role_join.py` with `ffn_gate`,
`ffn_up`, and `ffn_gateup_pair`, then run:

```bash
DEV=AMD AMD_AQL=1 QK_PRIMITIVE=1 python3 extra/qk_att_inmodel_role_join.py ffn_gate ffn_up
python3 extra/qk_decode_native_tooling_readiness.py
```

Expected result after DTR-1:

- readiness may still be `TOOLING_NOT_READY`;
- the missing row should no longer be "q8 `ffn_gate/up` role-joined body evidence";
- the next blocker should be feature-level attribution or bucket classification.
