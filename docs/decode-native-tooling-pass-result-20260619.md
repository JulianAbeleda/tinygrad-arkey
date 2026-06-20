# Decode Native Tooling Pass Result

Date: 2026-06-19

Scope:

- `docs/decode-native-tooling-pass-scope-20260619.md`

Artifacts:

- `extra/qk_decode_native_pmc_decode.py`
- `extra/qk_decode_native_sqtt_decode_probe.py`
- `extra/qk_decode_native_role_timing_join.py`
- `extra/qk_decode_native_scheduler_ablation_scope.py`
- `extra/qk_decode_native_wd_projection.py`
- `extra/qk_decode_native_tooling_readiness.py`
- `bench/qk-decode-native-tooling/pmc_decode.json`
- `bench/qk-decode-native-tooling/timeline_attribution.json`
- `bench/qk-decode-native-tooling/role_timing_join.json`
- `bench/qk-decode-native-tooling/scheduler_ablation_scope.json`
- `bench/qk-decode-native-tooling/wd_projection.json`
- `bench/qk-decode-native-tooling/readiness.json`

## Verdict

`ROADMAP_ONLY`.

The decode-native tooling verdict now passes: it has enough evidence to make a final implementation decision. That
decision is **do not start a bounded q8/native scheduler-renderer N2 patch**.

## Gate Summary

| Gate | Result |
|---|---:|
| q8 `ffn_gate/up` role body evidence | PASS |
| N2 candidate count | `0` |
| Max timing-grade movement | `14.087us` |
| Required q8 feature movement | `>=30us` |
| W==D projectable native feature | none |
| Final readiness | `ROADMAP_ONLY` |

## P1 - PMC Decode

Verdict: `BLOCKED_COUNTER_DECODE`.

PMC profiling is runnable and records event layouts, but the persisted artifacts do not include raw PMC counter blobs or
decoded counter values. Therefore PMC cannot provide counter-grade feature attribution from current artifacts.

This is not an N2 blocker anymore because the scheduler ablation result can still classify the remaining work as
roadmap-only.

## P2 - SQTT / Timeline Attribution

Verdict: `BLOCKED_TIMELINE_DECODE`.

SQTT capture is runnable, but local decode still fails on the q8 RDNA3 instruction-trace blobs:

```text
ValueError('unknown cdna format word=0xf4080100')
```

ATT body evidence exists for the role, but ATT packet counts are visibility-only and do not become timing authority.

## P3 - Same-Binary Timing Join

Verdict: `PROXY_ONLY`.

The role evidence joins to the in-model native Q4_K program hash `236fd9e8841b577f`, but the only same-interval timing
from ATT is profiler-wall timing and is not promotion authority. Existing q8 native/oracle timings remain standalone
proxy authority.

## P4 - Scheduler Ablation Decision

Verdict: `ROADMAP_ONLY`.

Known isolated feature movements remain below gate:

| Feature | Movement | Decision |
|---|---:|---|
| load shape/coalescing | `14.087us` | closed below gate |
| waitcnt grouping | `0.837us` | closed below gate |
| reduction topology | `13.305us` | closed below gate |
| dot4 instruction selection | `0us` | closed |

Remaining scheduler/resource features are classified as compound project-level backend work, not bounded N2:

- `scheduler_markers`
- `instruction_order`
- `register_lifetime`
- `resource_descriptor`

## P5 - W==D Projection

Verdict: `NO_PROJECTABLE_FEATURE`.

No feature clears the local N2 movement gate, so no native W==D projection is justified.

## Final Decision

Do not start native q8 scheduler/renderer implementation.

Allowed future work:

- broad AMD backend scheduler/resource project, if explicitly funded;
- external decoder/counter tooling if the project wants deeper attribution for future backend work;
- keep the q8 hipcc/LLD artifact as the default-off research route.

Disallowed:

- q8-specific N2 codegen patch;
- manual `s_clause` / `s_delay_alu` insertion from static diff alone;
- load-shape, waitcnt, or reduction-topology standalone reopen;
- treating ATT packet count or PMC/SQTT existence as timing authority.

The native decode route is now a roadmap/backend investment item, not a bounded implementation task.
