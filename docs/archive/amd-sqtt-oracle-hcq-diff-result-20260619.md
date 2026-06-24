# AMD SQTT oracle-to-HCQ diff result - 2026-06-19

Purpose: execute `amd-sqtt-oracle-hcq-diff-scope-20260619.md` end to end after the ATT decoder became available.

Artifacts:

- `extra/amd_sqtt_oracle_hcq_diff.py`
- `bench/amd-scheduler-tooling-backend/att_oracle_capture.json`
- `bench/amd-scheduler-tooling-backend/hcq_sqtt_baseline_capture.json`
- `bench/amd-scheduler-tooling-backend/att_hcq_setup_diff.json`
- `bench/amd-scheduler-tooling-backend/hcq_sqtt_oracle_patch_result.json`
- `bench/amd-scheduler-tooling-backend/q8_body_attribution_smoke.json`
- `bench/amd-scheduler-tooling-backend/sqtt_oracle_hcq_diff_result.json`

## Verdict

**KILL_PATCH_NO_BODY.**

The external ROCprofiler ATT oracle is valid and instruction-rich, but the only bounded, observable HCQ patch did not
produce tinygrad body instruction packets. Close Track T as a small tooling patch.

Further progress would require a broader ROCprofiler command-service integration project, not another SQTT register
sweep.

## Gates

| phase | result |
|---|---|
| O0 oracle capture | pass |
| O1 HCQ baseline reproduced | pass |
| O2 patchable diff found | pass |
| O3 env-gated patch produced body packets | fail |
| O4 attribution usability | not run; no body packets |

## O0 - Oracle Capture

The existing `rocprofv3 --att` output from the fixed decoder path is enough to prove the oracle:

- decoded code rows: `537`;
- decoded wave instruction records: `110446`;
- traced CUs: `[(1, 684)]`;
- dispatch `1`: `__amd_rocclr_fillBufferAligned`, `4736` wave instruction records;
- dispatch `3`: `body_kernel(float*, float const*, float const*, int)`, `105710` wave instruction records.

This closes the old uncertainty: ROCprofiler can emit body instruction records on this machine when using the mature ATT
path.

## O1 - HCQ Baseline

The baseline tinygrad HCQ proof reproduced the prior failure:

- verdict: `NO_LOCAL_REGISTER_KNOB_BODY_MAPPING`;
- baseline SQTT events: `12`;
- baseline itrace events: `2`;
- baseline total SQTT bytes: `1777760`;
- mapped instruction events exist, but they map only to `S_ENDPGM`;
- raw body packet classes: `0`;
- mapped body instructions: `0`.

So the tinygrad failure is stable and not a stale artifact.

## O2 - Diff

The only new observable and bounded difference from the oracle output is target selection:

- ROCprofiler's decoded waves all land on CU1;
- tinygrad's HCQ path uses computed `COMPUTE_STATIC_THREAD_MGMT_SE*` masks plus a first-WGP/SIMD assumption.

Previously tested and still closed:

- `SQ_THREAD_TRACE_MASK`;
- `SQ_THREAD_TRACE_TOKEN_MASK`;
- `SQ_THREAD_TRACE_CTRL`;
- `SQTT_MODE`;
- `SQTT_TTRACE_EXEC`.

Not observable from the ATT JSON:

- the full ROCprofiler command-service setup;
- exact PM4/AQL ordering around trace start/stop;
- any hidden ROCprofiler service packets outside the decoded trace output.

## O3 - Env-Gated Patch

Added one bounded env-gated patch:

```text
SQTT_ORACLE_TARGET_CU=<n>
```

When set, tinygrad's SQTT setup forces `COMPUTE_STATIC_THREAD_MGMT_SE*` to the selected CU bit instead of using the
derived CU mask. Default behavior is unchanged when unset.

Trial matrix:

| env | SQTT bytes | raw body packets | mapped body instructions |
|---|---:|---:|---:|
| `SQTT_ORACLE_TARGET_CU=1` | `1966080` | `0` | `0` |
| `SQTT_ORACLE_TARGET_CU=1 SQTT_SIMD_SEL=1` | `1965408` | `0` | `0` |
| `SQTT_ORACLE_TARGET_CU=1` + AQLprofile raw regs | `3529568` | `0` | `0` |
| `SQTT_ORACLE_TARGET_CU=1 SQTT_SIMD_SEL=1` + AQLprofile raw regs | `3526272` | `0` | `0` |

The patch changes trace volume and packet mix, but it does not create body packet classes. The top packets remain
lifecycle-style: `NOP`, `WAVESTART`, `WAVEEND`, `TS_WAVE_STATE`, and time deltas.

## Meaning

This narrows the blocker again.

It is not:

- decoder availability;
- basic body-trace support on this GPU;
- the local tinygrad SQTT decoder;
- simple CU/SIMD target selection;
- the three AQLprofile raw SQTT registers;
- trace volume.

The missing piece is very likely a ROCprofiler command-service detail around trace setup/start/stop that is not exposed
by the ATT UI JSON and not represented by the bounded AQLprofile register recovery.

## Decision

Do not keep funding Track T as a small primitive-observability patch.

Use the evidence we already have for decode:

- PMCs;
- static disassembly;
- isolated-vs-in-model bandwidth reconciliation;
- q8 lifecycle artifacts.

Reopen only as a broader ROCprofiler command-service integration project with a way to observe or replay the full
ROCprofiler command sequence, not as another local register sweep.
