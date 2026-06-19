# AMD scheduler tooling/backend T0-T4 + B0 result - 2026-06-19

Executed the first combined pass from `amd-scheduler-tooling-backend-project-scope-20260619.md`.

Artifacts:

- `extra/amd_scheduler_tooling_backend_execute.py`
- `bench/amd-scheduler-tooling-backend/execution.json`
- `bench/amd-scheduler-tooling-backend/t0_capture_blobs.json`
- `bench/amd-scheduler-tooling-backend/t0_capture_q8_gateup_full.json`

## Verdict

**TRACK_T_PARTIAL_NO_FEATURE_B0_PASS**.

Track T is started and useful, but it does not yet justify backend codegen patches. Track B0 is complete as an
independent oracle suite.

## Track T

| phase | result |
|---|---|
| T0 evidence inventory | PASS |
| T1 RDNA3 HCQ SQTT decoder | PARTIAL / gate fail |
| T2 PMC blob decoder | PASS structural parse |
| T3 primitive timeline join | PASS Level-3 + PMC rows |
| T4 attribution verdict | NO FEATURE |

T0 captured replayable q8 evidence:

- program: `q8_b2b_fullrow_reduce`;
- program code hash: `cdb3201adb657655ae3006b387709bb82529c46f93bd1401b7d1df0c6392fad5`;
- PMC events: `2`;
- SQTT events: `12`;
- SQTT itrace events: `2`;
- SQTT bytes: `1775488`;
- PMC bytes: `4880`.

T1 improved over the older N1 artifact in one way: SQTT replay is structurally decodable. But it still fails the real
gate:

- mapped instruction events: `4114`;
- body instruction events: `0`;
- mapped instruction class: `S_ENDPGM` only.

So this is not usable scheduler attribution. It proves capture/replay works, but not PC-level body mapping.

A follow-up proof swept baseline, `SQTT_MODE=3`, `SQTT_TTRACE_EXEC=1`, and both together. Every config still produced
`0` raw body packet events and `0` mapped body instruction events. See
`docs/amd-scheduler-tooling-t1-body-mapping-proof-20260619.md`.

T2 parsed the PMC blobs structurally:

| metric | run 1 | run 2 |
|---|---:|---:|
| GL2 hit rate | `0.181318` | `0.182276` |
| LDS bank conflict sum | `0` | `0` |
| VALU/SALU inst ratio | `26.166667` | `26.166667` |

This is enough to say the q8 ASM run is not obviously LDS-bank-conflict-bound and has low GL2 locality. It is **not**
enough to assign the remaining `73.109us` q8 gap to one scheduler feature.

T4 decision:

```text
Do not claim scheduler/resource feature attribution yet.
Do not start B1/B2 backend patches from this evidence alone.
```

## Track B0

B0 oracle suite is complete.

| oracle | tinygrad baseline | oracle | movement |
|---|---:|---:|---:|
| q8 decode gate/up consumer | `166.649us` | `93.54us` | `73.109us` gap |
| prefill Tensile ffn_gate/up | `42.0 TFLOPS` | `66.8 TFLOPS` | `1.59x` |
| prefill Tensile ffn_down | `42.0 TFLOPS` | `68.9 TFLOPS` | `1.64x` |
| small smoke kernel | present | present | attribution/control only |

This means Track B has a stable target suite, but it should not proceed to B1/B2 as a performance patch until one of two
things happens:

1. Track T maps body instructions/counters well enough to assign a large feature bucket; or
2. the project explicitly funds the AMD backend as a reusable compiler investment despite incomplete attribution.

## Next

The next tooling step is narrower than "build a scheduler":

1. obtain body instruction packets through ROCprofiler/AQLprofile-compatible SQTT setup;
2. use the PMC parser plus body mapping to label the q8 gap;
3. only then start B1/B2, unless we explicitly choose the larger backend investment path.
