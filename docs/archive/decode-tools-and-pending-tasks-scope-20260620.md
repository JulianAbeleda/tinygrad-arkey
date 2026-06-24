# Decode Tools and Pending Tasks Scope - 2026-06-20

Verdict: `PASS_DECODE_TOOLS_PENDING_SCOPE_READY`

This is the current decode work ledger after oracle extraction, OES-4 semantic mapping, rocprof-visible oracle kernel trace, DNR-4 live-range scoping, and DNR4-T1 timing.

## Tooling Ledger

| tool | status | answers | remaining gap |
| --- | --- | --- | --- |
| T0 LLVM code-object tools | ready | HSACO identity, metadata, symbols, static ISA | no timing attribution |
| T1 oracle extraction | ready | exact `q8_mmvq_gateup` artifact, metadata, grouped ISA | metadata/profiler resource fields must not be mixed blindly |
| T2 semantic map | ready | oracle S0-S5 stages, S3 body, S4/S5 reduction/writeback | native/C7C PCs still need equivalent stage labels |
| T3 HIP rocprof runner | ready | rocprof-visible oracle dispatch, kernel-trace resource/timing | kernel trace is coarse |
| T4 ATT/thread trace | blocked | PC-level oracle timeline and stalls | missing rocprof trace-decoder `.so` |
| T5 native resource ledger | ready | native VGPR bands, phase pressure, private/LDS descriptor | needs DNR4 per-candidate live interval assertions |
| T6 same-harness timing | ready | material movement and promotion/no-promotion | does not explain why by itself |
| T7 native PMC | ready partial | SQ wait/busy/cache/LDS direction | not PC-level and cannot directly see HIP oracle PCs |
| T8 search/BEAM | not ready | candidate exploration after objective exists | static similarity was refuted; no trusted search objective yet |

## Pending Tasks

| id | status | task | done when |
| --- | --- | --- | --- |
| P0 | blocked external | Install/provide rocprof trace decoder | `extra/qk_decode_oracle_att_probe.py` passes and emits ATT artifacts |
| P1 | ready | Native stage-label map | native DNR-2, best-static, C7C, and DNR4 candidates have S0-S5 live intervals |
| P2 | next | DNR4-T2 dot-body compression scope | scoped candidate names how to reduce S2/S3 VGPR bands without breaking 16 dot4 correctness |
| P3 | pending | DNR4-T2 structural candidate | launches, correct, preserves dot4, reduces register band toward `<=40` |
| P4 | pending | DNR4-T2 same-harness timing | `>=30us` vs native, `>=15us` vs best static, or `>=10us` vs C7C |
| P5 | pending if timing moves | DNR4-T2 PMC confirmation | counters agree with the claimed mechanism, or ATT attributes the PC-stage win |
| P6 | pending | Route-level decode decision | promote/park decision with timing, resource, and quality policy |
| P7 | not ready | Search objective definition | objective terms are concrete enough for search |

## Current Decision

Do next: `P2-DNR4-T2-dot-body-compression-scope`.

Do not do next:

- do not run BEAM/search yet;
- do not continue DNR4-T1 except as structural cleanup;
- do not add static count matching patches;
- do not claim PC-level attribution until ATT decoder is available.

Why: DNR4-T1 reduced the reduction/tail register band and stayed correct, but timing moved only `1.833us` vs native and lost to best-static/C7C. The remaining executable native path is S2/S3 dot-body vector/live-range compression.

Probe: `extra/qk_decode_tools_pending_scope.py`

