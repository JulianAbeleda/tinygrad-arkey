# 8B Prefill Current State

Canonical current-state page for the prefill S-phase work. This is the single source of truth for
"which phase is active and what is the current route/number". Scope docs describe intent and detail;
this page tracks state.

Last updated: 2026-07-13. Numbers pulled from
`docs/8b-prefill-hybrid-machine-search-over-backend-atom-scope.md` and
`docs/8b-prefill-s10-lds2-ownership-migration-scope.md`. Do not invent numbers here; update from the
authority harness (`extra/qk/prefill_whole_synced.py --mode authority --pin-clock`) and the scope docs.

## S-Phase Ledger

| Phase | Status | Meaning |
|---|---|---|
| S0 | done | Pin the fast graph-GEMM oracle/baseline. |
| S1 | done | Extract LDS2 register layout. |
| S2 | done | Extract LDS2 memory layout. |
| S3 | done | Extract LDS2 wait policy. |
| S4 | done | Extract LDS2 cadence. |
| S5 | done | Extract LDS2 lifecycle template. |
| S6 | done | Extract LDS2 primitive emitter. |
| S7 | done | Extract shell/epilogue emitter. |
| S8 | done | Make `build_gemm_lds2` a wrapper around `lower_lds2_gemm_kernel`. |
| S9 | refuted oracle | Historical 4.4k/4.1k runs under-dispatched the pipe roles and are not valid performance authorities. |
| S10-A | done | Hybrid S9/S10 scope: S10 owns metadata/spec/search gates while S9 emits backend atoms. |
| S10-B | done | Repeatable hybrid role trace over S9 backend atoms. |
| S10-C | done | Isolate the hard DBUF epoch choreography as `DBUFEpochPrimitive`. |
| S10-D | superseded | The `>=4000` condition depended on incomplete pipe execution and is retired. |
| S10-E | pending | Promotion/rollback gate for the hybrid route. |
| S10-F | pending | Real parameterized epoch primitive interface beyond metadata. |
| S10-G | later | Partial generated replacement around the epoch primitive. |
| S10-H | parked | Full generated DBUF lifecycle replacement. |
| hybrid_machine_search | correct / refuted on speed | Explicit pipe geometry and buffer effects restore whole-model parity; honest pinned pp512 is `2096`, below the generated candidate's `3482` performance reference. |

## Current route

- Active phase: **generated candidate whole-model correctness blocker**.
- Route family: `prefill_pipe_role_selective_generated`.
- Classification: `external_handwritten_kernel` / `hand_external_reference`; the schedule is spec-described, but
  final pipe/LDS2 instruction streams come from the raw backend atoms.
- Root cause: `_emit_schedule` paired `build_gemm_pipe` with the alternate LDS tile geometry. The model launched
  `global=(32,4,1) local=(256,1,1)` instead of the pipe-owned `global=(128,16,1) local=(32,1,1)`, leaving exactly
  `93.75%` of attn_qo output zero. Raw PROGRAM metadata also omitted its A/B-read and C-write effects.
- Corrected authority: pinned `pp512 2095.70`, `pp4096 1823.67`; maximum timing CV `0.155%`. Three deterministic
  whole-model cases match baseline greedy output and both isolated children leave the GPU healthy.
- Therefore historical `~4413` and recreated `4099` are **refuted performance oracles**, not targets: they measured
  incomplete computation. The generated four-role route's stored pinned `pp512 3481.78` is faster, but its newly
  added whole-model gate fails, so it remains a candidate until its multi-kernel graph integration is corrected.

## Refuted / deferred branches

- Spec-owned composed LDS/DBUF transport (`prefill_wmma_pipe_lds_dbuf_primitive_generated`) is the
  correct route but slow ASM-backed transport: pinned `pp512 ~1332` vs the `~4413` backend-atom band.
  Retired by `hybrid_machine_search` as correct-but-slow. Lessons banked in `docs/prefill-lessons-ledger.md`
  (DBUF/LDS operand staging + "why hand beats generated" density thesis).
