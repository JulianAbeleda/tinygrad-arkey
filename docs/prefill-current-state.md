# 8B Prefill Current State

Canonical current-state page for the prefill S-phase work. This is the single source of truth for
"which phase is active and what is the current route/number". Scope docs describe intent and detail;
this page tracks state.

Last updated: 2026-07-10. Numbers pulled from
`docs/8b-prefill-s10_5-machine-search-over-backend-atom-scope.md` and
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
| S9 | done / baseline | Safe search over extracted knobs; keep opt-in; S9 authority path preserves the 4k pp512 band. |
| S10-A | done | Hybrid S9/S10 scope: S10 owns metadata/spec/search gates while S9 emits backend atoms. |
| S10-B | done | Repeatable hybrid role trace over S9 backend atoms. |
| S10-C | done | Isolate the hard DBUF epoch choreography as `DBUFEpochPrimitive`. |
| S10-D | next | Search/control safe S10-owned knobs around the hybrid boundary while preserving pp512 `>=4000`. |
| S10-E | pending | Promotion/rollback gate for the hybrid route. |
| S10-F | pending | Real parameterized epoch primitive interface beyond metadata. |
| S10-G | later | Partial generated replacement around the epoch primitive. |
| S10-H | parked | Full generated DBUF lifecycle replacement. |
| S10.5 | current | Machine-search / spec-ownership over the proven hand-coded backend atom. Route `prefill_pipe_role_selective_generated`, hybrid backend-atom (not pure generated). Pinned pp512 `~4413`. See `docs/8b-prefill-s10_5-machine-search-over-backend-atom-scope.md`. |

## Current route

- Active phase: **S10.5** (machine-search over backend atom).
- Route family: `prefill_pipe_role_selective_generated`.
- Classification: `compiler_primitive_spec_owned__asm_backend_atom` — hybrid compiler primitive plus
  hand-coded reusable DBUF backend atom. Not pure generated; not full hand-kernel ownership.
- Authority number: pinned `pp512 ~4413`, `pp4096 ~3237` (unpinned/boost: `~5111` / `~3677`).
- Authority gate: pinned `pp512 >= 4000` through
  `extra/qk/prefill_whole_synced.py --mode authority --pin-clock`.

## Refuted / deferred branches

- Generated composed LDS/DBUF transport (`prefill_wmma_pipe_lds_dbuf_primitive_generated`) is the
  correct route but slow generated transport: pinned `pp512 ~1332` vs the `~4413` backend-atom band.
  Retired by S10.5 as correct-but-slow. Lessons banked in `docs/prefill-lessons-ledger.md`
  (DBUF/LDS operand staging + "why hand beats generated" density thesis).
