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
| S9 | done / baseline | Safe search over extracted knobs; keep opt-in; S9 authority path preserves the 4k pp512 band. |
| S10-A | done | Hybrid S9/S10 scope: S10 owns metadata/spec/search gates while S9 emits backend atoms. |
| S10-B | done | Repeatable hybrid role trace over S9 backend atoms. |
| S10-C | done | Isolate the hard DBUF epoch choreography as `DBUFEpochPrimitive`. |
| S10-D | next | Search/control safe S10-owned knobs around the hybrid boundary while preserving pp512 `>=4000`. |
| S10-E | pending | Promotion/rollback gate for the hybrid route. |
| S10-F | pending | Real parameterized epoch primitive interface beyond metadata. |
| S10-G | later | Partial generated replacement around the epoch primitive. |
| S10-H | parked | Full generated DBUF lifecycle replacement. |
| hybrid_machine_search | reference-only / quality-blocked | The external backend-atom route was recreated on clean HEAD at pinned pp512 `4099`, but deterministic whole-model greedy parity fails. It is a performance reference, not a usable route. |

## Current route

- Active phase: **hybrid recreation quality blocker**.
- Route family: `prefill_pipe_role_selective_generated`.
- Classification: `external_handwritten_kernel` / `hand_external_reference`; the schedule is spec-described, but
  final pipe/LDS2 instruction streams come from the raw backend atoms.
- Clean recreation (`7cc2e6447`): pinned `pp512 4098.95`, `pp4096 3102.37`; route binding passes and the maximum
  timing CV is `0.248%`.
- Quality: **FAIL**. A deterministic TinyJit `argmax(model.logits)` comparison against the ordinary scheduler gives
  baseline token `198` versus hybrid token `0` for the bounded case. Both children dispatch and both post-run GPU
  health checks pass. Gate/up-only parity passes; an attn_qo-only production-graph probe fails even though the exact
  standalone attn_qo pipe GEMM is correct. This isolates the remaining blocker to raw pipe model-graph integration,
  not pipe instruction bytes (ordinary `AMD` and `AMD:ISA` compile the same `9be1e239...` binary).
- Therefore the `~4413` historical number remains a performance reference only. Do not promote or use this route
  until custom-kernel graph ownership is fixed, or replace the three raw pipe roles with the compiler-owned direct
  register transport and repeat correctness plus pinned timing.

## Refuted / deferred branches

- Spec-owned composed LDS/DBUF transport (`prefill_wmma_pipe_lds_dbuf_primitive_generated`) is the
  correct route but slow ASM-backed transport: pinned `pp512 ~1332` vs the `~4413` backend-atom band.
  Retired by `hybrid_machine_search` as correct-but-slow. Lessons banked in `docs/prefill-lessons-ledger.md`
  (DBUF/LDS operand staging + "why hand beats generated" density thesis).
