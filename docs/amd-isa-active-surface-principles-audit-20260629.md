# AMD ISA Active-Surface Principles Audit — 2026-06-29

Audit scope: the active AMD ISA / decode machine-search surface, not the whole historical repo. Evidence came from the
current H-N artifacts, `tinygrad/llm/model.py`, `tinygrad/renderer/isa/amd.py`, native tile injection tools, and the
repo's own principles in `structure/Development/*`.

## Verdict

The active path is principled in the important ways: every major performance decision since grid parallelism was gated by
route attribution, token match, W==D, and a targeted counter/diff. The main risk has shifted from "wrong optimization" to
"the successful levers are still encoded as research flags and one-off phase scripts rather than a small search-owned
manifest."

## Scorecard

| principle | grade | audit read |
|---|---:|---|
| Whole-primitive measurement | A- | N4 proves whole-step attribution before further tile work. |
| Audit-before-build | A | N1B and N5A stopped on evidence instead of pushing risky code. |
| Correctness/fallback discipline | A- | H/N gates check token match and fallback absence; block-tile partial-stack guard is fail-loud. |
| Centralized authority | C+ | Route flags and candidate stacks are copied across model and phase gates. |
| Modularity/orthogonality | B- | Native tile injection is well-contained, but compile/cache/grid policy/profiling live together. |
| Tiny / anti-re-sprawl | C | Phase tools are now numerous one-off scripts; good for research, weak as durable search surface. |
| Invariant encoding | B- | Verdict artifacts are strong; flag contracts are still stringly and manually assembled. |
| Dangerous-power/profiling containment | B | PMC/SQTT use is explicit and artifacted; per-PC ATT wall is documented. |

## High-Value Findings

| id | severity | finding | evidence | recommendation |
|---|---|---|---|---|
| A1 | high | Proven levers are not yet search-owned. Dynamic-S and hardware exp are real wins, but the route is still assembled by hand through flags and gate scripts. | `extra/amd_isa_phase_n3f_gate.py:28-30`; `bench/amd-isa-backend-phase-n3/n3f_latest.json` shows `66.91 tok/s` at ctx512; `bench/amd-isa-backend-phase-n1a/latest.json` passed hardware exp. | N6 should create a native candidate manifest whose default candidate includes hardware exp and dynamic-S, with refuted axes excluded. |
| A2 | high | N5A register accumulators are now a deferred/regalloc feature, not an implementation task. | `bench/amd-isa-backend-phase-n5/native_tile_residual/latest.json` verdict `AMD_ISA_PHASE_N5A_BLOCKED_REGALLOC`; it says the scaffold was reverted and the tree is byte-identical to N4. | Do not include register accumulators as a BubbleBeam axis. Record as deferred until a backend-generic loop-carried physical accumulator/regalloc capability exists. |
| A3 | high | The active scope doc is now stale after N5A. It still points N5 at native tile residual, but the N5A artifact recommends N6/N7. | `docs/amd-isa-backend-phase-h-o-claude-scope-20260629.md` Phase N5 says branch to N5A; N5A artifact says "DEFER N5A" and "next: N6". | Update the scope doc before the next agent run: completed N5A blocker, proceed N6 search binding then N7 package. |
| A4 | medium | Native route flag contract is still scattered. The candidate route requires `DECODE_ATTN_AMDGCN_TILE=0`, generated whole-cache, fused xlane, block tile, native ISA, and absence of fixed-S for dynamic-S. | `tinygrad/llm/model.py:1047-1054`; `extra/amd_isa_phase_i_gate.py:28-31`; `extra/amd_isa_phase_n3f_gate.py:28-30`. | Add one route/candidate helper or manifest for `owned`, `native_fixed_s`, and `native_dynamic_s`. Gate scripts should import it instead of retyping env maps. |
| A5 | medium | Dynamic-S activation is a negative flag / absence contract, not a positive capability. | `extra/amd_isa_phase_i_gate.py:30` pops `DECODE_ATTN_BLOCK_TILE_FIXED_S` when `QK_I_DYNAMIC_S` is set; `extra/amd_isa_phase_n3f_gate.py:28-30` omits fixed-S. | Introduce an explicit candidate key such as `native_dynamic_s` in the manifest. Keep old flags for back-compat, but stop making "absence of fixed-S" the durable interface. |
| A6 | medium | Phase gate sprawl is becoming the next tiny-principle problem. Each phase earned its existence diagnostically, but N6 should not add more cloned scripts. | `extra/amd_isa_phase_*_gate.py` family; anti-re-sprawl rule in `structure/Development/tinygrad-coding-overrides.md`. | After N7, collapse active N evaluator paths into a table-driven runner. Do not interrupt N6/N7 for this cleanup unless a script clone causes a wrong result. |
| A7 | medium | N4's per-kernel owner taxonomy is hardcoded in the N4 tool. It is the current authority for "what dominates" but not reusable by N6. | `extra/amd_isa_phase_n4_whole_step_attribution.py:24-33`. | Move owner classification into a small helper or candidate evaluator module before BubbleBeam consumes whole-step rows. |
| A8 | medium | N2 per-PC ATT/SQTT is a documented infra wall; keeping it as a default expectation would mislead future agents. | `bench/amd-isa-backend-phase-n2/latest.json` degraded timing attribution; `bench/amd-isa-backend-phase-n2b/latest.json` PMC category pass. | In N6/N7 docs, make PMC category attribution the supported active profiler; ATT/SQTT per-PC is deferred infra, not a required gate. |
| A9 | low | Model route still allows silent fallback for the owned default route by design. That is okay for shipped behavior but not for candidate evaluation. | `tinygrad/llm/model.py:1098-1104` falls back to `gqa_coop_vec`; H/N gates separately check fallback absence. | Keep shipped fallback. For candidates, route helper must expose `strict_candidate=True` so failure is a blocker, not a fallback. |

## Flag Matrix Summary

| flag / interface | current status | audit disposition |
|---|---|---|
| `DECODE_ATTN_AMDGCN_TILE=1` | shipped owned default for validated shape | keep; comparator/oracle and fallback reference |
| `DECODE_ATTN_KV_IDENTITY=1` | shipped owned sub-route | keep default-on; proven route cleanliness |
| `DECODE_ATTN_GENERATED_WHOLECACHE=1` | required native route stack | bind through manifest, not hand-typed |
| `DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE=1` | required native route stack | bind through manifest, not hand-typed |
| `DECODE_ATTN_BLOCK_TILE=1` | required native route stack; partial stack guarded | keep fail-loud guard; route helper should imply full stack for native candidates |
| `DECODE_ATTN_NATIVE_ISA_BLOCK_TILE=1` | native tile injection selector | bind into N6 candidate manifest |
| `DECODE_ATTN_BLOCK_TILE_FIXED_S=1` | fixed-S legacy/native baseline | keep only as A/B/back-compat; not the preferred candidate |
| dynamic-S via absence of fixed-S / `QK_I_DYNAMIC_S` | proven +9.5% short-ctx | promote to explicit candidate key; stop relying on negative flag semantics |
| `AMD_ISA_SCHED=1` | default-on scheduler, small gain, correctness preserved | keep default-on; not a primary search axis unless profiling says scheduler pressure returned |
| hardware `EXP2 -> v_exp_f32` | renderer-owned default by supported op | keep; no flag needed |
| `AMD_ISA_N1B` | opt-in scalar address path, dead/faulting on live tile | keep off or remove from candidate surface; record as refuted |
| `AMD_ISA_NO_GRID` | A/B escape for old serial grid | keep test-only; never candidate |
| `AMD_ISA_WAITCNT_CONSERVATIVE` | A/B baseline for waitcnt | keep diagnostic-only |
| `AMD_ISA_REG_ACCUM` | attempted and reverted, no active code | do not reintroduce as flag until backend-generic regalloc support exists |

## Decoupling Targets

1. **Route manifest / helper.** Centralize route env maps and strictness for `owned`, `native_fixed_s`, and
   `native_dynamic_s`. This closes the highest-risk flag duplication without changing runtime behavior.
2. **Candidate evaluator helper.** Share token-match, fallback absence, W==D, per-kernel owner attribution, and PMC row
   recording across N6/N7. Do not let every future candidate add a new standalone gate.
3. **Profiler capability boundary.** Treat PMC as the active supported profiler and ATT/SQTT per-PC as a deferred
   backend. This prevents future N phases from blocking on a known infra wall.
4. **Refutation ledger.** N1B, Phase M occupancy/LDS, scheduler-only, and N5A reg accumulators need durable
   `do_not_search` rows before BubbleBeam binding.

## Bugs / Stale Gates / Hidden-Risk Notes

- The `bench/amd-isa-backend-phase-n5/native_tile_residual/latest.json` blocker is not reflected in the H-O scope doc.
  This is the highest-priority documentation fix.
- N4's selected branch was correct at the time (`N5A`), but N5A's blocker changes the continuation to `N6 -> N7`, not
  "try harder on registers."
- Dynamic-S is proven but not expressed as a positive public route. This is the most likely source of future accidental
  fixed-S regressions.
- The shipped fallback in `model.py` is healthy for users, but all candidate gates must keep using strict no-fallback
  checks.
- No active dirty source change remained when this audit artifact was written (`git status --short --branch` showed only
  branch ahead of origin). Earlier uncommitted reg-accum scaffolding was reverted by the N5A path and is captured in the
  N5A artifact.

## Recommended Next Actions

1. **Update the H-O/N scope doc for N5A completion.** Mark N5A blocked/deferred and make N6 the next phase.
2. **Implement N6 as manifest-first search binding.** Use proven levers only: native dynamic-S and hardware exp; record
   refuted axes explicitly.
3. **Build N7 pre-promotion package.** Measure `ctx512/1024/2048/4096`, token match, route attribution, fallback
   absence, and search provenance.
4. **Defer tiny cleanup until N7 unless it blocks correctness.** After N7, collapse N-phase gate sprawl into a
   table-driven evaluator.

## Non-Actions

- Do not delete historical provenance.
- Do not revive N1B scalarization.
- Do not keep pushing register accumulators locally without first changing regalloc semantics.
- Do not make native attention the shipped default before Phase O.
- Do not require per-PC ATT/SQTT for N6/N7.
