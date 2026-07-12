# Audit + Cleanup Handoff — 2026-07-10

Handoff for continuing in another tool. This session audited the week's "pure machine search" build against
`structure/Development/{coding,performance-primitive-research}-principles.md`, then executed cleanup. All work is
committed on `master` (workflow rule: commit on master, never branch). Full unit suite green (476 passed, 1 xfailed).

## What was done (in order)

1. **6-plane principle audit** of the hybrid machine-search build → fixed 6 sections: bound the purity guard to the
   real dispatcher (was a parallel model that couldn't fail); single-sourced route authority (census now generated
   from `route_manifest`); made hybrid-machine-search/whole-synced promotion verdicts honest (a research route was stamped `READY`
   on a bare proxy with a failing binding gate); added fp16 WMMA + int8 value-semantics tests (via remu); routed
   `model.py`'s leaked `extra.qk` import through the adapter (`test_tinygrad_boundary` green); retired 12 orphan
   probes; restored 9 principle-cited refutation docs that delete-to-git-history had orphaned + added a dead-link
   linter (`extra/tools/check_doc_links.py`).
2. **Fixed route labeling**: hybrid = `prefill_pipe_role_selective_generated`
   (`GRAPH_GEMM=1` only, ~4413 pp512, fast hand backend atom + machine schedule); spec-owned =
   `prefill_wmma_pipe_lds_dbuf_primitive_generated` (`+PIPE+LDS+DBUF`, ~1332, ASM-backed LDS2 lifecycle).
   Neither route is strictly pure. **Do not invert.**
   The shipping renderer is `HIPRenderer`; `AMDISARenderer` is opt-in research (`DEV=AMD:ISA`).
3. **Flag collapse** — classified all 131 `PREFILL_*`/`AMD_ISA_*` flags (`docs/prefill-flag-classification.md`):
   4 selector-owned, ~25 promote-to-route-spec, **49 deleted**, ~53 keep-for-debug.
   - Phase 1 (done): built `PrefillRouteSpec` + `resolve_prefill_route` + `DebugFlags` in
     `tinygrad/codegen/opt/prefill_route_spec.py` (core, no `extra.qk` import) + manifest-backed
     `CanonicalRoute.select` in `extra/qk/prefill_route_select.py`. Additive, behavior-preserving.
   - Phase 2 (done): deleted the 49 dead flags in 4 NFC commits, **byte-identical proof** (7/7 machine-code sha256
     hashes identical to pristine baseline across direct + kmajor route families), **−887 LOC** from
     `amd.py`/`postrange.py`. Each flag banked in `docs/prefill-flag-graveyard.md` (`REMOVED`).
4. **Doc decouple** — `docs/` reduced 108 → **34 live files**. The other 75 were distilled into
   `docs/prefill-lessons-ledger.md` (thematic, deduped) and rm'd; code/doc references repointed to the ledger.
5. **tinygrad/ core size audit** (findings below).

## Core audit findings (does tinygrad/ core need to be that big?)

Mostly no — most core is legitimate **upstream** tinygrad (framework, renderers, drivers), correctly single-sourced,
leave it. The fork's real issue is **orthogonality (research living in framework core), not bulk**:

- **~1,350 LOC** of WMMA/DBUF machine-search inside `renderer/isa/amd.py` (~1,000) + `codegen/opt/postrange.py`
  (~350). `AMDISARenderer` is an off-default research backend, not the shipping renderer. → **Extract to `extra/qk`
  behind one adapter** (research registers *into* core; do NOT add `import extra.qk` to `tinygrad/` — keep
  `test_tinygrad_boundary` green). **Highest-leverage cleanup.**
- JSON-spec taxonomy now lives in `extra/qk`; the old `tinygrad/llm/{runtime_specs,quant_specs,generated_candidates}.py`
  shims were deleted after import census showed no live direct users.
- **~200 LOC** PSP debug tracing + `AM_*` experiment hooks in `runtime/support/am/ip/psp.py` → slim after AM boot
  stabilizes.
- Inline experimental lowerings in `codegen/__init__.py` + prefill guards in `codegen/late/devectorizer.py` →
  register through one AMD-lowering extension point.
- **NV + CUDA runtime** (`ops_nv`, `support/nv/*`, `ops_cuda`, `graph/cuda`) ≈ **1,930 LOC**, imported by no `llm/`
  path — deletable ONLY if committing to a gfx1100-only tree; it's upstream breadth, not a principle violation. The
  AM/AMD bare-metal stack next to it IS live (PSP/GART logs, `gc=11,0,0`) — keep.

## Open follow-ups (pick up here)

1. **Flag-collapse Phase 3:** bake the ~25 promote flags into route specs (byte-identical for their route);
   relocate the ~53 keep-for-debug survivors into `DebugFlags`; `PREFILL_V2` default-`auto` promotion as its OWN
   gated commit (it's the one real global-default change). Map in `docs/prefill-flag-classification.md`.
2. **ISA-renderer extraction:** move the ~1,350 LOC machine-search out of `amd.py`/`postrange.py` into `extra/qk`
   behind an adapter. Seed proof now exists in `test/unit/test_amd_isa_extraction_fixtures.py`; extend it to direct
   2x2/4x2/2x4 + kmajor 2x2/4x2/2x4/4x4 before moving renderer policy.
3. **NV/CUDA decision:** gfx1100-only? If yes, delete ~1,930 LOC.
4. **Deferred audit items** (need AMD GPU, see memory `pure-machine-search-audit-2026-07`): re-measure the dropped
   schedule-table shapes (4096x4096 +8); backfill new quality/comparator fields into old authority artifacts;
   F3 dual-activation policy unification; F4 hardcoded-shape → data-table move.
5. **Minor:** `extra/tools/doc_link_baseline.txt` is gitignored (`*.txt`), so the linter isn't reproducibly green on
   a fresh checkout — either track it or fix the remaining baselined links.

## Hazards / rules for whoever continues

- **NEVER `timeout`/`pkill` a live `DEV=AMD` run** — jams the RDNA3 MES ring, needs reboot or
  `echo 1 > /sys/kernel/debug/dri/*/amdgpu_gpu_recover`. Bound the work or run in background instead.
- Commit on `master`, never branch. Split commits by subsystem; mark NFC; don't bundle bench artifacts with source.
- Behavior-preserving refactors must be proven byte-identical (remu machine-code hash), not asserted.
- Authoritative current state: `docs/prefill-current-state.md`; lessons: `docs/prefill-lessons-ledger.md`;
  flag map: `docs/prefill-flag-classification.md` + `docs/prefill-flag-graveyard.md`. Research plane: `extra/qk/`.
