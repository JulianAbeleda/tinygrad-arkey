# Register-store devectorizer AMD authority verification

Date: 2026-07-28

Status: open input task; centralization is accepted, but authority verification is outstanding.

Owner: tinygrad codegen/reorganization maintainer

Branch rule: run from a clean `exp` or dedicated experiment worktree first, then repeat on the exact promoted commit.
Do not run from a dirty source tree and do not update authority artifacts from a different commit.

## Objective

Prove that centralizing `pm_reg_store_devec` in `tinygrad/codegen/late/reg_store.py` did not change the generated AMD
lowering output or the existing lowering fingerprints. This task validates authority artifacts; it does not decide
whether the separate matcher should be merged with `pm_distinct_reg_store_devec`.

## Current blocker

The Mac environment currently has no `llvm-readelf` (or equivalent ROCm LLVM tool), so the lowering baseline cannot
extract AMD binary metadata. The checked-in lowering fingerprint also differs from the freshly computed fingerprint.
Neither observation is a matcher test failure, but neither may be silently accepted as a passing authority gate.

## Commit-history triage

The checked-in CPU fingerprint artifact was last captured at `05f86ca73`. The current tip contains subsequent codegen,
cache-key, gate-inventory, and compiler-ownership commits before the reg-store/fdot2 centralizations. The `fdot2` hook
is AMD-only and default-off, so its promotion cannot explain a CPU-only fingerprint delta. This provenance inference is
enough to continue the branch reorganization and review later slices; it is not a substitute for rerunning the
fingerprint authority or for extracting AMDHSA metadata from a compiled binary.

## Prerequisites

- A clean promoted commit containing the centralization and its tests.
- ROCm LLVM tools available on `PATH`, or an explicitly recorded equivalent tool path providing `llvm-readelf`.
- The repository's supported Python environment and dependencies.
- No GPU workload is required for the baseline/fingerprint commands, but any hardware command must follow the shared
  GPU-lock policy.

## Exact verification sequence

From the clean worktree:

```sh
command -v llvm-readelf
PYTHONPATH=. python3 extra/audit/lowering_baseline.py --check
PYTHONPATH=. python3 extra/audit/lowering_fingerprint.py --check
PYTHONPATH=. pytest -q \
  test/unit/test_lowering_baseline.py \
  test/unit/test_lowering_fingerprint.py \
  test/unit/test_reg_store_devec.py
```

If an authority intentionally changes, regenerate only from that same clean commit, review the diff, and rerun the
checks:

```sh
PYTHONPATH=. python3 extra/audit/lowering_baseline.py
PYTHONPATH=. python3 extra/audit/lowering_fingerprint.py
```

Record the tool path, commit, generated artifact hashes, pass/fail output, and any justified source-shape delta in a
task-workflow output record. Do not overwrite historical `docs/artifacts/**` snapshots merely to make this check pass.

## Acceptance criteria

- `llvm-readelf` or the recorded equivalent is available and its version/path is captured.
- `lowering_baseline.py --check` passes at the exact clean commit, or a reviewed baseline update explains every delta.
- `lowering_fingerprint.py --check` passes at the exact clean commit, or a reviewed fingerprint update explains every
  delta.
- The focused matcher suite and the existing lowering tests pass.
- No GPU result is claimed unless a separately locked hardware run is recorded.
- When complete, move this input task to `docs/task_workflow/output/` and link its result from the reorganization report.

Until these criteria are met, label the authority verification as open rather than failed and do not use it as evidence
that the centralized matcher changed behavior.
