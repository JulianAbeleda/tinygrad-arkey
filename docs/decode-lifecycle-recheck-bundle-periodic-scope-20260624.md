# Decode Lifecycle Recheck Bundle Periodic Scope (2026-06-24)

## Goal

Create a repeatable periodic refresh loop so every decode baseline sweep is both
audited and compared to the prior snapshot.

## Scope

1. Run full lifecycle recheck bundle (`extra/qk_decode_lifecycle_recheck_bundle.py`) for a given `run-id`.
2. Compare the resulting bundle against `bench/qk-decode-lifecycle-recheck-bundle/latest.json`.
3. Emit a single drift artifact (`periodic_diff.json`) and a short markdown summary
   (`periodic_diff.md`) in the same run directory.

## Coverage required in each periodic run

- oracle gate pre + post (`qk_decode_search_gate.py`):
  - route fire
  - `E_49152` materialization absence
  - ISA + resource checks
  - token correctness
  - verdict

- unknown-bucket lockstep pre + post (`qk_decode_unknown_bucket_lockstep_audit.py`)
  at `512,1024,2048,4096`

- throughput matrix (`qk_ctx_slope_driver.py`):
  - current-context sweep (`512,1024,2048,4096`)
  - long-context sweep (`4096`)
  - alternative capture mode (`DECODE_ATTN_AMDGCN_TILE=0`)

## Commands

Run one full periodic refresh:

```bash
cd /home/ubuntu/tinygrad-arkey
.venv/bin/python extra/qk_decode_lifecycle_recheck_periodic.py --out-root bench/qk-decode-lifecycle-recheck-bundle
```

Run-id handling:
- If no `--run-id` is passed, the script auto-generates a timestamp run-id and
  auto-increments with a suffix when a directory collision is detected, so
  artifacts never get overwritten.

Force an explicit run id:

```bash
.venv/bin/python extra/qk_decode_lifecycle_recheck_periodic.py --run-id 20260624-<HHMMSS> --out-root bench/qk-decode-lifecycle-recheck-bundle
```

Compare-only refresh (fast, no heavy workload):

```bash
.venv/bin/python extra/qk_decode_lifecycle_recheck_periodic.py --compare-only --out-root bench/qk-decode-lifecycle-recheck-bundle
```

## Operational notes

- Recommended cadence: at least daily or after any decode default/on flag change.
- Safe recovery pattern:
  - run inside tmux (`tmux new -s qk-decode-periodic`) so output survives shell drops.
  - if interrupted, rerun with the same command; only completed bundles update `latest.json`.
  - use compare-only to validate next-step drift before starting any heavy rerun.

## Acceptance

Emit `DECODE_LIFECYCLE_PERIODIC_DIFF` as PASS only when:

- current bundle verdict is `DECODE_LIFECYCLE_RECHECK_BUNDLE_PASS`
- pre/post gate ok
- pre/post unknown-lockstep proven
- min current-context delta does not regress by >1.0pp vs prior

Record outcomes in session handoff as:

- periodic baseline run id
- periodic diff artifact path
- current_context min delta change vs prior
