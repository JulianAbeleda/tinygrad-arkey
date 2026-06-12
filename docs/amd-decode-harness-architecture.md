# AMD Decode Harness Architecture

Date: 2026-06-12

Status: harness hardening implemented.

## Purpose

The QK work is now infrastructure. The next useful layer is not another kernel
variant or a 32B-specific chase; it is a reliable loop that can generate a
policy, validate it, benchmark it, and explain the decision without bespoke
script assembly.

The storage track should continue only when it enables that loop. A 32B scaling
point is useful if storage dedup makes it cheap, but it is not the objective.
Likewise, kernel search should not resume from this track.

## Experiment Contract

`extra/qk_policy_pipeline.py` now writes a `manifest.json` into every run
directory. The manifest records:

- git commit;
- repo path;
- model path, size, and mtime;
- experiment spec;
- relevant storage/runtime environment variables;
- current stage statuses.

`--reuse` validates the manifest before trusting existing artifacts. Wrong
commit, model, storage cap, benchmark config, or other spec changes fail loudly
unless `--force` is used. `--force` rewrites the manifest and regenerates stages;
it does not bless stale artifacts for reuse.

## Stages

Each pipeline stage writes `<stage>.status.json`:

| stage | purpose |
|---|---|
| `search` | run generated candidate search or reuse `search.json` |
| `policy` | emit or regenerate `policy.json` |
| `semantic` | write semantic report from search output |
| `parity` | compare generated policy with reference policy |
| `decode` | run repeated reference/generated decode and summarize stability |
| `ab` | run greedy output A/B |
| `profile` | profile accepted large wins or skip explicitly |
| `decide` | write normalized `decision.json` |
| `report` | write human README |

Statuses are `running`, `passed`, `failed`, `blocked`, or `skipped`. A killed
or failed run can be resumed only when the manifest still matches.

## Decision Schema

`decision.json` now carries:

- `kind=qk_policy_pipeline_decision`;
- `schema_version`;
- status, reasons, gain, reference mode;
- explicit/generated decode windows;
- parity summary;
- greedy A/B result;
- storage policy;
- runtime storage accounting parsed from debug logs;
- stage summary.

This makes decisions comparable across runs instead of being prose-only.

## Matrix Summary

`extra/qk_experiment_matrix.py` summarizes multiple experiment directories or a
JSON file containing an `experiments` list. It emits:

- `matrix-summary.json`;
- `matrix-summary.md`;
- model, status, reference mode, speed, gain, percent of llama.cpp reference;
- policy and runtime storage MB.

The committed harness matrices are covered by
`test/external/test_qk_experiment_matrix.py`, which regenerates them from the
committed `decision.json` directories and compares both JSON and Markdown
exactly. A stale matrix is now a test failure, not a prose audit finding.

This is the layer that should answer 8B/14B/32B questions. Do not perfect 32B by
hand; add it to the matrix when storage/harness changes make the run natural.

## Stop Rules

- Do not use `QK_PRIMITIVE_STORAGE=q4_ondemand` as a performance path.
- Do not resume kernel search from the storage track.
- Do not chase a third scaling point by perfecting 32B.
- Do not trust `--reuse` without a matching manifest.
- Do not accept a policy without parity, greedy A/B, stable decode, and storage
  accounting when debug logs are available.
