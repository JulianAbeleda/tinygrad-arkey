# QK Primitive Storage Control

Date: 2026-06-12

Purpose: validate runtime storage accounting, a fail-closed primitive storage
cap, and one Q4_K non-duplicated storage probe.

## 8B Smokes

| mode | Q4 wrappers | Q6 wrappers | persistent storage MB | avg tok/s | warm tok/s | verdict |
|---|---:|---:|---:|---:|---:|---|
| generated sidecar | `162` | `18` | `3786.75` | `45.80` | `57.77` | baseline fast path |
| generated runtime cap `512 MB` | `28` | `0` | `504.00` | `12.27` | `13.46` | cap prevents over-allocation, but install-order selection is slow |
| generated `q4_ondemand` | `162` | `18` | `708.75` | `2.87` | `0.55` | rejects on-demand Q4 as a production dedup path |

The on-demand path proves the storage lever exists: persistent Q4 sidecar bytes
drop to zero. It also proves this is not the right runtime architecture: moving
Q4 packed copies into decode destroys throughput.

## 32B Smokes

| mode | policy | Q4 wrappers | Q6 wrappers | persistent storage MB | avg tok/s | warm tok/s | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| static generated cap | `qk-policy-cap-20260612/32b-1536mb/policy.json` | `112` | `32` | `1526.25` | `4.01` | `4.30` | useful selected policy |
| full policy + runtime cap `1536 MB` | `qk-policy-pipeline-20260612/32b/policy.json` | `43` | `0` | `1535.62` | `3.57` | `3.71` | guard works, but selection is not optimal |

The runtime cap prevents the full generated policy from OOMing, but because it
accepts tensors in install order, it spends the budget on early Q4 tensors and
selects no Q6 tensors. The static generated cap remains the correct mechanism
for performance because it ranks by benefit per MB.

## Interpretation

- Accounting is now runtime-measured, not inferred from policy JSON.
- `QK_PRIMITIVE_MAX_STORAGE_MB` is a safety guard and diagnostic tool.
- `QK_PRIMITIVE_STORAGE=q4_ondemand` is a negative prototype: it reduces
  persistent storage but is much too slow.
- The next useful storage work is shared/lazy ownership that avoids duplicate
  persistent storage without copying Q4 weights every decode token.
- Do not resume kernel search from this track; the useful next move is the
  higher-level harness/infrastructure work.

## Artifacts

- `8b-generated-storage-debug.log`
- `8b-generated-runtime-cap512.log`
- `8b-generated-q4-ondemand.log`
- `8b-storage-summary.json`
- `8b-storage-summary.md`
- `32b-static-cap1536.log`
- `32b-full-policy-runtime-cap1536.log`
- `32b-storage-summary.json`
- `32b-storage-summary.md`
