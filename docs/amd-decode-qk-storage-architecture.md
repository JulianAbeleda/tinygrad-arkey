# QK Policy Storage Architecture

Date: 2026-06-12

Status: memory-aware policy and runtime storage accounting implemented;
long-term shared storage architecture still open.

## Problem

The generated Q4_K/Q6_K policy can choose good primitive coverage for 8B and
14B, but the uncapped 32B policy does not fit in VRAM. The failure is not search
or correctness. The generated 32B policy passed semantic search and parity, then
failed during primitive storage install:

```text
MemoryError: Allocation of 70.31 MB failed on AMD. Used: 23.80 GB
```

The cause is duplicated storage. The model keeps the fallback GGUF-backed weight
graph, and each primitive wrapper also installs its own persistent packed-weight
Tensor. For 32B, the full generated sidecar would add about `17.8 GiB` of
primitive storage across `448` wrappers, which cannot coexist with the model on
the 24 GB card.

## Current Fix

The policy format now supports tensor-scoped entries in addition to the original
shape-scoped entries.

- Old format: one entry per `(ggml_type, rows, cols)` shape.
- New format: exact tensor entries keyed by tensor name, with fallback to shape
  entries for old artifacts.
- Runtime lookup checks tensor entry first, then shape entry.
- Each entry can carry storage metadata and a `policy_reason`.

`extra/qk_ansor.py --policy-max-storage-mb N` now emits a memory-capped policy:

1. Expand shape search results to real model tensors.
2. Estimate persistent primitive bytes per tensor.
3. Estimate benefit from fused baseline device time minus primitive device time.
4. Rank primitive candidates by `benefit_ms_per_mb`.
5. Select candidates until the byte cap is reached.
6. Emit fused-graph fallback entries for over-budget tensors.

The pipeline can compare this capped generated policy against a generic fused
baseline with `--reference-mode generic`, needed for 32B because the full
explicit primitive baseline also OOMs.

## Runtime Controls

The runtime now reports actual primitive sidecar bytes under
`Q4K_PRIMITIVE_DEBUG`, `Q6K_PRIMITIVE_DEBUG`, or `QK_GENERATED_POLICY_DEBUG`.

Relevant environment variables:

- `QK_PRIMITIVE_MAX_STORAGE_MB=N`: cap total persistent primitive sidecar bytes
  across Q4_K and Q6_K installers. Tensors that would exceed the cap fall back
  and increment `runtime_storage_cap`.
- `QK_GENERATED_POLICY_STRICT=1`: turn storage-cap fallback into a loud error
  for generated-policy runs.
- `QK_PRIMITIVE_STORAGE=sidecar`: default fast path, preloads packed primitive
  storage on device.
- `QK_PRIMITIVE_STORAGE=q4_ondemand`: experimental Q4_K-only non-persistent
  path. It keeps the Q4 packed slice disk-backed and copies at decode time.

`q4_ondemand` is a negative prototype, not a recommended mode. It proves that
Q4 persistent sidecar bytes can be removed, but the per-token copy cost destroys
decode throughput.

## 32B Result

Artifact: `bench/qk-policy-cap-20260612/32b-1536mb/`.

Command:

```sh
DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_policy_pipeline.py \
  --model ~/models/Qwen3-32B-Q4_K_M.gguf \
  --out bench/qk-policy-cap-20260612/32b-1536mb \
  --device AMD --level 2 --iters 2 --benchmark 128 --repeats 3 \
  --max-extra-repeats 1 --profile auto --reference-mode generic \
  --policy-max-storage-mb 1536 --reuse
```

Decision:

| model | reference | capped policy | cap | selected primitive tensors | result |
|---|---|---|---:|---:|---|
| Qwen3-32B-Q4_K_M | generic fused graph | generated tensor policy | `1536 MB` | `144` | accept |

Decode:

| mode | avg tok/s | stable | note |
|---|---:|---|---|
| generic fused baseline | `3.44` | yes | pipeline labels this as `explicit` because it is the reference side |
| capped generated policy | `4.16` | yes | `20.98%` faster, greedy output A/B match |

Storage selection:

| selected role | tensors |
|---|---:|
| `attn_k` | `64` |
| `attn_v` | `64` |
| `ffn_down` | `16` |

Selected bytes: `1,600,389,120` under a cap of `1,610,612,736`.

The same policy re-run with runtime accounting reports exactly
`1,600,389,120` persistent bytes (`1526.25 MB`) and reaches about `4.30 tok/s`
after warmup in a short benchmark-4 smoke.

Important caveat: this is not an explicit-full-primitive comparison. The full
explicit Q4/Q6 primitive baseline does not fit for 32B. This result says a
memory-aware generated policy can recover some speed while fitting beside the
generic model graph.

## Why This Is Architectural

Shape-level policy is the wrong abstraction for memory pressure. A single shape
decision can imply dozens of layer tensors. On 32B, choosing "primitive for this
shape" can silently allocate gigabytes of persistent sidecar storage.

Tensor-scoped policy is the minimum required control surface:

- it allows selecting high-benefit layers while leaving low-benefit layers on
  the fused graph;
- it records why each tensor was selected or fused;
- it makes memory a first-class search constraint rather than an after-the-fact
  OOM.

This moves the project in the Ansor direction without pretending the storage
problem is solved. The searcher now reasons about a model-level resource budget,
not only per-shape speed.

## Storage-Control Validation

Artifact: `bench/qk-storage-20260612/`.

8B smokes:

| mode | Q4 wrappers | Q6 wrappers | persistent storage MB | warm tok/s | verdict |
|---|---:|---:|---:|---:|---|
| generated sidecar | `162` | `18` | `3786.75` | `57.77` | fast path |
| generated runtime cap `512 MB` | `28` | `0` | `504.00` | `13.46` | cap works, install-order selection is slow |
| generated `q4_ondemand` | `162` | `18` | `708.75` | `0.55` | rejects per-token Q4 copy |

32B smokes:

| mode | Q4 wrappers | Q6 wrappers | persistent storage MB | warm tok/s | verdict |
|---|---:|---:|---:|---:|---|
| static generated cap `1536 MB` | `112` | `32` | `1526.25` | `4.30` | useful selected policy |
| full generated policy + runtime cap `1536 MB` | `43` | `0` | `1535.62` | `3.71` | guard works, but not an optimizer |

Interpretation:

- Runtime accounting matches the generated policy storage estimate.
- The runtime cap prevents OOM and explains skipped tensors.
- Install-order capping is slower than generated benefit-per-MB capping.
- Q4 on-demand removes persistent Q4 sidecar bytes but is too slow to use.

## Long-Term Design

The cap is a bridge. The long-term design should reduce or remove duplicate
storage:

1. Shared packed storage
   - A primitive wrapper should reference the existing GGUF-backed packed weight
     storage when the runtime can consume the same layout.
   - Avoid making a second persistent device copy when the source storage is
     already resident or can be viewed safely.

2. Lazy materialization
   - Install primitive metadata at model load, not necessarily all sidecar
     buffers.
   - Materialize packed buffers only for tensors selected by a memory policy or
     first use.
   - Keep allocation failure local to one tensor and fall back loudly.

3. Budget-aware policy generation
   - Keep `persistent_bytes`, `benefit_ms`, and `benefit_ms_per_mb` in policy
     entries.
   - Support caps by absolute bytes and by remaining device memory.
   - Make policy decisions reproducible and artifact-pinned.

4. Runtime diagnostics
   - Report selected/skipped tensor counts and reasons.
   - Report estimated and actual persistent primitive bytes.
   - Fail loudly when a policy requests more storage than the runtime cap.

5. Search integration
   - Treat storage as part of the candidate objective:
     `speedup subject to persistent_bytes <= budget`.
   - Do not let BEAM or generated search choose a policy that cannot fit the
     target model/device pair.

## Next Work

Ordered by architectural value:

1. Move up to the harness/infrastructure layer. Treat storage caps and accounting
   as available primitives for future experiments.
2. If storage work resumes, prototype real shared ownership for one Q4_K family:
   no duplicate persistent buffer and no per-token disk copy.
3. Re-run a cheap 32B true comparison only if shared ownership makes full
   explicit/full generated policies fit naturally.
4. Do not resume kernel search from this track. The storage work is now
   infrastructure; use it to support higher-level loop/harness work.

Stop rule: do not spend more time on 32B candidate search until the storage
architecture can explain where every persistent packed-weight byte lives.
